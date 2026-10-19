/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package capacityadmission implements the 'capacity-admission' plugin: a
// capacity-aware admission policy for a heterogeneous pool that pairs a
// large-context DGX fleet with a small RTX 6000 host, used as the shared
// qwen38-flash-next route's capacity gate.
//
// The RTX endpoint is an accelerator, not a fallback: a request may land on it
// only when its full prompt token count (converted natively by the RTX engine
// itself) plus an explicit maximum output reservation fit inside the RTX KV
// budget. Everything else - uncountable bodies, missing output caps, budget
// pressure, an unproven engine state after an EPP restart - keeps the request
// on the DGX path, whose cache-affinity and active-request scoring the plugin
// deliberately leaves untouched.
//
// The plugin occupies several extension points of one request cycle:
//
//   - DataProducer: asks the RTX engine's own /tokenize (OpenAI chat and
//     completions) or /v1/messages/count_tokens (Anthropic Messages) for the
//     input token count, once per request, and caches the admission decision
//     on the request as an attribute. All HTTP happens here, off the
//     scheduling hot path.
//   - Filter: runs before the affinity scorers. It applies the capacity
//     policy and, when the RTX endpoint passes, atomically reserves the
//     request (idempotent per request identity, so repeated profile runs in
//     one cycle neither double count nor race). A losing request gets the RTX
//     endpoint dropped from its candidate set and proceeds on DGX through
//     the unchanged scorers - the bounded reschedule that keeps races off
//     the client as an avoidable 503.
//   - PreRequest: confirms the reservation against the final destination,
//     releases it when scoring or a competing profile took the request to
//     DGX anyway, marks the session as relocated, and rejects RTX-only
//     (hardware alias) requests the RTX engine cannot account for.
//   - ResponseBody: releases the reservation at end-of-stream, including the
//     abort path the director guarantees for every request that picked a pod
//     but never completed (client disconnect, cancellation, failed
//     forwarding).
//
// Fail-closed rules, all deliberate:
//
//   - Until the RTX engine has been observed idle (zero running and waiting
//     requests in a fresh metrics snapshot) since this process started, the
//     RTX endpoint stays excluded: an EPP restart lost the in-process
//     reservation table, so only a proven-drained engine is safe to admit
//     against again. Stale reservations are then reconciled away only when
//     an idle snapshot postdates the hold by more than the grace period.
//   - A request whose tokens cannot be counted never occupies the RTX budget.
//     Hardware-alias (RTX-only) requests that cannot be counted are rejected
//     with a clear error rather than silently placed on DGX or force-admitted
//     on an unknown budget.
//   - A session that was relocated to DGX after its first RTX turn stays on
//     DGX (sticky, bounded store keyed by the conversation root hash or the
//     configured session header) so later turns do not bounce back to RTX
//     when capacity frees up.
//   - Session reference-only requests (Responses with previous_response_id
//     and no new input) of an RTX-homed session are rejected: their context
//     lives on the bound endpoint only, and it cannot be counted.
//
// Limitations (documented, not silently degraded): budgets are per EPP
// replica; a multi-replica deployment multiplies the effective RTX budget.
// The reservation flow assumes a single primary profile; extra warm/test
// profiles are handled via idempotent reserve plus PreRequest
// confirm/release, but prefill/decode disagg pairing is not accounted per
// role. gRPC-payload requests are not countable and therefore DGX-only.
package capacityadmission

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/conversation"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// PluginType is the registered type name of the capacity-admission plugin.
const PluginType = "capacity-admission"

// defaultMaxSessionEntries bounds the relocated/pinned session store. Session
// entries are small fixed-size records; the bound protects a long-lived EPP
// from unbounded memory the contract explicitly calls out.
const defaultMaxSessionEntries = 65536

// decisionKey names the per-request admission decision attribute published by
// this plugin. It is private to the plugin: the framework never schedules a
// second producer for this key.
var decisionKey = fwkplugin.NewDataKey("CapacityAdmissionDecision", PluginType)

// Reason classifies why an request is or is not allowed to occupy the RTX
// budget. It is logged (never with prompt content) so routing decisions are
// reviewable after the fact.
type Reason string

const (
	// ReasonAllowed indicates the RTX endpoint passed every check.
	ReasonAllowed Reason = ""
	// ReasonNotGoverned means the target model is outside this plugin's route.
	ReasonNotGoverned Reason = "model_not_governed"
	// ReasonNotParsed means the body was not parsed into a countable shape.
	ReasonNotParsed Reason = "body_not_parsed"
	// ReasonNoRTXEndpoint means no candidate matched the RTX identity.
	ReasonNoRTXEndpoint Reason = "no_rtx_candidate"
	// ReasonStartupGate means the engine has not been observed idle since this
	// EPP process started (fail closed after restart).
	ReasonStartupGate Reason = "startup_gate_closed"
	// ReasonStaleMetrics means the RTX endpoint's metrics snapshot is missing,
	// stale, or predates this process - it cannot prove the engine's state.
	ReasonStaleMetrics Reason = "rtx_metrics_unusable"
	// ReasonRelocated means the session was previously relocated to DGX and
	// must stay there (no bounce back).
	ReasonRelocated Reason = "session_relocated_sticky_dgx"
	// ReasonReferenceOnlyUnbound means the request only references session
	// state but no RTX binding exists to honor, so it goes DGX-only.
	ReasonReferenceOnlyUnbound Reason = "session_reference_only_unbound"
	// ReasonMissingMaxOutput means the client gave no explicit maximum output
	// token cap, so the full reservation is unknown.
	ReasonMissingMaxOutput Reason = "missing_explicit_max_output"
	// ReasonUncountableAPI means this request API shape is one the RTX engine
	// cannot count for us (e.g. /v1/responses on the shared route).
	ReasonUncountableAPI Reason = "api_not_countable_on_rtx"
	// ReasonUncountablePayload means the payload shape (gRPC, unparsed, or a
	// field we refuse to approximate, such as a per-request chat_template
	// override) cannot be faithfully tokenized by the RTX counter.
	ReasonUncountablePayload Reason = "payload_not_countable_on_rtx"
	// ReasonCounterFailed means the RTX counter endpoint errored, timed out,
	// or returned a shape we do not trust. Fail closed: RTX ineligible.
	ReasonCounterFailed Reason = "rtx_counter_unavailable"
	// ReasonRequestTooLarge means input + reserved output exceeds maxRequestTokens.
	ReasonRequestTooLarge Reason = "request_exceeds_rtx_max_request_tokens"
	// ReasonBudgetConcurrency means the RTX concurrency limit is reached.
	ReasonBudgetConcurrency Reason = "rtx_concurrency_budget_exceeded"
	// ReasonBudgetResident means the RTX resident token budget is exhausted.
	ReasonBudgetResident Reason = "rtx_resident_token_budget_exceeded"
	// ReasonRTXOnlyNoCandidate means an RTX-only (hardware alias) request has
	// no RTX endpoint among candidates: never silently fall back to DGX.
	ReasonRTXOnlyNoCandidate Reason = "rtx_only_model_no_rtx_endpoint"
	// ReasonReferenceOnlyNotAccountable means the request only reads stored
	// session state (Responses previous_response_id with no new input) and
	// the engine cannot count that context: it can neither be reserved nor
	// relocated, so it is rejected with a clear error.
	ReasonReferenceOnlyNotAccountable Reason = "session_reference_only_not_accountable"
)

// isCapacityExhaustion reports whether the reason reflects transient
// capacity/gate pressure (the request shape is admissible in principle) as
// opposed to a fundamentally uncountable request. Alias rejections use this
// to answer 503 for "RTX is full" and 400 for "RTX cannot count this".
func (r Reason) isCapacityExhaustion() bool {
	switch r {
	case ReasonBudgetConcurrency, ReasonBudgetResident, ReasonStartupGate, ReasonStaleMetrics, ReasonRelocated:
		return true
	default:
		return false
	}
}

// Parameters configures the capacity-admission plugin. The four fields the
// route contract pins are rtxEndpointName, maxConcurrentRequests,
// maxResidentTokens and maxRequestTokens; the rest carry safe defaults.
type Parameters struct {
	// RTXEndpointName matches the RTX endpoint by EndpointMetadata.Name,
	// tolerating the Deployment's random pod suffix: a candidate matches when
	// its name equals the configured value or starts with it followed by '-'.
	RTXEndpointName string `json:"rtxEndpointName,omitempty"`
	// RTXEndpointLabel optionally matches instead (or additionally) on a pod
	// label, formatted "key" (present) or "key=value".
	RTXEndpointLabel string `json:"rtxEndpointLabel,omitempty"`
	// CombinedModels lists target model names served by the shared route:
	// RTX-eligible when capacity permits, DGX otherwise.
	CombinedModels []string `json:"combinedModels,omitempty"`
	// RTXOnlyModels lists hardware alias model names that must never leave
	// the RTX endpoint (e.g. "qwen38-flash-next-rtx6000"). Admission uses the
	// same accounting; alias requests that cannot be counted are rejected.
	RTXOnlyModels []string `json:"rtxOnlyModels,omitempty"`
	// MaxConcurrentRequests caps concurrently admitted RTX requests.
	MaxConcurrentRequests int `json:"maxConcurrentRequests"`
	// MaxResidentTokens caps the summed (input + reserved output) tokens of
	// all live RTX reservations.
	MaxResidentTokens int64 `json:"maxResidentTokens"`
	// MaxRequestTokens caps the (input + reserved output) of a single request.
	MaxRequestTokens int64 `json:"maxRequestTokens"`
	// Counter configures the RTX token-count HTTP client.
	Counter CounterParameters `json:"counter"`
	// SessionHeader optionally names a request header (e.g. "x-session-id")
	// whose opaque value keys the session store. When unset or empty on a
	// request, the store keys on the conversation root hash instead. Raw
	// prompts are never stored either way.
	SessionHeader string `json:"sessionHeader,omitempty"`
	// MaxSessionEntries bounds the session store (LRU). Default 65536.
	MaxSessionEntries int `json:"maxSessionEntries,omitempty"`
	// SessionTTL evicts session placements unused for this long (duration
	// string, e.g. "2h"). Default "2h". Zero disables expiry (LRU bound still
	// applies).
	SessionTTL string `json:"sessionTTL,omitempty"`
	// MetricsMaxAge is how fresh an RTX endpoint's metrics snapshot must be
	// to count as evidence of engine state (duration string). Default "30s".
	MetricsMaxAge string `json:"metricsMaxAge,omitempty"`
	// StaleReservationGrace is how long a reservation must have been held
	// before an engine-reported-idle reconciliation may clear it, covering
	// releases lost to a crashed hook (duration string). Default "30s".
	StaleReservationGrace string `json:"staleReservationGrace,omitempty"`
}

// CounterParameters configures the outbound token-count calls to the RTX
// engine. The URLs must point at the RTX engine Service directly (not through
// the EPP) so counting a request never re-enters the routing path.
type CounterParameters struct {
	// BaseURL is the RTX engine base URL without a trailing slash, e.g.
	// "http://rtx6000-qwen38-flash-next:8000". Required.
	BaseURL string `json:"baseURL"`
	// OpenAITokenizePath overrides the default "/tokenize".
	OpenAITokenizePath string `json:"openAITokenizePath,omitempty"`
	// AnthropicCountPath overrides the default "/v1/messages/count_tokens".
	AnthropicCountPath string `json:"anthropicCountPath,omitempty"`
	// Timeout bounds one count request (duration string). Default "2s".
	Timeout string `json:"timeout,omitempty"`
	// Model overrides the model name sent to the counter. Empty uses the
	// request's target model, which the RTX engine must accept for every
	// governed name (the common name and the hardware alias alike).
	Model string `json:"model,omitempty"`
	// CACertPath, ClientCertPath and ClientKeyPath configure TLS like the
	// tokenizer producer's vLLM backend; InsecureSkipVerify accepts
	// self-signed certificates for development.
	CACertPath         string `json:"caCertPath,omitempty"`
	ClientCertPath     string `json:"clientCertPath,omitempty"`
	ClientKeyPath      string `json:"clientKeyPath,omitempty"`
	InsecureSkipVerify bool   `json:"insecureSkipVerify,omitempty"`
}

func (p *Parameters) applyDefaults() {
	if p.MaxSessionEntries <= 0 {
		p.MaxSessionEntries = defaultMaxSessionEntries
	}
	if p.SessionTTL == "" {
		p.SessionTTL = "2h"
	}
	if p.MetricsMaxAge == "" {
		p.MetricsMaxAge = "30s"
	}
	if p.StaleReservationGrace == "" {
		p.StaleReservationGrace = "30s"
	}
	if p.Counter.OpenAITokenizePath == "" {
		p.Counter.OpenAITokenizePath = "/tokenize"
	}
	if p.Counter.AnthropicCountPath == "" {
		p.Counter.AnthropicCountPath = "/v1/messages/count_tokens"
	}
	if p.Counter.Timeout == "" {
		p.Counter.Timeout = "2s"
	}
}

func (p *Parameters) validate() error {
	if strings.TrimSpace(p.RTXEndpointName) == "" && strings.TrimSpace(p.RTXEndpointLabel) == "" {
		return fmt.Errorf("'%s': one of rtxEndpointName or rtxEndpointLabel is required", PluginType)
	}
	if p.MaxConcurrentRequests <= 0 {
		return fmt.Errorf("'%s': maxConcurrentRequests must be positive", PluginType)
	}
	if p.MaxResidentTokens <= 0 {
		return fmt.Errorf("'%s': maxResidentTokens must be positive", PluginType)
	}
	if p.MaxRequestTokens <= 0 {
		return fmt.Errorf("'%s': maxRequestTokens must be positive", PluginType)
	}
	if p.MaxRequestTokens > p.MaxResidentTokens {
		return fmt.Errorf("'%s': maxRequestTokens (%d) must not exceed maxResidentTokens (%d), or no request could ever fit the resident budget",
			PluginType, p.MaxRequestTokens, p.MaxResidentTokens)
	}
	if len(p.CombinedModels) == 0 && len(p.RTXOnlyModels) == 0 {
		return fmt.Errorf("'%s': at least one of combinedModels or rtxOnlyModels must be configured", PluginType)
	}
	if strings.TrimSpace(p.Counter.BaseURL) == "" {
		return fmt.Errorf("'%s': counter.baseURL is required", PluginType)
	}
	if _, err := time.ParseDuration(p.SessionTTL); err != nil {
		return fmt.Errorf("'%s': invalid sessionTTL: %w", PluginType, err)
	}
	if _, err := time.ParseDuration(p.MetricsMaxAge); err != nil {
		return fmt.Errorf("'%s': invalid metricsMaxAge: %w", PluginType, err)
	}
	if _, err := time.ParseDuration(p.StaleReservationGrace); err != nil {
		return fmt.Errorf("'%s': invalid staleReservationGrace: %w", PluginType, err)
	}
	if _, err := time.ParseDuration(p.Counter.Timeout); err != nil {
		return fmt.Errorf("'%s': invalid counter.timeout: %w", PluginType, err)
	}
	if !strings.HasPrefix(p.Counter.OpenAITokenizePath, "/") {
		return fmt.Errorf("'%s': counter.openAITokenizePath must start with '/'", PluginType)
	}
	if !strings.HasPrefix(p.Counter.AnthropicCountPath, "/") {
		return fmt.Errorf("'%s': counter.anthropicCountPath must start with '/'", PluginType)
	}
	return nil
}

// apiKind classifies which request API a governed body presents.
type apiKind string

const (
	apiNone             apiKind = "none"
	apiChat             apiKind = "chat"
	apiCompletions      apiKind = "completions"
	apiMessages         apiKind = "messages"
	apiResponses        apiKind = "responses"
	apiGenerate         apiKind = "generate"
	apiOtherUnsupported apiKind = "other"
)

// decision is the cached, per-request admission decision. It is stored on the
// request under decisionKey by the DataProducer hook and reused by the Filter
// and PreRequest hooks of the same cycle, so the counter HTTP call
// happens exactly once regardless of how many profiles run.
type decision struct {
	governed   bool
	rtxOnly    bool
	kind       apiKind
	reference  bool // session reference-only (no new prompt content)
	countable  bool
	tokens     int64 // input + reserved output, meaningful when countable
	input      int64
	output     int64
	tooLarge   bool
	reason     Reason // exclusion reason when not admissible to RTX
	reqKey     string // reservation identity
	sessionKey string // bounded-store identity ("" = untracked)
	binding    string // last placement endpoint id ("" = unbound)
	boundRTX   bool
	relocated  bool        // sticky: relocated to DGX, do not bounce back
	wasRTX     bool        // the session's last recorded home was RTX-class
	reserved   bool        // set once the filter holds the reservation
	endpoint   string      // RTX endpoint id used for reservation
	stopGuard  func() bool // context.AfterFunc stop: releases the hold on request cancellation
}

// placement is the bounded session-store record.
type placement struct {
	endpoint  string
	relocated bool
	rtx       bool // the bound endpoint was RTX-class when recorded
	lastSeen  time.Time
}

// reservation is one live RTX capacity hold.
type reservation struct {
	endpointID string
	tokens     int64
	reservedAt time.Time
}

// compile-time interface checks.
var (
	_ fwkplugin.Plugin            = &CapacityAdmission{}
	_ fwkplugin.ProducerPlugin    = &CapacityAdmission{}
	_ fwkrc.DataProducer          = &CapacityAdmission{}
	_ fwkrc.TimeoutAwareProducer  = &CapacityAdmission{}
	_ fwkrc.PreRequest            = &CapacityAdmission{}
	_ fwkrc.ResponseBodyProcessor = &CapacityAdmission{}
	_ fwksched.Filter             = &CapacityAdmission{}
	_ fwkplugin.StateDumper       = &CapacityAdmission{}
)

// openGate marks the RTX engine as proven drained since process start, from
// an idle-and-fresh metrics observation made outside the admission path
// (e.g. by the RTX-only gate). It is idempotent.
func (p *CapacityAdmission) openGate(ctx context.Context, rtx fwksched.Endpoint) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.gateOpen {
		return
	}
	p.gateOpen = true
	log.FromContext(ctx).Info("capacity-admission opened: RTX engine observed idle with fresh metrics",
		"plugin", p.typedName.String(), "endpoint", p.rtxID(rtx))
}

// CapacityAdmission is the capacity-aware admission plugin.
type CapacityAdmission struct {
	typedName fwkplugin.TypedName
	params    Parameters

	sessionTTL            time.Duration
	metricsMaxAge         time.Duration
	staleReservationGrace time.Duration
	counterTimeout        time.Duration

	counter *counterClient

	// mu guards the reservation table, its per-endpoint totals, the startup
	// gate and the session store.
	mu           sync.Mutex
	reservations map[string]reservation
	epTokens     map[string]int64
	epCount      map[string]int
	gateOpen     bool
	// sessions is size-bounded; entries additionally expire after sessionTTL
	// via the lastSeen check (the LRU bounds memory even without TTL hits).
	sessions *lru.Cache[string, *placement]

	// processStart anchors metric freshness: snapshots scraped before this
	// process began are never evidence of the current engine state.
	processStart time.Time
	reqKeySeq    atomic.Uint64
}

// Factory builds the plugin from its JSON parameters.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", PluginType, err)
		}
	}
	params.applyDefaults()
	if err := params.validate(); err != nil {
		return nil, err
	}
	return New(name, params)
}

// New constructs a configured CapacityAdmission.
func New(name string, params Parameters) (*CapacityAdmission, error) {
	params.applyDefaults()
	if err := params.validate(); err != nil {
		return nil, err
	}
	sessionTTL, err := time.ParseDuration(params.SessionTTL)
	if err != nil {
		return nil, fmt.Errorf("invalid sessionTTL: %w", err)
	}
	metricsMaxAge, err := time.ParseDuration(params.MetricsMaxAge)
	if err != nil {
		return nil, fmt.Errorf("invalid metricsMaxAge: %w", err)
	}
	grace, err := time.ParseDuration(params.StaleReservationGrace)
	if err != nil {
		return nil, fmt.Errorf("invalid staleReservationGrace: %w", err)
	}
	countTimeout, err := time.ParseDuration(params.Counter.Timeout)
	if err != nil {
		return nil, fmt.Errorf("invalid counter.timeout: %w", err)
	}
	client, err := newCounterClient(params.Counter)
	if err != nil {
		return nil, err
	}
	sessions, err := lru.New[string, *placement](params.MaxSessionEntries)
	if err != nil {
		return nil, fmt.Errorf("creating session store: %w", err)
	}
	return &CapacityAdmission{
		typedName:             fwkplugin.TypedName{Type: PluginType, Name: name},
		params:                params,
		sessionTTL:            sessionTTL,
		metricsMaxAge:         metricsMaxAge,
		staleReservationGrace: grace,
		counterTimeout:        countTimeout,
		processStart:          time.Now(),
		counter:               client,
		reservations:          map[string]reservation{},
		epTokens:              map[string]int64{},
		epCount:               map[string]int{},
		sessions:              sessions,
	}, nil
}

// TypedName returns the type and name tuple of the plugin instance.
func (p *CapacityAdmission) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the per-request admission decision key.
func (p *CapacityAdmission) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{decisionKey: decision{}}
}

// Produce computes the admission decision (and the single counter HTTP call)
// before scheduling. It never returns an error: a failed count is expressed
// as an uncountable decision, which excludes the RTX endpoint and leaves the
// DGX path running.
func (p *CapacityAdmission) Produce(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	p.decide(ctx, request, endpoints)
	return nil
}

// ProduceTimeout gives the counter call room beyond the framework default.
func (p *CapacityAdmission) ProduceTimeout() time.Duration {
	return p.counterTimeout + 500*time.Millisecond
}

// governs reports whether the model participates in this route at all.
func (p *CapacityAdmission) governs(model string) (combined, rtxOnly bool) {
	for _, m := range p.params.CombinedModels {
		if strings.EqualFold(m, model) {
			return true, false
		}
	}
	for _, m := range p.params.RTXOnlyModels {
		if strings.EqualFold(m, model) {
			return false, true
		}
	}
	return false, false
}

// isRTX reports whether the endpoint is the RTX host, matching
// EndpointMetadata.Name with the Deployment random-suffix boundary
// ("name" or "name-<suffix>") and/or the configured label.
func (p *CapacityAdmission) isRTX(ep fwksched.Endpoint) bool {
	md := ep.GetMetadata()
	if md == nil {
		return false
	}
	if p.params.RTXEndpointName != "" {
		name := md.Name
		if name == p.params.RTXEndpointName || strings.HasPrefix(name, p.params.RTXEndpointName+"-") {
			return true
		}
	}
	if p.params.RTXEndpointLabel != "" {
		key, value, wantValue := strings.Cut(p.params.RTXEndpointLabel, "=")
		if v, ok := md.Labels[key]; ok && (!wantValue || v == value) {
			return true
		}
	}
	return false
}

func (p *CapacityAdmission) rtxID(ep fwksched.Endpoint) string {
	return p.epID(ep)
}

// epID returns a candidate's stable identity string, used to match session
// placements against live candidates (DGX pinning after relocation).
func (p *CapacityAdmission) epID(ep fwksched.Endpoint) string {
	md := ep.GetMetadata()
	if md == nil {
		return ""
	}
	return md.ID.String()
}

// findRTX returns the first RTX endpoint among candidates.
func (p *CapacityAdmission) findRTX(endpoints []fwksched.Endpoint) fwksched.Endpoint {
	for _, ep := range endpoints {
		if p.isRTX(ep) {
			return ep
		}
	}
	return nil
}

// decisionFor returns the cached decision, or nil.
func (p *CapacityAdmission) decisionFor(request *fwksched.InferenceRequest) *decision {
	if request == nil {
		return nil
	}
	if d, ok := fwksched.ReadRequestAttribute[*decision](request, decisionKey); ok {
		return d
	}
	return nil
}

// decide returns the request's admission decision, computing it on first call.
// All request-cycle hooks run on the director goroutine sequentially, so the
// compute-once guard is a plain attribute check.
func (p *CapacityAdmission) decide(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) *decision {
	if d := p.decisionFor(request); d != nil {
		return d
	}
	d := p.computeDecision(ctx, request, endpoints)
	request.PutAttribute(decisionKey, d)
	return d
}

// forceDecide recomputes the decision after in-cycle state changes (the
// startup gate opening on this request's own metrics check). The reservation
// identity stays stable so an idempotent hold is never orphaned by the
// recompute.
func (p *CapacityAdmission) forceDecide(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) *decision {
	old := p.decisionFor(request)
	d := p.computeDecision(ctx, request, endpoints)
	if old != nil {
		d.reqKey = old.reqKey
		d.reserved = old.reserved
	}
	request.PutAttribute(decisionKey, d)
	return d
}

// computeDecision classifies the request and, when the RTX endpoint is a
// plausible target, asks the RTX engine for the token count. The counter is
// only consulted when its answer can change the outcome.
func (p *CapacityAdmission) computeDecision(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) *decision {
	logger := log.FromContext(ctx)
	d := &decision{}

	combined, rtxOnly := p.governs(request.TargetModel)
	if !combined && !rtxOnly {
		d.reason = ReasonNotGoverned
		return d
	}
	d.governed = true
	d.rtxOnly = rtxOnly

	d.reqKey = request.RequestID
	if d.reqKey == "" {
		d.reqKey = "cap-" + strconv.FormatUint(p.reqKeySeq.Add(1), 10)
	}

	rtx := p.findRTX(endpoints)
	if rtx != nil {
		d.endpoint = p.rtxID(rtx)
	}

	d.kind, d.reference = classifyAPI(request.Body)

	if request.Body == nil {
		d.reason = ReasonNotParsed
		return d
	}

	d.sessionKey = p.sessionKeyFor(request)
	if key := d.sessionKey; key != "" {
		p.mu.Lock()
		if pl, ok := p.lookupPlacement(key); ok {
			d.binding = pl.endpoint
			d.relocated = pl.relocated
			d.wasRTX = pl.rtx
			d.boundRTX = pl.endpoint != "" && pl.endpoint == d.endpoint
		}
		p.mu.Unlock()
	}

	// A DGX-only candidate set has nothing to decide about - except
	// RTX-homed reference-only reads, which the filter must fail when the
	// home replica is gone, and that hinges on the placement just read.
	if rtx == nil && !rtxOnly {
		d.reason = ReasonNoRTXEndpoint
		return d
	}

	// Relocated sessions stick to DGX: skip the counter call entirely, its
	// answer cannot change the outcome.
	if d.relocated && !d.reference && !rtxOnly {
		d.reason = ReasonRelocated
		return d
	}
	// Reference-only reads cannot be counted (the engine has no Responses
	// count endpoint). A session homed on RTX would lose its context on any
	// other endpoint, and admitting it would consume an unaccountable budget
	// - so it is rejected (see PreRequest). A session never homed on RTX has
	// nothing to honor there: keep it off the accelerator.
	if d.reference {
		if d.wasRTX {
			d.reason = ReasonReferenceOnlyNotAccountable
		} else {
			d.reason = ReasonReferenceOnlyUnbound
		}
		return d
	}
	// Gate closed: RTX stays excluded, do not spend a count call on it.
	if !p.isGateOpen() && !rtxOnly {
		d.reason = ReasonStartupGate
		return d
	}

	// Cheap pre-check: a fully occupied concurrency budget excludes RTX
	// regardless of the token count, so skip the counter round-trip. (The
	// authoritative budget check runs under the lock at reservation time.)
	p.mu.Lock()
	busy := p.epCount[d.endpoint] >= p.params.MaxConcurrentRequests
	p.mu.Unlock()
	if busy {
		d.reason = ReasonBudgetConcurrency
		return d
	}

	switch d.kind {
	case apiResponses:
		// vLLM on RTX exposes no /v1/responses count endpoint (405): the
		// shared route keeps Responses on DGX; RTX-only aliases are rejected
		// at PreRequest (there is no unaccountable-forward escape hatch).
		d.reason = ReasonUncountableAPI
		return d
	case apiOtherUnsupported, apiNone:
		d.reason = ReasonUncountableAPI
		return d
	}

	// Explicit output cap is mandatory: the reservation must cover the full
	// generation, and an estimate is not a reservation.
	if request.Body.MaxOutputTokens == nil {
		d.reason = ReasonMissingMaxOutput
		return d
	}
	if *request.Body.MaxOutputTokens < 0 {
		d.reason = ReasonMissingMaxOutput
		return d
	}
	d.output = *request.Body.MaxOutputTokens

	input, reason := p.countInput(ctx, request, d.kind)
	if reason != ReasonAllowed {
		d.reason = reason
		return d
	}
	d.input = input
	d.countable = true
	// Overflow-safe per-request ceiling: never add input+output unchecked.
	if d.output > p.params.MaxRequestTokens || input > p.params.MaxRequestTokens-d.output {
		d.tooLarge = true
		d.reason = ReasonRequestTooLarge
		return d
	}
	d.tokens = input + d.output
	logger.V(logutil.DEBUG).Info("capacity-admission decision",
		"requestID", d.reqKey, "model", request.TargetModel, "kind", d.kind,
		"inputTokens", d.input, "maxOutputTokens", d.output, "endpoint", d.endpoint)
	return d
}

// classifyAPI maps the body to its API kind and detects session reference-only
// requests: Responses carrying previous_response_id with no new input.
func classifyAPI(body *fwkrh.InferenceRequestBody) (apiKind, bool) {
	if body == nil {
		return apiNone, false
	}
	switch {
	case body.ChatCompletions != nil:
		return apiChat, false
	case body.Messages != nil:
		return apiMessages, false
	case body.Completions != nil:
		return apiCompletions, false
	case body.Responses != nil:
		reference := responsesReferenceOnly(body)
		return apiResponses, reference
	case body.Generate != nil || body.TokenizedRequest != nil:
		return apiGenerate, false
	default:
		return apiOtherUnsupported, false
	}
}

// responsesReferenceOnly reports whether a Responses request adds no new
// prompt content and only references stored state (previous_response_id).
func responsesReferenceOnly(body *fwkrh.InferenceRequestBody) bool {
	if body.Responses == nil {
		return false
	}
	inputEmpty := body.Responses.Input == nil
	if s, ok := body.Responses.Input.(string); ok && strings.TrimSpace(s) == "" {
		inputEmpty = true
	}
	if arr, ok := body.Responses.Input.([]any); ok && len(arr) == 0 {
		inputEmpty = true
	}
	if !inputEmpty {
		return false
	}
	if m := payloadMap(body); m != nil {
		if ref, ok := m["previous_response_id"].(string); ok && ref != "" {
			return true
		}
	}
	// An empty input with no resolvable reference is not a state read; leave
	// it to the generic uncountable path.
	return false
}

// sessionKeyFor derives the bounded-store identity: the configured opaque
// session header when present, else the FNV hash of the conversation root.
// Neither stores raw prompts.
func (p *CapacityAdmission) sessionKeyFor(request *fwksched.InferenceRequest) string {
	if p.params.SessionHeader != "" && request.Headers != nil {
		if v, ok := metadata.GetLowerCaseHeaderValue(request.Headers, p.params.SessionHeader); ok && strings.TrimSpace(v) != "" {
			return "h:" + strings.TrimSpace(v)
		}
	}
	if root, ok := conversation.Root(request); ok {
		return "r:" + strconv.FormatUint(conversation.Hash(root), 16)
	}
	return ""
}

// ---------------------------------------------------------------------------
// Startup gate and reconciliation
// ---------------------------------------------------------------------------

// metricsUsable reports whether a snapshot is present, fresh, and new enough
// to describe the post-restart engine: a snapshot scraped before this plugin
// was constructed (process restart) proves nothing about the reservations we
// no longer hold, so it can neither open the gate nor reconcile.
func (p *CapacityAdmission) metricsUsable(m *fwkdl.Metrics, now time.Time) bool {
	if m == nil {
		return false
	}
	if m.UpdateTime.IsZero() || !m.UpdateTime.After(p.processStart) || m.UpdateTime.After(now) {
		return false
	}
	if now.Sub(m.UpdateTime) > p.metricsMaxAge {
		return false
	}
	return true
}

func engineIdle(m *fwkdl.Metrics) bool {
	return m != nil && m.RunningRequestsSize == 0 && m.WaitingQueueSize == 0
}

// isGateOpen reports whether the RTX endpoint has been proven drained since
// process start.
func (p *CapacityAdmission) isGateOpen() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.gateOpen
}

// checkRTXMetrics inspects the RTX endpoint's live metrics: it opens the
// startup gate on the first fresh idle observation, and reconciles stale
// reservations away while the engine proves idle. It returns the reason the
// RTX endpoint must be excluded (ReasonAllowed when it may serve).
func (p *CapacityAdmission) checkRTXMetrics(ctx context.Context, rtx fwksched.Endpoint) Reason {
	now := time.Now()
	m := rtx.GetMetrics()
	if !p.metricsUsable(m, now) {
		if !p.isGateOpen() {
			return ReasonStaleMetrics
		}
		// Gate opened earlier: a momentary stale scrape must not flip an
		// already-proven engine back to closed, but we cannot reconcile on it
		// either.
		return ReasonAllowed
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if engineIdle(m) {
		if !p.gateOpen {
			p.gateOpen = true
			log.FromContext(ctx).Info("capacity-admission opened: RTX engine observed idle with fresh metrics",
				"plugin", p.typedName.String(), "endpoint", p.rtxID(rtx))
		}
		p.reconcileLocked(ctx, m.UpdateTime)
		return ReasonAllowed
	}
	if !p.gateOpen {
		return ReasonStartupGate
	}
	return ReasonAllowed
}

// reconcileLocked clears reservations only when the engine's own idle
// snapshot POSTDATES the reservation by more than the grace period: a hold
// taken after (or shortly before) the scrape can legitimately coexist with
// idle metrics (the request has not reached the engine yet), so only holds
// the idle evidence contradicts outright are dropped. It runs under mu.
func (p *CapacityAdmission) reconcileLocked(ctx context.Context, idleObserved time.Time) {
	var cleared int64
	for key, res := range p.reservations {
		if idleObserved.After(res.reservedAt.Add(p.staleReservationGrace)) {
			p.dropLocked(key)
			cleared++
		}
	}
	if cleared > 0 {
		log.FromContext(ctx).Info("capacity-admission reconciled stale reservations against idle RTX engine",
			"plugin", p.typedName.String(), "cleared", cleared)
	}
}

// ---------------------------------------------------------------------------
// Reservation accounting
// ---------------------------------------------------------------------------

// feasibleLocked applies the budget to a candidate (endpoint, tokens) hold.
// When reqKey already holds a reservation it is an idempotent refresh and
// always succeeds. There is no bypass: a hold that cannot be accounted
// honestly is never taken (uncountable requests are excluded from RTX).
// It runs under mu.
func (p *CapacityAdmission) feasibleLocked(reqKey, endpointID string, tokens int64) Reason {
	if _, ok := p.reservations[reqKey]; ok {
		// Idempotent repeat (re-filtered request) or re-booking onto
		// another RTX replica: reserve() reconciles the totals.
		return ReasonAllowed
	}
	if p.epCount[endpointID] >= p.params.MaxConcurrentRequests {
		return ReasonBudgetConcurrency
	}
	if p.epTokens[endpointID]+tokens > p.params.MaxResidentTokens {
		return ReasonBudgetResident
	}
	return ReasonAllowed
}

// reserve atomically records the hold. Idempotent per request key.
func (p *CapacityAdmission) reserve(reqKey, endpointID string, tokens int64) Reason {
	p.mu.Lock()
	defer p.mu.Unlock()
	reason := p.feasibleLocked(reqKey, endpointID, tokens)
	if reason != ReasonAllowed {
		return reason
	}
	if old, ok := p.reservations[reqKey]; ok {
		p.subtractLocked(old)
	}
	p.reservations[reqKey] = reservation{
		endpointID: endpointID,
		tokens:     tokens,
		reservedAt: time.Now(),
	}
	p.epCount[endpointID]++
	p.epTokens[endpointID] += tokens
	return ReasonAllowed
}

// guardCancel releases the request's hold if the request context is
// cancelled before the destination is even chosen (an ext-proc teardown in
// the middle of scheduling would otherwise strand the reservation until an
// idle-engine reconciliation). Registered once per decision; the end-of-stream
// hook stops it so completed requests leave no callbacks behind.
func (p *CapacityAdmission) guardCancel(ctx context.Context, d *decision) {
	if d.stopGuard != nil || d.reqKey == "" {
		return
	}
	d.stopGuard = context.AfterFunc(ctx, func() { p.release(d.reqKey) })
}

// release removes a hold. It is idempotent and returns true when a
// reservation was actually freed.
func (p *CapacityAdmission) release(reqKey string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, ok := p.reservations[reqKey]; !ok {
		return false
	}
	p.dropLocked(reqKey)
	return true
}

func (p *CapacityAdmission) dropLocked(reqKey string) {
	if res, ok := p.reservations[reqKey]; ok {
		p.subtractLocked(res)
		delete(p.reservations, reqKey)
	}
}

func (p *CapacityAdmission) subtractLocked(res reservation) {
	p.epCount[res.endpointID]--
	if p.epCount[res.endpointID] <= 0 {
		delete(p.epCount, res.endpointID)
	}
	p.epTokens[res.endpointID] -= res.tokens
	if p.epTokens[res.endpointID] <= 0 {
		delete(p.epTokens, res.endpointID)
	}
}

// held reports whether the request currently holds a reservation and on which
// endpoint (used by the PreRequest confirmation).
func (p *CapacityAdmission) held(reqKey string) (reservation, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	res, ok := p.reservations[reqKey]
	return res, ok
}

// ---------------------------------------------------------------------------
// Session placements (bounded)
// ---------------------------------------------------------------------------

// lookupPlacement returns the live placement for a session key. Callers hold
// mu. Expired entries read as absent (the LRU bounds total memory).
func (p *CapacityAdmission) lookupPlacement(key string) (*placement, bool) {
	pl, ok := p.sessions.Get(key)
	if !ok || pl == nil {
		return nil, false
	}
	if p.sessionTTL > 0 && time.Since(pl.lastSeen) > p.sessionTTL {
		p.sessions.Remove(key)
		return nil, false
	}
	return pl, true
}

// recordPlacement stores the endpoint a session is served by, marking the
// session relocated when it leaves a previously RTX-bound home for DGX.
// Reference-only requests never relocate (their caller must not move them).
func (p *CapacityAdmission) recordPlacement(key, endpointID string, relocated, rtx bool) {
	if key == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	now := time.Now()
	existing, ok := p.lookupPlacement(key)
	if !ok {
		p.sessions.Add(key, &placement{endpoint: endpointID, relocated: relocated, rtx: rtx, lastSeen: now})
		return
	}
	if relocated {
		existing.relocated = true
	}
	existing.endpoint = endpointID
	existing.rtx = rtx
	p.sessions.Add(key, existing)
}

// touchPlacement refreshes lastSeen without changing the binding.
func (p *CapacityAdmission) touchPlacement(key string) {
	if key == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if pl, ok := p.lookupPlacement(key); ok {
		pl.lastSeen = time.Now()
		p.sessions.Add(key, pl)
	}
}

// liveBudgetSnapshot renders the reservation state for DumpState.
type stateDump struct {
	GateOpen     bool              `json:"gateOpen"`
	Reservations []reservationDump `json:"reservations"`
	SessionCount int               `json:"sessions"`
}

type reservationDump struct {
	Request    string `json:"request"`
	Endpoint   string `json:"endpoint"`
	Tokens     int64  `json:"tokens"`
	HeldForSec int64  `json:"heldSeconds"`
}

// DumpState exposes sanitized, bounded plugin state for the debug endpoint:
// live reservations (ids and token counts only - never prompts, headers or
// payloads) plus the gate state.
func (p *CapacityAdmission) DumpState() (json.RawMessage, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	now := time.Now()
	dump := stateDump{GateOpen: p.gateOpen, SessionCount: p.sessions.Len()}
	for key, res := range p.reservations {
		if len(dump.Reservations) >= 128 { // bounded
			break
		}
		dump.Reservations = append(dump.Reservations, reservationDump{
			Request:    key,
			Endpoint:   res.endpointID,
			Tokens:     res.tokens,
			HeldForSec: int64(now.Sub(res.reservedAt).Seconds()),
		})
	}
	return json.Marshal(dump)
}
