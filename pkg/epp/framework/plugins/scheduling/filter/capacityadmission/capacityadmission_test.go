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

package capacityadmission

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const (
	rtxName     = "rtx6000-qwen38-flash-next"
	rtxPod      = rtxName + "-7d9f8b-abcde" // Deployment-style random suffix
	dgxName     = "dgx-qwen38-flash-next"
	combinedMdl = "qwen38-flash-next"
	aliasMdl    = "qwen38-flash-next-rtx6000"
	testNS      = "llm-d"

	bigMarker   = "BIGCTX"
	bigCount    = int64(450_000) // exceeds MaxRequestTokens (400k) to exercise relocation
	smallCount  = int64(5_000)
	defaultRoot = "root message for capacity session"
)

func rtxIDStr() string {
	return types.NamespacedName{Namespace: testNS, Name: rtxPod}.String()
}

func newEndpoint(name string, m *fwkdl.Metrics) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      types.NamespacedName{Namespace: testNS, Name: name},
			Name:    name,
			Address: "10.0.0.1",
			Port:    "8000",
		},
		m, nil)
}

func idleMetrics() *fwkdl.Metrics {
	return &fwkdl.Metrics{UpdateTime: time.Now()}
}

func busyMetrics() *fwkdl.Metrics {
	return &fwkdl.Metrics{
		RunningRequestsSize: 2,
		WaitingQueueSize:    1,
		UpdateTime:          time.Now(),
	}
}

func rtxEndpoint() fwksched.Endpoint { return newEndpoint(rtxPod, idleMetrics()) }
func dgxEndpoint() fwksched.Endpoint { return newEndpoint(dgxName, idleMetrics()) }
func dgxEndpoint2() fwksched.Endpoint {
	return newEndpoint(dgxName+"-second", idleMetrics())
}

// counterScript is a fake RTX engine token-count service: /tokenize answers
// with 450k when the body contains the BIGCTX marker, 5k otherwise.
type counterScript struct {
	chatCalls   atomic.Int64
	nativeCalls atomic.Int64
	failChat    atomic.Bool
	failNative  atomic.Bool
}

func newCounterServer(t *testing.T) (*counterScript, string) {
	t.Helper()
	cs := &counterScript{}
	mux := http.NewServeMux()
	mux.HandleFunc("/tokenize", func(w http.ResponseWriter, r *http.Request) {
		cs.chatCalls.Add(1)
		if cs.failChat.Load() {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		body, _ := io.ReadAll(r.Body)
		n := smallCount
		if strings.Contains(string(body), bigMarker) {
			n = bigCount
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"count": n})
	})
	mux.HandleFunc("/v1/messages/count_tokens", func(w http.ResponseWriter, r *http.Request) {
		cs.nativeCalls.Add(1)
		if cs.failNative.Load() {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		body, _ := io.ReadAll(r.Body)
		n := smallCount
		if strings.Contains(string(body), bigMarker) {
			n = bigCount
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"input_tokens": n})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return cs, srv.URL
}

// chatRequest builds a governed chat request. The conversation root
// (system + first user) is constant so growing the conversation across turns
// keeps the session key stable; `big` appends history beyond the root.
func chatRequest(reqID, model, session string, big bool, maxOut *int64) *fwksched.InferenceRequest {
	messages := []any{
		map[string]any{"role": "system", "content": "You are helpful."},
		map[string]any{"role": "user", "content": defaultRoot},
	}
	if big {
		messages = append(messages,
			map[string]any{"role": "assistant", "content": "ok"},
			map[string]any{"role": "user", "content": "carry on with " + bigMarker},
		)
	}
	payload := fwkrh.PayloadMap{
		"model":    model,
		"messages": messages,
	}
	if maxOut != nil {
		payload["max_tokens"] = *maxOut
	}
	headers := map[string]string{}
	if session != "" {
		headers["x-session-id"] = session
	}
	return &fwksched.InferenceRequest{
		RequestID:   reqID,
		TargetModel: model,
		Headers:     headers,
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{Role: "system", Content: fwkrh.Content{Raw: "You are helpful."}},
					{Role: "user", Content: fwkrh.Content{Raw: defaultRoot}},
				},
			},
			MaxOutputTokens: maxOut,
			Payload:         payload,
		},
	}
}

func messagesRequest(reqID, model, session string, big bool, maxOut *int64) *fwksched.InferenceRequest {
	userText := defaultRoot
	if big {
		userText = defaultRoot + " " + bigMarker
	}
	payload := fwkrh.PayloadMap{
		"model":    model,
		"system":   "You are helpful.",
		"messages": []any{map[string]any{"role": "user", "content": userText}},
		"tools":    []any{map[string]any{"name": "lookup"}},
	}
	if maxOut != nil {
		payload["max_tokens"] = *maxOut
	}
	headers := map[string]string{}
	if session != "" {
		headers["x-session-id"] = session
	}
	return &fwksched.InferenceRequest{
		RequestID:   reqID,
		TargetModel: model,
		Headers:     headers,
		Body: &fwkrh.InferenceRequestBody{
			Messages: &fwkrh.MessagesRequest{
				System:   fwkrh.AnthropicContent{Raw: "You are helpful."},
				Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: userText}}},
			},
			MaxOutputTokens: maxOut,
			Payload:         payload,
		},
	}
}

func responsesRequest(reqID, model, session string, input any, previousID string, maxOut *int64) *fwksched.InferenceRequest {
	payload := fwkrh.PayloadMap{"model": model}
	if input != nil {
		payload["input"] = input
	}
	if previousID != "" {
		payload["previous_response_id"] = previousID
	}
	if maxOut != nil {
		payload["max_output_tokens"] = *maxOut
	}
	headers := map[string]string{}
	if session != "" {
		headers["x-session-id"] = session
	}
	return &fwksched.InferenceRequest{
		RequestID:   reqID,
		TargetModel: model,
		Headers:     headers,
		Body: &fwkrh.InferenceRequestBody{
			Responses:       &fwkrh.ResponsesRequest{Input: input},
			MaxOutputTokens: maxOut,
			Payload:         payload,
		},
	}
}

func newTestPlugin(t *testing.T, counterURL string, mutate func(*Parameters)) *CapacityAdmission {
	t.Helper()
	params := Parameters{
		RTXEndpointName:       rtxName,
		CombinedModels:        []string{combinedMdl},
		RTXOnlyModels:         []string{aliasMdl},
		MaxConcurrentRequests: 2,
		MaxResidentTokens:     900_000,
		MaxRequestTokens:      400_000,
		SessionHeader:         "x-session-id",
	}
	params.Counter.BaseURL = counterURL
	if mutate != nil {
		mutate(&params)
	}
	p, err := New("test", params)
	require.NoError(t, err)
	return p
}

func schedResult(endpoints ...fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: endpoints},
		},
		PrimaryProfileName: "default",
	}
}

func (p *CapacityAdmission) holds(reqKey string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	_, ok := p.reservations[reqKey]
	return ok
}

// ---------------------------------------------------------------------------
// Reservation accounting
// ---------------------------------------------------------------------------

func TestReserveBudgetIdempotent(t *testing.T) {
	p := newTestPlugin(t, "http://127.0.0.1:1", nil)

	// Budget: maxConcurrent 2, resident 900k.
	require.Equal(t, ReasonAllowed, p.reserve("r1", "ep", 300_000))
	require.Equal(t, ReasonAllowed, p.reserve("r1", "ep", 300_000), "idempotent repeat never double counts")
	require.Equal(t, 1, p.epCount["ep"])

	require.Equal(t, ReasonAllowed, p.reserve("r2", "ep", 300_000))
	require.Equal(t, ReasonBudgetConcurrency, p.reserve("r3", "ep", 300_000))
	require.Equal(t, ReasonBudgetConcurrency, p.reserve("r4", "ep", 100))

	require.True(t, p.release("r2"))
	require.False(t, p.release("r2"), "release is idempotent")
	require.Equal(t, 1, p.epCount["ep"])

	// Resident budget bites once concurrency frees up.
	require.Equal(t, ReasonAllowed, p.reserve("r3", "ep", 600_000), "600k on top of 300k equals the resident cap")
	p.params.MaxConcurrentRequests = 3
	require.Equal(t, ReasonBudgetResident, p.reserve("r5", "ep", 100), "resident cap must reject a third hold even when concurrency permits it")

	// Re-booking the same key on another endpoint replaces without leaking.
	require.Equal(t, ReasonAllowed, p.reserve("r3", "other", 100))
	require.Equal(t, 1, p.epCount["ep"])
	require.Equal(t, 1, p.epCount["other"])
}

func TestReserveNoForceBypassExists(t *testing.T) {
	// The API surface no longer offers a bypass: the ceilings are absolute.
	p := newTestPlugin(t, "http://127.0.0.1:1", func(par *Parameters) { par.MaxConcurrentRequests = 1 })
	require.Equal(t, ReasonAllowed, p.reserve("only", "ep", 10))
	require.Equal(t, ReasonBudgetConcurrency, p.reserve("second", "ep", 10))
	require.Equal(t, ReasonBudgetConcurrency, p.reserve("tiny", "ep", 0), "even zero tokens cannot bypass concurrency accounting")
}

func TestReserveConcurrentExclusion(t *testing.T) {
	p := newTestPlugin(t, "http://127.0.0.1:1", func(par *Parameters) {
		par.MaxConcurrentRequests = 10
		par.MaxResidentTokens = 400_000
		par.MaxRequestTokens = 400_000
	})

	var wg sync.WaitGroup
	var wins atomic.Int64
	for i := range 10 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if p.reserve(fmt.Sprintf("req-%d", i), "ep", 200_000) == ReasonAllowed {
				wins.Add(1)
			}
		}(i)
	}
	wg.Wait()
	// Resident budget 400k / 200k per request => exactly 2 winners.
	require.Equal(t, int64(2), wins.Load())
	require.Equal(t, 2, p.epCount["ep"])
	require.EqualValues(t, 400_000, p.epTokens["ep"])
}

// ---------------------------------------------------------------------------
// Startup gate & reconciliation
// ---------------------------------------------------------------------------

func TestStartupGateBlocksUntilIdleObserved(t *testing.T) {
	cs, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	ctx := context.Background()

	// Gate closed and metrics report busy: RTX excluded, no counter call.
	rtx := newEndpoint(rtxPod, busyMetrics())
	out := p.Filter(ctx, chatRequest("g1", combinedMdl, "s-g1", false, new(int64(100))), []fwksched.Endpoint{dgxEndpoint(), rtx})
	require.Len(t, out, 1)
	require.False(t, p.isRTX(out[0]))
	require.Equal(t, int64(0), cs.chatCalls.Load())

	// Fresh idle metrics open the gate; the same cycle recomputes and admits.
	out = p.Filter(ctx, chatRequest("g2", combinedMdl, "s-g2", false, new(int64(100))), []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.True(t, p.isGateOpen())
	require.Len(t, out, 2)
	require.Equal(t, int64(1), cs.chatCalls.Load(), "count runs once the gate opens")

	// Stale metrics after the gate opened do not re-close it.
	stale := idleMetrics()
	stale.UpdateTime = time.Now().Add(-time.Hour)
	out = p.Filter(ctx, chatRequest("g3", combinedMdl, "s-g3", false, new(int64(100))), []fwksched.Endpoint{dgxEndpoint(), newEndpoint(rtxPod, stale)})
	require.Len(t, out, 2)
}

func TestMetricsOlderThanProcessNeverOpensGate(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	ancient := idleMetrics()
	ancient.UpdateTime = time.Now().Add(-24 * time.Hour)
	out := p.Filter(context.Background(),
		chatRequest("old-1", combinedMdl, "s-old", false, new(int64(100))),
		[]fwksched.Endpoint{dgxEndpoint(), newEndpoint(rtxPod, ancient)})
	require.Len(t, out, 1, "a day-old snapshot cannot prove a drained engine")
	require.False(t, p.isGateOpen())
}

func TestReconcileClearsOnlyHoldsTheIdleEvidenceContradicts(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true

	require.Equal(t, ReasonAllowed, p.reserve("stale", rtxIDStr(), 1_000))
	require.Equal(t, ReasonAllowed, p.reserve("fresh", rtxIDStr(), 1_000))
	// Age only the stale hold beyond the grace window.
	p.mu.Lock()
	res := p.reservations["stale"]
	res.reservedAt = time.Now().Add(-time.Minute)
	p.reservations["stale"] = res
	p.mu.Unlock()

	p.Filter(context.Background(), chatRequest("probe", combinedMdl, "s-probe", false, new(int64(10))), []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})

	// The idle snapshot (UpdateTime=now) postdates the stale hold by more
	// than the grace: it is contradicted and cleared. The fresh hold is
	// compatible with an idle engine (not forwarded yet) and must survive.
	require.False(t, p.holds("stale"), "idle engine postdating the hold by more than the grace must clear it")
	require.True(t, p.holds("fresh"), "a fresh hold must not be reconciled away by an idle scrape that cannot cover it yet")
}

func TestCancelGuardReleasesHoldWithoutDestination(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true

	ctx, cancel := context.WithCancel(context.Background())
	req := chatRequest("cg-1", combinedMdl, "s-cg", false, new(int64(100)))
	out := p.Filter(ctx, req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 2, "filter reserved the slot")
	require.True(t, p.holds("cg-1"))

	// The request dies before the destination is captured: no response hook
	// will ever fire, only the cancel guard can free the hold.
	cancel()
	require.Eventually(t, func() bool { return !p.holds("cg-1") }, time.Second, 10*time.Millisecond,
		"request-context cancellation must release the reservation")
}

// ---------------------------------------------------------------------------
// Filter policy
// ---------------------------------------------------------------------------

func TestFilterExclusionReasons(t *testing.T) {
	cases := []struct {
		name   string
		req    *fwksched.InferenceRequest
		expect int // remaining endpoints; 1 => RTX dropped, DGX kept
	}{
		{"chat ok", chatRequest("f1", combinedMdl, "s-f1", false, new(int64(100))), 2},
		{"chat over max request tokens", chatRequest("f2", combinedMdl, "s-f2", true, new(int64(100))), 1},
		{"missing explicit max output", chatRequest("f3", combinedMdl, "s-f3", false, nil), 1},
		{"responses uncountable", responsesRequest("f4", combinedMdl, "s-f4", "tell me", "", new(int64(100))), 1},
		{"unmanaged model inert", chatRequest("f5", "other-model", "s-f5", false, new(int64(100))), 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, url := newCounterServer(t)
			p := newTestPlugin(t, url, nil)
			p.gateOpen = true
			out := p.Filter(context.Background(), tc.req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
			require.Len(t, out, tc.expect)
			found := false
			for _, ep := range out {
				if !p.isRTX(ep) {
					found = true
				}
			}
			require.True(t, found, "DGX endpoint must never be dropped by capacity admission")
		})
	}
}

func TestFilterCounterFailureFailsClosedToDGX(t *testing.T) {
	cs, url := newCounterServer(t)
	cs.failChat.Store(true)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	req := chatRequest("cf", combinedMdl, "s-cf", false, new(int64(100)))
	out := p.Filter(context.Background(), req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 1)
	require.False(t, p.isRTX(out[0]))
	require.Equal(t, ReasonCounterFailed, p.decisionFor(req).reason)
}

func TestFilterReservesAtomicallyAndExcludesLoser(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, func(par *Parameters) { par.MaxConcurrentRequests = 1 })
	p.gateOpen = true
	ctx := context.Background()

	first := chatRequest("w1", combinedMdl, "s-w1", false, new(int64(100)))
	out := p.Filter(ctx, first, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 2, "first request keeps RTX and holds the reservation")
	require.True(t, p.holds("w1"))

	// The same request re-filtered (warm/test profile pass) must not
	// double count or flip to excluded.
	out = p.Filter(ctx, first, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 2)

	second := chatRequest("w2", combinedMdl, "s-w2", false, new(int64(100)))
	out = p.Filter(ctx, second, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 1, "racing loser is confined to DGX - no avoidable 503")
	require.False(t, p.isRTX(out[0]))
}

func TestOverflowSafePerRequestCeiling(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true

	// An absurd explicit max output must trip the per-request ceiling
	// without int64 overflow (input + output is never added unchecked).
	req := chatRequest("ov-1", combinedMdl, "s-ov", false, new(int64(1<<62)))
	out := p.Filter(context.Background(), req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 1)
	d := p.decisionFor(req)
	require.True(t, d.tooLarge)
	require.Equal(t, ReasonRequestTooLarge, d.reason)
}

func TestMultiSequenceAndThinkingRefused(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()
	cands := []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()}

	// chat n>1
	chatN := chatRequest("ms-1", combinedMdl, "s-ms1", false, new(int64(100)))
	chatN.Body.Payload.(fwkrh.PayloadMap)["n"] = float64(2)
	require.NoError(t, p.Produce(ctx, chatN, cands))
	require.False(t, p.decisionFor(chatN).countable)
	require.Equal(t, ReasonUncountablePayload, p.decisionFor(chatN).reason)

	// completions best_of>1
	comp := &fwksched.InferenceRequest{RequestID: "ms-2", TargetModel: combinedMdl, Headers: map[string]string{}, Body: &fwkrh.InferenceRequestBody{
		Completions:     &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}},
		MaxOutputTokens: new(int64(10)),
		Payload:         fwkrh.PayloadMap{"model": combinedMdl, "prompt": "hello", "best_of": float64(4)},
	}}
	require.NoError(t, p.Produce(ctx, comp, cands))
	require.Equal(t, ReasonUncountablePayload, p.decisionFor(comp).reason)

	// completions batched strings
	compBatch := &fwksched.InferenceRequest{RequestID: "ms-3", TargetModel: combinedMdl, Headers: map[string]string{}, Body: &fwkrh.InferenceRequestBody{
		Completions:     &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Strings: []string{"a", "b"}}},
		MaxOutputTokens: new(int64(10)),
		Payload:         fwkrh.PayloadMap{"model": combinedMdl},
	}}
	require.NoError(t, p.Produce(ctx, compBatch, cands))
	require.Equal(t, ReasonUncountablePayload, p.decisionFor(compBatch).reason)

	// anthropic thinking
	msg := messagesRequest("ms-4", combinedMdl, "s-ms4", false, new(int64(100)))
	msg.Body.Payload.(fwkrh.PayloadMap)["thinking"] = map[string]any{"type": "enabled", "budget_tokens": 1000}
	require.NoError(t, p.Produce(ctx, msg, cands))
	require.Equal(t, ReasonUncountablePayload, p.decisionFor(msg).reason)
}

// ---------------------------------------------------------------------------
// RTX-only (hardware alias)
// ---------------------------------------------------------------------------

func TestRTXOnlyNeverFallsBackToDGX(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()

	out := p.Filter(ctx, chatRequest("a1", aliasMdl, "s-a1", false, new(int64(100))), []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 1)
	require.True(t, p.isRTX(out[0]), "alias traffic must be pinned to RTX")

	// No RTX endpoint at all: candidates go empty; never DGX.
	out = p.Filter(ctx, chatRequest("a2", aliasMdl, "s-a2", false, new(int64(100))), []fwksched.Endpoint{dgxEndpoint()})
	require.Empty(t, out)
}

func TestRTXOnlyRequiresStartupEvidence(t *testing.T) {
	// Alias traffic must not land on an RTX engine the process has not
	// proven drained: same gate as the shared route.
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil) // gate closed
	ctx := context.Background()
	out := p.Filter(ctx, chatRequest("ag-1", aliasMdl, "s-ag1", false, new(int64(100))), []fwksched.Endpoint{newEndpoint(rtxPod, busyMetrics())})
	require.Empty(t, out, "busy unproven engine closes the alias path too")
	require.False(t, p.isGateOpen())

	// Engine saturated (idle gate already open) -> empty candidates, never DGX.
	p2 := newTestPlugin(t, url, func(par *Parameters) { par.MaxConcurrentRequests = 2 })
	p2.gateOpen = true
	out = p2.Filter(ctx, chatRequest("ag-2", aliasMdl, "s-ag2", false, new(int64(100))), []fwksched.Endpoint{newEndpoint(rtxPod, busyMetrics())})
	require.Empty(t, out, "running+waiting at the ceiling excludes alias traffic")
}

func TestRTXOnlyUncountableRejected(t *testing.T) {
	_, url := newCounterServer(t)
	ctx := context.Background()

	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	req := responsesRequest("a3", aliasMdl, "s-a3", "hello", "", new(int64(50)))
	p.Filter(ctx, req, []fwksched.Endpoint{rtxEndpoint()})
	err := p.PreRequest(ctx, req, schedResult(rtxEndpoint()))
	require.Error(t, err)
	var e errcommon.Error
	require.ErrorAs(t, err, &e)
	require.Equal(t, errcommon.BadRequest, e.Code)
}

func TestRTXOnlyOverBudgetFailsExplicitlyNotDGX(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, func(par *Parameters) { par.MaxConcurrentRequests = 1 })
	p.gateOpen = true
	ctx := context.Background()

	// First alias request takes the single slot through the full cycle.
	first := chatRequest("b1", aliasMdl, "s-b1", false, new(int64(100)))
	p.Filter(ctx, first, []fwksched.Endpoint{rtxEndpoint()})
	require.NoError(t, p.PreRequest(ctx, first, schedResult(rtxEndpoint())))
	require.True(t, p.holds("b1"))

	// Second alias request: budget gone at confirmation -> typed 503, never
	// a DGX placement.
	second := chatRequest("b2", aliasMdl, "s-b2", false, new(int64(100)))
	p.Filter(ctx, second, []fwksched.Endpoint{rtxEndpoint()})
	err := p.PreRequest(ctx, second, schedResult(rtxEndpoint()))
	require.Error(t, err)
	var e errcommon.Error
	require.ErrorAs(t, err, &e)
	require.Equal(t, errcommon.ServiceUnavailable, e.Code)
	require.False(t, p.holds("b2"))
}

// ---------------------------------------------------------------------------
// Session relocation stickiness & reference-only rejection
// ---------------------------------------------------------------------------

func TestRelocatedSessionStaysOnDGX(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()

	// Turn 1: small request lands on RTX and binds the session there.
	turn1 := chatRequest("s1-t1", combinedMdl, "sess-1", false, new(int64(100)))
	out := p.Filter(ctx, turn1, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 2)
	require.NoError(t, p.PreRequest(ctx, turn1, schedResult(rtxEndpoint())))
	require.True(t, p.holds("s1-t1"))

	// Complete turn 1: reservation released, binding kept.
	p.ResponseBody(ctx, turn1, &fwkrc.Response{EndOfStream: true}, rtxEndpoint().GetMetadata())
	require.False(t, p.holds("s1-t1"))

	// Turn 2: the conversation grew past the per-request ceiling (same
	// root, more history): relocates to DGX and the move is marked sticky.
	turn2 := chatRequest("s1-t2", combinedMdl, "sess-1", true, new(int64(100)))
	out = p.Filter(ctx, turn2, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 1)
	require.False(t, p.isRTX(out[0]))
	require.NoError(t, p.PreRequest(ctx, turn2, schedResult(dgxEndpoint())))

	// Turn 3: same session, small request and free capacity: the session
	// must NOT bounce back to RTX.
	turn3 := chatRequest("s1-t3", combinedMdl, "sess-1", false, new(int64(100)))
	out = p.Filter(ctx, turn3, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 1, "relocated session excludes RTX even when budget is free")
	require.False(t, p.isRTX(out[0]))
	require.NoError(t, p.PreRequest(ctx, turn3, schedResult(dgxEndpoint())))

	// A different session is unaffected by sess-1's stickiness.
	other := chatRequest("s2-t1", combinedMdl, "sess-2", false, new(int64(100)))
	out = p.Filter(ctx, other, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 2)
}

func TestRelocatedSessionPinsExactDGXEndpoint(t *testing.T) {
	// No-bounce is endpoint-exact: after relocating to a specific DGX
	// replica, later turns stay on that replica while it is a candidate.
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()

	dgxA := newEndpoint("dgx-a", idleMetrics())
	dgxB := newEndpoint("dgx-b", idleMetrics())

	// Bind to RTX, complete, then relocate via an oversized turn to dgxA.
	turn1 := chatRequest("pin-t1", combinedMdl, "sess-pin", false, new(int64(100)))
	p.Filter(ctx, turn1, []fwksched.Endpoint{dgxA, dgxB, rtxEndpoint()})
	require.NoError(t, p.PreRequest(ctx, turn1, schedResult(rtxEndpoint())))
	p.ResponseBody(ctx, turn1, &fwkrc.Response{EndOfStream: true}, rtxEndpoint().GetMetadata())

	turn2 := chatRequest("pin-t2", combinedMdl, "sess-pin", true, new(int64(100)))
	out := p.Filter(ctx, turn2, []fwksched.Endpoint{dgxA, dgxB, rtxEndpoint()})
	require.Len(t, out, 2, "relocation only drops RTX")
	require.NoError(t, p.PreRequest(ctx, turn2, schedResult(dgxA)))

	// Turn 3: small again; candidates include both DGX replicas.
	turn3 := chatRequest("pin-t3", combinedMdl, "sess-pin", false, new(int64(100)))
	out = p.Filter(ctx, turn3, []fwksched.Endpoint{dgxA, dgxB, rtxEndpoint()})
	require.Len(t, out, 1, "relocated session is pinned to its DGX endpoint")
	require.Equal(t, p.epID(dgxA), p.epID(out[0]))

	// If the pinned replica disappears, DGX selection resumes normally.
	out = p.Filter(ctx, chatRequest("pin-t4", combinedMdl, "sess-pin", false, new(int64(100))), []fwksched.Endpoint{dgxB, rtxEndpoint()})
	require.Len(t, out, 1)
	require.Equal(t, p.epID(dgxB), p.epID(out[0]))
}

func TestBudgetPressureRelocationWithoutOversizedHistory(t *testing.T) {
	// Relocation also triggers on pure budget pressure (no tooLarge): the
	// session binds to RTX, a concurrent tenant consumes the budget, and
	// the next turn relocates to DGX and stays.
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, func(par *Parameters) { par.MaxConcurrentRequests = 1 })
	p.gateOpen = true
	ctx := context.Background()

	turn1 := chatRequest("p-t1", combinedMdl, "sess-p", false, new(int64(100)))
	p.Filter(ctx, turn1, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.NoError(t, p.PreRequest(ctx, turn1, schedResult(rtxEndpoint())))
	p.ResponseBody(ctx, turn1, &fwkrc.Response{EndOfStream: true}, rtxEndpoint().GetMetadata())

	// Another request saturates the single slot (held, not completed).
	holder := chatRequest("p-holder", combinedMdl, "sess-holder", false, new(int64(100)))
	p.Filter(ctx, holder, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.True(t, p.holds("p-holder"))

	turn2 := chatRequest("p-t2", combinedMdl, "sess-p", false, new(int64(100)))
	out := p.Filter(ctx, turn2, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 1)
	require.False(t, p.isRTX(out[0]))
	require.NoError(t, p.PreRequest(ctx, turn2, schedResult(dgxEndpoint())))

	// Budget frees; turn 3 must still go DGX (sticky relocation).
	p.release("p-holder")
	turn3 := chatRequest("p-t3", combinedMdl, "sess-p", false, new(int64(100)))
	out = p.Filter(ctx, turn3, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 1)
	require.False(t, p.isRTX(out[0]))
}

func TestReferenceOnlyBoundRTXRejectedNeverMovedOrAdmitted(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, func(par *Parameters) { par.MaxConcurrentRequests = 1 })
	p.gateOpen = true
	ctx := context.Background()

	// Bind the session to RTX with a live held reservation.
	turn1 := chatRequest("ro-t1", combinedMdl, "sess-ro", false, new(int64(100)))
	p.Filter(ctx, turn1, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.NoError(t, p.PreRequest(ctx, turn1, schedResult(rtxEndpoint())))

	// Reference-only Responses for the same session: rejected clearly.
	ref := responsesRequest("ro-t2", combinedMdl, "sess-ro", nil, "resp-1", nil)
	ref.Body.RawBody = []byte(`{"model":"qwen38-flash-next","previous_response_id":"resp-1"}`)
	out := p.Filter(ctx, ref, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 1, "reference-only is unaccountable: RTX is dropped from its candidates")
	require.False(t, p.isRTX(out[0]))

	// DGX destination: rejected (must not be relocated).
	err := p.PreRequest(ctx, ref, schedResult(dgxEndpoint()))
	require.Error(t, err)
	var e errcommon.Error
	require.ErrorAs(t, err, &e)
	require.Equal(t, errcommon.PreconditionFailed, e.Code)

	// RTX destination: also rejected (no force-admit on an unknown budget).
	ref2 := responsesRequest("ro-t3", combinedMdl, "sess-ro", nil, "resp-1", nil)
	ref2.Body.Payload.(fwkrh.PayloadMap)["previous_response_id"] = "resp-1"
	require.NoError(t, p.Produce(ctx, ref2, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()}))
	err = p.PreRequest(ctx, ref2, schedResult(rtxEndpoint()))
	require.Error(t, err)
	require.ErrorAs(t, err, &e)
	require.Equal(t, errcommon.PreconditionFailed, e.Code)

	// Nothing was force-reserved for the references.
	require.False(t, p.holds("ro-t2"))
	require.False(t, p.holds("ro-t3"))
}

func TestReferenceOnlyUnboundGoesDGX(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()

	ref := responsesRequest("ru-1", combinedMdl, "sess-unknown", nil, "resp-x", new(int64(50)))
	out := p.Filter(ctx, ref, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 1)
	require.False(t, p.isRTX(out[0]))
	require.Equal(t, ReasonReferenceOnlyUnbound, p.decisionFor(ref).reason)
	require.NoError(t, p.PreRequest(ctx, ref, schedResult(dgxEndpoint())))
}

// ---------------------------------------------------------------------------
// PreRequest / ResponseBody completion semantics
// ---------------------------------------------------------------------------

func TestPreRequestReleasesWhenDGXWinsScoring(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()

	req := chatRequest("rel-1", combinedMdl, "s-rel", false, new(int64(100)))
	p.Filter(ctx, req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.True(t, p.holds("rel-1"), "filter reserved")

	require.NoError(t, p.PreRequest(ctx, req, schedResult(dgxEndpoint())))
	require.False(t, p.holds("rel-1"), "PreRequest must release a reservation the destination does not use")

	// Idempotent repeat is safe.
	require.NoError(t, p.PreRequest(ctx, req, schedResult(dgxEndpoint())))
}

func TestPreRequestRebooksWhenHoldLost(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()

	req := chatRequest("nr-1", combinedMdl, "s-nr", false, new(int64(100)))
	p.Filter(ctx, req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.True(t, p.holds("nr-1"))
	p.release("nr-1") // simulate a lost hold after filtering

	// Budget permits: PreRequest re-books rather than 503-ing.
	require.NoError(t, p.PreRequest(ctx, req, schedResult(rtxEndpoint())))
	require.True(t, p.holds("nr-1"))

	// Budget denies: the single slot is held by an unrelated tenant. Even
	// though the filter cached a countable decision (and excluded RTX), a
	// destination that contradicts it must fail with a typed 503 rather
	// than forward unaccounted traffic to RTX.
	p2 := newTestPlugin(t, url, func(par *Parameters) { par.MaxConcurrentRequests = 1 })
	p2.gateOpen = true
	require.Equal(t, ReasonAllowed, p2.reserve("blocker", rtxIDStr(), 100))
	req2 := chatRequest("nr-2", combinedMdl, "s-nr2", false, new(int64(100)))
	out2 := p2.Filter(ctx, req2, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out2, 1, "filter excludes RTX while the slot is taken")
	err := p2.PreRequest(ctx, req2, schedResult(rtxEndpoint()))
	require.Error(t, err)
	var e errcommon.Error
	require.ErrorAs(t, err, &e)
	require.Equal(t, errcommon.ServiceUnavailable, e.Code)
}

func TestPreRequestRTXOnlyNonRTXDestinationFailsClosed(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()

	req := chatRequest("pin-1", aliasMdl, "s-pin", false, new(int64(100)))
	p.Filter(ctx, req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	err := p.PreRequest(ctx, req, schedResult(dgxEndpoint()))
	require.Error(t, err, "alias traffic must never be forwarded to DGX")
	var e errcommon.Error
	require.ErrorAs(t, err, &e)
	require.Equal(t, errcommon.PreconditionFailed, e.Code)
}

func TestResponseBodyReleasesOnce(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()

	req := chatRequest("rb-1", combinedMdl, "s-rb", false, new(int64(100)))
	p.Filter(ctx, req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.NoError(t, p.PreRequest(ctx, req, schedResult(rtxEndpoint())))
	require.True(t, p.holds("rb-1"))

	p.ResponseBody(ctx, req, &fwkrc.Response{}, rtxEndpoint().GetMetadata())
	require.True(t, p.holds("rb-1"), "non-final chunks must not release")

	p.ResponseBody(ctx, req, &fwkrc.Response{EndOfStream: true, TerminationCause: fwkrc.TerminationCauseClientDisconnect}, rtxEndpoint().GetMetadata())
	require.False(t, p.holds("rb-1"))

	// Cancellation/abort repeat calls are idempotent.
	p.ResponseBody(ctx, req, &fwkrc.Response{EndOfStream: true}, rtxEndpoint().GetMetadata())
	require.False(t, p.holds("rb-1"))
}

// ---------------------------------------------------------------------------
// Counter behavior
// ---------------------------------------------------------------------------

func TestCounterProtocolsAndFailure(t *testing.T) {
	cs, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()
	cands := []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()}

	// Chat uses /tokenize exactly once per request across the full cycle.
	chatBefore := cs.chatCalls.Load()
	req := chatRequest("c1", combinedMdl, "s-c1", false, new(int64(100)))
	require.NoError(t, p.Produce(ctx, req, cands))
	p.Filter(ctx, req, cands)
	require.NoError(t, p.PreRequest(ctx, req, schedResult(rtxEndpoint())))
	require.Equal(t, int64(1), cs.chatCalls.Load()-chatBefore, "one tokenize call per request lifecycle")
	require.EqualValues(t, smallCount+100, p.decisionFor(req).tokens)

	// Anthropic uses count_tokens with system+tools preserved.
	nativeBefore := cs.nativeCalls.Load()
	nreq := messagesRequest("c2", combinedMdl, "s-c2", false, new(int64(100)))
	require.NoError(t, p.Produce(ctx, nreq, cands))
	p.Filter(ctx, nreq, cands)
	require.NoError(t, p.PreRequest(ctx, nreq, schedResult(rtxEndpoint())))
	require.Equal(t, int64(1), cs.nativeCalls.Load()-nativeBefore)
	require.True(t, p.decisionFor(nreq).countable)
	p.ResponseBody(ctx, req, &fwkrc.Response{EndOfStream: true}, rtxEndpoint().GetMetadata())
	p.ResponseBody(ctx, nreq, &fwkrc.Response{EndOfStream: true}, rtxEndpoint().GetMetadata())

	// Counter failure => uncountable => DGX-only candidates.
	cs.failChat.Store(true)
	freq := chatRequest("c3", combinedMdl, "s-c3", false, new(int64(100)))
	require.NoError(t, p.Produce(ctx, freq, cands))
	out := p.Filter(ctx, freq, cands)
	require.Len(t, out, 1)
	require.False(t, p.isRTX(out[0]))
}

func TestCounterRefusesChatTemplateOverride(t *testing.T) {
	cs, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()

	req := chatRequest("ct-1", combinedMdl, "s-ct", false, new(int64(100)))
	req.Body.Payload.(fwkrh.PayloadMap)["chat_template"] = "custom {{messages}}"
	before := cs.chatCalls.Load()
	require.NoError(t, p.Produce(ctx, req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()}))
	d := p.decisionFor(req)
	require.False(t, d.countable)
	require.Equal(t, ReasonUncountablePayload, d.reason)
	require.Equal(t, before, cs.chatCalls.Load(), "refused fields must not hit the counter")
}

func TestCompletionsTokenIDsCountedLocally(t *testing.T) {
	cs, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()

	req := &fwksched.InferenceRequest{RequestID: "tok-1", TargetModel: combinedMdl, Headers: map[string]string{}, Body: &fwkrh.InferenceRequestBody{
		Completions:     &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{TokenIDs: [][]uint32{{1, 2, 3, 4}}}},
		MaxOutputTokens: new(int64(10)),
		Payload:         fwkrh.PayloadMap{"model": combinedMdl},
	}}
	before := cs.chatCalls.Load()
	require.NoError(t, p.Produce(ctx, req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()}))
	d := p.decisionFor(req)
	require.True(t, d.countable)
	require.EqualValues(t, 4, d.input)
	require.Equal(t, before, cs.chatCalls.Load())
}

func TestNativeLargeHistoryCountsViaCounter(t *testing.T) {
	// Genuine >400k-token next-turn history: the RTX engine's own counter
	// reports the size and the request is excluded from RTX (DGX serves).
	cs, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()

	req := chatRequest("big-1", combinedMdl, "s-big", true, new(int64(100)))
	require.NoError(t, p.Produce(ctx, req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()}))
	d := p.decisionFor(req)
	require.True(t, d.tooLarge)
	require.EqualValues(t, bigCount, d.input)
	require.False(t, p.holds("big-1"), "oversized history must never reserve RTX capacity")
	require.Positive(t, cs.chatCalls.Load(), "the real engine counter decides size")

	out := p.Filter(ctx, req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})
	require.Len(t, out, 1)
	require.False(t, p.isRTX(out[0]))
}

// ---------------------------------------------------------------------------
// Session store boundedness
// ---------------------------------------------------------------------------

func TestSessionStoreBounded(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, func(par *Parameters) { par.MaxSessionEntries = 4 })
	for i := range 50 {
		p.recordPlacement(fmt.Sprintf("k%d", i), "ep", false, false)
	}
	p.mu.Lock()
	size := p.sessions.Len()
	p.mu.Unlock()
	require.LessOrEqual(t, size, 4, "session store must stay LRU-bounded")
}

func TestSessionTTLExpiry(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, func(par *Parameters) { par.SessionTTL = "10ms" })
	p.recordPlacement("gone", rtxIDStr(), false, true)
	time.Sleep(30 * time.Millisecond)
	p.mu.Lock()
	_, ok := p.lookupPlacement("gone")
	p.mu.Unlock()
	require.False(t, ok, "expired placements read as absent")
}

// ---------------------------------------------------------------------------
// Factory validation
// ---------------------------------------------------------------------------

func TestFactoryRequiresCoreParameters(t *testing.T) {
	_, err := New("bad", Parameters{})
	require.Error(t, err)

	_, err = New("bad", Parameters{
		RTXEndpointName: rtxName, CombinedModels: []string{combinedMdl},
		MaxConcurrentRequests: 2, MaxResidentTokens: 900_000, MaxRequestTokens: 1_000_000,
	})
	require.Error(t, err, "maxRequestTokens must not exceed maxResidentTokens")

	_, err = New("bad", Parameters{
		RTXEndpointName:       rtxName,
		CombinedModels:        []string{combinedMdl},
		MaxConcurrentRequests: 2,
		MaxResidentTokens:     900_000,
		MaxRequestTokens:      400_000,
		// counter.baseURL missing
	})
	require.Error(t, err)
}

// ---------------------------------------------------------------------------
// DumpState sanity (must never leak prompt material)
// ---------------------------------------------------------------------------

func TestDumpStateSanitized(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	p.gateOpen = true
	ctx := context.Background()
	req := chatRequest("dump-1", combinedMdl, "s-dump", false, new(int64(100)))
	p.Filter(ctx, req, []fwksched.Endpoint{dgxEndpoint(), rtxEndpoint()})

	raw, err := p.DumpState()
	require.NoError(t, err)
	require.NotContains(t, string(raw), defaultRoot)
	require.NotContains(t, string(raw), "helpful")
	require.Contains(t, string(raw), "dump-1")
	require.NotContains(t, string(raw), "forced")
}

// ---------------------------------------------------------------------------
// Naming and label identification
// ---------------------------------------------------------------------------

func TestRTXIdentification(t *testing.T) {
	_, url := newCounterServer(t)
	p := newTestPlugin(t, url, nil)
	require.True(t, p.isRTX(newEndpoint(rtxPod, nil)), "deployment suffix tolerated")
	require.True(t, p.isRTX(newEndpoint(rtxName, nil)), "bare name matches")
	require.False(t, p.isRTX(newEndpoint(rtxName+"x", nil)), "no prefix-boundary violation")
	require.False(t, p.isRTX(newEndpoint(dgxName, nil)))

	pl := newTestPlugin(t, url, func(par *Parameters) {
		par.RTXEndpointName = ""
		par.RTXEndpointLabel = "llm-d.ai/capacity-class=rtx6000"
	})
	labelled := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:     types.NamespacedName{Namespace: testNS, Name: "anything"},
		Name:   "anything",
		Labels: map[string]string{"llm-d.ai/capacity-class": "rtx6000"},
	}, nil, nil)
	require.True(t, pl.isRTX(labelled))
	require.False(t, pl.isRTX(dgxEndpoint()))
}
