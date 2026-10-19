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
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// counterClient asks the RTX engine itself to count prompt tokens, so the
// admission number is produced by the tokenizer that will run the request.
// Two endpoints are used, matching the protocol of the request:
//
//   - OpenAI chat/completions: vLLM POST {base}/tokenize
//   - Anthropic Messages:      POST {base}/v1/messages/count_tokens
//
// The base URL must address the RTX engine Service directly; routing the
// count through the EPP would recurse.
type counterClient struct {
	client  *http.Client
	baseURL string
	timeout time.Duration

	openAITokenizePath string
	anthropicCountPath string
	modelOverride      string
}

func newCounterClient(cfg CounterParameters) (*counterClient, error) {
	base := strings.TrimRight(cfg.BaseURL, "/")
	timeout, err := time.ParseDuration(cfg.Timeout)
	if err != nil {
		return nil, fmt.Errorf("invalid counter.timeout: %w", err)
	}
	transport, err := newCounterTransport(cfg)
	if err != nil {
		return nil, err
	}
	return &counterClient{
		client: &http.Client{Transport: otelhttp.NewTransport(transport, otelhttp.WithSpanNameFormatter(
			// Name the outbound span after the count route so traces
			// identify admission counting calls.
			func(_ string, r *http.Request) string { return "capacity_count " + r.URL.Path },
		))},
		baseURL:            base,
		timeout:            timeout,
		openAITokenizePath: cfg.OpenAITokenizePath,
		anthropicCountPath: cfg.AnthropicCountPath,
		modelOverride:      cfg.Model,
	}, nil
}

// newCounterTransport mirrors the tokenizer producer's render transport:
// HTTP/2 off (vLLM does not support it), modest idle pool, explicit TLS
// options.
func newCounterTransport(cfg CounterParameters) (*http.Transport, error) {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ForceAttemptHTTP2 = false
	transport.MaxIdleConns = 16
	transport.MaxIdleConnsPerHost = 8
	transport.TLSNextProto = map[string]func(string, *tls.Conn) http.RoundTripper{}

	if cfg.InsecureSkipVerify || cfg.CACertPath != "" || cfg.ClientCertPath != "" || cfg.ClientKeyPath != "" {
		tlsCfg := &tls.Config{
			MinVersion:         tls.VersionTLS12,
			InsecureSkipVerify: cfg.InsecureSkipVerify, //nolint:gosec // operator opt-in for self-signed dev endpoints
		}
		if cfg.CACertPath != "" {
			pem, err := os.ReadFile(cfg.CACertPath)
			if err != nil {
				return nil, fmt.Errorf("reading counter.caCertPath: %w", err)
			}
			pool := x509.NewCertPool()
			if !pool.AppendCertsFromPEM(pem) {
				return nil, fmt.Errorf("counter.caCertPath %q contains no usable certificates", cfg.CACertPath)
			}
			tlsCfg.RootCAs = pool
		}
		if (cfg.ClientCertPath == "") != (cfg.ClientKeyPath == "") {
			return nil, fmt.Errorf("counter.clientCertPath and counter.clientKeyPath must be set together")
		}
		if cfg.ClientCertPath != "" {
			cert, err := tls.LoadX509KeyPair(cfg.ClientCertPath, cfg.ClientKeyPath)
			if err != nil {
				return nil, fmt.Errorf("loading counter client certificate: %w", err)
			}
			tlsCfg.Certificates = []tls.Certificate{cert}
		}
		transport.TLSClientConfig = tlsCfg
	}
	return transport, nil
}

// countOpenAI counts a chat (messages) or completions (prompt) body via the
// engine's /tokenize endpoint. Only fields that affect tokenization and are
// understood by the endpoint are forwarded; anything else in the body is
// irrelevant to the token count.
func (c *counterClient) countOpenAI(ctx context.Context, requestModel string, payload fwkrh.PayloadMap) (int64, error) {
	body := map[string]any{"model": c.model(requestModel)}
	for _, field := range []string{"messages", "prompt", "tools", "chat_template_kwargs", "add_generation_prompt", "continue_final_message"} {
		if v, ok := payload[field]; ok && v != nil {
			body[field] = v
		}
	}
	if _, hasMessages := body["messages"]; !hasMessages {
		if _, hasPrompt := body["prompt"]; !hasPrompt {
			return 0, fmt.Errorf("payload carries neither messages nor prompt")
		}
	}
	return c.postCount(ctx, c.openAITokenizePath, body, "count", "tokens")
}

// countAnthropic counts a Messages body via the engine's native
// /v1/messages/count_tokens endpoint, forwarding system, messages and tools
// exactly as the generation request carries them.
func (c *counterClient) countAnthropic(ctx context.Context, requestModel string, payload fwkrh.PayloadMap) (int64, error) {
	body := map[string]any{"model": c.model(requestModel)}
	for _, field := range []string{"system", "messages", "tools"} {
		if v, ok := payload[field]; ok && v != nil {
			body[field] = v
		}
	}
	if _, ok := body["messages"]; !ok {
		return 0, fmt.Errorf("payload carries no messages")
	}
	return c.postCount(ctx, c.anthropicCountPath, body, "input_tokens", "")
}

func (c *counterClient) model(requestModel string) string {
	if c.modelOverride != "" {
		return c.modelOverride
	}
	return requestModel
}

// postCount posts the count request and extracts the token count. countField
// names the numeric field carrying the result; when it is absent (vLLM
// /tokenize variants), tokenArrayField, if non-empty, is measured instead.
func (c *counterClient) postCount(ctx context.Context, path string, reqBody map[string]any, countField, tokenArrayField string) (int64, error) {
	raw, err := json.Marshal(reqBody)
	if err != nil {
		return 0, fmt.Errorf("marshaling count request: %w", err)
	}

	callCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(callCtx, http.MethodPost, c.baseURL+path, bytes.NewReader(raw))
	if err != nil {
		return 0, fmt.Errorf("building count request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return 0, fmt.Errorf("calling count endpoint: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("count endpoint %s returned status %d", path, resp.StatusCode)
	}

	var parsed map[string]json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return 0, fmt.Errorf("decoding count response: %w", err)
	}
	if v, ok := parsed[countField]; ok {
		n, err := strconv.ParseInt(strings.Trim(string(v), `"`), 10, 64)
		if err != nil {
			return 0, fmt.Errorf("count endpoint returned unreadable %s: %s", countField, string(v))
		}
		if n < 0 {
			return 0, fmt.Errorf("count endpoint returned negative %s: %d", countField, n)
		}
		return n, nil
	}
	if tokenArrayField != "" {
		if v, ok := parsed[tokenArrayField]; ok {
			var arr []json.RawMessage
			if err := json.Unmarshal(v, &arr); err != nil {
				return 0, fmt.Errorf("count endpoint returned unreadable %s: %w", tokenArrayField, err)
			}
			return int64(len(arr)), nil
		}
	}
	return 0, fmt.Errorf("count response carries neither %q nor %q", countField, tokenArrayField)
}

// payloadMap returns the decoded JSON body map for the request, or nil when
// the payload is not a JSON map (gRPC, unparsed passthrough).
func payloadMap(body *fwkrh.InferenceRequestBody) fwkrh.PayloadMap {
	if body == nil {
		return nil
	}
	switch payload := body.WirePayload().(type) {
	case fwkrh.PayloadMap:
		return payload
	case fwkrh.RawPayload:
		var parsed fwkrh.PayloadMap
		if err := json.Unmarshal(payload, &parsed); err != nil {
			return nil
		}
		return parsed
	default:
		return nil
	}
}

// countInput dispatches to the counter that matches the request's API kind.
// It returns ReasonAllowed with the input token count, or the exclusion
// reason; any non-allowed reason means "RTX ineligible" and the DGX path is
// unaffected.
//
// Chat-template overrides (chat_template) and HF documents fields are
// refused: we will not pretend to know how the engine would render them, so
// such requests simply never occupy the RTX budget.
func (p *CapacityAdmission) countInput(ctx context.Context, request *fwksched.InferenceRequest, kind apiKind) (int64, Reason) {
	logger := log.FromContext(ctx)
	fail := func(reason Reason, err error) (int64, Reason) {
		logger.V(logutil.DEBUG).Info("capacity-admission count refused",
			"requestID", request.RequestID, "model", request.TargetModel,
			"kind", kind, "reason", reason, "error", fmt.Sprint(err))
		return 0, reason
	}

	switch kind {
	case apiChat:
		payload := payloadMap(request.Body)
		if payload == nil {
			return fail(ReasonUncountablePayload, nil)
		}
		if refusedChatFields(payload) || multiSequence(payload) {
			// n>1 / best_of>1 replicates sequences and outputs beyond the
			// single explicit max we reserve; no multi-sequence accounting
			// yet, so such requests stay off RTX.
			return fail(ReasonUncountablePayload, nil)
		}
		n, err := p.counter.countOpenAI(ctx, request.TargetModel, payload)
		if err != nil {
			return fail(ReasonCounterFailed, err)
		}
		return n, ReasonAllowed

	case apiMessages:
		payload := payloadMap(request.Body)
		if payload == nil {
			return fail(ReasonUncountablePayload, nil)
		}
		if _, hasThinking := payload["thinking"]; hasThinking || multiSequence(payload) {
			// Extended thinking changes the generated-token shape beyond
			// the single explicit max_tokens we reserve, and n>1 multiplies
			// sequences; the engine count endpoints do not settle either, so
			// we refuse rather than under-reserve.
			return fail(ReasonUncountablePayload, nil)
		}
		n, err := p.counter.countAnthropic(ctx, request.TargetModel, payload)
		if err != nil {
			return fail(ReasonCounterFailed, err)
		}
		return n, ReasonAllowed

	case apiCompletions:
		c := request.Body.Completions
		if c == nil {
			return fail(ReasonNotParsed, nil)
		}
		// Prompt delivered as token ids: the client chose the tokens, the
		// engine renders them verbatim, so the count is exact without a call.
		if len(c.Prompt.TokenIDs) == 1 {
			return int64(len(c.Prompt.TokenIDs[0])), ReasonAllowed
		}
		if len(c.Prompt.TokenIDs) > 1 || len(c.Prompt.Strings) > 1 {
			// Batched prompts (token ids or strings) multiply sequences and
			// outputs; refusing is honest - estimating either dimension is
			// not.
			return fail(ReasonUncountablePayload, nil)
		}
		payload := payloadMap(request.Body)
		if payload == nil || multiSequence(payload) {
			return fail(ReasonUncountablePayload, nil)
		}
		n, err := p.counter.countOpenAI(ctx, request.TargetModel, payload)
		if err != nil {
			return fail(ReasonCounterFailed, err)
		}
		return n, ReasonAllowed

	case apiGenerate:
		g := request.Body.Generate
		switch {
		case g != nil && g.Features == nil && len(g.TokenIDs) > 0:
			return int64(len(g.TokenIDs)), ReasonAllowed
		case request.Body.TokenizedRequest != nil && request.Body.TokenizedRequest.TokenCount() > 0:
			// Parser-provided tokenization is only trusted when it is real
			// token ids, not an estimate.
			return int64(request.Body.TokenizedRequest.TokenCount()), ReasonAllowed
		default:
			return fail(ReasonUncountablePayload, nil)
		}

	default:
		return fail(ReasonUncountableAPI, nil)
	}
}

// refusedChatFields reports whether a chat payload carries fields whose
// tokenization effect the counter cannot faithfully reproduce.
func refusedChatFields(payload fwkrh.PayloadMap) bool {
	for _, field := range []string{"chat_template", "documents"} {
		if v, ok := payload[field]; ok && v != nil {
			switch t := v.(type) {
			case string:
				if strings.TrimSpace(t) != "" {
					return true
				}
			default:
				return true
			}
		}
	}
	return false
}

// multiSequence reports whether a payload requests more than one independent
// completion (n or best_of above one). Each extra sequence replicates input
// and output KV beyond the single explicit max-output we reserve, and there
// is no multi-sequence accounting yet: such requests stay off RTX.
func multiSequence(payload fwkrh.PayloadMap) bool {
	for _, field := range []string{"n", "best_of"} {
		switch v := payload[field].(type) {
		case float64:
			if v > 1 {
				return true
			}
		case int64:
			if v > 1 {
				return true
			}
		case json.Number:
			if f, err := v.Float64(); err == nil && f > 1 {
				return true
			}
		}
	}
	return false
}
