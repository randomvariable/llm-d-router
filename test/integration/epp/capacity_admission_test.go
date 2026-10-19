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

package epp

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	testutil "github.com/llm-d/llm-d-router/pkg/epp/util/testing"
	"github.com/llm-d/llm-d-router/test/integration"
)

// Capacity-admission integration fixtures: one RTX pod (Deployment-style
// random suffix on the name) plus one DGX pod, served under the common
// model name. The counter is a fake RTX engine token-count service whose
// answer depends only on whether the request text carries the size marker.

const (
	capRTXPodName = "rtx6000-qwen38-flash-next-7c9d8e-rtxpt"
	capDGXPodName = "dgx-qwen38-flash-next-5b6f7a-dgxpt"
	capRTXIP      = "192.168.1.231"
	capDGXIP      = "192.168.1.232"

	capCombinedModel = "qwen38-flash-next"
	capAliasModel    = "qwen38-flash-next-rtx6000"
	capBigMarker     = "BIGCTX"
	capRoot          = "root message for capacity session"
)

// newCapCounter starts the fake RTX token counter. /tokenize and
// /v1/messages/count_tokens report a huge count for bodies carrying the
// marker and a small one otherwise; the failure switch proves DGX keeps
// serving when counting is unavailable.
func newCapCounter(t *testing.T, fail *atomic.Bool) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	count := func(body []byte) int {
		if strings.Contains(string(body), capBigMarker) {
			return 450_000
		}
		return 5_000
	}
	mux.HandleFunc("/tokenize", func(w http.ResponseWriter, r *http.Request) {
		if fail != nil && fail.Load() {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.NewEncoder(w).Encode(map[string]any{"count": count(body)})
	})
	mux.HandleFunc("/v1/messages/count_tokens", func(w http.ResponseWriter, r *http.Request) {
		if fail != nil && fail.Load() {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.NewEncoder(w).Encode(map[string]any{"input_tokens": count(body)})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func capConfig(counterURL string) string {
	return fmt.Sprintf(`
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: mock-metrics-source
  - type: queue-scorer
  - type: capacity-admission
    name: qwen-capacity
    parameters:
      rtxEndpointName: rtx6000-qwen38-flash-next
      combinedModels: ["%[2]s"]
      rtxOnlyModels: ["%[3]s"]
      maxConcurrentRequests: 2
      maxResidentTokens: 900000
      maxRequestTokens: 400000
      metricsMaxAge: "10m"
      staleReservationGrace: "10m"
      counter:
        baseURL: "%[1]s"
        timeout: "2s"
      sessionHeader: x-session-id
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-scorer
      - pluginRef: qwen-capacity
      - pluginRef: max-score-picker
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  sources:
  - pluginRef: mock-metrics-source
`, counterURL, capCombinedModel, capAliasModel)
}

// capWithPods creates the RTX and DGX pods with ready-state metrics, keyed
// the way the mock metrics source expects (name + "-rank-0").
func capWithPods(t *testing.T, h *TestHarness) {
	t.Helper()
	metricsMap := map[types.NamespacedName]*fwkdl.Metrics{}
	for _, name := range []string{capRTXPodName, capDGXPodName} {
		metricsMap[types.NamespacedName{Namespace: h.Namespace, Name: name + "-rank-0"}] = &fwkdl.Metrics{
			RunningRequestsSize: 0,
			WaitingQueueSize:    0,
			KVCacheUsagePercent: 0.05,
			ActiveModels:        map[string]int{},
			WaitingModels:       map[string]int{},
			UpdateTime:          time.Now(),
		}
	}
	// Prefer idle RTX initially; capacity exclusion must still move the next
	// turn to DGX even though its queue score is worse.
	metricsMap[types.NamespacedName{Namespace: h.Namespace, Name: capDGXPodName + "-rank-0"}].WaitingQueueSize = 1
	h.metricsBackend.SetPodMetrics(metricsMap)

	for i, name := range []string{capRTXPodName, capDGXPodName} {
		ip := capRTXIP
		if i == 1 {
			ip = capDGXIP
		}
		pod := testutil.MakePod(name).
			Namespace(h.Namespace).
			ReadyCondition().
			Labels(map[string]string{"app": testPoolName}).
			IP(ip).
			Complete().
			ObjRef()
		intendedStatus := pod.Status
		require.NoError(t, k8sClient.Create(h.ctx, pod), "failed to create pod %s", name)
		pod.Status = intendedStatus
		require.NoError(t, k8sClient.Status().Update(h.ctx, pod), "failed to update status for pod %s", name)
	}
}

// sendCapacityChat drives one chat-completions transaction through the EPP
// and returns the routed destination endpoint (host:port).
func sendCapacityChat(t *testing.T, h *TestHarness, reqID, session, model, content string) string {
	t.Helper()
	body := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": content},
		},
		"max_tokens": 50,
	}
	raw, err := json.Marshal(body)
	require.NoError(t, err)

	reqHeaders := map[string]string{
		metadata.ObjectiveKey:        modelMyModel,
		metadata.ModelNameRewriteKey: model,
		reqcommon.RequestIDHeaderKey: reqID,
		":path":                      "/v1/chat/completions",
		":method":                    "POST",
	}
	if session != "" {
		reqHeaders["x-session-id"] = session
	}

	requests := integration.ReqRaw(reqHeaders, string(raw))
	requests = append(requests, ReqResponseOnly(
		map[string]string{"content-type": "application/json", "status": "200"},
		`{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"hi","role":"assistant"}}],"model":"`+model+`","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":5}}`,
	)...)

	client, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(t.Context())
	require.NoError(t, err)
	responses, err := integration.StreamedRequest(t, client, requests, 4)
	require.NoError(t, err)
	require.Len(t, responses, 4)

	endpoint := headerValue(responses[0].GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), metadata.DestinationEndpointKey)
	require.NotEmpty(t, endpoint, "request headers response must set the destination endpoint")
	return endpoint
}

func capRTXHost() string { return capRTXIP + ":8000" }
func capDGXHost() string { return capDGXIP + ":8000" }

// TestCapacityAdmission_NextTurnRelocationAndSticky proves the core route
// contract end to end: a session starts on RTX while capacity permits, its
// grown next turn relocates to DGX when the RTX per-request ceiling rejects
// it, and the relocated session does not bounce back to RTX when later turns
// shrink again.
func TestCapacityAdmission_NextTurnRelocationAndSticky(t *testing.T) {
	srv := newCapCounter(t, nil)
	configText := capConfig(srv.URL)

	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithStandardMode(), WithConfigText(configText)).WithBaseResources()
	capWithPods(t, h)
	h.WaitForSync(2, modelMyModel)
	h.WaitForReadyPodsMetric(2)
	h = h.WaitForMetricsDelivery()

	// Turn 1: small history -> RTX.
	ep := sendCapacityChat(t, h, "cap-t1", "cap-session-1", capCombinedModel, capRoot)
	require.Equal(t, capRTXHost(), ep, "turn 1 should accelerate onto RTX")

	// Turn 2: same session root, history grew past the RTX per-request
	// ceiling -> must relocate to DGX.
	turn2 := capRoot + " then " + capBigMarker + " context"
	ep = sendCapacityChat(t, h, "cap-t2", "cap-session-1", capCombinedModel, turn2)
	require.Equal(t, capDGXHost(), ep, "over-budget next turn must relocate to DGX")

	// Turn 3: same session, request small again and RTX budget free: the
	// relocation must be sticky - no bounce back to RTX.
	ep = sendCapacityChat(t, h, "cap-t3", "cap-session-1", capCombinedModel, capRoot)
	require.Equal(t, capDGXHost(), ep, "relocated session must stay on DGX")

	// A fresh session is unaffected and can still accelerate.
	ep = sendCapacityChat(t, h, "cap-t4", "cap-session-2", capCombinedModel, capRoot)
	require.Equal(t, capRTXHost(), ep, "unrelated session keeps RTX eligibility")
}

// TestCapacityAdmission_DGXUnaffectedWhenCounterDown proves the DGX route
// keeps operating when the RTX token counter is unavailable: every request
// is served, and none touches RTX.
func TestCapacityAdmission_DGXUnaffectedWhenCounterDown(t *testing.T) {
	fail := new(atomic.Bool)
	fail.Store(true)
	srv := newCapCounter(t, fail)
	configText := capConfig(srv.URL)

	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithStandardMode(), WithConfigText(configText)).WithBaseResources()
	capWithPods(t, h)
	h.WaitForSync(2, modelMyModel)
	h.WaitForReadyPodsMetric(2)
	h = h.WaitForMetricsDelivery()

	for i := range 2 {
		ep := sendCapacityChat(t, h, fmt.Sprintf("cap-down-%d", i), fmt.Sprintf("down-session-%d", i), capCombinedModel, capRoot)
		require.Equal(t, capDGXHost(), ep, "counter outage must never block DGX")
	}
}

// TestCapacityAdmission_AliasResponsesRejectedAndChatAccelerated proves the
// hardware alias keeps its own policy: RTX-only, uncountable Responses
// rejected with a clear error, chat accelerated onto RTX under the same
// accounting.
func TestCapacityAdmission_AliasResponsesRejectedAndChatAccelerated(t *testing.T) {
	srv := newCapCounter(t, nil)
	configText := capConfig(srv.URL)
	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithStandardMode(), WithConfigText(configText)).WithBaseResources()
	capWithPods(t, h)
	h.WaitForSync(2, modelMyModel)
	h.WaitForReadyPodsMetric(2)
	h = h.WaitForMetricsDelivery()

	// Alias chat goes to RTX.
	ep := sendCapacityChat(t, h, "cap-alias-1", "alias-session-1", capAliasModel, capRoot)
	require.Equal(t, capRTXHost(), ep, "alias chat must pin to RTX")

	// Alias Responses: the engine cannot count it and the alias cannot move
	// to DGX - a clear 4xx rejection, never a silent DGX placement.
	body := `{"model":"` + capAliasModel + `","input":"hello","max_output_tokens":50}`
	reqHeaders := map[string]string{
		metadata.ObjectiveKey:        modelMyModel,
		metadata.ModelNameRewriteKey: capAliasModel,
		reqcommon.RequestIDHeaderKey: "cap-alias-2",
		":path":                      "/v1/responses",
		":method":                    "POST",
	}
	client, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(t.Context())
	require.NoError(t, err)
	responses, err := integration.StreamedRequest(t, client, integration.ReqRaw(reqHeaders, body), 1)
	require.NoError(t, err)
	require.Len(t, responses, 1)
	immediate := responses[0].GetImmediateResponse()
	require.NotNil(t, immediate, "alias Responses must be rejected immediately")
	require.EqualValues(t, 400, int32(immediate.Status.GetCode()), "unsupported-accounting rejection is a clear 400")
}
