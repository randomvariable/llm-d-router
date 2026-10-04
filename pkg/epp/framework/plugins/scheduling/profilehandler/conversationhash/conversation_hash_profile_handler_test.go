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

package conversationhash

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type fakeSchedulerProfile struct{ label string }

func (f *fakeSchedulerProfile) Run(_ context.Context, _ *fwksched.InferenceRequest, _ []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
	return &fwksched.ProfileRunResult{}, nil
}

func chatRequest(systemPrompt, firstUserText string, extraTurns int) *fwksched.InferenceRequest {
	messages := []fwkrh.Message{
		{Role: "system", Content: fwkrh.Content{Raw: systemPrompt}},
		{Role: "user", Content: fwkrh.Content{Raw: firstUserText}},
	}
	for i := range extraTurns {
		messages = append(messages,
			fwkrh.Message{Role: "assistant", Content: fwkrh.Content{Raw: fmt.Sprintf("answer %d", i)}},
			fwkrh.Message{Role: "user", Content: fwkrh.Content{Raw: fmt.Sprintf("follow-up %d", i)}},
		)
	}
	return &fwksched.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{Messages: messages},
		},
	}
}

func anthropicRequest(system, firstUser string, turns int) *fwksched.InferenceRequest {
	messages := []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: firstUser}}}
	for i := range turns {
		messages = append(messages,
			fwkrh.AnthropicMessage{Role: "assistant", Content: fwkrh.AnthropicContent{Raw: fmt.Sprintf("a %d", i)}},
			fwkrh.AnthropicMessage{Role: "user", Content: fwkrh.AnthropicContent{Raw: fmt.Sprintf("q %d", i)}},
		)
	}
	return &fwksched.InferenceRequest{Body: &fwkrh.InferenceRequestBody{
		Messages: &fwkrh.MessagesRequest{
			System:   fwkrh.AnthropicContent{Raw: system},
			Messages: messages,
		},
	}}
}

func threeArms() []profileWeight {
	return []profileWeight{{Name: "A", Weight: 50}, {Name: "B", Weight: 25}, {Name: "C", Weight: 25}}
}

func mustHandler(t *testing.T, profiles []profileWeight, defaultProfile string) *ConversationHashProfileHandler {
	t.Helper()
	handler, err := NewConversationHashProfileHandler(profiles, defaultProfile)
	require.NoError(t, err)
	return handler
}

func profileMap() map[string]fwksched.SchedulerProfile {
	return map[string]fwksched.SchedulerProfile{
		"A": &fakeSchedulerProfile{label: "A"},
		"B": &fakeSchedulerProfile{label: "B"},
		"C": &fakeSchedulerProfile{label: "C"},
	}
}

// pickName returns the single profile name Pick selected.
func pickName(t *testing.T, handler *ConversationHashProfileHandler, request *fwksched.InferenceRequest) string {
	t.Helper()
	selected := handler.Pick(context.Background(), request, profileMap(), map[string]*fwksched.ProfileRunResult{})
	require.Len(t, selected, 1, "exactly one profile must run per request")
	for name := range selected {
		return name
	}
	panic("unreachable")
}

func TestConversationRootIsStableAcrossTurns(t *testing.T) {
	handler := mustHandler(t, threeArms(), "A")

	first := pickName(t, handler, chatRequest("You are a coding agent.", "hello", 0))
	for _, turns := range []int{1, 2, 5, 12} {
		assert.Equal(t, first, pickName(t, handler, chatRequest("You are a coding agent.", "hello", turns)),
			"appending turns must not move a session to another arm")
	}
}

func TestConversationRootSeparatesConversations(t *testing.T) {
	handler := mustHandler(t, threeArms(), "A")

	// A constant hash (for example reading the wrong field) would pass the
	// stability test above and fail here.
	seen := map[string]int{}
	for i := range 400 {
		seen[pickName(t, handler, chatRequest("system", fmt.Sprintf("prompt %d", i), 0))]++
	}
	assert.Len(t, seen, 3, "all arms must be reachable")
}

func TestProfileSplitMatchesWeights(t *testing.T) {
	handler := mustHandler(t, threeArms(), "A")

	const samples = 30000
	counts := map[string]int{}
	for i := range samples {
		counts[pickName(t, handler, chatRequest(
			fmt.Sprintf("instructions %d", i), fmt.Sprintf("first message %d", i), 0))]++
	}

	for _, want := range []struct {
		profile  string
		expected float64
	}{
		{profile: "A", expected: 50},
		{profile: "B", expected: 25},
		{profile: "C", expected: 25},
	} {
		share := 100 * float64(counts[want.profile]) / float64(samples)
		assert.InDelta(t, want.expected, share, 1.5,
			"profile %s took %.2f%% of %d conversations", want.profile, share, samples)
	}
}

func TestNoExtractableRootUsesDefaultProfile(t *testing.T) {
	handler := mustHandler(t, threeArms(), "A")

	cases := map[string]*fwksched.InferenceRequest{
		"nil request": nil,
		"nil body":    {Body: nil},
		"empty body":  {Body: &fwkrh.InferenceRequestBody{}},
		"legacy completions": {Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "once upon a time"}},
		}},
		"responses api": {Body: &fwkrh.InferenceRequestBody{
			Responses: &fwkrh.ResponsesRequest{Input: "anything", Instructions: "anything"},
		}},
		"chat without user message": {Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{Role: "system", Content: fwkrh.Content{Raw: "only instructions"}}},
			},
		}},
	}

	for name, request := range cases {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, "A", pickName(t, handler, request))
		})
	}
}

func TestAnthropicRootUsesSystemAndFirstUser(t *testing.T) {
	handler := mustHandler(t, threeArms(), "A")

	first := pickName(t, handler, anthropicRequest("be terse", "hi", 0))
	assert.Equal(t, first, pickName(t, handler, anthropicRequest("be terse", "hi", 4)),
		"an anthropic session must keep its arm across turns")
	assert.NotEqual(t, first, pickName(t, handler, anthropicRequest("be verbose", "hi", 0)),
		"a different system prompt is a different conversation root")

	// Structured text blocks flatten to their text and non-text blocks are
	// skipped, so this request shares the arm of the raw-string one.
	blockRequest := &fwksched.InferenceRequest{Body: &fwkrh.InferenceRequestBody{
		Messages: &fwkrh.MessagesRequest{
			System: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{{Type: "text", Text: "be terse"}}},
			Messages: []fwkrh.AnthropicMessage{
				{Role: "user", Content: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
					{Type: "text", Text: "hi"}, {Type: "image"},
				}}},
			},
		},
	}}
	assert.Equal(t, first, pickName(t, handler, blockRequest))
}

func TestSingleConfiguredProfileAlwaysRuns(t *testing.T) {
	handler := mustHandler(t, []profileWeight{{Name: "only", Weight: 10}}, "only")
	profiles := map[string]fwksched.SchedulerProfile{"only": &fakeSchedulerProfile{label: "only"}}

	selected := handler.Pick(context.Background(), chatRequest("s", "u", 0), profiles, map[string]*fwksched.ProfileRunResult{})
	require.Len(t, selected, 1)
	_, ok := selected["only"]
	assert.True(t, ok, "a one-profile deployment runs without a hash decision")
}

func TestAlreadyRunProfilesAreNotSelectedAgain(t *testing.T) {
	handler := mustHandler(t, threeArms(), "A")
	selected := handler.Pick(context.Background(), chatRequest("s", "u", 0), profileMap(),
		map[string]*fwksched.ProfileRunResult{"A": {}})
	assert.Empty(t, selected)
}

func TestUnknownHashedProfileFallsBackToDefault(t *testing.T) {
	handler := mustHandler(t, threeArms(), "A")

	// Only A and B are configured; a conversation hashing to C must still run A.
	profiles := map[string]fwksched.SchedulerProfile{
		"A": &fakeSchedulerProfile{label: "A"},
		"B": &fakeSchedulerProfile{label: "B"},
	}
	resetMetrics()
	for i := range 500 {
		request := chatRequest(fmt.Sprintf("sys %d", i), "u", 0)
		root, ok := conversationRoot(request)
		require.True(t, ok)
		if handler.profileFor(fnv1a64(root)) != "C" {
			continue
		}
		result := handler.Pick(context.Background(), request, profiles, map[string]*fwksched.ProfileRunResult{})
		require.Len(t, result, 1)
		_, ranDefault := result["A"]
		assert.True(t, ranDefault, "an unconfigured arm must fall back to defaultProfile")
		assert.Equal(t, 1.0, testutil.ToFloat64(profileSelections.WithLabelValues("A")),
			"the fallback is counted under the profile that actually ran")
		return
	}
	t.Fatal("test prerequisite: no sample hashed to the unconfigured arm")
}

func TestProcessResults(t *testing.T) {
	handler := mustHandler(t, threeArms(), "A")
	ctx := context.Background()

	t.Run("single result", func(t *testing.T) {
		result, err := handler.ProcessResults(ctx, nil, map[string]*fwksched.ProfileRunResult{
			"B": {TargetEndpoints: nil},
		})
		require.NoError(t, err)
		assert.Equal(t, "B", result.PrimaryProfileName)
	})

	t.Run("no result", func(t *testing.T) {
		_, err := handler.ProcessResults(ctx, nil, map[string]*fwksched.ProfileRunResult{})
		require.Error(t, err)
	})

	t.Run("several results", func(t *testing.T) {
		_, err := handler.ProcessResults(ctx, nil, map[string]*fwksched.ProfileRunResult{
			"A": {}, "B": {},
		})
		require.Error(t, err)
	})

	t.Run("failed run", func(t *testing.T) {
		_, err := handler.ProcessResults(ctx, nil, map[string]*fwksched.ProfileRunResult{"A": nil})
		require.Error(t, err)
	})
}

func TestFactoryValidatesParameters(t *testing.T) {
	cases := []struct {
		name  string
		raw   string
		valid bool
	}{
		{
			name:  "valid",
			raw:   `{"profiles":[{"name":"A","weight":50},{"name":"B","weight":50}],"defaultProfile":"A"}`,
			valid: true,
		},
		{name: "no profiles", raw: `{"defaultProfile":"A"}`},
		{name: "empty profiles", raw: `{"profiles":[],"defaultProfile":"A"}`},
		{name: "zero total weight", raw: `{"profiles":[{"name":"A","weight":0}],"defaultProfile":"A"}`},
		{name: "negative weight", raw: `{"profiles":[{"name":"A","weight":-5}],"defaultProfile":"A"}`},
		{name: "duplicate names", raw: `{"profiles":[{"name":"A","weight":1},{"name":"A","weight":2}],"defaultProfile":"A"}`},
		{name: "blank name", raw: `{"profiles":[{"name":"  ","weight":1}],"defaultProfile":"A"}`},
		{name: "missing default", raw: `{"profiles":[{"name":"A","weight":1}]}`},
		{name: "default not configured", raw: `{"profiles":[{"name":"A","weight":1}],"defaultProfile":"Z"}`},
		{name: "fractional weight", raw: `{"profiles":[{"name":"A","weight":25.5}],"defaultProfile":"A"}`},
		// The framework hands factories a strict decoder, so a typo in a parameter
		// name fails at load rather than being ignored.
		{name: "unknown field", raw: `{"profiles":[{"name":"A","weight":1}],"defaultProfile":"A","bogus":1}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plugin, err := ConversationHashProfileHandlerFactory("h", fwkplugin.StrictDecoder(json.RawMessage(tc.raw)), nil)
			if tc.valid {
				require.NoError(t, err)
				require.NotNil(t, plugin)
				return
			}
			require.Error(t, err)
		})
	}
}

func TestSelectionCounter(t *testing.T) {
	handler := mustHandler(t, threeArms(), "A")
	resetMetrics()

	for i := range 100 {
		pickName(t, handler, chatRequest(fmt.Sprintf("s%d", i), "u", 0))
	}

	// The vector is package-level, so sum the arms rather than assert on one
	// series: ToFloat64 requires exactly one metric and would panic on three.
	var total float64
	for _, profile := range handler.profiles {
		total += testutil.ToFloat64(profileSelections.WithLabelValues(profile.Name))
	}
	assert.Equal(t, 100.0, total, "every request contributes exactly one arm observation")
	assert.Greater(t, testutil.ToFloat64(profileSelections.WithLabelValues("A")), 0.0,
		"the default arm must see traffic")
}
