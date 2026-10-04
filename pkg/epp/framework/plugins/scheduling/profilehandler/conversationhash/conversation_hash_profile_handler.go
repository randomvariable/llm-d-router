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

// Package conversationhash provides a profile handler that assigns every
// conversation to one scheduling profile by hashing its root, so competing
// placement policies can be compared inside a single EPP process on live
// traffic.
package conversationhash

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"maps"
	"slices"
	"strings"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// ConversationHashProfileHandlerType is the type of the ConversationHashProfileHandler.
	ConversationHashProfileHandlerType = "conversation-hash-profile-handler"

	// roleSeparator and messageSeparator keep the serialisation of a conversation
	// root injective: without them "a"+"bc" and "ab"+"c" would hash alike and two
	// different conversations would share an arm.
	roleSeparator    = "\x00"
	messageSeparator = "\x01"

	roleSystem    = "system"
	roleDeveloper = "developer"
	roleUser      = "user"

	// textBlockType is the only Anthropic content block that carries prompt text.
	textBlockType = "text"
)

// parameters configures the ConversationHashProfileHandler.
type parameters struct {
	// Profiles are the candidate scheduling profiles and their selection
	// weights. Weights are relative and their sum must be positive.
	Profiles []profileWeight `json:"profiles"`
	// DefaultProfile runs when a request has no extractable conversation root (a
	// legacy completions body, an unparsed payload), and is the fallback when a
	// hashed profile is not configured. It must name one of Profiles.
	DefaultProfile string `json:"defaultProfile"`
}

// profileWeight is one arm: a scheduling profile name and its relative weight.
// Weight is int64 so the framework's strict decoder rejects a fractional value
// at parse time rather than silently truncating it.
type profileWeight struct {
	Name   string `json:"name"`
	Weight int64  `json:"weight"`
}

// compile-time type assertion
var _ fwksched.ProfileHandler = &ConversationHashProfileHandler{}

// ConversationHashProfileHandler runs exactly one scheduling profile per
// request, chosen by hashing the conversation root. A conversation's history is
// append-only, so its root is identical on every turn and the whole session
// stays on one arm, while different conversations spread across arms by weight.
//
// Hashing the root rather than the request id is what makes the comparison
// valid: a session split across arms would contaminate both, because its later
// turns inherit the cache its earlier turns built.
type ConversationHashProfileHandler struct {
	typedName      fwkplugin.TypedName
	profiles       []profileWeight
	totalWeight    int64
	defaultProfile string

	// warnedOnce keeps the misconfiguration log to one line per profile name
	// instead of one per request.
	warnedOnce sync.Map
}

// ConversationHashProfileHandlerFactory defines the factory function for ConversationHashProfileHandler.
func ConversationHashProfileHandlerFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' profile handler - %w", ConversationHashProfileHandlerType, err)
		}
	}

	handler, err := NewConversationHashProfileHandler(params.Profiles, params.DefaultProfile)
	if err != nil {
		return nil, fmt.Errorf("invalid configuration for the '%s' profile handler: %w", ConversationHashProfileHandlerType, err)
	}

	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
	}

	return handler.WithName(name), nil
}

// NewConversationHashProfileHandler validates the arms and returns the handler.
// The configured order is significant: weights are cumulative, so the same
// weights in a different order split traffic differently.
func NewConversationHashProfileHandler(profiles []profileWeight, defaultProfile string) (*ConversationHashProfileHandler, error) {
	if len(profiles) == 0 {
		return nil, fmt.Errorf("profiles must not be empty")
	}

	normalized := make([]profileWeight, 0, len(profiles))
	seen := make(map[string]struct{}, len(profiles))
	var total int64
	for _, profile := range profiles {
		profileName := strings.TrimSpace(profile.Name)
		if profileName == "" {
			return nil, fmt.Errorf("a profile name must not be empty")
		}
		if _, duplicate := seen[profileName]; duplicate {
			return nil, fmt.Errorf("duplicate profile name %q", profileName)
		}
		seen[profileName] = struct{}{}
		if profile.Weight < 0 {
			return nil, fmt.Errorf("profile %q: weight must be non-negative, got %d", profileName, profile.Weight)
		}
		total += profile.Weight
		normalized = append(normalized, profileWeight{Name: profileName, Weight: profile.Weight})
	}
	if total <= 0 {
		return nil, fmt.Errorf("the sum of profile weights must be positive, got %d", total)
	}

	defaultProfile = strings.TrimSpace(defaultProfile)
	if defaultProfile == "" {
		return nil, fmt.Errorf("defaultProfile is required and must name one of the configured profiles")
	}
	if _, configured := seen[defaultProfile]; !configured {
		return nil, fmt.Errorf("defaultProfile %q is not among the configured profiles", defaultProfile)
	}

	return &ConversationHashProfileHandler{
		typedName:      fwkplugin.TypedName{Type: ConversationHashProfileHandlerType, Name: ConversationHashProfileHandlerType},
		profiles:       normalized,
		totalWeight:    total,
		defaultProfile: defaultProfile,
	}, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (h *ConversationHashProfileHandler) TypedName() fwkplugin.TypedName {
	return h.typedName
}

// WithName sets the name of the profile handler.
func (h *ConversationHashProfileHandler) WithName(name string) *ConversationHashProfileHandler {
	h.typedName.Name = name
	return h
}

// Pick implements fwksched.ProfileHandler.Pick; see README.md for behavior.
func (h *ConversationHashProfileHandler) Pick(ctx context.Context, request *fwksched.InferenceRequest,
	profiles map[string]fwksched.SchedulerProfile,
	profileResults map[string]*fwksched.ProfileRunResult) map[string]fwksched.SchedulerProfile {
	if len(profileResults) > 0 { // the selected profile has already run
		return map[string]fwksched.SchedulerProfile{}
	}

	// With a single configured profile there is nothing to choose, so run it. This
	// keeps the handler usable as a drop-in for single-profile-handler once an
	// experiment ends.
	if len(profiles) == 1 {
		for name, profile := range profiles {
			recordProfileSelection(name)
			return map[string]fwksched.SchedulerProfile{name: profile}
		}
	}

	selected := h.defaultProfile
	var hash uint64
	if root, ok := conversationRoot(request); ok {
		hash = fnv1a64(root)
		selected = h.profileFor(hash)
	}

	profile, configured := profiles[selected]
	if !configured {
		h.warnUnknownProfile(ctx, selected, slices.Sorted(maps.Keys(profiles)))
		selected = h.defaultProfile
		profile, configured = profiles[selected]
		if !configured {
			// Neither the hashed arm nor the default exists. The client gets the
			// generic failure the director returns for an empty selection; this line
			// gives the operator the reason.
			log.FromContext(ctx).Info("no scheduling profile selected for request",
				"resolvedProfileName", selected,
				"registeredProfiles", slices.Sorted(maps.Keys(profiles)))
			return map[string]fwksched.SchedulerProfile{}
		}
	}

	logger := log.FromContext(ctx)
	if logger.V(logutil.DEBUG).Enabled() {
		logger.V(logutil.DEBUG).Info("Conversation hash profile selected",
			"profile", selected, "sessionHash", fmt.Sprintf("%016x", hash))
	}
	recordProfileSelection(selected)

	return map[string]fwksched.SchedulerProfile{selected: profile}
}

// ProcessResults handles the outcome of the single profile run selected by Pick.
// It specifies in the SchedulingResult the key of the primary profile that should
// be used to get the request's selected destination.
func (h *ConversationHashProfileHandler) ProcessResults(_ context.Context, _ *fwksched.InferenceRequest,
	profileResults map[string]*fwksched.ProfileRunResult) (*fwksched.SchedulingResult, error) {
	switch len(profileResults) {
	case 0:
		return nil, fmt.Errorf("conversation hash profile handler: no scheduling profile was run")
	case 1:
		// exactly one profile ran, handled below
	default:
		return nil, fmt.Errorf("conversation hash profile handler is intended to run a single profile per request, got %d", len(profileResults))
	}

	var profileName string
	for name := range profileResults {
		profileName = name
	}

	if profileResults[profileName] == nil { // there was an error while running the profile
		return nil, fmt.Errorf("failed to run scheduler profile '%s'", profileName)
	}

	return &fwksched.SchedulingResult{
		ProfileResults:     profileResults,
		PrimaryProfileName: profileName,
	}, nil
}

// profileFor maps a conversation hash onto a profile through cumulative
// weights, in configured order.
func (h *ConversationHashProfileHandler) profileFor(hash uint64) string {
	// #nosec G115 -- the modulo keeps this below totalWeight, a positive int64.
	remaining := int64(hash % uint64(h.totalWeight))
	for _, profile := range h.profiles {
		remaining -= profile.Weight
		if remaining < 0 {
			return profile.Name
		}
	}
	return h.profiles[len(h.profiles)-1].Name
}

// warnUnknownProfile logs a selected arm that has no scheduling profile
// configured once per name, so a config typo is visible without flooding the
// log at request rate.
func (h *ConversationHashProfileHandler) warnUnknownProfile(ctx context.Context, profileName string, registered []string) {
	if _, loaded := h.warnedOnce.LoadOrStore(profileName, struct{}{}); loaded {
		return
	}
	log.FromContext(ctx).Info("hashed profile is not configured, falling back to defaultProfile",
		"profileName", profileName, "defaultProfile", h.defaultProfile,
		"registeredProfiles", registered)
}

// conversationRoot returns the leading system/developer instructions plus the
// first user message, serialised so it is stable across the turns of one
// conversation. It reports false when no root can be extracted, which routes
// the request to defaultProfile: a body the parsers did not fill (legacy
// completions, an unparsed payload, or the responses API whose Input and
// Instructions are untyped) is not worth guessing at, since the traffic this
// pool serves is chat completions.
func conversationRoot(request *fwksched.InferenceRequest) (string, bool) {
	if request == nil || request.Body == nil {
		return "", false
	}

	var parts []string
	switch {
	case request.Body.ChatCompletions != nil:
		var sawUser bool
		for _, message := range request.Body.ChatCompletions.Messages {
			role := strings.ToLower(strings.TrimSpace(message.Role))
			if role == roleSystem || role == roleDeveloper {
				parts = append(parts, role+roleSeparator+message.Content.PlainText())
				continue
			}
			if role == roleUser {
				parts = append(parts, role+roleSeparator+message.Content.PlainText())
				sawUser = true
			}
			break // the root ends at the first non-instruction message
		}
		if !sawUser {
			// Instructions alone are not a conversation. Assigning them an arm by
			// hash would be wrong twice over: the same session lands on a different
			// arm as soon as its first user message arrives, because the root grows.
			return "", false
		}
	case request.Body.Messages != nil:
		var sawUser bool
		if system := anthropicPlainText(request.Body.Messages.System); system != "" {
			parts = append(parts, roleSystem+roleSeparator+system)
		}
		for _, message := range request.Body.Messages.Messages {
			if strings.EqualFold(strings.TrimSpace(message.Role), roleUser) {
				parts = append(parts, roleUser+roleSeparator+anthropicPlainText(message.Content))
				sawUser = true
			}
			break
		}
		if !sawUser {
			return "", false
		}
	default:
		return "", false
	}
	if len(parts) == 0 {
		return "", false
	}
	return strings.Join(parts, messageSeparator), true
}

// anthropicPlainText flattens an Anthropic content value to text. The
// requesthandling package exposes only a length for this type, so the plugin
// owns the text form; non-text blocks are skipped for the same reason the
// OpenAI flattener skips them - their bytes are not part of the prompt text.
func anthropicPlainText(content fwkrh.AnthropicContent) string {
	if content.Raw != "" {
		return content.Raw
	}
	var builder strings.Builder
	for _, block := range content.Structured {
		if block.Type == textBlockType {
			builder.WriteString(block.Text)
			builder.WriteString(" ")
		}
	}
	return builder.String()
}

func fnv1a64(text string) uint64 {
	hasher := fnv.New64a()
	_, _ = hasher.Write([]byte(text)) // hash.Hash.Write never returns an error
	return hasher.Sum64()
}
