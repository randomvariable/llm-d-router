/*
Copyright 2026 The Kubernetes Authors.

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

// Package conversation provides the conversation-root helper shared by plugins
// that need a stable per-session key without storing request payloads.
//
// The root of a conversation is its leading system/developer instructions plus
// the first user message. Because a conversation's history is append-only, the
// root is identical on every turn, so it identifies the session across turns
// while never revealing the (potentially growing, potentially private) body of
// the conversation.
package conversation

import (
	"hash/fnv"
	"strings"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// roleSeparator and messageSeparator keep the serialisation of a conversation
	// root injective: without them "a"+"bc" and "ab"+"c" would hash alike and two
	// different conversations would collide.
	roleSeparator    = "\x00"
	messageSeparator = "\x01"

	roleSystem    = "system"
	roleDeveloper = "developer"
	roleUser      = "user"

	// textBlockType is the only Anthropic content block that carries prompt text.
	textBlockType = "text"
)

// Root returns the leading system/developer instructions plus the first user
// message, serialised so it is stable across the turns of one conversation. It
// reports false when no root can be extracted: a body the parsers did not fill
// (legacy completions, an unparsed payload, or the responses API whose Input
// and Instructions are untyped) has no root to key on.
func Root(request *fwksched.InferenceRequest) (string, bool) {
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
			// Instructions alone are not a conversation. Keying them would be
			// wrong twice over: the same session lands on a different key as
			// soon as its first user message arrives, because the root grows.
			return "", false
		}
	case request.Body.Messages != nil:
		var sawUser bool
		if system := AnthropicPlainText(request.Body.Messages.System); system != "" {
			parts = append(parts, roleSystem+roleSeparator+system)
		}
		for _, message := range request.Body.Messages.Messages {
			if strings.EqualFold(strings.TrimSpace(message.Role), roleUser) {
				parts = append(parts, roleUser+roleSeparator+AnthropicPlainText(message.Content))
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

// AnthropicPlainText flattens an Anthropic content value to text. The
// requesthandling package exposes only a length for this type, so this package
// owns the text form; non-text blocks are skipped for the same reason the
// OpenAI flattener skips them - their bytes are not part of the prompt text.
func AnthropicPlainText(content fwkrh.AnthropicContent) string {
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

// Hash returns the FNV-1a 64-bit hash of the given text, used to key
// conversation roots into bounded stores without retaining their content.
func Hash(text string) uint64 {
	hasher := fnv.New64a()
	_, _ = hasher.Write([]byte(text)) // hash.Hash.Write never returns an error
	return hasher.Sum64()
}
