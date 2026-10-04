# Conversation Hash Profile Handler (`conversation-hash-profile-handler`)

Runs exactly one scheduling profile per request, choosing it by hashing the
conversation root. Use it to compare placement policies on live traffic without
restarting the EPP: every arm shares the same KV-cache index, latency predictor
and flow control, so a difference in outcomes is a difference in placement.

## Behavior

| Situation | Result |
|---|---|
| Request has a conversation root | `hash % sum(weights)` selects a profile through cumulative weights, in configured order |
| Same conversation, later turn | Same profile: the root is the leading system/developer messages plus the first user message, and history is append-only |
| No extractable root (legacy completions, unparsed payload, responses API, chat with no user message) | `defaultProfile` |
| Hashed profile is not among the configured scheduling profiles | `defaultProfile`, logged once per unknown name |
| Neither the hashed profile nor `defaultProfile` is configured | No profile runs; the request fails the way an empty selection always does |
| Exactly one profile configured | That profile runs, so the handler can replace `single-profile-handler` when an experiment ends |
| A profile has already run for this request | Returns an empty selection |

The conversation root, not the request id, is what makes the comparison valid. A
session split across arms would contaminate both, because its later turns
inherit the prefix cache its earlier turns built on one endpoint.

## Root extraction

| API | Root |
|---|---|
| `/v1/chat/completions` | leading `system`/`developer` messages, then the first `user` message, using `Content.PlainText()` |
| `/v1/messages` (Anthropic) | top-level `system`, then the first `user` message; non-text blocks are skipped, as the OpenAI flattener skips them |
| anything else | none - falls back to `defaultProfile` |

Each contributing message serialises as `role + "\x00" + text`, joined by
`"\x01"`, and is hashed with FNV-1a 64.

## Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `profiles` | `[{name, weight}]` | yes | Arms and their relative weights. Names must be unique and non-blank; weights non-negative with a positive sum. Fractional weights are rejected. |
| `defaultProfile` | `string` | yes | Must name one of `profiles`. Runs when no root is extractable and when a hashed arm is not configured. |

## Metrics

`llm_d_epp_conversation_hash_profile_selections_total{profile}` counts one
selection per request, under the profile that actually ran (so a fallback is
counted under the default). Pair it with the per-arm outcomes the audit script
derives from the `-v=4` logs.

At `-v=4` the handler logs `Conversation hash profile selected` with `profile`
and `sessionHash` (16 hex digits). The audit tool groups requests into sessions
by `sessionHash` and resamples whole sessions when it builds confidence
intervals, which is what keeps a chatty session from posing as many
observations.

## Configuration example

```yaml
plugins:
  - type: conversation-hash-profile-handler
    parameters:
      profiles:
        - name: "A"
          weight: 50
        - name: "B"
          weight: 25
        - name: "C"
          weight: 25
      defaultProfile: "A"
schedulingProfiles:
  - name: "A"
    plugins:
      - pluginRef: latency-scorer
      - pluginRef: weighted-random-picker
  - name: "B"
    plugins:
      - pluginRef: latency-scorer
      - pluginRef: max-score-picker
  - name: "C"
    plugins:
      - pluginRef: latency-scorer
      - pluginRef: weighted-random-picker
```

## Related Documentation

- [Header Profile Handler](../headerprofile/README.md) - selects by request
  header, which suits request-level experiments and mis-splits sessions when the
  policy being compared is cache affinity
- [Single Profile Handler](../single/README.md) - no selection at all
