# Capacity Admission

**Type:** `capacity-admission` (Alpha - requires `--allow-experimental-plugins`)

Capacity-aware admission for heterogeneous pools that pair a large-context DGX
fleet with a small RTX 6000 host on one shared model route. The RTX endpoint is
an accelerator, never a fallback: a request may occupy RTX capacity only when
its **full prompt token count - converted by the RTX engine itself** - plus an
**explicit maximum output reservation** fit inside the RTX budget. Anything the
plugin cannot prove (unknown tokens, missing `max_tokens`, budget pressure, an
engine state this EPP process has not observed) keeps the request on DGX, whose
cache-affinity and active-request scorers run untouched.

## Extension points

One plugin instance fills several roles; reference it once in `plugins:` and
once in the profile (it registers as a **Filter**; the
DataProducer/PreRequest/ResponseBody hooks wire automatically from the plugin
handle). Keep `max-score-picker` and the DGX scorers unchanged.

1. **DataProducer** - classifies the request and asks the RTX engine for the
   token count exactly once per request:
   - `/v1/chat/completions`, `/v1/completions` → vLLM `POST {counter.baseURL}/tokenize`
   - `/v1/messages` (Anthropic) → `POST {counter.baseURL}/v1/messages/count_tokens`
     (system, messages and tools included; validated against the engine)
   - pre-tokenized prompts (completions token-ids, native generate token_ids)
     → counted locally, no call
   - everything the engine cannot count for us → **uncountable → RTX
     ineligible** (vLLM exposes no Responses count endpoint, so `/v1/responses`
     is uncountable by definition)
2. **Filter** - runs before the affinity scorers. Applies the capacity policy
   and, when the RTX endpoint passes, atomically reserves the request
   (idempotent per request identity, so warm/test profile re-runs neither
   double count nor race). A request that loses the race simply proceeds with
   the DGX-only candidate set, scored normally: no avoidable race 503.
3. **PreRequest** - confirms the hold against the actual destination
   (re-reserving if a competing profile misplaced it), releases holds for
   DGX-routed requests, marks relocated sessions, and rejects RTX-only
   (hardware alias) traffic the engine cannot account - with a clear typed
   error, never a silent DGX placement.
4. **ResponseBody** - releases at end-of-stream, including the
   director-guaranteed abort path (client disconnect, cancellation, failed
   forwarding). A `context.AfterFunc` guard additionally releases holds if the
   request dies before a destination is captured. All releases are idempotent.

## Session behavior

- **Relocation is sticky and endpoint-exact.** When a session that previously
  ran on RTX cannot be admitted there this turn (budget, size, countability,
  engine evidence), it moves to DGX and is thereafter pinned to that exact DGX
  endpoint while it remains a candidate - later turns do not bounce between
  DGX replicas or back to RTX when capacity frees. If the pinned DGX replica
  disappears, normal DGX selection resumes for that session.
- **Reference-only reads are rejected, not relocated or force-admitted.** A
  `/v1/responses` request that only references stored state
  (`previous_response_id`, empty input) of a session homed on RTX fails with
  `PreconditionFailed (session_reference_only_not_accountable)`: the engine
  cannot count that context, so admitting it would consume an unaccountable
  budget, and DGX cannot resolve the reference.
- Session identity is the opaque configured `sessionHeader` value when
  present, else a FNV hash of the conversation root (system/developer
  instructions + first user message - the same extraction the
  conversation-hash profile handler uses, shared via
  `scheduling/util/conversation`). Raw prompts are never stored. The store is
  LRU-bounded (`maxSessionEntries`) with TTL expiry (`sessionTTL`).

## Fail-closed rules

- Until the RTX endpoint reports `running == 0 && waiting == 0` in a metrics
  snapshot that (a) postdates this process start, and (b) is fresher than
  `metricsMaxAge`, RTX stays excluded - for shared-route **and** alias
  traffic: an EPP restart lost the in-process reservation table, so only a
  proven-drained engine is safe to admit against. Reconciliation then clears a
  hold only when an idle snapshot **postdates the reservation by more than**
  `staleReservationGrace`: fresh holds legitimately coexist with idle metrics
  (the request has not reached the engine yet).
- A request whose tokens cannot be counted never occupies the RTX budget:
  uncountable APIs or payloads (missing explicit `max_tokens`, `chat_template`
  overrides, `documents`, `thinking`, `n`/`best_of` > 1, batched prompts, gRPC
  payloads) → DGX. Hardware-alias (`rtxOnlyModels`) requests that cannot be
  counted are rejected with `400` - the alias pin has no DGX escape.
- Counter failure/timeout ⇒ RTX ineligible; DGX keeps serving.
- Alias traffic is additionally gated by the engine's own
  `running+waiting < maxConcurrentRequests` report (the table restarts empty;
  the engine's load does not).

## Parameters

- `rtxEndpointName` (string, required unless label given): RTX
  `EndpointMetadata.Name`; Deployment random suffixes tolerated via a
  `name-` prefix-boundary match.
- `rtxEndpointLabel` (string, optional): `key` or `key=value` pod-label
  alternative/extra matcher.
- `combinedModels` ([]string): shared-route target model names
  (e.g. `qwen38-flash-next`).
- `rtxOnlyModels` ([]string): hardware alias names pinned to RTX (e.g.
  `qwen38-flash-next-rtx6000`).
- `maxConcurrentRequests` (int, required): live RTX request ceiling.
- `maxResidentTokens` (int64, required): summed (input + reserved output)
  token ceiling across live reservations.
- `maxRequestTokens` (int64, required): per-request (input + reserved output)
  ceiling (must be ≤ `maxResidentTokens`).
- `counter.baseURL` (string, required): RTX engine Service URL - never the
  EPP route, or counting would recurse.
- `counter.timeout` (duration, default `2s`): per count call; size the
  longest expected legitimate `/tokenize` (e.g. 10s for 400k-token prompts).
  Failure only drops RTX eligibility.
- `counter.model` (string, optional): override for the count-call model name;
  default is the request's target model (RTX must accept every governed name,
  common and alias alike).
- `counter.openAITokenizePath` (default `/tokenize`),
  `counter.anthropicCountPath` (default `/v1/messages/count_tokens`),
  `counter.caCertPath` / `counter.clientCertPath` / `counter.clientKeyPath` /
  `counter.insecureSkipVerify` (TLS, mirroring the tokenizer producer).
- `sessionHeader` (string, optional): header whose opaque value keys the
  session store.
- `maxSessionEntries` (int, default 65536), `sessionTTL` (duration, default
  `2h`).
- `metricsMaxAge` (duration, default `30s`), `staleReservationGrace`
  (duration, default `30s`).

## Configuration example

```yaml
plugins:
  - type: capacity-admission
    name: qwen-capacity
    parameters:
      rtxEndpointName: rtx6000-qwen38-flash-next
      combinedModels: ["qwen38-flash-next"]
      rtxOnlyModels: ["qwen38-flash-next-rtx6000"]
      maxConcurrentRequests: 2
      maxResidentTokens: 900000
      maxRequestTokens: 400000
      counter:
        baseURL: http://rtx6000-qwen38-flash-next.openai.svc.cluster.local:8000
        timeout: 10s
      sessionHeader: x-session-id
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: prefix-scorer        # unchanged DGX affinity
      - pluginRef: active-tracker       # unchanged DGX load signal
      - pluginRef: qwen-capacity        # filter role; place before any
                                        # session-affinity filter
      - pluginRef: max-score-picker
```

## Limitations

- Budgets are per EPP replica; multiple replicas multiply the effective RTX
  budget.
- The reservation flow assumes one primary profile decides the destination;
  disagg prefill/decode pairing is not accounted per role.
- Multi-sequence requests (`n`/`best_of` > 1, batched prompts) and Anthropic
  `thinking` are refused RTX eligibility: single-max-output accounting would
  under-reserve them.
- gRPC-payload requests, per-request `chat_template` overrides, and HF
  `documents` are not countable and stay on DGX.
- Multimodal content is counted only insofar as the engine's own tokenizer
  reports it; a count call that fails keeps the request on DGX.
