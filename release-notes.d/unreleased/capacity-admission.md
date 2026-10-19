---
pr: 0
url: ""
author: naadir
date: 2026-10-19
---
New Alpha plugin `capacity-admission`: capacity-aware admission for heterogeneous DGX+RTX pools on one shared model route. Counts prompts with the RTX engine's own `/tokenize` (OpenAI chat/completions) and `/v1/messages/count_tokens` (Anthropic), requires an explicit max-output reservation, reserves atomically before forwarding (idempotent per request, released on completion/abort, reconciled against engine-idle metrics), excludes RTX before the affinity scorers under budget pressure, keeps relocated sessions pinned to DGX (no bounce-back), never relocates session reference-only requests, and enforces hardware-alias (RTX-only) models without silent DGX fallback. Requires `--allow-experimental-plugins`. Rename this file to its PR number.
