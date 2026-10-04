# KV-Block Index

The block-level store that maps KV-block keys to the pods holding them, plus the
token-to-block-key processing that feeds it. This is the shared state the
[`kvcache.Indexer`](../README.md) reads and the [`kvevents`](../../kvevents/README.md)
subscriber writes.

## What It Does

- **Token processing.** `TokenProcessor` (backed by `ChunkedTokenDatabase`)
  turns a token sequence into KV-block keys: it chunks tokens into fixed-size
  blocks and hashes each block, chaining the previous block's hash so a key
  encodes its full prefix. `ComputeBlockExtraFeatures` folds multimodal
  metadata into per-block `BlockExtraFeatures` that taint the hash.
- **Block index.** `Index` maps each block key to the set of pods (`PodEntry`,
  carrying a pod identifier and device tier) that currently hold it. Producers
  `Add` and `Evict` entries; the indexer's `Lookup` reads them.
- **Ordered walk.** `KeyWalker` is an optional capability for callers that
  fold entries in key order: it visits the requested keys in sequence,
  reporting misses, and lends each key's entries as `EntryRef` values that
  carry the index's pod and tier ordinals. The in-memory backend has it and
  the decorators keep it; the [`kvcache`](../README.md) prefix matcher walks
  when it can and falls back to `Lookup`.

## Backends

`NewIndex` selects a backend from `IndexConfig` (first configured wins):

| Backend | Constructor | Use |
|---------|-------------|-----|
| In-memory | `NewInMemoryIndex` | Single-replica, LRU-bounded local index. |
| Cost-aware memory | `NewCostAwareMemoryIndex` | Local index with cost-aware eviction. |
| Redis | `NewRedisIndex` | Shared index across EPP replicas (Redis-protocol server). |

Two decorators wrap any backend: `NewTracedIndex` adds OpenTelemetry spans
(no-op when tracing is off) and `NewInstrumentedIndex` records the
[metrics](../metrics/README.md).

## Grouped (HMA) Entries

`GroupCatalog` tracks group identity for heterogeneous-memory-aware (HMA)
models. A `PodEntry` is one holder: `(pod, device tier)`, optionally carrying a
group. For `full_attention` blocks the group is deliberately not part of the
holder identity, because the engine serves such a block only when every group
of the spec holds it (`BlockPool.get_cached_block` in
`vllm/v1/core/block_pool.py:198-223`), so the groups are one cache rather than
competing holders. Announcing one block per group (13 of 50 groups on a
hybrid 48-layer model plus MTP) would otherwise multiply entries per key and
let a single store evict every other pod under the per-key cap. Groups of
other kinds keep `HasGroup`/`GroupIdx`, and eviction stays reference-counted
per group so a block is dropped only once no announcement still holds it.

## Key Types

| Symbol | Role |
|--------|------|
| `Index` | Block-key -> pods mapping (`Add` / `Lookup` / `Evict`). |
| `KeyWalker` / `EntryRef` | Optional ordered walk over requested keys; entries with pod and tier ordinals. |
| `TokenProcessor` | Tokens -> block keys; exposes `BlockSize`. |
| `BlockHash` | A single block key. |
| `PodEntry` | A holder of a block: a pod and device tier, plus a group for the kinds where groups are separate holders. |
| `BlockExtraFeatures` / `ComputeBlockExtraFeatures` | Per-block multimodal metadata folded into the hash. |
| `PlaceholderRange` | Placeholder-token range for a multimodal item. |

## Related Documentation

- [KV-Cache Indexer](../README.md) -- reads this index to score pods
- [KV-Events](../../kvevents/README.md) -- writes to this index from engine events
