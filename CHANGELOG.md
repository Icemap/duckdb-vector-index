# Changelog

All notable changes to `vindex` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-28

First tagged release. Ships as an unsigned DuckDB extension built against
DuckDB v1.5.2 (`LOAD 'path/to/vindex.duckdb_extension'` with
`SET allow_unsigned_extensions = true;`).

### Added

- **HNSW index** (`USING HNSW`) — in-house graph implementation (`HnswCore`)
  over `IndexBlockStore`; replaces the `usearch` wrapper carried by
  upstream `duckdb-vss`. WITH options: `metric`, `quantizer`, `bits`,
  `rerank`, `M`, `M0`, `ef_construction`, `ef_search`.
- **IVF index** (`USING IVF`) — k-means++ centroids plus per-list posting
  buffers; supports IVF-Flat, IVF-RaBitQ, IVF-PQ. WITH options: `metric`,
  `quantizer`, `bits`, `rerank`, `nlist`, `nprobe`.
- **DiskANN index** (`USING DISKANN`) — Vamana graph with codes held
  out-of-band so graph blocks can evict past RAM via the DuckDB buffer
  pool. WITH options: `metric`, `quantizer`, `bits`, `rerank`,
  `diskann_r`, `diskann_l`, `diskann_alpha`. Accepts `quantizer='pq'` or
  `quantizer='rabitq'`; rejects `flat`.
- **RaBitQ quantizer** (`quantizer='rabitq'`) — rotated + bit-packed
  codes at `bits ∈ {1,2,3,4,5,7,8}`; 3-bit is the default and hits >99%
  Recall@10 on SIFT1M with `rerank=10`.
- **PQ quantizer** (`quantizer='pq'`) — classical product quantization
  with k-means++ per-segment codebooks; `bits ∈ {4, 8}`, `m` defaults to
  `dim/4`.
- **Flat quantizer** (`quantizer='flat'`) — float32 passthrough,
  bit-for-bit exact distances.
- **Rerank pass** — `WITH (rerank = N)` (or session pragma
  `SET vindex_rerank_multiple = N`) has the planner pull `k × N`
  candidates from the index and re-score against the authoritative
  `FLOAT[d]` column. Uniform `TOP_N ← PROJECTION ← VINDEX_INDEX_SCAN`
  plan shape across algorithms.
- **Session pragmas** — `vindex_ef_search`, `vindex_nprobe`,
  `vindex_diskann_l_search`, `vindex_rerank_multiple`,
  `vindex_enable_experimental_persistence`.
- **Info pragmas** — `pragma_vindex_hnsw_index_info()`,
  `pragma_vindex_ivf_index_info()`,
  `pragma_vindex_diskann_index_info()`.
- **Compact pragma** — `CALL vindex_compact_index('<idx>')` reclaims
  tombstoned entries (IVF rebuilds posting lists in place; HNSW/DiskANN
  currently clear the tombstone set and mark the index dirty).
- **Persistence** — indexes round-trip through checkpoint and WAL; state
  stream carries quantizer blob + core state + row mapping + tombstones.
- **Recall harness** — `make bench` downloads siftsmall on first run and
  fails non-zero on Recall@10 regressions. `run_recall.py --dataset
  sift1m` is wired but gated.
- **GitHub release pipeline** —
  `.github/workflows/MainDistributionPipeline.yml` builds the multi-arch
  matrix on tag push (`v*`) and publishes a GitHub Release with
  per-arch `vindex.<arch>.duckdb_extension` assets.

### Deprecated

The following names are retained as aliases of their `vindex_*`
replacements for at least one release; they will be removed in a
future version:

- `hnsw_enable_experimental_persistence` → `vindex_enable_experimental_persistence`
- `hnsw_ef_search` → `vindex_ef_search`
- `hnsw_compact_index(...)` → `vindex_compact_index(...)`
- `pragma_hnsw_index_info()` → `pragma_vindex_hnsw_index_info()`

[Unreleased]: https://github.com/Icemap/duckdb-vector-index/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Icemap/duckdb-vector-index/releases/tag/v0.1.0
