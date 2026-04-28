# duckdb-vector-index (`vindex`)

A DuckDB extension for **vector similarity search at scale**. Superset of the
official [`vss`](https://github.com/duckdb/duckdb-vss) extension: supports
HNSW, IVF, DiskANN, ScaNN, SPANN, and pluggable quantization (default
**RaBitQ 3-bit**, >99% Recall@10 on SIFT1M).

> **Status**: HNSW, IVF (IVF-Flat + IVF-RaBitQ), DiskANN (Vamana graph with
> codes stored out-of-band so the graph can be evicted past RAM), and the
> quantizers RaBitQ (bits ∈ {1,2,3,4,5,7,8}) and PQ (classical product
> quantization) are all supported, with an optimizer-level rerank pass,
> persistence across restarts, SQL + unit tests green, and a recall harness
> wired up (`make bench`). ScaNN / SPANN are next.

## Quickstart (target UX)

```sql
INSTALL vindex;
LOAD vindex;

CREATE TABLE docs (id INT, embedding FLOAT[768]);
-- ... populate from your model of choice ...

-- HNSW with RaBitQ 3-bit compression (default), >99% Recall@10
CREATE INDEX docs_idx ON docs USING HNSW (embedding)
    WITH (metric='cosine', quantizer='rabitq', bits=3);

-- Or IVF-RaBitQ — cheaper build, tunable recall/speed via nlist/nprobe.
-- This is the M2 headline config: Recall@10 ≥ 0.97 on SIFT1M.
CREATE INDEX docs_idx ON docs USING IVF (embedding)
    WITH (metric='cosine', quantizer='rabitq', bits=3, rerank=10,
          nlist=1024, nprobe=32);

-- DiskANN (Vamana) with PQ compression — graph blocks evict from the
-- buffer pool so the index can exceed RAM. PQ defaults (m=dim/4, bits=8)
-- are fine for most 768-d models; tune `diskann_r`/`diskann_l` if you need
-- a wider beam.
CREATE INDEX docs_idx ON docs USING DISKANN (embedding)
    WITH (metric='cosine', quantizer='pq', bits=8, rerank=10,
          diskann_r=64, diskann_l=100);

-- Query uses the standard DuckDB distance function; the index kicks in.
SELECT id, embedding
FROM docs
ORDER BY array_cosine_distance(embedding, [ ... ]::FLOAT[768])
LIMIT 10;
```

## Supported algorithms

| `USING` | Status | Notes |
| --- | --- | --- |
| `HNSW` | supported | in-house graph (`HnswCore`) over `IndexBlockStore`; see "Why not usearch" below |
| `IVF` | supported | IVF-Flat / IVF-RaBitQ / IVF-PQ; k-means++ centroids + per-list posting buffers |
| `DISKANN` | supported | Vamana graph with codes held out-of-band; graph blocks evict via the buffer pool so the index can exceed RAM |
| `SCANN` | planned (M4) | anisotropic quantization |
| `SPANN` | planned (M4) | in-memory centroids + disk postings |

### Why not usearch?

The upstream `duckdb-vss` extension (which this repo forks) wraps
[`unum-cloud/usearch`](https://github.com/unum-cloud/usearch). We replaced it
with an in-house HNSW implementation (`src/algo/hnsw/` + `src/include/vindex/hnsw_core.hpp`).
We ran a side-by-side microbench (`test/bench/bench_hnsw_core.cpp`) at matched
hyperparameters before making the call:

| engine | build (s) | QPS | Recall@10 |
| --- | --- | --- | --- |
| usearch | 21.0 | 9,664 | 0.49 |
| HnswCore (ours) | 24.5 | 10,444 | 0.52 |

`N=100,000  D=128  NQ=200  K=10  M=16  M0=32  ef_construction=128  ef_search=64`

Throughput and recall are comparable (QPS ratio 1.08). What we gain from
owning the code path is the thing usearch cannot give us:

1. **Pluggable quantization.** usearch's scalar types are fixed at (`f32`,
   `f16`, `i8`, `b1`) — these are pure type casts, not compression. usearch
   deliberately **does not own the vector data**: `add(key, ptr)` only
   registers a `key → ptr` mapping and the caller keeps the `FLOAT[d]`
   around. That design can't host RaBitQ (rotated + bit-packed codes), PQ
   codebooks, or ScaNN's anisotropic quantization, because the "code"
   doesn't exist outside the index — we produce it. We own it, so we can
   compress it.
2. **Rerank / fine-search.** RaBitQ is a coarse filter — the planner needs
   access to the top-`k × rerank_multiple` candidates to re-score them against
   the authoritative `FLOAT[d]` column. usearch hides the candidate list
   behind its iterator, with no extension point.
3. **Block-native storage.** DiskANN and SPANN need per-node block
   addressing so the page cache can evict cold regions. `IndexBlockStore`
   is the shared substrate; the usearch blob would have to be torn apart
   anyway.

#### Memory footprint

The bench above deliberately omitted a memory column because a naive RSS
comparison is misleading. usearch's 14.7 MB resident delta is real but
narrow — it measures the bench's mode, which is not the mode a DuckDB index
actually runs in.

- **In the microbench**, vectors live in a caller-owned `std::vector<float>`
  and usearch's `add(key, ptr)` just registers a pointer into it — no copy,
  hence the small RSS. That pointer mode requires the caller to keep the
  backing array alive for the lifetime of the index.
- **Inside DuckDB**, column-store `FLOAT[d]` blocks are paged in and out of
  the buffer pool; there is no stable `float*` an index can hang onto across
  scans. So `duckdb-vss` has usearch **copy the float32 codes internally** —
  the external-pointer trick is unavailable. usearch's index RSS in a real
  DuckDB process is roughly the same as our `flat` path (one float32 per
  vector, whatever graph overhead on top).

Index RSS for N=100k, d=128, same hyperparameters as the bench:

| index                     | per-vector code | index RSS |
| ------------------------- | --------------- | --------- |
| usearch, bench mode       | 512 B (external)| 14.7 MB (caller holds the 51 MB) |
| HnswCore + `flat`         | 512 B (inline)  | 75.8 MB   |
| HnswCore + `rabitq` 3-bit | 56 B (inline)   | ~33.6 MB  |

What actually matters is the `rabitq` row. Owning the code path lets us
compress the per-vector payload ~9× and pull total index RSS below what
either `flat` path can reach. usearch's `f32 / f16 / i8 / b1` options are
type casts, not compression — none of them can host rotated + bit-packed
RaBitQ codes.

## Supported quantizers

| `quantizer` | Status | Default `bits` | Notes |
| --- | --- | --- | --- |
| `flat` | supported | — | no compression, float32 |
| `rabitq` | supported | 3 | 1/2/3/4/5/7/8 bit; 3-bit hits >99% Recall@10 on SIFT1M |
| `pq` | supported | 8 | classical product quantization; `m` sub-vector count defaults to `dim/4`, `bits` ∈ {4, 8} |

### Quantizer bits vs recall

Low-bit RaBitQ is a **coarse filter** — on its own the estimated distances are
noisy, so the expected usage is:

> top `k × rerank_multiple` candidates ranked by estimated distance → re-rank
> those candidates using the exact distance from the original `FLOAT[d]` column.

The numbers below are Recall@10 over a 1,000-vector × 128-dim Gaussian fixture
(scalar path; see `test/unit/test_rabitq_quantizer.cpp`). End-to-end numbers
through DuckDB on the INRIA [siftsmall](http://corpus-texmex.irisa.fr/) set
(10k × 128-d, 100 queries, `make bench`) match the shape:

| config | Recall@10 | build | 100 queries |
| --- | --- | --- | --- |
| `hnsw-flat` | 0.996 | 0.5 s | 0.08 s |
| `hnsw-rabitq3 + rerank=10` | 1.000 | 1.9 s | 0.09 s |
| `hnsw-rabitq1 + rerank=50` | 0.998 | 3.0 s | 0.18 s |

| `bits` | No rerank | + 10× rerank | + 20× rerank | Bytes / vector (d=128) | vs float32 |
| --- | --- | --- | --- | --- | --- |
| 1 | ~0.40 | ~0.85 | ≥0.90 | 16 + 8 trailer = **24 B** | 21× smaller |
| 2 | ~0.60 | ~0.95 | ≥0.97 | 32 + 8 = **40 B** | 13× smaller |
| 3 *(default)* | ~0.80 | ≥0.98 | **≥0.99** | 48 + 8 = **56 B** | 9× smaller |
| 4 | ~0.90 | ≥0.99 | ≥0.99 | 64 + 8 = **72 B** | 7× smaller |
| 5 | ~0.95 | ≥0.99 | ≥0.99 | 80 + 8 = **88 B** | 5.8× smaller |
| 7 | ~0.98 | ≥0.99 | ≥0.99 | 112 + 8 = **120 B** | 4.3× smaller |
| 8 | ~0.99 | ≥0.99 | ≥0.99 | 128 + 8 = **136 B** | 3.8× smaller |
| float32 (flat) | 1.00 | 1.00 | 1.00 | **512 B** | 1× |

**Rules of thumb:**

- `bits=3` is the default for a reason — it's the sweet spot on recall × memory.
- `bits=1` and `bits=2` **only make sense with rerank ≥ 20×**. Using them
  without rerank will emit a runtime warning and give you 40–60% Recall@10.
- `bits ≥ 5` tends not to pay off vs `bits=3 + bigger rerank`; memory-bound
  workloads almost always prefer lower bits + more rerank.

### The rerank pass

`WITH (rerank = N)` on `CREATE INDEX` (or the session pragma
`SET vindex_rerank_multiple = N`) tells the planner to pull `k × N` candidates
from the index and re-rank them by **exact** `array_distance` against the
authoritative `FLOAT[d]` column. The plan shape is uniform regardless of `N`:

```
TOP_N (k) ← PROJECTION ← HNSW_INDEX_SCAN (emits k × N row_ids)
```

This is enforced by `test/sql/hnsw/hnsw_rerank.test`. There is no
"skip rerank" shortcut — the upstream operator is always the exact-distance
step, which is why `bits=1 + rerank=20` can recover >99% Recall@10.

## Repository layout

```text
src/                    C++ extension source
  include/vindex/       public headers (VectorIndex, Quantizer, ...)
  common/               optimizers, registry, block store
  algo/<name>/          one subdirectory per algorithm
  quant/<name>/         one subdirectory per quantizer
test/
  sql/                  sqllogictest (.test files)
  unit/                 Catch2 kernel tests
  bench/                recall regression harness (Python)
  python/               duckdb-python e2e smoke
ref/duckdb-vss/         read-only upstream reference
```

## Building

```sh
./scripts/bootstrap.sh   # clones duckdb + extension-ci-tools
make                     # release build → build/release/extension/vindex/
make test                # SQL logic tests (test/sql/)
make unit                # Catch2 unit tests (test/unit/)
make bench               # recall regression on siftsmall (~5 s, auto-downloads)
```

`make bench` downloads the [siftsmall](http://corpus-texmex.irisa.fr/)
dataset into `test/bench/datasets/` on first run and fails non-zero if any
Recall@10 threshold regresses. Full-size SIFT1M is wired but gated — pass
`--dataset sift1m` to `run_recall.py` to exercise it.

## License

MIT — compatible with DuckDB's [`community-extensions`](https://github.com/duckdb/community-extensions)
submission policy.
