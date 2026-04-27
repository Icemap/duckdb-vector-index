# duckdb-vector-index (`vindex`)

A DuckDB extension for **vector similarity search at scale**. Superset of the
official [`vss`](https://github.com/duckdb/duckdb-vss) extension: supports
HNSW, IVF, DiskANN, ScaNN, SPANN, and pluggable quantization (default
**RaBitQ 3-bit**, >99% Recall@10 on SIFT1M).

> **Status**: early development. See `doc/plan.md` for milestones. M0 work in
> progress — only scaffolding is checked in so far.

## Quickstart (target UX)

```sql
INSTALL vindex;
LOAD vindex;

CREATE TABLE docs (id INT, embedding FLOAT[768]);
-- ... populate from your model of choice ...

-- HNSW with RaBitQ 3-bit compression (default), >99% Recall@10
CREATE INDEX docs_idx ON docs USING HNSW (embedding)
    WITH (metric='cosine', quantizer='rabitq', bits=3);

-- Query uses the standard DuckDB distance function; the index kicks in.
SELECT id, embedding
FROM docs
ORDER BY array_cosine_distance(embedding, [ ... ]::FLOAT[768])
LIMIT 10;
```

## Supported algorithms

| `USING` | Status | Notes |
| --- | --- | --- |
| `HNSW` | M0 (port) | backed by [usearch](https://github.com/unum-cloud/usearch), same as official `vss` |
| `IVF` | M2 | IVF-Flat / IVF-PQ / IVF-RaBitQ |
| `DISKANN` | M3 | Vamana graph + PQ, indexes larger than RAM |
| `SCANN` | M4 | anisotropic quantization |
| `SPANN` | M4 | in-memory centroids + disk postings |

## Supported quantizers

| `quantizer` | Status | Default `bits` | Notes |
| --- | --- | --- | --- |
| `flat` | M0 | — | no compression, float32 |
| `rabitq` | M1 | 3 | 1/2/3/4/5/7/8 bit; 3-bit hits >99% Recall@10 on SIFT1M |
| `pq` | M2 | — | classical product quantization |

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
doc/                    design & plan documents
ref/duckdb-vss/         read-only upstream reference
```

## Building

```sh
./scripts/bootstrap.sh   # clones duckdb + extension-ci-tools
make                     # release build → build/release/extension/vindex/
make test                # SQL logic tests
make unit                # Catch2 unit tests
```

See `AGENTS.md` for full contributor conventions (naming, testing thresholds,
community-extensions constraints).

## License

MIT — compatible with DuckDB's [`community-extensions`](https://github.com/duckdb/community-extensions)
submission policy.
