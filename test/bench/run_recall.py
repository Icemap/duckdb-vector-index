#!/usr/bin/env python3
"""Recall regression harness for duckdb-vector-index.

Loads a dataset (default: siftsmall, ~5 MB), builds an HNSW index over the
base vectors through the built vindex extension, runs each of N queries
through the index, and measures Recall@10 against the ground truth.

Fails non-zero if any configured algorithm falls below its threshold; writes
a machine-readable JSON report to test/bench/results/latest.json.

Usage:
    make bench                                # runs siftsmall, default matrix
    python3 test/bench/run_recall.py --dataset siftsmall
    python3 test/bench/run_recall.py --dataset sift1m --algo hnsw-rabitq3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

# Ensure sibling module import works regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent))
import datasets  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTENSION = REPO_ROOT / "build" / "release" / "extension" / "vindex" / "vindex.duckdb_extension"

K = 10  # Recall@K target

# (algo-key → (USING clause, WITH-clause)). The harness issues the same
# ORDER BY + LIMIT query for each; rerank pairings match README
# §"Quantizer bits vs recall".
CONFIGS = {
    "hnsw-flat":           ("HNSW", ""),
    "hnsw-rabitq3":        ("HNSW", "WITH (quantizer='rabitq', bits=3, rerank=10)"),
    "hnsw-rabitq1-rerank": ("HNSW", "WITH (quantizer='rabitq', bits=1, rerank=50)"),
    "hnsw-pq":             ("HNSW", "WITH (quantizer='pq', bits=8, rerank=10)"),
    "hnsw-scann":          ("HNSW", "WITH (quantizer='scann', bits=8, eta=4.0, rerank=10)"),
    "ivf-flat":            ("IVF",  "WITH (nlist=128, nprobe=16)"),
    "ivf-rabitq3":         ("IVF",  "WITH (quantizer='rabitq', bits=3, rerank=10, nlist=128, nprobe=16)"),
    "ivf-pq":              ("IVF",  "WITH (quantizer='pq', m=16, bits=8, rerank=10, nlist=128, nprobe=16)"),
    "ivf-scann":           ("IVF",  "WITH (quantizer='scann', m=16, bits=8, eta=4.0, rerank=10, nlist=128, nprobe=16)"),
    "diskann-pq":          ("DISKANN", "WITH (quantizer='pq', m=16, bits=8, rerank=10, diskann_r=64, diskann_l=100)"),
    "diskann-rabitq3":     ("DISKANN", "WITH (quantizer='rabitq', bits=3, rerank=10, diskann_r=64, diskann_l=100)"),
    "diskann-scann":       ("DISKANN", "WITH (quantizer='scann', m=16, bits=8, eta=4.0, rerank=10, diskann_r=64, diskann_l=100)"),
    "spann-flat":          ("SPANN", "WITH (nlist=128, nprobe=16, replica_count=8, closure_factor=1.1)"),
    "spann-rabitq3":       ("SPANN", "WITH (quantizer='rabitq', bits=3, rerank=10, nlist=128, nprobe=16, replica_count=8, closure_factor=1.1)"),
    "spann-pq":            ("SPANN", "WITH (quantizer='pq', m=16, bits=8, rerank=10, nlist=128, nprobe=16, replica_count=8, closure_factor=1.1)"),
    "spann-scann":         ("SPANN", "WITH (quantizer='scann', m=16, bits=8, eta=4.0, rerank=10, nlist=128, nprobe=16, replica_count=8, closure_factor=1.1)"),
}

# Recall@10 thresholds per (algo, dataset). Anything below → non-zero exit.
# sift1m thresholds for IVF follow AGENTS.md §6.3 — ivf-rabitq3 is the M2
# milestone gate (≥0.97 at nlist=1024/nprobe=32). The bench default config
# keeps the more conservative nlist=128/nprobe=16 for fast CI runs; the
# milestone-grade threshold is enforced under the `-sift1m` profile.
THRESHOLDS = {
    ("hnsw-flat",           "siftsmall"): 0.98,
    ("hnsw-rabitq3",        "siftsmall"): 0.95,
    ("hnsw-rabitq1-rerank", "siftsmall"): 0.90,
    ("hnsw-pq",             "siftsmall"): 0.90,
    ("hnsw-scann",          "siftsmall"): 0.90,
    ("ivf-flat",            "siftsmall"): 0.90,
    ("ivf-rabitq3",         "siftsmall"): 0.90,
    ("ivf-pq",              "siftsmall"): 0.90,
    ("ivf-scann",           "siftsmall"): 0.90,
    ("diskann-pq",          "siftsmall"): 0.90,
    ("diskann-rabitq3",     "siftsmall"): 0.90,
    ("diskann-scann",       "siftsmall"): 0.90,
    ("spann-flat",          "siftsmall"): 0.90,
    ("spann-rabitq3",       "siftsmall"): 0.90,
    ("spann-pq",            "siftsmall"): 0.90,
    ("spann-scann",         "siftsmall"): 0.90,
    ("hnsw-flat",           "sift1m"):    0.98,
    ("hnsw-rabitq3",        "sift1m"):    0.99,
    ("hnsw-rabitq1-rerank", "sift1m"):    0.90,
    ("hnsw-pq",             "sift1m"):    0.98,
    ("hnsw-scann",          "sift1m"):    0.98,
    ("ivf-flat",            "sift1m"):    0.95,
    ("ivf-rabitq3",         "sift1m"):    0.97,
    ("ivf-pq",              "sift1m"):    0.95,
    ("ivf-scann",           "sift1m"):    0.95,
    ("diskann-pq",          "sift1m"):    0.95,
    ("diskann-rabitq3",     "sift1m"):    0.95,
    ("diskann-scann",       "sift1m"):    0.95,
    ("spann-flat",          "sift1m"):    0.95,
    ("spann-rabitq3",       "sift1m"):    0.95,
    ("spann-pq",            "sift1m"):    0.95,
    ("spann-scann",         "sift1m"):    0.95,
}


@dataclass
class RunResult:
    algo: str
    dataset: str
    build_s: float
    query_s: float
    recall_at_k: float
    threshold: float

    @property
    def passed(self) -> bool:
        return self.recall_at_k >= self.threshold


def _connect(extension_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{extension_path}'")
    return con


def _load_base_table(con: duckdb.DuckDBPyConnection, base: np.ndarray) -> None:
    dim = base.shape[1]
    con.execute(f"CREATE TABLE base (id INTEGER, vec FLOAT[{dim}])")
    # Arrow path: build a RecordBatch with a FixedSizeList<float32, dim>.
    import pyarrow as pa
    vec_type = pa.list_(pa.float32(), dim)
    flat = pa.array(base.reshape(-1).astype(np.float32, copy=False), type=pa.float32())
    vec_col = pa.FixedSizeListArray.from_arrays(flat, dim)
    ids = pa.array(np.arange(base.shape[0], dtype=np.int32))
    table = pa.table({"id": ids, "vec": vec_col})
    con.register("base_df", table)
    con.execute("INSERT INTO base SELECT id, vec FROM base_df")
    con.unregister("base_df")


def _run_queries(con: duckdb.DuckDBPyConnection, queries: np.ndarray) -> tuple[np.ndarray, float]:
    # Build one query-vector-keyed scan per query. sqllogictest uses literal
    # FLOAT[] casts; we do the same to exercise the optimizer's constant-vector
    # path. (Subquery-sourced vectors don't trigger VINDEX_INDEX_SCAN — see
    # test/sql/rabitq/rabitq_basic.test line 100.)
    q_count = queries.shape[0]
    out = np.full((q_count, K), -1, dtype=np.int32)
    t0 = time.perf_counter()
    for i, q in enumerate(queries):
        literal = "[" + ",".join(f"{float(x):.7g}" for x in q) + f"]::FLOAT[{queries.shape[1]}]"
        rows = con.execute(
            f"SELECT id FROM base ORDER BY array_distance(vec, {literal}) LIMIT {K}"
        ).fetchall()
        for j, (rid,) in enumerate(rows):
            out[i, j] = rid
    return out, time.perf_counter() - t0


def _recall_at_k(retrieved: np.ndarray, gt: np.ndarray, k: int) -> float:
    assert retrieved.shape[0] == gt.shape[0]
    hits = 0
    total = retrieved.shape[0] * k
    for i in range(retrieved.shape[0]):
        truth = set(gt[i, :k].tolist())
        hits += sum(1 for rid in retrieved[i] if rid in truth)
    return hits / total


def _run_one(extension_path: Path, ds: datasets.Dataset, algo: str, using: str, with_clause: str) -> RunResult:
    threshold = THRESHOLDS.get((algo, ds.name), 0.0)
    print(f"[bench] {algo} on {ds.name}: build…", file=sys.stderr)
    con = _connect(extension_path)
    _load_base_table(con, ds.base)
    t0 = time.perf_counter()
    con.execute(f"CREATE INDEX idx ON base USING {using} (vec) {with_clause}")
    build_s = time.perf_counter() - t0
    print(f"[bench] {algo} on {ds.name}: build={build_s:.2f}s, query…", file=sys.stderr)
    retrieved, query_s = _run_queries(con, ds.query)
    recall = _recall_at_k(retrieved, ds.groundtruth, K)
    con.close()
    print(f"[bench] {algo} on {ds.name}: Recall@{K}={recall:.4f} "
          f"(threshold {threshold:.2f}) query={query_s:.2f}s", file=sys.stderr)
    return RunResult(algo=algo, dataset=ds.name, build_s=build_s, query_s=query_s,
                     recall_at_k=recall, threshold=threshold)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="siftsmall")
    parser.add_argument("--algo", default="all", help="all or one of: " + ",".join(CONFIGS))
    parser.add_argument("--extension", default=str(DEFAULT_EXTENSION))
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    extension_path = Path(args.extension)
    if not extension_path.exists():
        print(f"error: extension not found: {extension_path}. Run `make` first.",
              file=sys.stderr)
        return 2

    ds = datasets.load(args.dataset)
    algos = list(CONFIGS) if args.algo == "all" else [args.algo]

    results: list[RunResult] = []
    for algo in algos:
        if algo not in CONFIGS:
            print(f"error: unknown algo: {algo}", file=sys.stderr)
            return 2
        using, with_clause = CONFIGS[algo]
        results.append(_run_one(extension_path, ds, algo, using, with_clause))

    out_path = Path(args.output) if args.output else Path(__file__).parent / "results" / "latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "dataset": ds.name,
        "k": K,
        "runs": [r.__dict__ | {"passed": r.passed} for r in results],
    }, indent=2))

    all_passed = all(r.passed for r in results)
    print(("[bench] ALL PASS" if all_passed else "[bench] FAIL"), file=sys.stderr)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
