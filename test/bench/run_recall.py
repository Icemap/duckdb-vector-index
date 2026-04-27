#!/usr/bin/env python3
"""Recall regression harness for duckdb-vector-index.

Runs the matrix defined in AGENTS.md §6.3 (algorithm × dataset × metric) and
fails non-zero if any Recall@10 falls below the configured threshold. Writes
machine-readable output to test/bench/results/<timestamp>.json so CI can
surface regressions.

Usage:
    python3 test/bench/run_recall.py --dataset sift1m
    python3 test/bench/run_recall.py --all

Datasets must be downloaded into test/bench/datasets/ ahead of time; the
datasets module handles download + ground-truth caching.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# TODO(m1): wire up real runs. Scaffold only.

THRESHOLDS = {
    ("hnsw-flat", "sift1m"): 0.98,
    ("hnsw-rabitq3", "sift1m"): 0.99,
    ("hnsw-rabitq1", "sift1m"): 0.90,
    ("ivf-rabitq3", "sift1m"): 0.97,
    ("diskann-pq", "sift1m"): 0.95,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="sift1m")
    parser.add_argument("--algo", default="all")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    print(f"[vindex-bench] dataset={args.dataset} algo={args.algo}", file=sys.stderr)
    print("[vindex-bench] not implemented yet — scaffold only", file=sys.stderr)

    results = {"status": "scaffold", "thresholds": {f"{a}:{d}": v for (a, d), v in THRESHOLDS.items()}}
    out_path = Path(args.output) if args.output else Path(__file__).parent / "results" / "latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
