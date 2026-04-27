"""Dataset download + loading helpers for the recall harness.

Supported datasets (added incrementally):
    sift1m      : 1M 128-d SIFT vectors   (~500 MB)
    gist1m      : 1M 960-d GIST vectors   (~3.6 GB)
    deep10m     : 10M 96-d DEEP vectors   (~3.8 GB)

All files live under test/bench/datasets/<name>/; large files are gitignored.
"""

from __future__ import annotations

from pathlib import Path

DATASETS_DIR = Path(__file__).parent / "datasets"


def dataset_path(name: str) -> Path:
    return DATASETS_DIR / name


def require(name: str) -> Path:
    # TODO(m1): actually download + verify checksums.
    p = dataset_path(name)
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {p}. Run `scripts/download_dataset.sh {name}`.")
    return p
