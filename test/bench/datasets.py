"""Dataset download + loading helpers for the recall harness.

Supported datasets:
    siftsmall   : 10k 128-d SIFT vectors  (~5 MB)   — default smoke set
    sift1m      : 1M  128-d SIFT vectors  (~500 MB) — full regression

INRIA TEXMEX ships both as `.fvecs` / `.ivecs`:
    fvecs  = [int32 dim][float32 * dim] per row
    ivecs  = [int32 dim][int32   * dim] per row

All files live under test/bench/datasets/<name>/; large files are gitignored.
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASETS_DIR = Path(__file__).parent / "datasets"

_URLS = {
    "siftsmall": "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
    "sift1m": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
}

# Extracted tarball directory name != download name for sift1m.
_EXTRACT_DIR = {
    "siftsmall": "siftsmall",
    "sift1m": "sift",
}

# Within the extracted dir, file stems also differ between the two.
_FILE_STEM = {
    "siftsmall": "siftsmall",
    "sift1m": "sift",
}


@dataclass
class Dataset:
    name: str
    base: np.ndarray       # (N, d) float32
    query: np.ndarray      # (Q, d) float32
    groundtruth: np.ndarray  # (Q, k) int32 — top-k base row ids per query
    dim: int

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Dataset({self.name}: base={self.base.shape} "
                f"query={self.query.shape} gt={self.groundtruth.shape})")


def _read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    dim = int(raw[0])
    stride = dim + 1
    assert raw.size % stride == 0, f"malformed fvecs: {path}"
    return raw.reshape(-1, stride)[:, 1:].copy().view(np.float32)


def _read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    dim = int(raw[0])
    stride = dim + 1
    assert raw.size % stride == 0, f"malformed ivecs: {path}"
    return raw.reshape(-1, stride)[:, 1:].copy()


def _download_and_extract(name: str) -> Path:
    url = _URLS[name]
    out_dir = DATASETS_DIR / _EXTRACT_DIR[name]
    if out_dir.exists():
        return out_dir
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATASETS_DIR / f"{name}.tar.gz"
    if not tar_path.exists():
        print(f"[datasets] downloading {url}")
        urllib.request.urlretrieve(url, tar_path)
    print(f"[datasets] extracting {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(DATASETS_DIR)
    return out_dir


def load(name: str) -> Dataset:
    if name not in _URLS:
        raise ValueError(f"unknown dataset: {name}. supported: {list(_URLS)}")
    dir_path = _download_and_extract(name)
    stem = _FILE_STEM[name]
    base = _read_fvecs(dir_path / f"{stem}_base.fvecs")
    query = _read_fvecs(dir_path / f"{stem}_query.fvecs")
    gt = _read_ivecs(dir_path / f"{stem}_groundtruth.ivecs")
    assert base.ndim == 2 and query.ndim == 2
    assert base.shape[1] == query.shape[1]
    assert gt.shape[0] == query.shape[0]
    return Dataset(name=name, base=base, query=query, groundtruth=gt, dim=base.shape[1])
