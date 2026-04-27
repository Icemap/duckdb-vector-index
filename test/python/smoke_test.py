"""End-to-end smoke via the duckdb Python package.

Verifies LOAD vindex; succeeds and the expected index type name is present.
Populated once the extension can be built and picked up by duckdb-python's
`extension_directory` config.
"""

from __future__ import annotations

import pytest


def test_extension_loads(tmp_path) -> None:
    pytest.skip("scaffold: wire up once `make` produces vindex.duckdb_extension")
