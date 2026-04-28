PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# DuckDB community-extensions identity
EXT_NAME=vindex
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Forward vindex-specific cmake flags into DuckDB's build invocation.
EXT_FLAGS=-DVINDEX_BUILD_UNIT=ON -DVINDEX_BUILD_BENCH=ON

# Pull in DuckDB's standard extension makefile (clone/build/test targets)
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# -----------------------------------------------------------------------------
# Extra targets layered on top of the standard extension makefile
# -----------------------------------------------------------------------------

# Run Catch2 unit tests (built alongside the extension when VINDEX_BUILD_UNIT=ON)
.PHONY: unit
unit: release
	ctest --test-dir build/release/extension/vindex --output-on-failure -L unit

# Recall benchmark — requires datasets under test/bench/datasets/
.PHONY: bench
bench:
	python3 test/bench/run_recall.py

# HnswCore vs usearch microbench (built with VINDEX_BUILD_BENCH=ON).
# Prints build time / QPS / recall; see test/bench/bench_hnsw_core.cpp.
.PHONY: bench_hnsw_core
bench_hnsw_core: release
	./build/release/extension/vindex/test/bench/bench_hnsw_core

# One-shot formatter (src/ only, skips third_party/)
.PHONY: format
format:
	scripts/format.sh

# Algorithm × quantizer × metric acceptance matrix. Opt-in because each
# CREATE INDEX pays the full build pipeline; the file is skipped by default
# under `make test` via `require-env VINDEX_RUN_MATRIX`.
.PHONY: matrix
matrix: release
	VINDEX_RUN_MATRIX=1 ./build/release/test/unittest "test/sql/common/combo_matrix.test"
