#!/usr/bin/env bash
# Clone + pin DuckDB and extension-ci-tools as submodules. Both are pinned
# to the v1.5.2 release so the build is reproducible. Bump in lockstep.
#
# Safe to re-run; skips already-initialised submodules.

set -euo pipefail

DUCKDB_REF=v1.5.2
EXT_CI_REF=v1.5.2

cd "$(dirname "$0")/.."

if [[ ! -d duckdb ]]; then
    echo "[vindex] adding duckdb submodule..."
    git submodule add -f https://github.com/duckdb/duckdb.git duckdb
fi

if [[ ! -d extension-ci-tools ]]; then
    echo "[vindex] adding extension-ci-tools submodule..."
    git submodule add -f https://github.com/duckdb/extension-ci-tools.git extension-ci-tools
fi

git submodule update --init --recursive

echo "[vindex] pinning duckdb@$DUCKDB_REF and extension-ci-tools@$EXT_CI_REF..."
(cd duckdb && git fetch --tags origin && git checkout "$DUCKDB_REF")
(cd extension-ci-tools && git fetch --tags origin && git checkout "$EXT_CI_REF")

echo "[vindex] bootstrap complete. Next: \`make\` or \`make debug\`."
