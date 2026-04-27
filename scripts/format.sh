#!/usr/bin/env bash
# clang-format runner. Skips vendored third-party headers.
set -euo pipefail

cd "$(dirname "$0")/.."

files=$(find src test/unit \
    \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) \
    -not -path 'src/include/third_party/*')

if [[ -z "${files}" ]]; then
    echo "[vindex] no sources to format."
    exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
    echo "${files}" | xargs clang-format --dry-run --Werror
else
    echo "${files}" | xargs clang-format -i
fi
