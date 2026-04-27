# This file is included by DuckDB's build system.
# It specifies which extension(s) to load into the duckdb build.

# The vindex extension lives in this repository.
duckdb_extension_load(vindex
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
        LOAD_TESTS
        )

# Any extra extensions that should be built alongside for testing purposes can
# be loaded here, e.g.:
# duckdb_extension_load(json)
