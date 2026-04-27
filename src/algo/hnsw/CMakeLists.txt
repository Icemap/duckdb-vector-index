# M0: HNSW implementation ported from ref/duckdb-vss. Keep the file split
# close to the reference so that diffing remains tractable.
set(EXTENSION_SOURCES
    ${EXTENSION_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_register.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_build.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index_scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_pragmas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_core.cpp
    PARENT_SCOPE)
