# Standalone HnswCore microbench. Built only when -DVINDEX_BUILD_BENCH=ON is
# passed to cmake (see top-level CMakeLists.txt).
#
# Catch2 is NOT used here because the bench needs full control over timing and
# output format.

add_executable(bench_hnsw_core
    bench_hnsw_core.cpp
    ${PROJECT_SOURCE_DIR}/src/common/metric.cpp
    ${PROJECT_SOURCE_DIR}/src/common/index_block_store.cpp
    ${PROJECT_SOURCE_DIR}/src/algo/hnsw/hnsw_core.cpp
    ${PROJECT_SOURCE_DIR}/src/quant/flat/flat_quantizer.cpp
    ${PROJECT_SOURCE_DIR}/src/quant/rabitq/rabitq_rotate.cpp
    ${PROJECT_SOURCE_DIR}/src/quant/rabitq/rabitq_quantizer.cpp
    ${PROJECT_SOURCE_DIR}/src/quant/rabitq/rabitq_kernels.cpp
)

target_include_directories(bench_hnsw_core PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/src/include/third_party
)
target_link_libraries(bench_hnsw_core PRIVATE duckdb)
