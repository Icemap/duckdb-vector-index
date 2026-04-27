# Catch2 unit tests for algorithm kernels. Built only when
# -DVINDEX_BUILD_UNIT=ON is passed to cmake (see top-level CMakeLists.txt).

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.3
)
FetchContent_MakeAvailable(Catch2)

add_executable(vindex_unit
    test_metric.cpp
    ${PROJECT_SOURCE_DIR}/src/common/metric.cpp
    # test_rabitq_encode.cpp   # M1
    # test_rabitq_recall.cpp   # M1
    # test_kmeans.cpp          # M2
)

target_include_directories(vindex_unit PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
# duckdb_static brings in `duckdb/common/...` headers + string types that
# metric.cpp / the tests depend on. We only need its interface (no runtime
# execution), so a direct link is fine.
# Link the shared libduckdb rather than duckdb_static — the static lib assumes
# the `duckdb_generated_extension_loader` object is present, which would drag
# in the full extension bootstrap (parquet + core_functions + vindex vtables).
# We only need duckdb for its `StringUtil` + exception symbols.
target_link_libraries(vindex_unit PRIVATE Catch2::Catch2WithMain duckdb)

include(Catch)
catch_discover_tests(vindex_unit PROPERTIES LABELS "unit")
