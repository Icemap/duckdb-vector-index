# Matches DuckDB's own upper bound. The Windows mingw runner in
# extension-ci-tools ships CMake 3.24, so we can't raise this without a
# custom pre-install step in the reusable workflow.
cmake_minimum_required(VERSION 3.5...3.29)

# -----------------------------------------------------------------------------
# Extension identity
# -----------------------------------------------------------------------------
set(TARGET_NAME vindex)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME} CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(VINDEX_BUILD_UNIT
       "Build Catch2 unit tests in test/unit/" OFF)
option(VINDEX_BUILD_BENCH
       "Build standalone microbenchmarks in test/bench/" OFF)

# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------
include_directories(src)
include_directories(src/include)
include_directories(src/include/third_party)

set(EXTENSION_SOURCES src/vindex_extension.cpp)

# Each of these subdirs appends to EXTENSION_SOURCES via PARENT_SCOPE.
add_subdirectory(src/common)
add_subdirectory(src/algo)
add_subdirectory(src/quant)

# -----------------------------------------------------------------------------
# Build the static + loadable extension (DuckDB build macros)
# -----------------------------------------------------------------------------
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

# -----------------------------------------------------------------------------
# Optional: Catch2 unit tests
# -----------------------------------------------------------------------------
# The unit tests and microbenchmarks reach into DuckDB-internal C++ classes
# (FixedSizeAllocator, FixedSizeBuffer, DatabaseManager::Get,
# AttachedDatabase::GetStorageManager, ...). On Linux/macOS the DuckDB shared
# library exports these symbols by default. On Windows duckdb.lib only exports
# the public C API — internal symbols lack __declspec(dllexport), so the test
# binary fails to link. Silently skip building the test/bench binaries on
# Windows; SQL logic tests (`make test`) still run on every platform.
if(VINDEX_BUILD_UNIT AND NOT WIN32)
  enable_testing()
  add_subdirectory(test/unit)
endif()

# -----------------------------------------------------------------------------
# Optional: microbenchmarks
# -----------------------------------------------------------------------------
if(VINDEX_BUILD_BENCH AND NOT WIN32)
  add_subdirectory(test/bench)
endif()
