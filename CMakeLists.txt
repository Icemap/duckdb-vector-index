cmake_minimum_required(VERSION 3.29)

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
if(VINDEX_BUILD_UNIT)
  enable_testing()
  add_subdirectory(test/unit)
endif()

# -----------------------------------------------------------------------------
# Optional: microbenchmarks
# -----------------------------------------------------------------------------
if(VINDEX_BUILD_BENCH)
  add_subdirectory(test/bench)
endif()
