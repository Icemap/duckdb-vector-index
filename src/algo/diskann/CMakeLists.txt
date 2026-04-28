# M3: DiskANN — Vamana graph ANN with out-of-band (RAM-resident) codes.
set(EXTENSION_SOURCES
    ${EXTENSION_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/diskann_register.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/diskann_index.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/diskann_build.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/diskann_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/diskann_pragmas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/diskann_core.cpp
    PARENT_SCOPE)
