# M1: RaBitQ (default 3-bit, >99% Recall@10). Sources listed here once
# implemented. Kernel split:
#   - rabitq_quantizer.cpp : Train / Encode / EstimateDistance / Serialize
#   - rabitq_rotate.cpp    : deterministic random rotation (Gaussian/Householder)
#   - rabitq_kernels.cpp   : SIMD distance kernels, reuses simsimd
set(EXTENSION_SOURCES
    ${EXTENSION_SOURCES}
    # ${CMAKE_CURRENT_SOURCE_DIR}/rabitq_quantizer.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/rabitq_rotate.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/rabitq_kernels.cpp
    PARENT_SCOPE)
