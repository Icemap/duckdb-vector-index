add_subdirectory(hnsw)
add_subdirectory(ivf)
add_subdirectory(diskann)
# ScaNN / SPANN land in later milestones; add when their directories gain
# implementations.

set(EXTENSION_SOURCES ${EXTENSION_SOURCES} PARENT_SCOPE)
