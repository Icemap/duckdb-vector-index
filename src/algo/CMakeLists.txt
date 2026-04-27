add_subdirectory(hnsw)
add_subdirectory(ivf)
# DiskANN / ScaNN / SPANN land in later milestones; add when their directories
# gain implementations.

set(EXTENSION_SOURCES ${EXTENSION_SOURCES} PARENT_SCOPE)
