# M4: SPANN — IVF + closure replicas for boundary recall.
set(EXTENSION_SOURCES
    ${EXTENSION_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/spann_register.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/spann_index.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/spann_build.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/spann_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/spann_pragmas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/spann_core.cpp
    PARENT_SCOPE)
