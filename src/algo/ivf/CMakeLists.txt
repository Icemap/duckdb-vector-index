# M2: IVF — posting-list / inverted-file ANN.
set(EXTENSION_SOURCES
    ${EXTENSION_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/ivf_register.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ivf_index.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ivf_build.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ivf_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ivf_pragmas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ivf_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kmeans.cpp
    PARENT_SCOPE)
