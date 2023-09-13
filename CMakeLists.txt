# Specify the minimum version of CMake required to build the project
cmake_minimum_required(VERSION 3.21)
project(cuSZp
        VERSION 0.0.1
        DESCRIPTION "Error-bounded GPU lossy compression library"
        LANGUAGES CXX CUDA)

option(BUILD_SHARED_LIBS "prefer building shared libraries" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE Release)
endif()

add_library(cuSZp)
target_compile_features(cuSZp PUBLIC
    cxx_std_17
    cuda_std_17
    )
set_target_properties(cuSZp PROPERTIES
    CXX_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    )

target_sources(cuSZp
        PRIVATE
        src/cuSZp.cu
        src/cuSZp_utility.cu
        src/cuSZp_timer.cu
        src/cuSZp_entry.cu
        )

target_include_directories(cuSZp
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
target_link_libraries(cuSZp PUBLIC CUDA::cudart)

set(public_headers
        include/cuSZp.h
        include/cuSZp_utility.h
        include/cuSZp_timer.h
        include/cuSZp_entry.h
        )


set_target_properties(cuSZp PROPERTIES PUBLIC_HEADER "${public_headers}")
set_target_properties(cuSZp PROPERTIES DEBUG_POSTFIX "d")
install(TARGETS cuSZp
        EXPORT "cuSZpTargets"
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuSZp
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

# generate and install export file
install(EXPORT "cuSZpTargets"
        FILE "cuSZpTargets.cmake"
        NAMESPACE cuSZp::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuSZp
        )


write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/cuSZpConfigVersion.cmake"
        VERSION "${version}"
        COMPATIBILITY AnyNewerVersion)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/cuSZpConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuSZp/)
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/cuSZpConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/cuSZpConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuSZp/)
export(EXPORT "cuSZpTargets"
        FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/cuSZpTargets.cmake"
        NAMESPACE cuSZp::)

option(CUSZP_BUILD_EXAMPLES "Option to enable building example programs" ON)
if (CUSZP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()
