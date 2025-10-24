# Specify the minimum version of CMake required to build the project
cmake_minimum_required(VERSION 3.21)

# Project configuration
project(cuSZp VERSION 3.0.0 DESCRIPTION "Error-bounded GPU lossy compression library")

# Enable languages
enable_language(CXX CUDA)

# Find required CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA host compiler
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 75 80 86)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE Release)
endif()

# Disable specific warnings (adjust as needed)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-unused-result")

# List of source files
set(cuSZp_SOURCES
    src/cuSZp_kernels_1D_f32.cu
    src/cuSZp_kernels_1D_f64.cu
    src/cuSZp_kernels_2D_f32.cu
    src/cuSZp_kernels_2D_f64.cu
    src/cuSZp_kernels_3D_f32.cu
    src/cuSZp_kernels_3D_f64.cu
    src/cuSZp_utility.cu
    src/cuSZp_timer.cu
    src/cuSZp_entry_1D_f32.cu
    src/cuSZp_entry_1D_f64.cu
    src/cuSZp_entry_2D_f32.cu
    src/cuSZp_entry_2D_f64.cu
    src/cuSZp_entry_3D_f32.cu
    src/cuSZp_entry_3D_f64.cu
    src/cuSZp.cu
)

# Create static and shared libraries
add_library(cuSZp_static STATIC ${cuSZp_SOURCES})
add_library(cuSZp_shared SHARED ${cuSZp_SOURCES})

# Include directories for both libraries
target_include_directories(cuSZp_static
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/cuSZp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(cuSZp_shared
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/cuSZp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link CUDA runtime
target_link_libraries(cuSZp_static PRIVATE CUDA::cudart)
target_link_libraries(cuSZp_shared PRIVATE CUDA::cudart)

# Set output name for libraries
set_target_properties(cuSZp_static PROPERTIES OUTPUT_NAME cuSZp)
set_target_properties(cuSZp_shared PROPERTIES OUTPUT_NAME cuSZp)

# Headers for installation
set(public_headers
    include/cuSZp.h
)

# Optionally build examples
option(cuSZp_BUILD_EXAMPLES "Option to enable building example programs" ON)
if (cuSZp_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

# Include the Installing.cmake script
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Installing)

# Enable CTest
enable_testing()

# Add test for installed cuSZp_test binary
add_test(NAME cuSZpTest_f32_1D COMMAND ${CMAKE_INSTALL_PREFIX}/bin/cuSZp_test_f32_1D)
add_test(NAME cuSZpTest_f32_2D COMMAND ${CMAKE_INSTALL_PREFIX}/bin/cuSZp_test_f32_2D)
add_test(NAME cuSZpTest_f32_3D COMMAND ${CMAKE_INSTALL_PREFIX}/bin/cuSZp_test_f32_3D)
add_test(NAME cuSZpTest_f64_1D COMMAND ${CMAKE_INSTALL_PREFIX}/bin/cuSZp_test_f64_1D)
add_test(NAME cuSZpTest_f64_2D COMMAND ${CMAKE_INSTALL_PREFIX}/bin/cuSZp_test_f64_2D)
add_test(NAME cuSZpTest_f64_3D COMMAND ${CMAKE_INSTALL_PREFIX}/bin/cuSZp_test_f64_3D)

# Optional: Set working directory if required
set_tests_properties(cuSZpTest_f32_1D PROPERTIES WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)
set_tests_properties(cuSZpTest_f32_2D PROPERTIES WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)
set_tests_properties(cuSZpTest_f32_3D PROPERTIES WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)
set_tests_properties(cuSZpTest_f64_1D PROPERTIES WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)
set_tests_properties(cuSZpTest_f64_2D PROPERTIES WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)
set_tests_properties(cuSZpTest_f64_3D PROPERTIES WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)