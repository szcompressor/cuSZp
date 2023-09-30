# Specify the minimum version of CMake required to build the project
cmake_minimum_required(VERSION 3.21)

project(cuSZp
        VERSION 0.0.2
        DESCRIPTION "Error-bounded GPU lossy compression library"
        )
set(namespace "cuSZp")
enable_language(CXX)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -debug -Wall -diag-disable=10441")
#set(CMAKE_CXX_FLAGS_RELEASE "-diag-disable=10441 -g -ftz -fma -O2 -fp-model precise -prec-div -Wall")

#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -ftz=true -G -allow-unsupported-compiler")
#set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -allow-unsupported-compiler")

set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_FLAGS_INIT "-std=c++17 -allow-unsupported-compiler")
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 75)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_LIBRARY CUDA::cudart)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE Release)
endif()

add_library(${PROJECT_NAME} STATIC)

target_sources(${PROJECT_NAME}
        PRIVATE
        src/cuSZp_f32.cu
        src/cuSZp_f64.cu
        src/cuSZp_utility.cu
        src/cuSZp_timer.cu
        src/cuSZp_entry_f32.cu
        src/cuSZp_entry_f64.cu
        )

target_include_directories(${PROJECT_NAME}
        PRIVATE
        # where the library itself will look for its internal headers
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        PUBLIC
        # where top-level project will look for the library's public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # where external projects will look for the library's public headers
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

#target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

set(public_headers
        include/cuSZp_f32.h
        include/cuSZp_f64.h
        include/cuSZp_utility.h
        include/cuSZp_timer.h
        include/cuSZp_entry_f32.h
        include/cuSZp_entry_f64.h
        )

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Installing)

option(CUSZP_BUILD_EXAMPLES "Option to enable building example programs" ON)
if (CUSZP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()