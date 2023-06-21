# Find CUDA package
find_package(CUDA REQUIRED)

set(install_dir ${PROJECT_BINARY_DIR}/examples/bin)
set(execName_gpu "cuSZpExample_gpu_api")
set(execName_cpu "cuSZpExample_cpu_api")
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# Add include and library directories
include_directories(${INCLUDE_DIR})

# Compile headers as a library
cuda_add_library(cuSZp_example STATIC ${SRC_DIR}/cuSZp.cu
                                      ${SRC_DIR}/cuSZp_utility.cu
                                      ${SRC_DIR}/cuSZp_timer.cu
                                      ${SRC_DIR}/cuSZp_entry.cu)

# Compile executable binary
cuda_add_executable(${execName_gpu} example_gpu_api.cpp)
cuda_add_executable(${execName_cpu} example_cpu_api.cpp)

# Link with headers
target_link_libraries(${execName_gpu} cuSZp_example)
target_link_libraries(${execName_cpu} cuSZp_example)

# Set output paths for the compiled binary
set_target_properties(${execName_gpu} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${execName_cpu} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})

# Set installation paths for the compiled binary.
install(TARGETS ${execName_gpu} DESTINATION bin)
install(TARGETS ${execName_cpu} DESTINATION bin)