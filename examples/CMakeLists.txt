# Find CUDA package
find_package(CUDA REQUIRED)

set(install_dir ${PROJECT_BINARY_DIR}/examples/bin)
set(execName_gpu_f32 "cuSZp_gpu_f32_api")
set(execName_cpu_f32 "cuSZp_cpu_f32_api")
set(execName_gpu_f64 "cuSZp_gpu_f64_api")
set(execName_cpu_f64 "cuSZp_cpu_f64_api")
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# Add include and library directories
include_directories(${INCLUDE_DIR})

# Compile headers as a library
cuda_add_library(cuSZp_libs STATIC ${SRC_DIR}/cuSZp_f32.cu
                                   ${SRC_DIR}/cuSZp_f64.cu 
                                   ${SRC_DIR}/cuSZp_utility.cu
                                   ${SRC_DIR}/cuSZp_timer.cu
                                   ${SRC_DIR}/cuSZp_entry_f32.cu
                                   ${SRC_DIR}/cuSZp_entry_f64.cu)

# Compile executable binary
cuda_add_executable(${execName_gpu_f32} cuSZp_gpu_f32_api.cpp)
cuda_add_executable(${execName_cpu_f32} cuSZp_cpu_f32_api.cpp)
cuda_add_executable(${execName_gpu_f64} cuSZp_gpu_f64_api.cpp)
cuda_add_executable(${execName_cpu_f64} cuSZp_cpu_f64_api.cpp)

# Link with headers
target_link_libraries(${execName_gpu_f32} cuSZp_libs)
target_link_libraries(${execName_cpu_f32} cuSZp_libs)
target_link_libraries(${execName_gpu_f64} cuSZp_libs)
target_link_libraries(${execName_cpu_f64} cuSZp_libs)

# Set output paths for the compiled binary
set_target_properties(${execName_gpu_f32} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${execName_cpu_f32} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${execName_gpu_f64} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${execName_cpu_f64} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})

# Set installation paths for the compiled binary.
install(TARGETS ${execName_gpu_f32} DESTINATION bin)
install(TARGETS ${execName_cpu_f32} DESTINATION bin)
install(TARGETS ${execName_gpu_f64} DESTINATION bin)
install(TARGETS ${execName_cpu_f64} DESTINATION bin)