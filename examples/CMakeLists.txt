# Compile executable binary
add_executable(cuSZpExample_gpu_api example_gpu_api.cpp)
add_executable(cuSZpExample_cpu_api example_cpu_api.cpp)

# Link with headers
target_link_libraries(cuSZpExample_gpu_api PRIVATE cuSZp)
target_link_libraries(cuSZpExample_cpu_api PRIVATE cuSZp)

# Set installation paths for the compiled binary.
install(TARGETS cuSZpExample_gpu_api DESTINATION bin)
install(TARGETS cuSZpExample_cpu_api DESTINATION bin)
