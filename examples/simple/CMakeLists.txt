set(install_dir ${PROJECT_BINARY_DIR}/examples/bin)
set(execName "cuSZpExample")

add_executable(${execName}
        main.cpp
        )

target_link_libraries(${execName}
        PRIVATE
        cuSZp
        CUDA::cudart
        )

target_include_directories(${execName}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )

set_target_properties(${execName} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})

install(TARGETS ${execName} DESTINATION bin)





