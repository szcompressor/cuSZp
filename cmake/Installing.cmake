include(GNUInstallDirs)

# Set default install prefix
if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "CMAKE_INSTALL_PREFIX is not set, defaulting to ${CMAKE_SOURCE_DIR}/install")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Install path" FORCE)
else()
    message(STATUS "CMAKE_INSTALL_PREFIX set to ${CMAKE_INSTALL_PREFIX}")
endif()

# Install cuSZp.h in the include directory
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/cuSZp.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# **Gather all .h files from include/cuSZp/** and install them
file(GLOB cuSZp_headers ${CMAKE_CURRENT_SOURCE_DIR}/include/cuSZp/*.h)

# Install the cuSZp headers into include/cuSZp directory
install(FILES ${cuSZp_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuSZp
)

# Install shared library and headers
install(TARGETS cuSZp_shared
        EXPORT cuSZpTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}    # for executables
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}    # for shared libraries
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}  # for public headers (cuSZp.h)
)

# Install static library without headers to avoid duplication
install(TARGETS cuSZp_static
        EXPORT cuSZpTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}    # for static libraries
)

# Generate and install export file only once
install(EXPORT cuSZpTargets
        FILE cuSZpTargets.cmake
        NAMESPACE cuSZp::
        DESTINATION cmake
)

include(CMakePackageConfigHelpers)

# Generate and install version and config files
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/cuSZpConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/cuSZpConfig.cmake"
        INSTALL_DESTINATION cmake
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/cuSZpConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/cuSZpConfigVersion.cmake"
        DESTINATION cmake
)

# Export the build tree
export(EXPORT cuSZpTargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/cuSZpTargets.cmake"
        NAMESPACE cuSZp::
)
