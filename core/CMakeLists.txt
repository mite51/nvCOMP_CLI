cmake_minimum_required(VERSION 3.18)
project(nvcomp_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find nvCOMP package (should be configured from parent)
find_package(nvcomp REQUIRED)

# Core library sources
set(CORE_SOURCES
    src/archive.cpp
    src/volume.cpp
    src/nvcomp_cpu.cpp
    src/nvcomp_core.cu
    src/nvcomp_c_api.cpp
)

# Create shared library
add_library(nvcomp_core SHARED ${CORE_SOURCES})

# Set CUDA architectures
set_property(TARGET nvcomp_core PROPERTY CUDA_ARCHITECTURES 75 80 86 89 90)

# Define export macro
target_compile_definitions(nvcomp_core PRIVATE NVCOMP_CORE_EXPORTS)

# Include directories
target_include_directories(nvcomp_core 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link dependencies (provided by parent CMakeLists)
target_link_libraries(nvcomp_core 
    PUBLIC
        nvcomp::nvcomp
    PRIVATE
        lz4_static
        snappy
        libzstd_static
)

# Installation rules
install(TARGETS nvcomp_core
    EXPORT nvcomp_coreTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

# Export targets
install(EXPORT nvcomp_coreTargets
    FILE nvcomp_coreTargets.cmake
    NAMESPACE nvcomp_core::
    DESTINATION lib/cmake/nvcomp_core
)

