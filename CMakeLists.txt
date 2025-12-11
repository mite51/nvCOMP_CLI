cmake_minimum_required(VERSION 3.18)
project(NvCompCLI LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# ============================================================================
# Fetch nvCOMP SDK
# ============================================================================

message(STATUS "Detecting platform for nvCOMP download...")

if(WIN32)
    set(NVCOMP_URL "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/windows-x86_64/nvcomp-windows-x86_64-5.1.0.21_cuda13-archive.zip")
    message(STATUS "Platform: Windows")
elseif(UNIX AND NOT APPLE)
    set(NVCOMP_URL "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda13-archive.tar.xz")
    message(STATUS "Platform: Linux")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

FetchContent_Declare(
    nvcomp_archive
    URL ${NVCOMP_URL}
)

FetchContent_GetProperties(nvcomp_archive)
if(NOT nvcomp_archive_POPULATED)
    message(STATUS "Downloading nvCOMP SDK...")
    FetchContent_Populate(nvcomp_archive)
endif()

# Find nvCOMP path
if(EXISTS "${nvcomp_archive_SOURCE_DIR}/include/nvcomp.hpp")
    set(NVCOMP_PATH "${nvcomp_archive_SOURCE_DIR}")
else()
    file(GLOB CHILDREN "${nvcomp_archive_SOURCE_DIR}/*")
    foreach(CHILD ${CHILDREN})
        if(IS_DIRECTORY "${CHILD}" AND EXISTS "${CHILD}/include/nvcomp.hpp")
            set(NVCOMP_PATH "${CHILD}")
            break()
        endif()
    endforeach()
endif()

if(NOT DEFINED NVCOMP_PATH)
    message(FATAL_ERROR "Could not find nvCOMP in ${nvcomp_archive_SOURCE_DIR}")
endif()

message(STATUS "nvCOMP path: ${NVCOMP_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${NVCOMP_PATH}/lib/cmake/nvcomp")

# ============================================================================
# Fetch and Patch LZ4
# ============================================================================

message(STATUS "Fetching LZ4...")

FetchContent_Declare(
    lz4
    URL https://github.com/lz4/lz4/archive/refs/tags/v1.9.4.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(lz4)
if(NOT lz4_POPULATED)
    FetchContent_Populate(lz4)
    
    # Patch CMakeLists.txt to use compatible CMake version
    set(LZ4_CMAKE_FILE "${lz4_SOURCE_DIR}/build/cmake/CMakeLists.txt")
    if(EXISTS ${LZ4_CMAKE_FILE})
        file(READ ${LZ4_CMAKE_FILE} LZ4_CMAKE_CONTENT)
        string(REGEX REPLACE "cmake_minimum_required\\(VERSION [^)]+\\)" 
               "cmake_minimum_required(VERSION 3.5)" 
               LZ4_CMAKE_CONTENT "${LZ4_CMAKE_CONTENT}")
        file(WRITE ${LZ4_CMAKE_FILE} "${LZ4_CMAKE_CONTENT}")
        message(STATUS "Patched LZ4 CMakeLists.txt")
    endif()
    
    # Build LZ4
    set(LZ4_BUILD_CLI OFF CACHE BOOL "" FORCE)
    set(LZ4_BUILD_LEGACY_LZ4C OFF CACHE BOOL "" FORCE)
    add_subdirectory(${lz4_SOURCE_DIR}/build/cmake ${lz4_BINARY_DIR})
endif()

# ============================================================================
# Fetch and Patch Snappy
# ============================================================================

message(STATUS "Fetching Snappy...")

FetchContent_Declare(
    snappy
    URL https://github.com/google/snappy/archive/refs/tags/1.2.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(snappy)
if(NOT snappy_POPULATED)
    FetchContent_Populate(snappy)
    
    # Patch CMakeLists.txt
    set(SNAPPY_CMAKE_FILE "${snappy_SOURCE_DIR}/CMakeLists.txt")
    if(EXISTS ${SNAPPY_CMAKE_FILE})
        file(READ ${SNAPPY_CMAKE_FILE} SNAPPY_CMAKE_CONTENT)
        string(REGEX REPLACE "cmake_minimum_required\\(VERSION [^)]+\\)" 
               "cmake_minimum_required(VERSION 3.5)" 
               SNAPPY_CMAKE_CONTENT "${SNAPPY_CMAKE_CONTENT}")
        file(WRITE ${SNAPPY_CMAKE_FILE} "${SNAPPY_CMAKE_CONTENT}")
        message(STATUS "Patched Snappy CMakeLists.txt")
    endif()
    
    # Build Snappy
    set(SNAPPY_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SNAPPY_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${snappy_SOURCE_DIR} ${snappy_BINARY_DIR})
endif()

# ============================================================================
# Fetch and Patch Zstd
# ============================================================================

message(STATUS "Fetching Zstd...")

FetchContent_Declare(
    zstd
    URL https://github.com/facebook/zstd/archive/refs/tags/v1.5.5.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(zstd)
if(NOT zstd_POPULATED)
    FetchContent_Populate(zstd)
    
    # Patch CMakeLists.txt
    set(ZSTD_CMAKE_FILE "${zstd_SOURCE_DIR}/build/cmake/CMakeLists.txt")
    if(EXISTS ${ZSTD_CMAKE_FILE})
        file(READ ${ZSTD_CMAKE_FILE} ZSTD_CMAKE_CONTENT)
        string(REGEX REPLACE "cmake_minimum_required\\(VERSION [^)]+\\)" 
               "cmake_minimum_required(VERSION 3.5)" 
               ZSTD_CMAKE_CONTENT "${ZSTD_CMAKE_CONTENT}")
        file(WRITE ${ZSTD_CMAKE_FILE} "${ZSTD_CMAKE_CONTENT}")
        message(STATUS "Patched Zstd CMakeLists.txt")
    endif()
    
    # Build Zstd
    set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
    add_subdirectory(${zstd_SOURCE_DIR}/build/cmake ${zstd_BINARY_DIR})
endif()

# ============================================================================
# Find nvCOMP package
# ============================================================================

find_package(nvcomp REQUIRED)

# ============================================================================
# Build Core Library
# ============================================================================

# Pass dependency include directories to core library via parent scope includes
# This allows core to find lz4, snappy, zstd headers
include_directories(
    ${lz4_SOURCE_DIR}/lib
    ${snappy_SOURCE_DIR}
    ${snappy_BINARY_DIR}
    ${zstd_SOURCE_DIR}/lib
)

# Build core library subdirectory
add_subdirectory(core)

# ============================================================================
# Create CLI executable
# ============================================================================

add_executable(nvcomp_cli main.cu)

# Set CUDA architectures
set_property(TARGET nvcomp_cli PROPERTY CUDA_ARCHITECTURES 75 80 86 89 90)

# Include directories
target_include_directories(nvcomp_cli PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    ${lz4_SOURCE_DIR}/lib
    ${snappy_SOURCE_DIR}
    ${snappy_BINARY_DIR}
    ${zstd_SOURCE_DIR}/lib
)

# Link libraries
# NOTE: main.cu still contains old code (not refactored yet - that's Task 1.3)
# So it needs direct access to compression libraries for now
target_link_libraries(nvcomp_cli PRIVATE 
    nvcomp_core
    nvcomp::nvcomp
    lz4_static
    snappy
    libzstd_static
)

# Windows-specific: Copy DLLs to output directory
if(WIN32)
    add_custom_command(TARGET nvcomp_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NVCOMP_PATH}/bin/nvcomp64_5.dll"
        $<TARGET_FILE_DIR:nvcomp_cli>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NVCOMP_PATH}/bin/nvcomp_cpu64_5.dll"
        $<TARGET_FILE_DIR:nvcomp_cli>
        COMMENT "Copying nvCOMP DLLs to output directory"
    )
endif()

# ============================================================================
# Create Test Executable for C API
# ============================================================================

add_executable(test_c_api unit_test/test_c_api.cpp)

# Include directories
target_include_directories(test_c_api PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)

# Link libraries
target_link_libraries(test_c_api PRIVATE 
    nvcomp_core
)

# Windows-specific: Copy DLLs to test output directory
if(WIN32)
    add_custom_command(TARGET test_c_api POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:nvcomp_core>
        $<TARGET_FILE_DIR:test_c_api>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NVCOMP_PATH}/bin/nvcomp64_5.dll"
        $<TARGET_FILE_DIR:test_c_api>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NVCOMP_PATH}/bin/nvcomp_cpu64_5.dll"
        $<TARGET_FILE_DIR:test_c_api>
        COMMENT "Copying DLLs to test output directory"
    )
endif()

# ============================================================================
# Installation (optional)
# ============================================================================

install(TARGETS nvcomp_cli DESTINATION bin)

message(STATUS "")
message(STATUS "========================================")
message(STATUS "nvCOMP CLI Configuration Complete")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "nvCOMP: ${NVCOMP_PATH}")
message(STATUS "LZ4: ${lz4_SOURCE_DIR}")
message(STATUS "Snappy: ${snappy_SOURCE_DIR}")
message(STATUS "Zstd: ${zstd_SOURCE_DIR}")
message(STATUS "========================================")
message(STATUS "")
