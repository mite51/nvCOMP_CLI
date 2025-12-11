/**
 * @file nvcomp_c_api.cpp
 * @brief C API wrapper implementation
 */

#include "nvcomp_c_api.h"
#include "nvcomp_core.hpp"

#include <string>
#include <cstring>
#include <mutex>
#include <stdexcept>

// ============================================================================
// Thread-local Error Storage
// ============================================================================

thread_local std::string g_last_error;

// ============================================================================
// Operation Handle Implementation
// ============================================================================

struct nvcomp_operation_t {
    nvcomp_progress_callback_t callback;
    void* user_data;
    std::mutex mutex;
    
    nvcomp_operation_t() : callback(nullptr), user_data(nullptr) {}
    
    void reportProgress(uint64_t current, uint64_t total) {
        std::lock_guard<std::mutex> lock(mutex);
        if (callback) {
            callback(current, total, user_data);
        }
    }
};

// ============================================================================
// Error Handling Helper
// ============================================================================

/**
 * @brief Helper template to execute C++ code and convert exceptions to error codes
 */
template<typename Func>
nvcomp_error_t executeSafely(Func&& func) {
    try {
        g_last_error.clear();
        func();
        return NVCOMP_SUCCESS;
    } catch (const std::invalid_argument& e) {
        g_last_error = std::string("Invalid argument: ") + e.what();
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        g_last_error = msg;
        
        // Categorize runtime errors
        if (msg.find("file") != std::string::npos || msg.find("File") != std::string::npos) {
            if (msg.find("not found") != std::string::npos || msg.find("does not exist") != std::string::npos) {
                return NVCOMP_ERROR_FILE_NOT_FOUND;
            }
            return NVCOMP_ERROR_FILE_IO;
        }
        if (msg.find("format") != std::string::npos || msg.find("invalid") != std::string::npos) {
            return NVCOMP_ERROR_INVALID_FORMAT;
        }
        if (msg.find("compress") != std::string::npos) {
            return NVCOMP_ERROR_COMPRESSION_FAILED;
        }
        if (msg.find("decompress") != std::string::npos) {
            return NVCOMP_ERROR_DECOMPRESSION_FAILED;
        }
        if (msg.find("memory") != std::string::npos || msg.find("allocation") != std::string::npos) {
            return NVCOMP_ERROR_OUT_OF_MEMORY;
        }
        if (msg.find("CUDA") != std::string::npos || msg.find("GPU") != std::string::npos) {
            return NVCOMP_ERROR_CUDA_ERROR;
        }
        if (msg.find("algorithm") != std::string::npos || msg.find("unsupported") != std::string::npos) {
            return NVCOMP_ERROR_UNSUPPORTED_ALGORITHM;
        }
        
        return NVCOMP_ERROR_UNKNOWN;
    } catch (const std::bad_alloc& e) {
        g_last_error = std::string("Out of memory: ") + e.what();
        return NVCOMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        g_last_error = std::string("Unknown error: ") + e.what();
        return NVCOMP_ERROR_UNKNOWN;
    } catch (...) {
        g_last_error = "Unknown error occurred";
        return NVCOMP_ERROR_UNKNOWN;
    }
}

// ============================================================================
// Algorithm Conversion Helpers
// ============================================================================

nvcomp_core::AlgoType toCorealgo(nvcomp_algorithm_t algo) {
    switch (algo) {
        case NVCOMP_ALGO_LZ4: return nvcomp_core::ALGO_LZ4;
        case NVCOMP_ALGO_SNAPPY: return nvcomp_core::ALGO_SNAPPY;
        case NVCOMP_ALGO_ZSTD: return nvcomp_core::ALGO_ZSTD;
        case NVCOMP_ALGO_GDEFLATE: return nvcomp_core::ALGO_GDEFLATE;
        case NVCOMP_ALGO_ANS: return nvcomp_core::ALGO_ANS;
        case NVCOMP_ALGO_BITCOMP: return nvcomp_core::ALGO_BITCOMP;
        default: return nvcomp_core::ALGO_UNKNOWN;
    }
}

nvcomp_algorithm_t fromCoreAlgo(nvcomp_core::AlgoType algo) {
    switch (algo) {
        case nvcomp_core::ALGO_LZ4: return NVCOMP_ALGO_LZ4;
        case nvcomp_core::ALGO_SNAPPY: return NVCOMP_ALGO_SNAPPY;
        case nvcomp_core::ALGO_ZSTD: return NVCOMP_ALGO_ZSTD;
        case nvcomp_core::ALGO_GDEFLATE: return NVCOMP_ALGO_GDEFLATE;
        case nvcomp_core::ALGO_ANS: return NVCOMP_ALGO_ANS;
        case nvcomp_core::ALGO_BITCOMP: return NVCOMP_ALGO_BITCOMP;
        default: return NVCOMP_ALGO_UNKNOWN;
    }
}

// ============================================================================
// Error Handling Functions
// ============================================================================

const char* nvcomp_get_last_error(void) {
    return g_last_error.c_str();
}

void nvcomp_clear_last_error(void) {
    g_last_error.clear();
}

// ============================================================================
// Algorithm Utility Functions
// ============================================================================

nvcomp_algorithm_t nvcomp_parse_algorithm(const char* algo_str) {
    if (!algo_str) return NVCOMP_ALGO_UNKNOWN;
    
    try {
        auto coreAlgo = nvcomp_core::parseAlgorithm(algo_str);
        return fromCoreAlgo(coreAlgo);
    } catch (...) {
        return NVCOMP_ALGO_UNKNOWN;
    }
}

const char* nvcomp_algorithm_to_string(nvcomp_algorithm_t algo) {
    thread_local std::string algo_str;
    try {
        algo_str = nvcomp_core::algoToString(toCorealgo(algo));
        return algo_str.c_str();
    } catch (...) {
        return "unknown";
    }
}

bool nvcomp_is_cross_compatible(nvcomp_algorithm_t algo) {
    try {
        return nvcomp_core::isCrossCompatible(toCorealgo(algo));
    } catch (...) {
        return false;
    }
}

bool nvcomp_is_cuda_available(void) {
    try {
        return nvcomp_core::isCudaAvailable();
    } catch (...) {
        return false;
    }
}

// ============================================================================
// File I/O Utilities
// ============================================================================

bool nvcomp_is_directory(const char* path) {
    if (!path) return false;
    
    try {
        return nvcomp_core::isDirectory(path);
    } catch (...) {
        return false;
    }
}

nvcomp_error_t nvcomp_create_directories(const char* path) {
    if (!path) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    
    return executeSafely([&]() {
        nvcomp_core::createDirectories(path);
    });
}

// ============================================================================
// Volume Support Functions
// ============================================================================

bool nvcomp_is_volume_file(const char* filename) {
    if (!filename) return false;
    
    try {
        return nvcomp_core::isVolumeFile(filename);
    } catch (...) {
        return false;
    }
}

uint64_t nvcomp_parse_volume_size(const char* size_str) {
    if (!size_str) return 0;
    
    try {
        return nvcomp_core::parseVolumeSize(size_str);
    } catch (...) {
        return 0;
    }
}

bool nvcomp_check_gpu_memory_for_volume(uint64_t volume_size) {
    try {
        return nvcomp_core::checkGPUMemoryForVolume(volume_size);
    } catch (...) {
        return false;
    }
}

// ============================================================================
// Operation Handle Functions
// ============================================================================

nvcomp_operation_handle nvcomp_create_operation_handle(void) {
    try {
        return new nvcomp_operation_t();
    } catch (...) {
        return nullptr;
    }
}

void nvcomp_destroy_operation_handle(nvcomp_operation_handle handle) {
    if (handle) {
        delete handle;
    }
}

nvcomp_error_t nvcomp_set_progress_callback(
    nvcomp_operation_handle handle,
    nvcomp_progress_callback_t callback,
    void* user_data
) {
    if (!handle) {
        g_last_error = "Null operation handle";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        std::lock_guard<std::mutex> lock(handle->mutex);
        handle->callback = callback;
        handle->user_data = user_data;
        return NVCOMP_SUCCESS;
    } catch (...) {
        g_last_error = "Failed to set progress callback";
        return NVCOMP_ERROR_UNKNOWN;
    }
}

// ============================================================================
// Compression Functions
// ============================================================================

nvcomp_error_t nvcomp_compress_gpu_batched(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_path,
    const char* output_file,
    uint64_t max_volume_size
) {
    if (!input_path || !output_file) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    
    auto result = executeSafely([&]() {
        nvcomp_core::compressGPUBatched(
            toCorealgo(algo),
            input_path,
            output_file,
            max_volume_size
        );
    });
    
    if (result == NVCOMP_SUCCESS && handle) {
        handle->reportProgress(100, 100);
    }
    
    return result;
}

nvcomp_error_t nvcomp_decompress_gpu_batched(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_file,
    const char* output_path
) {
    if (!input_file || !output_path) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    
    auto result = executeSafely([&]() {
        nvcomp_core::decompressGPUBatched(
            toCorealgo(algo),
            input_file,
            output_path
        );
    });
    
    if (result == NVCOMP_SUCCESS && handle) {
        handle->reportProgress(100, 100);
    }
    
    return result;
}

nvcomp_error_t nvcomp_compress_gpu_manager(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_path,
    const char* output_file,
    uint64_t max_volume_size
) {
    if (!input_path || !output_file) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    
    auto result = executeSafely([&]() {
        nvcomp_core::compressGPUManager(
            toCorealgo(algo),
            input_path,
            output_file,
            max_volume_size
        );
    });
    
    if (result == NVCOMP_SUCCESS && handle) {
        handle->reportProgress(100, 100);
    }
    
    return result;
}

nvcomp_error_t nvcomp_decompress_gpu_manager(
    nvcomp_operation_handle handle,
    const char* input_file,
    const char* output_path
) {
    if (!input_file || !output_path) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    
    auto result = executeSafely([&]() {
        nvcomp_core::decompressGPUManager(
            input_file,
            output_path
        );
    });
    
    if (result == NVCOMP_SUCCESS && handle) {
        handle->reportProgress(100, 100);
    }
    
    return result;
}

nvcomp_error_t nvcomp_compress_cpu(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_path,
    const char* output_file,
    uint64_t max_volume_size
) {
    if (!input_path || !output_file) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    
    auto result = executeSafely([&]() {
        nvcomp_core::compressCPU(
            toCorealgo(algo),
            input_path,
            output_file,
            max_volume_size
        );
    });
    
    if (result == NVCOMP_SUCCESS && handle) {
        handle->reportProgress(100, 100);
    }
    
    return result;
}

nvcomp_error_t nvcomp_decompress_cpu(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_file,
    const char* output_path
) {
    if (!input_file || !output_path) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    
    auto result = executeSafely([&]() {
        nvcomp_core::decompressCPU(
            toCorealgo(algo),
            input_file,
            output_path
        );
    });
    
    if (result == NVCOMP_SUCCESS && handle) {
        handle->reportProgress(100, 100);
    }
    
    return result;
}

// ============================================================================
// Algorithm Detection
// ============================================================================

nvcomp_algorithm_t nvcomp_detect_algorithm_from_file(const char* filename) {
    if (!filename) return NVCOMP_ALGO_UNKNOWN;
    
    try {
        auto coreAlgo = nvcomp_core::detectAlgorithmFromFile(filename);
        return fromCoreAlgo(coreAlgo);
    } catch (...) {
        return NVCOMP_ALGO_UNKNOWN;
    }
}

// ============================================================================
// Archive Listing
// ============================================================================

nvcomp_error_t nvcomp_list_compressed_archive(
    nvcomp_algorithm_t algo,
    const char* input_file,
    bool use_cpu,
    bool cuda_available
) {
    if (!input_file) {
        g_last_error = "Null input file provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    
    return executeSafely([&]() {
        nvcomp_core::listCompressedArchive(
            toCorealgo(algo),
            input_file,
            use_cpu,
            cuda_available
        );
    });
}


