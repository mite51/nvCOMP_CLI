/**
 * @file nvcomp_c_api.h
 * @brief C API wrapper for nvcomp_core library
 * 
 * Provides a C-compatible interface for cross-language compatibility
 * and potential future bindings (Python, C#, etc.)
 */

#ifndef NVCOMP_C_API_H
#define NVCOMP_C_API_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Windows DLL export/import macros
#ifdef _WIN32
    #ifdef NVCOMP_CORE_EXPORTS
        #define NVCOMP_C_API __declspec(dllexport)
    #else
        #define NVCOMP_C_API __declspec(dllimport)
    #endif
#else
    #define NVCOMP_C_API
#endif

// ============================================================================
// Error Codes
// ============================================================================

typedef enum {
    NVCOMP_SUCCESS = 0,
    NVCOMP_ERROR_INVALID_ARGUMENT,
    NVCOMP_ERROR_FILE_NOT_FOUND,
    NVCOMP_ERROR_FILE_IO,
    NVCOMP_ERROR_INVALID_FORMAT,
    NVCOMP_ERROR_COMPRESSION_FAILED,
    NVCOMP_ERROR_DECOMPRESSION_FAILED,
    NVCOMP_ERROR_OUT_OF_MEMORY,
    NVCOMP_ERROR_CUDA_ERROR,
    NVCOMP_ERROR_UNSUPPORTED_ALGORITHM,
    NVCOMP_ERROR_UNKNOWN
} nvcomp_error_t;

// ============================================================================
// Algorithm Types
// ============================================================================

typedef enum {
    NVCOMP_ALGO_LZ4 = 0,
    NVCOMP_ALGO_SNAPPY,
    NVCOMP_ALGO_ZSTD,
    NVCOMP_ALGO_GDEFLATE,
    NVCOMP_ALGO_ANS,
    NVCOMP_ALGO_BITCOMP,
    NVCOMP_ALGO_UNKNOWN
} nvcomp_algorithm_t;

// ============================================================================
// Operation Handle (Opaque Type for Progress Callbacks)
// ============================================================================

typedef struct nvcomp_operation_t* nvcomp_operation_handle;

// ============================================================================
// Progress Callback
// ============================================================================

/**
 * @brief Progress callback function type
 * @param current Current progress value
 * @param total Total progress value
 * @param user_data User-provided context data
 */
typedef void (*nvcomp_progress_callback_t)(uint64_t current, uint64_t total, void* user_data);

// ============================================================================
// Error Handling Functions
// ============================================================================

/**
 * @brief Get the last error message for the current thread
 * @return Error message string (thread-local, do not free)
 */
NVCOMP_C_API const char* nvcomp_get_last_error(void);

/**
 * @brief Clear the last error for the current thread
 */
NVCOMP_C_API void nvcomp_clear_last_error(void);

// ============================================================================
// Algorithm Utility Functions
// ============================================================================

/**
 * @brief Parse algorithm string to enum
 * @param algo_str Algorithm name (e.g., "lz4", "snappy", "zstd")
 * @return Algorithm enum value
 */
NVCOMP_C_API nvcomp_algorithm_t nvcomp_parse_algorithm(const char* algo_str);

/**
 * @brief Convert algorithm enum to string
 * @param algo Algorithm enum value
 * @return Algorithm name string (do not free)
 */
NVCOMP_C_API const char* nvcomp_algorithm_to_string(nvcomp_algorithm_t algo);

/**
 * @brief Check if algorithm is cross-compatible (CPU/GPU)
 * @param algo Algorithm to check
 * @return true if cross-compatible, false otherwise
 */
NVCOMP_C_API bool nvcomp_is_cross_compatible(nvcomp_algorithm_t algo);

/**
 * @brief Check if CUDA is available
 * @return true if CUDA is available, false otherwise
 */
NVCOMP_C_API bool nvcomp_is_cuda_available(void);

// ============================================================================
// File I/O Utilities
// ============================================================================

/**
 * @brief Check if path is a directory
 * @param path Path to check
 * @return true if directory, false otherwise
 */
NVCOMP_C_API bool nvcomp_is_directory(const char* path);

/**
 * @brief Create directories recursively
 * @param path Directory path to create
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_create_directories(const char* path);

// ============================================================================
// Volume Support Functions
// ============================================================================

/**
 * @brief Check if filename indicates a volume file
 * @param filename Filename to check
 * @return true if volume file, false otherwise
 */
NVCOMP_C_API bool nvcomp_is_volume_file(const char* filename);

/**
 * @brief Parse volume size string (e.g., "2.5GB", "100MB")
 * @param size_str Size string to parse
 * @return Size in bytes, or 0 on error
 */
NVCOMP_C_API uint64_t nvcomp_parse_volume_size(const char* size_str);

/**
 * @brief Check if GPU memory is sufficient for volume size
 * @param volume_size Volume size in bytes
 * @return true if sufficient, false otherwise
 */
NVCOMP_C_API bool nvcomp_check_gpu_memory_for_volume(uint64_t volume_size);

// ============================================================================
// Operation Handle Functions (for Progress Callbacks)
// ============================================================================

/**
 * @brief Create an operation handle for tracking progress
 * @return Operation handle, or NULL on error
 */
NVCOMP_C_API nvcomp_operation_handle nvcomp_create_operation_handle(void);

/**
 * @brief Destroy an operation handle
 * @param handle Handle to destroy
 */
NVCOMP_C_API void nvcomp_destroy_operation_handle(nvcomp_operation_handle handle);

/**
 * @brief Set progress callback for an operation
 * @param handle Operation handle
 * @param callback Callback function
 * @param user_data User data to pass to callback
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_set_progress_callback(
    nvcomp_operation_handle handle,
    nvcomp_progress_callback_t callback,
    void* user_data
);

// ============================================================================
// Compression Functions
// ============================================================================

/**
 * @brief Compress file or folder using GPU (batched API)
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param input_path Input file or folder path
 * @param output_file Output compressed file path
 * @param max_volume_size Maximum volume size (0 for no splitting)
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_compress_gpu_batched(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_path,
    const char* output_file,
    uint64_t max_volume_size
);

/**
 * @brief Decompress file using GPU (batched API)
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param input_file Input compressed file path
 * @param output_path Output file or folder path
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_decompress_gpu_batched(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_file,
    const char* output_path
);

/**
 * @brief Compress file or folder using GPU (manager API)
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param input_path Input file or folder path
 * @param output_file Output compressed file path
 * @param max_volume_size Maximum volume size (0 for no splitting)
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_compress_gpu_manager(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_path,
    const char* output_file,
    uint64_t max_volume_size
);

/**
 * @brief Decompress file using GPU (manager API - auto-detects algorithm)
 * @param handle Operation handle (can be NULL)
 * @param input_file Input compressed file path
 * @param output_path Output file or folder path
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_decompress_gpu_manager(
    nvcomp_operation_handle handle,
    const char* input_file,
    const char* output_path
);

/**
 * @brief Compress file or folder using CPU
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param input_path Input file or folder path
 * @param output_file Output compressed file path
 * @param max_volume_size Maximum volume size (0 for no splitting)
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_compress_cpu(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_path,
    const char* output_file,
    uint64_t max_volume_size
);

/**
 * @brief Decompress file using CPU
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param input_file Input compressed file path
 * @param output_path Output file or folder path
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_decompress_cpu(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_file,
    const char* output_path
);

// ============================================================================
// Algorithm Detection
// ============================================================================

/**
 * @brief Detect compression algorithm from file
 * @param filename Compressed file path
 * @return Detected algorithm, or NVCOMP_ALGO_UNKNOWN on error
 */
NVCOMP_C_API nvcomp_algorithm_t nvcomp_detect_algorithm_from_file(const char* filename);

// ============================================================================
// Archive Listing
// ============================================================================

/**
 * @brief List contents of compressed archive
 * @param algo Algorithm used for compression
 * @param input_file Compressed archive path
 * @param use_cpu Use CPU for decompression
 * @param cuda_available Whether CUDA is available
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_list_compressed_archive(
    nvcomp_algorithm_t algo,
    const char* input_file,
    bool use_cpu,
    bool cuda_available
);

#ifdef __cplusplus
}
#endif

#endif // NVCOMP_C_API_H


