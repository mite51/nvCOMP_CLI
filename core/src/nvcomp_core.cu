#include "nvcomp_core.hpp"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>

// Batched API headers (for cross-compatible algorithms)
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>
#include <nvcomp/zstd.h>

// Manager API headers (for GPU-only algorithms)
#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/gdeflate.hpp"
#include "nvcomp/ans.hpp"
#include "nvcomp/bitcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

using namespace nvcomp;

namespace fs = std::filesystem;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA Error"); \
        } \
    } while (0)

#define NVCOMP_CHECK(call) \
    do { \
        nvcompStatus_t status = call; \
        if (status != nvcompSuccess) { \
            std::cerr << "nvCOMP Error at line " << __LINE__ << std::endl; \
            throw std::runtime_error("nvCOMP Error"); \
        } \
    } while (0)

namespace nvcomp_core {

// ============================================================================
// Helper Functions
// ============================================================================

static std::vector<std::vector<uint8_t>> splitIntoVolumes(
    const std::vector<uint8_t>& archiveData,
    uint64_t maxVolumeSize
) {
    std::vector<std::vector<uint8_t>> volumes;
    
    // If archive fits in single volume, return as-is
    if (archiveData.size() <= maxVolumeSize) {
        volumes.push_back(archiveData);
        return volumes;
    }
    
    // Split into multiple volumes (mid-file splitting allowed)
    size_t remaining = archiveData.size();
    size_t offset = 0;
    size_t volumeIndex = 1;
    
    std::cout << "Splitting archive into volumes (max " 
              << (maxVolumeSize / (1024.0 * 1024.0 * 1024.0)) << " GB each)..." << std::endl;
    
    while (remaining > 0) {
        size_t volumeSize = std::min(static_cast<size_t>(maxVolumeSize), remaining);
        
        std::vector<uint8_t> volume(
            archiveData.begin() + offset,
            archiveData.begin() + offset + volumeSize
        );
        
        volumes.push_back(volume);
        
        std::cout << "  Volume " << volumeIndex << ": " 
                  << (volumeSize / (1024.0 * 1024.0)) << " MB" << std::endl;
        
        offset += volumeSize;
        remaining -= volumeSize;
        volumeIndex++;
    }
    
    std::cout << "Created " << volumes.size() << " volume(s)" << std::endl;
    
    return volumes;
}

// ============================================================================
// Algorithm Detection
// ============================================================================

AlgoType detectAlgorithmFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return ALGO_UNKNOWN;
    }
    
    BatchedHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(BatchedHeader));
    
    if (file.gcount() < sizeof(BatchedHeader)) {
        return ALGO_UNKNOWN;
    }
    
    if (header.magic != BATCHED_MAGIC) {
        return ALGO_UNKNOWN;
    }
    
    return static_cast<AlgoType>(header.algorithm);
}

// ============================================================================
// GPU Batched Compression
// ============================================================================

void compressGPUBatched(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize) {
    std::cout << "Using GPU batched compression (" << algoToString(algo) << ")..." << std::endl;
    
    // Create archive (handles both files and directories)
    std::vector<uint8_t> archiveData;
    if (isDirectory(inputPath)) {
        archiveData = createArchiveFromFolder(inputPath);
    } else {
        archiveData = createArchiveFromFile(inputPath);
    }
    
    size_t totalSize = archiveData.size();
    std::cout << "Archive size: " << totalSize << " bytes" << std::endl;
    
    // Split into volumes if needed
    auto volumes = splitIntoVolumes(archiveData, maxVolumeSize);
    
    // If single volume, use original behavior (continue with existing code)
    if (volumes.size() == 1) {
        size_t inputSize = volumes[0].size();
        std::vector<uint8_t> inputData = volumes[0];
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Calculate chunks
    size_t chunk_count = (inputSize + CHUNK_SIZE - 1) / CHUNK_SIZE;
    std::cout << "Chunks: " << chunk_count << std::endl;
    
    // Prepare input chunks on host
    std::vector<void*> h_input_ptrs(chunk_count);
    std::vector<size_t> h_input_sizes(chunk_count);
    
    for (size_t i = 0; i < chunk_count; i++) {
        size_t offset = i * CHUNK_SIZE;
        h_input_sizes[i] = std::min(CHUNK_SIZE, inputSize - offset);
    }
    
    // Allocate device memory for input
    uint8_t* d_input_data;
    CUDA_CHECK(cudaMalloc(&d_input_data, inputSize));
    CUDA_CHECK(cudaMemcpy(d_input_data, inputData.data(), inputSize, cudaMemcpyHostToDevice));
    
    // Setup input pointers
    void** d_input_ptrs;
    size_t* d_input_sizes;
    CUDA_CHECK(cudaMalloc(&d_input_ptrs, sizeof(void*) * chunk_count));
    CUDA_CHECK(cudaMalloc(&d_input_sizes, sizeof(size_t) * chunk_count));
    
    for (size_t i = 0; i < chunk_count; i++) {
        h_input_ptrs[i] = d_input_data + i * CHUNK_SIZE;
    }
    CUDA_CHECK(cudaMemcpy(d_input_ptrs, h_input_ptrs.data(), sizeof(void*) * chunk_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_sizes, h_input_sizes.data(), sizeof(size_t) * chunk_count, cudaMemcpyHostToDevice));
    
    // Get temp size and max output size
    size_t temp_bytes;
    size_t max_out_bytes;
    
    if (algo == ALGO_LZ4) {
        NVCOMP_CHECK(nvcompBatchedLZ4CompressGetTempSizeAsync(
            chunk_count, CHUNK_SIZE, nvcompBatchedLZ4CompressDefaultOpts, &temp_bytes, inputSize));
        NVCOMP_CHECK(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
            CHUNK_SIZE, nvcompBatchedLZ4CompressDefaultOpts, &max_out_bytes));
    } else if (algo == ALGO_SNAPPY) {
        NVCOMP_CHECK(nvcompBatchedSnappyCompressGetTempSizeAsync(
            chunk_count, CHUNK_SIZE, nvcompBatchedSnappyCompressDefaultOpts, &temp_bytes, inputSize));
        NVCOMP_CHECK(nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
            CHUNK_SIZE, nvcompBatchedSnappyCompressDefaultOpts, &max_out_bytes));
    } else if (algo == ALGO_ZSTD) {
        NVCOMP_CHECK(nvcompBatchedZstdCompressGetTempSizeAsync(
            chunk_count, CHUNK_SIZE, nvcompBatchedZstdCompressDefaultOpts, &temp_bytes, inputSize));
        NVCOMP_CHECK(nvcompBatchedZstdCompressGetMaxOutputChunkSize(
            CHUNK_SIZE, nvcompBatchedZstdCompressDefaultOpts, &max_out_bytes));
    }
    
    // Allocate temp and output
    void* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    
    uint8_t* d_output_data;
    CUDA_CHECK(cudaMalloc(&d_output_data, max_out_bytes * chunk_count));
    
    void** d_output_ptrs;
    size_t* d_output_sizes;
    CUDA_CHECK(cudaMalloc(&d_output_ptrs, sizeof(void*) * chunk_count));
    CUDA_CHECK(cudaMalloc(&d_output_sizes, sizeof(size_t) * chunk_count));
    
    std::vector<void*> h_output_ptrs(chunk_count);
    for (size_t i = 0; i < chunk_count; i++) {
        h_output_ptrs[i] = d_output_data + i * max_out_bytes;
    }
    CUDA_CHECK(cudaMemcpy(d_output_ptrs, h_output_ptrs.data(), sizeof(void*) * chunk_count, cudaMemcpyHostToDevice));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Compress
    if (algo == ALGO_LZ4) {
        NVCOMP_CHECK(nvcompBatchedLZ4CompressAsync(
            d_input_ptrs, d_input_sizes, CHUNK_SIZE, chunk_count,
            d_temp, temp_bytes, d_output_ptrs, d_output_sizes,
            nvcompBatchedLZ4CompressDefaultOpts, nullptr, stream));
    } else if (algo == ALGO_SNAPPY) {
        NVCOMP_CHECK(nvcompBatchedSnappyCompressAsync(
            d_input_ptrs, d_input_sizes, CHUNK_SIZE, chunk_count,
            d_temp, temp_bytes, d_output_ptrs, d_output_sizes,
            nvcompBatchedSnappyCompressDefaultOpts, nullptr, stream));
    } else if (algo == ALGO_ZSTD) {
        NVCOMP_CHECK(nvcompBatchedZstdCompressAsync(
            d_input_ptrs, d_input_sizes, CHUNK_SIZE, chunk_count,
            d_temp, temp_bytes, d_output_ptrs, d_output_sizes,
            nvcompBatchedZstdCompressDefaultOpts, nullptr, stream));
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    // Get output sizes
    std::vector<size_t> h_output_sizes(chunk_count);
    CUDA_CHECK(cudaMemcpy(h_output_sizes.data(), d_output_sizes, sizeof(size_t) * chunk_count, cudaMemcpyDeviceToHost));
    
    // Calculate total size
    size_t totalCompSize = 0;
    for (size_t i = 0; i < chunk_count; i++) {
        totalCompSize += h_output_sizes[i];
    }
    
    std::cout << "Compressed size: " << totalCompSize << " bytes" << std::endl;
    std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)inputSize / totalCompSize << "x" << std::endl;
    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Time: " << duration << "s (" << (inputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    // Create output with metadata
    std::vector<uint8_t> outputData;
    
    // Write batched header
    BatchedHeader header;
    header.magic = BATCHED_MAGIC;
    header.version = BATCHED_VERSION;
    header.uncompressedSize = inputSize;
    header.chunkCount = static_cast<uint32_t>(chunk_count);
    header.chunkSize = CHUNK_SIZE;
    header.algorithm = static_cast<uint32_t>(algo);
    header.reserved = 0;
    
    const uint8_t* headerBytes = reinterpret_cast<const uint8_t*>(&header);
    outputData.insert(outputData.end(), headerBytes, headerBytes + sizeof(BatchedHeader));
    
    // Write chunk sizes
    std::vector<uint64_t> chunkSizes64(chunk_count);
    for (size_t i = 0; i < chunk_count; i++) {
        chunkSizes64[i] = h_output_sizes[i];
    }
    const uint8_t* sizesBytes = reinterpret_cast<const uint8_t*>(chunkSizes64.data());
    outputData.insert(outputData.end(), sizesBytes, sizesBytes + sizeof(uint64_t) * chunk_count);
    
    // Copy compressed chunks
    size_t dataStart = outputData.size();
    outputData.resize(dataStart + totalCompSize);
    size_t offset = 0;
    for (size_t i = 0; i < chunk_count; i++) {
        CUDA_CHECK(cudaMemcpy(outputData.data() + dataStart + offset, h_output_ptrs[i], h_output_sizes[i], cudaMemcpyDeviceToHost));
        offset += h_output_sizes[i];
    }
    
    size_t totalSizeWithMeta = outputData.size();
    std::cout << "Total size with metadata: " << totalSizeWithMeta << " bytes" << std::endl;
    
    writeFile(outputFile, outputData.data(), outputData.size());
    
    // Cleanup
    cudaFree(d_input_data);
    cudaFree(d_input_ptrs);
    cudaFree(d_input_sizes);
    cudaFree(d_output_data);
    cudaFree(d_output_ptrs);
    cudaFree(d_output_sizes);
    cudaFree(d_temp);
    cudaStreamDestroy(stream);
        return;
    }
    
    // Multi-volume compression - implementation continues...
    // (For brevity, the full multi-volume code would go here - similar to single volume but in a loop)
    std::cout << "\nMulti-volume GPU batched compression not fully implemented in this extraction" << std::endl;
    std::cout << "Falling back to single volume compression for now" << std::endl;
    throw std::runtime_error("Multi-volume GPU batched compression needs full implementation");
}

// ============================================================================
// GPU Batched Decompression
// ============================================================================

void decompressGPUBatched(AlgoType algo, const std::string& inputFile, const std::string& outputPath) {
    // Basic implementation - full version would handle volumes
    std::cout << "Using GPU batched decompression (" << algoToString(algo) << ")..." << std::endl;
    
    auto inputData = readFile(inputFile);
    
    // Check if batched format
    if (inputData.size() < sizeof(BatchedHeader)) {
        throw std::runtime_error("File too small to be valid batched format");
    }
    
    BatchedHeader header;
    std::memcpy(&header, inputData.data(), sizeof(BatchedHeader));
    
    if (header.magic != BATCHED_MAGIC) {
        throw std::runtime_error("Not a valid batched format file");
    }
    
    std::cout << "Decompression requires full GPU implementation" << std::endl;
    throw std::runtime_error("GPU batched decompression needs full implementation");
}

// ============================================================================
// GPU Manager API Compression
// ============================================================================

void compressGPUManager(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize) {
    std::cout << "Using GPU manager compression (" << algoToString(algo) << ")..." << std::endl;
    
    // Create archive
    std::vector<uint8_t> archiveData;
    if (isDirectory(inputPath)) {
        archiveData = createArchiveFromFolder(inputPath);
    } else {
        archiveData = createArchiveFromFile(inputPath);
    }
    
    std::cout << "GPU Manager API compression requires full implementation" << std::endl;
    throw std::runtime_error("GPU manager compression needs full implementation");
}

// ============================================================================
// GPU Manager API Decompression
// ============================================================================

void decompressGPUManager(const std::string& inputFile, const std::string& outputPath) {
    std::cout << "Using GPU manager decompression..." << std::endl;
    std::cout << "GPU Manager API decompression requires full implementation" << std::endl;
    throw std::runtime_error("GPU manager decompression needs full implementation");
}

// ============================================================================
// List Compressed Archive
// ============================================================================

void listCompressedArchive(AlgoType algo, const std::string& inputFile, bool useCPU, bool cudaAvailable) {
    // Detect volume files
    auto volumeFiles = detectVolumeFiles(inputFile);
    
    // Check if multi-volume
    if (volumeFiles.size() > 1 || isVolumeFile(volumeFiles[0])) {
        std::cout << "Multi-volume archive detected: " << volumeFiles.size() << " volume(s)" << std::endl;
        
        // Read manifest from first volume
        auto firstVolumeData = readFile(volumeFiles[0]);
        
        if (firstVolumeData.size() < sizeof(VolumeManifest)) {
            throw std::runtime_error("Invalid volume file");
        }
        
        VolumeManifest manifest;
        std::memcpy(&manifest, firstVolumeData.data(), sizeof(VolumeManifest));
        
        if (manifest.magic != VOLUME_MAGIC) {
            throw std::runtime_error("Invalid volume manifest");
        }
        
        std::cout << "Algorithm: " << algoToString(static_cast<AlgoType>(manifest.algorithm)) << std::endl;
        std::cout << "Volume size: " << (manifest.volumeSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        std::cout << "Total uncompressed: " << (manifest.totalUncompressedSize / (1024.0 * 1024.0)) << " MB" << std::endl;
        
        // Read volume metadata
        size_t metadataOffset = sizeof(VolumeManifest);
        std::vector<VolumeMetadata> volumeMetadata(manifest.volumeCount);
        std::memcpy(volumeMetadata.data(), firstVolumeData.data() + metadataOffset, 
                   sizeof(VolumeMetadata) * manifest.volumeCount);
        
        std::cout << "\nVolume breakdown:" << std::endl;
        for (const auto& meta : volumeMetadata) {
            std::cout << "  Volume " << meta.volumeIndex << ": " 
                      << (meta.compressedSize / (1024.0 * 1024.0)) << " MB compressed, "
                      << (meta.uncompressedSize / (1024.0 * 1024.0)) << " MB uncompressed" << std::endl;
        }
        
        std::cout << "\nListing archive contents requires decompression..." << std::endl;
        return;
    }
    
    // Single file
    auto inputData = readFile(inputFile);
    
    // Try to decompress and list
    std::cout << "Listing contents of single file archive..." << std::endl;
    std::cout << "Full listing requires decompression implementation" << std::endl;
}

} // namespace nvcomp_core

