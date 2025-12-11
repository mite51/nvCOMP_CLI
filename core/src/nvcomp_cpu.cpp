#include "nvcomp_core.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cuda_runtime.h>

// CPU compression libraries
#include "lz4.h"
#include "lz4hc.h"
#include "snappy.h"
#include "zstd.h"

namespace nvcomp_core {

// ============================================================================
// Algorithm Utilities
// ============================================================================

AlgoType parseAlgorithm(const std::string& algo) {
    if (algo == "lz4") return ALGO_LZ4;
    if (algo == "snappy") return ALGO_SNAPPY;
    if (algo == "zstd") return ALGO_ZSTD;
    if (algo == "gdeflate") return ALGO_GDEFLATE;
    if (algo == "ans") return ALGO_ANS;
    if (algo == "bitcomp") return ALGO_BITCOMP;
    return ALGO_UNKNOWN;
}

std::string algoToString(AlgoType algo) {
    switch(algo) {
        case ALGO_LZ4: return "lz4";
        case ALGO_SNAPPY: return "snappy";
        case ALGO_ZSTD: return "zstd";
        case ALGO_GDEFLATE: return "gdeflate";
        case ALGO_ANS: return "ans";
        case ALGO_BITCOMP: return "bitcomp";
        default: return "unknown";
    }
}

bool isCrossCompatible(AlgoType algo) {
    return algo == ALGO_LZ4 || algo == ALGO_SNAPPY || algo == ALGO_ZSTD;
}

bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return error == cudaSuccess && deviceCount > 0;
}

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

static std::vector<uint8_t> compressDataCPU(AlgoType algo, const std::vector<uint8_t>& inputData) {
    size_t inputSize = inputData.size();
    std::vector<uint8_t> outputData;
    size_t compSize = 0;
    
    if (algo == ALGO_LZ4) {
        size_t maxSize = LZ4_compressBound(inputSize);
        outputData.resize(maxSize);
        compSize = LZ4_compress_HC(
            reinterpret_cast<const char*>(inputData.data()),
            reinterpret_cast<char*>(outputData.data()),
            inputSize,
            maxSize,
            LZ4HC_CLEVEL_DEFAULT
        );
        if (compSize == 0) {
            throw std::runtime_error("LZ4 CPU compression failed");
        }
    } else if (algo == ALGO_SNAPPY) {
        size_t maxSize = snappy::MaxCompressedLength(inputSize);
        outputData.resize(maxSize);
        snappy::RawCompress(
            reinterpret_cast<const char*>(inputData.data()),
            inputSize,
            reinterpret_cast<char*>(outputData.data()),
            &compSize
        );
        if (compSize == 0) {
            throw std::runtime_error("Snappy CPU compression failed");
        }
    } else if (algo == ALGO_ZSTD) {
        size_t maxSize = ZSTD_compressBound(inputSize);
        outputData.resize(maxSize);
        compSize = ZSTD_compress(
            outputData.data(),
            maxSize,
            inputData.data(),
            inputSize,
            ZSTD_CLEVEL_DEFAULT
        );
        if (ZSTD_isError(compSize)) {
            throw std::runtime_error("Zstd CPU compression failed");
        }
    } else {
        throw std::runtime_error("Algorithm not supported for CPU compression");
    }
    
    outputData.resize(compSize);
    return outputData;
}

static std::vector<uint8_t> decompressDataCPU(AlgoType algo, const std::vector<uint8_t>& inputData) {
    size_t inputSize = inputData.size();
    std::vector<uint8_t> outputData;
    size_t decompSize = 0;
    
    if (algo == ALGO_LZ4) {
        // Try different output sizes
        for (size_t multiplier = 10; multiplier <= 1000; multiplier *= 10) {
            outputData.resize(inputSize * multiplier);
            int result = LZ4_decompress_safe(
                reinterpret_cast<const char*>(inputData.data()),
                reinterpret_cast<char*>(outputData.data()),
                inputSize,
                outputData.size()
            );
            if (result > 0) {
                decompSize = result;
                break;
            }
        }
        if (decompSize == 0) {
            throw std::runtime_error("LZ4 CPU decompression failed");
        }
    } else if (algo == ALGO_SNAPPY) {
        size_t uncompressedLength;
        if (!snappy::GetUncompressedLength(
            reinterpret_cast<const char*>(inputData.data()),
            inputSize,
            &uncompressedLength
        )) {
            throw std::runtime_error("Snappy: Failed to get uncompressed length");
        }
        outputData.resize(uncompressedLength);
        if (!snappy::RawUncompress(
            reinterpret_cast<const char*>(inputData.data()),
            inputSize,
            reinterpret_cast<char*>(outputData.data())
        )) {
            throw std::runtime_error("Snappy CPU decompression failed");
        }
        decompSize = uncompressedLength;
    } else if (algo == ALGO_ZSTD) {
        unsigned long long uncompressedSize = ZSTD_getFrameContentSize(inputData.data(), inputSize);
        if (uncompressedSize == ZSTD_CONTENTSIZE_ERROR || uncompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
            throw std::runtime_error("Zstd: Failed to get uncompressed size");
        }
        outputData.resize(uncompressedSize);
        decompSize = ZSTD_decompress(
            outputData.data(),
            uncompressedSize,
            inputData.data(),
            inputSize
        );
        if (ZSTD_isError(decompSize)) {
            throw std::runtime_error("Zstd CPU decompression failed");
        }
    } else {
        throw std::runtime_error("Algorithm not supported for CPU decompression");
    }
    
    outputData.resize(decompSize);
    return outputData;
}

static std::vector<uint8_t> decompressBatchedFormat(AlgoType algo, const std::vector<uint8_t>& compressedData) {
    // Check if it's batched format
    if (compressedData.size() < sizeof(BatchedHeader)) {
        // Not batched format, use CPU decompression directly
        return decompressDataCPU(algo, compressedData);
    }
    
    BatchedHeader header;
    std::memcpy(&header, compressedData.data(), sizeof(BatchedHeader));
    
    if (header.magic != BATCHED_MAGIC) {
        // Not batched format, use CPU decompression directly
        return decompressDataCPU(algo, compressedData);
    }
    
    // It's a batched format - extract the compressed chunks and decompress with CPU
    // Use algorithm from header (auto-detect)
    AlgoType actualAlgo = static_cast<AlgoType>(header.algorithm);
    std::cout << "Auto-detected algorithm: " << algoToString(actualAlgo) << std::endl;
    
    size_t chunk_count = header.chunkCount;
    size_t uncompressedSize = header.uncompressedSize;
    
    // Read chunk sizes
    size_t offset = sizeof(BatchedHeader);
    std::vector<uint64_t> chunkSizes64(chunk_count);
    std::memcpy(chunkSizes64.data(), compressedData.data() + offset, sizeof(uint64_t) * chunk_count);
    offset += sizeof(uint64_t) * chunk_count;
    
    // Decompress each chunk with CPU
    std::vector<uint8_t> result;
    result.reserve(uncompressedSize);
    
    for (size_t i = 0; i < chunk_count; i++) {
        size_t chunkSize = static_cast<size_t>(chunkSizes64[i]);
        std::vector<uint8_t> chunk(compressedData.begin() + offset, 
                                   compressedData.begin() + offset + chunkSize);
        
        auto decompressed = decompressDataCPU(actualAlgo, chunk);
        result.insert(result.end(), decompressed.begin(), decompressed.end());
        
        offset += chunkSize;
    }
    
    return result;
}

// ============================================================================
// CPU Compression
// ============================================================================

void compressCPU(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize) {
    std::cout << "Using CPU compression (" << algoToString(algo) << ")..." << std::endl;
    
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
    
    // If single volume, use original behavior
    if (volumes.size() == 1) {
        auto start = std::chrono::high_resolution_clock::now();
        auto compressedData = compressDataCPU(algo, volumes[0]);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end - start).count();
        size_t compSize = compressedData.size();
        
        std::cout << "Compressed size: " << compSize << " bytes" << std::endl;
        std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)totalSize / compSize << "x" << std::endl;
        std::cout << "Time: " << duration << "s (" << (totalSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
        
        // Write with BatchedHeader for algorithm detection compatibility
        // Use a simple single-chunk format
        BatchedHeader header;
        header.magic = BATCHED_MAGIC;
        header.version = BATCHED_VERSION;
        header.uncompressedSize = totalSize;
        header.chunkCount = 1;
        header.chunkSize = totalSize;
        header.algorithm = static_cast<uint32_t>(algo);
        header.reserved = 0;
        
        // Build output: header + chunk size + compressed data
        std::vector<uint8_t> outputData;
        outputData.reserve(sizeof(BatchedHeader) + sizeof(uint64_t) + compSize);
        
        // Append header
        outputData.insert(outputData.end(), 
                         reinterpret_cast<uint8_t*>(&header),
                         reinterpret_cast<uint8_t*>(&header) + sizeof(BatchedHeader));
        
        // Append chunk size
        uint64_t chunkSize64 = compSize;
        outputData.insert(outputData.end(),
                         reinterpret_cast<uint8_t*>(&chunkSize64),
                         reinterpret_cast<uint8_t*>(&chunkSize64) + sizeof(uint64_t));
        
        // Append compressed data
        outputData.insert(outputData.end(), compressedData.begin(), compressedData.end());
        
        writeFile(outputFile, outputData.data(), outputData.size());
        return;
    }
    
    // Multi-volume compression
    std::cout << "\nCompressing " << volumes.size() << " volume(s)..." << std::endl;
    
    std::vector<VolumeMetadata> volumeMetadata;
    uint64_t uncompressedOffset = 0;
    double totalDuration = 0;
    size_t totalCompressedSize = 0;
    
    for (size_t i = 0; i < volumes.size(); i++) {
        // Show progress on single line
        std::cout << "\r  Processing volume " << (i + 1) << "/" << volumes.size() << "..." << std::flush;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto compressed = compressDataCPU(algo, volumes[i]);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end - start).count();
        totalDuration += duration;
        
        // Create volume metadata
        VolumeMetadata meta;
        meta.volumeIndex = i + 1;
        meta.compressedSize = compressed.size();
        meta.uncompressedOffset = uncompressedOffset;
        meta.uncompressedSize = volumes[i].size();
        volumeMetadata.push_back(meta);
        
        uncompressedOffset += volumes[i].size();
        totalCompressedSize += compressed.size();
        
        // Write volume file
        std::string volumeFile = generateVolumeFilename(outputFile, i + 1);
        writeFile(volumeFile, compressed.data(), compressed.size());
    }
    
    std::cout << "\r  Processing volume " << volumes.size() << "/" << volumes.size() << "... Done!" << std::endl;
    
    // Create and prepend manifest to first volume
    VolumeManifest manifest;
    manifest.magic = VOLUME_MAGIC;
    manifest.version = VOLUME_VERSION;
    manifest.volumeCount = static_cast<uint32_t>(volumes.size());
    manifest.algorithm = static_cast<uint32_t>(algo);
    manifest.volumeSize = maxVolumeSize;
    manifest.totalUncompressedSize = totalSize;
    manifest.reserved = 0;
    
    // Read first volume
    std::string firstVolumeFile = generateVolumeFilename(outputFile, 1);
    auto firstVolumeData = readFile(firstVolumeFile);
    
    // Create new first volume with manifest
    std::vector<uint8_t> newFirstVolume;
    
    // Add manifest header
    const uint8_t* manifestBytes = reinterpret_cast<const uint8_t*>(&manifest);
    newFirstVolume.insert(newFirstVolume.end(), manifestBytes, manifestBytes + sizeof(VolumeManifest));
    
    // Add volume metadata array
    const uint8_t* metadataBytes = reinterpret_cast<const uint8_t*>(volumeMetadata.data());
    newFirstVolume.insert(newFirstVolume.end(), metadataBytes, 
                         metadataBytes + sizeof(VolumeMetadata) * volumeMetadata.size());
    
    // Add original compressed data
    newFirstVolume.insert(newFirstVolume.end(), firstVolumeData.begin(), firstVolumeData.end());
    
    // Write updated first volume
    writeFile(firstVolumeFile, newFirstVolume.data(), newFirstVolume.size());
    
    // Update metadata for first volume
    volumeMetadata[0].compressedSize = newFirstVolume.size();
    totalCompressedSize = totalCompressedSize - firstVolumeData.size() + newFirstVolume.size();
    
    std::cout << "\n=== Multi-Volume Compression SUCCESSFUL ===" << std::endl;
    std::cout << "Volumes created: " << volumes.size() << std::endl;
    std::cout << "Total uncompressed: " << (totalSize / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Total compressed: " << (totalCompressedSize / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Overall ratio: " << std::fixed << std::setprecision(2) 
              << (double)totalSize / totalCompressedSize << "x" << std::endl;
    std::cout << "Total time: " << totalDuration << "s (" 
              << (totalSize / (1024.0 * 1024.0 * 1024.0)) / totalDuration << " GB/s)" << std::endl;
}

void decompressCPU(AlgoType algo, const std::string& inputFile, const std::string& outputPath) {
    // Detect volume files
    auto volumeFiles = detectVolumeFiles(inputFile);
    
    // Check if multi-volume
    if (volumeFiles.size() > 1 || isVolumeFile(volumeFiles[0])) {
        // Read manifest from first volume
        auto firstVolumeData = readFile(volumeFiles[0]);
        
        if (firstVolumeData.size() < sizeof(VolumeManifest)) {
            throw std::runtime_error("Invalid volume file: too small for manifest");
        }
        
        VolumeManifest manifest;
        std::memcpy(&manifest, firstVolumeData.data(), sizeof(VolumeManifest));
        
        if (manifest.magic != VOLUME_MAGIC) {
            // Not a multi-volume archive, treat as single file
            std::cout << "Using CPU decompression (" << algoToString(algo) << ")..." << std::endl;
            
            auto start = std::chrono::high_resolution_clock::now();
            auto archiveData = decompressBatchedFormat(algo, firstVolumeData);
            auto end = std::chrono::high_resolution_clock::now();
            
            double duration = std::chrono::duration<double>(end - start).count();
            size_t decompSize = archiveData.size();
            std::cout << "Decompressed size: " << decompSize << " bytes" << std::endl;
            std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
            
            extractArchive(archiveData, outputPath);
            return;
        }
        
        // Multi-volume archive
        std::cout << "Multi-volume archive detected: " << manifest.volumeCount << " volume(s)" << std::endl;
        std::cout << "Using CPU decompression (" << algoToString(static_cast<AlgoType>(manifest.algorithm)) << ")..." << std::endl;
        
        // Read volume metadata
        size_t metadataOffset = sizeof(VolumeManifest);
        std::vector<VolumeMetadata> volumeMetadata(manifest.volumeCount);
        std::memcpy(volumeMetadata.data(), firstVolumeData.data() + metadataOffset, 
                   sizeof(VolumeMetadata) * manifest.volumeCount);
        
        // Check all volumes exist
        if (volumeFiles.size() != manifest.volumeCount) {
            std::cerr << "Error: Expected " << manifest.volumeCount << " volumes, found " << volumeFiles.size() << std::endl;
            std::cerr << "Missing volumes!" << std::endl;
            throw std::runtime_error("Missing volume files");
        }
        
        // Decompress all volumes
        std::vector<uint8_t> fullArchive;
        fullArchive.reserve(manifest.totalUncompressedSize);
        double totalDuration = 0;
        
        for (size_t i = 0; i < volumeFiles.size(); i++) {
            std::cout << "\nDecompressing volume " << (i + 1) << "/" << volumeFiles.size() << "..." << std::endl;
            
            auto volumeData = readFile(volumeFiles[i]);
            
            // Skip manifest and metadata in first volume
            size_t dataOffset = 0;
            if (i == 0) {
                dataOffset = sizeof(VolumeManifest) + sizeof(VolumeMetadata) * manifest.volumeCount;
                volumeData = std::vector<uint8_t>(volumeData.begin() + dataOffset, volumeData.end());
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            auto decompressed = decompressBatchedFormat(static_cast<AlgoType>(manifest.algorithm), volumeData);
            auto end = std::chrono::high_resolution_clock::now();
            
            double duration = std::chrono::duration<double>(end - start).count();
            totalDuration += duration;
            
            std::cout << "  Decompressed: " << decompressed.size() << " bytes in " << duration << "s" << std::endl;
            
            fullArchive.insert(fullArchive.end(), decompressed.begin(), decompressed.end());
        }
        
        std::cout << "\n=== Decompression Summary ===" << std::endl;
        std::cout << "Total decompressed: " << fullArchive.size() << " bytes" << std::endl;
        std::cout << "Total time: " << totalDuration << "s (" 
                  << (fullArchive.size() / (1024.0 * 1024.0 * 1024.0)) / totalDuration << " GB/s)" << std::endl;
        
        // Extract archive
        extractArchive(fullArchive, outputPath);
        return;
    }
    
    // Single file (non-volume)
    // Try to auto-detect algorithm if not specified
    AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
    if (detectedAlgo != ALGO_UNKNOWN) {
        algo = detectedAlgo;
        std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << std::endl;
    }
    
    std::cout << "Using CPU decompression (" << algoToString(algo) << ")..." << std::endl;
    
    auto inputData = readFile(inputFile);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Use batched format handler which works for both batched and standard formats
    auto archiveData = decompressBatchedFormat(algo, inputData);
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    size_t decompSize = archiveData.size();
    std::cout << "Decompressed size: " << decompSize << " bytes" << std::endl;
    std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    // Extract archive
    extractArchive(archiveData, outputPath);
}

} // namespace nvcomp_core


