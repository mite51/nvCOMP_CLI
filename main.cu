/*
 * nvCOMP CLI with CPU Fallback
 * 
 * Two implementations:
 * - Batched API (LZ4, Snappy, Zstd): Cross-compatible with CPU
 * - Manager API (GDeflate, ANS, Bitcomp): GPU-only
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <sstream>

#include <cuda_runtime.h>

namespace fs = std::filesystem;

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

// CPU compression libraries
#include "lz4.h"
#include "lz4hc.h"
#include "snappy.h"
#include "zstd.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

#define NVCOMP_CHECK(call) \
    do { \
        nvcompStatus_t status = call; \
        if (status != nvcompSuccess) { \
            std::cerr << "nvCOMP Error at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// Chunk size for batched API
constexpr size_t CHUNK_SIZE = 1 << 16; // 64KB

// Archive magic number for identification
constexpr uint32_t ARCHIVE_MAGIC = 0x4E564152; // "NVAR" (NvCOMP ARchive)
constexpr uint32_t ARCHIVE_VERSION = 1;

// Batched compression metadata magic
constexpr uint32_t BATCHED_MAGIC = 0x4E564243; // "NVBC" (NvCOMP Batched Compression)
constexpr uint32_t BATCHED_VERSION = 1;

// Volume manifest magic
constexpr uint32_t VOLUME_MAGIC = 0x4E56564D; // "NVVM" (NvCOMP Volume Manifest)
constexpr uint32_t VOLUME_VERSION = 1;

// Default volume size: 2.5GB (safe for 8GB VRAM GPUs)
constexpr uint64_t DEFAULT_VOLUME_SIZE = 2684354560ULL; // 2.5GB in bytes

// Archive header structure
struct ArchiveHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t fileCount;
    uint32_t reserved;
};

// File entry in archive
struct FileEntry {
    uint32_t pathLength;
    uint64_t fileSize;
    // Followed by: path (pathLength bytes), then file data (fileSize bytes)
};

// Batched compression header (for GPU batched API)
struct BatchedHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t uncompressedSize;
    uint32_t chunkCount;
    uint32_t chunkSize;
    uint32_t algorithm; // AlgoType
    uint32_t reserved;
    // Followed by: chunk sizes array (chunkCount * uint64_t), then compressed data
};

// Volume manifest (stored in first volume)
struct VolumeManifest {
    uint32_t magic;                  // VOLUME_MAGIC
    uint32_t version;                // VOLUME_VERSION
    uint32_t volumeCount;            // Total number of volumes
    uint32_t algorithm;              // AlgoType used
    uint64_t volumeSize;             // Max uncompressed size per volume (bytes)
    uint64_t totalUncompressedSize;  // Total archive size before compression
    uint64_t reserved;
    // Followed by: VolumeMetadata array (volumeCount entries)
};

// Metadata for each volume
struct VolumeMetadata {
    uint64_t volumeIndex;            // 1, 2, 3, ...
    uint64_t compressedSize;         // Actual compressed size of this volume
    uint64_t uncompressedOffset;     // Where this volume's data starts in original archive
    uint64_t uncompressedSize;       // How much uncompressed data this volume contains
};

// Algorithm types
enum AlgoType {
    ALGO_LZ4,
    ALGO_SNAPPY,
    ALGO_ZSTD,
    ALGO_GDEFLATE,
    ALGO_ANS,
    ALGO_BITCOMP,
    ALGO_UNKNOWN
};

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

// Forward declarations
AlgoType detectAlgorithmFromFile(const std::string& filename);
std::vector<uint8_t> decompressBatchedFormat(AlgoType algo, const std::vector<uint8_t>& compressedData);

// Volume support functions (forward declarations)
std::string generateVolumeFilename(const std::string& baseFile, size_t volumeIndex);
std::vector<std::string> detectVolumeFiles(const std::string& firstVolume);
bool isVolumeFile(const std::string& filename);
uint64_t parseVolumeSize(const std::string& sizeStr);
bool checkGPUMemoryForVolume(uint64_t volumeSize);

std::vector<uint8_t> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open input file: " + filename);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }
    throw std::runtime_error("Failed to read file: " + filename);
}

void writeFile(const std::string& filename, const void* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }
    file.write(reinterpret_cast<const char*>(data), size);
}

// ============================================================================
// CROSS-PLATFORM PATH AND DIRECTORY UTILITIES
// ============================================================================

// Normalize path separators to forward slashes (cross-platform standard)
std::string normalizePath(const std::string& path) {
    std::string normalized = path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    return normalized;
}

// Get relative path from base directory
std::string getRelativePath(const fs::path& path, const fs::path& base) {
    fs::path relativePath = fs::relative(path, base);
    return normalizePath(relativePath.string());
}

// Check if path is a directory
bool isDirectory(const std::string& path) {
    try {
        return fs::is_directory(path);
    } catch (...) {
        return false;
    }
}

// Recursively collect all files in a directory
std::vector<fs::path> collectFiles(const fs::path& dirPath) {
    std::vector<fs::path> files;
    
    if (!fs::exists(dirPath)) {
        throw std::runtime_error("Directory does not exist: " + dirPath.string());
    }
    
    if (!fs::is_directory(dirPath)) {
        throw std::runtime_error("Not a directory: " + dirPath.string());
    }
    
    for (const auto& entry : fs::recursive_directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    
    return files;
}

// Create directories recursively
void createDirectories(const fs::path& path) {
    if (!path.empty() && path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
}

// ============================================================================
// VOLUME SUPPORT FUNCTIONS
// ============================================================================

// Generate volume filename (e.g., output.lz4 + 1 -> output.vol001.lz4)
std::string generateVolumeFilename(const std::string& baseFile, size_t volumeIndex) {
    fs::path p(baseFile);
    std::string stem = p.stem().string();
    std::string ext = p.extension().string();
    
    std::ostringstream oss;
    oss << stem << ".vol" << std::setw(3) << std::setfill('0') << volumeIndex << ext;
    
    if (p.has_parent_path()) {
        return (p.parent_path() / oss.str()).string();
    }
    return oss.str();
}

// Check if filename is a volume file (contains .vol001, .vol002, etc.)
bool isVolumeFile(const std::string& filename) {
    return filename.find(".vol") != std::string::npos;
}

// Detect all volume files for a given first volume or base name
std::vector<std::string> detectVolumeFiles(const std::string& inputFile) {
    std::vector<std::string> volumes;
    
    // If it's a volume file, extract the base name
    fs::path p(inputFile);
    std::string filename = p.filename().string();
    
    // Check if this is already a volume file
    size_t volPos = filename.find(".vol");
    std::string baseName;
    std::string ext;
    
    if (volPos != std::string::npos) {
        // Extract base name (e.g., "output.vol001.lz4" -> "output" and ".lz4")
        baseName = filename.substr(0, volPos);
        size_t extPos = filename.find('.', volPos + 4);
        if (extPos != std::string::npos) {
            ext = filename.substr(extPos);
        }
    } else {
        // Not a volume file, check if volume 001 exists
        baseName = p.stem().string();
        ext = p.extension().string();
        
        std::string vol001 = generateVolumeFilename(inputFile, 1);
        if (!fs::exists(vol001)) {
            // No volumes exist, return single file
            volumes.push_back(inputFile);
            return volumes;
        }
    }
    
    // Find all volumes
    fs::path dir = p.parent_path();
    if (dir.empty()) dir = ".";
    
    for (size_t i = 1; i <= 9999; i++) {
        std::ostringstream oss;
        oss << baseName << ".vol" << std::setw(3) << std::setfill('0') << i << ext;
        
        fs::path volumePath = dir / oss.str();
        if (fs::exists(volumePath)) {
            volumes.push_back(volumePath.string());
        } else {
            break; // No more volumes
        }
    }
    
    return volumes;
}

// Parse volume size string (e.g., "2.5GB", "500MB", "1TB")
uint64_t parseVolumeSize(const std::string& sizeStr) {
    if (sizeStr.empty()) {
        return DEFAULT_VOLUME_SIZE;
    }
    
    // Find the numeric part
    size_t pos = 0;
    double value = std::stod(sizeStr, &pos);
    
    // Find the unit part
    std::string unit = sizeStr.substr(pos);
    // Convert to uppercase for comparison
    std::transform(unit.begin(), unit.end(), unit.begin(), ::toupper);
    
    uint64_t multiplier = 1;
    if (unit == "KB" || unit == "K") {
        multiplier = 1024ULL;
    } else if (unit == "MB" || unit == "M") {
        multiplier = 1024ULL * 1024ULL;
    } else if (unit == "GB" || unit == "G" || unit.empty()) {
        multiplier = 1024ULL * 1024ULL * 1024ULL;
    } else if (unit == "TB" || unit == "T") {
        multiplier = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
    } else {
        throw std::runtime_error("Invalid volume size unit: " + unit);
    }
    
    uint64_t result = static_cast<uint64_t>(value * multiplier);
    
    // Minimum 1KB to avoid excessive volumes (but allow small sizes for testing)
    const uint64_t MIN_VOLUME_SIZE = 1024ULL;
    if (result < MIN_VOLUME_SIZE) {
        throw std::runtime_error("Volume size too small (minimum 1KB)");
    }
    
    return result;
}

// Check if GPU has sufficient memory for a given volume size
bool checkGPUMemoryForVolume(uint64_t volumeSize) {
    if (!isCudaAvailable()) {
        return false;
    }
    
    size_t free, total;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) {
        return false;
    }
    
    // Need ~2.1x volume size for input + output + temp buffers
    uint64_t requiredWithOverhead = static_cast<uint64_t>(volumeSize * 2.1);
    
    return free >= requiredWithOverhead;
}

// ============================================================================
// ARCHIVE CREATION AND EXTRACTION
// ============================================================================

// Create an uncompressed archive from directory
std::vector<uint8_t> createArchive(const std::string& inputPath) {
    std::vector<uint8_t> archiveData;
    std::vector<fs::path> files;
    fs::path basePath;
    
    if (isDirectory(inputPath)) {
        basePath = fs::path(inputPath);
        files = collectFiles(basePath);
        std::cout << "Collecting files from directory: " << inputPath << std::endl;
        std::cout << "Found " << files.size() << " file(s)" << std::endl;
    } else {
        // Single file - create archive with just this file
        basePath = fs::path(inputPath).parent_path();
        files.push_back(fs::path(inputPath));
        std::cout << "Adding single file: " << inputPath << std::endl;
    }
    
    if (files.empty()) {
        throw std::runtime_error("No files to archive");
    }
    
    // Write header
    ArchiveHeader header;
    header.magic = ARCHIVE_MAGIC;
    header.version = ARCHIVE_VERSION;
    header.fileCount = static_cast<uint32_t>(files.size());
    header.reserved = 0;
    
    const uint8_t* headerBytes = reinterpret_cast<const uint8_t*>(&header);
    archiveData.insert(archiveData.end(), headerBytes, headerBytes + sizeof(ArchiveHeader));
    
    // Write each file
    for (const auto& filePath : files) {
        std::string relativePath = getRelativePath(filePath, basePath);
        if (relativePath.empty() || relativePath == ".") {
            relativePath = filePath.filename().string();
        }
        
        std::cout << "  Adding: " << relativePath << std::flush;
        
        auto fileData = readFile(filePath.string());
        
        FileEntry entry;
        entry.pathLength = static_cast<uint32_t>(relativePath.length());
        entry.fileSize = fileData.size();
        
        // Write entry header
        const uint8_t* entryBytes = reinterpret_cast<const uint8_t*>(&entry);
        archiveData.insert(archiveData.end(), entryBytes, entryBytes + sizeof(FileEntry));
        
        // Write path
        archiveData.insert(archiveData.end(), relativePath.begin(), relativePath.end());
        
        // Write file data
        archiveData.insert(archiveData.end(), fileData.begin(), fileData.end());
        
        std::cout << " (" << fileData.size() << " bytes)" << std::endl;
    }
    
    return archiveData;
}

// Split archive data into volumes
std::vector<std::vector<uint8_t>> splitIntoVolumes(
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

// Extract archive to directory
void extractArchive(const std::vector<uint8_t>& archiveData, const std::string& outputPath) {
    if (archiveData.size() < sizeof(ArchiveHeader)) {
        throw std::runtime_error("Invalid archive: too small");
    }
    
    size_t offset = 0;
    
    // Read header
    ArchiveHeader header;
    std::memcpy(&header, archiveData.data() + offset, sizeof(ArchiveHeader));
    offset += sizeof(ArchiveHeader);
    
    if (header.magic != ARCHIVE_MAGIC) {
        throw std::runtime_error("Invalid archive: bad magic number");
    }
    
    if (header.version != ARCHIVE_VERSION) {
        throw std::runtime_error("Unsupported archive version");
    }
    
    std::cout << "Extracting " << header.fileCount << " file(s) to: " << outputPath << std::endl;
    
    // Create output directory if it doesn't exist
    if (!outputPath.empty()) {
        fs::create_directories(outputPath);
    }
    
    // Extract each file
    for (uint32_t i = 0; i < header.fileCount; i++) {
        if (offset + sizeof(FileEntry) > archiveData.size()) {
            throw std::runtime_error("Invalid archive: truncated file entry");
        }
        
        FileEntry entry;
        std::memcpy(&entry, archiveData.data() + offset, sizeof(FileEntry));
        offset += sizeof(FileEntry);
        
        if (offset + entry.pathLength + entry.fileSize > archiveData.size()) {
            throw std::runtime_error("Invalid archive: truncated file data");
        }
        
        // Read path
        std::string filePath(
            reinterpret_cast<const char*>(archiveData.data() + offset),
            entry.pathLength
        );
        offset += entry.pathLength;
        
        std::cout << "  Extracting: " << filePath << " (" << entry.fileSize << " bytes)" << std::endl;
        
        // Construct full output path
        fs::path fullPath = fs::path(outputPath) / fs::path(filePath);
        
        // Create parent directories
        createDirectories(fullPath);
        
        // Write file
        writeFile(fullPath.string(), archiveData.data() + offset, entry.fileSize);
        offset += entry.fileSize;
    }
    
    std::cout << "Extraction complete." << std::endl;
}

// List archive contents
void listArchive(const std::vector<uint8_t>& archiveData) {
    if (archiveData.size() < sizeof(ArchiveHeader)) {
        throw std::runtime_error("Invalid archive: too small");
    }
    
    size_t offset = 0;
    
    // Read header
    ArchiveHeader header;
    std::memcpy(&header, archiveData.data() + offset, sizeof(ArchiveHeader));
    offset += sizeof(ArchiveHeader);
    
    if (header.magic != ARCHIVE_MAGIC) {
        throw std::runtime_error("Invalid archive: bad magic number");
    }
    
    if (header.version != ARCHIVE_VERSION) {
        throw std::runtime_error("Unsupported archive version");
    }
    
    std::cout << "Archive contains " << header.fileCount << " file(s):" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    uint64_t totalSize = 0;
    
    // List each file
    for (uint32_t i = 0; i < header.fileCount; i++) {
        if (offset + sizeof(FileEntry) > archiveData.size()) {
            throw std::runtime_error("Invalid archive: truncated file entry");
        }
        
        FileEntry entry;
        std::memcpy(&entry, archiveData.data() + offset, sizeof(FileEntry));
        offset += sizeof(FileEntry);
        
        if (offset + entry.pathLength + entry.fileSize > archiveData.size()) {
            throw std::runtime_error("Invalid archive: truncated file data");
        }
        
        // Read path
        std::string filePath(
            reinterpret_cast<const char*>(archiveData.data() + offset),
            entry.pathLength
        );
        offset += entry.pathLength;
        
        // Skip file data
        offset += entry.fileSize;
        totalSize += entry.fileSize;
        
        // Format size with appropriate unit
        double displaySize = static_cast<double>(entry.fileSize);
        std::string sizeUnit = "B";
        
        if (displaySize >= 1024 * 1024 * 1024) {
            displaySize /= (1024.0 * 1024.0 * 1024.0);
            sizeUnit = "GB";
        } else if (displaySize >= 1024 * 1024) {
            displaySize /= (1024.0 * 1024.0);
            sizeUnit = "MB";
        } else if (displaySize >= 1024) {
            displaySize /= 1024.0;
            sizeUnit = "KB";
        }
        
        std::cout << "  " << std::left << std::setw(50) << filePath
                  << std::right << std::setw(8) << std::fixed << std::setprecision(2) 
                  << displaySize << " " << sizeUnit << std::endl;
    }
    
    std::cout << std::string(60, '-') << std::endl;
    
    // Total size
    double totalDisplaySize = static_cast<double>(totalSize);
    std::string totalUnit = "B";
    
    if (totalDisplaySize >= 1024 * 1024 * 1024) {
        totalDisplaySize /= (1024.0 * 1024.0 * 1024.0);
        totalUnit = "GB";
    } else if (totalDisplaySize >= 1024 * 1024) {
        totalDisplaySize /= (1024.0 * 1024.0);
        totalUnit = "MB";
    } else if (totalDisplaySize >= 1024) {
        totalDisplaySize /= 1024.0;
        totalUnit = "KB";
    }
    
    std::cout << "Total: " << std::fixed << std::setprecision(2) 
              << totalDisplaySize << " " << totalUnit << std::endl;
}

// ============================================================================
// CPU COMPRESSION/DECOMPRESSION (LZ4, Snappy, Zstd)
// ============================================================================

std::vector<uint8_t> compressDataCPU(AlgoType algo, const std::vector<uint8_t>& inputData) {
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

std::vector<uint8_t> decompressDataCPU(AlgoType algo, const std::vector<uint8_t>& inputData) {
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

void compressCPU(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize) {
    std::cout << "Using CPU compression (" << algoToString(algo) << ")..." << std::endl;
    
    // Create archive (handles both files and directories)
    std::vector<uint8_t> archiveData;
    if (isDirectory(inputPath)) {
        archiveData = createArchive(inputPath);
    } else {
        archiveData = createArchive(inputPath);
    }
    
    size_t totalSize = archiveData.size();
    std::cout << "Archive size: " << totalSize << " bytes" << std::endl;
    
    // Split into volumes if needed
    auto volumes = splitIntoVolumes(archiveData, maxVolumeSize);
    
    // If single volume, use original behavior
    if (volumes.size() == 1) {
        auto start = std::chrono::high_resolution_clock::now();
        auto outputData = compressDataCPU(algo, volumes[0]);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end - start).count();
        size_t compSize = outputData.size();
        
        std::cout << "Compressed size: " << compSize << " bytes" << std::endl;
        std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)totalSize / compSize << "x" << std::endl;
        std::cout << "Time: " << duration << "s (" << (totalSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
        
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

// ============================================================================
// GPU BATCHED API (LZ4, Snappy, Zstd) - Cross-compatible with CPU
// ============================================================================

void compressGPUBatched(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize) {
    std::cout << "Using GPU batched compression (" << algoToString(algo) << ")..." << std::endl;
    
    // Create archive (handles both files and directories)
    std::vector<uint8_t> archiveData;
    if (isDirectory(inputPath)) {
        archiveData = createArchive(inputPath);
    } else {
        archiveData = createArchive(inputPath);
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
    
    // Multi-volume compression
    std::cout << "\nCompressing " << volumes.size() << " volume(s)..." << std::endl;
    
    std::vector<VolumeMetadata> volumeMetadata;
    uint64_t uncompressedOffset = 0;
    double totalDuration = 0;
    size_t totalCompressedSize = 0;
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    for (size_t volIdx = 0; volIdx < volumes.size(); volIdx++) {
        // Show progress on single line
        std::cout << "\r  Processing volume " << (volIdx + 1) << "/" << volumes.size() << "..." << std::flush;
        
        std::vector<uint8_t>& inputData = volumes[volIdx];
        size_t inputSize = inputData.size();
        
        // Calculate chunks
        size_t chunk_count = (inputSize + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
        // Prepare input chunks
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
        
        double duration = std::chrono::duration<double>(end - start).count();
        totalDuration += duration;
        
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
        
        // Create volume metadata
        VolumeMetadata meta;
        meta.volumeIndex = volIdx + 1;
        meta.compressedSize = outputData.size();
        meta.uncompressedOffset = uncompressedOffset;
        meta.uncompressedSize = inputSize;
        volumeMetadata.push_back(meta);
        
        uncompressedOffset += inputSize;
        totalCompressedSize += outputData.size();
        
        // Write volume file
        std::string volumeFile = generateVolumeFilename(outputFile, volIdx + 1);
        writeFile(volumeFile, outputData.data(), outputData.size());
        
        // Cleanup for this volume
        cudaFree(d_input_data);
        cudaFree(d_input_ptrs);
        cudaFree(d_input_sizes);
        cudaFree(d_output_data);
        cudaFree(d_output_ptrs);
        cudaFree(d_output_sizes);
        cudaFree(d_temp);
    }
    
    cudaStreamDestroy(stream);
    
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

// Detect algorithm from batched format file header
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

std::vector<uint8_t> decompressBatchedFormat(AlgoType algo, const std::vector<uint8_t>& compressedData) {
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
        size_t chunkCompSize = static_cast<size_t>(chunkSizes64[i]);
        
        // Extract this chunk's data
        std::vector<uint8_t> chunkCompressed(chunkCompSize);
        std::memcpy(chunkCompressed.data(), compressedData.data() + offset, chunkCompSize);
        offset += chunkCompSize;
        
        // Decompress this chunk (use detected algorithm)
        auto chunkDecompressed = decompressDataCPU(actualAlgo, chunkCompressed);
        
        // Append to result
        result.insert(result.end(), chunkDecompressed.begin(), chunkDecompressed.end());
    }
    
    if (result.size() != uncompressedSize) {
        std::cerr << "Warning: Decompressed size (" << result.size() << ") doesn't match expected size (" << uncompressedSize << ")" << std::endl;
    }
    
    return result;
}

void decompressGPUBatched(AlgoType algo, const std::string& inputFile, const std::string& outputPath) {
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
            AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
            if (detectedAlgo != ALGO_UNKNOWN) {
                algo = detectedAlgo;
                std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << std::endl;
            }
            
            std::cout << "Decompressing (" << algoToString(algo) << ")..." << std::endl;
            
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
        
        // Check GPU memory
        if (!checkGPUMemoryForVolume(manifest.volumeSize)) {
            std::cout << "Insufficient GPU memory for " << (manifest.volumeSize / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB volumes (need ~" << (manifest.volumeSize * 2.1 / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB VRAM)." << std::endl;
            std::cout << "Falling back to CPU decompression..." << std::endl;
            decompressCPU(algo, inputFile, outputPath);
            return;
        }
        
        std::cout << "Using GPU decompression (" << algoToString(static_cast<AlgoType>(manifest.algorithm)) << ")..." << std::endl;
        
        // Read volume metadata
        size_t metadataOffset = sizeof(VolumeManifest);
        std::vector<VolumeMetadata> volumeMetadata(manifest.volumeCount);
        std::memcpy(volumeMetadata.data(), firstVolumeData.data() + metadataOffset, 
                   sizeof(VolumeMetadata) * manifest.volumeCount);
        
        // Check all volumes exist
        if (volumeFiles.size() != manifest.volumeCount) {
            std::cerr << "Error: Expected " << manifest.volumeCount << " volumes, found " << volumeFiles.size() << std::endl;
            throw std::runtime_error("Missing volume files");
        }
        
        // Decompress all volumes (using CPU for batched format since it's easier)
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
    AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
    if (detectedAlgo != ALGO_UNKNOWN) {
        algo = detectedAlgo;
        std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << std::endl;
    }
    
    std::cout << "Decompressing (" << algoToString(algo) << ")..." << std::endl;
    
    auto compressedData = readFile(inputFile);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Decompress (handles both batched and standard formats)
    auto archiveData = decompressBatchedFormat(algo, compressedData);
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    size_t decompSize = archiveData.size();
    std::cout << "Decompressed size: " << decompSize << " bytes" << std::endl;
    std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    // Extract archive
    extractArchive(archiveData, outputPath);
}

// ============================================================================
// GPU MANAGER API (GDeflate, ANS, Bitcomp) - GPU-only
// ============================================================================

void compressGPUManager(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize) {
    std::cout << "Using GPU manager compression (" << algoToString(algo) << ")..." << std::endl;
    
    // Create archive (handles both files and directories)
    std::vector<uint8_t> archiveData;
    if (isDirectory(inputPath)) {
        archiveData = createArchive(inputPath);
    } else {
        archiveData = createArchive(inputPath);
    }
    
    size_t totalSize = archiveData.size();
    std::cout << "Archive size: " << totalSize << " bytes" << std::endl;
    
    // Split into volumes if needed
    auto volumes = splitIntoVolumes(archiveData, maxVolumeSize);
    
    // If single volume, use original behavior
    if (volumes.size() == 1) {
        size_t inputSize = volumes[0].size();
        std::vector<uint8_t> inputData = volumes[0];
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));
    
    std::shared_ptr<nvcompManagerBase> manager;
    
    if (algo == ALGO_GDEFLATE) {
        manager = std::make_shared<GdeflateManager>(
            CHUNK_SIZE, nvcompBatchedGdeflateCompressDefaultOpts, nvcompBatchedGdeflateDecompressDefaultOpts, stream);
    } else if (algo == ALGO_ANS) {
        manager = std::make_shared<ANSManager>(
            CHUNK_SIZE, nvcompBatchedANSCompressDefaultOpts, nvcompBatchedANSDecompressDefaultOpts, stream);
    } else if (algo == ALGO_BITCOMP) {
        manager = std::make_shared<BitcompManager>(
            CHUNK_SIZE, nvcompBatchedBitcompCompressDefaultOpts, nvcompBatchedBitcompDecompressDefaultOpts, stream);
    }
    
    CompressionConfig comp_config = manager->configure_compression(inputSize);
    
    uint8_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, comp_config.max_compressed_buffer_size));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    manager->compress(d_input, d_output, comp_config);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    size_t compSize = manager->get_compressed_output_size(d_output);
    
    std::cout << "Compressed size: " << compSize << " bytes" << std::endl;
    std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)inputSize / compSize << "x" << std::endl;
    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Time: " << duration << "s (" << (inputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    std::vector<uint8_t> outputData(compSize);
    CUDA_CHECK(cudaMemcpy(outputData.data(), d_output, compSize, cudaMemcpyDeviceToHost));
    
    writeFile(outputFile, outputData.data(), outputData.size());
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
        return;
    }
    
    // Multi-volume compression
    std::cout << "\nCompressing " << volumes.size() << " volume(s)..." << std::endl;
    
    std::vector<VolumeMetadata> volumeMetadata;
    uint64_t uncompressedOffset = 0;
    double totalDuration = 0;
    size_t totalCompressedSize = 0;
    
    for (size_t volIdx = 0; volIdx < volumes.size(); volIdx++) {
        // Show progress on single line
        std::cout << "\r  Processing volume " << (volIdx + 1) << "/" << volumes.size() << "..." << std::flush;
        
        std::vector<uint8_t>& inputData = volumes[volIdx];
        size_t inputSize = inputData.size();
        
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        uint8_t* d_input;
        CUDA_CHECK(cudaMalloc(&d_input, inputSize));
        CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));
        
        std::shared_ptr<nvcompManagerBase> manager;
        
        if (algo == ALGO_GDEFLATE) {
            manager = std::make_shared<GdeflateManager>(
                CHUNK_SIZE, nvcompBatchedGdeflateCompressDefaultOpts, nvcompBatchedGdeflateDecompressDefaultOpts, stream);
        } else if (algo == ALGO_ANS) {
            manager = std::make_shared<ANSManager>(
                CHUNK_SIZE, nvcompBatchedANSCompressDefaultOpts, nvcompBatchedANSDecompressDefaultOpts, stream);
        } else if (algo == ALGO_BITCOMP) {
            manager = std::make_shared<BitcompManager>(
                CHUNK_SIZE, nvcompBatchedBitcompCompressDefaultOpts, nvcompBatchedBitcompDecompressDefaultOpts, stream);
        }
        
        CompressionConfig comp_config = manager->configure_compression(inputSize);
        
        uint8_t* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, comp_config.max_compressed_buffer_size));
        
        auto start = std::chrono::high_resolution_clock::now();
        
        manager->compress(d_input, d_output, comp_config);
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        
        size_t compSize = manager->get_compressed_output_size(d_output);
        
        double duration = std::chrono::duration<double>(end - start).count();
        totalDuration += duration;
        
        std::vector<uint8_t> outputData(compSize);
        CUDA_CHECK(cudaMemcpy(outputData.data(), d_output, compSize, cudaMemcpyDeviceToHost));
        
        // Create volume metadata
        VolumeMetadata meta;
        meta.volumeIndex = volIdx + 1;
        meta.compressedSize = compSize;
        meta.uncompressedOffset = uncompressedOffset;
        meta.uncompressedSize = inputSize;
        volumeMetadata.push_back(meta);
        
        uncompressedOffset += inputSize;
        totalCompressedSize += compSize;
        
        // Write volume file
        std::string volumeFile = generateVolumeFilename(outputFile, volIdx + 1);
        writeFile(volumeFile, outputData.data(), outputData.size());
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
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

void decompressGPUManager(const std::string& inputFile, const std::string& outputPath) {
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
            std::cout << "Using GPU manager decompression (auto-detect)..." << std::endl;
            
            size_t inputSize = firstVolumeData.size();
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            
            uint8_t* d_input;
            CUDA_CHECK(cudaMalloc(&d_input, inputSize));
            CUDA_CHECK(cudaMemcpyAsync(d_input, firstVolumeData.data(), inputSize, cudaMemcpyHostToDevice, stream));
            
            auto manager = create_manager(d_input, stream);
            DecompressionConfig decomp_config = manager->configure_decompression(d_input);
            size_t outputSize = decomp_config.decomp_data_size;
            std::cout << "Detected original size: " << outputSize << " bytes" << std::endl;
            
            uint8_t* d_output;
            CUDA_CHECK(cudaMalloc(&d_output, outputSize));
            
            auto start = std::chrono::high_resolution_clock::now();
            manager->decompress(d_output, d_input, decomp_config);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto end = std::chrono::high_resolution_clock::now();
            
            double duration = std::chrono::duration<double>(end - start).count();
            std::cout << "Time: " << duration << "s (" << (outputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
            
            std::vector<uint8_t> archiveData(outputSize);
            CUDA_CHECK(cudaMemcpy(archiveData.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
            
            cudaFree(d_input);
            cudaFree(d_output);
            cudaStreamDestroy(stream);
            
            extractArchive(archiveData, outputPath);
            return;
        }
        
        // Multi-volume archive
        std::cout << "Multi-volume archive detected: " << manifest.volumeCount << " volume(s)" << std::endl;
        
        // Check GPU memory
        if (!checkGPUMemoryForVolume(manifest.volumeSize)) {
            std::cout << "Insufficient GPU memory for " << (manifest.volumeSize / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB volumes (need ~" << (manifest.volumeSize * 2.1 / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB VRAM)." << std::endl;
            throw std::runtime_error("Insufficient GPU memory for GPU-only algorithm. Cannot fall back to CPU.");
        }
        
        std::cout << "Using GPU manager decompression..." << std::endl;
        
        // Read volume metadata
        size_t metadataOffset = sizeof(VolumeManifest);
        std::vector<VolumeMetadata> volumeMetadata(manifest.volumeCount);
        std::memcpy(volumeMetadata.data(), firstVolumeData.data() + metadataOffset, 
                   sizeof(VolumeMetadata) * manifest.volumeCount);
        
        // Check all volumes exist
        if (volumeFiles.size() != manifest.volumeCount) {
            std::cerr << "Error: Expected " << manifest.volumeCount << " volumes, found " << volumeFiles.size() << std::endl;
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
            
            size_t inputSize = volumeData.size();
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            
            uint8_t* d_input;
            CUDA_CHECK(cudaMalloc(&d_input, inputSize));
            CUDA_CHECK(cudaMemcpyAsync(d_input, volumeData.data(), inputSize, cudaMemcpyHostToDevice, stream));
            
            auto manager = create_manager(d_input, stream);
            DecompressionConfig decomp_config = manager->configure_decompression(d_input);
            size_t outputSize = decomp_config.decomp_data_size;
            
            uint8_t* d_output;
            CUDA_CHECK(cudaMalloc(&d_output, outputSize));
            
            auto start = std::chrono::high_resolution_clock::now();
            manager->decompress(d_output, d_input, decomp_config);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto end = std::chrono::high_resolution_clock::now();
            
            double duration = std::chrono::duration<double>(end - start).count();
            totalDuration += duration;
            
            std::cout << "  Decompressed: " << outputSize << " bytes in " << duration << "s" << std::endl;
            
            std::vector<uint8_t> decompressed(outputSize);
            CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
            
            fullArchive.insert(fullArchive.end(), decompressed.begin(), decompressed.end());
            
            cudaFree(d_input);
            cudaFree(d_output);
            cudaStreamDestroy(stream);
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
    std::cout << "Using GPU manager decompression (auto-detect)..." << std::endl;
    
    auto inputData = readFile(inputFile);
    size_t inputSize = inputData.size();
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));
    
    auto manager = create_manager(d_input, stream);
    
    DecompressionConfig decomp_config = manager->configure_decompression(d_input);
    size_t outputSize = decomp_config.decomp_data_size;
    std::cout << "Detected original size: " << outputSize << " bytes" << std::endl;
    
    uint8_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    manager->decompress(d_output, d_input, decomp_config);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Time: " << duration << "s (" << (outputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    std::vector<uint8_t> archiveData(outputSize);
    CUDA_CHECK(cudaMemcpy(archiveData.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    
    // Extract archive
    extractArchive(archiveData, outputPath);
}

// ============================================================================
// LIST MODE - Show archive contents
// ============================================================================

void listCompressedArchive(AlgoType algo, const std::string& inputFile, bool useCPU, bool cudaAvailable) {
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
            // Not a multi-volume archive, treat as single file - continue with normal logic below
        } else {
            // Multi-volume archive
            std::cout << "Multi-volume archive detected: " << manifest.volumeCount << " volume(s)" << std::endl;
            std::cout << "Algorithm: " << algoToString(static_cast<AlgoType>(manifest.algorithm)) << std::endl;
            
            // List volumes
            std::cout << "\nVolume files:" << std::endl;
            for (size_t i = 0; i < volumeFiles.size(); i++) {
                fs::path p(volumeFiles[i]);
                size_t fileSize = fs::file_size(p);
                std::cout << "  " << p.filename().string() << " - " 
                          << (fileSize / (1024.0 * 1024.0)) << " MB" << std::endl;
            }
            
            std::cout << "\nDecompressing archive to list contents..." << std::endl;
            
            // Check GPU memory if needed
            AlgoType archiveAlgo = static_cast<AlgoType>(manifest.algorithm);
            bool needGPU = !isCrossCompatible(archiveAlgo);
            
            if (needGPU && !checkGPUMemoryForVolume(manifest.volumeSize)) {
                std::cout << "Insufficient GPU memory. Cannot decompress GPU-only algorithm." << std::endl;
                throw std::runtime_error("Insufficient GPU memory");
            }
            
            // Read volume metadata
            size_t metadataOffset = sizeof(VolumeManifest);
            std::vector<VolumeMetadata> volumeMetadata(manifest.volumeCount);
            std::memcpy(volumeMetadata.data(), firstVolumeData.data() + metadataOffset, 
                       sizeof(VolumeMetadata) * manifest.volumeCount);
            
            // Decompress all volumes
            std::vector<uint8_t> fullArchive;
            fullArchive.reserve(manifest.totalUncompressedSize);
            
            for (size_t i = 0; i < volumeFiles.size(); i++) {
                auto volumeData = readFile(volumeFiles[i]);
                
                // Skip manifest and metadata in first volume
                if (i == 0) {
                    size_t dataOffset = sizeof(VolumeManifest) + sizeof(VolumeMetadata) * manifest.volumeCount;
                    volumeData = std::vector<uint8_t>(volumeData.begin() + dataOffset, volumeData.end());
                }
                
                // Decompress based on algorithm type
                if (isCrossCompatible(archiveAlgo)) {
                    auto decompressed = decompressBatchedFormat(archiveAlgo, volumeData);
                    fullArchive.insert(fullArchive.end(), decompressed.begin(), decompressed.end());
                } else {
                    // GPU Manager
                    cudaStream_t stream;
                    CUDA_CHECK(cudaStreamCreate(&stream));
                    
                    uint8_t* d_input;
                    CUDA_CHECK(cudaMalloc(&d_input, volumeData.size()));
                    CUDA_CHECK(cudaMemcpyAsync(d_input, volumeData.data(), volumeData.size(), cudaMemcpyHostToDevice, stream));
                    
                    auto manager = create_manager(d_input, stream);
                    DecompressionConfig decomp_config = manager->configure_decompression(d_input);
                    size_t outputSize = decomp_config.decomp_data_size;
                    
                    uint8_t* d_output;
                    CUDA_CHECK(cudaMalloc(&d_output, outputSize));
                    
                    manager->decompress(d_output, d_input, decomp_config);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    
                    std::vector<uint8_t> decompressed(outputSize);
                    CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
                    
                    fullArchive.insert(fullArchive.end(), decompressed.begin(), decompressed.end());
                    
                    cudaFree(d_input);
                    cudaFree(d_output);
                    cudaStreamDestroy(stream);
                }
            }
            
            std::cout << std::endl;
            listArchive(fullArchive);
            return;
        }
    }
    
    // Single file (non-volume) or non-volume file named like a volume
    // Try to auto-detect algorithm if not specified
    AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
    if (detectedAlgo != ALGO_UNKNOWN) {
        algo = detectedAlgo;
        std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << std::endl;
    }
    
    std::cout << "Listing archive contents..." << std::endl;
    
    // Read compressed file
    auto compressedData = readFile(inputFile);
    
    // Decompress
    std::vector<uint8_t> archiveData;
    
    if (isCrossCompatible(algo)) {
        // Use helper function that handles both batched and standard formats (auto-detects algorithm)
        archiveData = decompressBatchedFormat(algo, compressedData);
    } else {
        // GPU Manager decompression
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        uint8_t* d_input;
        CUDA_CHECK(cudaMalloc(&d_input, compressedData.size()));
        CUDA_CHECK(cudaMemcpyAsync(d_input, compressedData.data(), compressedData.size(), cudaMemcpyHostToDevice, stream));
        
        auto manager = create_manager(d_input, stream);
        
        DecompressionConfig decomp_config = manager->configure_decompression(d_input);
        size_t outputSize = decomp_config.decomp_data_size;
        
        uint8_t* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, outputSize));
        
        manager->decompress(d_output, d_input, decomp_config);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        archiveData.resize(outputSize);
        CUDA_CHECK(cudaMemcpy(archiveData.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
    }
    
    // List archive
    listArchive(archiveData);
}

// ============================================================================
// MAIN
// ============================================================================

void printUsage(const char* appName) {
    std::cerr << "nvCOMP CLI with CPU Fallback & Multi-Volume Support\n\n";
    std::cerr << "Usage:\n";
    std::cerr << "  Compress:   " << appName << " -c <input> <output> <algorithm> [options]\n";
    std::cerr << "  Decompress: " << appName << " -d <input> <output> [algorithm] [options]\n";
    std::cerr << "  List:       " << appName << " -l <archive> [algorithm] [options]\n\n";
    std::cerr << "Arguments:\n";
    std::cerr << "  <input>     Input file or directory (for compression)\n";
    std::cerr << "  <output>    Output file (compression) or directory (decompression)\n";
    std::cerr << "  <archive>   Compressed archive file to list (e.g., output.vol001.lz4 or output.lz4)\n";
    std::cerr << "  [algorithm] Optional for -d/-l (auto-detected), required for -c\n\n";
    std::cerr << "Algorithms:\n";
    std::cerr << "  Cross-compatible (GPU/CPU): lz4, snappy, zstd\n";
    std::cerr << "  GPU-only: gdeflate, ans, bitcomp\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  --cpu              Force CPU mode\n";
    std::cerr << "  --volume-size <N>  Set max volume size (default: 2.5GB)\n";
    std::cerr << "                     Examples: 1GB, 500MB, 5GB\n";
    std::cerr << "  --no-volumes       Disable volume splitting (single file)\n";
    std::cerr << "\nExamples:\n";
    std::cerr << "  # Compress with default 2.5GB volumes\n";
    std::cerr << "  " << appName << " -c input.txt output.lz4 lz4\n\n";
    std::cerr << "  # Compress entire folder with custom volume size\n";
    std::cerr << "  " << appName << " -c mydata/ output.zstd zstd --volume-size 1GB\n\n";
    std::cerr << "  # Compress without volume splitting\n";
    std::cerr << "  " << appName << " -c mydata/ output.lz4 lz4 --no-volumes\n\n";
    std::cerr << "  # Decompress multi-volume archive (auto-detects volumes)\n";
    std::cerr << "  " << appName << " -d output.vol001.lz4 restored/\n\n";
    std::cerr << "  # Decompress single file with auto-detection\n";
    std::cerr << "  " << appName << " -d output.lz4 restored/\n\n";
    std::cerr << "  # List multi-volume archive\n";
    std::cerr << "  " << appName << " -l output.vol001.zstd\n\n";
    std::cerr << "  # Force CPU mode\n";
    std::cerr << "  " << appName << " -c input.txt output.lz4 lz4 --cpu\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            printUsage(argv[0]);
            return 1;
        }
        
        std::string mode = argv[1];
        
        // List mode requires at least 3 args, compress/decompress require at least 4
        if (mode == "-l") {
            if (argc < 3) {
                printUsage(argv[0]);
                return 1;
            }
        } else if (argc < 4) {
            printUsage(argv[0]);
            return 1;
        }
        
        std::string inputPath = argv[2];
        std::string outputPath = (argc >= 4) ? argv[3] : "";
        std::string algoStr = "";
        bool forceCPU = false;
        uint64_t maxVolumeSize = DEFAULT_VOLUME_SIZE;
        bool noVolumes = false;
        
        // Parse algorithm and flags
        for (int i = (mode == "-l" ? 3 : 4); i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--cpu") {
                forceCPU = true;
            } else if (arg == "--no-volumes") {
                noVolumes = true;
            } else if (arg == "--volume-size") {
                if (i + 1 < argc) {
                    i++;
                    try {
                        maxVolumeSize = parseVolumeSize(argv[i]);
                    } catch (const std::exception& e) {
                        std::cerr << "Error: " << e.what() << std::endl;
                        return 1;
                    }
                } else {
                    std::cerr << "Error: --volume-size requires a size argument" << std::endl;
                    printUsage(argv[0]);
                    return 1;
                }
            } else {
                // Assume it's an algorithm
                algoStr = arg;
            }
        }
        
        // If --no-volumes is set, use a very large volume size
        if (noVolumes) {
            maxVolumeSize = UINT64_MAX;
        }
        
        // Parse algorithm (default to ALGO_UNKNOWN for auto-detection)
        AlgoType algo = ALGO_UNKNOWN;
        if (!algoStr.empty()) {
            algo = parseAlgorithm(algoStr);
            if (algo == ALGO_UNKNOWN) {
                std::cerr << "Unknown algorithm: " << algoStr << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        } else {
            // No algorithm specified, try auto-detection (only for decompression/list modes)
            if (mode == "-c") {
                std::cerr << "Error: Algorithm required for compression mode" << std::endl;
                printUsage(argv[0]);
                return 1;
            }
            // For decompression and list modes, will auto-detect from file
            algo = ALGO_LZ4; // Default fallback if auto-detection fails
        }
        
        bool cudaAvailable = isCudaAvailable();
        bool useCPU = forceCPU || !cudaAvailable;
        
        if (!cudaAvailable && !forceCPU) {
            std::cout << "CUDA not available, falling back to CPU..." << std::endl;
            useCPU = true;
        }
        
        if (useCPU && !isCrossCompatible(algo)) {
            std::cerr << "Error: Algorithm '" << algoStr << "' is GPU-only and cannot run on CPU" << std::endl;
            return 1;
        }
        
        if (mode == "-c") {
            // Compression mode
            std::cout << "Volume size: " << (maxVolumeSize == UINT64_MAX ? "unlimited" : 
                         std::to_string(maxVolumeSize / (1024.0 * 1024.0 * 1024.0)) + " GB") << std::endl;
            
            if (useCPU) {
                compressCPU(algo, inputPath, outputPath, maxVolumeSize);
            } else {
                if (isCrossCompatible(algo)) {
                    compressGPUBatched(algo, inputPath, outputPath, maxVolumeSize);
                } else {
                    compressGPUManager(algo, inputPath, outputPath, maxVolumeSize);
                }
            }
        } else if (mode == "-d") {
            // Decompression mode
            if (useCPU) {
                decompressCPU(algo, inputPath, outputPath);
            } else {
                if (isCrossCompatible(algo)) {
                    // Use GPU batched decompression
                    decompressGPUBatched(algo, inputPath, outputPath);
                } else {
                    decompressGPUManager(inputPath, outputPath);
                }
            }
        } else if (mode == "-l") {
            // List mode
            listCompressedArchive(algo, inputPath, useCPU, cudaAvailable);
        } else {
            printUsage(argv[0]);
            return 1;
        }
        
        std::cout << "Done." << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
