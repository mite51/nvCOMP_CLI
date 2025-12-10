#include "nvcomp_core.hpp"
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

namespace nvcomp_core {

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

bool isVolumeFile(const std::string& filename) {
    return filename.find(".vol") != std::string::npos;
}

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

} // namespace nvcomp_core

