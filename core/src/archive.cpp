#include "nvcomp_core.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

namespace fs = std::filesystem;

namespace nvcomp_core {

// ============================================================================
// File I/O Utilities
// ============================================================================

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

std::string normalizePath(const std::string& path) {
    std::string normalized = path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    return normalized;
}

std::string getRelativePath(const std::string& path, const std::string& base) {
    fs::path fsPath(path);
    fs::path fsBase(base);
    fs::path relativePath = fs::relative(fsPath, fsBase);
    return normalizePath(relativePath.string());
}

bool isDirectory(const std::string& path) {
    try {
        return fs::is_directory(path);
    } catch (...) {
        return false;
    }
}

void createDirectories(const std::string& path) {
    fs::path fsPath(path);
    if (!fsPath.empty() && fsPath.has_parent_path()) {
        fs::create_directories(fsPath.parent_path());
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

static std::vector<fs::path> collectFiles(const fs::path& dirPath) {
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

// ============================================================================
// Archive Creation
// ============================================================================

std::vector<uint8_t> createArchiveFromFolder(const std::string& folderPath) {
    std::vector<uint8_t> archiveData;
    fs::path basePath(folderPath);
    
    if (!isDirectory(folderPath)) {
        throw std::runtime_error("Not a directory: " + folderPath);
    }
    
    std::vector<fs::path> files = collectFiles(basePath);
    std::cout << "Collecting files from directory: " << folderPath << std::endl;
    std::cout << "Found " << files.size() << " file(s)" << std::endl;
    
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
        std::string relativePath = getRelativePath(filePath.string(), basePath.string());
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

std::vector<uint8_t> createArchiveFromFile(const std::string& filePath) {
    std::vector<uint8_t> archiveData;
    
    fs::path p(filePath);
    if (!fs::exists(p) || !fs::is_regular_file(p)) {
        throw std::runtime_error("File does not exist or is not a regular file: " + filePath);
    }
    
    std::cout << "Adding single file: " << filePath << std::endl;
    
    // Write header
    ArchiveHeader header;
    header.magic = ARCHIVE_MAGIC;
    header.version = ARCHIVE_VERSION;
    header.fileCount = 1;
    header.reserved = 0;
    
    const uint8_t* headerBytes = reinterpret_cast<const uint8_t*>(&header);
    archiveData.insert(archiveData.end(), headerBytes, headerBytes + sizeof(ArchiveHeader));
    
    // Read file data
    auto fileData = readFile(filePath);
    std::string filename = p.filename().string();
    
    FileEntry entry;
    entry.pathLength = static_cast<uint32_t>(filename.length());
    entry.fileSize = fileData.size();
    
    // Write entry header
    const uint8_t* entryBytes = reinterpret_cast<const uint8_t*>(&entry);
    archiveData.insert(archiveData.end(), entryBytes, entryBytes + sizeof(FileEntry));
    
    // Write path
    archiveData.insert(archiveData.end(), filename.begin(), filename.end());
    
    // Write file data
    archiveData.insert(archiveData.end(), fileData.begin(), fileData.end());
    
    std::cout << "  Added: " << filename << " (" << fileData.size() << " bytes)" << std::endl;
    
    return archiveData;
}

// ============================================================================
// Archive Extraction
// ============================================================================

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
        createDirectories(fullPath.string());
        
        // Write file
        writeFile(fullPath.string(), archiveData.data() + offset, entry.fileSize);
        offset += entry.fileSize;
    }
    
    std::cout << "Extraction complete." << std::endl;
}

// ============================================================================
// Archive Listing
// ============================================================================

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

} // namespace nvcomp_core

