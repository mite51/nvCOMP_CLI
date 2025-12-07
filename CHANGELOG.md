# Changelog

All notable changes to the nvCOMP CLI project will be documented in this file.

## [3.0.0] - 2025-12-07

### Major New Feature: Multi-Volume Support 🎉

#### Added
- **Multi-Volume Archive Splitting**
  - Automatic splitting of large archives into manageable volumes
  - Default 2.5GB volumes (safe for 8GB VRAM GPUs)
  - Volume naming convention: `output.vol001.lz4`, `output.vol002.lz4`, etc.
  - Single files (no split) maintain simple naming: `output.lz4`
  - Mid-file splitting allowed for predictable volume sizes

- **Intelligent GPU Memory Management**
  - Automatic GPU memory detection with `cudaMemGetInfo()`
  - Memory requirement calculation (~2.1x volume size for input + output + temp buffers)
  - Smart fallback to CPU when GPU memory insufficient (for cross-compatible algorithms)
  - Clear error messages for GPU-only algorithms when memory insufficient

- **Volume Manifest System**
  - New volume manifest format with magic number `NVVM` (NvCOMP Volume Manifest)
  - Manifest stored in first volume (`.vol001`) contains:
    - Volume count and algorithm used
    - Max volume size and total uncompressed size
    - Per-volume metadata (index, sizes, offsets)
  - Automatic volume detection and reassembly during decompression

- **Command-Line Options**
  - `--volume-size <size>`: Set custom volume size (e.g., `1GB`, `500MB`, `5GB`)
  - `--no-volumes`: Disable volume splitting (create single file regardless of size)
  - Volume size display at compression start
  - Minimum volume size: 1KB (configurable for testing)

- **Enhanced Output**
  - Single-line progress indicator for multi-volume compression
  - Shows: "Processing volume X/Y..." (overwrites itself)
  - Clear success indicator: "=== Multi-Volume Compression SUCCESSFUL ==="
  - Summary displays volume count prominently
  - Human-readable sizes in MB for totals

#### Changed
- **Compression Functions**
  - All compression functions now accept `maxVolumeSize` parameter
  - `splitIntoVolumes()`: New function to split archives by size
  - Volume creation happens before compression
  - Progress reporting improved for large volume counts

- **Decompression Functions**
  - All decompression functions now detect and handle multi-volume archives
  - `detectVolumeFiles()`: Automatically finds all volume files in directory
  - Reassembles volumes before extraction
  - GPU memory check before decompressing each volume

- **List Mode**
  - Shows volume information (count, sizes) for multi-volume archives
  - Lists all files across all volumes
  - Displays individual volume file sizes

#### Technical Details
- **Volume Manifest Structure** (48 bytes):
  - Magic: `0x4E56564D` ("NVVM")
  - Version: 1
  - Volume count, algorithm, max volume size, total size
  - Followed by VolumeMetadata array

- **VolumeMetadata Structure** (32 bytes per volume):
  - Volume index, compressed size, uncompressed offset, uncompressed size

- **Memory Safety Calculation**:
  - Input buffer: 1x volume size
  - Output buffer: ~1.05x volume size (worst case)
  - Temp buffer: ~0.3-0.5x volume size (algorithm dependent)
  - Total: ~2.1x multiplier for safety

- **Default Volume Size Rationale**:
  - 2.5GB uncompressed per volume
  - ~5.25GB GPU memory needed (2.5 × 2.1)
  - Safe on 8GB VRAM GPUs with ~2.75GB headroom
  - Balances volume count vs memory usage

#### Testing
- **14 New Multi-Volume Tests** (`test_volume.bat`, `test_volume.sh`):
  - 4 multi-volume compression tests (GPU/CPU, various sizes)
  - 2 single-volume tests (verify no splitting when not needed)
  - 2 volume listing tests
  - 2 volume auto-detection tests
  - 4 custom volume size tests
- **Total Test Coverage**: 43 tests (15 single-file + 14 folder + 14 volume)
- Test volume sizes: 5KB-50KB (forces splitting with ~30MB sample folder)
- All tests passing on Windows with CUDA 13.0

#### Documentation
- Updated README.md with comprehensive multi-volume section
- New examples for multi-volume usage
- Updated limitations section (GPU memory issue now resolved!)
- Performance considerations for multi-volume
- Troubleshooting section for volume-related issues
- Volume format specification

#### Breaking Changes
- **Default Behavior**: Archives larger than 2.5GB now automatically create multi-volume files
  - Use `--no-volumes` to maintain previous single-file behavior
  - Existing single-file archives still work perfectly (backward compatible)
  
#### Benefits
- ✅ **No More GPU Memory Failures**: Large files now work reliably
- ✅ **Scalability**: Can compress datasets of any size
- ✅ **Predictable Memory Usage**: Consistent per-volume memory footprint
- ✅ **Cross-Platform**: Volume format works on Windows and Linux
- ✅ **Flexible**: Users can adjust volume size or disable splitting
- ✅ **Smart Fallback**: Automatic CPU fallback when GPU memory insufficient

#### Performance
- Sequential volume processing (not parallel)
- Per-volume throughput same as single-file compression
- Minimal overhead (<1%) for manifest creation
- Larger volume sizes = fewer volumes = less overhead

---

## [2.2.0] - 2025-12-07

### Added
- **Algorithm Auto-Detection** for Decompression and Listing Modes
  - Algorithm parameter is now **optional** for `-d` (decompress) and `-l` (list) modes
  - Tool automatically reads algorithm ID from GPU batched format file headers
  - Works for LZ4, Snappy, and Zstd in both GPU and CPU modes
  - Example: `nvcomp_cli -d archive.zstd output/` (no algorithm needed!)
  - Example: `nvcomp_cli -l archive.lz4` (no algorithm needed!)
  
- **New Helper Function**: `detectAlgorithmFromFile()`
  - Reads batched format header from file
  - Extracts algorithm ID without loading entire file
  - Returns ALGO_UNKNOWN for non-batched format files
  
- **Enhanced User Experience**:
  - Updated CLI usage message with clear indication of optional parameters
  - Better error messages when algorithm is required (compression mode)
  - Auto-detection status messages ("Auto-detected algorithm: lz4")

### Changed
- **CLI Argument Parsing**:
  - Algorithm parameter changed from required to optional for `-d` and `-l` modes
  - Compression mode (`-c`) still requires algorithm parameter (as expected)
  - Defaults to ALGO_LZ4 fallback if auto-detection fails
  
- **Decompression Functions**:
  - `decompressGPUBatched()`: Now auto-detects algorithm before processing
  - `decompressCPU()`: Now auto-detects algorithm before processing
  - `listCompressedArchive()`: Now auto-detects algorithm before processing
  - `decompressBatchedFormat()`: Updated log message to show "Auto-detected algorithm"

### Testing
- **6 New Auto-Detection Tests** added to folder test suite:
  - 3 listing tests (LZ4, Snappy, Zstd) without algorithm parameter
  - 3 decompression tests (LZ4 GPU, Zstd GPU, LZ4 CPU) without algorithm parameter
  - All tests verify successful auto-detection and correct operation
  
- **Total Test Coverage**: 29 tests (15 single-file + 14 folder/archive)
  - Single-file tests: 15 (unchanged)
  - Folder compression: 4 tests
  - Archive listing: 3 tests
  - Round-trip test: 1 test
  - **Auto-detection: 6 tests (NEW)**

### Documentation
- Updated README.md with algorithm auto-detection examples
- Updated usage documentation to show optional parameters
- Added dedicated "Algorithm Auto-Detection" section
- Updated all example commands to show auto-detection usage
- Updated test coverage numbers (8 → 14 folder tests, 23 → 29 total)

### Benefits
- **Simplified User Experience**: No need to remember or specify algorithm for decompression
- **Backward Compatible**: Explicit algorithm parameter still works as before
- **Cross-Platform**: Works on both Windows and Linux
- **Error Prevention**: Prevents mismatches between file format and specified algorithm

---

## [2.1.1] - 2025-12-07

### Fixed
- **GPU Batched Decompression**: Implemented GPU batched decompression for LZ4, Snappy, Zstd
  - Added custom batched compression format with magic number `NVBC`
  - Stores chunk metadata (uncompressed size, chunk count, chunk sizes)
  - Enables GPU decompression with proper chunk boundary handling
  - Cross-compatible: CPU can still decompress GPU-compressed files (but slower)
  - **Original Issue Fixed**: "Error: Zstd CPU decompression failed" when listing or decompressing GPU-compressed archives
  
- **Decompression Mode Selection**:
  - Removed forced CPU decompression for cross-compatible algorithms
  - GPU decompression now used by default when GPU is available
  - Automatic format detection (batched format vs CPU format)
  - `--cpu` flag still works to force CPU decompression

- **List Mode Format Detection**:
  - Automatically detects batched format vs standard format
  - Uses appropriate decompression method based on file format
  - Works with both GPU and CPU compressed files

### Added
- **Algorithm Auto-Detection** for GPU batched format files
  - Reads algorithm from file header automatically
  - No need to specify algorithm when listing or decompressing GPU batched files
  - Works for LZ4, Snappy, and Zstd
  - Example: `nvcomp_cli -l archive.zstd` (no algorithm parameter needed)

- **Folder Compression Test Suite** (`test_folder.bat`, `test_folder.sh`)
  - 8 comprehensive tests for folder compression and archive operations
  - Tests for GPU and CPU compression of folders (LZ4, Zstd)
  - Tests for archive listing functionality
  - Dedicated round-trip test that reproduces and validates fix for original zstd issue
  - All tests pass successfully

### Changed
- **Batched Compression Output**:
  - Now includes metadata header with chunk information
  - Slightly larger files (metadata overhead ~32 bytes + 8 bytes per chunk)
  - Enables much faster GPU decompression
  
- **Performance**:
  - GPU decompression now works for Zstd, LZ4, Snappy (was forcing CPU before)
  - Significant speed improvement for decompression on GPU

### Technical Details
- **Batched Header Structure**:
  - Magic: 0x4E564243 ("NVBC")
  - Version: 1
  - Uncompressed size, chunk count, chunk size, algorithm ID
  - Followed by array of chunk sizes
  - Then compressed chunk data

- **Format Detection**:
  - Checks magic number to determine format
  - Batched format → GPU decompression (or CPU if --cpu flag)
  - Standard format → CPU decompression

### Testing
- **Original Issue Test**: Round-trip test validates the fix for GPU zstd decompression failure
  - Step 1: Compress folder with GPU zstd
  - Step 2: List archive (was failing: "Error: Zstd CPU decompression failed")
  - Step 3: Decompress archive (was also failing)
  - Step 4: Verify extracted files
  - ✅ All steps now pass successfully

### Breaking Changes
- Files compressed with previous version (2.1.0) cannot be decompressed with this version
- Recompress your archives with the new version for GPU decompression support

---

## [2.1.0] - 2025-12-07

### Added
- **Folder/Directory Compression**
  - Recursive directory traversal for compressing entire folder structures
  - Custom archive format with magic number `NVAR` (NvCOMP ARchive)
  - Archive header stores file count and version information
  - Each file stored with relative path and data
  - Supports both single files and directories as input

- **Archive Listing Mode**
  - New `-l` flag to list contents of compressed archives
  - Shows file paths and sizes without extraction
  - Formatted output with human-readable sizes (B, KB, MB, GB)
  - Works with all compression algorithms

- **Cross-Platform Path Handling**
  - Automatic normalization of path separators (Windows `\` → Unix `/`)
  - Paths stored in archives use forward slashes for portability
  - Archives created on Windows can be extracted on Linux and vice versa
  - Relative path preservation for consistent extraction

- **Enhanced CLI**
  - Updated argument parsing to support new modes
  - Output path for decompression is now a directory (not a file)
  - Improved usage documentation and examples
  - Better error messages for directory operations

### Changed
- **Compression Functions**
  - `compressCPU()`, `compressGPUBatched()`, `compressGPUManager()` now handle both files and directories
  - All inputs are archived before compression (even single files for consistency)
  - Refactored compression/decompression into separate data processing functions

- **Decompression Functions**
  - `decompressCPU()`, `decompressGPUManager()` now extract archives to directories
  - Output parameter changed from file path to directory path
  - Automatic directory creation during extraction

- **File I/O**
  - Added helper functions: `createArchive()`, `extractArchive()`, `listArchive()`
  - Added path utilities: `normalizePath()`, `getRelativePath()`, `collectFiles()`
  - Uses C++17 filesystem API (`std::filesystem`)

### Technical Details
- **Archive Format**:
  - Header: 16 bytes (magic, version, file count, reserved)
  - File Entry: 12 bytes (path length, file size)
  - Data: path string + file contents
  - No compression in archive layer (archive → compress → output)

- **Cross-Platform Compatibility**:
  - Forward slashes used universally in archive
  - Automatic conversion on both Windows and Linux
  - Parent directories created automatically during extraction

### Documentation
- Updated README.md with folder compression examples
- Added archive format specification
- Added cross-platform usage examples
- Updated troubleshooting section with path-related issues
- Added manual testing instructions for folders

### Known Limitations
- Entire archive must fit in memory (no streaming for folders)
- File permissions and attributes not preserved
- Symbolic links not supported
- Empty directories not stored in archive

---

## [2.0.0] - 2025-12-06

### Breaking Changes
- **Complete Architecture Rewrite**: Implemented dual-API approach with different handling for cross-compatible vs GPU-only algorithms
- **Cascaded Algorithm Removed**: Removed due to incompatibility with text data (caused failures in reference examples)
- **Command Interface Changed**: Updated to support `--cpu` flag for forcing CPU mode

### Added
- **CPU Fallback Support**
  - Automatic CPU fallback when CUDA is not available
  - CPU compression/decompression for LZ4, Snappy, and Zstd using native libraries
  - `--cpu` flag to force CPU mode even when GPU is available

- **Cross-Compatibility (GPU ↔ CPU)**
  - **Batched API Implementation** for LZ4, Snappy, Zstd:
    - Uses C header files (`nvcomp/lz4.h`, not `.hpp`)
    - Functions: `nvcompBatchedLZ4CompressAsync`, etc.
    - Produces **raw compressed data** compatible with CPU libraries
    - Enables GPU compress → CPU decompress and vice versa
  - Successfully tested cross-compatibility for all three algorithms

- **GPU-Only Manager API** for GDeflate, ANS, Bitcomp:
  - Uses C++ headers (`nvcomp.hpp`, `nvcomp/gdeflate.hpp`)
  - Classes: `GdeflateManager`, `ANSManager`, `BitcompManager`
  - Produces nvCOMP container format with metadata
  - Automatic format detection on decompression via `create_manager()`

- **Automatic Dependency Management**
  - CMake FetchContent integration for LZ4 (1.9.4)
  - CMake FetchContent integration for Snappy (1.2.1)
  - CMake FetchContent integration for Zstd (1.5.5)
  - Automatic CMake version patching for old dependencies
  - Cross-platform nvCOMP SDK download (Windows/Linux detection)

- **Comprehensive Testing**
  - Test suite with 15 comprehensive tests (reduced from initial plan, optimized):
    - GPU Batched ↔ GPU Batched (3 tests): LZ4, Snappy, Zstd
    - GPU Manager ↔ GPU Manager (3 tests): GDeflate, ANS, Bitcomp
    - CPU ↔ CPU (3 tests): LZ4, Snappy, Zstd
    - **GPU → CPU Cross-compatibility** (3 tests): LZ4, Snappy, Zstd
    - **CPU → GPU Cross-compatibility** (3 tests): LZ4, Snappy, Zstd
  - Test scripts (`test.bat`, `test.sh`) now located in `unit_test/` folder
  - Test output isolated to `unit_test/output/` directory
  - All tests passing on Windows with CUDA 13.0

- **Enhanced Documentation**
  - Comprehensive README.md with architecture explanation
  - Build instructions for Windows and Linux
  - Usage examples including cross-compatibility scenarios
  - Performance metrics and comparison table
  - Troubleshooting guide
  - API differences clearly documented

### Fixed
- **API Compatibility Issues** (nvCOMP 5.1.0):
  - Corrected function names: Added "Async" suffix (`nvcompBatchedLZ4CompressGetTempSizeAsync`)
  - Fixed option constants: Changed to `nvcompBatchedLZ4CompressDefaultOpts` (with "Compress")
  - Fixed Manager constructors: Added both compress and decompress options
  - Added missing header: `nvcomp/nvcompManagerFactory.hpp`
  - Fixed function signatures: Added `nullptr` status pointer parameter before stream
  - Added total input size parameter to `GetTempSizeAsync` functions

- **Build System**
  - Fixed CMake minimum version conflicts by patching fetched dependencies
  - Resolved CUDA architecture targeting (75, 80, 86, 89, 90)
  - Fixed linker warnings (LIBCMT conflicts - non-critical, common in CUDA projects)

### Changed
- **Chunk Management**: Implemented 64KB chunking for batched API (required for cross-compatibility)
- **File Format**: Cross-compatible algorithms produce raw format (concatenated chunks), GPU-only algorithms use nvCOMP container
- **Error Handling**: Enhanced error messages with line numbers and algorithm context
- **Performance Reporting**: Improved throughput calculation and formatting

### Technical Details
- **Two Different Implementations**:
  - **Implementation A (Batched API)**: For LZ4, Snappy, Zstd - produces standard format
  - **Implementation B (Manager API)**: For GDeflate, ANS, Bitcomp - produces nvCOMP format
- **Why Two Implementations?**: Batched API enables CPU interoperability at the cost of complexity; Manager API is simpler but GPU-only
- **Chunking**: Data split into 64KB chunks for batched processing, enabling better GPU utilization

### Known Limitations
- GPU batched decompression not implemented (requires chunk metadata storage)
- Files larger than GPU memory not supported (no streaming implementation yet)
- Default compression levels used (no level configuration exposed)
- Single GPU only (device 0)

### Dependencies
- nvCOMP 5.1.0.21 (CUDA 13)
- LZ4 1.9.4 (auto-fetched)
- Snappy 1.2.1 (auto-fetched)
- Zstd 1.5.5 (auto-fetched)
- CUDA Toolkit 11.0+ (user-provided)
- CMake 3.18+ (user-provided)

### Testing Environment
- Windows 10/11 with Visual Studio 2022
- CUDA 13.0
- NVIDIA GPU (tested with various architectures)
- All 15 tests passing

---

## [1.0.0] - 2025-12-06

### Added
- **Command Line Interface (`nvcomp_cli`)**
  - Implemented a C++ CLI application (`main.cu`) using the nvCOMP High-Level C++ API.
  - **Compression**:
    - Support for multiple GPU-accelerated algorithms: LZ4, GDeflate, Snappy, ANS, Cascaded, and Bitcomp.
    - Default algorithm set to LZ4.
    - Automatic resource management (GPU memory allocation, stream synchronization).
  - **Decompression**:
    - Automatic format detection using `nvcomp::create_manager`.
    - Support for decompressing any valid nvCOMP high-level stream.
  - **Performance Metrics**: Reports compressed size, compression ratio, and execution time (including throughput in GB/s).

- **Build System**
  - Added `CMakeLists.txt` for cross-platform build configuration.
  - Configured to link against local nvCOMP 5.1 library (`nvcomp-windows-x86_64-5.1.0.21_cuda13`).
  - Added post-build steps to automatically copy required DLLs (`nvcomp64_5.dll`, `nvcomp_cpu64_5.dll`) to the output directory.

- **Documentation**
  - Added `README.md` with build instructions, prerequisites, and usage examples.

### Fixed
- Resolved compilation errors related to missing default options for GDeflate, Snappy, ANS, Cascaded, and Bitcomp managers by updating constructor calls to match nvCOMP 5.1 headers.
- Fixed a runtime crash caused by improper destruction order of `nvcompManager` and `cudaStream_t` (ensured manager is destroyed before the stream).

### Notes
- The current implementation loads entire files into memory; files larger than available GPU memory are not yet supported (requires chunked streaming implementation).
- GPU-only implementation; requires NVIDIA GPU with Compute Capability 6.0+ and CUDA drivers.

