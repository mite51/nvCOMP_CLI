# nvCOMP Qt UI Implementation Plan

## Overview

Transform the nvCOMP CLI into a cross-platform GUI application using Qt 6, with full OS integration for Windows and Linux (Ubuntu). The approach maintains the existing CUDA compression core while adding a modern, native UI layer.

**Architecture:**
- **Core Library**: Shared C++/CUDA compression logic (DLL/SO)
- **CLI Application**: Existing command-line interface (thin wrapper)
- **GUI Application**: Qt 6 desktop application (new)
- **Platform Integration**: Windows context menus, Linux desktop integration

**Timeline:** 8-10 weeks (assuming ~2-3 LLM sessions per week)

---

## Project Structure (Target)

```
nvCOMP_Project/
├── CMakeLists.txt                    (root - builds all targets)
├── core/                             (shared compression library)
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── nvcomp_core.hpp          (C++ API)
│   │   └── nvcomp_c_api.h           (C API for external bindings)
│   └── src/
│       ├── nvcomp_core.cu           (GPU compression)
│       ├── nvcomp_cpu.cpp           (CPU fallback)
│       ├── archive.cpp              (archive management)
│       ├── volume.cpp               (multi-volume support)
│       └── nvcomp_c_api.cpp         (C API implementation)
│
├── cli/                              (command-line interface)
│   ├── CMakeLists.txt
│   └── main.cu                      (existing, will be refactored)
│
├── gui/                              (Qt GUI application)
│   ├── CMakeLists.txt
│   ├── resources/
│   │   ├── icons/
│   │   ├── nvcomp.qrc
│   │   └── styles/
│   ├── src/
│   │   ├── main.cpp
│   │   ├── mainwindow.cpp/h
│   │   ├── compression_worker.cpp/h
│   │   ├── settings_dialog.cpp/h
│   │   ├── archive_viewer.cpp/h
│   │   └── gpu_monitor.cpp/h
│   └── ui/
│       ├── mainwindow.ui
│       ├── settings_dialog.ui
│       └── archive_viewer.ui
│
├── platform/                         (OS-specific integrations)
│   ├── windows/
│   │   ├── context_menu.cpp/h
│   │   ├── file_associations.cpp/h
│   │   └── installer.wxs
│   └── linux/
│       ├── desktop_integration.cpp/h
│       ├── nautilus_extension.py
│       └── debian/
│
└── docs/
    ├── BUILD.md
    ├── ARCHITECTURE.md
    └── USER_GUIDE.md
```

---

## Unit Testing Strategy

### Overview

This project follows a **test-driven development (TDD) approach** where all code changes must be accompanied by appropriate unit tests. Testing is integrated from the start to ensure reliability and prevent regressions. 

### 

Unit tests can be found in /unit_tests, there are a number of .bat/.sh files to run tests for major features. Before starting a new feature and when completing a feature, all tests must pass. If the feature being added does not have sufficient test coverage, create a new test or extend an existing one. Ensure tests work for windows and linux.

---

## Phase 1: Core Refactoring (Weeks 1-2)

> **⚠️ Testing Requirement**: All tasks in all phases must pass existing tests and add new tests for new functionality. See Unit Testing Strategy above.

### Task 1.1: Extract Core Library from main.cu
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** None

**Objective:** Separate reusable compression logic from CLI-specific code into a shared library.

**Deliverables:**
- `core/src/nvcomp_core.cu` - GPU compression functions
- `core/src/nvcomp_cpu.cpp` - CPU fallback implementations  
- `core/src/archive.cpp` - Archive creation/extraction
- `core/src/volume.cpp` - Multi-volume support
- `core/include/nvcomp_core.hpp` - C++ API header
- `core/CMakeLists.txt` - Build configuration for shared library

**Success Criteria:**
- Core library compiles as shared library (.dll/.so)
- All compression algorithms accessible via C++ API
- No CLI-specific code in core library
- ✅ All existing unit tests pass
- ✅ New tests added: `test_core_api.cpp` (test C++ API functions)
- ✅ Test coverage >80% for new code

---

### Task 1.2: Create C API Wrapper
**Duration:** 1 session  
**Complexity:** Low-Medium  
**Dependencies:** Task 1.1

**Objective:** Provide a clean C API for cross-language compatibility and potential future bindings.

**Deliverables:**
- `core/include/nvcomp_c_api.h` - C API header with extern "C"
- `core/src/nvcomp_c_api.cpp` - C API implementation wrapping C++ core
- Progress callback system for UI integration
- Error handling via error codes and messages

**Success Criteria:**
- C API can compress/decompress files
- Progress callbacks work correctly
- Thread-safe error message retrieval
- ✅ All existing unit tests pass
- ✅ New tests added: `test_c_api.cpp` (test C API wrapper)
- ✅ Test coverage >80% for C API code

---

### Task 1.3: Refactor CLI to Use Core Library
**Duration:** 1 session  
**Complexity:** Low  
**Dependencies:** Tasks 1.1, 1.2

**Objective:** Convert existing `main.cu` to a thin wrapper around the core library.

**Deliverables:**
- Simplified `cli/main.cu` using core library
- Maintained backward compatibility (same CLI arguments)
- Updated `cli/CMakeLists.txt` linking to core library

**Success Criteria:**
- All existing CLI functionality works unchanged
- CLI executable size reduced significantly
- CLI can be built independently
- ✅ All existing CLI tests pass
- ✅ New tests added for refactored CLI interface
- ✅ Integration tests verify CLI uses core library correctly

---

### Task 1.4: Update Root CMake Build System
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** Tasks 1.1-1.3

**Objective:** Create unified CMake system that builds core, CLI, and prepares for GUI.

**Deliverables:**
- Root `CMakeLists.txt` with options for CLI/GUI builds
- Proper library installation targets
- Cross-platform configuration (Windows/Linux)
- RPATH/DLL copy logic

**Success Criteria:**
- `cmake -DBUILD_CLI=ON` builds CLI successfully
- Core library installs to system locations
- Windows: DLLs copied to executable directory
- Linux: RPATH configured correctly
- ✅ All tests build and pass with new CMake configuration
- ✅ Tests verify library installation paths

---

## Phase 2: Basic Qt GUI (Weeks 3-4)

> **⚠️ Testing Requirement**: All GUI tasks must include Qt Test-based unit tests. Mock heavy operations for fast test execution.

### Task 2.1: Set Up Qt Project Structure
**Duration:** 1 session  
**Complexity:** Low  
**Dependencies:** Phase 1 complete

**Objective:** Initialize Qt 6 GUI project with basic window and build configuration.

**Deliverables:**
- `gui/CMakeLists.txt` with Qt 6 configuration
- `gui/src/main.cpp` - Qt application entry point
- `gui/src/mainwindow.cpp/h` - Main window class skeleton
- `gui/ui/mainwindow.ui` - Qt Designer UI file
- Basic application icon and resources

**Success Criteria:**
- Qt application compiles and runs
- Empty main window displays correctly
- Application icon shows properly
- Cross-platform compatibility (Windows/Linux)
- ✅ Basic GUI test suite created: `test_mainwindow.cpp`
- ✅ Tests verify window initialization and basic properties

---

### Task 2.2: Implement Main Window UI Layout
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** Task 2.1

**Objective:** Design and implement the main user interface with all controls.

**Deliverables:**
- File/folder selection area (drag-drop support)
- Algorithm selector (combo box with GPU/CPU indicators)
- Compression/Decompression buttons
- Progress bar with status text
- Settings button (opens dialog)
- GPU availability indicator
- Volume size configuration

**UI Components:**
- QGroupBox for input/output selection
- QComboBox for algorithm selection
- QProgressBar for operations
- QListWidget for file display
- QCheckBox for options (CPU mode, volumes)
- QPushButton for actions

**Success Criteria:**
- UI layout is intuitive and professional
- Responsive design (resizing works)
- All controls properly connected to slots
- Drag-and-drop file acceptance works
- ✅ Tests added for UI interactions: button clicks, combo box selections
- ✅ Tests verify drag-and-drop acceptance (mocked)

---

### Task 2.3: Create Compression Worker Thread
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** Tasks 2.2, Phase 1

**Objective:** Implement background worker for compression operations to keep UI responsive.

**Deliverables:**
- `gui/src/compression_worker.cpp/h` - QThread-based worker
- Integration with core library C++ API
- Signal/slot communication for progress updates
- Cancellation support
- Error handling and reporting

**Key Features:**
- Runs compression in separate thread
- Emits progress signals (percentage, speed, ETA)
- Handles all algorithm types
- Graceful cancellation
- Memory-efficient operation

**Success Criteria:**
- Compression doesn't freeze UI
- Progress updates smoothly in real-time
- Can cancel long-running operations
- Errors reported to main thread correctly
- ✅ Worker thread tests added: `test_worker.cpp`
- ✅ Tests verify signal emissions and thread safety
- ✅ Tests verify cancellation mechanism

---

### Task 2.4: Implement Compress/Decompress Functionality
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** Tasks 2.2, 2.3

**Objective:** Connect UI controls to compression worker and implement full workflow.

**Deliverables:**
- Compress button handler (file/folder selection)
- Decompress button handler (output folder selection)
- Progress bar updates from worker signals
- Completion notifications (QMessageBox)
- Compression statistics display (ratio, speed, size)
- Error handling and user feedback

**Success Criteria:**
- Can compress single files successfully
- Can compress entire folders
- Can decompress archives
- Multi-volume archives work correctly
- User receives clear feedback on success/failure
- ✅ All existing tests pass
- ✅ Integration tests added for compress/decompress workflows
- ✅ Tests verify error handling and user feedback

---

## Phase 3: Advanced Features (Weeks 5-6)

### Task 3.1: Implement Archive Viewer
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** Task 2.4

**Objective:** Create a dialog to browse archive contents without extracting.

**Deliverables:**
- `gui/src/archive_viewer.cpp/h` - Archive browser dialog
- `gui/ui/archive_viewer.ui` - UI layout
- Tree view showing archive structure
- File size, path, and statistics display
- "Extract Selected" functionality
- Sorting and filtering capabilities

**Success Criteria:**
- Can list contents of any compressed archive
- Multi-volume archives fully supported
- Tree structure mirrors original folders
- Shows file sizes and compression ratios
- Can extract individual files/folders
- ✅ All existing tests pass
- ✅ Tests added for archive viewer: `test_archive_viewer.cpp`
- ✅ Tests verify tree building and extraction functionality

---

### Task 3.2: Create Settings Dialog
**Duration:** 1 session  
**Complexity:** Low-Medium  
**Dependencies:** Task 2.4

**Objective:** Implement user preferences and configuration management.

**Deliverables:**
- `gui/src/settings_dialog.cpp/h` - Settings dialog
- `gui/ui/settings_dialog.ui` - UI layout
- QSettings integration for persistence
- Default algorithm selection
- Default volume size
- CPU/GPU preference
- Output path templates
- Theme selection (light/dark)

**Success Criteria:**
- Settings persist between sessions
- Changes apply immediately
- Validation for volume sizes
- Cross-platform settings storage
- ✅ All existing tests pass
- ✅ Tests added: `test_settings.cpp` for QSettings persistence
- ✅ Tests verify validation and default values

---

### Task 3.3: Add GPU Monitoring Widget
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** Task 2.3, Phase 1

**Objective:** Display real-time GPU status and memory usage.

**Deliverables:**
- `gui/src/gpu_monitor.cpp/h` - GPU monitoring widget
- CUDA memory query integration
- Real-time VRAM usage display
- GPU model and driver info
- Warning when VRAM insufficient for volume
- Optional: Temperature monitoring

**Success Criteria:**
- Shows available VRAM correctly
- Updates during compression operations
- Warns before attempting too-large compression
- Graceful handling when no GPU available
- ✅ All existing tests pass
- ✅ Tests added: `test_gpu_monitor.cpp` with mocked CUDA calls
- ✅ Tests verify VRAM calculation and warning thresholds

---

### Task 3.4: Implement Batch Operations
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** Task 2.4

**Objective:** Allow compression of multiple files/folders in sequence.

**Deliverables:**
- Queue system for multiple operations
- Batch progress indicator (file 3 of 10)
- Option to stop after current file
- Summary report at completion
- CSV export of compression statistics

**Success Criteria:**
- Can queue multiple compress/decompress operations
- Operations run sequentially
- Overall and per-file progress shown
- Can pause/resume batch operations
- ✅ All existing tests pass
- ✅ Tests added: `test_batch_operations.cpp` for queue management
- ✅ Tests verify sequential processing and error handling

---

## Phase 4: Windows Integration (Week 7)

### Task 4.1: Windows Context Menu Integration
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** GUI functional (Phase 2-3)

**Objective:** Add "Compress with nvCOMP" to Windows Explorer right-click menu.

**Deliverables:**
- `platform/windows/context_menu.cpp/h` - Registry management
- Context menu for files
- Context menu for folders
- "Compress here" option
- "Compress and email" option (creates archive)
- Dynamic submenu with algorithm choices
- Installer integration (register on install)

**Implementation:**
- Registry keys under `HKEY_CLASSES_ROOT\*\shell`
- Registry keys under `HKEY_CLASSES_ROOT\Directory\shell`
- Icon display in context menu
- Command-line argument parsing in GUI app

**Success Criteria:**
- Right-click on file shows nvCOMP option
- Right-click on folder shows nvCOMP option
- Clicking launches GUI with file pre-selected
- Uninstaller removes context menu
- ✅ All existing tests pass
- ✅ Tests added: `test_windows_context_menu.cpp` for registry operations
- ✅ Tests verify registration/unregistration (may require admin)

---

### Task 4.2: Windows File Associations
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** Task 4.1

**Objective:** Associate compressed file extensions with nvCOMP.

**Deliverables:**
- `platform/windows/file_associations.cpp/h` - Association management
- Register extensions: `.lz4`, `.zstd`, `.snappy`, `.nvcomp`
- Custom icons for each archive type
- "Open with nvCOMP" as default action
- Shell thumbnail provider (shows compression info)
- Property sheet extension (archive properties)

**Success Criteria:**
- Double-clicking `.lz4` opens nvCOMP
- Archive files show custom icons
- File properties show compression details
- Windows Search can index archive contents
- ✅ All existing tests pass
- ✅ Tests added: `test_file_associations.cpp` for ProgID registration
- ✅ Tests verify extension associations and icon extraction

---

### Task 4.3: Windows Installer (WiX)
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** Tasks 4.1, 4.2

**Objective:** Create professional MSI installer for Windows.

**Deliverables:**
- `platform/windows/installer.wxs` - WiX configuration
- Install nvCOMP GUI and CLI
- Install nvCOMP core DLL and dependencies
- Register context menu and file associations
- Create Start Menu shortcuts
- Create Desktop shortcut (optional)
- Add to Programs and Features
- Clean uninstall support

**Features:**
- Per-user or system-wide installation
- Custom installation directory
- Optional components (context menu, file associations)
- Version checking and upgrade logic
- Silent install mode

**Success Criteria:**
- MSI installs all components correctly
- Context menu and associations work immediately
- Uninstaller removes all traces
- Upgrade preserves user settings
- ✅ All existing tests pass
- ✅ Manual testing checklist for installer validation
- ✅ Tests verify installer components and registry entries

---

## Phase 5: Linux Integration (Week 8)

### Task 5.1: Linux Desktop Integration
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** GUI functional (Phase 2-3)

**Objective:** Integrate nvCOMP into Linux desktop environment.

**Deliverables:**
- `platform/linux/desktop_integration.cpp/h` - Desktop file generation
- `.desktop` file for application launcher
- MIME type definitions for archives
- Icon installation (various sizes)
- Application menu entry (Utilities → Archiving)
- Default application registration
- XDG integration

**Files Created:**
- `/usr/share/applications/nvcomp.desktop`
- `/usr/share/mime/packages/nvcomp.xml`
- `/usr/share/icons/hicolor/*/apps/nvcomp.png`
- `~/.local/share/applications/` (user install)

**Success Criteria:**
- App appears in application menu
- Double-clicking archives opens nvCOMP
- Icons display correctly in file manager
- System recognizes archive MIME types
- ✅ All existing tests pass
- ✅ Tests added: `test_linux_desktop.cpp` for .desktop file generation
- ✅ Tests verify XDG integration and MIME type registration

---

### Task 5.2: Nautilus/Nemo Context Menu Extension
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** Task 5.1

**Objective:** Add right-click menu integration for Nautilus and Nemo file managers.

**Deliverables:**
- `platform/linux/nautilus_extension.py` - Python-Nautilus extension
- Nemo script support
- Context menu for files
- Context menu for folders
- "Compress with nvCOMP" submenu
- Algorithm selection in submenu
- Integration with both file managers

**Installation:**
- Nautilus: `~/.local/share/nautilus-python/extensions/`
- Nemo: `~/.local/share/nemo/scripts/`
- System-wide: `/usr/share/nautilus-python/extensions/`

**Success Criteria:**
- Right-click on file shows nvCOMP option
- Right-click on folder shows nvCOMP option
- Works in both Nautilus and Nemo
- Respects user preferences
- ✅ All existing tests pass
- ✅ Tests added for Nautilus extension (Python unittest)
- ✅ Manual testing checklist for file manager integration

---

### Task 5.3: Debian/Ubuntu Package Creation
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** Tasks 5.1, 5.2

**Objective:** Create .deb package for easy installation on Ubuntu/Debian.

**Deliverables:**
- `platform/linux/debian/control` - Package metadata
- `platform/linux/debian/rules` - Build rules
- `platform/linux/debian/postinst` - Post-install script
- `platform/linux/debian/prerm` - Pre-removal script
- Package dependencies (Qt6, CUDA runtime)
- Desktop integration in postinst
- Proper library paths and RPATH

**Package Contents:**
```
/usr/
├── bin/
│   ├── nvcomp-gui
│   └── nvcomp-cli
├── lib/
│   └── libnvcomp_core.so
└── share/
    ├── applications/nvcomp.desktop
    ├── icons/
    ├── mime/packages/nvcomp.xml
    └── doc/nvcomp/
```

**Success Criteria:**
- Package installs without errors
- All dependencies resolved automatically
- Desktop integration works immediately
- Package can be removed cleanly
- Package works on Ubuntu 20.04, 22.04, 24.04
- ✅ All existing tests pass
- ✅ Package passes lintian with no errors
- ✅ Manual installation testing on target Ubuntu versions

---

### Task 5.4: AppImage Creation
**Duration:** 1 session  
**Complexity:** Low-Medium  
**Dependencies:** Task 5.1

**Objective:** Create portable AppImage for broader Linux compatibility.

**Deliverables:**
- AppImage build script
- Bundled Qt libraries
- Bundled nvCOMP core library
- CUDA runtime bundling (or detection)
- Desktop integration support
- First-run setup dialog

**Success Criteria:**
- Single-file executable
- Runs on multiple Linux distros
- Can optionally integrate with desktop
- CUDA works when available
- Graceful CPU fallback
- ✅ All existing tests pass
- ✅ AppImage tested on Ubuntu, Fedora, Arch
- ✅ Tests verify bundled dependencies are complete

---

## Phase 6: Polish and Testing (Weeks 9-10)

### Task 6.1: Internationalization (i18n)
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** All GUI tasks

**Objective:** Prepare application for translation to multiple languages.

**Deliverables:**
- Extract all user-facing strings using Qt's tr()
- Create base translation file (.ts)
- Example translations (English, Spanish, Chinese)
- Language selector in settings
- Dynamic language switching

**Success Criteria:**
- All UI text translatable
- Language changes without restart
- Date/number formatting respects locale
- RTL language support basic framework
- ✅ All existing tests pass
- ✅ Tests added for translation loading and switching
- ✅ Tests verify all tr() strings are properly extracted

---

### Task 6.2: Comprehensive Testing Suite
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** All previous tasks

**Objective:** Create automated tests and testing documentation.

**Deliverables:**
- Unit tests for core library (Google Test)
- Integration tests for compression workflows
- UI tests (Qt Test framework)
- Platform integration tests (Windows/Linux)
- Performance benchmarks
- Memory leak detection setup
- CI/CD configuration (GitHub Actions)

**Test Coverage:**
- All compression algorithms
- Multi-volume archives
- Error conditions
- Platform-specific features
- Performance regression tests

**Success Criteria:**
- >80% code coverage in core library
- All critical paths tested
- Tests run in CI automatically
- Performance benchmarks tracked
- ✅ Comprehensive test suite created and documented
- ✅ CI/CD pipeline configured and passing
- ✅ Test data generation scripts working

---

### Task 6.3: Documentation and Help System
**Duration:** 1 session  
**Complexity:** Low  
**Dependencies:** All previous tasks

**Objective:** Create comprehensive user and developer documentation.

**Deliverables:**
- User guide (Markdown + PDF)
- In-app help system (Qt Assistant)
- Context-sensitive help (F1 key)
- Developer documentation (Doxygen)
- Build instructions for all platforms
- Troubleshooting guide
- FAQ document

**Success Criteria:**
- User can learn application without external help
- Developers can build from source easily
- API documentation complete
- Help accessible offline
- ✅ All existing tests pass
- ✅ Documentation reviewed and complete
- ✅ Help system tested on both platforms

---

### Task 6.4: Performance Optimization
**Duration:** 1 session  
**Complexity:** Medium  
**Dependencies:** Task 6.2

**Objective:** Profile and optimize application performance.

**Focus Areas:**
- Compression throughput optimization
- Memory usage reduction
- UI responsiveness during operations
- Startup time optimization
- Large file handling (>10GB)
- Multi-volume overhead reduction

**Deliverables:**
- Performance profiling results
- Optimization implementations
- Benchmark comparisons (before/after)
- Memory leak fixes
- Bottleneck documentation

**Success Criteria:**
- 10GB file compresses without UI freeze
- Memory usage <2x file size
- Startup time <2 seconds
- No memory leaks in 24-hour stress test
- ✅ All existing tests pass
- ✅ Performance benchmarks show improvement
- ✅ Profiling results documented

---

### Task 6.5: Final Integration and Release Preparation
**Duration:** 1 session  
**Complexity:** Low  
**Dependencies:** All previous tasks

**Objective:** Final testing, packaging, and release preparation.

**Deliverables:**
- Version number finalization
- Release notes
- Change log
- Windows installer final build
- Linux packages final build
- Code signing (Windows)
- Release announcement draft
- Website/GitHub release preparation

**Success Criteria:**
- All installers build successfully
- Windows code signed
- Linux packages pass lintian checks
- Documentation complete and accurate
- Ready for public release
- ✅ All tests pass on all platforms
- ✅ Final integration testing complete
- ✅ Release checklist verified

---

## LLM Session Prompts

### **Prompt for Task 1.1: Extract Core Library**

```
I have a CUDA compression CLI application in a single main.cu file (2355 lines). I need to refactor it into a reusable core library and separate CLI application for Qt GUI integration.

Context:
- Current file: main.cu contains everything (compression, archive management, CLI)
- Need to create: core library with C++ API for GUI to use
- Platform: Windows and Linux support required
- Keep all existing functionality intact

Please help me:
1. Identify which code should go into the core library vs CLI
2. Create the file structure for core/src/ and core/include/
3. Extract compression functions, archive management, and volume support into separate files
4. Design a clean C++ API (nvcomp_core.hpp) that the GUI will use
5. Create CMakeLists.txt for building core as a shared library

Requirements:
- Core library should have NO iostream/cout (library shouldn't print)
- Provide callback mechanism for progress reporting
- Thread-safe design
- Export proper symbols for shared library (.dll/.so)

Files attached: @main.cu @CMakeLists.txt

Start with analyzing the code and proposing the architecture.
```

---

### **Prompt for Task 1.2: Create C API Wrapper**

```
I have a C++ core library for GPU compression. I need to create a C API wrapper for broader compatibility and potential future language bindings (Python, C#, etc.).

Context:
- Core library is C++/CUDA with std::vector, exceptions, etc.
- Need extern "C" API for ABI stability
- Will be used by Qt GUI (C++) and potentially other languages
- Must handle errors gracefully without exceptions crossing ABI boundary

Please help me:
1. Design a C API header (nvcomp_c_api.h) with:
   - Handle-based API (opaque pointers)
   - Error codes instead of exceptions
   - Progress callbacks
   - Platform-specific DLL export macros
2. Implement the C API wrapper (nvcomp_c_api.cpp) that:
   - Wraps C++ exceptions into error codes
   - Manages object lifetimes safely
   - Provides thread-local error messages
   - Handles progress callbacks

Key functions needed:
- Compression (file/buffer)
- Decompression (file/buffer)
- Archive listing
- GPU availability check
- Volume support

Files attached: @core/include/nvcomp_core.hpp @core/src/nvcomp_core.cu

Provide complete C API design with implementation.
```

---

### **Prompt for Task 1.3: Refactor CLI**

```
I've extracted my compression logic into a core library. Now I need to refactor the CLI (main.cu) to be a thin wrapper around this library instead of containing all the logic.

Context:
- Original main.cu: 2355 lines with all compression logic
- New core library: provides C++ API for all operations
- Need to maintain backward compatibility (same command-line arguments)
- CLI should be simple and focused on user interaction

Please help me:
1. Refactor cli/main.cu to use the core library API
2. Keep all existing CLI arguments and behavior
3. Remove all compression logic (now in core)
4. Add proper error handling and user feedback
5. Update cli/CMakeLists.txt to link against core library

The CLI should handle:
- Argument parsing
- File path resolution
- User-facing error messages
- Progress display (using core library callbacks)
- Exit codes

Files attached: @main.cu @core/include/nvcomp_core.hpp @CMakeLists.txt

Make the CLI clean and maintainable (target: <500 lines).
```

---

### **Prompt for Task 1.4: Update Root CMake**

```
I'm restructuring my project into core library, CLI, and (future) GUI. I need a robust root CMakeLists.txt that handles all build configurations.

Current structure:
```
nvCOMP_Project/
├── CMakeLists.txt (root - needs update)
├── core/
│   ├── CMakeLists.txt (builds shared library)
│   └── src/
├── cli/
│   ├── CMakeLists.txt (builds CLI exe)
│   └── main.cu
└── gui/ (future)
```

Please help me create a root CMakeLists.txt that:
1. Provides options: BUILD_CLI, BUILD_GUI, BUILD_TESTS
2. Finds and configures dependencies (CUDA, nvCOMP, LZ4, Snappy, Zstd)
3. Handles platform differences (Windows DLL paths, Linux RPATH)
4. Sets up proper installation targets
5. Configures Qt6 when BUILD_GUI=ON (for future)
6. Supports both Debug and Release builds
7. Generates proper export targets for the core library

Platform-specific requirements:
- Windows: Copy DLLs to executable directory
- Linux: Set RPATH to find .so files
- Both: Install to standard system locations

Files attached: @CMakeLists.txt @core/CMakeLists.txt @cli/CMakeLists.txt

Create a professional, maintainable build system.
```

---

### **Prompt for Task 2.1: Set Up Qt Project**

```
I'm adding a Qt 6 GUI to my CUDA compression project. I need to set up the initial Qt project structure and get a basic window displaying.

Context:
- Existing C++/CUDA core library (already built)
- First time adding Qt to this project
- Need cross-platform support (Windows + Linux)
- Using Qt 6 (not Qt 5)

Please help me:
1. Create gui/CMakeLists.txt with Qt 6 configuration
2. Create gui/src/main.cpp - Qt application entry point
3. Create gui/src/mainwindow.h/cpp - Main window class
4. Create gui/ui/mainwindow.ui - Qt Designer UI file
5. Set up resources (gui/resources/nvcomp.qrc) for icons
6. Create a basic application icon
7. Update root CMakeLists.txt to include gui/ subdirectory when BUILD_GUI=ON

Requirements:
- Use Qt Widgets (not QML)
- Modern C++ (C++17)
- Link against core library
- Window should display "nvCOMP Compressor" with basic menu bar
- Application icon should appear in taskbar/titlebar

Files attached: @CMakeLists.txt @core/include/nvcomp_core.hpp

Provide complete initial Qt setup.
```

---

### **Prompt for Task 2.2: Implement Main Window UI**

```
I need to design and implement the main window UI for my GPU compression application using Qt 6.

Application purpose: Compress/decompress files and folders using GPU-accelerated algorithms (LZ4, Snappy, Zstd, GDeflate, ANS, Bitcomp).

Please help me create a professional, intuitive UI with:

1. **Input/Output Area:**
   - File/folder selection (button + drag-drop zone)
   - Display selected file/folder path
   - Output path selection
   - QListWidget to show files (especially for folders)

2. **Compression Settings:**
   - Algorithm selector (QComboBox) with GPU/CPU indicators
   - Volume size input with unit selector (MB/GB)
   - "No volumes" checkbox
   - "Force CPU" checkbox
   - Settings button (opens dialog - to be implemented later)

3. **Action Buttons:**
   - Compress button (large, prominent)
   - Decompress button
   - View Archive button (shows contents without extracting)
   - Cancel button (for running operations)

4. **Status Area:**
   - Progress bar (QProgressBar)
   - Status text (current operation, speed, ETA)
   - GPU availability indicator
   - Statistics (ratio, file size, time)

5. **Menu Bar:**
   - File → Open, Exit
   - Tools → Settings, GPU Monitor
   - Help → About, Documentation

Design requirements:
- Professional appearance (modern, clean)
- Intuitive layout (logical flow)
- Responsive (resizes properly)
- Tooltips on all controls
- Disabled states for unavailable options

Files attached: @gui/src/mainwindow.h @gui/ui/mainwindow.ui

Provide both the .ui file (XML) and the mainwindow.h/cpp implementation with all slots.
```

---

### **Prompt for Task 2.3: Create Compression Worker**

```
I need to implement a QThread-based worker class to run compression operations in the background without freezing the Qt GUI.

Context:
- GUI has compress/decompress buttons
- Core library provides synchronous compression API
- Need to keep UI responsive during long operations (multi-GB files)
- Progress updates should flow to UI smoothly

Please help me create:

1. **CompressionWorker class (QThread-based):**
   - Header: gui/src/compression_worker.h
   - Implementation: gui/src/compression_worker.cpp

2. **Features:**
   - Compress file/folder in separate thread
   - Decompress archive in separate thread
   - Call core library API safely from thread
   - Emit signals for progress (percentage, speed, ETA)
   - Emit signals for completion/error
   - Support cancellation (stop flag)
   - Handle all algorithm types

3. **Signals to emit:**
   - progressUpdated(int percent, QString status)
   - speedUpdated(double mbPerSec)
   - etaUpdated(int secondsRemaining)
   - compressionComplete(QString resultPath, qint64 originalSize, qint64 compressedSize)
   - decompressionComplete(QString resultPath)
   - errorOccurred(QString errorMessage)

4. **Integration:**
   - How to create and start worker from MainWindow
   - How to connect signals to UI updates
   - Proper cleanup and thread management
   - Avoiding race conditions

Files attached: @core/include/nvcomp_core.hpp @gui/src/mainwindow.h

Provide complete worker implementation with usage example.
```

---

### **Prompt for Task 2.4: Implement Compress/Decompress**

```
I have a Qt GUI with UI controls and a background worker thread. Now I need to wire everything together to actually compress and decompress files.

Current state:
- MainWindow UI is complete (buttons, progress bar, etc.)
- CompressionWorker class runs operations in background
- Core library provides compression API

Please help me implement:

1. **Compress functionality:**
   - Slot: void MainWindow::onCompressClicked()
   - Show file picker (QFileDialog) or use selected file
   - Validate input (file/folder exists)
   - Show output path dialog with suggested name (e.g., input.lz4)
   - Get algorithm from UI combo box
   - Get settings (volume size, CPU mode)
   - Create and start CompressionWorker
   - Connect worker signals to UI updates
   - Disable UI during operation
   - Show completion message with statistics

2. **Decompress functionality:**
   - Slot: void MainWindow::onDecompressClicked()
   - Show file picker for compressed archive
   - Auto-detect algorithm (from file)
   - Show folder picker for extraction
   - Detect multi-volume archives
   - Create and start CompressionWorker
   - Update UI with progress
   - Show completion message

3. **Progress updates:**
   - Update QProgressBar from worker signals
   - Show current status (compressing, decompressing)
   - Display speed (MB/s or GB/s)
   - Display ETA (estimated time remaining)
   - Show final statistics (compression ratio, time, sizes)

4. **Error handling:**
   - Catch worker errors
   - Show user-friendly QMessageBox
   - Re-enable UI after error
   - Log errors to console/file

5. **Cancellation:**
   - Cancel button stops current operation
   - Worker cleans up gracefully
   - UI returns to ready state

Files attached: @gui/src/mainwindow.h @gui/src/compression_worker.h

Provide complete implementation of compress/decompress workflow.
```

---

### **Prompt for Task 3.1: Implement Archive Viewer**

```
I need to create a dialog that displays the contents of a compressed archive without extracting it, similar to how WinRAR or 7-Zip works.

Requirements:

1. **ArchiveViewerDialog class:**
   - Header: gui/src/archive_viewer.h
   - Implementation: gui/src/archive_viewer.cpp
   - UI file: gui/ui/archive_viewer.ui

2. **Features:**
   - Display archive contents in tree view (QTreeWidget)
   - Show file path, size, and compression ratio
   - Support multi-volume archives
   - Sort by name, size, date (if available)
   - Search/filter files
   - Select multiple files/folders
   - "Extract Selected" button
   - "Extract All" button
   - Total archive statistics (file count, total size, compressed size)

3. **Tree structure:**
   - Root shows archive name
   - Folders are expandable nodes
   - Files are leaf nodes
   - Icons for folders and files
   - Right-click context menu (extract, properties)

4. **Integration:**
   - Call from MainWindow: Tools → View Archive
   - Can also open by double-clicking archive in file list
   - Use core library to read archive manifest
   - Handle errors (corrupted archives, missing volumes)

5. **Performance:**
   - Load large archives efficiently (1000+ files)
   - Don't block UI while reading archive
   - Progress indicator for loading

Files attached: @core/include/nvcomp_core.hpp @gui/src/mainwindow.h

Provide complete archive viewer implementation with UI.
```

---

### **Prompt for Task 3.2: Create Settings Dialog**

```
I need a settings dialog for user preferences in my Qt compression application. Settings should persist between sessions.

Requirements:

1. **SettingsDialog class:**
   - Header: gui/src/settings_dialog.h
   - Implementation: gui/src/settings_dialog.cpp
   - UI file: gui/ui/settings_dialog.ui

2. **Settings categories (use QTabWidget):**

   **Tab 1: Compression**
   - Default algorithm (combo box)
   - Default volume size (spinbox + unit combo)
   - Default compression level (if applicable)
   - Enable/disable multi-volume by default
   - Output path template (e.g., "{filename}.{ext}")

   **Tab 2: Performance**
   - Prefer GPU/CPU (radio buttons)
   - VRAM usage limit (slider)
   - Thread count for CPU mode
   - Chunk size for batched operations

   **Tab 3: Interface**
   - Language selection (English, Spanish, etc.)
   - Theme (Light/Dark/System)
   - Show advanced options
   - Confirm before overwriting files
   - Show statistics after compression

   **Tab 4: Integration**
   - Enable context menu (Windows/Linux)
   - File associations
   - Start with system (checkbox)

3. **Persistence:**
   - Use QSettings to save preferences
   - Load settings on application start
   - Apply changes immediately (or on OK button)
   - Restore defaults button

4. **Validation:**
   - Validate volume size (min/max)
   - Check for valid output templates
   - Alert user to invalid settings

5. **Integration:**
   - Open from MainWindow: Tools → Settings
   - Apply settings to compression operations
   - Update MainWindow UI based on settings

Files attached: @gui/src/mainwindow.h

Provide complete settings dialog with QSettings integration.
```

---

### **Prompt for Task 3.3: Add GPU Monitoring Widget**

```
I need a widget to display real-time GPU status and memory usage for my CUDA compression application.

Requirements:

1. **GPUMonitorWidget class:**
   - Can be a QWidget or QDialog
   - Header: gui/src/gpu_monitor.h
   - Implementation: gui/src/gpu_monitor.cpp

2. **Display information:**
   - GPU name and model
   - CUDA driver version
   - Total VRAM
   - Available VRAM (updates in real-time)
   - VRAM usage percentage (progress bar)
   - GPU temperature (if available via NVML)
   - Warning when VRAM insufficient for current operation

3. **Real-time updates:**
   - Use QTimer to poll GPU status every 500ms
   - Update progress bar and labels
   - Color-code based on usage (green/yellow/red)
   - Show warning icon when <10% VRAM available

4. **CUDA integration:**
   - Use cudaGetDeviceProperties() for static info
   - Use cudaMemGetInfo() for VRAM usage
   - Optionally use NVML for temperature
   - Handle cases where no GPU is available

5. **Features:**
   - "Refresh" button to update immediately
   - Display multiple GPUs if available (tabs or list)
   - Predict VRAM needed for current file/volume size
   - Warning dialog if user tries to compress file too large for VRAM

6. **Integration:**
   - Open from MainWindow: Tools → GPU Monitor
   - Show warning icon in MainWindow when GPU at capacity
   - Embedded mini-version in MainWindow status bar

Files attached: @core/include/nvcomp_core.hpp @gui/src/mainwindow.h

Provide GPU monitoring implementation with CUDA integration.
```

---

### **Prompt for Task 3.4: Implement Batch Operations**

```
I need to add batch processing capability to compress/decompress multiple files in sequence.

Requirements:

1. **Queue system:**
   - QueueManager class (manages list of operations)
   - Each queue item: input path, output path, operation type, settings
   - Items processed sequentially (one at a time)
   - Can pause/resume/cancel queue
   - Can reorder queue items (move up/down)
   - Can remove items from queue

2. **UI integration:**
   - "Add to Queue" button alongside Compress/Decompress
   - Queue panel/dialog showing all queued operations
   - Per-item progress and overall queue progress
   - Status for each item: Waiting, In Progress, Complete, Failed
   - "Start Queue" button
   - "Pause Queue" and "Clear Queue" buttons

3. **Features:**
   - Multiple files can be added via drag-drop
   - Batch compress entire folder (each file separately)
   - Option to preserve folder structure in output
   - Option to compress folder as single archive
   - Summary report at completion (CSV export)
   - Notifications when queue completes

4. **Statistics tracking:**
   - Total files processed
   - Total time elapsed
   - Average compression ratio
   - Total space saved
   - Any errors encountered
   - Export stats to CSV

5. **Integration with CompressionWorker:**
   - Worker processes one queue item at a time
   - Queue manager starts next item after completion
   - Handle errors without stopping entire queue
   - Option: "Stop after current file" vs "Stop immediately"

6. **Persistence:**
   - Save queue on exit (optional)
   - Restore queue on start (optional)
   - Resume interrupted queue

Files attached: @gui/src/mainwindow.h @gui/src/compression_worker.h

Provide complete batch processing implementation.
```

---

### **Prompt for Task 4.1: Windows Context Menu Integration**

```
I need to add "Compress with nvCOMP" to the Windows Explorer right-click context menu for files and folders.

Context:
- Qt application for Windows
- Need registry integration
- Should work for both files and folders
- Installer should register/unregister context menu

Please help me create:

1. **ContextMenuManager class:**
   - Header: platform/windows/context_menu.h
   - Implementation: platform/windows/context_menu.cpp

2. **Registry keys to create:**
   - HKEY_CLASSES_ROOT\*\shell\nvCOMP (for files)
   - HKEY_CLASSES_ROOT\Directory\shell\nvCOMP (for folders)
   - Submenus for different algorithms
   - Icon display in context menu
   - Position relative to other items

3. **Features:**
   - Main item: "Compress with nvCOMP"
   - Submenu items:
     - "Compress here (LZ4)"
     - "Compress here (Zstd)"
     - "Compress and move to..."
     - "Add to archive..."
   - Dynamic menu (show GPU options only if GPU available)
   - Icon next to menu item
   - Cascading menu support

4. **Command-line handling:**
   - GUI app should accept command-line arguments
   - Parse: nvcomp-gui.exe --compress "C:\path\to\file.txt"
   - Parse: nvcomp-gui.exe --context-menu --algorithm lz4 "file.txt"
   - GUI opens with file pre-selected or compresses immediately

5. **Installer integration:**
   - Function to register context menu (called by installer)
   - Function to unregister context menu (called by uninstaller)
   - Check for administrator privileges
   - Graceful failure if can't write registry

6. **Testing:**
   - How to test without installer
   - Manual registry import/export
   - Verify on Windows 10 and 11

Provide complete Windows context menu implementation with registry code.
```

---

### **Prompt for Task 4.2: Windows File Associations**

```
I need to associate compressed file extensions (.lz4, .zstd, etc.) with my Qt application on Windows so double-clicking opens the archive viewer.

Requirements:

1. **FileAssociationManager class:**
   - Header: platform/windows/file_associations.h
   - Implementation: platform/windows/file_associations.cpp

2. **File extensions to register:**
   - .lz4 (LZ4 compressed archive)
   - .zstd (Zstd compressed archive)
   - .snappy (Snappy compressed archive)
   - .nvcomp (generic nvCOMP archive)
   - .gdeflate, .ans, .bitcomp (GPU algorithms)

3. **Registry keys:**
   - Create ProgID: nvCOMP.LZ4Archive, nvCOMP.ZstdArchive, etc.
   - Associate extension with ProgID
   - Set description (friendly name)
   - Set default icon (different for each type)
   - Set default action (Open with nvCOMP)
   - Add "Extract here" context menu
   - Add "Extract to folder" context menu

4. **Custom icons:**
   - Generate icon resources for each archive type
   - Icon shows compression algorithm visually
   - Embed in executable or separate .ico files
   - Icon extraction from executable

5. **Shell integration:**
   - Windows property sheet (shows compression info in Properties dialog)
   - Thumbnail provider (optional: shows file count / compression ratio)
   - Windows Search integration (index archive contents)
   - Preview handler (optional: show archive contents in preview pane)

6. **Advanced features:**
   - Set as default program for extension
   - Check if already associated (don't overwrite user preference)
   - Integration with "Open With" menu
   - File type grouping in Explorer

7. **Management functions:**
   - RegisterAssociation(extension, progId)
   - UnregisterAssociation(extension)
   - IsAssociated(extension)
   - SetAsDefault(extension)

Provide complete file association implementation with icons.
```

---

### **Prompt for Task 4.3: Windows Installer (WiX)**

```
I need to create a professional MSI installer for Windows using WiX Toolset for my Qt/CUDA compression application.

Application components:
- nvcomp-gui.exe (Qt GUI, ~10MB)
- nvcomp-cli.exe (CLI tool, ~5MB)
- nvcomp_core.dll (core library)
- nvcomp64_5.dll, nvcomp_cpu64_5.dll (NVIDIA dependencies)
- Qt6Core.dll, Qt6Widgets.dll, Qt6Gui.dll (Qt dependencies)
- Various resource files (icons, translations)

Requirements:

1. **WiX project files:**
   - platform/windows/installer.wxs (main installer definition)
   - platform/windows/ui.wxs (custom UI dialogs)
   - platform/windows/product.wxi (shared properties)

2. **Installation features:**
   - Install to Program Files by default
   - User can choose installation directory
   - Components:
     - nvCOMP GUI (required)
     - nvCOMP CLI (optional)
     - Context Menu Integration (optional)
     - File Associations (optional)
     - Start Menu Shortcuts (optional)
     - Desktop Shortcut (optional)

3. **Registry integration:**
   - Call ContextMenuManager::Register()
   - Call FileAssociationManager::Register()
   - Add uninstall entry to Programs and Features
   - Store installation path for updater

4. **Prerequisites:**
   - Check for Visual C++ Redistributable
   - Check for .NET Runtime (if needed)
   - Optionally check for CUDA Toolkit (for GPU)
   - Install missing prerequisites automatically

5. **Upgrade logic:**
   - Detect previous version
   - Major upgrade (remove old, install new)
   - Preserve user settings during upgrade
   - Version number scheme

6. **UI customization:**
   - Custom banner and dialog images
   - License agreement (RTF)
   - Component selection dialog
   - Installation progress
   - Finish dialog with "Launch nvCOMP" checkbox

7. **Build process:**
   - CMake integration (generate WiX files)
   - Automatic version number from Git/CMake
   - Code signing integration (SignTool)
   - Build both x64 and x86 (if needed)

8. **Uninstaller:**
   - Clean removal of all files
   - Unregister context menu
   - Unregister file associations
   - Remove registry keys
   - Option to keep user settings

Provide complete WiX installer implementation with build instructions.
```

---

### **Prompt for Task 5.1: Linux Desktop Integration**

```
I need to integrate my Qt compression application with Linux desktop environments (GNOME, KDE, XFCE) following freedesktop.org standards.

Requirements:

1. **DesktopIntegration class:**
   - Header: platform/linux/desktop_integration.h
   - Implementation: platform/linux/desktop_integration.cpp

2. **Desktop entry file (.desktop):**
   - Generate: /usr/share/applications/nvcomp.desktop
   - Or: ~/.local/share/applications/nvcomp.desktop (user install)
   - Fields:
     - Name, GenericName, Comment
     - Exec (command to run)
     - Icon (application icon)
     - Terminal=false
     - Categories (Utility;Archiving;Compression)
     - MimeType (associated file types)
     - Actions (additional actions: Compress, Decompress)

3. **MIME type definitions:**
   - Create: /usr/share/mime/packages/nvcomp.xml
   - Or: ~/.local/share/mime/packages/nvcomp.xml
   - Define MIME types:
     - application/x-lz4
     - application/x-zstd
     - application/x-snappy
     - application/x-nvcomp
   - Add glob patterns, magic numbers, descriptions
   - Set icons for each MIME type

4. **Icon installation:**
   - Install icons in multiple sizes: 16x16, 32x32, 48x48, 64x64, 128x128, 256x256
   - Path: /usr/share/icons/hicolor/[size]/apps/nvcomp.png
   - Or: ~/.local/share/icons/hicolor/[size]/apps/nvcomp.png
   - Create SVG icon (scalable)
   - MIME type icons: nvcomp-lz4, nvcomp-zstd, etc.

5. **Implementation:**
   - Install() method: Copies files to appropriate locations
   - Uninstall() method: Removes files
   - IsInstalled() method: Checks if already integrated
   - UpdateDesktopDatabase() method: Runs update-desktop-database
   - UpdateMimeDatabase() method: Runs update-mime-database
   - Handle both system-wide and user-only installation

6. **XDG utilities:**
   - Use xdg-utils commands where appropriate
   - xdg-mime: Set default applications
   - xdg-icon-resource: Install icons
   - xdg-desktop-menu: Install desktop files

7. **Testing:**
   - Verify in GNOME (Nautilus)
   - Verify in KDE (Dolphin)
   - Verify in XFCE (Thunar)
   - Check icon display
   - Check MIME type recognition
   - Check default application

Files attached: @gui/resources/ (for icons)

Provide complete Linux desktop integration implementation.
```

---

### **Prompt for Task 5.2: Nautilus/Nemo Context Menu**

```
I need to add right-click context menu integration for Nautilus (GNOME) and Nemo (Cinnamon) file managers to compress files/folders.

Requirements:

1. **Nautilus Python extension:**
   - File: platform/linux/nautilus_extension.py
   - Uses python-nautilus bindings
   - Install to: ~/.local/share/nautilus-python/extensions/
   - Or: /usr/share/nautilus-python/extensions/

2. **Nemo script:**
   - File: platform/linux/nemo_script.sh
   - Simple bash script calling our application
   - Install to: ~/.local/share/nemo/scripts/
   - Or: /usr/share/nemo/scripts/

3. **Context menu items:**
   - Main item: "Compress with nvCOMP"
   - Submenu items:
     - "Compress with LZ4"
     - "Compress with Zstd"
     - "Compress with Snappy"
     - "Compress (GPU)..." (if GPU available)
     - "Settings..."
   - Only show for valid file types
   - Support multiple selection

4. **Nautilus extension features:**
   - Inherit from Nautilus.MenuProvider
   - Implement get_file_items() or get_background_items()
   - Check CUDA availability (show/hide GPU options)
   - Pass selected files to GUI application
   - Environment variables: NAUTILUS_SCRIPT_SELECTED_FILE_PATHS

5. **Integration with GUI:**
   - Launch: nvcomp-gui --compress --algorithm lz4 "/path/file"
   - Handle multiple files: nvcomp-gui --batch-compress "file1" "file2"
   - Show progress notification (libnotify)
   - Handle errors gracefully

6. **Installation:**
   - Installer copies extension files
   - Restarts Nautilus: nautilus -q
   - Checks for python-nautilus dependency
   - Provides manual installation instructions

7. **Compatibility:**
   - Test on Ubuntu 20.04, 22.04, 24.04
   - Test on Fedora
   - Test on Linux Mint (Nemo)
   - Handle different Python versions (3.6+)

8. **Decompression integration:**
   - Add "Extract here" for .lz4, .zstd files
   - Add "Extract to folder" (creates subdirectory)
   - Add "View archive" (opens archive viewer)

Provide complete Nautilus extension and Nemo script with installation code.
```

---

### **Prompt for Task 5.3: Debian/Ubuntu Package**

```
I need to create a .deb package for my Qt/CUDA compression application for Ubuntu/Debian Linux.

Application details:
- Qt 6 GUI application
- CUDA-enabled (optional, falls back to CPU)
- Dependencies: Qt6, CUDA runtime (optional), standard C++ libraries
- Size: ~50MB installed

Requirements:

1. **Debian package structure:**
   ```
   platform/linux/debian/
   ├── control           (package metadata)
   ├── rules             (build rules)
   ├── changelog         (version history)
   ├── copyright         (license info)
   ├── postinst          (post-installation script)
   ├── prerm             (pre-removal script)
   ├── postrm            (post-removal script)
   └── nvcomp-gui.install (file installation list)
   ```

2. **control file:**
   - Package name: nvcomp-gui
   - Version: 1.0.0
   - Architecture: amd64
   - Depends: libqt6core6, libqt6widgets6, libqt6gui6, libc6, libstdc++6
   - Recommends: nvidia-cuda-toolkit (for GPU support)
   - Description: GPU-accelerated compression tool
   - Homepage, Maintainer, Section, Priority

3. **Installation layout:**
   ```
   /usr/
   ├── bin/
   │   ├── nvcomp-gui
   │   └── nvcomp-cli
   ├── lib/
   │   ├── libnvcomp_core.so
   │   └── nvcomp/  (private libraries)
   └── share/
       ├── applications/nvcomp.desktop
       ├── icons/hicolor/.../nvcomp.png
       ├── mime/packages/nvcomp.xml
       ├── doc/nvcomp-gui/
       │   ├── README.md
       │   ├── copyright
       │   └── changelog.gz
       └── man/man1/
           ├── nvcomp-gui.1.gz
           └── nvcomp-cli.1.gz
   ```

4. **postinst script:**
   - Update desktop database
   - Update MIME database
   - Update icon cache
   - Install Nautilus extension (if available)
   - Set up file associations
   - Print success message with GPU status

5. **prerm/postrm scripts:**
   - Unregister file associations
   - Remove Nautilus extension
   - Clean up configuration (optional)
   - Update databases

6. **Build process:**
   - Use debhelper (dh)
   - CMake integration
   - Automatic dependency detection (shlibs)
   - Lintian checks (no errors/warnings)
   - GPG signing

7. **Multiple package strategy:**
   - nvcomp-gui (GUI application)
   - nvcomp-cli (CLI only, lighter)
   - nvcomp-dev (development headers)
   - All depend on libvcomp-core

8. **Repository setup:**
   - PPA structure (for Ubuntu)
   - Add GPG key
   - Sources.list entry
   - apt update integration

9. **Version compatibility:**
   - Build for Ubuntu 20.04 LTS
   - Build for Ubuntu 22.04 LTS
   - Build for Ubuntu 24.04 LTS
   - Build for Debian 11, 12

Provide complete Debian packaging with build and testing instructions.
```

---

### **Prompt for Task 5.4: AppImage Creation**

```
I need to create a portable AppImage for my Qt/CUDA compression application that works across many Linux distributions.

Requirements:

1. **AppImage structure:**
   - Self-contained executable
   - Bundles all dependencies (Qt, core library)
   - No system installation required
   - Optional desktop integration

2. **Build script:**
   - File: platform/linux/build-appimage.sh
   - Uses linuxdeploy or appimagetool
   - Bundles Qt6 libraries
   - Bundles core library and dependencies
   - Optionally bundles CUDA runtime
   - Creates .AppImage file

3. **AppDir structure:**
   ```
   nvcomp.AppDir/
   ├── AppRun              (entry script)
   ├── nvcomp-gui.desktop  (desktop file)
   ├── nvcomp.png          (icon)
   ├── usr/
   │   ├── bin/
   │   │   ├── nvcomp-gui
   │   │   └── nvcomp-cli
   │   └── lib/
   │       ├── libnvcomp_core.so
   │       ├── libQt6Core.so.6
   │       ├── libQt6Widgets.so.6
   │       └── ... (other dependencies)
   ```

4. **Dependency handling:**
   - Bundle Qt libraries (not system Qt)
   - Bundle C++ runtime
   - Check CUDA availability at runtime
   - If no CUDA: disable GPU options, use CPU mode
   - If CUDA: verify version compatibility
   - Use AppImage update information

5. **Desktop integration:**
   - First-run prompt: "Integrate with desktop?"
   - Uses appimaged or libappimage
   - Adds to application menu
   - Sets file associations
   - Can be reversed (unintegrate)

6. **Features:**
   - Works on glibc 2.27+ (Ubuntu 18.04+)
   - Supports both x86_64 and aarch64 (if needed)
   - Self-update capability (via AppImageUpdate)
   - Portable settings (stored in ~/.config/nvcomp/ or beside AppImage)

7. **Build dependencies:**
   - linuxdeploy
   - linuxdeploy-plugin-qt
   - appimagetool
   - CMake build of application

8. **Testing:**
   - Test on Ubuntu 20.04, 22.04, 24.04
   - Test on Fedora 38+
   - Test on Arch Linux
   - Test on Linux Mint
   - Test on elementary OS

9. **CI/CD integration:**
   - GitHub Actions workflow
   - Build AppImage on every release
   - Upload to GitHub Releases
   - Automatic update information

Provide complete AppImage build script and instructions.
```

---

### **Prompt for Task 6.1: Internationalization**

```
I need to prepare my Qt compression application for translation into multiple languages using Qt's internationalization (i18n) system.

Current state:
- Qt GUI application with English text
- Many hard-coded strings in C++ code
- Need to support at least: English, Spanish, Chinese, French, German

Requirements:

1. **Code preparation:**
   - Wrap all user-facing strings with tr()
   - Properly handle singular/plural forms
   - Use tr() with context where needed
   - Handle dynamic strings (file names, sizes, etc.)
   - Avoid concatenating translated strings

2. **Translation files:**
   - Extract strings: lupdate
   - Create .ts files: gui/translations/nvcomp_*.ts
   - Languages: en_US, es_ES, zh_CN, fr_FR, de_DE
   - Compile to .qm files: lrelease
   - Embed in resources or install separately

3. **Translation workflow:**
   - How to extract new strings
   - How translators work with .ts files
   - How to test translations
   - How to add new language

4. **UI components to translate:**
   - All button text
   - Labels and tooltips
   - Menu items
   - Dialog titles and messages
   - Error messages
   - Status messages
   - File dialog filters

5. **Special cases:**
   - Date and time formatting (QLocale)
   - Number formatting (sizes, ratios)
   - File sizes (MB, GB - localized)
   - Plurals: "1 file" vs "5 files"
   - Context-specific translations

6. **Language selector:**
   - Add to Settings dialog
   - Dropdown with language names (in their own language)
   - Apply without restart (retranslateUi())
   - Save preference in QSettings
   - Detect system language on first run

7. **Implementation:**
   - Create TranslationManager class
   - Load appropriate .qm file
   - Install translators on QApplication
   - Support fallback to English
   - Handle missing translations gracefully

8. **CMake integration:**
   - Find Qt6::LinguistTools
   - Add lupdate and lrelease commands
   - Generate .qm files during build
   - Install .qm files or embed in resources

9. **Testing:**
   - Test all languages
   - Verify layout doesn't break (German text is longer)
   - Check right-to-left languages (if supported)
   - Verify string replacements work (%1, %2)

10. **Example translations:**
    - Provide English (source)
    - Provide Spanish translation (for demonstration)
    - Template for other languages

Provide complete i18n implementation with example translations.
```

---

### **Prompt for Task 6.2: Comprehensive Testing Suite**

```
I need to create a comprehensive testing suite for my Qt/CUDA compression application to ensure reliability and catch regressions.

Application components:
- Core library (C++/CUDA compression logic)
- CLI application
- Qt GUI application
- Platform integrations (Windows/Linux)

Requirements:

1. **Testing framework setup:**
   - Google Test for C++ unit tests
   - Qt Test for GUI testing
   - CMake integration (enable with BUILD_TESTS=ON)
   - Separate test executables

2. **Core library tests:**
   - File: tests/core/test_compression.cpp
   - Tests for each algorithm: LZ4, Snappy, Zstd, GDeflate, ANS, Bitcomp
   - Test compress + decompress = original
   - Test various data sizes: 1KB, 1MB, 100MB, 1GB
   - Test with random data, compressible data, incompressible data
   - Test multi-volume archives
   - Test error conditions (corrupted data, missing volumes)
   - Test CPU fallback
   - Memory leak detection (Valgrind/ASAN)

3. **Archive tests:**
   - File: tests/core/test_archive.cpp
   - Create archive from files
   - Extract archive
   - Verify files match after extract
   - Test with nested directories
   - Test with special characters in filenames
   - Test with large files (>2GB)
   - Test archive listing

4. **CLI tests:**
   - File: tests/cli/test_cli.cpp
   - Test command-line argument parsing
   - Test compress operations
   - Test decompress operations
   - Test list operations
   - Test error handling
   - Verify exit codes
   - Test batch operations

5. **GUI tests (Qt Test):**
   - File: tests/gui/test_mainwindow.cpp
   - Test window initialization
   - Test button clicks (mock operations)
   - Test file selection dialogs (mock)
   - Test progress updates
   - Test error message display
   - Test settings persistence
   - Verify UI state transitions

6. **Integration tests:**
   - File: tests/integration/test_workflows.cpp
   - End-to-end compress/decompress workflow
   - Multi-volume workflow
   - Archive viewer workflow
   - Batch operations workflow
   - Settings changes applied correctly

7. **Performance benchmarks:**
   - File: tests/benchmarks/benchmark_compression.cpp
   - Measure compression speed (MB/s)
   - Compare CPU vs GPU performance
   - Compare different algorithms
   - Memory usage profiling
   - Regression detection (flag if 10% slower)

8. **Platform-specific tests:**
   - Windows: Context menu registration
   - Windows: File associations
   - Linux: Desktop integration
   - Linux: MIME type handling

9. **CMake test configuration:**
   ```cmake
   if(BUILD_TESTS)
       enable_testing()
       add_subdirectory(tests)
       # Add test targets
       # Add valgrind/ASAN options
   endif()
   ```

10. **CI/CD integration:**
    - GitHub Actions workflow
    - Run tests on push/PR
    - Test on Windows and Linux
    - Code coverage reporting (gcov/lcov)
    - Fail build if tests fail

11. **Test data:**
    - Small test files in repository
    - Script to generate various test data
    - Pre-compressed archives for decompression tests

12. **Documentation:**
    - How to run tests locally
    - How to add new tests
    - Coverage requirements
    - Performance benchmarks baseline

Provide complete testing suite with build configuration and example tests.
```

---

### **Prompt for Task 6.3: Documentation and Help System**

```
I need comprehensive documentation for my Qt/CUDA compression application covering user guides, developer documentation, and in-app help.

Requirements:

1. **User documentation:**
   - File: docs/USER_GUIDE.md
   - Getting started
   - Installation instructions (Windows/Linux)
   - Basic usage (compress, decompress)
   - Advanced features (multi-volume, batch operations)
   - Settings and configuration
   - Troubleshooting
   - FAQ
   - Supported formats

2. **Developer documentation:**
   - File: docs/DEVELOPER.md
   - Architecture overview
   - Building from source (all platforms)
   - Code structure
   - Core library API
   - Adding new compression algorithms
   - Contributing guidelines
   - Testing procedures

3. **API documentation (Doxygen):**
   - Doxygen configuration: docs/Doxyfile
   - Document all public API functions
   - Document core library classes
   - Generate HTML documentation
   - Generate PDF (optional)
   - Include examples and code snippets

4. **In-app help system:**
   - Qt Assistant integration
   - Help menu → Documentation
   - Context-sensitive help (F1 key)
   - What's This mode (Shift+F1)
   - Tooltips on all controls
   - Status bar tips

5. **Help content (.qhp files):**
   - File: docs/help/nvcomp.qhp
   - Table of contents
   - Index
   - Search capability
   - Screenshots and diagrams
   - Step-by-step tutorials

6. **Build documentation:**
   - File: docs/BUILD.md
   - Prerequisites for each platform
   - CMake configuration options
   - Dependency installation
   - Building core library
   - Building CLI
   - Building GUI
   - Creating installers/packages
   - Troubleshooting build issues

7. **Platform-specific guides:**
   - File: docs/WINDOWS.md (Windows-specific features)
   - File: docs/LINUX.md (Linux-specific features)
   - Context menu usage
   - File associations
   - Integration with desktop environment

8. **README.md update:**
   - Project overview
   - Features
   - Screenshots
   - Quick start
   - Installation
   - Build badges (CI status)
   - License
   - Credits

9. **Man pages (Linux):**
   - File: docs/man/nvcomp-gui.1
   - File: docs/man/nvcomp-cli.1
   - Standard Unix man page format
   - Install to /usr/share/man/man1/

10. **Video tutorials (scripts):**
    - Basic compression tutorial
    - Multi-volume archives tutorial
    - Batch operations tutorial
    - Settings and preferences

11. **Help implementation:**
    - HelpManager class
    - Open documentation from menu
    - Open specific help page by ID
    - Online help fallback (if local docs missing)

12. **Accessibility documentation:**
    - Keyboard shortcuts
    - Screen reader compatibility
    - High contrast mode

Provide complete documentation structure with example content for each section.
```

---

### **Prompt for Task 6.4: Performance Optimization**

```
I need to profile and optimize my Qt/CUDA compression application for better performance, lower memory usage, and improved user experience.

Current state:
- Functional compression/decompression
- Some performance issues with large files (>10GB)
- Occasional UI freezing
- Memory usage higher than expected

Requirements:

1. **Performance profiling:**
   - Profile with CUDA Nsight Systems
   - Profile with gprof/perf (CPU code)
   - Memory profiling with Valgrind/Heaptrack
   - Qt Creator profiler
   - Identify bottlenecks

2. **GPU optimization:**
   - Analyze CUDA kernel performance
   - Optimize memory transfers (minimize CPU↔GPU copies)
   - Use pinned memory for faster transfers
   - Stream overlapping (transfer while computing)
   - Optimize chunk sizes
   - Batch operations more efficiently

3. **CPU optimization:**
   - Parallelize CPU compression (OpenMP or std::thread)
   - Optimize file I/O (buffered reads/writes)
   - Reduce memory allocations
   - Use move semantics
   - Optimize archive parsing

4. **Memory optimization:**
   - Reduce peak memory usage
   - Stream large files instead of loading entirely
   - Reuse buffers
   - Fix memory leaks (run Valgrind)
   - Implement memory usage limits
   - Monitor VRAM usage

5. **UI responsiveness:**
   - Ensure all long operations in worker threads
   - Reduce main thread blocking
   - Optimize progress updates (don't update every byte)
   - Lazy loading in archive viewer
   - Implement cancellation properly

6. **File I/O optimization:**
   - Use memory-mapped files for large files
   - Optimize buffer sizes (test 4KB, 64KB, 1MB, 4MB)
   - Async I/O where possible
   - Minimize disk seeks

7. **Startup time optimization:**
   - Lazy load Qt plugins
   - Defer non-critical initialization
   - Cache settings
   - Optimize icon loading

8. **Multi-volume optimization:**
   - Reduce overhead between volumes
   - Optimize manifest size
   - Parallel volume processing (if independent)

9. **Benchmarking:**
   - Create benchmark suite
   - Test with various file sizes: 1MB, 100MB, 1GB, 10GB
   - Test with various data types: text, binary, images, video
   - Compare before/after optimization
   - Track regressions

10. **Optimization targets:**
    - Compress 1GB file: <10 seconds (GPU), <30 seconds (CPU)
    - Decompress 1GB file: <8 seconds (GPU), <25 seconds (CPU)
    - Memory usage: <2x file size
    - UI freeze: <100ms any operation
    - Startup time: <2 seconds

11. **Implementation:**
    - Document optimizations made
    - Add performance tests
    - Create performance monitoring dashboard
    - Set up performance regression CI

Provide performance analysis, optimization implementation, and benchmarking code.
```

---

### **Prompt for Task 6.5: Final Integration and Release**

```
I'm ready to release v1.0 of my Qt/CUDA compression application. I need to finalize everything for public release on Windows and Linux.

Current state:
- All features implemented
- Testing complete
- Documentation written
- Installers/packages ready

Requirements:

1. **Version management:**
   - Finalize version number: v1.0.0
   - Update CMakeLists.txt version
   - Update package versions (deb, MSI)
   - Update About dialog version
   - Tag Git repository: git tag v1.0.0

2. **Release notes:**
   - File: RELEASE_NOTES.md
   - New features
   - Bug fixes
   - Known issues
   - System requirements
   - Upgrade notes

3. **Change log:**
   - File: CHANGELOG.md
   - Follow Keep a Changelog format
   - List all changes since last release
   - Categorize: Added, Changed, Fixed, Removed

4. **Final testing:**
   - Test all installers
   - Test on clean VMs (no dev tools)
   - Windows: Test on Windows 10 and 11
   - Linux: Test on Ubuntu 20.04, 22.04, 24.04
   - Test with and without GPU
   - Test upgrades from previous version (if applicable)

5. **Code signing:**
   - Windows: Sign .exe and .msi with Authenticode
   - Linux: Sign packages with GPG key
   - Create code signing instructions

6. **Build final packages:**
   - Windows:
     - nvcomp-gui-1.0.0-win64.msi (installer)
     - nvcomp-gui-1.0.0-win64-portable.zip (portable)
   - Linux:
     - nvcomp-gui_1.0.0_amd64.deb (Ubuntu/Debian)
     - nvcomp-gui-1.0.0-x86_64.AppImage (universal)
     - nvcomp-gui-1.0.0.tar.gz (source)

7. **GitHub release:**
   - Create release on GitHub
   - Upload all packages
   - Write release description
   - Include checksums (SHA256)
   - Add screenshots
   - Link to documentation

8. **Website/landing page:**
   - Project website content
   - Download links
   - Screenshots and videos
   - Feature list
   - System requirements
   - Installation instructions

9. **Announcement:**
   - Draft announcement text
   - Where to announce:
     - GitHub
     - Reddit (r/linux, r/programming, r/CUDA)
     - Hacker News
     - Twitter/X
     - LinkedIn
   - Press release template

10. **Distribution channels:**
    - Windows: Consider Microsoft Store submission
    - Linux: Submit to Snap Store, Flathub
    - Submit to package managers (AUR for Arch)
    - List on alternativeto.net

11. **Post-release checklist:**
    - Monitor bug reports
    - Set up issue templates on GitHub
    - Create CONTRIBUTING.md
    - Set up pull request template
    - Create project roadmap for v1.1

12. **Analytics (optional):**
    - Download tracking
    - Usage statistics (opt-in)
    - Crash reporting (opt-in)

13. **Support:**
    - Create support channels (GitHub Issues, Discord, Email)
    - Create support documentation
    - Set up FAQ based on common questions

Provide release checklist, final build scripts, and release announcement template.
```

---

## Execution Strategy

### Recommended Approach:

1. **Sequential execution**: Complete Phase 1 before starting Phase 2
2. **Testing at every step**: Run all tests before and after each task
3. **Test-driven development**: Write tests for new features, ensure they pass
4. **Testing between phases**: Comprehensive testing after each major milestone
5. **Iterative refinement**: Revisit earlier tasks if issues found
6. **Documentation as you go**: Don't save all docs for the end

### Time Estimates:

- **Phase 1** (Core Refactoring): 8-10 LLM sessions
- **Phase 2** (Basic Qt GUI): 8-12 LLM sessions
- **Phase 3** (Advanced Features): 8-10 LLM sessions
- **Phase 4** (Windows Integration): 6-8 LLM sessions
- **Phase 5** (Linux Integration): 8-10 LLM sessions
- **Phase 6** (Polish & Testing): 10-12 LLM sessions

**Total**: ~50-60 LLM sessions over 8-10 weeks

### Success Metrics:

- ✅ **All unit tests pass** on Windows and Linux
- ✅ Core library tests pass (>80% coverage)
- ✅ GUI tests pass (>60% coverage)
- ✅ GUI compiles and runs on both platforms
- ✅ Context menus work on both platforms
- ✅ File associations registered correctly
- ✅ Installers create working installations
- ✅ User documentation complete
- ✅ No critical bugs in final testing
- ✅ CI/CD pipeline passing on all platforms

---

## Notes

- Each prompt assumes the previous tasks are complete
- File paths can be adjusted based on your preferences
- Some tasks may expand into multiple sessions if complex
- Testing should be continuous throughout
- Keep the CLI functional throughout (don't break existing functionality)

---

## Next Steps

1. Review this plan
2. Adjust priorities if needed
3. Start with **Task 1.1** (Extract Core Library)
4. Use the provided prompt for that task
5. Proceed sequentially through the plan

**Good luck with the Qt GUI implementation!** 🚀

