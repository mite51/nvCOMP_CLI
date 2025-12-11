# Multi-Volume Test Suite

This test suite validates the multi-volume splitting and reassembly functionality of nvCOMP CLI.

## Running the Tests

### Windows
```cmd
cd unit_test
test_volume.bat
```

### Linux
```bash
cd unit_test
chmod +x test_volume.sh
./test_volume.sh
```

## Test Coverage (14 Tests)

### 1. Multi-Volume Compression Tests (4 tests)
Tests volume splitting with small volume sizes to force multiple volumes:
- **GPU LZ4** with 10KB volumes
- **GPU Zstd** with 20KB volumes  
- **CPU LZ4** with 15KB volumes
- **CPU Snappy** with 25KB volumes

**Validates:**
- Multiple `.vol001`, `.vol002`, `.vol003` files are created
- Volume naming convention is correct
- Volumes can be decompressed and reassembled
- Original folder structure is preserved

### 2. Single Volume Tests (2 tests)
Tests that volume splitting is NOT applied when not needed:
- **--no-volumes** flag (forces single file)
- **Large volume size** (10GB, larger than input)

**Validates:**
- No `.vol001` suffix added to filename
- Output file is named `output.lz4` (not `output.vol001.lz4`)
- Single file can be decompressed normally
- Folder structure is preserved

### 3. Volume List Tests (2 tests)
Tests the `-l` (list) command with multi-volume archives:
- **LZ4** multi-volume archive listing
- **Zstd** multi-volume archive listing

**Validates:**
- Can list contents of multi-volume archives
- Shows volume information (count, sizes)
- Displays all files across all volumes

### 4. Volume Auto-Detection Tests (2 tests)
Tests algorithm auto-detection with multi-volume archives:
- **LZ4** auto-detection
- **Zstd** auto-detection

**Validates:**
- Can decompress without specifying algorithm
- Automatically detects algorithm from manifest
- Correctly reassembles all volumes
- Folder structure is preserved

### 5. Custom Volume Size Tests (4 tests)
Tests various custom volume sizes:
- **5KB volumes** (very small, creates many volumes)
- **50KB volumes** (larger, fewer volumes)
- Both **LZ4** and **Zstd** algorithms

**Validates:**
- `--volume-size` parameter works correctly
- Different sizes create appropriate number of volumes
- All sizes decompress successfully
- Folder structure is preserved

## Why Small Volume Sizes?

The tests use small volume sizes (5KB - 50KB) instead of the default 2.5GB because:

1. **Sample Data Size**: The `sample_folder` is relatively small (typically < 100KB)
2. **Force Volume Splitting**: Small sizes ensure multiple volumes are created for testing
3. **Test Speed**: Smaller volumes = faster test execution
4. **Predictable Behavior**: Easy to verify the correct number of volumes are created

## Sample Folder Requirements

The tests expect a `sample_folder` directory containing test files. The existing `sample_folder/PineTools.com_files/` directory works perfectly.

**Minimum Requirements:**
- At least 30-50KB of total file content (to force multiple volumes with small sizes)
- Some folder structure to verify preservation after decompression
- Multiple files to validate the archive format

## Expected Test Output

### Successful Test Example:
```
[Test 1] Multi-volume GPU LZ4 (10KB volumes)
  Compressing folder with 10KB volumes...
  Created 5 volumes
  Verifying volume naming (should be .vol001, .vol002, etc.)...
  Decompressing multi-volume archive...
  Verifying decompressed folder structure...
  PASSED (Created 5 volumes, decompressed successfully)
```

### Test Summary Example:
```
========================================
Test Summary
========================================
Total tests: 14
Passed: 14
Failed: 0
========================================

ALL TESTS PASSED!
```

## Naming Convention Verification

The tests specifically verify that:
- **Single files**: `output.lz4` (no volume suffix)
- **Multi-volume**: `output.vol001.lz4`, `output.vol002.lz4`, `output.vol003.lz4`, etc.
- **No legacy formats**: Does NOT create `output_V001.lz4` or other variations

## Integration with Existing Tests

This test suite complements the existing test suites:
- `test.bat/sh` (15 tests): Single-file compression
- `test_folder.bat/sh` (14 tests): Folder compression and auto-detection  
- `test_volume.bat/sh` (14 tests): Multi-volume support ← **NEW**

**Total Coverage: 43 tests**

## Troubleshooting

### All tests fail immediately
- Verify `nvcomp_cli.exe` is built: `cd build && cmake --build . --config Release`
- Check the executable path in the test script

### "sample_folder not found"
- Ensure you're in the `unit_test` directory
- Verify `sample_folder/` exists with files inside

### Volume tests fail but others pass
- Check that `--volume-size` parameter is properly parsed
- Verify volume manifest structures are correct
- Look for "Invalid volume file" errors in output

### Decompression fails
- Ensure all `.vol001`, `.vol002`, `.vol003` files are present
- Check that volume sequence has no gaps
- Verify you're decompressing from `.vol001` file


