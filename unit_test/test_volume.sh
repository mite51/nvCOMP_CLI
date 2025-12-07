#!/bin/bash
# nvCOMP CLI Multi-Volume Tests
# Tests for multi-volume splitting and reassembly

echo "========================================"
echo "nvCOMP CLI Multi-Volume Tests"
echo "========================================"
echo ""

TEST_COUNT=0
PASS_COUNT=0
FAIL_COUNT=0

# Create output directory
mkdir -p output

EXE="../build/nvcomp_cli"
if [ ! -f "$EXE" ]; then
    echo "ERROR: nvcomp_cli not found. Build the project first."
    exit 1
fi

# Verify sample_folder exists
if [ ! -d "sample_folder" ]; then
    echo "ERROR: sample_folder not found. Create sample_folder directory with test files."
    exit 1
fi

# ============================================================================
# Helper Functions
# ============================================================================

run_volume_test() {
    local TEST_NAME="$1"
    local ALGO="$2"
    local VOL_SIZE="$3"
    local FLAGS="$4"
    ((TEST_COUNT++))
    
    echo ""
    echo "[Test $TEST_COUNT] $TEST_NAME"
    echo "  Compressing folder with $VOL_SIZE volumes..."
    
    # Clean up old volume files
    rm -f output/volume_test_*.vol*.$ALGO 2>/dev/null
    
    $EXE -c sample_folder output/volume_test_$ALGO.$ALGO $ALGO --volume-size $VOL_SIZE $FLAGS
    if [ $? -ne 0 ]; then
        echo "  FAILED: Volume compression failed"
        ((FAIL_COUNT++))
        return
    fi
    
    # Check if multiple volumes were created
    VOLUME_COUNT=$(ls output/volume_test_$ALGO.vol*.$ALGO 2>/dev/null | wc -l)
    
    if [ $VOLUME_COUNT -lt 2 ]; then
        echo "  FAILED: Expected multiple volumes, found $VOLUME_COUNT"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  Created $VOLUME_COUNT volumes"
    
    echo "  Verifying volume naming (should be .vol001, .vol002, etc.)..."
    if [ ! -f "output/volume_test_$ALGO.vol001.$ALGO" ]; then
        echo "  FAILED: First volume (.vol001) not found"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  Decompressing multi-volume archive..."
    rm -rf output/volume_restored_$ALGO
    $EXE -d output/volume_test_$ALGO.vol001.$ALGO output/volume_restored_$ALGO $FLAGS
    if [ $? -ne 0 ]; then
        echo "  FAILED: Volume decompression failed"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  Verifying decompressed folder structure..."
    if [ ! -d "output/volume_restored_$ALGO/PineTools.com_files" ]; then
        echo "  FAILED: Folder structure not preserved after volume decompression"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  PASSED (Created $VOLUME_COUNT volumes, decompressed successfully)"
    ((PASS_COUNT++))
}

run_single_volume_test() {
    local TEST_NAME="$1"
    local ALGO="$2"
    local FLAGS_BASE="$3"
    local VOL_FLAG="$4"
    ((TEST_COUNT++))
    
    echo ""
    echo "[Test $TEST_COUNT] $TEST_NAME"
    echo "  Compressing folder WITHOUT volume splitting..."
    
    # Clean up old files
    rm -f output/single_test_$ALGO.* 2>/dev/null
    
    $EXE -c sample_folder output/single_test_$ALGO.$ALGO $ALGO $VOL_FLAG $FLAGS_BASE
    if [ $? -ne 0 ]; then
        echo "  FAILED: Single volume compression failed"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  Verifying single file created (no .vol001 suffix)..."
    if [ -f "output/single_test_$ALGO.vol001.$ALGO" ]; then
        echo "  FAILED: Unexpected volume files created"
        ((FAIL_COUNT++))
        return
    fi
    
    if [ ! -f "output/single_test_$ALGO.$ALGO" ]; then
        echo "  FAILED: Output file not created"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  Decompressing single file..."
    rm -rf output/single_restored_$ALGO
    $EXE -d output/single_test_$ALGO.$ALGO output/single_restored_$ALGO $FLAGS_BASE
    if [ $? -ne 0 ]; then
        echo "  FAILED: Single file decompression failed"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  Verifying folder structure..."
    if [ ! -d "output/single_restored_$ALGO/PineTools.com_files" ]; then
        echo "  FAILED: Folder structure not preserved"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  PASSED (Single file created, no volume splitting)"
    ((PASS_COUNT++))
}

run_volume_list_test() {
    local TEST_NAME="$1"
    local ALGO="$2"
    local VOL_SIZE="$3"
    ((TEST_COUNT++))
    
    echo ""
    echo "[Test $TEST_COUNT] $TEST_NAME"
    echo "  Compressing folder with $VOL_SIZE volumes..."
    
    # Clean up old volume files
    rm -f output/list_vol_test_*.vol*.$ALGO 2>/dev/null
    
    $EXE -c sample_folder output/list_vol_test_$ALGO.$ALGO $ALGO --volume-size $VOL_SIZE
    if [ $? -ne 0 ]; then
        echo "  FAILED: Volume compression failed"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  Listing multi-volume archive..."
    $EXE -l output/list_vol_test_$ALGO.vol001.$ALGO
    if [ $? -ne 0 ]; then
        echo "  FAILED: Volume listing failed"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  PASSED (Multi-volume archive listed successfully)"
    ((PASS_COUNT++))
}

run_volume_autodetect_test() {
    local TEST_NAME="$1"
    local ALGO="$2"
    local VOL_SIZE="$3"
    ((TEST_COUNT++))
    
    echo ""
    echo "[Test $TEST_COUNT] $TEST_NAME"
    echo "  Compressing folder with $VOL_SIZE volumes..."
    
    # Clean up old volume files
    rm -f output/autodetect_vol_*.vol*.$ALGO 2>/dev/null
    
    $EXE -c sample_folder output/autodetect_vol_$ALGO.$ALGO $ALGO --volume-size $VOL_SIZE
    if [ $? -ne 0 ]; then
        echo "  FAILED: Volume compression failed"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  Decompressing WITHOUT algorithm parameter (auto-detect)..."
    rm -rf output/autodetect_vol_restored_$ALGO
    $EXE -d output/autodetect_vol_$ALGO.vol001.$ALGO output/autodetect_vol_restored_$ALGO
    if [ $? -ne 0 ]; then
        echo "  FAILED: Auto-detection decompression failed"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  Verifying folder structure..."
    if [ ! -d "output/autodetect_vol_restored_$ALGO/PineTools.com_files" ]; then
        echo "  FAILED: Folder structure not preserved"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  PASSED (Algorithm auto-detected from multi-volume archive)"
    ((PASS_COUNT++))
}

run_custom_size_test() {
    local TEST_NAME="$1"
    local ALGO="$2"
    local VOL_SIZE="$3"
    ((TEST_COUNT++))
    
    echo ""
    echo "[Test $TEST_COUNT] $TEST_NAME"
    echo "  Compressing with custom volume size $VOL_SIZE..."
    
    # Clean up old volume files
    rm -f output/custom_vol_*.vol*.$ALGO 2>/dev/null
    
    $EXE -c sample_folder output/custom_vol_$ALGO.$ALGO $ALGO --volume-size $VOL_SIZE
    if [ $? -ne 0 ]; then
        echo "  FAILED: Custom size compression failed"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  Decompressing custom-sized volumes..."
    rm -rf output/custom_vol_restored_$ALGO
    $EXE -d output/custom_vol_$ALGO.vol001.$ALGO output/custom_vol_restored_$ALGO
    if [ $? -ne 0 ]; then
        echo "  FAILED: Custom size decompression failed"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  Verifying folder structure..."
    if [ ! -d "output/custom_vol_restored_$ALGO/PineTools.com_files" ]; then
        echo "  FAILED: Folder structure not preserved"
        ((FAIL_COUNT++))
        return
    fi
    
    echo "  PASSED (Custom volume size worked correctly)"
    ((PASS_COUNT++))
}

# ============================================================================
# Multi-Volume Compression Tests
# ============================================================================

echo ""
echo "========================================"
echo "Multi-Volume Compression Tests"
echo "========================================"

run_volume_test "Multi-volume GPU LZ4 (10KB volumes)" "lz4" "10KB" ""
run_volume_test "Multi-volume GPU Zstd (20KB volumes)" "zstd" "20KB" ""
run_volume_test "Multi-volume CPU LZ4 (15KB volumes)" "lz4" "15KB" "--cpu"
run_volume_test "Multi-volume CPU Snappy (25KB volumes)" "snappy" "25KB" "--cpu"

# ============================================================================
# Single Volume Tests (No Split)
# ============================================================================

echo ""
echo "========================================"
echo "Single Volume Tests (No Split)"
echo "========================================"

run_single_volume_test "Single volume GPU LZ4 (--no-volumes)" "lz4" "" "--no-volumes"
run_single_volume_test "Single volume GPU Zstd (large size)" "zstd" "" "--volume-size 10GB"

# ============================================================================
# Volume List and Decompress Tests
# ============================================================================

echo ""
echo "========================================"
echo "Volume List and Decompress Tests"
echo "========================================"

run_volume_list_test "List multi-volume LZ4 archive" "lz4" "10KB"
run_volume_list_test "List multi-volume Zstd archive" "zstd" "20KB"

# ============================================================================
# Volume Auto-Detection Tests
# ============================================================================

echo ""
echo "========================================"
echo "Volume Auto-Detection Tests"
echo "========================================"

run_volume_autodetect_test "Auto-detect multi-volume LZ4" "lz4" "15KB"
run_volume_autodetect_test "Auto-detect multi-volume Zstd" "zstd" "20KB"

# ============================================================================
# Custom Volume Size Tests
# ============================================================================

echo ""
echo "========================================"
echo "Custom Volume Size Tests"
echo "========================================"

run_custom_size_test "Custom 5KB volumes (LZ4)" "lz4" "5KB"
run_custom_size_test "Custom 50KB volumes (Zstd)" "zstd" "50KB"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total tests: $TEST_COUNT"
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "========================================"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "SOME TESTS FAILED!"
    exit 1
else
    echo ""
    echo "ALL TESTS PASSED!"
    exit 0
fi

