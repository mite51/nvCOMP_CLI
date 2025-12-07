@echo off
REM nvCOMP CLI Multi-Volume Tests
REM Tests for multi-volume splitting and reassembly

setlocal enabledelayedexpansion

echo ========================================
echo nvCOMP CLI Multi-Volume Tests
echo ========================================
echo.

set TEST_COUNT=0
set PASS_COUNT=0
set FAIL_COUNT=0

REM Create output directory
if not exist output mkdir output

set EXE=..\build\Release\nvcomp_cli.exe
if not exist %EXE% (
    set EXE=..\build\x64\Release\nvcomp_cli.exe
)
if not exist %EXE% (
    echo ERROR: nvcomp_cli.exe not found. Build the project first.
    exit /b 1
)

REM Verify sample_folder exists
if not exist sample_folder (
    echo ERROR: sample_folder not found. Create sample_folder directory with test files.
    exit /b 1
)

REM ============================================================================
REM Multi-Volume Compression Tests
REM ============================================================================

echo.
echo ========================================
echo Multi-Volume Compression Tests
echo ========================================

call :run_volume_test "Multi-volume GPU LZ4 (10KB volumes)" lz4 "10KB" ""
call :run_volume_test "Multi-volume GPU Zstd (20KB volumes)" zstd "20KB" ""
call :run_volume_test "Multi-volume CPU LZ4 (15KB volumes)" lz4 "15KB" "--cpu"
call :run_volume_test "Multi-volume CPU Snappy (25KB volumes)" snappy "25KB" "--cpu"

REM ============================================================================
REM Single Volume Tests (No Split)
REM ============================================================================

echo.
echo ========================================
echo Single Volume Tests (No Split)
echo ========================================

call :run_single_volume_test "Single volume GPU LZ4 (--no-volumes)" lz4 "" "--no-volumes"
call :run_single_volume_test "Single volume GPU Zstd (large size)" zstd "" "--volume-size 10GB"

REM ============================================================================
REM Volume List and Decompress Tests
REM ============================================================================

echo.
echo ========================================
echo Volume List and Decompress Tests
echo ========================================

call :run_volume_list_test "List multi-volume LZ4 archive" lz4 "10KB"
call :run_volume_list_test "List multi-volume Zstd archive" zstd "20KB"

REM ============================================================================
REM Volume Auto-Detection Tests
REM ============================================================================

echo.
echo ========================================
echo Volume Auto-Detection Tests
echo ========================================

call :run_volume_autodetect_test "Auto-detect multi-volume LZ4" lz4 "15KB"
call :run_volume_autodetect_test "Auto-detect multi-volume Zstd" zstd "20KB"

REM ============================================================================
REM Custom Volume Size Tests
REM ============================================================================

echo.
echo ========================================
echo Custom Volume Size Tests
echo ========================================

call :run_custom_size_test "Custom 5KB volumes (LZ4)" lz4 "5KB"
call :run_custom_size_test "Custom 50KB volumes (Zstd)" zstd "50KB"

REM ============================================================================
REM Summary
REM ============================================================================

echo.
echo ========================================
echo Test Summary
echo ========================================
echo Total tests: %TEST_COUNT%
echo Passed: %PASS_COUNT%
echo Failed: %FAIL_COUNT%
echo ========================================

if %FAIL_COUNT% gtr 0 (
    echo.
    echo SOME TESTS FAILED!
    exit /b 1
) else (
    echo.
    echo ALL TESTS PASSED!
    exit /b 0
)

REM ============================================================================
REM Helper Functions
REM ============================================================================

:run_volume_test
set TEST_NAME=%~1
set ALGO=%~2
set VOL_SIZE=%~3
set FLAGS=%~4
set /a TEST_COUNT+=1

echo.
echo [Test %TEST_COUNT%] %TEST_NAME%
echo   Compressing folder with %VOL_SIZE% volumes...

REM Clean up old volume files
del /q output\volume_test_*.vol*.%ALGO% 2>nul

%EXE% -c sample_folder output\volume_test_%ALGO%.%ALGO% %ALGO% --volume-size %VOL_SIZE% %FLAGS%
if errorlevel 1 (
    echo   FAILED: Volume compression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

REM Check if multiple volumes were created
set VOLUME_COUNT=0
for %%f in (output\volume_test_%ALGO%.vol*.%ALGO%) do (
    set /a VOLUME_COUNT+=1
)

if %VOLUME_COUNT% lss 2 (
    echo   FAILED: Expected multiple volumes, found %VOLUME_COUNT%
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Created %VOLUME_COUNT% volumes

echo   Verifying volume naming (should be .vol001, .vol002, etc.)...
if not exist output\volume_test_%ALGO%.vol001.%ALGO% (
    echo   FAILED: First volume (.vol001) not found
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Decompressing multi-volume archive...
if exist output\volume_restored_%ALGO% rmdir /s /q output\volume_restored_%ALGO%
%EXE% -d output\volume_test_%ALGO%.vol001.%ALGO% output\volume_restored_%ALGO% %FLAGS%
if errorlevel 1 (
    echo   FAILED: Volume decompression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Verifying decompressed folder structure...
if not exist output\volume_restored_%ALGO%\PineTools.com_files (
    echo   FAILED: Folder structure not preserved after volume decompression
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   PASSED (Created %VOLUME_COUNT% volumes, decompressed successfully)
set /a PASS_COUNT+=1
goto :eof

:run_single_volume_test
set TEST_NAME=%~1
set ALGO=%~2
set FLAGS_BASE=%~3
set VOL_FLAG=%~4
set /a TEST_COUNT+=1

echo.
echo [Test %TEST_COUNT%] %TEST_NAME%
echo   Compressing folder WITHOUT volume splitting...

REM Clean up old files
del /q output\single_test_%ALGO%.* 2>nul

%EXE% -c sample_folder output\single_test_%ALGO%.%ALGO% %ALGO% %VOL_FLAG% %FLAGS_BASE%
if errorlevel 1 (
    echo   FAILED: Single volume compression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Verifying single file created (no .vol001 suffix)...
if exist output\single_test_%ALGO%.vol001.%ALGO% (
    echo   FAILED: Unexpected volume files created
    set /a FAIL_COUNT+=1
    goto :eof
)

if not exist output\single_test_%ALGO%.%ALGO% (
    echo   FAILED: Output file not created
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Decompressing single file...
if exist output\single_restored_%ALGO% rmdir /s /q output\single_restored_%ALGO%
%EXE% -d output\single_test_%ALGO%.%ALGO% output\single_restored_%ALGO% %FLAGS_BASE%
if errorlevel 1 (
    echo   FAILED: Single file decompression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Verifying folder structure...
if not exist output\single_restored_%ALGO%\PineTools.com_files (
    echo   FAILED: Folder structure not preserved
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   PASSED (Single file created, no volume splitting)
set /a PASS_COUNT+=1
goto :eof

:run_volume_list_test
set TEST_NAME=%~1
set ALGO=%~2
set VOL_SIZE=%~3
set /a TEST_COUNT+=1

echo.
echo [Test %TEST_COUNT%] %TEST_NAME%
echo   Compressing folder with %VOL_SIZE% volumes...

REM Clean up old volume files
del /q output\list_vol_test_*.vol*.%ALGO% 2>nul

%EXE% -c sample_folder output\list_vol_test_%ALGO%.%ALGO% %ALGO% --volume-size %VOL_SIZE%
if errorlevel 1 (
    echo   FAILED: Volume compression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Listing multi-volume archive...
%EXE% -l output\list_vol_test_%ALGO%.vol001.%ALGO%
if errorlevel 1 (
    echo   FAILED: Volume listing failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   PASSED (Multi-volume archive listed successfully)
set /a PASS_COUNT+=1
goto :eof

:run_volume_autodetect_test
set TEST_NAME=%~1
set ALGO=%~2
set VOL_SIZE=%~3
set /a TEST_COUNT+=1

echo.
echo [Test %TEST_COUNT%] %TEST_NAME%
echo   Compressing folder with %VOL_SIZE% volumes...

REM Clean up old volume files
del /q output\autodetect_vol_*.vol*.%ALGO% 2>nul

%EXE% -c sample_folder output\autodetect_vol_%ALGO%.%ALGO% %ALGO% --volume-size %VOL_SIZE%
if errorlevel 1 (
    echo   FAILED: Volume compression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Decompressing WITHOUT algorithm parameter (auto-detect)...
if exist output\autodetect_vol_restored_%ALGO% rmdir /s /q output\autodetect_vol_restored_%ALGO%
%EXE% -d output\autodetect_vol_%ALGO%.vol001.%ALGO% output\autodetect_vol_restored_%ALGO%
if errorlevel 1 (
    echo   FAILED: Auto-detection decompression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Verifying folder structure...
if not exist output\autodetect_vol_restored_%ALGO%\PineTools.com_files (
    echo   FAILED: Folder structure not preserved
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   PASSED (Algorithm auto-detected from multi-volume archive)
set /a PASS_COUNT+=1
goto :eof

:run_custom_size_test
set TEST_NAME=%~1
set ALGO=%~2
set VOL_SIZE=%~3
set /a TEST_COUNT+=1

echo.
echo [Test %TEST_COUNT%] %TEST_NAME%
echo   Compressing with custom volume size %VOL_SIZE%...

REM Clean up old volume files
del /q output\custom_vol_*.vol*.%ALGO% 2>nul

%EXE% -c sample_folder output\custom_vol_%ALGO%.%ALGO% %ALGO% --volume-size %VOL_SIZE%
if errorlevel 1 (
    echo   FAILED: Custom size compression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Decompressing custom-sized volumes...
if exist output\custom_vol_restored_%ALGO% rmdir /s /q output\custom_vol_restored_%ALGO%
%EXE% -d output\custom_vol_%ALGO%.vol001.%ALGO% output\custom_vol_restored_%ALGO%
if errorlevel 1 (
    echo   FAILED: Custom size decompression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Verifying folder structure...
if not exist output\custom_vol_restored_%ALGO%\PineTools.com_files (
    echo   FAILED: Folder structure not preserved
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   PASSED (Custom volume size worked correctly)
set /a PASS_COUNT+=1
goto :eof

