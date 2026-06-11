@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
REM ============================================================================
REM build.bat - nn_framework auto build script (Windows + MinGW)
REM
REM What this script does (step by step):
REM   1. Locate project root (where this .bat file lives, must have CMakeLists.txt)
REM   2. Check if cmake and gcc are available in PATH
REM   3. Detect non-ASCII characters in path (e.g. Chinese)
REM      -> If found, map a temporary drive N: to bypass MinGW path bug
REM   4. Clean old build/ directory to avoid stale CMake cache
REM   5. Run cmake configure (Release mode, MinGW Makefiles generator)
REM   6. Run cmake --build to compile all targets
REM   7. Copy exe files to project root for easy access
REM   8. Release the temporary N: drive mapping
REM
REM Known issue:
REM   MinGW make cannot handle non-ASCII paths (e.g. "神经网络").
REM   Error: "No rule to make target". Workaround: subst N: to get ASCII path.
REM   Permanent fix: move project to an all-ASCII path.
REM
REM Output targets:
REM   nn_demo.exe    - Interactive training demo (9 tasks)
REM   ex_minimal.exe - Minimal API usage example
REM   ex_export.exe  - Model export/import example
REM
REM Log file: build.log (in project root, overwritten each run)
REM ============================================================================

REM -- Redirect all output to console AND log file --
set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "LOGFILE=%PROJECT_DIR%\build.log"

echo ========================================  > "%LOGFILE%"
echo   nn_framework - CMake + MinGW Build    >> "%LOGFILE%"
echo ========================================  >> "%LOGFILE%"
echo.                                          >> "%LOGFILE%"

echo ========================================
echo   nn_framework - CMake + MinGW Build
echo   Log: build.log
echo ========================================
echo.

REM -- Step 1: Verify project root --
echo [1/7] Checking project directory...
echo [1/7] Checking project directory... >> "%LOGFILE%"
if not exist "%PROJECT_DIR%\CMakeLists.txt" (
    echo [ERROR] CMakeLists.txt not found in %PROJECT_DIR%
    echo         Please run this script from the nn_framework root directory.
    goto :done
)
echo        OK: %PROJECT_DIR%
echo        OK: %PROJECT_DIR% >> "%LOGFILE%"
echo.

REM -- Step 2: Detect non-ASCII path --
echo [2/7] Checking path encoding...
set "USE_SUBST=0"
echo %PROJECT_DIR% | findstr /r "[^\x00-\x7F]" >nul 2>&1
if not %errorlevel%==0 goto :ascii_path

REM --- Non-ASCII path: use subst to map parent dir ---
echo        Path contains non-ASCII characters, mapping N: drive...
for %%I in ("%PROJECT_DIR%") do set "PROJECT_NAME=%%~nxI"
for %%I in ("%PROJECT_DIR%\..") do set "PARENT_DIR=%%~fI"
subst N: /d >nul 2>&1
subst N: "%PARENT_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] subst N: failed!
    echo         Move project to an ASCII-only path.
    goto :done
)
set "BUILD_DIR=N:\%PROJECT_NAME%\build"
set "CMAKE_SRC=N:\%PROJECT_NAME%"
set "USE_SUBST=1"
echo        OK: Using N:\%PROJECT_NAME%
goto :path_done

:ascii_path
set "BUILD_DIR=%PROJECT_DIR%\build"
set "CMAKE_SRC=%PROJECT_DIR%"
echo        OK: ASCII path, no mapping needed

:path_done
echo.

REM -- Step 3: Check toolchain --
echo [3/7] Checking toolchain...
echo [3/7] Checking toolchain... >> "%LOGFILE%"
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] cmake not found in PATH!
    echo         Download from https://cmake.org/download/
    goto :done
)
where gcc >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] gcc not found in PATH!
    echo         Install MinGW from https://sourceforge.net/projects/mingw/
    goto :done
)
for /f "tokens=1-3" %%a in ('gcc --version 2^>^&1 ^| findstr /i "gcc"') do (
    echo        Compiler: %%a %%b %%c
    echo        Compiler: %%a %%b %%c >> "%LOGFILE%"
)
for /f "tokens=1-3" %%a in ('cmake --version 2^>^&1 ^| findstr /r "cmake version"') do (
    echo        CMake: %%c
    echo        CMake: %%c >> "%LOGFILE%"
)
echo.

REM -- Step 4: Clean old build directory --
echo [4/7] Cleaning old build directory...
echo [4/7] Cleaning old build directory... >> "%LOGFILE%"
if exist "%BUILD_DIR%" (
    rmdir /s /q "%BUILD_DIR%"
    echo        Removed old build/
)
mkdir "%BUILD_DIR%"
echo        OK: build/ ready
echo.

REM -- Step 5: CMake configure --
echo [5/7] Running cmake configure (Release)...
cmake -S "%CMAKE_SRC%" -B "%BUILD_DIR%" -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 (
    echo [ERROR] CMake configure failed!
    goto :done
)
echo        OK: CMake configured
echo.

REM -- Step 6: Build --
echo [6/7] Compiling...
cmake --build "%BUILD_DIR%"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed!
    goto :done
)
echo        OK: All targets compiled
echo.

REM -- Step 7: Copy executables to project root --
echo [7/7] Copying executables to project root...
if exist "%BUILD_DIR%\nn_demo.exe" (
    copy /y "%BUILD_DIR%\nn_demo.exe" "%PROJECT_DIR%\nn_demo.exe" >nul
    echo        ^> nn_demo.exe
)
if exist "%BUILD_DIR%\ex_minimal.exe" (
    copy /y "%BUILD_DIR%\ex_minimal.exe" "%PROJECT_DIR%\ex_minimal.exe" >nul
    echo        ^> ex_minimal.exe
)
if exist "%BUILD_DIR%\ex_export.exe" (
    copy /y "%BUILD_DIR%\ex_export.exe" "%PROJECT_DIR%\ex_export.exe" >nul
    echo        ^> ex_export.exe
)

REM -- Release N: drive if mapped --
if "%USE_SUBST%"=="1" (
    subst N: /d >nul 2>&1
    echo        Released N: drive mapping.
)

echo.
echo ========================================
echo   Build SUCCESS!
echo   Run: nn_demo.exe
echo ========================================

:done
echo.
pause
