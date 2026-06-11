@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   nn_framework - CMake + MinGW Build
echo ========================================
echo.

:: ====== 获取脚本所在目录 ======
set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

:: ====== 检查中文路径 ======
echo %PROJECT_DIR% | findstr /r "[^\x00-\x7F]" >nul
if %errorlevel%==0 (
    echo [WARN] Path contains non-ASCII characters.
    echo        Mapping temporary drive N: to avoid MinGW path issues...
    
    :: 清除可能已存在的 N: 映射
    subst N: /d >nul 2>&1
    :: 映射当前项目路径
    subst N: "%PROJECT_DIR%"
    if %errorlevel% neq 0 (
        echo [ERROR] subst failed! Cannot map drive N:
        echo         Please move project to a path without Chinese characters.
        pause
        exit /b 1
    )
    set "BUILD_DIR=N:\build"
    set "CMAKE_SRC=N:"
    set "USE_SUBST=1"
    echo        Using N: drive for build.
) else (
    set "BUILD_DIR=%PROJECT_DIR%\build"
    set "CMAKE_SRC=%PROJECT_DIR%"
    set "USE_SUBST=0"
)
echo.

:: ====== 检查工具链 ======
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] cmake not found in PATH!
    pause
    exit /b 1
)

where gcc >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] gcc not found in PATH!
    pause
    exit /b 1
)

for /f "tokens=1-3" %%a in ('gcc --version 2^>^&1 ^| findstr /i "gcc"') do (
    echo [INFO] Compiler: %%a %%b %%c
)
echo.

:: ====== 清理旧构建 ======
if exist "%BUILD_DIR%" (
    echo [INFO] Cleaning old build directory...
    rmdir /s /q "%BUILD_DIR%"
)
mkdir "%BUILD_DIR%"

:: ====== CMake 配置 ======
echo [INFO] Configuring with CMake (Release)...
cmake -S "%CMAKE_SRC%" -B "%BUILD_DIR%" -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 (
    echo [ERROR] CMake configure failed!
    if "%USE_SUBST%"=="1" subst N: /d >nul 2>&1
    pause
    exit /b 1
)
echo.

:: ====== 编译 ======
echo [INFO] Building...
cmake --build "%BUILD_DIR%"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed!
    if "%USE_SUBST%"=="1" subst N: /d >nul 2>&1
    pause
    exit /b 1
)
echo.

:: ====== 复制产物 ======
echo [INFO] Copying executables...
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

:: ====== 清理临时映射 ======
if "%USE_SUBST%"=="1" (
    subst N: /d >nul 2>&1
    echo        Removed temporary N: drive mapping.
)

echo.
echo ========================================
echo   Build SUCCESS!
echo   Run: nn_demo.exe
echo ========================================
pause
