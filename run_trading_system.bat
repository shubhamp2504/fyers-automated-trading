@echo off
REM 🎯 FYERS Trading System - Python 3.11 Launcher
REM 🏁 CHECKPOINT: 100% SUCCESS STATE (February 7, 2026)
REM This script uses Python 3.11 for full compatibility with Fyers API
REM No Visual C++ compilation issues!

echo ===============================================
echo 🚀 FYERS TRADING SYSTEM (Python 3.11)
echo 🏁 CHECKPOINT: 100%% SUCCESS STATE
echo ===============================================
echo.
echo 🐍 Using Python 3.11.9 for maximum compatibility
echo 📊 Official Fyers API v3 integration
echo 💰 Real account trading with live data
echo ✅ Historical data API working (100%% validation)
echo.

REM Set Python 3.11 path
set PYTHON311=C:\Users\shubh\AppData\Local\Programs\Python\Python311\python.exe

REM Check if Python 3.11 exists
if not exist "%PYTHON311%" (
    echo ❌ Python 3.11 not found at: %PYTHON311%
    echo Please install Python 3.11.9 first
    pause
    exit /b 1
)

REM Show Python version
echo 🔍 Verifying Python version:
"%PYTHON311%" --version
echo.

REM Menu for different operations
:MENU
echo Choose an option:
echo 1. Run Live Trading System Validation
echo 2. Run Main Trading System
echo 3. Test Python 3.11 Compatibility
echo 4. Verify Checkpoint State (100%% Success)
echo 5. Restore to Checkpoint
echo 6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo 🔍 Running Live Trading System Validation...
    "%PYTHON311%" validate_live_fyers_system.py
    echo.
    goto MENU
) else if "%choice%"=="2" (
    echo.
    echo 🚀 Starting Live Trading System...
    echo ⚠️  WARNING: This will trade with REAL MONEY!
    set /p confirm="Are you sure? (y/N): "
    if /i "%confirm%"=="y" (
        "%PYTHON311%" main.py
    ) else (
        echo Trading cancelled.
    )
    echo.
    goto MENU
) else if "%choice%"=="3" (
    echo.
    echo 🧪 Testing Python 3.11 Compatibility...
    "%PYTHON311%" test_python311_success.py
    echo.
    goto MENU
) else if "%choice%"=="4" (
    echo.
    echo � Verifying Checkpoint State...
    "%PYTHON311%" verify_checkpoint.py
    echo.
    goto MENU
) else if "%choice%"=="5" (
    echo.
    echo 🔧 Restoring to Checkpoint State...
    call restore_checkpoint.bat
    echo.
    goto MENU
) else if "%choice%"=="6" (
    echo.
    echo �👋 Goodbye!
    exit /b 0
) else (
    echo.
    echo ❌ Invalid choice. Please try again.
    echo.
    goto MENU
)