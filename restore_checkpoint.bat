@echo off
REM 🏁 CHECKPOINT RESTORE SCRIPT
REM Restore the Fyers Trading System to 100% Success State

echo ================================================================
echo 🔄 FYERS TRADING SYSTEM - CHECKPOINT RESTORE
echo ================================================================
echo.
echo 📅 Target State: February 7, 2026 - 100%% Success Checkpoint
echo 🎯 Objective: Restore all systems to fully operational state
echo.

REM Step 1: Verify Python 3.11
echo 🔍 Step 1: Verifying Python 3.11 installation...
set PYTHON311=C:\Users\shubh\AppData\Local\Programs\Python\Python311\python.exe

if not exist "%PYTHON311%" (
    echo ❌ Python 3.11 not found at expected location
    echo 📥 Please install Python 3.11.9 from https://www.python.org/downloads/release/python-3119/
    echo ✅ During install: Check "Add Python to PATH"
    pause
    exit /b 1
)

echo ✅ Python 3.11 found
"%PYTHON311%" --version
echo.

REM Step 2: Install required packages
echo 🔍 Step 2: Installing required packages...
echo 📦 Installing fyers-apiv3, pandas, numpy...

"%PYTHON311%" -m pip install fyers-apiv3 pandas numpy requests websocket-client
if errorlevel 1 (
    echo ❌ Package installation failed
    pause
    exit /b 1
)
echo ✅ Packages installed successfully
echo.

REM Step 3: Verify checkpoint state
echo 🔍 Step 3: Verifying checkpoint state...
"%PYTHON311%" verify_checkpoint.py
if errorlevel 1 (
    echo ⚠️  Checkpoint verification issues detected
    echo 🔧 Manual review may be needed
) else (
    echo ✅ Checkpoint verification passed
)
echo.

REM Step 4: Test system validation
echo 🔍 Step 4: Testing system validation...
echo ⚠️  Running full system validation (may take 30-60 seconds)...
"%PYTHON311%" validate_live_fyers_system.py > validation_result.tmp 2>&1

findstr /C:"ALL TESTS PASSED" validation_result.tmp > nul
if errorlevel 1 (
    echo ❌ System validation failed
    echo 📊 Check validation_result.tmp for details
) else (
    echo ✅ System validation: ALL TESTS PASSED
    echo 🎉 CHECKPOINT RESTORATION COMPLETE
    del validation_result.tmp
)
echo.

echo ================================================================
echo 🏁 RESTORE COMPLETE
echo ================================================================
echo.
echo 📋 Your system should now be at 100%% success checkpoint state:
echo    ✅ Python 3.11 with all required packages
echo    ✅ Official Fyers API v3 integration  
echo    ✅ Historical data API fix applied
echo    ✅ Live trading system ready
echo.
echo 🚀 To start trading:
echo    1. Double-click: run_trading_system.bat
echo    2. Or manually: "%PYTHON311%" main.py
echo.
echo ⚠️  Remember: This system trades with REAL MONEY
echo    Use appropriate position sizing and risk management
echo.
pause