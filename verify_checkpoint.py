#!/usr/bin/env Python 3.11
"""
🔄 CHECKPOINT RESTORE VERIFICATION
Verify that the system is at the 100% success checkpoint state
"""

import sys
import os
import subprocess
from datetime import datetime

def check_python_version():
    """Verify Python 3.11 is being used"""
    print("🐍 Checking Python version...")
    
    if sys.version_info[:2] == (3, 11):
        print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - CORRECT")
        return True
    else:
        print(f"   ❌ Python {sys.version_info.major}.{sys.version_info.minor} - Should be 3.11")
        return False

def check_packages():
    """Verify critical packages are installed"""
    print("\n📦 Checking required packages...")
    
    package_imports = [
        ('fyers_apiv3', 'fyers_apiv3'),
        ('pandas', 'pandas'), 
        ('numpy', 'numpy'),
        ('aiohttp', 'aiohttp'),
        ('requests', 'requests'),
        ('websocket_client', 'websocket')
    ]
    
    all_good = True
    
    for package_name, import_name in package_imports:
        try:
            __import__(import_name)
            print(f"   ✅ {package_name} - INSTALLED")
        except ImportError:
            print(f"   ❌ {package_name} - MISSING")
            all_good = False
    
    return all_good

def check_critical_files():
    """Verify critical files exist"""
    print("\n📁 Checking critical files...")
    
    critical_files = [
        'fyers_config.json',
        'fyers_client.py', 
        'validate_live_fyers_system.py',
        'main.py',
        'run_trading_system.bat',
        'api_reference/market_data/market_data_complete.py'
    ]
    
    all_good = True
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path} - EXISTS")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_good = False
    
    return all_good

def verify_historical_data_fix():
    """Verify historical data API fix is in place"""
    print("\n🔧 Checking historical data API fix...")
    
    try:
        with open('api_reference/market_data/market_data_complete.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'range_from' in content and 'range_to' in content and 'date_format' in content:
            print("   ✅ Historical data API fix - APPLIED")
            return True
        else:
            print("   ❌ Historical data API fix - MISSING")
            return False
            
    except Exception as e:
        print(f"   ❌ Could not check fix: {e}")
        return False

def run_quick_validation():
    """Run a quick system validation"""
    print("\n🧪 Running system validation...")
    
    try:
        from fyers_client import FyersClient
        fyers = FyersClient()
        print("   ✅ Fyers client initialization - SUCCESS")
        return True
    except Exception as e:
        print(f"   ❌ Fyers client initialization - FAILED: {e}")
        return False

def main():
    """Main checkpoint verification"""
    print("🔄 CHECKPOINT RESTORE VERIFICATION")
    print("=" * 50)
    print(f"📅 Verification timestamp: {datetime.now()}")
    print("🎯 Target: 100% Success Checkpoint State")
    
    checks = [
        ("Python 3.11", check_python_version()),
        ("Required Packages", check_packages()),
        ("Critical Files", check_critical_files()), 
        ("Historical Data Fix", verify_historical_data_fix()),
        ("System Initialization", run_quick_validation())
    ]
    
    print("\n" + "=" * 50)
    print("📋 CHECKPOINT VERIFICATION RESULTS:")
    print("=" * 50)
    
    passed = 0
    total = len(checks)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} | {check_name}")
        if result:
            passed += 1
    
    print("-" * 50)
    success_rate = (passed / total) * 100
    
    if success_rate == 100:
        print("🎉 CHECKPOINT VERIFICATION: 100% SUCCESS")
        print("✅ System is at the target checkpoint state")
        print("🚀 Ready for live trading operations")
    else:
        print(f"⚠️  CHECKPOINT VERIFICATION: {success_rate:.0f}% ({passed}/{total})")
        print("❌ System needs restoration to checkpoint state")
        print("\n🔧 TO RESTORE:")
        print("1. Ensure Python 3.11 is installed")
        print("2. Install missing packages with pip")
        print("3. Verify all critical files exist")
        print("4. Apply historical data API fix if missing")
    
    print("=" * 50)

if __name__ == "__main__":
    main()