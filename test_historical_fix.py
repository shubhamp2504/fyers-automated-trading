#!/usr/bin/env python
"""
🔧 FYERS Historical Data Fix Test  
Test the corrected API parameters: range_from, range_to, date_format
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fyers_client import FyersClient
from datetime import datetime, timedelta

def test_fixed_historical_api():
    """Test the fixed historical data API"""
    
    print("🔧 TESTING FIXED FYERS HISTORICAL DATA API")
    print("=" * 50)
    
    # Initialize Fyers client
    try:
        fyers = FyersClient()
        print("✅ Fyers client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False
    
    # Test with corrected format
    print(f"\n🧪 Testing corrected API format")
    print("-" * 40)
    
    try:
        # Test RELIANCE historical data
        df = fyers.get_historical_data(
            symbol="NSE:RELIANCE-EQ",
            resolution="1D",
            start_date="2024-01-15",
            end_date="2024-01-19"
        )
        
        if df is not None and len(df) > 0:
            print(f"✅ SUCCESS: Historical data retrieved!")
            print(f"   📊 Records: {len(df)} candles")
            print(f"   📅 Period: {df.index[0]} to {df.index[-1]}")
            print(f"   💰 Latest close: ₹{df['close'].iloc[-1]:.2f}")
            print(f"   📈 Data sample:")
            print(df.head(2).to_string())
            return True
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_nifty_historical():
    """Test NIFTY historical data specifically"""
    
    print(f"\n🎯 Testing NIFTY Historical Data")
    print("-" * 40)
    
    try:
        fyers = FyersClient()
        
        # Test NIFTY with different symbol formats
        nifty_symbols = [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX"
        ]
        
        for symbol in nifty_symbols:
            print(f"\n   📊 Testing {symbol}")
            
            df = fyers.get_historical_data(
                symbol=symbol,
                resolution="1D",
                start_date="2024-01-15",
                end_date="2024-01-19"
            )
            
            if df is not None and len(df) > 0:
                print(f"   ✅ {symbol}: {len(df)} candles, Close: ₹{df['close'].iloc[-1]:.2f}")
                return True
            else:
                print(f"   ❌ {symbol}: No data")
        
    except Exception as e:
        print(f"   ❌ NIFTY test error: {e}")
    
    return False

if __name__ == "__main__":
    print("🚀 Testing Fixed Fyers Historical Data API")
    print("=" * 60)
    
    # Test 1: Basic fix
    success1 = test_fixed_historical_api()
    
    # Test 2: NIFTY specific
    success2 = test_nifty_historical()
    
    print("\n" + "=" * 60)
    if success1 or success2:
        print("🎉 HISTORICAL DATA API FIXED!")
        print("✅ Ready to update validation script")
    else:
        print("❌ Historical data still not working")
        print("🔍 May need further API investigation")