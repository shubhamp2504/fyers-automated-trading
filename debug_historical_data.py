#!/usr/bin/env python
"""
🔧 FYERS Historical Data Debug Tool
Fix the "Invalid input" error for historical data
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fyers_client import FyersClient
from datetime import datetime, timedelta

def test_historical_data_formats():
    """Test different symbol formats and date ranges for historical data"""
    
    print("🔍 DEBUGGING FYERS HISTORICAL DATA API")
    print("=" * 50)
    
    # Initialize Fyers client
    try:
        fyers = FyersClient()
        print("✅ Fyers client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Fyers client: {e}")
        return
    
    # Test different symbol formats
    test_symbols = [
        "NSE:NIFTY50-INDEX",    # Current format
        "NSE:NIFTYINDEX",       # Alternative 1
        "NSE:NIFTY",            # Alternative 2  
        "NSE:NIFTY_50",         # Alternative 3
        "NSE:RELIANCE-EQ",      # Known working symbol from quotes
    ]
    
    # Test different date ranges
    today = datetime.now()
    date_ranges = [
        # Recent 7 days
        (
            (today - timedelta(days=7)).strftime('%Y-%m-%d'),
            today.strftime('%Y-%m-%d'),
            "Last 7 days"
        ),
        # Recent 30 days  
        (
            (today - timedelta(days=30)).strftime('%Y-%m-%d'),
            today.strftime('%Y-%m-%d'),
            "Last 30 days"
        ),
        # Specific working dates
        (
            "2024-01-15",
            "2024-01-20", 
            "Jan 2024 (5 days)"
        )
    ]
    
    print(f"\n📊 Testing {len(test_symbols)} symbols with {len(date_ranges)} date ranges")
    
    for symbol in test_symbols:
        print(f"\n🎯 Testing symbol: {symbol}")
        print("-" * 30)
        
        for start_date, end_date, description in date_ranges:
            print(f"   📅 {description} ({start_date} to {end_date})")
            
            try:
                df = fyers.get_historical_data(
                    symbol=symbol,
                    resolution="1D",
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is not None and len(df) > 0:
                    print(f"   ✅ SUCCESS: {len(df)} candles retrieved")
                    print(f"   💰 Latest close: ₹{df['close'].iloc[-1]:.2f}")
                    
                    # Found working combination, show details
                    print(f"\n🎉 WORKING COMBINATION FOUND!")
                    print(f"   Symbol: {symbol}")
                    print(f"   Date range: {start_date} to {end_date}")
                    print(f"   Data points: {len(df)}")
                    return symbol, start_date, end_date
                    
                else:
                    print(f"   ❌ No data returned")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print()
    
    print("❌ No working combination found - API may need different approach")
    return None, None, None

if __name__ == "__main__":
    test_historical_data_formats()