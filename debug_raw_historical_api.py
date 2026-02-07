#!/usr/bin/env python
"""
🔧 FYERS Historical Data Raw API Test
Test direct API calls to understand the exact format needed
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fyers_client import FyersClient
from datetime import datetime, timedelta
import json

def test_raw_historical_api():
    """Test the raw Fyers historical data API with different formats"""
    
    print("🔍 TESTING RAW FYERS HISTORICAL DATA API")
    print("=" * 50)
    
    # Initialize Fyers client
    try:
        fyers = FyersClient()
        print("✅ Fyers client initialized")
        
        # Get the raw fyers model object
        fyers_model = fyers.market_data.fyers
        print("✅ Got raw Fyers model object")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test different parameter formats
    test_cases = [
        {
            "name": "Format 1: Standard RELIANCE",
            "data": {
                "symbol": "NSE:RELIANCE-EQ",
                "resolution": "1D",
                "date_from": "2024-01-15", 
                "date_to": "2024-01-20",
                "cont_flag": 1
            }
        },
        {
            "name": "Format 2: No cont_flag",
            "data": {
                "symbol": "NSE:RELIANCE-EQ",
                "resolution": "1D", 
                "date_from": "2024-01-15",
                "date_to": "2024-01-20"
            }
        },
        {
            "name": "Format 3: Different date format",
            "data": {
                "symbol": "NSE:RELIANCE-EQ",
                "resolution": "1D",
                "date_from": "2024-01-15T00:00:00Z",
                "date_to": "2024-01-20T00:00:00Z",
                "cont_flag": 1
            }
        },
        {
            "name": "Format 4: Unix timestamp",
            "data": {
                "symbol": "NSE:RELIANCE-EQ", 
                "resolution": "1D",
                "date_from": str(int(datetime(2024, 1, 15).timestamp())),
                "date_to": str(int(datetime(2024, 1, 20).timestamp())),
                "cont_flag": 1
            }
        },
        {
            "name": "Format 5: Different resolution",
            "data": {
                "symbol": "NSE:RELIANCE-EQ",
                "resolution": "60",  # Try minutes instead
                "date_from": "2024-01-19",
                "date_to": "2024-01-19", 
                "cont_flag": 1
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 {test_case['name']}")
        print("-" * 40)
        print(f"📋 Data: {json.dumps(test_case['data'], indent=2)}")
        
        try:
            # Make raw API call
            response = fyers_model.history(data=test_case['data'])
            
            print(f"📊 Response status: {response.get('s', 'Unknown')}")
            print(f"📊 Response message: {response.get('message', 'No message')}")
            
            if response.get('s') == 'ok':
                candles = response.get('candles', [])
                print(f"✅ SUCCESS: {len(candles)} candles retrieved")
                if candles:
                    print(f"📈 First candle: {candles[0]}")
                    print(f"📈 Last candle: {candles[-1]}")
                return True
            else:
                print(f"❌ Failed: {response}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print()
    
    print("❌ All formats failed - historical data may not be available for this account")
    
    # Try to get available symbols
    print("\n🔍 Checking account permissions and symbol format...")
    try:
        # Test if quotes work for comparison
        quotes_response = fyers_model.quotes(data={"symbols": "NSE:RELIANCE-EQ"})
        print(f"📊 Quotes work: {quotes_response.get('s') == 'ok'}")
        
        # Test profile to check account type
        profile = fyers_model.get_profile()
        if profile.get('s') == 'ok':
            print(f"👤 Account type: {profile.get('data', {}).get('user_type', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Additional checks failed: {e}")

if __name__ == "__main__":
    test_raw_historical_api()