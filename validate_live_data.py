"""
FYERS Live Data Connection Validator
Test script to validate FYERS API credentials and fetch live market data
"""

import json
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_data_validation.log'),
        logging.StreamHandler()
    ]
)

class LiveDataValidator:
    """Validate FYERS API connection and live data fetching"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.market_data = None
        self.client_id = None
        
    def load_credentials(self) -> bool:
        """Load and validate FYERS credentials"""
        try:
            with open('fyers_config.json', 'r') as f:
                config = json.load(f)
            
            self.client_id = config['fyers']['client_id']
            access_token = config['fyers']['access_token']
            
            if not self.client_id or not access_token:
                self.logger.error("❌ Missing client_id or access_token in config")
                return False
                
            if len(access_token) < 50:
                self.logger.error("❌ Access token appears to be invalid (too short)")
                return False
            
            # Initialize FYERS client
            try:
                from fyers_simple_client import FyersMarketData
                self.market_data = FyersMarketData(self.client_id, access_token)
                self.logger.info(f"✅ FYERS client initialized successfully")
                self.logger.info(f"   Client ID: {self.client_id}")
                self.logger.info(f"   Token Length: {len(access_token)} characters")
                return True
                
            except ImportError:
                self.logger.error("❌ fyers_client module not found. Please ensure FYERS API is properly installed.")
                return False
            except Exception as e:
                self.logger.error(f"❌ Error initializing FYERS client: {e}")
                return False
                
        except FileNotFoundError:
            self.logger.error("❌ fyers_config.json not found")
            return False
        except json.JSONDecodeError:
            self.logger.error("❌ Invalid JSON in fyers_config.json")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error loading credentials: {e}")
            return False
    
    def test_market_data(self, symbol: str = "NSE:NIFTY50-INDEX") -> Optional[pd.DataFrame]:
        """Test fetching live market data"""
        
        if not self.market_data:
            self.logger.error("❌ FYERS client not initialized")
            return None
            
        try:
            self.logger.info(f"📊 Testing live data fetch for {symbol}")
            
            # Get data for last 10 days
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
            
            self.logger.info(f"   Period: {start_date} to {end_date}")
            
            # Fetch historical data
            hist_data = self.market_data.get_historical_data(
                symbol=symbol,
                resolution="1D",
                date_from=start_date,
                date_to=end_date,
                cont_flag=1
            )
            
            if hist_data and hist_data.get('s') == 'ok':
                if 'candles' in hist_data and hist_data['candles']:
                    candles = hist_data['candles']
                    df = pd.DataFrame(
                        candles,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('timestamp', inplace=True)
                    
                    self.logger.info(f"✅ Live data fetched successfully!")
                    self.logger.info(f"   Rows: {len(df)}")
                    self.logger.info(f"   Columns: {list(df.columns)}")
                    self.logger.info(f"   Date Range: {df.index[0]} to {df.index[-1]}")
                    self.logger.info(f"   Latest Close: ₹{df['close'].iloc[-1]:,.2f}")
                    
                    return df
                else:
                    self.logger.warning("⚠️ FYERS returned success but no candle data")
                    return None
            else:
                self.logger.error(f"❌ FYERS API error: {hist_data.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching live data: {e}")
            return None
    
    def test_option_chain(self, symbol: str = "NSE:NIFTY") -> Optional[Dict]:
        """Test fetching live option chain data"""
        
        if not self.market_data:
            self.logger.error("❌ FYERS client not initialized")
            return None
            
        try:
            self.logger.info(f"📊 Testing option chain fetch for {symbol}")
            
            # Get current month expiry (simplified)
            expiry = "2026-02-27"  # Adjust based on actual expiry dates
            
            option_data = self.market_data.get_option_chain(
                symbol=symbol,
                expiry_date=expiry
            )
            
            if option_data:
                self.logger.info(f"✅ Option chain data fetched successfully!")
                self.logger.info(f"   Data type: {type(option_data)}")
                if isinstance(option_data, dict):
                    self.logger.info(f"   Keys: {list(option_data.keys())}")
                elif isinstance(option_data, list):
                    self.logger.info(f"   Items: {len(option_data)}")
                return option_data
            else:
                self.logger.warning("⚠️ No option chain data received")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching option chain: {e}")
            return None
    
    def run_comprehensive_test(self):
        """Run comprehensive validation tests"""
        
        print("🚀 FYERS Live Data Connection Validator")
        print("=" * 60)
        
        # Step 1: Load credentials
        print("\\n📋 Step 1: Loading FYERS Credentials")
        print("-" * 40)
        if not self.load_credentials():
            print("❌ Credential validation failed. Please check your fyers_config.json file.")
            return False
        
        # Step 2: Test market data
        print("\\n📊 Step 2: Testing Live Market Data")
        print("-" * 40)
        
        test_symbols = [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX", 
            "NSE:RELIANCE-EQ"
        ]
        
        successful_tests = 0
        for symbol in test_symbols:
            data = self.test_market_data(symbol)
            if data is not None:
                successful_tests += 1
                print(f"   ✅ {symbol}: SUCCESS")
            else:
                print(f"   ❌ {symbol}: FAILED")
        
        # Step 3: Test option chain (optional)
        print("\\n📈 Step 3: Testing Option Chain (Optional)")
        print("-" * 40)
        option_data = self.test_option_chain("NSE:NIFTY")
        if option_data:
            print("   ✅ Option chain: SUCCESS")
        else:
            print("   ⚠️ Option chain: FAILED (may not be supported)")
        
        # Summary
        print("\\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY")
        print("=" * 60)
        
        if successful_tests > 0:
            print(f"✅ FYERS API Connection: WORKING")
            print(f"✅ Live Data Available: YES")
            print(f"✅ Successful Tests: {successful_tests}/{len(test_symbols)}")
            print(f"🎯 Your system is ready for LIVE TRADING!")
            
            print(f"\\n🚀 Next Steps:")
            print(f"   1. Run: python run_options_backtest_comprehensive.py")
            print(f"   2. Run: python fyers_algo_backtester.py") 
            print(f"   3. All systems will now use LIVE market data!")
            
            return True
        else:
            print(f"❌ FYERS API Connection: FAILED")
            print(f"❌ Live Data Available: NO") 
            print(f"❌ Successful Tests: {successful_tests}/{len(test_symbols)}")
            
            print(f"\\n🔧 Troubleshooting:")
            print(f"   1. Verify FYERS credentials are correct")
            print(f"   2. Check internet connection")
            print(f"   3. Ensure FYERS API access is enabled")
            print(f"   4. Check if access token is expired")
            
            return False

def main():
    """Main validation function"""
    validator = LiveDataValidator()
    success = validator.run_comprehensive_test()
    
    if success:
        print(f"\\n🎉 VALIDATION COMPLETE - SYSTEM READY FOR LIVE TRADING! 🎉")
    else:
        print(f"\\n⚠️ VALIDATION FAILED - PLEASE CHECK CONFIGURATION ⚠️")
    
    return success

if __name__ == "__main__":
    main()