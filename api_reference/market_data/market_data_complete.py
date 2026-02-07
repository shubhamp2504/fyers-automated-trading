"""
FYERS API v3 - Market Data Reference Implementation
=================================================

Source: https://myapi.fyers.in/docsv3/#tag/Market-Data

All market data API calls with proper implementation examples.
"""

from fyers_apiv3 import fyersModel
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class FyersMarketData:
    """Complete market data reference implementation"""
    
    def __init__(self, client_id: str, access_token: str):
        self.fyers = fyersModel.FyersModel(client_id=client_id, token=access_token)
    
    def get_quotes(self, symbols: List[str]) -> Optional[Dict]:
        """
        Get market quotes for symbols
        API Doc: https://myapi.fyers.in/docsv3/#operation/quotes
        
        Args:
            symbols: List of symbols (max 50 per request)
        """
        try:
            # Join symbols with comma
            symbol_string = ",".join(symbols)
            data = {"symbols": symbol_string}
            
            response = self.fyers.quotes(data=data)
            
            if response['s'] == 'ok':
                print(f"✅ Quotes retrieved for {len(symbols)} symbols")
                return response['d']
            else:
                print(f"❌ Error getting quotes: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_quotes: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, resolution: str, 
                          date_from: str, date_to: str, 
                          cont_flag: int = 1) -> Optional[Dict]:
        """
        Get historical data for a symbol
        API Doc: https://myapi.fyers.in/docsv3/#operation/history
        
        Args:
            symbol: Trading symbol
            resolution: 1, 2, 3, 5, 10, 15, 30, 45, 60, 120, 240, 1D
            date_from: YYYY-MM-DD format
            date_to: YYYY-MM-DD format
            cont_flag: 1 for continuous, 0 for discrete
        """
        try:
            data = {
                "symbol": symbol,
                "resolution": resolution,
                "date_from": date_from,
                "date_to": date_to,
                "cont_flag": cont_flag
            }
            
            response = self.fyers.history(data=data)
            
            if response['s'] == 'ok':
                candles = response.get('candles', [])
                print(f"✅ Historical data: {len(candles)} candles for {symbol}")
                return response
            else:
                print(f"❌ Error getting historical data: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_historical_data: {str(e)}")
            return None
    
    def get_market_depth(self, symbol: str, ohlcv_flag: str = "1") -> Optional[Dict]:
        """
        Get market depth (Level 2 data)
        API Doc: https://myapi.fyers.in/docsv3/#operation/depth
        
        Args:
            symbol: Trading symbol
            ohlcv_flag: "1" to include OHLCV data, "0" for depth only
        """
        try:
            data = {
                "symbol": symbol,
                "ohlcv_flag": ohlcv_flag
            }
            
            response = self.fyers.depth(data=data)
            
            if response['s'] == 'ok':
                print(f"✅ Market depth retrieved for {symbol}")
                return response['d']
            else:
                print(f"❌ Error getting market depth: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_market_depth: {str(e)}")
            return None
    
    def get_option_chain(self, symbol: str, strike: Optional[str] = None,
                        expiry: Optional[str] = None) -> Optional[Dict]:
        """
        Get option chain data
        API Doc: https://myapi.fyers.in/docsv3/#operation/optionchain
        
        Args:
            symbol: Underlying symbol
            strike: Strike price (optional)
            expiry: Expiry date (optional)
        """
        try:
            data = {"symbol": symbol}
            
            if strike:
                data["strike"] = strike
            if expiry:
                data["expiry"] = expiry
                
            response = self.fyers.optionchain(data=data)
            
            if response['s'] == 'ok':
                print(f"✅ Option chain retrieved for {symbol}")
                return response['d']
            else:
                print(f"❌ Error getting option chain: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_option_chain: {str(e)}")
            return None
    
    def get_symbol_master(self, exchange: List[str]) -> Optional[List[Dict]]:
        """
        Get symbol master data
        API Doc: https://myapi.fyers.in/docsv3/#operation/symbolmaster
        
        Args:
            exchange: List of exchanges ["NSE", "BSE", "MCX", "NCDEX"]
        """
        try:
            data = {"exchange": exchange}
            
            response = self.fyers.symbolmaster(data=data)
            
            if response['s'] == 'ok':
                symbols = response.get('d', [])
                print(f"✅ Symbol master: {len(symbols)} symbols retrieved")
                return symbols
            else:
                print(f"❌ Error getting symbol master: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_symbol_master: {str(e)}")
            return None
    
    def get_market_status(self) -> Optional[Dict]:
        """
        Get market status
        API Doc: https://myapi.fyers.in/docsv3/#operation/marketstatus
        """
        try:
            response = self.fyers.market_status()
            
            if response['s'] == 'ok':
                print(f"✅ Market status retrieved")
                return response['d']
            else:
                print(f"❌ Error getting market status: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_market_status: {str(e)}")
            return None

def demo_market_data_apis():
    """Demonstrate all market data API calls"""
    
    print("📊 FYERS API v3 - Market Data Demo")
    print("=" * 50)
    
    # Load config
    try:
        with open("../config.json", "r") as f:
            config = json.load(f)
        
        client_id = config["client_id"]
        access_token = config["access_token"]
    except:
        print("❌ Please ensure config.json exists with credentials")
        return
    
    market = FyersMarketData(client_id, access_token)
    
    # Test symbols
    test_symbols = ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:INFY-EQ"]
    
    # 1. Get Quotes
    print("\n📋 1. Testing Market Quotes")
    quotes = market.get_quotes(test_symbols)
    if quotes:
        for quote in quotes:
            symbol = quote.get('n', 'Unknown')
            ltp = quote.get('lp', 0)
            change_pct = quote.get('chp', 0)
            print(f"   {symbol}: ₹{ltp} ({change_pct:.2f}%)")
    
    # 2. Get Historical Data
    print("\n📋 2. Testing Historical Data")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    hist_data = market.get_historical_data(
        symbol=test_symbols[0],
        resolution="1D",
        date_from=start_date,
        date_to=end_date
    )
    
    # 3. Get Market Depth
    print("\n📋 3. Testing Market Depth")
    depth = market.get_market_depth(test_symbols[0])
    
    # 4. Get Market Status
    print("\n📋 4. Testing Market Status")
    status = market.get_market_status()
    
    # 5. Get Option Chain (if applicable)
    print("\n📋 5. Testing Option Chain")
    option_chain = market.get_option_chain("NSE:NIFTY50-INDEX")
    
    print("\n✅ Market Data API demonstration completed!")

if __name__ == "__main__":
    demo_market_data_apis()