"""
FYERS Official API Client - Live Trading with Real Data
=======================================================

Official Fyers API v3 implementation for live trading.
Source: https://myapi.fyers.in/docsv3

Features:
- Real-time market data
- Live order placement and management  
- Portfolio tracking
- WebSocket streaming
- Full API compliance
"""

from fyers_apiv3 import fyersModel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'api_reference', 'market_data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'api_reference', 'orders'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'api_reference', 'portfolio'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'api_reference', 'websocket'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'api_reference', 'authentication'))

from market_data_complete import FyersMarketData
from orders_complete import FyersOrders
from portfolio_complete import FyersPortfolio
from websocket_complete import FyersWebSocketReference
from auth_complete import FyersAuthentication

import json
import logging
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime

class FyersClient:
    """Official Fyers API v3 Client for Live Trading with Real Data"""
    
    def __init__(self, config_path='fyers_config.json'):
        """Initialize Fyers client with official API"""
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            self.client_id = config['fyers']['client_id']
            self.access_token = config['fyers']['access_token']
            
            print(f"Initializing Fyers API Client...")
            print(f"   Client ID: {self.client_id[:10]}...")
            print(f"   Using LIVE FYERS API with REAL DATA")
            
            # Initialize official Fyers API components
            self.fyers = fyersModel.FyersModel(client_id=self.client_id, token=self.access_token)
            self.market_data = FyersMarketData(self.client_id, self.access_token)
            self.orders = FyersOrders(self.client_id, self.access_token) 
            self.portfolio = FyersPortfolio(self.client_id, self.access_token)
            self.websocket = FyersWebSocketReference(self.client_id, self.access_token)
            
            # Verify connection with real API call
            self._verify_connection()
                
        except Exception as e:
            print(f"Error initializing FYERS client: {e}")
            raise
    
    def _verify_connection(self):
        """Verify API connection with real Fyers account"""
        try:
            profile = self.portfolio.get_profile()
            if profile:
                print(f"Connected to REAL Fyers account")
                print(f"   Account: {profile.get('name', 'Unknown')}")
                print(f"   Email: {profile.get('email_id', 'Unknown')}")
            else:
                print(f"Connected but unable to fetch profile")
        except Exception as e:
            print(f"Connection verification failed: {e}")

    def place_order(self, symbol: str, qty: int, side: str, order_type: str = "MARKET", 
                   product_type: str = "INTRADAY", limit_price: float = 0, stop_price: float = 0, 
                   validity: str = "DAY") -> Optional[Dict]:
        """Place live order using official Fyers API"""
        
        # Convert parameters to Fyers API format
        side_map = {"BUY": 1, "SELL": -1}
        type_map = {"MARKET": 1, "LIMIT": 2, "STOP_MARKET": 3, "STOP_LIMIT": 4}
        
        api_side = side_map.get(side.upper(), 1)
        api_type = type_map.get(order_type.upper(), 1)
        
        print(f"Placing LIVE order via Fyers API:")
        print(f"   Symbol: {symbol}")
        print(f"   Side: {side} | Qty: {qty} | Type: {order_type}")
        
        return self.orders.place_order(
            symbol=symbol,
            qty=qty, 
            side=api_side,
            type=api_type,
            product_type=product_type,
            limit_price=limit_price,
            stop_price=stop_price,
            validity=validity
        )
        
    def get_historical_data(self, symbol: str, resolution: str = "1D", 
                           start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Get historical data using official Fyers API"""
        
        if not start_date:
            start_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Fetching REAL historical data from Fyers API:")
        print(f"   Symbol: {symbol} | Resolution: {resolution}")
        print(f"   Period: {start_date} to {end_date}")
        
        try:
            response = self.market_data.get_historical_data(
                symbol=symbol,
                resolution=resolution, 
                date_from=start_date,
                date_to=end_date,
                cont_flag=1  # Add continuous flag
            )
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return None
        
        if response and response.get('s') == 'ok' and 'candles' in response:
            # Convert to pandas DataFrame
            candles = response['candles']
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            print(f"Retrieved {len(df)} candles")
            return df
        else:
            print(f"API Response: {response}")
        return None
    
    def get_live_quotes(self, symbols: List[str]) -> Optional[Dict]:
        """Get live market quotes using official Fyers API"""
        print(f"Fetching LIVE quotes from Fyers API for {len(symbols)} symbols")
        response = self.market_data.get_quotes(symbols)
        
        # Handle the actual response format from Fyers API
        if response and isinstance(response, list) and len(response) > 0:
            # Convert list response to dictionary format for consistency
            quotes_dict = {'d': {}}
            for item in response:
                if 's' in item and item['s'] == 'ok':
                    symbol = item.get('n', '')
                    if symbol and 'v' in item:
                        quotes_dict['d'][symbol] = {'v': item['v']}
            quotes_dict['s'] = 'ok'
            return quotes_dict
        
        return response
    
    def get_market_depth(self, symbol: str) -> Optional[Dict]:
        """Get market depth using official Fyers API"""
        print(f"Fetching LIVE market depth from Fyers API: {symbol}")
        return self.market_data.get_market_depth(symbol)
    
    def get_positions(self) -> Optional[List[Dict]]:
        """Get current positions using official Fyers API"""
        print(f"Fetching LIVE positions from Fyers API")
        return self.portfolio.get_positions()
    
    def get_holdings(self) -> Optional[List[Dict]]:
        """Get holdings using official Fyers API"""
        print(f"Fetching LIVE holdings from Fyers API")
        return self.portfolio.get_holdings()
    
    def get_funds(self) -> Optional[Dict]:
        """Get fund details using official Fyers API"""
        print(f"💰 Fetching LIVE fund details from Fyers API")
        return self.portfolio.get_funds()
    
    def get_orders(self) -> Optional[List[Dict]]:
        """Get order book using official Fyers API"""
        print(f"📋 Fetching LIVE orders from Fyers API")
        return self.orders.get_orders()
    
    def cancel_order(self, order_id: str) -> Optional[Dict]:
        """Cancel order using official Fyers API"""
        print(f"❌ Cancelling LIVE order via Fyers API: {order_id}")
        return self.orders.cancel_order(order_id)
    
    def modify_order(self, order_id: str, **kwargs) -> Optional[Dict]:
        """Modify order using official Fyers API"""
        print(f"✏️ Modifying LIVE order via Fyers API: {order_id}")
        return self.orders.modify_order(order_id, **kwargs)
    
    def start_websocket(self, symbols: List[str], data_type: str = "symbolUpdate"):
        """Start WebSocket for live data streaming"""
        print(f"🔴 Starting LIVE WebSocket streaming for {len(symbols)} symbols")
        try:
            self.websocket.connect()
            if data_type == "symbolUpdate":
                self.websocket.subscribe_symbols_quotes(symbols)
            elif data_type == "depthUpdate": 
                self.websocket.subscribe_symbols_depth(symbols)
            return True
        except Exception as e:
            print(f"❌ WebSocket start failed: {e}")
            return False
    
    def stop_websocket(self):
        """Stop WebSocket streaming"""
        print(f"🔴 Stopping WebSocket streaming")
        try:
            self.websocket.disconnect()
            return True
        except Exception as e:
            print(f"❌ WebSocket stop failed: {e}")
            return False

    # Legacy compatibility methods
    def get_market_quote(self, symbols):
        """Legacy method for compatibility"""
        return self.get_live_quotes(symbols if isinstance(symbols, list) else [symbols])
    
    def get_profile(self):
        """Legacy method for compatibility"""
        return self.portfolio.get_profile()
