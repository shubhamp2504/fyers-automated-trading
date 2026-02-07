"""
Advanced FYERS API Features for Automated Trading System
⚠️  CRITICAL: Always consult https://myapi.fyers.in/docsv3 FIRST before implementing featuresBased on FYERS API v3 Documentation: https://myapi.fyers.in/docsv3
"""

import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from fyers_client import FyersClient

class AdvancedFyersFeatures:
    """Enhanced FYERS API features for professional trading"""
    
    def __init__(self):
        self.client = FyersClient()
        
    def get_market_depth(self, symbol: str) -> Optional[Dict]:
        """
        Get Level 2 market depth data (Bid/Ask levels)
        Useful for advanced order placement strategies
        """
        try:
            data = {
                "symbols": symbol,
                "ohlcv_flag": "1"  # Include OHLCV data
            }
            response = self.client.fyers.depth(data=data)
            
            if response['s'] == 'ok':
                return response['d']
            else:
                print(f"❌ Market depth error: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in market depth: {str(e)}")
            return None
    
    def get_option_chain(self, symbol: str, strike: Optional[str] = None, 
                        expiry: Optional[str] = None) -> Optional[Dict]:
        """
        Get options chain data for derivatives trading
        Essential for options strategies
        """
        try:
            data = {"symbol": symbol}
            if strike:
                data["strike"] = strike
            if expiry:
                data["expiry"] = expiry
                
            response = self.client.fyers.optionchain(data=data)
            
            if response['s'] == 'ok':
                return response['d']
            else:
                print(f"❌ Option chain error: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in option chain: {str(e)}")
            return None
    
    def get_multi_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols in a single API call
        Much more efficient for monitoring multiple stocks
        """
        try:
            # FYERS supports up to 50 symbols per request
            symbol_chunks = [symbols[i:i+50] for i in range(0, len(symbols), 50)]
            all_quotes = {}
            
            for chunk in symbol_chunks:
                symbol_string = ",".join(chunk)
                data = {"symbols": symbol_string}
                
                response = self.client.fyers.quotes(data=data)
                
                if response['s'] == 'ok':
                    for symbol_data in response['d']:
                        symbol = symbol_data['n']
                        all_quotes[symbol] = {
                            'ltp': symbol_data.get('lp', 0),
                            'volume': symbol_data.get('v', 0),
                            'high': symbol_data.get('h', 0),
                            'low': symbol_data.get('l', 0),
                            'open': symbol_data.get('o', 0),
                            'change': symbol_data.get('ch', 0),
                            'change_percent': symbol_data.get('chp', 0),
                            'prev_close': symbol_data.get('pc', 0),
                            'timestamp': symbol_data.get('tt', 0)
                        }
                else:
                    print(f"❌ Multi quotes error: {response.get('message', 'Unknown error')}")
                
                time.sleep(0.1)  # Rate limiting
            
            return all_quotes
            
        except Exception as e:
            print(f"❌ Exception in multi quotes: {str(e)}")
            return {}
    
    def get_symbol_master(self, exchange: str = "NSE") -> Optional[List[Dict]]:
        """
        Download complete symbol master from FYERS
        Contains all tradeable symbols with details
        """
        try:
            data = {"exchange": [exchange]}
            response = self.client.fyers.symbolmaster(data=data)
            
            if response['s'] == 'ok':
                return response['d']
            else:
                print(f"❌ Symbol master error: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in symbol master: {str(e)}")
            return None
    
    def place_advanced_order(self, symbol: str, side: int, qty: int, 
                           order_type: int, price: float = 0,
                           stop_loss: float = 0, take_profit: float = 0,
                           disclosed_qty: int = 0, validity: str = "DAY") -> Optional[Dict]:
        """
        Place advanced orders with stop loss and take profit
        
        Args:
            symbol: Trading symbol (e.g., "NSE:RELIANCE-EQ")
            side: 1 for BUY, -1 for SELL
            qty: Quantity to trade
            order_type: 1=MARKET, 2=LIMIT, 3=STOP_MARKET, 4=STOP_LIMIT
            price: Limit price (for LIMIT orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            disclosed_qty: Disclosed quantity for iceberg orders
            validity: "DAY" or "IOC"
        """
        try:
            order_data = {
                "symbol": symbol,
                "qty": qty,
                "type": order_type,
                "side": side,
                "productType": "CNC",  # CNC for delivery, INTRADAY for intraday
                "limitPrice": price if order_type in [2, 4] else 0,
                "stopPrice": 0,  # For stop orders
                "validity": validity,
                "disclosedQty": disclosed_qty,
                "offlineOrder": "False"
            }
            
            # Place main order
            response = self.client.fyers.place_order(data=order_data)
            
            if response['s'] == 'ok':
                order_id = response['id']
                print(f"✅ Order placed successfully: {order_id}")
                
                # Place stop loss order if specified
                if stop_loss > 0:
                    self._place_sl_order(symbol, -side, qty, stop_loss, order_id)
                
                # Place take profit order if specified  
                if take_profit > 0:
                    self._place_tp_order(symbol, -side, qty, take_profit, order_id)
                
                return {"order_id": order_id, "status": "success"}
            else:
                print(f"❌ Order placement error: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in order placement: {str(e)}")
            return None
    
    def _place_sl_order(self, symbol: str, side: int, qty: int, 
                       stop_price: float, parent_order_id: str):
        """Place stop loss order"""
        try:
            sl_data = {
                "symbol": symbol,
                "qty": qty,
                "type": 3,  # STOP_MARKET
                "side": side,
                "productType": "CNC",
                "stopPrice": stop_price,
                "validity": "DAY",
                "offlineOrder": "False"
            }
            
            response = self.client.fyers.place_order(data=sl_data)
            if response['s'] == 'ok':
                print(f"✅ Stop loss order placed: {response['id']}")
            else:
                print(f"❌ Stop loss order failed: {response.get('message')}")
                
        except Exception as e:
            print(f"❌ Stop loss order exception: {str(e)}")
    
    def _place_tp_order(self, symbol: str, side: int, qty: int, 
                       target_price: float, parent_order_id: str):
        """Place take profit order"""
        try:
            tp_data = {
                "symbol": symbol,
                "qty": qty,
                "type": 2,  # LIMIT
                "side": side,
                "productType": "CNC",
                "limitPrice": target_price,
                "validity": "DAY",
                "offlineOrder": "False"
            }
            
            response = self.client.fyers.place_order(data=tp_data)
            if response['s'] == 'ok':
                print(f"✅ Take profit order placed: {response['id']}")
            else:
                print(f"❌ Take profit order failed: {response.get('message')}")
                
        except Exception as e:
            print(f"❌ Take profit order exception: {str(e)}")
    
    def get_trade_book(self) -> Optional[List[Dict]]:
        """Get all executed trades for the day"""
        try:
            response = self.client.fyers.tradebook()
            
            if response['s'] == 'ok':
                return response.get('tradeBook', [])
            else:
                print(f"❌ Tradebook error: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in tradebook: {str(e)}")
            return None
    
    def modify_order(self, order_id: str, qty: int = 0, price: float = 0, 
                    order_type: int = 0) -> Optional[Dict]:
        """Modify existing order"""
        try:
            data = {"id": order_id}
            
            if qty > 0:
                data["qty"] = qty
            if price > 0:
                data["limitPrice"] = price
            if order_type > 0:
                data["type"] = order_type
            
            response = self.client.fyers.modify_order(data=data)
            
            if response['s'] == 'ok':
                print(f"✅ Order modified successfully: {order_id}")
                return response
            else:
                print(f"❌ Order modification error: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in order modification: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order"""
        try:
            data = {"id": order_id}
            response = self.client.fyers.cancel_order(data=data)
            
            if response['s'] == 'ok':
                print(f"✅ Order cancelled successfully: {order_id}")
                return True
            else:
                print(f"❌ Order cancellation error: {response.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Exception in order cancellation: {str(e)}")
            return False
    
    def get_holdings(self) -> Optional[List[Dict]]:
        """Get current stock holdings"""
        try:
            response = self.client.fyers.holdings()
            
            if response['s'] == 'ok':
                return response.get('holdings', [])
            else:
                print(f"❌ Holdings error: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in holdings: {str(e)}")
            return None

def demo_advanced_features():
    """Demonstrate advanced FYERS features"""
    
    print("🚀 FYERS Advanced Features Demo")
    print("=" * 50)
    
    advanced = AdvancedFyersFeatures()
    
    # Test multi-quotes for efficiency
    test_symbols = ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:INFY-EQ", "NSE:HDFCBANK-EQ", "NSE:ITC-EQ"]
    
    print("\n📊 Testing Multi-Quotes (Efficient)...")
    quotes = advanced.get_multi_quotes(test_symbols)
    
    for symbol, data in quotes.items():
        print(f"   {symbol}: ₹{data['ltp']} ({data['change_percent']:.2f}%)")
    
    # Test market depth
    print(f"\n🔍 Testing Market Depth for {test_symbols[0]}...")
    depth = advanced.get_market_depth(test_symbols[0])
    if depth:
        print(f"   ✅ Market depth data received")
    
    # Test holdings
    print(f"\n💼 Testing Holdings...")
    holdings = advanced.get_holdings()
    if holdings is not None:
        print(f"   ✅ Holdings: {len(holdings)} positions")
    
    # Test trade book
    print(f"\n📋 Testing Trade Book...")
    trades = advanced.get_trade_book()
    if trades is not None:
        print(f"   ✅ Trades: {len(trades)} executed trades today")

if __name__ == "__main__":
    demo_advanced_features()