"""
FYERS API v3 - WebSocket Streaming Reference Implementation
=========================================================

Source: https://myapi.fyers.in/docsv3/#tag/Data-Streaming

Complete WebSocket implementation for real-time data streaming.
"""

import json
import time
import threading
import websocket
from typing import Dict, List, Callable, Optional
from datetime import datetime
from fyers_apiv3 import fyersModel

class FyersWebSocketReference:
    """Complete WebSocket reference implementation"""
    
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.fyers = fyersModel.FyersModel(client_id=client_id, token=access_token)
        
        # WebSocket connection
        self.ws = None
        self.is_connected = False
        self.ws_thread = None
        
        # Subscription management
        self.subscribed_symbols = set()
        self.subscription_types = {}  # symbol -> subscription type mapping
        self.callbacks = {}  # symbol -> list of callbacks
        
        # Data storage
        self.live_data = {}
        self.tick_count = 0
        
        # WebSocket URL (will be set during connection)
        self.ws_url = None
    
    def on_open(self, ws):
        """WebSocket connection opened"""
        print("✅ WebSocket connection established")
        self.is_connected = True
        
        # Subscribe to any pre-configured symbols
        if self.subscribed_symbols:
            self._send_subscription_message()
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if 'type' in data:
                if data['type'] == 'connect':
                    print("📡 WebSocket connection confirmed")
                elif data['type'] == 'subscribe':
                    print(f"✅ Subscription confirmed for symbols")
                elif data['type'] == 'unsubscribe':
                    print(f"✅ Unsubscription confirmed")
                    
            # Handle market data
            if 'd' in data and isinstance(data['d'], list):
                for tick_data in data['d']:
                    self._process_tick_data(tick_data)
                    
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
        except Exception as e:
            print(f"❌ Message processing error: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"❌ WebSocket Error: {error}")
        
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print(f"🔌 WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
    
    def _process_tick_data(self, tick_data: Dict):
        """Process individual tick data"""
        try:
            symbol = tick_data.get('symbol', tick_data.get('n', ''))
            
            if symbol:
                # Store live data
                self.live_data[symbol] = {
                    'symbol': symbol,
                    'ltp': tick_data.get('ltp', tick_data.get('lp', 0)),
                    'volume': tick_data.get('vol_traded_today', tick_data.get('v', 0)),
                    'high': tick_data.get('high_price', tick_data.get('h', 0)),
                    'low': tick_data.get('low_price', tick_data.get('l', 0)),
                    'open': tick_data.get('open_price', tick_data.get('o', 0)),
                    'change': tick_data.get('ch', 0),
                    'change_percent': tick_data.get('chp', 0),
                    'bid': tick_data.get('bid', 0),
                    'ask': tick_data.get('ask', 0),
                    'timestamp': datetime.now().isoformat(),
                    'raw_data': tick_data  # Store raw data for reference
                }
                
                self.tick_count += 1
                
                # Call registered callbacks
                if symbol in self.callbacks:
                    for callback in self.callbacks[symbol]:
                        try:
                            callback(symbol, self.live_data[symbol])
                        except Exception as e:
                            print(f"❌ Callback error for {symbol}: {e}")
                            
        except Exception as e:
            print(f"❌ Tick processing error: {e}")
    
    def connect(self) -> bool:
        """Establish WebSocket connection"""
        try:
            # Generate WebSocket token/URL from FYERS API
            # Note: This is a simplified version - check latest docs for exact implementation
            
            # Create WebSocket URL (example format - verify with docs)
            self.ws_url = f"wss://api-t1.fyers.in/socket/v4/dataSock?user-agent=fyers-api&version=3.0&token={self.access_token}&type=symbolData"
            
            # Create WebSocket connection
            websocket.enableTrace(False)  # Set to True for debugging
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start WebSocket in separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection
            timeout = 10
            while not self.is_connected and timeout > 0:
                time.sleep(0.5)
                timeout -= 0.5
            
            if self.is_connected:
                print("🚀 WebSocket streaming started successfully")
                return True
            else:
                print("❌ WebSocket connection timeout")
                return False
                
        except Exception as e:
            print(f"❌ WebSocket connection error: {e}")
            return False
    
    def subscribe_symbols(self, symbols: List[str], 
                         subscription_type: str = "symbolData",
                         callback: Optional[Callable] = None):
        """
        Subscribe to symbols for real-time data
        
        Args:
            symbols: List of symbols to subscribe
            subscription_type: "symbolData", "depthUpdate", etc.
            callback: Optional callback function
        """
        for symbol in symbols:
            self.subscribed_symbols.add(symbol)
            self.subscription_types[symbol] = subscription_type
            
            # Register callback
            if callback:
                if symbol not in self.callbacks:
                    self.callbacks[symbol] = []
                self.callbacks[symbol].append(callback)
        
        # Send subscription if connected
        if self.is_connected:
            self._send_subscription_message()
        
        print(f"📊 Subscribed to {len(symbols)} symbols")
    
    def unsubscribe_symbols(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)
            self.subscription_types.pop(symbol, None)
            self.callbacks.pop(symbol, None)
        
        # Send unsubscription if connected
        if self.is_connected:
            self._send_unsubscription_message(symbols)
        
        print(f"📊 Unsubscribed from {len(symbols)} symbols")
    
    def _send_subscription_message(self):
        """Send subscription message to WebSocket"""
        if self.ws and self.is_connected:
            try:
                # Format subscription message (verify format with docs)
                sub_message = {
                    "T": "SUB_L2",
                    "L2LIST": list(self.subscribed_symbols),
                    "SUB_T": 1
                }
                
                self.ws.send(json.dumps(sub_message))
                print(f"📡 Subscription message sent for {len(self.subscribed_symbols)} symbols")
                
            except Exception as e:
                print(f"❌ Error sending subscription: {e}")
    
    def _send_unsubscription_message(self, symbols: List[str]):
        """Send unsubscription message"""
        if self.ws and self.is_connected:
            try:
                unsub_message = {
                    "T": "UNSUB_L2",
                    "L2LIST": symbols,
                    "SUB_T": 0
                }
                
                self.ws.send(json.dumps(unsub_message))
                print(f"📡 Unsubscription message sent for {len(symbols)} symbols")
                
            except Exception as e:
                print(f"❌ Error sending unsubscription: {e}")
    
    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get latest live data for a symbol"""
        return self.live_data.get(symbol)
    
    def get_all_live_data(self) -> Dict:
        """Get all live data"""
        return self.live_data.copy()
    
    def disconnect(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()
        self.is_connected = False
        print("🔌 WebSocket disconnected")
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            'connected': self.is_connected,
            'subscribed_symbols': len(self.subscribed_symbols),
            'symbols_with_data': len(self.live_data),
            'total_ticks_received': self.tick_count,
            'subscription_types': self.subscription_types.copy()
        }

class WebSocketDataLogger:
    """Log WebSocket data for analysis"""
    
    def __init__(self, log_file: str = "websocket_data.log"):
        self.log_file = log_file
        self.logged_ticks = 0
    
    def log_tick(self, symbol: str, data: Dict):
        """Log tick data to file"""
        try:
            with open(self.log_file, "a") as f:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'data': data
                }
                f.write(json.dumps(log_entry) + "\n")
                self.logged_ticks += 1
        except Exception as e:
            print(f"❌ Logging error: {e}")

def demo_websocket_streaming():
    """Demonstrate WebSocket streaming"""
    
    print("📡 FYERS API v3 - WebSocket Streaming Demo")
    print("=" * 60)
    
    # Load config
    try:
        with open("../config.json", "r") as f:
            config = json.load(f)
        
        client_id = config["client_id"]
        access_token = config["access_token"]
    except:
        print("❌ Please ensure config.json exists with credentials")
        return
    
    # Initialize WebSocket
    ws_client = FyersWebSocketReference(client_id, access_token)
    
    # Initialize data logger
    logger = WebSocketDataLogger("fyers_streaming_data.log")
    
    # Define callback functions
    def price_alert_callback(symbol: str, data: Dict):
        """Alert for significant price moves"""
        change_pct = data.get('change_percent', 0)
        if abs(change_pct) > 2.0:  # Alert for 2%+ moves
            direction = "🚀" if change_pct > 0 else "📉"
            ltp = data.get('ltp', 0)
            print(f"{direction} ALERT: {symbol} moved {change_pct:.2f}% to ₹{ltp}")
    
    def data_logging_callback(symbol: str, data: Dict):
        """Log all data"""
        logger.log_tick(symbol, data)
    
    def volume_alert_callback(symbol: str, data: Dict):
        """Alert for high volume"""
        volume = data.get('volume', 0)
        if volume > 10000000:  # 1 crore volume
            print(f"📊 HIGH VOLUME: {symbol} - {volume:,} shares")
    
    # Test symbols
    test_symbols = [
        "NSE:RELIANCE-EQ",
        "NSE:TCS-EQ", 
        "NSE:INFY-EQ",
        "NSE:HDFCBANK-EQ",
        "NSE:ITC-EQ",
        "NSE:NIFTY50-INDEX",
        "NSE:NIFTYBANK-INDEX"
    ]
    
    # Connect to WebSocket
    if ws_client.connect():
        print(f"✅ WebSocket connected successfully")
        
        # Subscribe with different callbacks
        ws_client.subscribe_symbols(
            test_symbols,
            callback=price_alert_callback
        )
        
        # Add additional callbacks
        for symbol in test_symbols:
            ws_client.callbacks.setdefault(symbol, []).extend([
                data_logging_callback,
                volume_alert_callback
            ])
        
        print(f"\n📊 Monitoring {len(test_symbols)} symbols...")
        print("💡 Press Ctrl+C to stop streaming")
        
        try:
            # Stream for demo duration
            start_time = time.time()
            while time.time() - start_time < 60:  # Stream for 60 seconds
                time.sleep(10)
                
                # Print statistics
                stats = ws_client.get_connection_stats()
                print(f"\n📈 Stats: {stats['total_ticks_received']} ticks | {stats['symbols_with_data']} symbols with data")
                
                # Show sample live data
                live_data = ws_client.get_all_live_data()
                if live_data:
                    for symbol, data in list(live_data.items())[:3]:  # Show first 3
                        ltp = data.get('ltp', 0)
                        change_pct = data.get('change_percent', 0)
                        volume = data.get('volume', 0)
                        direction = "🟢" if change_pct >= 0 else "🔴"
                        print(f"   {direction} {symbol}: ₹{ltp} ({change_pct:.2f}%) Vol: {volume:,}")
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping WebSocket streaming...")
        
        # Disconnect
        ws_client.disconnect()
        
        # Final statistics
        final_stats = ws_client.get_connection_stats()
        print(f"\n📊 Final Statistics:")
        print(f"   • Total ticks received: {final_stats['total_ticks_received']}")
        print(f"   • Data logged: {logger.logged_ticks} entries")
        print(f"   • Symbols monitored: {len(test_symbols)}")
        
    else:
        print("❌ Failed to connect to WebSocket")
    
    print("\n✅ WebSocket streaming demonstration completed!")

if __name__ == "__main__":
    demo_websocket_streaming()