"""
FYERS WebSocket Data Streaming for Real-time Trading
Based on FYERS API v3 Documentation: https://myapi.fyers.in/docsv3
"""

import json
import time
import threading
from datetime import datetime
from typing import List, Dict, Callable, Optional
from fyers_apiv3 import fyersModel
import websocket

class FyersWebSocketStream:
    """Real-time data streaming using FYERS WebSocket API"""
    
    def __init__(self, access_token: str, client_id: str):
        self.access_token = access_token
        self.client_id = client_id
        self.fyers = fyersModel.FyersModel(client_id=client_id, token=access_token)
        
        # WebSocket connection
        self.ws = None
        self.is_connected = False
        
        # Subscribed symbols and callbacks
        self.subscribed_symbols = set()
        self.callbacks = {}
        
        # Data storage
        self.live_data = {}
        self.tick_count = 0
        
        # Threading
        self.ws_thread = None
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if 'd' in data:  # Market data
                for tick in data['d']:
                    symbol = tick.get('symbol', '')
                    
                    # Update live data
                    self.live_data[symbol] = {
                        'ltp': tick.get('ltp', 0),
                        'volume': tick.get('v', 0), 
                        'high': tick.get('h', 0),
                        'low': tick.get('l', 0),
                        'open': tick.get('o', 0),
                        'change': tick.get('ch', 0),
                        'change_percent': tick.get('chp', 0),
                        'bid': tick.get('bid', 0),
                        'ask': tick.get('ask', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.tick_count += 1
                    
                    # Call registered callbacks
                    if symbol in self.callbacks:
                        for callback in self.callbacks[symbol]:
                            try:
                                callback(symbol, self.live_data[symbol])
                            except Exception as e:
                                print(f"❌ Callback error for {symbol}: {e}")
                                
                    # Print live updates (optional)
                    if self.tick_count % 10 == 0:  # Every 10th tick
                        ltp = self.live_data[symbol]['ltp']
                        chp = self.live_data[symbol]['change_percent']
                        direction = "🟢" if chp >= 0 else "🔴"
                        print(f"{direction} {symbol}: ₹{ltp} ({chp:.2f}%)")
                        
        except json.JSONDecodeError:
            print(f"❌ Failed to decode message: {message}")
        except Exception as e:
            print(f"❌ Message processing error: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"❌ WebSocket Error: {error}")
        
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print(f"🔌 WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        
    def on_open(self, ws):
        """Handle WebSocket open"""
        print("✅ WebSocket connection established")
        self.is_connected = True
        
        # Subscribe to symbols if any
        if self.subscribed_symbols:
            self._send_subscription()
    
    def connect(self):
        """Establish WebSocket connection"""
        try:
            # Get WebSocket connection details from FYERS
            ws_response = self.fyers.generate_token()
            
            if 's' in ws_response and ws_response['s'] == 'ok':
                ws_url = f"wss://api-t1.fyers.in/socket/v4/dataSock?user-agent=fyers-api&version=3.0&token={self.access_token}&type=symbolData"
                
                # Create WebSocket connection
                websocket.enableTrace(False)
                self.ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=self.on_message,
                    on_error=self.on_error, 
                    on_close=self.on_close,
                    on_open=self.on_open
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
                    print("🚀 Real-time data streaming started")
                    return True
                else:
                    print("❌ Failed to connect within timeout")
                    return False
                    
            else:
                print(f"❌ WebSocket token error: {ws_response}")
                return False
                
        except Exception as e:
            print(f"❌ WebSocket connection error: {e}")
            return False
    
    def subscribe(self, symbols: List[str], callback: Optional[Callable] = None):
        """Subscribe to symbols for real-time data"""
        for symbol in symbols:
            self.subscribed_symbols.add(symbol)
            
            # Register callback if provided
            if callback:
                if symbol not in self.callbacks:
                    self.callbacks[symbol] = []
                self.callbacks[symbol].append(callback)
        
        # Send subscription if connected
        if self.is_connected:
            self._send_subscription()
        
        print(f"📊 Subscribed to {len(symbols)} symbols")
    
    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)
            if symbol in self.callbacks:
                del self.callbacks[symbol]
        
        # Send unsubscription if connected
        if self.is_connected:
            self._send_unsubscription(symbols)
            
        print(f"📊 Unsubscribed from {len(symbols)} symbols")
    
    def _send_subscription(self):
        """Send subscription message to WebSocket"""
        if self.ws and self.is_connected:
            sub_message = {
                "T": "SUB_L2",
                "L2LIST": list(self.subscribed_symbols),
                "SUB_T": 1
            }
            self.ws.send(json.dumps(sub_message))
    
    def _send_unsubscription(self, symbols: List[str]):
        """Send unsubscription message to WebSocket"""
        if self.ws and self.is_connected:
            unsub_message = {
                "T": "UNSUB_L2", 
                "L2LIST": symbols,
                "SUB_T": 0
            }
            self.ws.send(json.dumps(unsub_message))
    
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
    
    def get_stats(self) -> Dict:
        """Get streaming statistics"""
        return {
            "connected": self.is_connected,
            "subscribed_symbols": len(self.subscribed_symbols),
            "symbols_with_data": len(self.live_data),
            "total_ticks": self.tick_count
        }

class TradingAlerts:
    """Real-time trading alerts based on streaming data"""
    
    def __init__(self, stream: FyersWebSocketStream):
        self.stream = stream
        self.alerts = {}
        
    def add_price_alert(self, symbol: str, target_price: float, 
                       direction: str = "above", callback: Optional[Callable] = None):
        """Add price-based alert"""
        alert_id = f"{symbol}_{direction}_{target_price}"
        
        self.alerts[alert_id] = {
            "symbol": symbol,
            "target_price": target_price,
            "direction": direction,  # "above" or "below"
            "callback": callback,
            "triggered": False,
            "created_at": datetime.now()
        }
        
        # Register callback for this symbol
        self.stream.subscribe([symbol], self._check_price_alerts)
        
        print(f"🔔 Price alert set: {symbol} {direction} ₹{target_price}")
    
    def add_volume_alert(self, symbol: str, volume_threshold: int, 
                        callback: Optional[Callable] = None):
        """Add volume-based alert"""
        alert_id = f"{symbol}_volume_{volume_threshold}"
        
        self.alerts[alert_id] = {
            "symbol": symbol,
            "volume_threshold": volume_threshold,
            "callback": callback,
            "triggered": False,
            "created_at": datetime.now()
        }
        
        self.stream.subscribe([symbol], self._check_volume_alerts)
        
        print(f"🔔 Volume alert set: {symbol} volume > {volume_threshold:,}")
    
    def _check_price_alerts(self, symbol: str, data: Dict):
        """Check price-based alerts"""
        current_price = data.get('ltp', 0)
        
        for alert_id, alert in self.alerts.items():
            if (alert['symbol'] == symbol and 
                'target_price' in alert and 
                not alert['triggered']):
                
                target = alert['target_price']
                direction = alert['direction']
                
                triggered = False
                if direction == "above" and current_price >= target:
                    triggered = True
                elif direction == "below" and current_price <= target:
                    triggered = True
                
                if triggered:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now()
                    alert['trigger_price'] = current_price
                    
                    print(f"🚨 PRICE ALERT: {symbol} is ₹{current_price} ({direction} ₹{target})")
                    
                    if alert['callback']:
                        alert['callback'](alert_id, alert, data)
    
    def _check_volume_alerts(self, symbol: str, data: Dict):
        """Check volume-based alerts"""
        current_volume = data.get('volume', 0)
        
        for alert_id, alert in self.alerts.items():
            if (alert['symbol'] == symbol and 
                'volume_threshold' in alert and 
                not alert['triggered']):
                
                if current_volume >= alert['volume_threshold']:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now()
                    alert['trigger_volume'] = current_volume
                    
                    print(f"🚨 VOLUME ALERT: {symbol} volume is {current_volume:,}")
                    
                    if alert['callback']:
                        alert['callback'](alert_id, alert, data)

def demo_streaming():
    """Demonstrate real-time data streaming"""
    
    # Load config
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except:
        print("❌ Please ensure config.json exists with FYERS credentials")
        return
    
    print("🚀 FYERS Real-time Data Streaming Demo")
    print("=" * 60)
    
    # Initialize streaming
    stream = FyersWebSocketStream(
        access_token=config["access_token"],
        client_id=config["client_id"]
    )
    
    # Connect to WebSocket
    if not stream.connect():
        print("❌ Failed to connect to WebSocket")
        return
    
    # Subscribe to popular stocks
    symbols = [
        "NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:INFY-EQ", 
        "NSE:HDFCBANK-EQ", "NSE:ITC-EQ", "NSE:NIFTY50-INDEX"
    ]
    
    # Custom callback for important stocks
    def important_stock_callback(symbol: str, data: Dict):
        if abs(data.get('change_percent', 0)) > 2.0:  # 2% move
            direction = "🚀" if data['change_percent'] > 0 else "📉"
            print(f"{direction} BIG MOVE: {symbol} at ₹{data['ltp']} ({data['change_percent']:.2f}%)")
    
    stream.subscribe(symbols, important_stock_callback)
    
    # Setup alerts
    alerts = TradingAlerts(stream)
    alerts.add_price_alert("NSE:RELIANCE-EQ", 2500, "above")
    alerts.add_volume_alert("NSE:ITC-EQ", 50000000)  # 5 crore volume
    
    print(f"\n📊 Streaming started for {len(symbols)} symbols")
    print("💡 Press Ctrl+C to stop streaming")
    
    try:
        # Let it run for demo
        while True:
            time.sleep(10)
            stats = stream.get_stats()
            print(f"📈 Stats: {stats['total_ticks']} ticks | {stats['symbols_with_data']} symbols with data")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping stream...")
        stream.disconnect()

if __name__ == "__main__":
    demo_streaming()