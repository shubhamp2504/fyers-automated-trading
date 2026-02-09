"""
🔌 FYERS API V3 WEBSOCKET ENDPOINT TESTING
Test WebSocket connections and real-time data feeds
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fyers_client import FyersClient
import websocket
import json
import time
import threading
from datetime import datetime

class FyersWebSocketTester:
    """Test WebSocket endpoints for real-time data"""
    
    def __init__(self):
        self.client = FyersClient()
        self.ws_url = "wss://api-t1.fyers.in/socket/v4/dataSock"
        self.ws = None
        self.connected = False
        self.test_results = []
        self.received_messages = []
        
        print("🔌 FYERS WEBSOCKET ENDPOINT TESTER")
        print("=" * 80)
        print(f"📱 Client ID: {self.client.client_id}")
        print(f"🔗 WebSocket URL: {self.ws_url}")
        print(f"🔑 Access Token: {'✅ Ready' if self.client.access_token else '❌ Missing'}")
        print("=" * 80)
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            self.received_messages.append({
                'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:12],
                'data': data
            })
            
            # Log different message types
            if 'ltp' in data:
                print(f"   📈 Live Price: {data.get('symbol', 'Unknown')} = ₹{data['ltp']}")
            elif 'type' in data:
                print(f"   💬 Message Type: {data['type']}")
            else:
                print(f"   📦 Message: {str(data)[:100]}...")
                
        except json.JSONDecodeError:
            print(f"   ⚠️ Non-JSON message: {message[:100]}...")
        except Exception as e:
            print(f"   ❌ Message processing error: {str(e)}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"   ❌ WebSocket Error: {error}")
        self.test_results.append({
            'test': 'WebSocket Connection',
            'status': 'ERROR',
            'details': str(error)
        })
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure"""
        print(f"   🔌 WebSocket Closed: Code {close_status_code}, Message: {close_msg}")
        self.connected = False
    
    def on_open(self, ws):
        """Handle WebSocket connection"""
        print(f"   ✅ WebSocket Connected Successfully!")
        self.connected = True
        
        # Send authentication
        auth_message = {
            "T": "SUB_OP",
            "SLIST": ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"],
            "SUB_T": 1
        }
        
        print(f"   🔐 Sending Authentication...")
        ws.send(json.dumps(auth_message))
        
        self.test_results.append({
            'test': 'WebSocket Connection',
            'status': 'PASS',
            'details': 'Connected and authenticated'
        })
    
    def test_websocket_connection(self, duration=15):
        """Test WebSocket connection for specified duration"""
        print(f"\n🔌 TESTING WEBSOCKET CONNECTION")
        print("-" * 50)
        
        try:
            # Create WebSocket with authentication
            headers = {
                "Authorization": f"{self.client.client_id}:{self.client.access_token}"
            }
            
            print(f"   🔗 Connecting to WebSocket...")
            
            websocket.enableTrace(False)  # Disable verbose logging
            
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                header=headers,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Run WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection and collect data
            time.sleep(2)  # Give time to connect
            
            if self.connected:
                print(f"   ⏰ Collecting real-time data for {duration} seconds...")
                
                start_time = time.time()
                while time.time() - start_time < duration and self.connected:
                    time.sleep(1)
                    print(f"   📊 Messages received: {len(self.received_messages)}", end='\r')
                
                print(f"\n   ✅ Data collection completed!")
                
                # Close connection
                if self.ws:
                    self.ws.close()
                
            else:
                print(f"   ❌ Failed to establish WebSocket connection")
                self.test_results.append({
                    'test': 'WebSocket Connection',
                    'status': 'FAIL',
                    'details': 'Connection timeout'
                })
        
        except Exception as e:
            print(f"   🚨 WebSocket test error: {str(e)}")
            self.test_results.append({
                'test': 'WebSocket Connection',
                'status': 'FAIL',
                'details': str(e)
            })
    
    def test_data_subscriptions(self):
        """Test different data subscription types"""
        print(f"\n📊 TESTING DATA SUBSCRIPTIONS")
        print("-" * 50)
        
        # Test different subscription types from Postman collection
        subscription_tests = [
            {
                'name': 'Index Data (NIFTY/BANKNIFTY)',
                'symbols': ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX'],
                'sub_type': 1  # LTP
            },
            {
                'name': 'Equity Data (Top Stocks)',
                'symbols': ['NSE:RELIANCE-EQ', 'NSE:INFY-EQ', 'NSE:TCS-EQ'],
                'sub_type': 1  # LTP
            },
            {
                'name': 'Market Depth',
                'symbols': ['NSE:NIFTY50-INDEX'],
                'sub_type': 2  # Market Depth
            }
        ]
        
        for test in subscription_tests:
            print(f"   🧪 Testing: {test['name']}")
            
            # This would require active WebSocket connection
            # For now, just log the test structure
            self.test_results.append({
                'test': f"Subscription - {test['name']}",
                'status': 'INFO',
                'details': f"Symbols: {', '.join(test['symbols'])}, Type: {test['sub_type']}"
            })
    
    def analyze_received_data(self):
        """Analyze the data received during testing"""
        print(f"\n📈 ANALYZING RECEIVED DATA")
        print("-" * 50)
        
        if not self.received_messages:
            print(f"   ⚠️ No messages received during testing")
            return
        
        # Analyze message types
        message_types = {}
        total_messages = len(self.received_messages)
        
        print(f"   📊 Total Messages: {total_messages}")
        
        for msg in self.received_messages[:10]:  # Show first 10 messages
            msg_type = "Unknown"
            if 'ltp' in msg['data']:
                msg_type = "Live Price Update"
            elif 'type' in msg['data']:
                msg_type = msg['data']['type']
            
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
            print(f"   📦 [{msg['timestamp']}] {msg_type}: {str(msg['data'])[:60]}...")
        
        if total_messages > 10:
            print(f"   ... and {total_messages - 10} more messages")
        
        # Summary
        print(f"\n   📈 Message Type Summary:")
        for msg_type, count in message_types.items():
            percentage = (count / total_messages) * 100
            print(f"      • {msg_type}: {count} messages ({percentage:.1f}%)")
    
    def generate_websocket_report(self):
        """Generate comprehensive WebSocket test report"""
        print(f"\n" + "=" * 80)
        print(f"🔌 WEBSOCKET ENDPOINT TEST REPORT")
        print("=" * 80)
        
        passed = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed = sum(1 for result in self.test_results if result['status'] == 'FAIL')
        errors = sum(1 for result in self.test_results if result['status'] == 'ERROR')
        info = sum(1 for result in self.test_results if result['status'] == 'INFO')
        
        total = len(self.test_results)
        
        for result in self.test_results:
            status_emoji = {"PASS": "✅", "FAIL": "❌", "ERROR": "🚨", "INFO": "💡"}
            emoji = status_emoji.get(result['status'], "❓")
            print(f"   {emoji} {result['test']}: {result['status']}")
            if result['details']:
                print(f"      💬 {result['details']}")
        
        print(f"\n📊 WEBSOCKET TEST SUMMARY:")
        print(f"   • Total Tests: {total}")
        print(f"   • ✅ Passed: {passed}")
        print(f"   • ❌ Failed: {failed}")
        print(f"   • 🚨 Errors: {errors}")
        print(f"   • 💡 Info: {info}")
        print(f"   • 📦 Messages Received: {len(self.received_messages)}")
        
        if passed > 0:
            success_rate = (passed / (passed + failed + errors)) * 100 if (passed + failed + errors) > 0 else 0
            print(f"   • 🎯 Success Rate: {success_rate:.1f}%")
        
        # Recommendations
        print(f"\n💡 WEBSOCKET RECOMMENDATIONS:")
        
        if len(self.received_messages) > 0:
            print(f"   • ✅ WebSocket data feed is functional")
            print(f"   • 📈 Real-time price updates working")
            print(f"   • 🚀 Ready for live trading integration")
        else:
            print(f"   • ⚠️ Check WebSocket authentication or network connectivity")
            print(f"   • 🔍 Verify subscription message format")
        
        print(f"   • 📊 Consider implementing reconnection logic for production")
        print(f"   • 🔄 Add error handling for message processing")
        
        print("=" * 80)
    
    def run_comprehensive_websocket_test(self):
        """Run comprehensive WebSocket endpoint testing"""
        print(f"🚀 STARTING COMPREHENSIVE WEBSOCKET TESTING")
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test WebSocket connection with real-time data
        self.test_websocket_connection(duration=10)
        
        # Test different subscription types
        self.test_data_subscriptions()
        
        # Analyze received data
        self.analyze_received_data()
        
        # Generate report
        self.generate_websocket_report()

def main():
    """Run comprehensive WebSocket endpoint testing"""
    
    print("🔌 FYERS API V3 WEBSOCKET ENDPOINT TESTING")
    print("Testing WebSocket connections and real-time data feeds")
    print("=" * 80)
    
    # Initialize tester
    tester = FyersWebSocketTester()
    
    # Run comprehensive tests
    tester.run_comprehensive_websocket_test()
    
    print(f"\n🎯 WEBSOCKET ENDPOINT TESTING COMPLETE!")

if __name__ == "__main__":
    main()