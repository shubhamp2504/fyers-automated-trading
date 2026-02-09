"""
🔌 FYERS API V3 WEBSOCKET ENDPOINT TESTING (FIXED VERSION)
Test WebSocket connections using proper Fyers API implementation
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fyers_client import FyersClient
import json
import time
import threading
from datetime import datetime
from websocket_stream import FyersWebSocketStream

class FyersWebSocketTesterFixed:
    """Test WebSocket endpoints using the proper Fyers implementation"""
    
    def __init__(self):
        self.client = FyersClient()
        self.ws_stream = None
        self.test_results = []
        self.received_data = []
        
        print("🔌 FYERS WEBSOCKET ENDPOINT TESTER (FIXED)")
        print("=" * 80)
        print(f"📱 Client ID: {self.client.client_id}")
        print(f"🔑 Access Token: {'✅ Ready' if self.client.access_token else '❌ Missing'}")
        print("=" * 80)
    
    def test_websocket_connection(self):
        """Test WebSocket connection using proper implementation"""
        print(f"\n🔌 TESTING WEBSOCKET CONNECTION")
        print("-" * 50)
        
        try:
            # Initialize WebSocket stream
            print(f"   🚀 Initializing WebSocket stream...")
            self.ws_stream = FyersWebSocketStream(
                access_token=self.client.access_token,
                client_id=self.client.client_id
            )
            
            # Connect to WebSocket
            print(f"   🔗 Connecting to WebSocket...")
            connected = self.ws_stream.connect()
            
            if connected:
                self.test_results.append({
                    'test': 'WebSocket Connection',
                    'status': 'PASS',
                    'details': 'Successfully connected to WebSocket'
                })
                print(f"   ✅ WebSocket connected successfully!")
                return True
            else:
                self.test_results.append({
                    'test': 'WebSocket Connection',
                    'status': 'FAIL',
                    'details': 'Failed to establish WebSocket connection'
                })
                print(f"   ❌ Failed to connect to WebSocket")
                return False
                
        except Exception as e:
            self.test_results.append({
                'test': 'WebSocket Connection', 
                'status': 'ERROR',
                'details': str(e)
            })
            print(f"   🚨 WebSocket connection error: {str(e)}")
            return False
    
    def test_symbol_subscriptions(self):
        """Test subscribing to different symbols"""
        print(f"\n📊 TESTING SYMBOL SUBSCRIPTIONS")
        print("-" * 50)
        
        if not self.ws_stream or not self.ws_stream.is_connected:
            print(f"   ⚠️ WebSocket not connected, skipping subscription tests")
            return
        
        # Test different symbol subscriptions
        test_symbols = [
            {
                'name': 'Index Symbols',
                'symbols': ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
            },
            {
                'name': 'Equity Symbols', 
                'symbols': ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:INFY-EQ']
            },
            {
                'name': 'Individual Test',
                'symbols': ['NSE:ITC-EQ']
            }
        ]
        
        for test_case in test_symbols:
            try:
                print(f"   🧪 Testing subscription: {test_case['name']}")
                
                # Subscribe with callback
                def data_callback(symbol, data):
                    self.received_data.append({
                        'symbol': symbol,
                        'ltp': data.get('ltp', 0),
                        'change_percent': data.get('change_percent', 0),
                        'timestamp': data.get('timestamp')
                    })
                
                # Subscribe to symbols
                self.ws_stream.subscribe(test_case['symbols'], callback=data_callback)
                
                # Wait for some data
                print(f"      ⏰ Collecting data for 5 seconds...")
                time.sleep(5)
                
                # Check if we received data
                symbol_data = []
                for symbol in test_case['symbols']:
                    live_data = self.ws_stream.get_live_data(symbol)
                    if live_data:
                        symbol_data.append(f"{symbol}: ₹{live_data['ltp']}")
                
                if symbol_data:
                    self.test_results.append({
                        'test': f'Subscription - {test_case["name"]}',
                        'status': 'PASS',
                        'details': f'Received data from {len(symbol_data)} symbols'
                    })
                    print(f"      ✅ Data received: {', '.join(symbol_data[:2])}...")
                else:
                    self.test_results.append({
                        'test': f'Subscription - {test_case["name"]}',
                        'status': 'WARN',
                        'details': 'No data received during test period'
                    })
                    print(f"      ⚠️ No data received")
                
                # Unsubscribe
                self.ws_stream.unsubscribe(test_case['symbols'])
                time.sleep(1)  # Wait between tests
                
            except Exception as e:
                self.test_results.append({
                    'test': f'Subscription - {test_case["name"]}',
                    'status': 'ERROR',
                    'details': str(e)
                })
                print(f"      🚨 Subscription error: {str(e)}")
    
    def test_real_time_data_quality(self):
        """Test real-time data quality and frequency"""
        print(f"\n📈 TESTING REAL-TIME DATA QUALITY")
        print("-" * 50)
        
        if not self.ws_stream or not self.ws_stream.is_connected:
            print(f"   ⚠️ WebSocket not connected, skipping quality tests")
            return
        
        # Subscribe to high-volume symbols
        test_symbols = ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
        
        print(f"   📊 Testing data quality for: {', '.join(test_symbols)}")
        
        # Track received data
        data_received = []
        
        def quality_callback(symbol, data):
            data_received.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'ltp': data.get('ltp', 0),
                'volume': data.get('volume', 0)
            })
        
        try:
            # Subscribe
            self.ws_stream.subscribe(test_symbols, callback=quality_callback)
            
            print(f"   ⏰ Collecting data for 10 seconds...")
            start_time = time.time()
            
            while time.time() - start_time < 10:
                print(f"      📦 Data points received: {len(data_received)}", end='\r')
                time.sleep(0.5)
            
            print(f"\n   📊 Analysis results:")
            
            # Analyze data quality
            if data_received:
                symbols_with_data = set(item['symbol'] for item in data_received)
                avg_frequency = len(data_received) / 10  # per second
                
                print(f"      • Total data points: {len(data_received)}")
                print(f"      • Symbols with data: {len(symbols_with_data)}")
                print(f"      • Average frequency: {avg_frequency:.2f} updates/sec")
                
                # Show latest prices
                for symbol in symbols_with_data:
                    latest_data = self.ws_stream.get_live_data(symbol)
                    if latest_data:
                        ltp = latest_data['ltp']
                        chp = latest_data.get('change_percent', 0)
                        print(f"      • {symbol}: ₹{ltp} ({chp:+.2f}%)")
                
                # Result assessment
                if len(data_received) >= 10:  # At least 1 update per second
                    self.test_results.append({
                        'test': 'Real-time Data Quality',
                        'status': 'PASS',
                        'details': f'{len(data_received)} updates, {avg_frequency:.2f}/sec'
                    })
                    print(f"   ✅ Data quality: GOOD")
                else:
                    self.test_results.append({
                        'test': 'Real-time Data Quality',
                        'status': 'WARN',
                        'details': f'Low update frequency: {avg_frequency:.2f}/sec'
                    })
                    print(f"   ⚠️ Data quality: LOW FREQUENCY")
            else:
                self.test_results.append({
                    'test': 'Real-time Data Quality',
                    'status': 'FAIL',
                    'details': 'No real-time data received'
                })
                print(f"   ❌ Data quality: NO DATA")
            
            # Unsubscribe
            self.ws_stream.unsubscribe(test_symbols)
            
        except Exception as e:
            self.test_results.append({
                'test': 'Real-time Data Quality',
                'status': 'ERROR',
                'details': str(e)
            })
            print(f"   🚨 Data quality test error: {str(e)}")
    
    def cleanup(self):
        """Clean up WebSocket connections"""
        print(f"\n🧹 CLEANING UP")
        print("-" * 50)
        
        if self.ws_stream:
            try:
                self.ws_stream.disconnect()
                print(f"   ✅ WebSocket disconnected")
            except Exception as e:
                print(f"   ⚠️ Cleanup warning: {str(e)}")
    
    def generate_report(self):
        """Generate comprehensive WebSocket test report"""
        print(f"\n" + "=" * 80)
        print(f"🔌 WEBSOCKET ENDPOINT TEST REPORT")
        print("=" * 80)
        
        passed = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAIL')
        errors = sum(1 for r in self.test_results if r['status'] == 'ERROR')
        warnings = sum(1 for r in self.test_results if r['status'] == 'WARN')
        
        total = len(self.test_results)
        
        # Show results
        for result in self.test_results:
            status_emoji = {"PASS": "✅", "FAIL": "❌", "ERROR": "🚨", "WARN": "⚠️"}
            emoji = status_emoji.get(result['status'], "❓")
            print(f"   {emoji} {result['test']}: {result['status']}")
            if result['details']:
                print(f"      💬 {result['details']}")
        
        # Summary
        print(f"\n📊 WEBSOCKET TEST SUMMARY:")
        print(f"   • Total Tests: {total}")
        print(f"   • ✅ Passed: {passed}")
        print(f"   • ❌ Failed: {failed}")
        print(f"   • 🚨 Errors: {errors}")
        print(f"   • ⚠️ Warnings: {warnings}")
        print(f"   • 📦 Data Points: {len(self.received_data)}")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"   • 🎯 Success Rate: {success_rate:.1f}%")
        
        # Assessment
        if passed >= 3:
            print(f"\n🎉 EXCELLENT: WebSocket functionality is working!")
            print(f"   • Real-time data streaming operational")
            print(f"   • Ready for live trading integration")
        elif passed >= 1:
            print(f"\n✅ PARTIAL: Basic WebSocket functionality working")
            print(f"   • Some features may need investigation")
        else:
            print(f"\n❌ POOR: WebSocket functionality has issues")
            print(f"   • Requires troubleshooting before live trading")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if passed > 0:
            print(f"   • ✅ WebSocket infrastructure ready for production")
            print(f"   • 📊 Consider implementing data buffering for high-frequency updates")
            print(f"   • 🔄 Add reconnection logic for network interruptions")
        
        if errors > 0 or failed > 0:
            print(f"   • 🔍 Investigate failed endpoints for production readiness")
            print(f"   • 📞 Contact Fyers support if authentication issues persist")
        
        print("=" * 80)
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "warnings": warnings,
            "success_rate": success_rate if total > 0 else 0
        }
    
    def run_comprehensive_test(self):
        """Run comprehensive WebSocket endpoint testing"""
        print(f"🚀 STARTING COMPREHENSIVE WEBSOCKET TESTING (FIXED)")
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Test connection
            connected = self.test_websocket_connection()
            
            if connected:
                # Test subscriptions
                self.test_symbol_subscriptions()
                
                # Test data quality
                self.test_real_time_data_quality()
            
            # Generate report
            results = self.generate_report()
            
            return results
            
        finally:
            # Always cleanup
            self.cleanup()

def main():
    """Run comprehensive WebSocket endpoint testing"""
    
    print("🔌 FYERS API V3 WEBSOCKET ENDPOINT TESTING (FIXED)")
    print("Testing WebSocket connections using proper Fyers implementation")
    print("=" * 80)
    
    # Initialize tester
    tester = FyersWebSocketTesterFixed()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    print(f"\n🎯 WEBSOCKET ENDPOINT TESTING COMPLETE!")

if __name__ == "__main__":
    main()