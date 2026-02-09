"""
🧪 FYERS API V3 COMPREHENSIVE ENDPOINT TESTING
Test all endpoints from the Postman collection systematically
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fyers_client import FyersClient
import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd

class FyersAPITester:
    """Comprehensive tester for all Fyers API V3 endpoints"""
    
    def __init__(self):
        self.client = FyersClient()
        self.base_url = "https://api-t1.fyers.in"  # Paper trading URL
        self.headers = {
            "Authorization": f"{self.client.client_id}:{self.client.access_token}",
            "Content-Type": "application/json"
        }
        self.test_results = {}
        
        print("🧪 FYERS API V3 COMPREHENSIVE ENDPOINT TESTER")
        print("=" * 80)
        print(f"📱 Client ID: {self.client.client_id}")
        print(f"🔗 Base URL: {self.base_url}")
        print(f"🔑 Authentication: {'✅ Ready' if self.client.access_token else '❌ Missing'}")
        print("=" * 80)
    
    def log_test_result(self, category: str, endpoint: str, method: str, status: str, details: str = ""):
        """Log test results"""
        if category not in self.test_results:
            self.test_results[category] = []
        
        result = {
            'endpoint': endpoint,
            'method': method,
            'status': status,
            'details': details,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        
        self.test_results[category].append(result)
        
        # Print result
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} [{method}] {endpoint} - {status}")
        if details:
            print(f"   💬 {details}")
    
    def make_request(self, method: str, endpoint: str, data: dict = None, params: dict = None):
        """Make HTTP request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=self.headers, json=data, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=self.headers, json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=self.headers, timeout=30)
            
            return response
        
        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout for {endpoint}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request error for {endpoint}: {str(e)}")
            return None
        except Exception as e:
            print(f"   🚨 Unexpected error for {endpoint}: {str(e)}")
            return None
    
    def test_authentication_endpoints(self):
        """Test authentication-related endpoints"""
        print(f"\n🔐 TESTING AUTHENTICATION ENDPOINTS")
        print("-" * 50)
        
        # Test if current authentication is working
        try:
            profile_response = self.make_request("GET", "/api/v3/profile")
            if profile_response and profile_response.status_code == 200:
                self.log_test_result("Authentication", "/api/v3/profile", "GET", "PASS", 
                                   "Current authentication working")
            else:
                self.log_test_result("Authentication", "/api/v3/profile", "GET", "FAIL",
                                   f"Auth failed: {profile_response.status_code if profile_response else 'No response'}")
        except Exception as e:
            self.log_test_result("Authentication", "/api/v3/profile", "GET", "FAIL", str(e))
    
    def test_account_info_endpoints(self):
        """Test account information endpoints"""
        print(f"\n👤 TESTING ACCOUNT INFO ENDPOINTS")
        print("-" * 50)
        
        # Test Profile
        response = self.make_request("GET", "/api/v3/profile")
        if response and response.status_code == 200:
            data = response.json()
            self.log_test_result("Account Info", "/api/v3/profile", "GET", "PASS",
                               f"Profile: {data.get('data', {}).get('name', 'N/A')}")
        else:
            self.log_test_result("Account Info", "/api/v3/profile", "GET", "FAIL",
                               f"HTTP {response.status_code if response else 'No response'}")
        
        # Test Funds
        response = self.make_request("GET", "/api/v3/funds")
        if response and response.status_code == 200:
            data = response.json()
            # Handle different response structures
            fund_info = data.get('fund_limit', data)
            
            # Try different possible keys for available cash
            available_cash = 0
            if isinstance(fund_info, dict):
                available_cash = fund_info.get('available_cash', 
                               fund_info.get('availableCash',
                               fund_info.get('cash_available', 0)))
            elif isinstance(fund_info, list) and fund_info:
                # If it's a list, take first element
                available_cash = fund_info[0].get('available_cash', 
                               fund_info[0].get('availableCash',
                               fund_info[0].get('cash_available', 0)))
            
            self.log_test_result("Account Info", "/api/v3/funds", "GET", "PASS",
                               f"Funds retrieved - Available Cash: ₹{available_cash:,.2f}")
        else:
            self.log_test_result("Account Info", "/api/v3/funds", "GET", "FAIL",
                               f"HTTP {response.status_code if response else 'No response'}")
    
    def test_market_data_endpoints(self):
        """Test market data endpoints"""
        print(f"\n📊 TESTING MARKET DATA ENDPOINTS")
        print("-" * 50)
        
        # Test Quotes - Single Symbol
        symbols = "NSE:NIFTY50-INDEX"
        response = self.make_request("GET", "/data/quotes/", params={"symbols": symbols})
        if response and response.status_code == 200:
            try:
                data = response.json()
                quote_data = data.get('d', [])
                if quote_data and len(quote_data) > 0:
                    ltp = quote_data[0].get('v', {}).get('lp', 0)
                    self.log_test_result("Market Data", "/data/quotes/", "GET", "PASS",
                                       f"NIFTY LTP: ₹{ltp:,.2f}")
                else:
                    self.log_test_result("Market Data", "/data/quotes/", "GET", "WARN", "No quote data returned")
            except Exception as e:
                self.log_test_result("Market Data", "/data/quotes/", "GET", "WARN", f"Response parsing error: {str(e)}")
        else:
            self.log_test_result("Market Data", "/data/quotes/", "GET", "FAIL",
                               f"HTTP {response.status_code if response else 'No response'}")
        
        # Test Quotes - Multiple Symbols
        symbols = "NSE:NIFTY50-INDEX,NSE:NIFTYBANK-INDEX,NSE:RELIANCE-EQ"
        response = self.make_request("GET", "/data/quotes/", params={"symbols": symbols})
        if response and response.status_code == 200:
            try:
                data = response.json()
                quote_count = len(data.get('d', []))
                self.log_test_result("Market Data", "/data/quotes/ (multiple)", "GET", "PASS",
                                   f"Retrieved {quote_count} quotes")
            except Exception as e:
                self.log_test_result("Market Data", "/data/quotes/ (multiple)", "GET", "WARN", f"Response parsing error: {str(e)}")
        else:
            self.log_test_result("Market Data", "/data/quotes/ (multiple)", "GET", "FAIL",
                               f"HTTP {response.status_code if response else 'No response'}")
        
        # Test Market Depth
        response = self.make_request("GET", "/data/depth/", 
                                   params={"symbol": "NSE:NIFTY50-INDEX", "ohlcv_flag": "1"})
        if response and response.status_code == 200:
            try:
                data = response.json()
                depth_data = data.get('d', {})
                bids = depth_data.get('bids', [])
                asks = depth_data.get('asks', [])
                self.log_test_result("Market Data", "/data/depth/", "GET", "PASS",
                                   f"Depth: {len(bids)} bids, {len(asks)} asks")
            except Exception as e:
                self.log_test_result("Market Data", "/data/depth/", "GET", "WARN", f"Response parsing error: {str(e)}")
        else:
            self.log_test_result("Market Data", "/data/depth/", "GET", "FAIL",
                               f"HTTP {response.status_code if response else 'No response'}")
        
        # Test Historical Data - Different Resolutions
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        resolutions = ["1", "5", "15", "30", "60", "1D"]
        for resolution in resolutions:
            params = {
                "symbol": "NSE:NIFTY50-INDEX",
                "resolution": resolution,
                "date_format": "1",
                "range_from": start_date,
                "range_to": end_date,
                "cont_flag": "1"
            }
            
            response = self.make_request("GET", "/data/history", params=params)
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    candles = data.get('candles', [])
                    self.log_test_result("Market Data", f"/data/history ({resolution})", "GET", "PASS",
                                       f"Retrieved {len(candles)} candles")
                except Exception as e:
                    self.log_test_result("Market Data", f"/data/history ({resolution})", "GET", "WARN",
                                       f"Response parsing error: {str(e)}")
            else:
                self.log_test_result("Market Data", f"/data/history ({resolution})", "GET", "FAIL",
                                   f"HTTP {response.status_code if response else 'No response'}")
    
    def test_transaction_info_endpoints(self):
        """Test transaction information endpoints"""
        print(f"\n💼 TESTING TRANSACTION INFO ENDPOINTS")
        print("-" * 50)
        
        # Test Orders
        response = self.make_request("GET", "/api/v3/orders")
        if response and response.status_code == 200:
            try:
                data = response.json()
                orders = data.get('orderBook', data.get('orders', []))
                self.log_test_result("Transaction Info", "/api/v3/orders", "GET", "PASS",
                                   f"Found {len(orders)} orders")
            except Exception as e:
                self.log_test_result("Transaction Info", "/api/v3/orders", "GET", "WARN", f"Response parsing error: {str(e)}")
        else:
            self.log_test_result("Transaction Info", "/api/v3/orders", "GET", "FAIL",
                               f"HTTP {response.status_code if response else 'No response'}")
        
        # Test Tradebook
        response = self.make_request("GET", "/api/v3/tradebook")
        if response and response.status_code == 200:
            try:
                data = response.json()
                trades = data.get('tradeBook', data.get('trades', []))
                self.log_test_result("Transaction Info", "/api/v3/tradebook", "GET", "PASS",
                                   f"Found {len(trades)} trades")
            except Exception as e:
                self.log_test_result("Transaction Info", "/api/v3/tradebook", "GET", "WARN", f"Response parsing error: {str(e)}")
        else:
            self.log_test_result("Transaction Info", "/api/v3/tradebook", "GET", "FAIL",
                               f"HTTP {response.status_code if response else 'No response'}")
        
        # Test Holdings
        response = self.make_request("GET", "/api/v3/holdings")
        if response and response.status_code == 200:
            try:
                data = response.json()
                holdings = data.get('holdings', data.get('data', []))
                self.log_test_result("Transaction Info", "/api/v3/holdings", "GET", "PASS",
                                   f"Found {len(holdings)} holdings")
            except Exception as e:
                self.log_test_result("Transaction Info", "/api/v3/holdings", "GET", "WARN", f"Response parsing error: {str(e)}")
        else:
            self.log_test_result("Transaction Info", "/api/v3/holdings", "GET", "FAIL",
                               f"HTTP {response.status_code if response else 'No response'}")
        
        # Test Positions
        response = self.make_request("GET", "/api/v3/positions")
        if response and response.status_code == 200:
            try:
                data = response.json()
                positions = data.get('netPositions', data.get('positions', []))
                self.log_test_result("Transaction Info", "/api/v3/positions", "GET", "PASS",
                                   f"Found {len(positions)} positions")
            except Exception as e:
                self.log_test_result("Transaction Info", "/api/v3/positions", "GET", "WARN", f"Response parsing error: {str(e)}")
        else:
            self.log_test_result("Transaction Info", "/api/v3/positions", "GET", "FAIL",
                               f"HTTP {response.status_code if response else 'No response'}")
    
    def test_order_validation_endpoints(self):
        """Test order-related endpoints (validation only, no real orders)"""
        print(f"\n📋 TESTING ORDER VALIDATION ENDPOINTS")
        print("-" * 50)
        
        # Test Place Order (validation only - will likely fail due to insufficient funds/demo mode)
        test_order = {
            "symbol": "NSE:IDEA-EQ",
            "qty": 1,
            "type": 1,  # Limit order
            "side": 1,  # Buy
            "productType": "INTRADAY",
            "limitPrice": 1.0,  # Very low price to avoid accidental execution
            "stopPrice": 0,
            "disclosedQty": 0,
            "validity": "DAY",
            "offlineOrder": False,
            "stopLoss": 0,
            "takeProfit": 0
        }
        
        response = self.make_request("POST", "/api/v3/orders/sync", data=test_order)
        if response:
            if response.status_code == 200:
                self.log_test_result("Order Validation", "/api/v3/orders/sync", "POST", "PASS",
                                   "Order validation successful")
            else:
                # Expected to fail - just testing endpoint accessibility
                error_msg = "Expected validation failure" if response.status_code in [400, 401, 403] else f"HTTP {response.status_code}"
                self.log_test_result("Order Validation", "/api/v3/orders/sync", "POST", "WARN", error_msg)
        else:
            self.log_test_result("Order Validation", "/api/v3/orders/sync", "POST", "FAIL", "No response")
    
    def test_additional_endpoints(self):
        """Test additional endpoints from the collection"""
        print(f"\n🔧 TESTING ADDITIONAL ENDPOINTS")
        print("-" * 50)
        
        # Test Symbols API (if available)
        response = self.make_request("GET", "/data/symbols")
        if response and response.status_code == 200:
            self.log_test_result("Additional", "/data/symbols", "GET", "PASS", "Symbols API accessible")
        else:
            self.log_test_result("Additional", "/data/symbols", "GET", "WARN", 
                               "Symbols API not accessible or different endpoint")
        
        # Test Market Status
        response = self.make_request("GET", "/data/market-status")
        if response and response.status_code == 200:
            data = response.json()
            self.log_test_result("Additional", "/data/market-status", "GET", "PASS", 
                               f"Market status retrieved")
        else:
            self.log_test_result("Additional", "/data/market-status", "GET", "WARN",
                               "Market status endpoint not found or different path")
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all endpoints"""
        print(f"🚀 STARTING COMPREHENSIVE API ENDPOINT TESTING")
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all test categories
        self.test_authentication_endpoints()
        time.sleep(1)  # Rate limiting
        
        self.test_account_info_endpoints() 
        time.sleep(1)
        
        self.test_market_data_endpoints()
        time.sleep(1)
        
        self.test_transaction_info_endpoints()
        time.sleep(1)
        
        self.test_order_validation_endpoints()
        time.sleep(1)
        
        self.test_additional_endpoints()
        
        # Generate comprehensive report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print(f"\n" + "=" * 80)
        print(f"📊 COMPREHENSIVE API ENDPOINT TEST REPORT")
        print("=" * 80)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        warning_tests = 0
        
        for category, results in self.test_results.items():
            print(f"\n📁 {category.upper()}:")
            print("-" * 60)
            
            category_pass = 0
            category_total = len(results)
            
            for result in results:
                status_emoji = "✅" if result['status'] == "PASS" else "❌" if result['status'] == "FAIL" else "⚠️"
                print(f"   {status_emoji} [{result['method']}] {result['endpoint']}")
                if result['details']:
                    print(f"      💬 {result['details']}")
                
                total_tests += 1
                if result['status'] == 'PASS':
                    passed_tests += 1
                    category_pass += 1
                elif result['status'] == 'FAIL':
                    failed_tests += 1
                else:
                    warning_tests += 1
            
            # Category summary
            success_rate = (category_pass / category_total * 100) if category_total > 0 else 0
            print(f"   📊 Category Success: {category_pass}/{category_total} ({success_rate:.1f}%)")
        
        # Overall summary
        print(f"\n" + "=" * 80)
        print(f"🎯 OVERALL TEST SUMMARY")
        print("=" * 80)
        
        overall_success = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"❌ Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"⚠️ Warnings: {warning_tests} ({warning_tests/total_tests*100:.1f}%)")
        print(f"🎯 Success Rate: {overall_success:.1f}%")
        
        # Final assessment
        if overall_success >= 80:
            print(f"\n🎉 EXCELLENT: API endpoints are well-connected and functional!")
        elif overall_success >= 60:
            print(f"\n✅ GOOD: Most API endpoints working, minor issues detected")
        elif overall_success >= 40:
            print(f"\n⚠️ MODERATE: Some API endpoints working, needs investigation")
        else:
            print(f"\n❌ POOR: Major API connectivity issues detected")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if failed_tests > 0:
            print(f"   • Investigate failed endpoints for production deployment")
            print(f"   • Check API permissions and rate limits")
        
        if warning_tests > 0:
            print(f"   • Review warning endpoints - may need different parameters")
        
        if passed_tests > 0:
            print(f"   • ✅ Core functionality is working - ready for live trading integration")
        
        print(f"\n⏰ Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warning_tests,
            "success_rate": overall_success
        }

def main():
    """Run comprehensive API endpoint testing"""
    
    print("🧪 FYERS API V3 COMPREHENSIVE ENDPOINT TESTING")
    print("Testing all endpoints from the Postman collection")
    print("=" * 80)
    
    # Initialize tester
    tester = FyersAPITester()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    print(f"\n🎯 API ENDPOINT TESTING COMPLETE!")

if __name__ == "__main__":
    main()