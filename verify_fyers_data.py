from fyers_client import FyersClient
import json
import time
from datetime import datetime, timedelta
import pandas as pd

class FyersDataVerifier:
    def __init__(self):
        self.fyers = FyersClient()
        self.test_symbols = []
        self.test_indices = []
        self.test_results = {
            'quotes': [],
            'indices': [],
            'historical': [],
            'profile': None,
            'funds': None,
            'positions': None,
            'orders': None
        }
    
    def load_verified_symbols(self):
        """Load verified symbols from our master data"""
        try:
            with open('nifty500_verified.json', 'r') as f:
                verified_data = json.load(f)
            
            # Get top 10 most liquid stocks for testing
            top_symbols = sorted(verified_data, 
                               key=lambda x: x['market_data']['volume'], 
                               reverse=True)[:10]
            
            self.test_symbols = [stock['symbol'] for stock in top_symbols]
            print(f"✅ Loaded {len(self.test_symbols)} symbols for verification")
            
            for i, symbol in enumerate(self.test_symbols, 1):
                stock_data = next(s for s in top_symbols if s['symbol'] == symbol)
                print(f"   {i}. {symbol} - Vol: {stock_data['market_data']['volume']:,}")
            
            # Load major Indian indices
            self.test_indices = [
                "NSE:NIFTY50-INDEX",    # Nifty 50
                "NSE:NIFTYBANK-INDEX", # Bank Nifty
                "NSE:NIFTYIT-INDEX",   # Nifty IT
                "NSE:NIFTYFMCG-INDEX", # Nifty FMCG
                "NSE:NIFTYAUTO-INDEX", # Nifty Auto
                "NSE:NIFTYPHARMA-INDEX", # Nifty Pharma
                "NSE:NIFTYMETAL-INDEX",  # Nifty Metal
                "NSE:NIFTYREALTY-INDEX", # Nifty Realty
                "BSE:SENSEX-INDEX",      # BSE Sensex
                "NSE:NIFTYNEXT50-INDEX", # Nifty Next 50
            ]
            
            print(f"✅ Loaded {len(self.test_indices)} indices for verification")
            for i, index in enumerate(self.test_indices, 1):
                print(f"   {i}. {index}")
                
            return True
        except Exception as e:
            print(f"❌ Error loading symbols: {e}")
            return False
    
    def test_market_quotes(self):
        """Test real-time market quote fetching"""
        print("\n🔍 TESTING MARKET QUOTES")
        print("=" * 50)
        
        successful_quotes = 0
        failed_quotes = 0
        
        for i, symbol in enumerate(self.test_symbols, 1):
            try:
                print(f"📊 Testing quotes for {symbol} ({i}/{len(self.test_symbols)})...")
                
                quote = self.fyers.get_market_quote(symbol)
                
                if quote.get('s') == 'ok' and quote.get('d'):
                    data = quote['d'][0]['v']
                    
                    quote_info = {
                        'symbol': symbol,
                        'last_price': data.get('lp', 0),
                        'volume': data.get('volume', 0),
                        'high': data.get('high_price', 0),
                        'low': data.get('low_price', 0),
                        'open': data.get('open_price', 0),
                        'change': data.get('ch', 0),
                        'change_percent': data.get('chp', 0),
                        'bid': data.get('bid', 0),
                        'ask': data.get('ask', 0),
                        'prev_close': data.get('prev_close_price', 0),
                        'timestamp': int(time.time())
                    }
                    
                    self.test_results['quotes'].append(quote_info)
                    successful_quotes += 1
                    
                    change_color = "🟢" if quote_info['change_percent'] >= 0 else "🔴"
                    print(f"   ✅ Price: ₹{quote_info['last_price']} | "
                          f"Vol: {quote_info['volume']:,} | "
                          f"{change_color}{quote_info['change_percent']:.2f}%")
                else:
                    failed_quotes += 1
                    print(f"   ❌ Failed to get quote: {quote.get('message', 'Unknown error')}")
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                failed_quotes += 1
                print(f"   ❌ Exception: {e}")
        
        print(f"\n📊 QUOTES SUMMARY: {successful_quotes} successful, {failed_quotes} failed")
        return successful_quotes > 0
    
    def test_index_quotes(self):
        """Test real-time index quote fetching"""
        print("\n📈 TESTING INDEX QUOTES")
        print("=" * 50)
        
        successful_indices = 0
        failed_indices = 0
        
        for i, index in enumerate(self.test_indices, 1):
            try:
                print(f"📊 Testing quotes for {index} ({i}/{len(self.test_indices)})...")
                
                quote = self.fyers.get_market_quote(index)
                
                if quote.get('s') == 'ok' and quote.get('d'):
                    data = quote['d'][0]['v']
                    
                    index_info = {
                        'symbol': index,
                        'last_price': data.get('lp', 0),
                        'high': data.get('high_price', 0),
                        'low': data.get('low_price', 0),
                        'open': data.get('open_price', 0),
                        'change': data.get('ch', 0),
                        'change_percent': data.get('chp', 0),
                        'prev_close': data.get('prev_close_price', 0),
                        'timestamp': int(time.time())
                    }
                    
                    self.test_results['indices'].append(index_info)
                    successful_indices += 1
                    
                    change_color = "🟢" if index_info['change_percent'] >= 0 else "🔴"
                    print(f"   ✅ Level: {index_info['last_price']:.2f} | "
                          f"High: {index_info['high']:.2f} | "
                          f"Low: {index_info['low']:.2f} | "
                          f"{change_color}{index_info['change_percent']:.2f}%")
                else:
                    failed_indices += 1
                    print(f"   ❌ Failed to get quote: {quote.get('message', 'Unknown error')}")
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                failed_indices += 1
                print(f"   ❌ Exception: {e}")
        
        print(f"\n📈 INDEX QUOTES SUMMARY: {successful_indices} successful, {failed_indices} failed")
        return successful_indices > 0
    
    def test_historical_data(self):
        """Test historical data fetching for stocks and indices"""
        print("\n📈 TESTING HISTORICAL DATA")
        print("=" * 50)
        
        # Test with top 3 symbols and 2 indices
        test_symbols = self.test_symbols[:3] + self.test_indices[:2]
        successful_hist = 0
        failed_hist = 0
        
        # Calculate date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        for symbol in test_symbols:
            try:
                symbol_type = "INDEX" if "INDEX" in symbol else "STOCK"
                print(f"📈 Testing historical data for {symbol} ({symbol_type})...")
                
                # FYERS historical data format
                hist_data = {
                    "symbol": symbol,
                    "resolution": "D",  # Daily data
                    "date_format": "1",
                    "range_from": start_date.strftime("%Y-%m-%d"),
                    "range_to": end_date.strftime("%Y-%m-%d"),
                    "cont_flag": "1"
                }
                
                # Try to get historical data using FYERS model
                response = self.fyers.fyers.history(hist_data)
                
                if response.get('s') == 'ok':
                    candles = response.get('candles', [])
                    print(f"   ✅ Got {len(candles)} historical records")
                    
                    if candles:
                        latest = candles[-1]
                        if symbol_type == "INDEX":
                            print(f"   📊 Latest: Open={latest[1]:.2f}, High={latest[2]:.2f}, "
                                  f"Low={latest[3]:.2f}, Close={latest[4]:.2f}")
                        else:
                            print(f"   📊 Latest: Open=₹{latest[1]}, High=₹{latest[2]}, "
                                  f"Low=₹{latest[3]}, Close=₹{latest[4]}, Vol={latest[5]:,.0f}")
                    
                    self.test_results['historical'].append({
                        'symbol': symbol,
                        'type': symbol_type,
                        'records': len(candles),
                        'data_sample': candles[-5:] if len(candles) >= 5 else candles
                    })
                    
                    successful_hist += 1
                else:
                    failed_hist += 1
                    print(f"   ❌ Failed: {response.get('message', 'Unknown error')}")
                
                time.sleep(1)  # Longer delay for historical data
                
            except Exception as e:
                failed_hist += 1
                print(f"   ❌ Exception: {e}")
        
        print(f"\n📈 HISTORICAL SUMMARY: {successful_hist} successful, {failed_hist} failed")
        return successful_hist > 0
    
    def test_account_info(self):
        """Test account-related API calls"""
        print("\n👤 TESTING ACCOUNT INFORMATION")
        print("=" * 50)
        
        # Test Profile
        try:
            print("📋 Testing profile...")
            profile = self.fyers.get_profile()
            if profile.get('s') == 'ok':
                data = profile['data']
                print(f"   ✅ Profile: {data.get('name', 'N/A')} ({data.get('fy_id', 'N/A')})")
                print(f"   📧 Email: {data.get('email_id', 'N/A')}")
                print(f"   📱 Mobile: {data.get('mobile_number', 'N/A')}")
                self.test_results['profile'] = profile
            else:
                print(f"   ❌ Profile failed: {profile.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"   ❌ Profile exception: {e}")
        
        # Test Funds
        try:
            print("💰 Testing funds...")
            funds = self.fyers.get_funds()
            if funds.get('s') == 'ok':
                fund_data = funds.get('fund_limit', [])
                if fund_data:
                    equity_funds = next((f for f in fund_data if f.get('title') == 'Equity'), {})
                    available = equity_funds.get('available_cash', 0)
                    utilized = equity_funds.get('utilised_amount', 0)
                    print(f"   ✅ Funds - Available: ₹{available:,.2f}, Utilized: ₹{utilized:,.2f}")
                    self.test_results['funds'] = funds
                else:
                    print("   ⚠️  No fund data available")
            else:
                print(f"   ❌ Funds failed: {funds.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"   ❌ Funds exception: {e}")
        
        # Test Positions
        try:
            print("📊 Testing positions...")
            positions = self.fyers.fyers.positions()
            if positions.get('s') == 'ok':
                pos_data = positions.get('netPositions', [])
                day_pos = positions.get('overall', {})
                print(f"   ✅ Positions: {len(pos_data)} net positions")
                
                if day_pos:
                    pnl = day_pos.get('pl_total', 0)
                    pnl_color = "🟢" if pnl >= 0 else "🔴"
                    print(f"   {pnl_color} Total P&L: ₹{pnl:,.2f}")
                
                self.test_results['positions'] = positions
            else:
                print(f"   ❌ Positions failed: {positions.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"   ❌ Positions exception: {e}")
        
        # Test Orders
        try:
            print("📝 Testing orders...")
            orders = self.fyers.fyers.orderbook()
            if orders.get('s') == 'ok':
                order_data = orders.get('orderBook', [])
                print(f"   ✅ Orders: {len(order_data)} orders in orderbook")
                self.test_results['orders'] = orders
            else:
                print(f"   ❌ Orders failed: {orders.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"   ❌ Orders exception: {e}")
    
    def test_market_status(self):
        """Test market status and other market info"""
        print("\n🏛️ TESTING MARKET STATUS")
        print("=" * 50)
        
        try:
            # Test market status
            print("📊 Testing market status...")
            
            # Use a simple quote call to infer market status
            sample_quote = self.fyers.get_market_quote(self.test_symbols[0])
            
            if sample_quote.get('s') == 'ok':
                print("   ✅ Market data is flowing - Market appears to be active")
                
                # Check timestamp to determine market hours
                current_hour = datetime.now().hour
                if 9 <= current_hour <= 15:
                    print("   🟢 Current time suggests market hours (9 AM - 3:30 PM)")
                else:
                    print("   🔴 Current time suggests after market hours")
            else:
                print("   ⚠️  Unable to determine market status")
                
        except Exception as e:
            print(f"   ❌ Market status exception: {e}")
    
    def generate_verification_report(self):
        """Generate a comprehensive verification report"""
        print("\n" + "=" * 80)
        print("📊 FYERS DATA FETCHING VERIFICATION REPORT")
        print("=" * 80)
        
        # Summary stats
        total_quotes = len(self.test_results['quotes'])
        total_indices = len(self.test_results['indices'])
        total_hist = len(self.test_results['historical'])
        
        print(f"🔍 REAL-TIME DATA:")
        print(f"   • Stock quotes tested: {total_quotes} symbols")
        if total_quotes > 0:
            avg_volume = sum(q['volume'] for q in self.test_results['quotes']) / total_quotes
            print(f"   • Average volume: {avg_volume:,.0f}")
            
            # Show price ranges
            prices = [q['last_price'] for q in self.test_results['quotes'] if q['last_price'] > 0]
            if prices:
                print(f"   • Price range: ₹{min(prices):.2f} - ₹{max(prices):,.2f}")
        
        print(f"   • Index quotes tested: {total_indices} indices")
        if total_indices > 0:
            index_levels = [i['last_price'] for i in self.test_results['indices'] if i['last_price'] > 0]
            if index_levels:
                print(f"   • Index level range: {min(index_levels):.2f} - {max(index_levels):,.2f}")
        
        print(f"\n📈 HISTORICAL DATA:")
        print(f"   • Historical data tested: {total_hist} symbols/indices")
        if total_hist > 0:
            total_records = sum(h['records'] for h in self.test_results['historical'])
            print(f"   • Total historical records: {total_records}")
            
            # Break down by type
            stocks_hist = [h for h in self.test_results['historical'] if h.get('type') == 'STOCK']
            indices_hist = [h for h in self.test_results['historical'] if h.get('type') == 'INDEX']
            print(f"   • Stocks with historical data: {len(stocks_hist)}")
            print(f"   • Indices with historical data: {len(indices_hist)}")
        
        print(f"\n👤 ACCOUNT ACCESS:")
        print(f"   • Profile access: {'✅ Success' if self.test_results['profile'] else '❌ Failed'}")
        print(f"   • Funds access: {'✅ Success' if self.test_results['funds'] else '❌ Failed'}")
        print(f"   • Positions access: {'✅ Success' if self.test_results['positions'] else '❌ Failed'}")
        print(f"   • Orders access: {'✅ Success' if self.test_results['orders'] else '❌ Failed'}")
        
        # Save detailed report
        report_data = {
            'verification_timestamp': datetime.now().isoformat(),
            'test_summary': {
                'quotes_tested': total_quotes,
                'indices_tested': total_indices,
                'historical_tested': total_hist,
                'profile_success': bool(self.test_results['profile']),
                'funds_success': bool(self.test_results['funds']),
                'positions_success': bool(self.test_results['positions']),
                'orders_success': bool(self.test_results['orders'])
            },
            'detailed_results': self.test_results
        }
        
        try:
            with open('fyers_verification_report.json', 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n💾 Detailed report saved to: fyers_verification_report.json")
        except Exception as e:
            print(f"⚠️  Could not save report: {e}")
        
        success_condition = total_quotes > 5 and total_indices > 3
        print(f"\n🎯 VERIFICATION STATUS: {'✅ PASSED' if success_condition else '⚠️  PARTIAL'}")
        print("=" * 80)
    
    def run_verification(self):
        """Run complete FYERS data fetching verification"""
        print("🚀 STARTING FYERS DATA FETCHING VERIFICATION")
        print("=" * 80)
        print("This will test all aspects of FYERS API data fetching")
        print("using your verified Nifty 500 master data plus major indices.\n")
        
        # Step 1: Load symbols and indices
        if not self.load_verified_symbols():
            print("❌ Cannot proceed without verified symbols")
            return False
        
        # Step 2: Test stock market quotes (most important)
        quotes_success = self.test_market_quotes()
        
        # Step 3: Test index quotes
        indices_success = self.test_index_quotes()
        
        # Step 4: Test historical data for stocks and indices
        hist_success = self.test_historical_data()
        
        # Step 5: Test account information
        self.test_account_info()
        
        # Step 6: Test market status
        self.test_market_status()
        
        # Step 7: Generate comprehensive report
        self.generate_verification_report()
        
        return quotes_success and indices_success and hist_success

if __name__ == "__main__":
    verifier = FyersDataVerifier()
    success = verifier.run_verification()
    
    if success:
        print("\n🎉 FYERS API verification completed successfully!")
        print("Your automated trading system is ready for live data!")
    else:
        print("\n⚠️  Some tests failed. Check the detailed report for issues.")