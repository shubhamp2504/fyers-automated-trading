from fyers_client import FyersClient
import json
import time
import requests
import pandas as pd
from io import StringIO
import csv

class NSENifty500Fetcher:
    def __init__(self):
        self.fyers = FyersClient()
        self.verified_symbols = []
        self.failed_symbols = []
        self.nifty500_master = []
    
    def fetch_nifty500_from_nse(self):
        """Fetch official Nifty 500 constituent list from NSE website"""
        print("🌐 Fetching official Nifty 500 list from NSE website...")
        
        try:
            # NSE Nifty 500 index constituents CSV URL
            nse_url = "https://www.nseindex.com/content/indices/ind_nifty500list.csv"
            
            # Headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            print("📡 Downloading Nifty 500 constituents CSV...")
            response = requests.get(nse_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                print("✅ Successfully downloaded NSE data")
                
                # Parse CSV data
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)
                
                print(f"📊 Found {len(df)} stocks in Nifty 500")
                
                # Extract symbols and create FYERS format
                nifty500_list = []
                for index, row in df.iterrows():
                    try:
                        symbol = str(row['Symbol']).strip()
                        company_name = str(row['Company Name']).strip()
                        series = str(row['Series']).strip() if 'Series' in row else 'EQ'
                        industry = str(row['Industry']).strip() if 'Industry' in row else 'N/A'
                        
                        # Create FYERS symbol format
                        fyers_symbol = f"NSE:{symbol}-EQ"
                        
                        stock_info = {
                            'symbol': fyers_symbol,
                            'nse_symbol': symbol,
                            'company_name': company_name,
                            'series': series,
                            'industry': industry,
                            'index': 'NIFTY500'
                        }
                        
                        nifty500_list.append(stock_info)
                        
                    except Exception as e:
                        print(f"⚠️  Error processing row {index}: {e}")
                        continue
                
                print(f"✅ Processed {len(nifty500_list)} Nifty 500 symbols")
                return nifty500_list
                
            else:
                print(f"❌ Failed to download NSE data. Status code: {response.status_code}")
                return self.get_fallback_nifty500_list()
                
        except Exception as e:
            print(f"❌ Error fetching from NSE: {e}")
            print("🔄 Using fallback method...")
            return self.get_fallback_nifty500_list()
    
    def get_fallback_nifty500_list(self):
        """Fallback method with manual Nifty 500 list if NSE site fails"""
        print("📋 Using fallback Nifty 500 list...")
        
        # Top 100+ Nifty 500 stocks (manually curated)
        fallback_symbols = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'BHARTIARTL', 'ICICIBANK', 'SBIN', 'INFY', 'LT', 'ITC', 'KOTAKBANK',
            'HINDUNILVR', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI', 'AXISBANK', 'SUNPHARMA', 'ULTRACEMCO', 'TITAN',
            'WIPRO', 'NESTLEIND', 'M&M', 'POWERGRID', 'NTPC', 'BAJAJFINSV', 'TECHM', 'HCLTECH', 'ONGC', 'TATASTEEL',
            'ADANIENT', 'COALINDIA', 'BRITANNIA', 'CIPLA', 'DIVISLAB', 'EICHERMOT', 'GRASIM', 'HINDALCO',
            'INDUSINDBK', 'JSWSTEEL', 'LTIM', 'TATACONSUM', 'TATAMOTORS', 'UPL', 'APOLLOHOSP', 'DRREDDY',
            'HDFCLIFE', 'HEROMOTOCO', 'PIDILITIND', 'SBILIFE', 'SHREECEM', 'BAJAJ-AUTO', 'BPCL', 'GODREJCP',
            'ICICIPRULI', 'ADANIPORTS', 'IOC', 'VEDL', 'BANDHANBNK', 'DABUR', 'MARICO', 'BERGEPAINT',
            'HAVELLS', 'COLPAL', 'MCDOWELL-N', 'ACC', 'AMBUJACEM', 'INDIGO', 'SIEMENS', 'DLF', 'GAIL',
            'BOSCHLTD', 'SRF', 'PAGEIND', 'ABB', 'BEL', 'CANBK', 'PNB', 'BANKBARODA', 'MOTHERSON',
            'BAJAJHLDNG', 'MUTHOOTFIN', 'PEL', 'TORNTPHARM', 'CHOLAFIN', 'LICI', 'TRENT', 'JINDALSTEL',
            'SAIL', 'RECLTD', 'CONCOR', 'OFSS', 'MPHASIS', 'PERSISTENT', 'MINDTREE', 'COFORGE', 'LTTS',
            'BIOCON', 'ALKEM', 'LUPIN', 'CADILAHC', 'GRANULES', 'ABBOTINDIA', 'PFIZER', 'GLAXO',
            'AUROPHARMA', 'REDDY', 'JUBLFOOD', 'BRITANNIA', 'GODREJIND', 'VOLTAS', 'WHIRLPOOL', 'CROMPTON',
            'BAJAJCON', 'BATAINDIA', 'RELAXO', 'VMART', 'TATAELXSI', 'POLYCAB', 'KEI', 'FINOLEX'
        ]
        
        fallback_list = []
        for symbol in fallback_symbols:
            stock_info = {
                'symbol': f"NSE:{symbol}-EQ",
                'nse_symbol': symbol,
                'company_name': symbol,  # Using symbol as name for fallback
                'series': 'EQ',
                'industry': 'N/A',
                'index': 'NIFTY500'
            }
            fallback_list.append(stock_info)
        
        print(f"✅ Created fallback list with {len(fallback_list)} symbols")
        return fallback_list
    
    def verify_symbols_batch(self, symbols, batch_size=10):
        """Verify symbols in batches to avoid rate limits"""
        print(f"🔍 Verifying {len(symbols)} symbols with FYERS API (batch size: {batch_size})...")
        
        verified = []
        failed = []
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            print(f"\n📦 Processing batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}")
            
            for j, stock in enumerate(batch):
                try:
                    symbol = stock['symbol']
                    print(f"📊 Verifying {symbol} ({i+j+1}/{len(symbols)})...")
                    
                    # Get market quote to verify symbol
                    quote = self.fyers.get_market_quote(symbol)
                    
                    if quote.get('s') == 'ok' and quote.get('d'):
                        price_data = quote['d'][0]['v']
                        
                        # Add market data to stock info
                        stock['market_data'] = {
                            'last_price': price_data.get('lp', 0),
                            'volume': price_data.get('volume', 0),
                            'high': price_data.get('high_price', 0),
                            'low': price_data.get('low_price', 0),
                            'open': price_data.get('open_price', 0),
                            'change': price_data.get('ch', 0),
                            'change_percent': price_data.get('chp', 0),
                            'prev_close': price_data.get('prev_close_price', 0),
                            'fyToken': price_data.get('fyToken', ''),
                            'last_updated': int(time.time())
                        }
                        
                        verified.append(stock)
                        print(f"✅ {symbol} - ₹{stock['market_data']['last_price']} | Vol: {stock['market_data']['volume']:,}")
                    else:
                        error_msg = quote.get('message', 'No data available')
                        failed.append({**stock, 'error': error_msg})
                        print(f"❌ {symbol} - Failed: {error_msg}")
                    
                    # Small delay to avoid rate limits
                    time.sleep(0.3)
                    
                except Exception as e:
                    failed.append({**stock, 'error': str(e)})
                    print(f"❌ {symbol} - Error: {e}")
                    time.sleep(0.3)
            
            # Longer delay between batches
            if i + batch_size < len(symbols):
                print("⏳ Waiting 2 seconds before next batch...")
                time.sleep(2)
        
        return verified, failed
    
    def create_master_files(self, nifty500_data, verified_symbols):
        """Create comprehensive master files for trading system"""
        try:
            # 1. Complete Nifty 500 master with all details
            with open('nifty500_master.json', 'w') as f:
                json.dump(nifty500_data, f, indent=2)
            print(f"💾 Saved complete master data to nifty500_master.json")
            
            # 2. Verified symbols with market data
            with open('nifty500_verified.json', 'w') as f:
                json.dump(verified_symbols, f, indent=2)
            print(f"💾 Saved {len(verified_symbols)} verified symbols to nifty500_verified.json")
            
            # 3. Simple symbol list for quick trading
            symbol_list = [stock['symbol'] for stock in verified_symbols]
            with open('nifty500_symbols.json', 'w') as f:
                json.dump(symbol_list, f, indent=2)
            print(f"💾 Saved symbol list to nifty500_symbols.json")
            
            # 4. Trading-ready CSV file
            csv_data = []
            for stock in verified_symbols:
                csv_row = {
                    'FYERS_Symbol': stock['symbol'],
                    'NSE_Symbol': stock['nse_symbol'],
                    'Company_Name': stock['company_name'],
                    'Industry': stock['industry'],
                    'Last_Price': stock['market_data']['last_price'],
                    'Volume': stock['market_data']['volume'],
                    'Change_Percent': stock['market_data']['change_percent'],
                    'fyToken': stock['market_data']['fyToken']
                }
                csv_data.append(csv_row)
            
            df = pd.DataFrame(csv_data)
            df.to_csv('nifty500_trading_master.csv', index=False)
            print(f"💾 Saved trading master CSV with {len(csv_data)} stocks")
            
        except Exception as e:
            print(f"❌ Error creating master files: {e}")
    
    def print_summary(self, nifty500_data, verified, failed):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("📊 NIFTY 500 MASTER DATA CREATION SUMMARY")
        print("="*80)
        print(f"🌐 Total symbols fetched from NSE: {len(nifty500_data)}")
        print(f"✅ Successfully verified: {len(verified)} symbols")
        print(f"❌ Failed verification: {len(failed)} symbols")
        print(f"📈 Success rate: {(len(verified)/(len(verified)+len(failed))*100):.1f}%")
        
        if verified:
            print(f"\n🎯 TOP 10 MOST ACTIVE VERIFIED STOCKS:")
            print(f"{'Symbol':<25} {'Company':<30} {'Price':<10} {'Volume':<15} {'Change%':<10}")
            print("-" * 90)
            
            # Sort by volume (most active stocks first)
            top_stocks = sorted(verified, key=lambda x: x['market_data']['volume'], reverse=True)[:10]
            
            for stock in top_stocks:
                change_pct = stock['market_data']['change_percent']
                change_color = "🟢" if change_pct >= 0 else "🔴"
                company_short = stock['company_name'][:28] + ".." if len(stock['company_name']) > 30 else stock['company_name']
                
                print(f"{stock['symbol']:<25} {company_short:<30} ₹{stock['market_data']['last_price']:<9.2f} {stock['market_data']['volume']:<15,} {change_color}{change_pct:<9.2f}%")
        
        print(f"\n📁 FILES CREATED:")
        print(f"   • nifty500_master.json - Complete data with company details")
        print(f"   • nifty500_verified.json - Verified symbols with market data")  
        print(f"   • nifty500_symbols.json - Simple symbol list")
        print(f"   • nifty500_trading_master.csv - Trading-ready CSV file")
        
        if failed:
            print(f"\n❌ SAMPLE FAILED SYMBOLS:")
            for fail in failed[:3]:
                print(f"   {fail['symbol']} - {fail.get('error', 'Unknown error')}")
        
        print(f"\n🚀 Your Nifty 500 master is ready for automated trading!")
    
    def run(self):
        """Main function to create Nifty 500 master data"""
        print("🚀 Creating Nifty 500 Master Data for Automated Trading System")
        print("="*70)
        
        # Step 1: Fetch official Nifty 500 list from NSE
        nifty500_data = self.fetch_nifty500_from_nse()
        if not nifty500_data:
            print("❌ Failed to fetch Nifty 500 data")
            return
        
        # Step 2: Verify symbols with FYERS API (in batches)
        verified, failed = self.verify_symbols_batch(nifty500_data, batch_size=10)
        
        # Step 3: Create master files
        if verified:
            self.create_master_files(nifty500_data, verified)
        
        # Step 4: Print comprehensive summary
        self.print_summary(nifty500_data, verified, failed)
        
        return verified

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import pandas as pd
    except ImportError:
        print("Installing pandas...")
        import subprocess
        subprocess.check_call(["pip", "install", "pandas"])
        import pandas as pd
    
    fetcher = NSENifty500Fetcher()
    verified_symbols = fetcher.run()