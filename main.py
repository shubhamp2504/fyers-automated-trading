"""
MAIN ENTRY POINT - FYERS LIVE TRADING SYSTEM (Python 3.11 Optimized)
=====================================================================

Real trading system using official Fyers API v3 with live data only.
Source: https://myapi.fyers.in/docsv3

✅ BREAKTHROUGH: Python 3.11 eliminates Visual C++ compilation issues!
✅ aiohttp-3.9.3 installs as pre-compiled wheel - no build errors!

Features:
- ✅ Official Fyers API integration (Python 3.11 compatible)
- ✅ Real market data streaming
- ✅ Live order execution 
- ✅ Portfolio tracking
- ✅ Risk management
- ❌ NO DEMO/DUMMY DATA

🚀 RECOMMENDED: Use run_trading_system.bat for automatic Python 3.11 selection
🔧 MANUAL: C:\Users\shubh\AppData\Local\Programs\Python\Python311\python.exe main.py

⚠️  WARNING: This system places REAL orders with REAL money.
⚠️  Use appropriate position sizing and risk management.
"""

from fyers_client import FyersClient
import sys
import os

# Import live trading system
sys.path.append(os.path.dirname(__file__))
from live_trading_system import LiveIndexTradingSystem
from index_intraday_strategy import IndexIntradayStrategy

def main():
    """Main entry point for live Fyers trading system"""
    
    print("=" * 60)
    print("🚀 FYERS LIVE TRADING SYSTEM - REAL DATA ONLY")
    print("=" * 60)
    print("⚠️  WARNING: LIVE TRADING WITH REAL MONEY")
    print("⚠️  Ensure proper risk management is in place")
    print("=" * 60)
    
    try:
        # Initialize Fyers client with real API
        print("\n🔑 Initializing Fyers API Client...")
        fyers = FyersClient('fyers_config.json')
        
        # Verify account funds before trading
        print("\n💰 Checking account funds...")
        funds = fyers.get_funds()
        if funds:
            available = funds.get('availableAmount', 0)
            print(f"   💵 Available: ₹{available:,.2f}")
            if available < 10000:  # Minimum balance check
                print("❌ Insufficient funds for trading. Minimum ₹10,000 required.")
                return
        
        # Initialize live trading system
        print("\n📊 Initializing Live Trading System...")
        trading_system = LiveIndexTradingSystem(fyers_client=fyers)
        
        # Start live trading
        print("\n🔴 Starting LIVE trading...")
        print("   📈 Monitoring: NIFTY & BANKNIFTY indices")
        print("   🎯 Strategy: Intraday momentum with support/resistance")
        print("   ⏰ Market hours: 9:15 AM to 3:30 PM")
        print("\n" + "=" * 60)
        print("LIVE TRADING ACTIVE - Press Ctrl+C to stop")
        print("=" * 60)
        
        # Run trading system
        trading_system.run()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Trading stopped by user")
        print("📊 Final portfolio status:")
        try:
            positions = fyers.get_positions()
            if positions:
                print(f"   📈 Open positions: {len(positions)}")
            else:
                print("   📈 No open positions")
        except:
            pass
            
    except Exception as e:
        print(f"\n❌ Error in trading system: {e}")
        print("🚨 Please check logs and restart if needed")
        
    finally:
        print("\n👋 Trading system shutdown complete")

if __name__ == "__main__":
    main()
