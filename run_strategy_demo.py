"""
Index Intraday Strategy Demo & Execution
========================================

Complete demonstration of the index intraday trading strategy:
1. Strategy overview and configuration
2. Backtesting with realistic market conditions
3. Parameter optimization
4. Live trading simulation
5. Performance analysis and reporting

⚠️ IMPORTANT: Always refer to https://myapi.fyers.in/docsv3 for latest API specifications
"""

import json
import time
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Import all our modules
from index_intraday_strategy import backtest_strategy, live_trading_demo
from advanced_backtester import run_full_backtest
from strategy_optimizer import run_strategy_optimization
from live_trading_system import LiveTradingSystem

def display_banner():
    """Display strategy banner"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                INDEX INTRADAY TRADING STRATEGY                ║
    ║                                                              ║
    ║  🎯 Multi-Timeframe Analysis (1H + 5M)                       ║
    ║  💰 Smart Profit Targets (20-30 points)                      ║
    ║  🛡️ Intelligent Stop Loss Management                        ║
    ║  📊 Focus on NIFTY 50 & BANK NIFTY                           ║
    ║  🤖 Fully Automated Execution                                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def show_strategy_overview():
    """Display strategy overview"""
    
    print("\n🎯 STRATEGY OVERVIEW")
    print("=" * 50)
    print("""
    📈 TIMEFRAMES:
    • Analysis: 1 Hour candles for trend identification
    • Execution: 5 Min candles for precise entry/exit
    
    💡 STRATEGY LOGIC:
    • EMA Crossover (9 vs 21) for trend direction
    • RSI (14) for momentum confirmation
    • VWAP for price strength validation
    • Support/Resistance levels for context
    
    🎯 PROFIT TARGETS:
    • Target 1: 20-25 points (Partial exit - 50% position)
    • Target 2: 25-30 points (Complete exit)
    • Stop Loss: Dynamic based on ATR and support/resistance
    
    ⚠️ RISK MANAGEMENT:
    • Maximum loss per trade: 15 points
    • Position sizing: 1 lot per trade
    • Daily loss limit: ₹5,000
    • Maximum concurrent positions: 2
    
    🕒 TRADING HOURS:
    • Market Hours: 09:15 - 15:15
    • Strategy Active: 09:15 - 14:30
    • Force exit: 15:00 (before market close)
    """)

def run_strategy_demo():
    """Run complete strategy demonstration"""
    
    display_banner()
    show_strategy_overview()
    
    print(f"\n🚀 STARTING STRATEGY DEMONSTRATION")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
    except:
        print("❌ config.json not found!")
        print("📝 Please ensure you have proper FYERS API credentials")
        return
    
    # Menu system
    while True:
        print(f"\n📋 STRATEGY DEMO MENU")
        print("-" * 30)
        print("1. 📊 Quick Backtest (Basic)")
        print("2. 🔬 Advanced Backtest (Comprehensive)")
        print("3. ⚙️ Parameter Optimization")
        print("4. 🎮 Live Trading Demo")
        print("5. 🚀 Full Live Trading System")
        print("6. 📈 Strategy Performance Report")
        print("7. ❓ Help & Documentation")
        print("0. 🚪 Exit")
        
        choice = input(f"\n🎯 Select option (0-7): ").strip()
        
        if choice == '0':
            print(f"👋 Thank you for using Index Intraday Strategy!")
            break
        elif choice == '1':
            run_quick_backtest()
        elif choice == '2':
            run_advanced_backtest()
        elif choice == '3':
            run_optimization()
        elif choice == '4':
            run_demo_trading()
        elif choice == '5':
            run_live_trading()
        elif choice == '6':
            show_performance_report()
        elif choice == '7':
            show_help_documentation()
        else:
            print("❌ Invalid choice. Please select 0-7.")
        
        input(f"\n⏸️ Press Enter to continue...")

def run_quick_backtest():
    """Run quick backtest demonstration"""
    
    print(f"\n1️⃣ QUICK BACKTEST")
    print("=" * 40)
    print("🔄 Running basic strategy backtest...")
    print("📊 This will test the strategy on simulated data")
    
    try:
        results = backtest_strategy(days_back=10)
        
        if results:
            print(f"\n📊 QUICK BACKTEST RESULTS")
            print("-" * 30)
            print(f"📈 Total Trades: {results.get('total_trades', 0)}")
            print(f"🟢 Winning Trades: {results.get('winning_trades', 0)}")
            print(f"🔴 Losing Trades: {results.get('losing_trades', 0)}")
            print(f"🎯 Win Rate: {results.get('win_rate', 0):.1f}%")
            print(f"💰 Total P&L: ₹{results.get('total_pnl', 0):,.2f}")
            print(f"📈 Best Trade: ₹{results.get('max_profit', 0):.2f}")
            print(f"📉 Worst Trade: ₹{results.get('max_loss', 0):.2f}")
            
            if results.get('win_rate', 0) > 60:
                print(f"✅ GOOD: Strategy shows promising results")
            elif results.get('win_rate', 0) > 50:
                print(f"🟡 FAIR: Strategy has potential with optimization")
            else:
                print(f"🔴 POOR: Strategy needs significant improvement")
        else:
            print("❌ Backtest failed - check configuration")
            
    except Exception as e:
        print(f"❌ Error running quick backtest: {e}")

def run_advanced_backtest():
    """Run advanced comprehensive backtest"""
    
    print(f"\n2️⃣ ADVANCED BACKTEST")
    print("=" * 40)
    print("🔬 Running comprehensive backtest with realistic market simulation...")
    print("⏳ This may take a few minutes...")
    
    try:
        results = run_full_backtest()
        
        if results:
            print(f"\n✅ Advanced backtest completed successfully!")
            print(f"📊 Detailed results displayed above")
            
            # Summary
            total_symbols = len(results)
            profitable_symbols = sum(1 for r in results.values() if r.get('total_return', 0) > 0)
            
            print(f"\n📋 SUMMARY:")
            print(f"   🔬 Symbols Tested: {total_symbols}")
            print(f"   💚 Profitable: {profitable_symbols}")
            print(f"   📊 Success Rate: {profitable_symbols/max(total_symbols,1)*100:.1f}%")
            
        else:
            print("❌ Advanced backtest failed")
            
    except Exception as e:
        print(f"❌ Error running advanced backtest: {e}")

def run_optimization():
    """Run strategy parameter optimization"""
    
    print(f"\n3️⃣ PARAMETER OPTIMIZATION")
    print("=" * 40)
    print("⚙️ Optimizing strategy parameters for maximum performance...")
    print("⏳ This process may take 10-15 minutes...")
    
    confirm = input(f"🤔 Continue with optimization? (y/N): ").lower()
    
    if confirm != 'y':
        print("📊 Optimization cancelled")
        return
    
    try:
        results = run_strategy_optimization()
        
        if results:
            print(f"\n✅ Parameter optimization completed!")
            print(f"📁 Optimized parameters saved to JSON files")
            print(f"🎯 Ready for live trading with optimized settings")
        else:
            print("❌ Parameter optimization failed")
            
    except Exception as e:
        print(f"❌ Error running optimization: {e}")

def run_demo_trading():
    """Run live trading demonstration"""
    
    print(f"\n4️⃣ LIVE TRADING DEMO")
    print("=" * 40)
    print("🎮 Running safe demo of live trading logic...")
    print("📊 No real trades will be placed")
    
    try:
        live_trading_demo()
        print(f"\n✅ Live trading demo completed!")
        
    except Exception as e:
        print(f"❌ Error running live demo: {e}")

def run_live_trading():
    """Run actual live trading system"""
    
    print(f"\n5️⃣ LIVE TRADING SYSTEM")
    print("=" * 40)
    print("🚀 This will start REAL live trading with actual money!")
    print("⚠️ RISK WARNING: Live trading involves real financial risk")
    print("💰 Ensure you understand the strategy before proceeding")
    
    confirm1 = input(f"\n🤔 Do you understand the risks? (yes/no): ").lower()
    if confirm1 != 'yes':
        print("📊 Live trading cancelled for safety")
        return
    
    confirm2 = input(f"🎯 Start live trading with real money? (START/cancel): ")
    if confirm2 != 'START':
        print("📊 Live trading cancelled")
        return
    
    try:
        trading_system = LiveTradingSystem()
        
        if not trading_system.is_market_open():
            print(f"⏰ Market is currently closed")
            print(f"📅 Trading hours: {trading_system.market_open_time} - {trading_system.market_close_time}")
            return
        
        print(f"🚀 Starting live trading system...")
        trading_system.start_live_trading()
        
    except Exception as e:
        print(f"❌ Error starting live trading: {e}")
        import traceback
        traceback.print_exc()

def show_performance_report():
    """Show comprehensive performance report"""
    
    print(f"\n6️⃣ PERFORMANCE REPORT")
    print("=" * 40)
    
    # Check for existing results
    print("📊 Checking for performance data...")
    
    # Try to load optimization results
    try:
        with open('optimized_params_nifty50-index.json', 'r') as f:
            nifty_results = json.load(f)
        print("✅ NIFTY 50 optimization data found")
        
        best_perf = nifty_results.get('best_performance', {})
        print(f"\n📈 NIFTY 50 OPTIMIZED PERFORMANCE:")
        print(f"   🎯 Best Return: {best_perf.get('total_return', 0):.2f}%")
        print(f"   📊 Win Rate: {best_perf.get('win_rate', 0):.1f}%")
        print(f"   ⚖️ Profit Factor: {best_perf.get('profit_factor', 0):.2f}")
        print(f"   📉 Max Drawdown: {best_perf.get('max_drawdown', 0):.2f}%")
        
    except FileNotFoundError:
        print("❌ No NIFTY 50 optimization data found")
    
    try:
        with open('optimized_params_niftybank-index.json', 'r') as f:
            bank_results = json.load(f)
        print("✅ BANK NIFTY optimization data found")
        
        best_perf = bank_results.get('best_performance', {})
        print(f"\n📈 BANK NIFTY OPTIMIZED PERFORMANCE:")
        print(f"   🎯 Best Return: {best_perf.get('total_return', 0):.2f}%")
        print(f"   📊 Win Rate: {best_perf.get('win_rate', 0):.1f}%")
        print(f"   ⚖️ Profit Factor: {best_perf.get('profit_factor', 0):.2f}")
        print(f"   📉 Max Drawdown: {best_perf.get('max_drawdown', 0):.2f}%")
        
    except FileNotFoundError:
        print("❌ No BANK NIFTY optimization data found")
    
    if not any([
        'optimized_params_nifty50-index.json',
        'optimized_params_niftybank-index.json'
    ]):
        print("\n💡 TIP: Run parameter optimization first to generate performance reports")

def show_help_documentation():
    """Show help and documentation"""
    
    print(f"\n7️⃣ HELP & DOCUMENTATION")
    print("=" * 40)
    
    print("""
    📚 STRATEGY DOCUMENTATION:
    
    🎯 CORE CONCEPT:
    • Multi-timeframe analysis combining 1H trend with 5M execution
    • Focus on NIFTY 50 and BANK NIFTY index trading
    • Optimized for 20-30 point profit targets with minimal losses
    
    🔧 TECHNICAL INDICATORS:
    • EMA (9, 21): Trend identification and crossover signals
    • RSI (14): Momentum confirmation and overbought/oversold levels
    • VWAP: Price strength and institutional interest
    • ATR: Dynamic stop loss calculation
    • Support/Resistance: Key level identification
    
    💰 RISK MANAGEMENT:
    • Position Size: 1 lot per trade (adjustable)
    • Stop Loss: Maximum 15 points loss per trade
    • Profit Targets: 22-25 points (Target 1), 28-30 points (Target 2)
    • Daily Loss Limit: ₹5,000 maximum
    • Maximum Positions: 2 concurrent trades
    
    ⏰ TRADING SCHEDULE:
    • Market Open: 09:15 AM
    • Strategy Active: 09:15 AM - 02:30 PM
    • Force Exit: 03:00 PM
    • Market Close: 03:15 PM
    
    🛠️ REQUIRED SETUP:
    1. Valid FYERS API credentials in config.json
    2. Active FYERS trading account with funds
    3. Python environment with required packages
    4. Stable internet connection for live trading
    
    📞 SUPPORT:
    • FYERS API Docs: https://myapi.fyers.in/docsv3
    • Always test strategies in paper trading first
    • Start with small position sizes
    • Monitor trades actively during market hours
    
    ⚠️ DISCLAIMER:
    • Trading involves substantial risk of loss
    • Past performance does not guarantee future results
    • Use only risk capital you can afford to lose
    • This is educational software, not investment advice
    """)

def main():
    """Main entry point"""
    
    print(f"🎯 INDEX INTRADAY TRADING STRATEGY")
    print(f"⏰ Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        run_strategy_demo()
    except KeyboardInterrupt:
        print(f"\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n👋 Thank you for using Index Intraday Strategy!")
        print(f"💡 Remember: Always test strategies thoroughly before live trading")

if __name__ == "__main__":
    main()