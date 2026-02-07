"""
FYERS BACKTESTING DEMO & TUTORIAL
Complete guide to backtesting your FYERS algorithmic trading strategies
"""

from datetime import datetime, timedelta
import json

def create_backtest_config():
    """Create backtesting configuration file"""
    
    config = {
        "backtest_settings": {
            "initial_capital": 100000,
            "commission_per_trade": 20,
            "slippage_percent": 0.05,
            "max_positions": 5,
            "risk_per_trade": 0.02,
            "stop_loss_percent": 2.0,
            "take_profit_percent": 4.0,
            "min_confidence": 75,
            "max_hold_days": 30
        },
        "test_symbols": {
            "single_stock": "RELIANCE",
            "portfolio": [
                "RELIANCE", "TCS", "INFY", "HDFCBANK", 
                "ICICIBANK", "SBIN", "ITC", "HINDUNILVR"
            ]
        },
        "test_periods": {
            "recent": {
                "start": "2023-01-01",
                "end": "2024-12-31"
            },
            "long_term": {
                "start": "2020-01-01", 
                "end": "2024-12-31"
            },
            "bull_market": {
                "start": "2020-03-01",
                "end": "2021-12-31"
            },
            "bear_market": {
                "start": "2022-01-01",
                "end": "2022-12-31"
            }
        }
    }
    
    with open('backtest_config.json', 'w') as f:
        json.dump(config, f, indent=2)
        
    print("📝 Created backtest_config.json")
    return config

def run_comprehensive_backtests():
    """Run multiple backtests with different scenarios"""
    
    try:
        from fyers_algo_backtester import FyersAlgoBacktester
        
        print("🧪 COMPREHENSIVE BACKTESTING SUITE")
        print("="*45)
        
        # Create backtester
        backtester = FyersAlgoBacktester()
        
        # Load configuration
        try:
            with open('backtest_config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = create_backtest_config()
            
        # Test scenarios
        scenarios = [
            {
                "name": "Single Stock - Recent Period",
                "type": "single",
                "symbol": config['test_symbols']['single_stock'],
                "period": config['test_periods']['recent']
            },
            {
                "name": "Portfolio - Bull Market",
                "type": "portfolio", 
                "symbols": config['test_symbols']['portfolio'][:4],  # Top 4 stocks
                "period": config['test_periods']['bull_market']
            },
            {
                "name": "Portfolio - Bear Market",
                "type": "portfolio",
                "symbols": config['test_symbols']['portfolio'][:4],
                "period": config['test_periods']['bear_market']
            }
        ]
        
        # Run each scenario
        results = {}
        
        for scenario in scenarios:
            print(f"\n🎯 Running: {scenario['name']}")
            print("-" * 35)
            
            try:
                if scenario['type'] == 'single':
                    result = backtester.run_backtest(
                        scenario['symbol'],
                        scenario['period']['start'],
                        scenario['period']['end']
                    )
                else:
                    result = backtester.run_portfolio_backtest(
                        scenario['symbols'],
                        scenario['period']['start'],
                        scenario['period']['end']
                    )
                    
                results[scenario['name']] = result
                
                # Quick summary
                print(f"   📊 Return: {result.total_return:+.2f}%")
                print(f"   🎯 Win Rate: {result.win_rate:.1f}%")
                print(f"   📈 Trades: {result.total_trades}")
                print(f"   📉 Max DD: {result.max_drawdown:.2f}%")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                
        # Generate comparison report
        if results:
            print("\n📋 BACKTEST COMPARISON SUMMARY")
            print("="*45)
            
            print(f"{'Scenario':<25} {'Return':<10} {'Win Rate':<10} {'Trades':<8} {'Max DD':<8}")
            print("-" * 65)
            
            for name, result in results.items():
                print(f"{name[:24]:<25} {result.total_return:+7.2f}% {result.win_rate:7.1f}% {result.total_trades:7d} {result.max_drawdown:6.2f}%")
                
        return results
        
    except ImportError:
        print("❌ Error: fyers_algo_backtester module not found")
        print("💡 Make sure fyers_algo_backtester.py exists in the same directory")
        return {}

def show_backtesting_guide():
    """Show comprehensive backtesting guide"""
    
    print("""
🧪 FYERS ALGORITHMIC TRADING BACKTESTING GUIDE
===============================================

🎯 WHAT IS BACKTESTING?
Backtesting tests your trading strategy on historical data to evaluate performance
before risking real money. It's essential for:
- Validating strategy effectiveness
- Understanding risk characteristics  
- Optimizing parameters
- Building confidence before live trading

📊 KEY METRICS EXPLAINED:

📈 RETURN METRICS:
   • Total Return: Overall profit/loss percentage
   • Win Rate: Percentage of profitable trades
   • Profit Factor: Total wins ÷ Total losses
   • Average Win/Loss: Mean profit per winning/losing trade

📉 RISK METRICS:
   • Maximum Drawdown: Worst peak-to-trough loss
   • Sharpe Ratio: Risk-adjusted returns (>1.0 is good)
   • Trade Duration: How long positions are held

🔧 BACKTEST PARAMETERS:

💰 CAPITAL SETTINGS:
   • Initial Capital: ₹1,00,000 (starting amount)
   • Risk per Trade: 2% (maximum loss per trade)
   • Max Positions: 5 (portfolio diversification)

⚙️ EXECUTION SETTINGS:
   • Commission: ₹20 per trade (realistic costs)
   • Stop Loss: 2% (risk management)
   • Take Profit: 4% (2:1 reward:risk ratio)
   • Min Confidence: 75% (signal quality filter)

📅 TEST PERIODS:
   • Recent: 2023-2024 (current market conditions)
   • Long-term: 2020-2024 (multiple market cycles)
   • Bull Market: 2020-2021 (rising market test)
   • Bear Market: 2022 (falling market test)

🚀 HOW TO RUN BACKTESTS:

1️⃣ SINGLE STOCK TEST:
   ```python
   backtester = FyersAlgoBacktester()
   results = backtester.run_backtest("RELIANCE", "2023-01-01", "2024-12-31")
   print(backtester.create_report(results))
   ```

2️⃣ PORTFOLIO TEST:
   ```python
   symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK"]  
   results = backtester.run_portfolio_backtest(symbols, "2023-01-01", "2024-12-31")
   backtester.plot_results(results)
   ```

3️⃣ COMPREHENSIVE TESTING:
   ```python
   python fyers_backtest_demo.py  # Run this file!
   ```

📊 INTERPRETING RESULTS:

✅ GOOD STRATEGY CHARACTERISTICS:
   • Total Return: >15% annually
   • Win Rate: >60%
   • Profit Factor: >1.5
   • Max Drawdown: <20%
   • Sharpe Ratio: >1.0

⚠️ WARNING SIGNS:
   • Very high win rate (>90%) - may be curve-fitted
   • Large drawdowns (>30%) - excessive risk
   • Few trades (<20) - insufficient data
   • Inconsistent performance across periods

🎯 NEXT STEPS AFTER BACKTESTING:

1. ✅ GOOD RESULTS: Proceed to paper trading
2. ⚠️ MIXED RESULTS: Optimize parameters and retest  
3. ❌ POOR RESULTS: Revise strategy or abandon
4. 🚀 EXCELLENT RESULTS: Start live trading with small amounts

💡 BACKTESTING LIMITATIONS:
   • Uses past data (future may differ)
   • Assumes perfect execution (reality has slippage)
   • May not capture all market conditions
   • Can lead to over-optimization

🔥 PRO TIPS:
   • Test across different market conditions
   • Use realistic commission and slippage
   • Don't over-optimize on limited data
   • Validate with out-of-sample testing
   • Start live trading with minimal capital

Ready to backtest your FYERS algorithmic trading strategy? 🚀
""")

def main():
    """Main demo function"""
    
    show_backtesting_guide()
    
    print("\n🚀 STARTING COMPREHENSIVE BACKTEST DEMO...")
    print("⏳ This may take a few minutes to download data and run tests...")
    
    # Create config
    create_backtest_config()
    
    # Run comprehensive tests
    results = run_comprehensive_backtests()
    
    if results:
        print("\n✅ BACKTESTING COMPLETED SUCCESSFULLY!")
        print("\n📊 Check the generated plots and reports")
        print("💡 If results look good, you're ready for live trading with FYERS!")
        print("\n🔗 NEXT STEPS:")
        print("   1. Configure FYERS API in fyers_config.json")
        print("   2. Run: python fyers_live_portfolio.py")
        print("   3. Start automated trading with real money!")
    else:
        print("\n⚠️ Some backtests failed - check your setup")
        print("💡 Ensure you have internet connection and all dependencies installed")

if __name__ == "__main__":
    main()