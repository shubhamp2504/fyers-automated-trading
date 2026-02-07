"""
🧪 HOW TO BACKTEST YOUR FYERS ALGORITHMIC TRADING SYSTEM
Complete step-by-step guide to validate your strategies before live trading
"""

def show_backtesting_tutorial():
    """Complete backtesting tutorial"""
    
    print("""
🧪 FYERS ALGORITHMIC TRADING BACKTESTING TUTORIAL
=================================================

🎯 WHY BACKTEST?
Backtesting is CRITICAL before live trading because:
✅ Validates strategy effectiveness on historical data
✅ Identifies optimal parameters and risk settings  
✅ Estimates expected returns and drawdowns
✅ Builds confidence before risking real money
✅ Prevents costly mistakes in live markets

📊 BACKTESTING PROCESS OVERVIEW:

1. HISTORICAL DATA ➜ Download price data (Yahoo Finance/FYERS)
2. SIGNAL GENERATION ➜ Apply JEAFX strategy rules
3. TRADE SIMULATION ➜ Execute virtual trades with real costs
4. PERFORMANCE ANALYSIS ➜ Calculate returns, win rate, risk metrics
5. OPTIMIZATION ➜ Fine-tune parameters for better performance

🚀 HOW TO BACKTEST YOUR FYERS SYSTEM:

📝 METHOD 1: SIMPLE SINGLE STOCK TEST
```python
from fyers_algo_backtester import FyersAlgoBacktester

# Initialize backtester
backtester = FyersAlgoBacktester()

# Run backtest on RELIANCE for 2 years
results = backtester.run_backtest("RELIANCE", "2022-01-01", "2024-12-31")

# Generate report
report = backtester.create_report(results)
print(report)

# Create visualizations
backtester.plot_results(results)
```

📈 METHOD 2: PORTFOLIO BACKTEST
```python
# Test multiple stocks together
symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
results = backtester.run_portfolio_backtest(symbols, "2022-01-01", "2024-12-31")

# Results will show diversification benefits
print(f"Portfolio Return: {results.total_return:.2f}%")
print(f"Win Rate: {results.win_rate:.1f}%")
```

⚙️ METHOD 3: COMPREHENSIVE TESTING
```python
# Run the demo script for multiple scenarios
python fyers_backtest_demo.py

# This tests:
# - Single stock performance
# - Portfolio performance  
# - Bull market periods
# - Bear market periods
# - Different time frames
```

📊 KEY METRICS TO ANALYZE:

📈 RETURN METRICS:
   • Total Return: Overall profit/loss percentage
   • Win Rate: % of profitable trades (target: >60%)
   • Profit Factor: Total wins ÷ Total losses (target: >1.5)
   • Average Win/Loss: Mean profit per winning/losing trade

📉 RISK METRICS:
   • Maximum Drawdown: Worst peak-to-trough loss (target: <20%)
   • Sharpe Ratio: Risk-adjusted returns (target: >1.0)
   • Trade Frequency: Number of trades per month
   • Average Holding Period: Days per trade

💰 GOOD STRATEGY CHARACTERISTICS:
✅ Total Return: >15% annually
✅ Win Rate: 60-80% (not >90%, may be over-fitted)
✅ Profit Factor: >1.5
✅ Max Drawdown: <20%
✅ Sharpe Ratio: >1.0
✅ Consistent across different market periods

⚠️ WARNING SIGNS:
❌ Very high win rate (>90%) - may be curve-fitted
❌ Large drawdowns (>30%) - excessive risk
❌ Few trades (<20 per year) - insufficient data
❌ Inconsistent performance across time periods
❌ Strategy only works in bull markets

🔧 BACKTEST CONFIGURATION:

💰 CAPITAL SETTINGS:
   • Initial Capital: ₹1,00,000 (realistic starting amount)
   • Risk per Trade: 2% (conservative risk management)
   • Max Positions: 5 (diversification limit)
   • Commission: ₹20 per trade (realistic brokerage costs)

⚙️ STRATEGY SETTINGS:
   • Stop Loss: 2% (risk control)
   • Take Profit: 4% (2:1 reward:risk ratio)
   • Min Confidence: 75% (high-quality signals only)
   • Max Hold Period: 30 days (avoid indefinite positions)

📅 TEST PERIODS TO USE:

🔥 RECENT PERIOD (2023-2024):
   - Tests current market conditions
   - Most relevant for near-term trading

📊 LONG-TERM (2020-2024):
   - Tests multiple market cycles
   - Shows strategy robustness

🚀 BULL MARKET (2020-2021):
   - Tests performance in rising markets
   - Should show good returns

📉 BEAR MARKET (2022):
   - Tests downside protection
   - Critical for risk assessment

🎯 STEP-BY-STEP BACKTESTING CHECKLIST:

□ 1. SETUP ENVIRONMENT
   - Install required packages: pandas, yfinance, matplotlib
   - Ensure internet connection for data download
   - Check that all strategy modules are working

□ 2. CONFIGURE PARAMETERS
   - Set realistic capital amount
   - Configure risk management rules
   - Select appropriate test symbols and time periods

□ 3. RUN INITIAL TESTS
   - Start with single stock backtest
   - Verify data download and processing
   - Check that trades are being generated

□ 4. ANALYZE RESULTS
   - Review return and risk metrics
   - Examine trade distribution and timing
   - Look for consistent performance patterns

□ 5. OPTIMIZE IF NEEDED
   - Adjust confidence thresholds
   - Modify stop loss/take profit levels
   - Test different holding periods

□ 6. VALIDATE ROBUSTNESS
   - Test across different time periods
   - Try different symbols and sectors
   - Ensure results are not over-fitted

□ 7. PREPARE FOR LIVE TRADING
   - If results are satisfactory, proceed to paper trading
   - Start live trading with minimal capital
   - Monitor performance vs backtest expectations

🚨 IMPORTANT LIMITATIONS:

⚠️ BACKTEST LIMITATIONS TO REMEMBER:
   • Uses historical data (future may be different)
   • Assumes perfect execution (no slippage/delays)
   • May not capture all market conditions
   • Can lead to over-optimization bias
   • Does not account for psychological factors

💡 BEST PRACTICES:
   • Test across multiple time periods and market conditions
   • Use realistic commission and slippage assumptions
   • Don't over-optimize parameters on limited data
   • Always validate with out-of-sample testing
   • Start live trading with small amounts first

🔗 NEXT STEPS AFTER BACKTESTING:

✅ GOOD RESULTS (>15% annual return, <20% drawdown):
   1. Proceed to paper trading for 1-2 months
   2. Configure FYERS API credentials
   3. Start live trading with ₹10,000-25,000
   4. Monitor closely and compare to backtest

⚠️ MIXED RESULTS (10-15% return, 15-25% drawdown):
   1. Optimize parameters and retest
   2. Consider longer test periods
   3. May need strategy refinements

❌ POOR RESULTS (<10% return, >25% drawdown):
   1. Revise strategy completely
   2. Consider different indicators or rules
   3. May need to abandon current approach

🎉 READY TO START BACKTESTING?

Run these commands to begin:

python fyers_algo_backtester.py     # Simple demo
python fyers_backtest_demo.py       # Comprehensive testing

Remember: Backtesting is your safety net before risking real money!
A few hours of backtesting can save you thousands in losses.

🚀 Once backtesting shows good results, you're ready for live FYERS trading!
""")

def show_quick_backtest_commands():
    """Show quick backtest commands"""
    
    print("""
🚀 QUICK BACKTEST COMMANDS FOR FYERS SYSTEM
===========================================

📊 SINGLE STOCK BACKTEST:
```python
from fyers_algo_backtester import FyersAlgoBacktester

backtester = FyersAlgoBacktester()
results = backtester.run_backtest("RELIANCE", "2023-01-01", "2024-12-31")
print(backtester.create_report(results))
backtester.plot_results(results)
```

📈 PORTFOLIO BACKTEST:
```python
symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK"]
results = backtester.run_portfolio_backtest(symbols, "2023-01-01", "2024-12-31")
backtester.plot_results(results)
```

⚡ QUICK TEST COMMANDS:
python fyers_algo_backtester.py     # Run single stock demo
python fyers_backtest_demo.py       # Run comprehensive tests

🔧 CUSTOMIZE SETTINGS:
```python
backtester.config['risk_per_trade'] = 0.01      # 1% risk per trade
backtester.config['stop_loss_percent'] = 1.5    # 1.5% stop loss
backtester.config['take_profit_percent'] = 3.0  # 3% take profit
backtester.config['min_confidence'] = 80        # Higher confidence threshold
```

📊 ACCESS DETAILED RESULTS:
```python
for trade in results.trades:
    print(f"Trade: {trade.symbol} {trade.side} P&L: ₹{trade.pnl:.2f}")
    
print(f"Best Trade: ₹{max([t.pnl for t in results.trades]):.2f}")
print(f"Worst Trade: ₹{min([t.pnl for t in results.trades]):.2f}")
```

🎯 That's it! Start backtesting now to validate your FYERS strategy!
""")

def main():
    """Main function"""
    
    print("🧪 FYERS BACKTESTING COMPLETE GUIDE")
    print("="*42)
    
    show_backtesting_tutorial()
    show_quick_backtest_commands()
    
    print("""
🎉 CONCLUSION: BACKTEST BEFORE YOU TRADE!

Your FYERS algorithmic trading system is now equipped with:
✅ Complete backtesting framework
✅ Historical data download capability  
✅ Performance analysis and visualization
✅ Risk management validation
✅ Multiple testing scenarios

💡 Remember: A few hours of backtesting can save you thousands in real money!

🚀 Ready to backtest? Run these files:
   • fyers_algo_backtester.py (simple test)
   • fyers_backtest_demo.py (comprehensive testing)

📈 After successful backtesting, proceed to live FYERS trading!
""")

if __name__ == "__main__":
    main()