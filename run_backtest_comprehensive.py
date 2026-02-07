"""
FYERS BACKTESTING EXECUTION
Run comprehensive backtests with different parameters to show trading activity
"""

from fyers_algo_backtester import FyersAlgoBacktester
import json

def run_comprehensive_backtest():
    """Run backtests with different configurations"""
    
    print("🚀 FYERS ALGORITHMIC TRADING BACKTEST EXECUTION")
    print("=" * 55)
    
    # Test configurations - progressively more aggressive
    test_configs = [
        {
            "name": "Conservative Strategy",
            "min_confidence": 70,
            "stop_loss": 2.0,
            "take_profit": 4.0,
            "risk_per_trade": 0.015
        },
        {
            "name": "Moderate Strategy", 
            "min_confidence": 60,
            "stop_loss": 2.5,
            "take_profit": 3.5,
            "risk_per_trade": 0.02
        },
        {
            "name": "Aggressive Strategy",
            "min_confidence": 50,
            "stop_loss": 3.0,
            "take_profit": 3.0,
            "risk_per_trade": 0.025
        }
    ]
    
    symbols_to_test = ["RELIANCE", "TCS", "INFY"]
    results_summary = []
    
    for config in test_configs:
        print(f"\n📊 TESTING: {config['name']}")
        print("-" * 40)
        
        backtester = FyersAlgoBacktester()
        
        # Override configuration
        backtester.config.update({
            'min_confidence': config['min_confidence'],
            'stop_loss_percent': config['stop_loss'],
            'take_profit_percent': config['take_profit'],
            'risk_per_trade': config['risk_per_trade']
        })
        
        print(f"   🎯 Min Confidence: {config['min_confidence']}%")
        print(f"   📉 Stop Loss: {config['stop_loss']}%")
        print(f"   📈 Take Profit: {config['take_profit']}%")
        print(f"   💰 Risk per Trade: {config['risk_per_trade']*100}%")
        
        # Test single symbol
        for symbol in symbols_to_test:
            try:
                print(f"\n   📊 Testing {symbol}...")
                
                results = backtester.run_backtest(
                    symbol, 
                    "2023-06-01",  # Shorter period for faster testing
                    "2024-06-01"
                )
                
                results_summary.append({
                    "Strategy": config['name'],
                    "Symbol": symbol,
                    "Return": f"{results.total_return:+.2f}%",
                    "Trades": results.total_trades,
                    "Win_Rate": f"{results.win_rate:.1f}%",
                    "Max_DD": f"{results.max_drawdown:.2f}%"
                })
                
                print(f"      💹 Return: {results.total_return:+.2f}%")
                print(f"      🎯 Trades: {results.total_trades}")
                print(f"      ✅ Win Rate: {results.win_rate:.1f}%")
                
            except Exception as e:
                print(f"      ❌ Error testing {symbol}: {e}")
    
    # Summary report
    print(f"\n📋 COMPREHENSIVE BACKTEST RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Strategy':<18} {'Symbol':<8} {'Return':<10} {'Trades':<7} {'Win %':<8} {'Max DD':<8}")
    print("-" * 65)
    
    for result in results_summary:
        print(f"{result['Strategy']:<18} {result['Symbol']:<8} {result['Return']:<10} "
              f"{result['Trades']:<7} {result['Win_Rate']:<8} {result['Max_DD']:<8}")
    
    print(f"\n🎯 ANALYSIS:")
    
    # Find best performing strategy
    best_returns = {}
    total_trades = {}
    
    for result in results_summary:
        strategy = result['Strategy']
        if strategy not in best_returns:
            best_returns[strategy] = 0
            total_trades[strategy] = 0
        
        # Parse return percentage
        return_val = float(result['Return'].replace('%', '').replace('+', ''))
        best_returns[strategy] += return_val
        total_trades[strategy] += result['Trades']
    
    # Show recommendations
    print(f"📊 Strategy Performance (Combined):")
    for strategy in best_returns:
        avg_return = best_returns[strategy] / 3  # 3 symbols tested
        print(f"   {strategy}: {avg_return:+.2f}% avg return, {total_trades[strategy]} total trades")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if max(total_trades.values()) == 0:
        print("   🔧 All strategies generated 0 trades - consider:")
        print("      • Lower minimum confidence further (< 50%)")
        print("      • Adjust technical indicator parameters")
        print("      • Test on different time periods")
        print("      • Check signal generation logic")
    else:
        best_strategy = max(best_returns, key=best_returns.get)
        print(f"   🏆 Best Overall Strategy: {best_strategy}")
        print(f"   📈 Ready for live trading with optimized parameters")

if __name__ == "__main__":
    run_comprehensive_backtest()