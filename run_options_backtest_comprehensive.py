"""
INDEX OPTIONS BACKTESTING EXECUTION
Comprehensive backtesting demo for Index Options Trading Strategies
"""

from index_options_backtester import IndexOptionsBacktester
import json
from datetime import datetime, timedelta

def run_comprehensive_options_backtest():
    """Execute comprehensive options backtesting across multiple strategies"""
    
    print("🚀 INDEX OPTIONS BACKTESTING EXECUTION")
    print("📈 NIFTY50, BANKNIFTY, FINNIFTY Options Strategies")
    print("=" * 60)
    
    try:
        backtester = IndexOptionsBacktester()
        
        # Comprehensive test matrix
        test_matrix = [
            # NIFTY50 Strategies
            {
                "name": "NIFTY50 Long Call (Bullish)",
                "index": "NIFTY50",
                "strategy": "LONG_CALL",
                "period": {"start": "2023-08-01", "end": "2023-10-01"},
                "params": {
                    "moneyness": "ATM",
                    "dte_range": [7, 14],
                    "entry_conditions": ["uptrend", "low_iv"],
                    "exit_conditions": ["50_profit", "expiry_minus_1"]
                }
            },
            {
                "name": "NIFTY50 Short Put (Income)",
                "index": "NIFTY50", 
                "strategy": "SHORT_PUT",
                "period": {"start": "2023-08-01", "end": "2023-10-01"},
                "params": {
                    "moneyness": "OTM",
                    "delta_range": [0.15, 0.25],
                    "dte_range": [14, 21],
                    "profit_target": 50,
                    "entry_conditions": ["support_level", "high_iv"]
                }
            },
            {
                "name": "NIFTY50 ATM Straddle (Volatility)",
                "index": "NIFTY50",
                "strategy": "STRADDLE", 
                "period": {"start": "2023-08-01", "end": "2023-10-01"},
                "params": {
                    "moneyness": "ATM",
                    "dte_range": [7, 14],
                    "iv_threshold": 20,
                    "breakeven_buffer": 2,
                    "entry_conditions": ["event_week", "rising_iv"]
                }
            },
            
            # BANKNIFTY Strategies  
            {
                "name": "BANKNIFTY Iron Condor (Range)",
                "index": "BANKNIFTY",
                "strategy": "IRON_CONDOR",
                "period": {"start": "2023-08-01", "end": "2023-10-01"}, 
                "params": {
                    "call_strikes": {"short": "ATM+200", "long": "ATM+400"},
                    "put_strikes": {"short": "ATM-200", "long": "ATM-400"},
                    "dte_range": [14, 21],
                    "profit_target": 25,
                    "max_loss": 75
                }
            },
            {
                "name": "BANKNIFTY Scalping Calls",
                "index": "BANKNIFTY",
                "strategy": "LONG_CALL",
                "period": {"start": "2023-08-01", "end": "2023-10-01"},
                "params": {
                    "moneyness": "ITM", 
                    "delta_range": [0.6, 0.8],
                    "dte_range": [0, 2],
                    "scalping": True,
                    "quick_profit": 20
                }
            },
            
            # FINNIFTY Strategies
            {
                "name": "FINNIFTY Weekly Strangles",
                "index": "FINNIFTY",
                "strategy": "STRANGLE",
                "period": {"start": "2023-08-01", "end": "2023-10-01"},
                "params": {
                    "call_moneyness": "OTM+100",
                    "put_moneyness": "OTM-100",
                    "dte_range": [2, 5],
                    "weekly_expiry": True
                }
            }
        ]
        
        results_summary = []
        
        print(f"\\n📊 EXECUTING {len(test_matrix)} OPTIONS STRATEGIES")
        print("="*60)
        
        for i, test_config in enumerate(test_matrix, 1):
            print(f"\\n🎯 Test {i}/{len(test_matrix)}: {test_config['name']}")
            print("-" * 50)
            
            print(f"   📈 Index: {test_config['index']}")
            print(f"   💡 Strategy: {test_config['strategy']}")
            print(f"   📅 Period: {test_config['period']['start']} to {test_config['period']['end']}")
            
            # Display key parameters
            key_params = test_config['params']
            if 'moneyness' in key_params:
                print(f"   🎯 Moneyness: {key_params['moneyness']}")
            if 'dte_range' in key_params:
                print(f"   ⏰ DTE Range: {key_params['dte_range']} days")
            if 'profit_target' in key_params:
                print(f"   💰 Profit Target: {key_params['profit_target']}%")
            
            try:
                # Test option chain first
                print(f"   📊 Fetching option chain...")
                option_chain = backtester.get_option_chain(test_config['index'])
                
                if option_chain:
                    spot_price = option_chain.get('underlying', {}).get('ltp', 0)
                    options_count = len(option_chain.get('options', []))
                    print(f"   ✅ Options available: {options_count} strikes")
                    print(f"   💹 Current Spot: ₹{spot_price:,.2f}")
                    
                    # Show ATM options for reference
                    if options_count > 0:
                        options_list = option_chain['options']
                        # Find ATM option
                        atm_option = min(options_list, key=lambda x: abs(x['strike_price'] - spot_price))
                        call_premium = atm_option['call']['ltp']
                        put_premium = atm_option['put']['ltp']
                        print(f"   🔵 ATM {atm_option['strike_price']} CE: ₹{call_premium}")
                        print(f"   🔴 ATM {atm_option['strike_price']} PE: ₹{put_premium}")
                
                # Run backtest
                print(f"   🚀 Running backtest...")
                
                results = backtester.run_options_backtest(
                    index=test_config['index'],
                    strategy=test_config['strategy'],
                    start_date=test_config['period']['start'],
                    end_date=test_config['period']['end'],
                    **test_config['params']
                )
                
                # Store results
                results_summary.append({
                    "name": test_config['name'],
                    "index": test_config['index'],
                    "strategy": test_config['strategy'],
                    "return": results.total_return,
                    "trades": results.total_trades,
                    "win_rate": results.win_rate,
                    "max_profit": results.max_profit_trade,
                    "max_loss": results.max_loss_trade,
                    "avg_days": results.avg_days_held
                })
                
                print(f"   ✅ Backtest completed!")
                print(f"   💰 Return: {results.total_return:+.2f}%")
                print(f"   🎯 Win Rate: {results.win_rate:.1f}%")
                print(f"   📈 Total Trades: {results.total_trades}")
                
                if results.total_trades > 0:
                    print(f"   🏆 Best Trade: ₹{results.max_profit_trade:+,.2f}")
                    print(f"   💸 Worst Trade: ₹{results.max_loss_trade:+,.2f}")
                    print(f"   ⏰ Avg Hold: {results.avg_days_held:.1f} days")
                
            except Exception as e:
                print(f"   ❌ Error in backtest: {e}")
                results_summary.append({
                    "name": test_config['name'],
                    "index": test_config['index'],
                    "strategy": test_config['strategy'],
                    "return": 0.0,
                    "trades": 0,
                    "win_rate": 0.0,
                    "max_profit": 0.0,
                    "max_loss": 0.0,
                    "avg_days": 0.0
                })
        
        # Generate comprehensive summary report
        print(f"\\n📋 OPTIONS BACKTESTING RESULTS SUMMARY")
        print("="*80)
        print(f"{'Strategy':<25} {'Index':<10} {'Return':<8} {'Trades':<7} {'Win%':<6} {'Best':<10} {'Worst':<10}")
        print("-"*80)
        
        for result in results_summary:
            print(f"{result['name'][:24]:<25} "
                  f"{result['index']:<10} "
                  f"{result['return']:+6.2f}% "
                  f"{result['trades']:<7d} "
                  f"{result['win_rate']:5.1f}% "
                  f"₹{result['max_profit']:7,.0f} "
                  f"₹{result['max_loss']:7,.0f}")
        
        # Analysis and recommendations
        print(f"\\n🎯 ANALYSIS & INSIGHTS:")
        
        # Calculate strategy type performance
        strategy_performance = {}
        for result in results_summary:
            strategy = result['strategy']
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {'total_return': 0, 'count': 0, 'trades': 0}
            
            strategy_performance[strategy]['total_return'] += result['return']
            strategy_performance[strategy]['count'] += 1
            strategy_performance[strategy]['trades'] += result['trades']
        
        print(f"\\n📊 Performance by Strategy Type:")
        for strategy, perf in strategy_performance.items():
            avg_return = perf['total_return'] / perf['count'] if perf['count'] > 0 else 0
            total_trades = perf['trades']
            print(f"   {strategy}: {avg_return:+.2f}% avg return, {total_trades} total trades")
        
        # Index-wise analysis
        index_performance = {}
        for result in results_summary:
            index = result['index'] 
            if index not in index_performance:
                index_performance[index] = {'total_return': 0, 'count': 0}
            
            index_performance[index]['total_return'] += result['return']
            index_performance[index]['count'] += 1
        
        print(f"\\n📈 Performance by Index:")
        for index, perf in index_performance.items():
            avg_return = perf['total_return'] / perf['count'] if perf['count'] > 0 else 0
            print(f"   {index}: {avg_return:+.2f}% average return across strategies")
        
        print(f"\\n💡 KEY RECOMMENDATIONS:")
        
        # Find best performing strategy
        best_strategy = max(results_summary, key=lambda x: x['return'])
        if best_strategy['return'] > 0:
            print(f"   🏆 Best Strategy: {best_strategy['name']}")
            print(f"      Return: {best_strategy['return']:+.2f}%, Win Rate: {best_strategy['win_rate']:.1f}%")
        
        # Check if any strategies generated trades
        total_trades_all = sum(r['trades'] for r in results_summary)
        if total_trades_all == 0:
            print(f"   🔧 No trades generated - Suggestions:")
            print(f"      • Configure real FYERS API for accurate options data")
            print(f"      • Implement specific entry/exit logic for each strategy")
            print(f"      • Add volatility-based triggers and market regime filters")
            print(f"      • Test during high volatility periods (earnings, events)")
        else:
            print(f"   ✅ Options backtesting infrastructure fully operational")
            print(f"   📊 Total trades across all strategies: {total_trades_all}")
            print(f"   🚀 Ready for live options trading implementation")
        
        print(f"\\n🔗 NEXT STEPS:")
        print(f"   1. 🔑 Configure FYERS API credentials for live options data")
        print(f"   2. 📊 Implement real-time Greeks calculations (Delta, Gamma, Theta, Vega)")
        print(f"   3. 🎯 Add advanced entry/exit triggers based on market conditions")
        print(f"   4. 💰 Implement position sizing and portfolio risk management")
        print(f"   5. 🚀 Deploy automated options trading with risk controls")
        
        # Generate sample detailed report
        sample_result = results_summary[0] if results_summary else None
        if sample_result and sample_result['trades'] > 0:
            print(f"\\n📄 SAMPLE DETAILED REPORT:")
            sample_backtest = backtester.run_options_backtest(
                index=sample_result['index'],
                strategy=sample_result['strategy'],
                start_date="2023-08-01",
                end_date="2023-10-01"
            )
            
            detailed_report = backtester.create_options_report(sample_backtest)
            print(detailed_report)
        
    except Exception as e:
        print(f"\\n❌ Error in options backtesting: {e}")
        print(f"💡 Troubleshooting:")
        print(f"   • Ensure all required packages are installed")
        print(f"   • Check FYERS API configuration")
        print(f"   • Verify internet connection for data retrieval")

if __name__ == "__main__":
    run_comprehensive_options_backtest()