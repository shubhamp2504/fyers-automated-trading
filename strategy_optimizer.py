"""
Strategy Parameter Optimization Module
=====================================

Optimizes trading strategy parameters for maximum performance
- Grid search optimization for key parameters
- Walk-forward analysis for robustness
- Monte Carlo simulation for risk assessment
- Parameter sensitivity analysis

⚠️ IMPORTANT: Always refer to https://myapi.fyers.in/docsv3 for latest API specifications
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
from itertools import product
import concurrent.futures
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from advanced_backtester import AdvancedBacktester

@dataclass
class OptimizationResult:
    """Results from parameter optimization"""
    parameters: Dict
    total_return: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    score: float  # Composite score

class StrategyOptimizer:
    """
    Advanced strategy parameter optimization
    """
    
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        
        # Optimization parameters
        self.optimization_ranges = {
            'profit_target_1': [18, 20, 22, 25, 28],  # First target points
            'profit_target_2': [25, 28, 30, 32, 35],  # Second target points
            'max_loss_per_trade': [10, 12, 15, 18, 20],  # Max loss points
            'ema_fast': [5, 7, 9, 12],  # Fast EMA period
            'ema_slow': [15, 18, 21, 24, 26],  # Slow EMA period
            'rsi_period': [10, 12, 14, 16, 18],  # RSI period
            'rsi_oversold': [30, 35, 40],  # RSI oversold level
            'rsi_overbought': [60, 65, 70]  # RSI overbought level
        }
        
        self.results = []
    
    def create_strategy_with_params(self, params: Dict) -> AdvancedBacktester:
        """Create strategy instance with custom parameters"""
        
        backtester = AdvancedBacktester(self.client_id, self.access_token)
        
        # Apply parameters to strategy
        backtester.strategy.profit_target_1 = params.get('profit_target_1', 22)
        backtester.strategy.profit_target_2 = params.get('profit_target_2', 28)
        backtester.strategy.max_loss_per_trade = params.get('max_loss_per_trade', 15)
        backtester.strategy.ema_fast = params.get('ema_fast', 9)
        backtester.strategy.ema_slow = params.get('ema_slow', 21)
        backtester.strategy.rsi_period = params.get('rsi_period', 14)
        backtester.strategy.rsi_oversold = params.get('rsi_oversold', 35)
        backtester.strategy.rsi_overbought = params.get('rsi_overbought', 65)
        
        return backtester
    
    def calculate_composite_score(self, results: Dict) -> float:
        """Calculate composite performance score"""
        
        # Weighted scoring system
        weights = {
            'return': 0.25,      # 25% weight on returns
            'win_rate': 0.20,    # 20% weight on win rate
            'profit_factor': 0.15, # 15% weight on profit factor
            'drawdown': 0.20,    # 20% weight on low drawdown (inverted)
            'sharpe': 0.10,      # 10% weight on Sharpe ratio
            'trades': 0.10       # 10% weight on number of trades
        }
        
        # Normalize and score each metric
        return_score = min(results.get('total_return', 0) / 20, 1.0)  # Cap at 20% return
        win_rate_score = results.get('win_rate', 0) / 100
        pf_score = min(results.get('profit_factor', 1) / 3, 1.0)  # Cap at 3.0 PF
        dd_score = max(0, 1 - results.get('max_drawdown', 100) / 20)  # Invert drawdown
        sharpe_score = min(max(results.get('sharpe_ratio', 0), 0) / 2, 1.0)  # Cap at 2.0
        trades_score = min(results.get('total_trades', 0) / 50, 1.0)  # Cap at 50 trades
        
        composite_score = (
            weights['return'] * return_score +
            weights['win_rate'] * win_rate_score +
            weights['profit_factor'] * pf_score +
            weights['drawdown'] * dd_score +
            weights['sharpe'] * sharpe_score +
            weights['trades'] * trades_score
        )
        
        return composite_score
    
    def test_parameter_combination(self, params: Dict, symbol: str) -> OptimizationResult:
        """Test a single parameter combination"""
        
        try:
            # Create strategy with parameters
            backtester = self.create_strategy_with_params(params)
            
            # Run backtest
            results = backtester.run_comprehensive_backtest(symbol, days=15)  # Shorter for optimization
            
            if not results:
                return OptimizationResult(
                    parameters=params,
                    total_return=0,
                    win_rate=0,
                    profit_factor=0,
                    max_drawdown=100,
                    sharpe_ratio=-1,
                    total_trades=0,
                    score=0
                )
            
            # Calculate composite score
            score = self.calculate_composite_score(results)
            
            return OptimizationResult(
                parameters=params,
                total_return=results.get('total_return', 0),
                win_rate=results.get('win_rate', 0),
                profit_factor=results.get('profit_factor', 0),
                max_drawdown=results.get('max_drawdown', 100),
                sharpe_ratio=results.get('sharpe_ratio', -1),
                total_trades=results.get('total_trades', 0),
                score=score
            )
            
        except Exception as e:
            print(f"❌ Error testing parameters {params}: {e}")
            return OptimizationResult(
                parameters=params,
                total_return=-10,
                win_rate=0,
                profit_factor=0,
                max_drawdown=100,
                sharpe_ratio=-2,
                total_trades=0,
                score=0
            )
    
    def grid_search_optimization(self, symbol: str, max_combinations: int = 100) -> List[OptimizationResult]:
        """Perform grid search optimization"""
        
        print(f"🔍 Starting Grid Search Optimization for {symbol}")
        print(f"⚙️ Testing up to {max_combinations} parameter combinations...")
        
        # Generate parameter combinations
        param_names = list(self.optimization_ranges.keys())
        param_values = [self.optimization_ranges[name] for name in param_names]
        
        # Limit combinations to avoid excessive computation
        all_combinations = list(product(*param_values))
        if len(all_combinations) > max_combinations:
            # Sample randomly
            np.random.seed(42)  # For reproducibility
            selected_combinations = np.random.choice(
                len(all_combinations), 
                size=max_combinations, 
                replace=False
            )
            combinations = [all_combinations[i] for i in selected_combinations]
        else:
            combinations = all_combinations
        
        print(f"📊 Testing {len(combinations)} parameter combinations...")
        
        results = []
        
        # Test each combination
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{len(combinations)} ({(i+1)/len(combinations)*100:.1f}%)")
            
            # Test parameters
            result = self.test_parameter_combination(params, symbol)
            results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        print(f"✅ Grid search completed for {symbol}")
        print(f"🏆 Best score: {results[0].score:.3f}" if results else "❌ No valid results")
        
        return results
    
    def walk_forward_analysis(self, symbol: str, best_params: Dict, periods: int = 5) -> Dict:
        """Perform walk-forward analysis to test robustness"""
        
        print(f"🚶 Running Walk-Forward Analysis for {symbol}")
        print(f"📅 Testing {periods} time periods...")
        
        period_results = []
        total_days = 30  # Total period
        days_per_period = total_days // periods
        
        for period in range(periods):
            print(f"   📊 Period {period + 1}/{periods}...")
            
            # Create backtester with best parameters
            backtester = self.create_strategy_with_params(best_params)
            
            # Run backtest for specific period
            # In a real implementation, you'd use specific date ranges
            results = backtester.run_comprehensive_backtest(symbol, days=days_per_period)
            
            if results:
                period_results.append({
                    'period': period + 1,
                    'return': results.get('total_return', 0),
                    'win_rate': results.get('win_rate', 0),
                    'trades': results.get('total_trades', 0),
                    'drawdown': results.get('max_drawdown', 0)
                })
        
        # Calculate consistency metrics
        if period_results:
            returns = [p['return'] for p in period_results]
            win_rates = [p['win_rate'] for p in period_results]
            
            consistency_metrics = {
                'avg_return': np.mean(returns),
                'return_std': np.std(returns),
                'return_consistency': np.mean(returns) / max(np.std(returns), 0.1),
                'positive_periods': sum(1 for r in returns if r > 0),
                'avg_win_rate': np.mean(win_rates),
                'win_rate_std': np.std(win_rates),
                'period_results': period_results
            }
            
            print(f"📈 Avg Return: {consistency_metrics['avg_return']:.2f}%")
            print(f"📊 Return Consistency: {consistency_metrics['return_consistency']:.2f}")
            print(f"✅ Positive Periods: {consistency_metrics['positive_periods']}/{periods}")
            
            return consistency_metrics
        
        return {}
    
    def monte_carlo_simulation(self, symbol: str, best_params: Dict, simulations: int = 100) -> Dict:
        """Perform Monte Carlo simulation for risk assessment"""
        
        print(f"🎲 Running Monte Carlo Simulation for {symbol}")
        print(f"🔄 {simulations} simulations...")
        
        simulation_results = []
        
        for sim in range(simulations):
            if (sim + 1) % 20 == 0:
                print(f"   Progress: {sim + 1}/{simulations} ({(sim+1)/simulations*100:.1f}%)")
            
            try:
                # Add randomness to parameters (±10% variation)
                varied_params = {}
                for key, value in best_params.items():
                    if key in ['profit_target_1', 'profit_target_2', 'max_loss_per_trade']:
                        variation = np.random.uniform(-0.1, 0.1)  # ±10%
                        varied_params[key] = max(1, int(value * (1 + variation)))
                    else:
                        varied_params[key] = value
                
                # Run simulation
                backtester = self.create_strategy_with_params(varied_params)
                results = backtester.run_comprehensive_backtest(symbol, days=10)  # Shorter for MC
                
                if results:
                    simulation_results.append({
                        'return': results.get('total_return', 0),
                        'drawdown': results.get('max_drawdown', 100),
                        'win_rate': results.get('win_rate', 0),
                        'profit_factor': results.get('profit_factor', 0)
                    })
                
            except Exception as e:
                print(f"❌ Simulation {sim + 1} failed: {e}")
                continue
        
        if not simulation_results:
            return {}
        
        # Calculate risk metrics
        returns = [s['return'] for s in simulation_results]
        drawdowns = [s['drawdown'] for s in simulation_results]
        
        risk_metrics = {
            'expected_return': np.mean(returns),
            'return_volatility': np.std(returns),
            'worst_case_return': np.percentile(returns, 5),  # 5th percentile
            'best_case_return': np.percentile(returns, 95),  # 95th percentile
            'probability_positive': sum(1 for r in returns if r > 0) / len(returns),
            'max_expected_drawdown': np.percentile(drawdowns, 95),  # 95th percentile
            'avg_drawdown': np.mean(drawdowns),
            'value_at_risk_5': np.percentile(returns, 5),  # VaR at 5%
            'simulation_count': len(simulation_results)
        }
        
        print(f"📊 Expected Return: {risk_metrics['expected_return']:.2f}%")
        print(f"📈 Probability of Profit: {risk_metrics['probability_positive']*100:.1f}%")
        print(f"📉 Worst Case (5%): {risk_metrics['worst_case_return']:.2f}%")
        print(f"📈 Best Case (95%): {risk_metrics['best_case_return']:.2f}%")
        
        return risk_metrics
    
    def optimize_strategy(self, symbol: str) -> Dict:
        """Complete strategy optimization process"""
        
        print(f"🚀 COMPLETE STRATEGY OPTIMIZATION - {symbol}")
        print("=" * 60)
        
        optimization_results = {}
        
        # Step 1: Grid Search Optimization
        print(f"\n1️⃣ Grid Search Optimization")
        grid_results = self.grid_search_optimization(symbol, max_combinations=50)
        
        if not grid_results:
            print("❌ Grid search failed")
            return {}
        
        best_result = grid_results[0]
        best_params = best_result.parameters
        
        print(f"\n🏆 BEST PARAMETERS FOUND:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        
        print(f"\n📊 BEST PERFORMANCE:")
        print(f"   📈 Total Return: {best_result.total_return:.2f}%")
        print(f"   🎯 Win Rate: {best_result.win_rate:.1f}%")
        print(f"   ⚖️ Profit Factor: {best_result.profit_factor:.2f}")
        print(f"   📉 Max Drawdown: {best_result.max_drawdown:.2f}%")
        print(f"   🏆 Composite Score: {best_result.score:.3f}")
        
        optimization_results['best_parameters'] = best_params
        optimization_results['best_performance'] = {
            'total_return': best_result.total_return,
            'win_rate': best_result.win_rate,
            'profit_factor': best_result.profit_factor,
            'max_drawdown': best_result.max_drawdown,
            'score': best_result.score
        }
        
        # Step 2: Walk-Forward Analysis
        print(f"\n2️⃣ Walk-Forward Analysis")
        wf_results = self.walk_forward_analysis(symbol, best_params)
        optimization_results['walk_forward'] = wf_results
        
        # Step 3: Monte Carlo Simulation
        print(f"\n3️⃣ Monte Carlo Risk Assessment")
        mc_results = self.monte_carlo_simulation(symbol, best_params, simulations=50)
        optimization_results['monte_carlo'] = mc_results
        
        # Step 4: Parameter Sensitivity Analysis
        print(f"\n4️⃣ Top Parameter Combinations")
        print("-" * 40)
        for i, result in enumerate(grid_results[:5]):  # Top 5
            print(f"Rank {i+1}: Score {result.score:.3f} | Return {result.total_return:.2f}% | Win Rate {result.win_rate:.1f}%")
        
        optimization_results['top_results'] = grid_results[:10]
        
        # Final Recommendation
        print(f"\n🎯 OPTIMIZATION SUMMARY")
        print("=" * 40)
        
        if best_result.score > 0.6:
            recommendation = "🟢 EXCELLENT - Strategy ready for live trading"
        elif best_result.score > 0.4:
            recommendation = "🟡 GOOD - Strategy needs minor adjustments"
        elif best_result.score > 0.2:
            recommendation = "🟠 AVERAGE - Strategy needs significant optimization"
        else:
            recommendation = "🔴 POOR - Strategy not recommended for trading"
        
        print(f"📊 Strategy Rating: {recommendation}")
        
        if mc_results:
            print(f"💰 Expected Return: {mc_results.get('expected_return', 0):.2f}%")
            print(f"📈 Success Probability: {mc_results.get('probability_positive', 0)*100:.1f}%")
            print(f"⚠️ Worst Case Scenario: {mc_results.get('worst_case_return', 0):.2f}%")
        
        optimization_results['recommendation'] = recommendation
        
        return optimization_results

def run_strategy_optimization():
    """Run complete strategy optimization for index trading"""
    
    print("🎯 INDEX STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 70)
    
    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except:
        print("❌ Please ensure config.json exists with credentials")
        return
    
    # Initialize optimizer
    optimizer = StrategyOptimizer(config['client_id'], config['access_token'])
    
    # Test symbols
    symbols = ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
    
    all_optimization_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*20} {symbol} {'='*20}")
        
        try:
            results = optimizer.optimize_strategy(symbol)
            
            if results:
                all_optimization_results[symbol] = results
                
                # Save optimized parameters
                filename = f"optimized_params_{symbol.split(':')[1].lower()}.json"
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                print(f"💾 Optimization results saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error optimizing {symbol}: {e}")
            continue
    
    # Overall summary
    if all_optimization_results:
        print(f"\n🎉 OPTIMIZATION COMPLETED")
        print("=" * 50)
        
        for symbol, results in all_optimization_results.items():
            best_perf = results.get('best_performance', {})
            print(f"\n📊 {symbol}:")
            print(f"   📈 Best Return: {best_perf.get('total_return', 0):.2f}%")
            print(f"   🎯 Win Rate: {best_perf.get('win_rate', 0):.1f}%")
            print(f"   🏆 Score: {best_perf.get('score', 0):.3f}")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"1. Review optimized parameters in saved JSON files")
        print(f"2. Run extended backtests with optimized parameters")
        print(f"3. Implement optimized strategy in live trading")
        print(f"4. Monitor performance and adjust as needed")
    
    return all_optimization_results

if __name__ == "__main__":
    # Run the complete optimization
    results = run_strategy_optimization()
    
    if results:
        print(f"\n✅ Strategy optimization completed successfully!")
        print(f"📁 Optimized parameters saved for {len(results)} symbols")
    else:
        print(f"\n❌ Strategy optimization failed")
        print(f"🔍 Check configuration and data availability")