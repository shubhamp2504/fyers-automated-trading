"""
Standalone Index Intraday Strategy Demo
=======================================

Complete demonstration of the strategy with simulated market data
No API credentials required - Educational purposes only

⚠️ This is a simulation using synthetic market data
Real trading requires live market data and proper API credentials
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import random

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class SimulatedTrade:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    signal: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_points: float
    exit_reason: str
    duration_minutes: int

class MarketDataSimulator:
    """Simulate realistic market data for demonstration"""
    
    def __init__(self):
        # Base prices for indices
        self.base_prices = {
            'NIFTY50': 18500.0,
            'BANKNIFTY': 43500.0
        }
        
        # Market characteristics
        self.volatility = 0.015  # 1.5% daily volatility
        self.trend_strength = 0.3
        self.noise_factor = 0.02
    
    def generate_intraday_data(self, symbol: str, days: int = 5) -> Dict[str, pd.DataFrame]:
        """Generate realistic intraday OHLCV data"""
        
        base_price = self.base_prices.get(symbol, 20000.0)
        
        # Generate 1H data
        hourly_data = []
        current_price = base_price
        
        for day in range(days):
            # Daily trend (random walk with slight bias)
            daily_trend = np.random.normal(0, self.volatility)
            
            # Generate hourly candles for market hours (9:15 AM to 3:15 PM = 6 hours)
            for hour in range(6):
                # Hourly price movement
                price_change = np.random.normal(daily_trend/6, self.volatility/4)
                
                # Generate OHLC
                open_price = current_price
                close_price = current_price * (1 + price_change)
                
                high_range = abs(close_price - open_price) * (1 + random.uniform(0.2, 0.8))
                low_range = abs(close_price - open_price) * (1 + random.uniform(0.2, 0.8))
                
                if close_price > open_price:  # Bullish candle
                    high_price = max(open_price, close_price) + high_range
                    low_price = min(open_price, close_price) - low_range * 0.5
                else:  # Bearish candle
                    high_price = max(open_price, close_price) + high_range * 0.5
                    low_price = min(open_price, close_price) - low_range
                
                volume = random.randint(100000, 500000)
                
                # Create timestamp
                date = datetime.now() - timedelta(days=days-day-1)
                timestamp = date.replace(hour=9+hour, minute=15, second=0, microsecond=0)
                
                hourly_data.append({
                    'timestamp': timestamp,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': volume
                })
                
                current_price = close_price
        
        # Create DataFrame
        df_1h = pd.DataFrame(hourly_data)
        df_1h.set_index('timestamp', inplace=True)
        
        # Generate 5M data (simplified - just split each 1H candle into 12 x 5min candles)
        minute_data = []
        
        for _, row in df_1h.iterrows():
            hour_timestamp = row.name
            hour_open = row['open']
            hour_close = row['close']
            hour_high = row['high']
            hour_low = row['low']
            
            # Generate 12 x 5-minute candles
            for minute_interval in range(0, 60, 5):
                # Linear progression from hour open to close with some noise
                progress = minute_interval / 60
                base_price = hour_open + (hour_close - hour_open) * progress
                
                # Add some intra-hour volatility
                noise = np.random.normal(0, (hour_high - hour_low) * 0.1)
                
                open_5m = base_price
                close_5m = base_price + noise
                
                # Ensure 5M candles stay within 1H range
                high_5m = min(hour_high, max(open_5m, close_5m) * (1 + random.uniform(0, 0.005)))
                low_5m = max(hour_low, min(open_5m, close_5m) * (1 - random.uniform(0, 0.005)))
                
                minute_timestamp = hour_timestamp + timedelta(minutes=minute_interval)
                
                minute_data.append({
                    'timestamp': minute_timestamp,
                    'open': round(open_5m, 2),
                    'high': round(high_5m, 2),
                    'low': round(low_5m, 2),
                    'close': round(close_5m, 2),
                    'volume': row['volume'] // 12
                })
        
        df_5m = pd.DataFrame(minute_data)
        df_5m.set_index('timestamp', inplace=True)
        
        return {'1h': df_1h, '5m': df_5m}

class SimulatedStrategy:
    """Simulated index intraday trading strategy"""
    
    def __init__(self):
        self.simulator = MarketDataSimulator()
        
        # Strategy parameters
        self.ema_fast = 9
        self.ema_slow = 21
        self.rsi_period = 14
        self.profit_target_1 = 22
        self.profit_target_2 = 28
        self.max_loss = 15
        
        # Trading results
        self.trades: List[SimulatedTrade] = []
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        
        # EMAs
        df['EMA_9'] = df['close'].ewm(span=self.ema_fast).mean()
        df['EMA_21'] = df['close'].ewm(span=self.ema_slow).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # VWAP
        df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # ATR for stop loss
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        return df
    
    def generate_signals(self, df_1h: pd.DataFrame) -> List[Dict]:
        """Generate trading signals from 1H data"""
        
        signals = []
        
        for i in range(self.ema_slow, len(df_1h) - 1):
            current = df_1h.iloc[i]
            previous = df_1h.iloc[i-1]
            
            # Skip if not enough data
            if pd.isna(current['EMA_9']) or pd.isna(current['RSI']):
                continue
            
            # Buy signal conditions
            if (current['EMA_9'] > current['EMA_21'] and
                previous['EMA_9'] <= previous['EMA_21'] and  # Fresh crossover
                35 < current['RSI'] < 70 and
                current['close'] > current['VWAP']):
                
                stop_loss = current['close'] - min(self.max_loss, current['ATR'] * 1.5)
                
                signals.append({
                    'timestamp': current.name,
                    'signal': 'BUY',
                    'entry_price': current['close'],
                    'stop_loss': stop_loss,
                    'target_1': current['close'] + self.profit_target_1,
                    'target_2': current['close'] + self.profit_target_2
                })
            
            # Sell signal conditions
            elif (current['EMA_9'] < current['EMA_21'] and
                  previous['EMA_9'] >= previous['EMA_21'] and  # Fresh crossover
                  30 < current['RSI'] < 65 and
                  current['close'] < current['VWAP']):
                
                stop_loss = current['close'] + min(self.max_loss, current['ATR'] * 1.5)
                
                signals.append({
                    'timestamp': current.name,
                    'signal': 'SELL',
                    'entry_price': current['close'],
                    'stop_loss': stop_loss,
                    'target_1': current['close'] - self.profit_target_1,
                    'target_2': current['close'] - self.profit_target_2
                })
        
        return signals
    
    def simulate_trade_execution(self, signal: Dict, df_5m: pd.DataFrame) -> Optional[SimulatedTrade]:
        """Simulate trade execution using 5M data"""
        
        signal_time = signal['timestamp']
        entry_price = signal['entry_price']
        
        # Find corresponding 5M candles
        future_candles = df_5m[df_5m.index > signal_time]
        
        if len(future_candles) < 10:  # Need at least 50 minutes of data
            return None
        
        # Simulate trade progression
        for i, (timestamp, candle) in enumerate(future_candles.iterrows()):
            high = candle['high']
            low = candle['low']
            close = candle['close']
            
            # Check exit conditions
            if signal['signal'] == 'BUY':
                # Stop loss hit
                if low <= signal['stop_loss']:
                    exit_price = signal['stop_loss']
                    exit_reason = 'STOP_LOSS'
                    pnl_points = exit_price - entry_price
                    break
                
                # Target 2 hit
                elif high >= signal['target_2']:
                    exit_price = signal['target_2']
                    exit_reason = 'TARGET_2'
                    pnl_points = exit_price - entry_price
                    break
                
                # Target 1 hit (70% chance to continue, 30% to exit)
                elif high >= signal['target_1'] and random.random() < 0.3:
                    exit_price = signal['target_1']
                    exit_reason = 'TARGET_1'
                    pnl_points = exit_price - entry_price
                    break
            
            else:  # SELL signal
                # Stop loss hit
                if high >= signal['stop_loss']:
                    exit_price = signal['stop_loss']
                    exit_reason = 'STOP_LOSS'
                    pnl_points = entry_price - exit_price
                    break
                
                # Target 2 hit
                elif low <= signal['target_2']:
                    exit_price = signal['target_2']
                    exit_reason = 'TARGET_2'
                    pnl_points = entry_price - exit_price
                    break
                
                # Target 1 hit (70% chance to continue, 30% to exit)
                elif low <= signal['target_1'] and random.random() < 0.3:
                    exit_price = signal['target_1']
                    exit_reason = 'TARGET_1'
                    pnl_points = entry_price - exit_price
                    break
            
            # Force exit after 6 hours (market close)
            if i >= 72:  # 6 hours = 72 x 5-minute candles
                exit_price = close
                exit_reason = 'MARKET_CLOSE'
                
                if signal['signal'] == 'BUY':
                    pnl_points = exit_price - entry_price
                else:
                    pnl_points = entry_price - exit_price
                break
        else:
            # No exit condition met
            return None
        
        # Calculate trade results
        duration_minutes = int((timestamp - signal_time).total_seconds() / 60)
        lot_size = 25  # NIFTY lot size
        pnl_rupees = pnl_points * lot_size - 20  # Subtract brokerage
        
        trade = SimulatedTrade(
            entry_time=signal_time,
            exit_time=timestamp,
            symbol='NIFTY50',
            signal=signal['signal'],
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl_rupees,
            pnl_points=pnl_points,
            exit_reason=exit_reason,
            duration_minutes=duration_minutes
        )
        
        return trade
    
    def run_backtest(self, symbol: str = 'NIFTY50', days: int = 5) -> Dict:
        """Run complete backtest simulation"""
        
        print(f"🔄 Running Strategy Backtest Simulation")
        print(f"📊 Symbol: {symbol} | Period: {days} days")
        print("-" * 50)
        
        # Generate market data
        market_data = self.simulator.generate_intraday_data(symbol, days)
        df_1h = market_data['1h']
        df_5m = market_data['5m']
        
        # Calculate indicators
        df_1h = self.calculate_indicators(df_1h)
        df_5m = self.calculate_indicators(df_5m)
        
        print(f"📈 Generated {len(df_1h)} hourly candles and {len(df_5m)} 5-minute candles")
        
        # Generate signals
        signals = self.generate_signals(df_1h)
        print(f"🎯 Generated {len(signals)} trading signals")
        
        # Execute trades
        self.trades = []
        for i, signal in enumerate(signals):
            trade = self.simulate_trade_execution(signal, df_5m)
            if trade:
                self.trades.append(trade)
                
                outcome = "🟢 WIN" if trade.pnl > 0 else "🔴 LOSS"
                print(f"  {outcome} Trade {i+1}: {trade.signal} @ ₹{trade.entry_price:.2f} → ₹{trade.exit_price:.2f}")
                print(f"      💰 P&L: ₹{trade.pnl:+.2f} ({trade.pnl_points:+.1f} pts) | {trade.exit_reason}")
        
        # Calculate performance metrics
        results = self.calculate_performance()
        
        print(f"\n📊 BACKTEST RESULTS")
        print("=" * 30)
        print(f"📈 Total Signals: {len(signals)}")
        print(f"✅ Executed Trades: {len(self.trades)}")
        print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
        print(f"💰 Total P&L: ₹{results['total_pnl']:+,.2f}")
        print(f"📈 Best Trade: ₹{results['best_trade']:+.2f}")
        print(f"📉 Worst Trade: ₹{results['worst_trade']:+.2f}")
        print(f"⚖️ Avg Win: ₹{results['avg_win']:.2f}")
        print(f"⚖️ Avg Loss: ₹{results['avg_loss']:.2f}")
        print(f"📊 Profit Factor: {results['profit_factor']:.2f}")
        
        return results
    
    def calculate_performance(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not self.trades:
            return {
                'total_pnl': 0, 'win_rate': 0, 'total_trades': 0,
                'winning_trades': 0, 'losing_trades': 0,
                'best_trade': 0, 'worst_trade': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0
            }
        
        total_pnl = sum(trade.pnl for trade in self.trades)
        winning_trades = [trade for trade in self.trades if trade.pnl > 0]
        losing_trades = [trade for trade in self.trades if trade.pnl <= 0]
        
        win_rate = (len(winning_trades) / len(self.trades)) * 100
        
        best_trade = max((trade.pnl for trade in self.trades), default=0)
        worst_trade = min((trade.pnl for trade in self.trades), default=0)
        
        avg_win = sum(trade.pnl for trade in winning_trades) / max(len(winning_trades), 1)
        avg_loss = abs(sum(trade.pnl for trade in losing_trades)) / max(len(losing_trades), 1)
        
        gross_profit = sum(trade.pnl for trade in winning_trades)
        gross_loss = abs(sum(trade.pnl for trade in losing_trades))
        profit_factor = gross_profit / max(gross_loss, 1)
        
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }

def demo_strategy_features():
    """Demonstrate key strategy features"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                INDEX INTRADAY STRATEGY DEMO                   ║
    ║                                                              ║
    ║  🎯 Educational Simulation with Synthetic Data                ║
    ║  📊 No Real API Calls - Safe for Learning                    ║
    ║  💰 Realistic Market Behavior Simulation                     ║
    ║  🛡️ Complete Risk Management Framework                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n🎯 STRATEGY FEATURES:")
    print("-" * 30)
    print("• Multi-timeframe analysis (1H + 5M)")
    print("• EMA crossover for trend identification")
    print("• RSI momentum confirmation")
    print("• VWAP strength validation")
    print("• Dynamic ATR-based stop losses")
    print("• 20-30 point profit targets")
    print("• Smart position sizing")
    print("• Comprehensive risk management")
    
    print("\n⚠️ IMPORTANT NOTES:")
    print("-" * 20)
    print("• This is a SIMULATION using synthetic market data")
    print("• Real trading requires live market data and API access")
    print("• Always test strategies thoroughly before live trading")
    print("• Never risk more than you can afford to lose")
    
    print("\n🚀 Running Strategy Demonstration...")
    
    # Initialize strategy
    strategy = SimulatedStrategy()
    
    # Run multiple backtests
    all_results = []
    
    for i in range(3):  # Run 3 different simulations
        print(f"\n🔄 Simulation {i+1}/3")
        results = strategy.run_backtest('NIFTY50', days=7)
        all_results.append(results)
    
    # Combined results
    print(f"\n🎯 COMBINED SIMULATION RESULTS")
    print("=" * 40)
    
    total_trades = sum(r['total_trades'] for r in all_results)
    total_pnl = sum(r['total_pnl'] for r in all_results)
    avg_win_rate = sum(r['win_rate'] for r in all_results) / len(all_results)
    avg_profit_factor = sum(r['profit_factor'] for r in all_results) / len(all_results)
    
    print(f"📊 Total Simulated Trades: {total_trades}")
    print(f"💰 Combined P&L: ₹{total_pnl:+,.2f}")
    print(f"🎯 Average Win Rate: {avg_win_rate:.1f}%")
    print(f"⚖️ Average Profit Factor: {avg_profit_factor:.2f}")
    
    # Strategy evaluation
    print(f"\n💡 STRATEGY EVALUATION:")
    if avg_win_rate > 65 and avg_profit_factor > 1.5:
        print("🟢 EXCELLENT: Strategy shows strong potential")
        print("   ✅ High win rate and good profit factor")
        print("   ✅ Risk-reward ratio is favorable")
    elif avg_win_rate > 55 and avg_profit_factor > 1.2:
        print("🟡 GOOD: Strategy shows promise with optimization")
        print("   ⚠️ Consider fine-tuning parameters")
        print("   ⚠️ Test with longer time periods")
    elif avg_win_rate > 45:
        print("🟠 AVERAGE: Strategy needs significant improvements")
        print("   ❌ Low win rate or poor profit factor")
        print("   ❌ Requires parameter optimization")
    else:
        print("🔴 POOR: Strategy not recommended without major changes")
        print("   ❌ Very low win rate")
        print("   ❌ Negative expected value")
    
    print(f"\n📋 NEXT STEPS FOR REAL TRADING:")
    print("1. Set up FYERS API credentials")
    print("2. Run backtests with real historical data")
    print("3. Optimize parameters using walk-forward analysis")
    print("4. Start with paper trading")
    print("5. Begin with small position sizes")
    print("6. Monitor and adjust strategy performance")
    
    print(f"\n🎉 Demo completed successfully!")
    print("Thank you for exploring the Index Intraday Strategy!")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    demo_strategy_features()