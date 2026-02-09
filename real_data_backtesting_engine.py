#!/usr/bin/env python3
"""
🔥 REAL DATA BACKTESTING ENGINE 🔥
================================================================================
Uses 100% AUTHENTIC historical data from YOUR Fyers API
Tests multiple strategies with REAL market movements
Prepares strategies for LIVE TRADING when market opens
================================================================================
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from fyers_client import FyersClient

@dataclass
class RealBacktestTrade:
    """Real backtest trade with authentic data"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    strategy: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    quantity: int
    points: float
    pnl: float
    result: str  # 'WIN' or 'LOSS'
    confidence: float
    method: str

class RealDataBacktestEngine:
    """Comprehensive backtesting engine using 100% real market data"""
    
    def __init__(self):
        print("🔥 REAL DATA BACKTESTING ENGINE 🔥")
        print("=" * 80)
        print("Uses 100% AUTHENTIC historical data from YOUR Fyers API")
        print("Tests multiple strategies with REAL market movements")
        print("Prepares strategies for LIVE TRADING when market opens")
        print("=" * 80)
        
        # Initialize Fyers client
        try:
            self.fyers_client = FyersClient('fyers_config.json')
            print("✅ Fyers client initialized")
        except Exception as e:
            print(f"❌ Fyers client error: {e}")
            return
            
        # Load configuration
        with open('fyers_config.json', 'r') as f:
            self.config = json.load(f)
            
        # Backtesting parameters
        self.initial_capital = 100000
        self.current_capital = self.initial_capital
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.commission = 20  # Rs. 20 per trade
        
        # Results tracking
        self.all_trades = []
        self.daily_pnl = {}
        self.strategy_performance = {}
        
        print(f"💰 Initial Capital: Rs.{self.initial_capital:,.0f}")
        print(f"🎯 Risk per trade: {self.risk_per_trade:.1%}")
        print(f"💸 Commission: Rs.{self.commission} per trade")
        
        # Check market status
        self.check_market_status()
        
    def check_market_status(self):
        """Check if market is open or closed"""
        now = datetime.now()
        current_time = now.time()
        
        # NSE market hours: 9:15 AM to 3:30 PM
        market_open = time(9, 15)
        market_close = time(15, 30)
        
        # Check if today is weekend
        is_weekend = now.weekday() >= 5  # Saturday=5, Sunday=6
        
        if is_weekend:
            print("📅 Today is weekend - Market CLOSED")
            print("🔍 Perfect time for BACKTESTING with real historical data")
        elif market_open <= current_time <= market_close:
            print("🔴 Market is currently OPEN")
            print("💡 Can do live analysis, but will focus on backtesting")
        else:
            print("📅 Market is currently CLOSED")
            print("🔍 Perfect time for BACKTESTING with real historical data")
            
        print("🎯 FOCUS: Backtesting with REAL historical data from Fyers")
        
    def get_real_historical_data(self, symbol: str, days: int = 90, timeframe: str = "5") -> pd.DataFrame:
        """Get 100% authentic historical data from Fyers API"""
        
        print(f"\n📊 FETCHING REAL HISTORICAL DATA")
        print("-" * 50)
        print(f"   🎯 Symbol: {symbol}")
        print(f"   📅 Period: Last {days} days")
        print(f"   ⏱️ Timeframe: {timeframe} minutes")
        print(f"   🔌 Source: YOUR Fyers API (100% authentic)")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data_request = {
                "symbol": symbol,
                "resolution": timeframe,
                "date_format": "1",
                "range_from": start_date.strftime('%Y-%m-%d'),
                "range_to": end_date.strftime('%Y-%m-%d'),
                "cont_flag": "1"
            }
            
            response = self.fyers_client.fyers.history(data_request)
            
            if response and response.get('s') == 'ok' and 'candles' in response:
                candles = response['candles']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Add technical indicators
                df = self.add_technical_indicators(df)
                
                print(f"   ✅ Successfully retrieved {len(df)} authentic candles")
                print(f"   📈 Price range: Rs.{df['low'].min():.2f} - Rs.{df['high'].max():.2f}")
                print(f"   📅 Data period: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
                print(f"   📊 Average volume: {df['volume'].mean():,.0f}")
                print(f"   💯 Data authenticity: GUARANTEED REAL MARKET DATA")
                
                return df
            else:
                print(f"   ❌ Failed to fetch data: {response}")
                return None
                
        except Exception as e:
            print(f"   ❌ Error fetching data: {e}")
            return None
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for strategy analysis"""
        
        # Moving averages
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility
        df['atr'] = self.calculate_atr(df)
        
        # Price levels
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def strategy_1_momentum_breakout(self, df: pd.DataFrame) -> List[RealBacktestTrade]:
        """Strategy 1: Momentum Breakout with Volume Confirmation"""
        
        print(f"\n⚡ STRATEGY 1: MOMENTUM BREAKOUT")
        print("-" * 40)
        
        trades = []
        
        for i in range(100, len(df) - 10):
            current = df.iloc[i]
            
            # Skip market hours outside 9:30 AM to 3:00 PM
            if not self.is_trading_time(current['datetime']):
                continue
                
            # Momentum breakout conditions
            if (pd.notna(current['ema_9']) and pd.notna(current['ema_21']) and 
                pd.notna(current['rsi']) and pd.notna(current['volume_ratio'])):
                
                # LONG signal conditions
                if (current['close'] > current['ema_9'] > current['ema_21'] and
                    current['close'] > current['high_20'] * 0.999 and  # Near 20-day high
                    current['rsi'] > 50 and current['rsi'] < 70 and
                    current['volume_ratio'] > 1.5):  # High volume
                    
                    # Execute trade
                    trade = self.execute_backtest_trade(
                        df, i, 'BUY', 'momentum_breakout', 
                        entry_price=current['close'],
                        stop_loss_pct=1.5,  # 1.5% stop loss
                        target_pct=3.0      # 3% target
                    )
                    if trade:
                        trades.append(trade)
                
                # SHORT signal conditions
                elif (current['close'] < current['ema_9'] < current['ema_21'] and
                      current['close'] < current['low_20'] * 1.001 and  # Near 20-day low
                      current['rsi'] < 50 and current['rsi'] > 30 and
                      current['volume_ratio'] > 1.5):  # High volume
                    
                    # Execute trade
                    trade = self.execute_backtest_trade(
                        df, i, 'SELL', 'momentum_breakout',
                        entry_price=current['close'],
                        stop_loss_pct=1.5,  # 1.5% stop loss
                        target_pct=3.0      # 3% target
                    )
                    if trade:
                        trades.append(trade)
        
        print(f"   ✅ Generated {len(trades)} momentum breakout trades")
        return trades
    
    def strategy_2_mean_reversion(self, df: pd.DataFrame) -> List[RealBacktestTrade]:
        """Strategy 2: Mean Reversion at Key Levels"""
        
        print(f"\n🎯 STRATEGY 2: MEAN REVERSION")
        print("-" * 40)
        
        trades = []
        
        for i in range(100, len(df) - 10):
            current = df.iloc[i]
            
            # Skip market hours outside 9:30 AM to 3:00 PM
            if not self.is_trading_time(current['datetime']):
                continue
                
            # Mean reversion conditions
            if (pd.notna(current['vwap']) and pd.notna(current['rsi']) and 
                pd.notna(current['atr'])):
                
                # LONG at oversold levels
                if (current['rsi'] < 30 and
                    current['close'] < current['vwap'] * 0.995 and  # Below VWAP
                    current['volume_ratio'] > 1.2):
                    
                    trade = self.execute_backtest_trade(
                        df, i, 'BUY', 'mean_reversion',
                        entry_price=current['close'],
                        stop_loss_pct=1.0,  # 1% stop loss
                        target_pct=2.0      # 2% target
                    )
                    if trade:
                        trades.append(trade)
                
                # SHORT at overbought levels
                elif (current['rsi'] > 70 and
                      current['close'] > current['vwap'] * 1.005 and  # Above VWAP
                      current['volume_ratio'] > 1.2):
                    
                    trade = self.execute_backtest_trade(
                        df, i, 'SELL', 'mean_reversion',
                        entry_price=current['close'],
                        stop_loss_pct=1.0,  # 1% stop loss
                        target_pct=2.0      # 2% target
                    )
                    if trade:
                        trades.append(trade)
        
        print(f"   ✅ Generated {len(trades)} mean reversion trades")
        return trades
    
    def strategy_3_scalping_5min(self, df: pd.DataFrame) -> List[RealBacktestTrade]:
        """Strategy 3: High-Frequency Scalping (5-min)"""
        
        print(f"\n⚡ STRATEGY 3: SCALPING (5-MIN)")
        print("-" * 40)
        
        trades = []
        
        for i in range(50, len(df) - 5):
            current = df.iloc[i]
            
            # Only during high-liquidity hours
            if not self.is_scalping_time(current['datetime']):
                continue
                
            # Quick scalping signals
            if (pd.notna(current['ema_9']) and pd.notna(current['volume_ratio'])):
                
                # LONG scalp
                if (current['close'] > current['ema_9'] and
                    current['close'] > current['open'] and  # Green candle
                    current['volume_ratio'] > 2.0):  # High volume spike
                    
                    trade = self.execute_backtest_trade(
                        df, i, 'BUY', 'scalping_5min',
                        entry_price=current['close'],
                        stop_loss_pct=0.5,  # 0.5% stop loss
                        target_pct=1.0      # 1% target
                    )
                    if trade:
                        trades.append(trade)
                
                # SHORT scalp
                elif (current['close'] < current['ema_9'] and
                      current['close'] < current['open'] and  # Red candle
                      current['volume_ratio'] > 2.0):  # High volume spike
                    
                    trade = self.execute_backtest_trade(
                        df, i, 'SELL', 'scalping_5min',
                        entry_price=current['close'],
                        stop_loss_pct=0.5,  # 0.5% stop loss
                        target_pct=1.0      # 1% target
                    )
                    if trade:
                        trades.append(trade)
        
        print(f"   ✅ Generated {len(trades)} scalping trades")
        return trades
    
    def is_trading_time(self, dt: datetime) -> bool:
        """Check if time is within regular trading hours"""
        t = dt.time()
        return time(9, 30) <= t <= time(15, 0)  # 9:30 AM to 3:00 PM
    
    def is_scalping_time(self, dt: datetime) -> bool:
        """Check if time is within high-liquidity scalping hours"""
        t = dt.time()
        return (time(9, 30) <= t <= time(11, 30) or  # Morning session
                time(13, 30) <= t <= time(15, 0))    # Afternoon session
    
    def execute_backtest_trade(self, df: pd.DataFrame, entry_idx: int, side: str, 
                              strategy: str, entry_price: float, 
                              stop_loss_pct: float, target_pct: float) -> Optional[RealBacktestTrade]:
        """Execute a complete backtest trade with authentic data"""
        
        entry_candle = df.iloc[entry_idx]
        
        # Calculate stop loss and target
        if side == 'BUY':
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            target = entry_price * (1 + target_pct / 100)
        else:  # SELL
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            target = entry_price * (1 - target_pct / 100)
        
        # Look for exit in future candles
        for i in range(entry_idx + 1, min(entry_idx + 50, len(df))):
            candle = df.iloc[i]
            
            if side == 'BUY':
                # Check target first
                if candle['high'] >= target:
                    return self.create_trade_result(
                        entry_candle, candle, 'BUY', strategy,
                        entry_price, target, 'WIN'
                    )
                # Check stop loss
                elif candle['low'] <= stop_loss:
                    return self.create_trade_result(
                        entry_candle, candle, 'BUY', strategy,
                        entry_price, stop_loss, 'LOSS'
                    )
            else:  # SELL
                # Check target first
                if candle['low'] <= target:
                    return self.create_trade_result(
                        entry_candle, candle, 'SELL', strategy,
                        entry_price, target, 'WIN'
                    )
                # Check stop loss
                elif candle['high'] >= stop_loss:
                    return self.create_trade_result(
                        entry_candle, candle, 'SELL', strategy,
                        entry_price, stop_loss, 'LOSS'
                    )
        
        # No exit found (should not happen with proper stops)
        return None
    
    def create_trade_result(self, entry_candle, exit_candle, side: str, strategy: str,
                           entry_price: float, exit_price: float, result: str) -> RealBacktestTrade:
        """Create trade result object"""
        
        # Calculate points and P&L
        if side == 'BUY':
            points = exit_price - entry_price
        else:  # SELL
            points = entry_price - exit_price
        
        # Position sizing based on risk
        risk_amount = self.current_capital * self.risk_per_trade
        if side == 'BUY':
            stop_distance = entry_price - (entry_price * 0.985)  # Approximate
        else:
            stop_distance = (entry_price * 1.015) - entry_price   # Approximate
        
        quantity = max(1, int(risk_amount / stop_distance)) if stop_distance > 0 else 1
        
        # Calculate P&L
        gross_pnl = points * quantity
        net_pnl = gross_pnl - self.commission
        
        # Update capital
        self.current_capital += net_pnl
        
        return RealBacktestTrade(
            entry_time=entry_candle['datetime'],
            exit_time=exit_candle['datetime'],
            symbol="NIFTY",
            strategy=strategy,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            points=points,
            pnl=net_pnl,
            result=result,
            confidence=0.75,  # Default confidence
            method="historical_backtest"
        )
    
    def run_comprehensive_backtest(self, symbol: str = "NSE:NIFTY50-INDEX", days: int = 90):
        """Run comprehensive backtesting across all strategies"""
        
        print(f"\n🔥 COMPREHENSIVE REAL DATA BACKTESTING")
        print("=" * 80)
        print(f"🎯 Symbol: {symbol}")
        print(f"📅 Period: Last {days} days")
        print(f"🔌 Data Source: YOUR Fyers API (100% authentic)")
        print(f"💰 Initial Capital: Rs.{self.initial_capital:,.0f}")
        
        # Get historical data
        df = self.get_real_historical_data(symbol, days, "5")
        if df is None:
            print("❌ Could not fetch historical data")
            return None
        
        print(f"\n🎯 TESTING MULTIPLE STRATEGIES ON REAL DATA")
        print("=" * 60)
        
        # Test all strategies
        strategy_1_trades = self.strategy_1_momentum_breakout(df)
        strategy_2_trades = self.strategy_2_mean_reversion(df)
        strategy_3_trades = self.strategy_3_scalping_5min(df)
        
        # Combine all trades
        all_trades = strategy_1_trades + strategy_2_trades + strategy_3_trades
        all_trades.sort(key=lambda x: x.entry_time)
        
        self.all_trades = all_trades
        
        # Generate comprehensive results
        results = self.generate_comprehensive_results(df, all_trades)
        
        return results
    
    def generate_comprehensive_results(self, df: pd.DataFrame, trades: List[RealBacktestTrade]):
        """Generate comprehensive backtesting results"""
        
        print(f"\n🔥💰 COMPREHENSIVE REAL DATA BACKTEST RESULTS 💰🔥")
        print("=" * 80)
        
        if not trades:
            print("💡 No trades generated - Market conditions may not be suitable")
            return
        
        # Calculate overall performance
        total_pnl = self.current_capital - self.initial_capital
        total_trades = len(trades)
        wins = len([t for t in trades if t.result == 'WIN'])
        losses = len([t for t in trades if t.result == 'LOSS'])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        roi = (total_pnl / self.initial_capital * 100)
        
        print(f"📊 DATA VERIFICATION:")
        print(f"   🔌 Source: YOUR Fyers API (Account: FAH92116)")
        print(f"   📅 Period: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
        print(f"   📈 Candles analyzed: {len(df):,} (5-minute NIFTY data)")
        print(f"   💯 Data authenticity: GUARANTEED REAL MARKET DATA")
        
        print(f"\n💰 OVERALL PERFORMANCE:")
        print(f"   💰 Starting Capital:     Rs.{self.initial_capital:10,.0f}")
        print(f"   🎯 Final Capital:        Rs.{self.current_capital:10,.0f}")
        print(f"   🚀 Total P&L:            Rs.{total_pnl:+9,.0f}")
        print(f"   📈 ROI:                  {roi:+8.1f}%")
        print(f"   ⚡ Total Trades:          {total_trades:10d}")
        print(f"   🏆 Win Rate:             {win_rate:9.1f}%")
        print(f"   ✅ Winning Trades:       {wins:10d}")
        print(f"   ❌ Losing Trades:        {losses:10d}")
        
        # Strategy-wise breakdown
        strategies = {}
        for trade in trades:
            if trade.strategy not in strategies:
                strategies[trade.strategy] = {'trades': 0, 'pnl': 0, 'wins': 0}
            strategies[trade.strategy]['trades'] += 1
            strategies[trade.strategy]['pnl'] += trade.pnl
            if trade.result == 'WIN':
                strategies[trade.strategy]['wins'] += 1
        
        print(f"\n🎯 STRATEGY PERFORMANCE BREAKDOWN:")
        for strategy, stats in strategies.items():
            strategy_win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            print(f"   {strategy.upper():20} {stats['trades']:3d} trades → Rs.{stats['pnl']:+8,.0f} (Win: {strategy_win_rate:.0f}%)")
        
        # Recent trade samples
        print(f"\n📋 RECENT TRADE SAMPLES (Real Market Data):")
        recent_trades = sorted(trades, key=lambda x: x.entry_time, reverse=True)[:10]
        for i, trade in enumerate(recent_trades):
            print(f"   {i+1:2d}. {trade.entry_time.strftime('%m-%d %H:%M')} {trade.strategy[:12]:12} {trade.side:4} Rs.{trade.entry_price:7.0f}→{trade.exit_price:7.0f} {trade.points:+4.0f}pts Rs.{trade.pnl:+6,.0f} {trade.result}")
        
        # Performance verdict
        print("\n" + "=" * 80)
        
        if roi > 20:
            print("🚀💰 EXCEPTIONAL PERFORMANCE: Ready for REAL MONEY!")
            print("   ✅ Strong returns with multiple strategies")
            print("   ✅ Proven with authentic market data")
            print("   🚀 READY FOR LIVE AUTOMATED TRADING!")
        elif roi > 10:
            print("🔥 EXCELLENT: Strong performance across strategies!")
        elif roi > 5:
            print("✅ GOOD: Solid performance with real data")
        elif roi > 0:
            print("📈 POSITIVE: Profitable with room for optimization")
        else:
            print("⚠️ NEEDS OPTIMIZATION: Strategies need refinement")
        
        print(f"\n🎯 LIVE TRADING READINESS:")
        print(f"   💯 Data: 100% authentic Fyers historical data")
        print(f"   🔧 Strategies: Multiple tested approaches")
        print(f"   🛡️ Risk: Conservative {self.risk_per_trade:.1%} per trade")
        print(f"   📊 Backtested: {len(df):,} candles of real market data")
        print(f"   🤖 Ready: For automated live trading when market opens")
        
        return {
            'total_pnl': total_pnl,
            'roi': roi,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'strategies': strategies,
            'data_period': f"{df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}",
            'candles_analyzed': len(df),
            'ready_for_live': roi > 5 and win_rate > 50
        }

if __name__ == "__main__":
    print("🔥 Starting Real Data Backtesting Engine...")
    
    try:
        # Initialize engine
        engine = RealDataBacktestEngine()
        
        # Run comprehensive backtest
        results = engine.run_comprehensive_backtest(
            symbol="NSE:NIFTY50-INDEX",
            days=60  # Test with 60 days of real data
        )
        
        if results and results.get('ready_for_live'):
            print(f"\n🎉 STRATEGIES VALIDATED WITH REAL DATA!")
            print(f"💰 Ready for live trading when market opens")
            print(f"🚀 All systems prepared for real money execution")
        else:
            print(f"\n📊 Strategies tested - Consider optimization")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()