#!/usr/bin/env python3
"""
⚡ PURE INTRADAY SCALPING SYSTEM ⚡
================================================================================
🎯 TRUE SCALPING: All trades close same day before 3:30 PM
⏰ INTRADAY ONLY: No overnight positions, no multi-day holds
💰 QUICK PROFITS: Target 20-50 points, 1-5 minute holds
🔥 HIGH FREQUENCY: Multiple trades per day, real market data
================================================================================
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from fyers_client import FyersClient

@dataclass
class ScalpTrade:
    """Individual scalp trade details"""
    trade_id: int
    entry_time: datetime
    exit_time: datetime  
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    points: float
    pnl: float
    duration_seconds: int
    scalp_signal: str

class IntradayScalpingSystem:
    """Pure intraday scalping system - NO overnight positions"""
    
    def __init__(self):
        print("⚡ PURE INTRADAY SCALPING SYSTEM ⚡")
        print("=" * 60)
        print("🎯 TRUE SCALPING: All trades close same day")
        print("⏰ INTRADAY ONLY: No overnight positions")
        print("💰 TARGET: 20-50 points, 1-5 minute holds")
        print("🔥 FREQUENCY: Multiple trades per day")
        print("=" * 60)
        
        # Initialize Fyers client
        try:
            self.fyers_client = FyersClient('fyers_config.json')
            print("✅ Connected to real Fyers account")
        except Exception as e:
            print(f"❌ Fyers connection error: {e}")
            return
            
        # Scalping parameters
        self.capital = 100000
        self.risk_per_trade = 0.005  # 0.5% risk per scalp
        self.commission = 20
        self.max_trades_per_day = 20
        self.min_points_target = 20
        self.max_points_target = 50
        self.max_hold_minutes = 15  # Maximum 15 minutes per trade
        
        # Market hours
        self.market_open = time(9, 15)
        self.market_close = time(15, 30)
        self.stop_new_trades = time(15, 0)  # Stop new trades at 3 PM
        
        # Tracking
        self.daily_trades = []
        self.daily_pnl = 0
        self.total_scalps = 0
        
    def run_intraday_scalping_backtest(self, symbol: str = "NSE:NIFTY50-INDEX", days: int = 30):
        """Run pure intraday scalping backtest"""
        
        print(f"\n🔥 STARTING INTRADAY SCALPING BACKTEST")
        print("=" * 50)
        print(f"📊 Symbol: {symbol}")
        print(f"📅 Period: {days} days")
        print(f"⏰ Trading Hours: 09:15 - 15:30")
        print(f"🎯 Max Hold Time: {self.max_hold_minutes} minutes")
        print(f"💰 Points Target: {self.min_points_target}-{self.max_points_target}")
        
        # Get 1-minute data for precise scalping
        df = self.get_intraday_data(symbol, days)
        if df is None:
            return None
            
        # Add scalping indicators
        df = self.add_scalping_indicators(df)
        
        # Run daily scalping sessions
        results = self.run_daily_scalping_sessions(df)
        
        # Generate scalping analysis
        self.generate_scalping_analysis(results)
        
        return results
    
    def get_intraday_data(self, symbol: str, days: int):
        """Get 1-minute intraday data"""
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get 1-minute data for precise scalping
            data_request = {
                "symbol": symbol,
                "resolution": "1",  # 1-minute candles
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
                df['time'] = df['datetime'].dt.time
                
                # Filter only market hours
                df = df[(df['time'] >= self.market_open) & 
                       (df['time'] <= self.market_close)]
                
                print(f"✅ Intraday data: {len(df):,} 1-minute candles")
                print(f"📅 Market hours only: {self.market_open} - {self.market_close}")
                
                return df
            else:
                print(f"❌ Data fetch failed: {response}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def add_scalping_indicators(self, df):
        """Add scalping-specific indicators"""
        
        print("🔧 Adding scalping indicators...")
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Fast EMAs for scalping
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_13'] = df['close'].ewm(span=13).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
        # Bollinger Bands (20 period)
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        
        # RSI (fast for scalping)
        df['rsi_fast'] = self.calculate_rsi(df['close'], 9)
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(10).mean()
        df['volume_spike'] = df['volume'] > df['volume_ma'] * 2
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['momentum'] = df['close'] - df['close'].shift(3)
        
        # Support/Resistance levels
        df['high_5'] = df['high'].rolling(5).max()
        df['low_5'] = df['low'].rolling(5).min()
        
        print("✅ Scalping indicators ready")
        return df
    
    def calculate_rsi(self, prices, period=9):
        """Fast RSI for scalping"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def run_daily_scalping_sessions(self, df):
        """Run scalping for each trading day"""
        
        print(f"\n⚡ RUNNING DAILY SCALPING SESSIONS")
        print("=" * 40)
        
        # Group by trading date
        df['date'] = df['datetime'].dt.date
        daily_results = {}
        
        for date, day_data in df.groupby('date'):
            day_data = day_data.reset_index(drop=True)
            
            if len(day_data) < 50:  # Skip days with insufficient data
                continue
                
            print(f"\n📅 {date} - Scalping Session")
            print("-" * 30)
            
            day_trades = self.scalp_trading_day(day_data, date)
            
            if day_trades:
                day_pnl = sum(trade.pnl for trade in day_trades)
                daily_results[date] = {
                    'trades': day_trades,
                    'count': len(day_trades),
                    'pnl': day_pnl,
                    'wins': len([t for t in day_trades if t.pnl > 0]),
                    'losses': len([t for t in day_trades if t.pnl < 0])
                }
                
                win_rate = (daily_results[date]['wins'] / len(day_trades)) * 100
                print(f"   📊 {len(day_trades)} scalps, {win_rate:.1f}% win rate, Rs.{day_pnl:+,.0f}")
            else:
                print(f"   ⚪ No scalps today")
                
        return daily_results
    
    def scalp_trading_day(self, day_data, date):
        """Execute scalping for one trading day"""
        
        trades = []
        trade_id = 1
        
        for i in range(50, len(day_data) - 10):
            current = day_data.iloc[i]
            
            # Stop new trades after 3 PM
            if current['time'] >= self.stop_new_trades:
                break
                
            # Check if we've hit daily trade limit
            if len(trades) >= self.max_trades_per_day:
                break
            
            # Look for scalping signals
            scalp_signal = self.detect_scalp_signal(day_data, i)
            
            if scalp_signal:
                trade = self.execute_scalp_trade(day_data, i, trade_id, scalp_signal)
                if trade:
                    trades.append(trade)
                    trade_id += 1
        
        # Force close any remaining positions at market close
        return trades
    
    def detect_scalp_signal(self, df, idx):
        """Detect scalping entry signals"""
        
        current = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        # Skip if indicators not available
        if pd.isna(current['ema_5']) or pd.isna(current['vwap']):
            return None
        
        # LONG SCALP SIGNALS
        
        # 1. VWAP Breakout Long
        if (current['close'] > current['vwap'] and 
            prev['close'] <= prev['vwap'] and
            current['volume_spike'] and
            current['rsi_fast'] > 50 and
            current['close'] > current['ema_5']):
            return "vwap_breakout_long"
        
        # 2. EMA Crossover Long
        if (current['ema_5'] > current['ema_13'] and
            prev['ema_5'] <= prev['ema_13'] and
            current['volume'] > current['volume_ma'] * 1.5 and
            current['close'] > current['open']):
            return "ema_cross_long"
        
        # 3. Bollinger Band Bounce Long
        if (prev['close'] <= prev['bb_lower'] and
            current['close'] > current['bb_lower'] and
            current['rsi_fast'] < 30 and
            current['volume_spike']):
            return "bb_bounce_long"
        
        # SHORT SCALP SIGNALS
        
        # 4. VWAP Breakdown Short
        if (current['close'] < current['vwap'] and
            prev['close'] >= prev['vwap'] and
            current['volume_spike'] and
            current['rsi_fast'] < 50 and
            current['close'] < current['ema_5']):
            return "vwap_breakdown_short"
        
        # 5. EMA Crossover Short
        if (current['ema_5'] < current['ema_13'] and
            prev['ema_5'] >= prev['ema_13'] and
            current['volume'] > current['volume_ma'] * 1.5 and
            current['close'] < current['open']):
            return "ema_cross_short"
        
        # 6. Bollinger Band Rejection Short
        if (prev['close'] >= prev['bb_upper'] and
            current['close'] < current['bb_upper'] and
            current['rsi_fast'] > 70 and
            current['volume_spike']):
            return "bb_rejection_short"
        
        return None
    
    def execute_scalp_trade(self, df, entry_idx, trade_id, signal):
        """Execute individual scalp trade"""
        
        entry_candle = df.iloc[entry_idx]
        entry_price = entry_candle['close']
        entry_time = entry_candle['datetime']
        
        # Determine side
        side = 'BUY' if 'long' in signal else 'SELL'
        
        # Set scalping targets (tight stops, quick profits)
        if side == 'BUY':
            stop_loss = entry_price - 15  # 15 points stop
            target_1 = entry_price + 25   # 25 points target
            target_2 = entry_price + 40   # 40 points extended
        else:
            stop_loss = entry_price + 15  # 15 points stop
            target_1 = entry_price - 25   # 25 points target
            target_2 = entry_price - 40   # 40 points extended
        
        # Position size for scalping
        risk_amount = self.capital * self.risk_per_trade
        quantity = max(1, int(risk_amount / 15))  # Based on 15 point stop
        
        # Look for exit within same day
        max_exit_idx = min(entry_idx + self.max_hold_minutes, len(df) - 1)
        
        for i in range(entry_idx + 1, max_exit_idx + 1):
            candle = df.iloc[i]
            
            # Force exit before market close
            if candle['time'] >= time(15, 25):
                return self.create_scalp_trade(
                    trade_id, entry_time, candle['datetime'], side,
                    entry_price, candle['close'], quantity, signal
                )
            
            # Check exit conditions
            if side == 'BUY':
                if candle['high'] >= target_1:  # Hit target
                    return self.create_scalp_trade(
                        trade_id, entry_time, candle['datetime'], side,
                        entry_price, target_1, quantity, signal
                    )
                elif candle['low'] <= stop_loss:  # Hit stop
                    return self.create_scalp_trade(
                        trade_id, entry_time, candle['datetime'], side,
                        entry_price, stop_loss, quantity, signal
                    )
            else:  # SELL
                if candle['low'] <= target_1:  # Hit target
                    return self.create_scalp_trade(
                        trade_id, entry_time, candle['datetime'], side,
                        entry_price, target_1, quantity, signal
                    )
                elif candle['high'] >= stop_loss:  # Hit stop
                    return self.create_scalp_trade(
                        trade_id, entry_time, candle['datetime'], side,
                        entry_price, stop_loss, quantity, signal
                    )
        
        # Time-based exit (max hold time reached)
        exit_candle = df.iloc[max_exit_idx]
        return self.create_scalp_trade(
            trade_id, entry_time, exit_candle['datetime'], side,
            entry_price, exit_candle['close'], quantity, signal
        )
    
    def create_scalp_trade(self, trade_id, entry_time, exit_time, side, 
                          entry_price, exit_price, quantity, signal):
        """Create scalp trade record"""
        
        # Calculate P&L
        if side == 'BUY':
            points = exit_price - entry_price
        else:
            points = entry_price - exit_price
            
        gross_pnl = points * quantity
        net_pnl = gross_pnl - self.commission
        
        # Duration in seconds
        duration = (exit_time - entry_time).total_seconds()
        
        return ScalpTrade(
            trade_id=trade_id,
            entry_time=entry_time,
            exit_time=exit_time,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            points=points,
            pnl=net_pnl,
            duration_seconds=int(duration),
            scalp_signal=signal
        )
    
    def generate_scalping_analysis(self, daily_results):
        """Generate comprehensive scalping analysis"""
        
        print(f"\n⚡ INTRADAY SCALPING ANALYSIS REPORT ⚡")
        print("=" * 60)
        
        if not daily_results:
            print("❌ No scalp trades executed")
            return
        
        # Aggregate statistics
        all_trades = []
        total_pnl = 0
        total_days = len(daily_results)
        
        for date, day_result in daily_results.items():
            all_trades.extend(day_result['trades'])
            total_pnl += day_result['pnl']
        
        if not all_trades:
            print("❌ No completed trades")
            return
        
        # Performance metrics
        total_scalps = len(all_trades)
        winning_scalps = len([t for t in all_trades if t.pnl > 0])
        losing_scalps = len([t for t in all_trades if t.pnl < 0])
        win_rate = (winning_scalps / total_scalps) * 100
        
        avg_win = np.mean([t.pnl for t in all_trades if t.pnl > 0]) if winning_scalps > 0 else 0
        avg_loss = np.mean([t.pnl for t in all_trades if t.pnl < 0]) if losing_scalps > 0 else 0
        
        roi = (total_pnl / self.capital) * 100
        avg_duration = np.mean([t.duration_seconds for t in all_trades]) / 60  # minutes
        
        # Daily averages
        avg_trades_per_day = total_scalps / total_days
        avg_pnl_per_day = total_pnl / total_days
        
        print(f"📊 SCALPING PERFORMANCE SUMMARY:")
        print(f"   ⚡ Total Scalps:          {total_scalps:8d}")
        print(f"   📅 Trading Days:          {total_days:8d}")
        print(f"   📈 Scalps/Day:            {avg_trades_per_day:8.1f}")
        print(f"   🏆 Win Rate:              {win_rate:7.1f}%")
        print(f"   ✅ Winners:               {winning_scalps:8d}")
        print(f"   ❌ Losers:                {losing_scalps:8d}")
        print(f"   💚 Avg Win:               Rs.{avg_win:+6.0f}")
        print(f"   💔 Avg Loss:              Rs.{avg_loss:+6.0f}")
        print(f"   ⏱️ Avg Duration:          {avg_duration:6.1f} min")
        print(f"   💰 Total P&L:             Rs.{total_pnl:+8,.0f}")
        print(f"   📊 ROI:                   {roi:+7.1f}%")
        print(f"   📈 P&L/Day:               Rs.{avg_pnl_per_day:+6.0f}")
        
        print(f"\n📋 DAILY SCALPING BREAKDOWN:")
        print("-" * 50)
        print(f"{'Date':<12} {'Scalps':<7} {'Win%':<6} {'P&L':<8} {'Best':<6} {'Worst':<6}")
        print("-" * 50)
        
        for date, day_result in daily_results.items():
            trades = day_result['trades']
            win_rate_day = (day_result['wins'] / len(trades)) * 100 if trades else 0
            best_trade = max([t.pnl for t in trades]) if trades else 0
            worst_trade = min([t.pnl for t in trades]) if trades else 0
            
            print(f"{date} {len(trades):>6d} {win_rate_day:>5.1f}% "
                  f"Rs.{day_result['pnl']:+6.0f} {best_trade:+6.0f} {worst_trade:+6.0f}")
        
        # Signal analysis
        self.analyze_scalp_signals(all_trades)
        
        print(f"\n🎯 SCALPING INSIGHTS:")
        print(f"   ⚡ Pure intraday: All positions closed same day")
        print(f"   🕒 No overnight risk: Zero gap exposure")
        print(f"   💨 Quick trades: Average {avg_duration:.1f} minutes")
        print(f"   🔥 High frequency: {avg_trades_per_day:.1f} scalps per day")
        
        if roi > 0:
            print(f"   ✅ PROFITABLE SCALPING SYSTEM: {roi:+.1f}% ROI")
        else:
            print(f"   ⚠️ NEEDS OPTIMIZATION: {roi:+.1f}% ROI")
    
    def analyze_scalp_signals(self, trades):
        """Analyze performance by scalp signal type"""
        
        print(f"\n🎯 SCALP SIGNAL PERFORMANCE:")
        print("-" * 40)
        
        signal_stats = {}
        
        for trade in trades:
            signal = trade.scalp_signal
            if signal not in signal_stats:
                signal_stats[signal] = {'count': 0, 'wins': 0, 'pnl': 0}
            
            signal_stats[signal]['count'] += 1
            signal_stats[signal]['pnl'] += trade.pnl
            if trade.pnl > 0:
                signal_stats[signal]['wins'] += 1
        
        for signal, stats in signal_stats.items():
            win_rate = (stats['wins'] / stats['count']) * 100
            avg_pnl = stats['pnl'] / stats['count']
            
            print(f"   {signal}:")
            print(f"      {stats['count']:2d} trades, {win_rate:5.1f}% win, "
                  f"Rs.{avg_pnl:+5.0f} avg")

if __name__ == "__main__":
    print("⚡ Starting Pure Intraday Scalping System...")
    
    try:
        scalper = IntradayScalpingSystem()
        
        # Run intraday scalping backtest
        results = scalper.run_intraday_scalping_backtest(
            symbol="NSE:NIFTY50-INDEX",
            days=30
        )
        
        if results:
            print(f"\n✅ INTRADAY SCALPING BACKTEST COMPLETE")
            print(f"⚡ Pure scalping with same-day exits only")
            print(f"💰 All positions closed before 3:30 PM daily")
        else:
            print(f"\n❌ Scalping backtest failed")
            
    except Exception as e:
        print(f"❌ Scalping system error: {e}")
        import traceback
        traceback.print_exc()