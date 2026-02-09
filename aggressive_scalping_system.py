#!/usr/bin/env python3
"""
💥 SUPER AGGRESSIVE INTRADAY SCALPING 💥
================================================================================
🔥 GUARANTEED TRADES: Simple conditions, real scalps
⚡ 5-15 POINT TARGETS: Quick small profits
🎯 HIGH FREQUENCY: Every small move captured
📈 REAL MARKET BEHAVIOR: Based on actual NIFTY movements
================================================================================
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
import warnings
warnings.filterwarnings('ignore')

from fyers_client import FyersClient

class AggressiveScalpingSystem:
    """Super aggressive scalping - finds every possible trade"""
    
    def __init__(self):
        print("💥 SUPER AGGRESSIVE INTRADAY SCALPING 💥")
        print("=" * 60)
        print("🔥 GUARANTEED TRADES: Simple conditions")
        print("⚡ 5-15 POINT TARGETS: Quick profits")  
        print("🎯 HIGH FREQUENCY: Every move captured")
        print("📈 REAL BEHAVIOR: Actual NIFTY scalps")
        print("=" * 60)
        
        # Initialize Fyers
        try:
            self.fyers_client = FyersClient('fyers_config.json')
            print("✅ Connected to real Fyers account")
        except Exception as e:
            print(f"❌ Fyers error: {e}")
            return
            
        # Aggressive scalping parameters  
        self.capital = 100000
        self.quantity = 3  # Fixed quantity for simplicity
        self.commission = 20
        
        # Very tight scalping targets
        self.profit_target = 10  # Just 10 points profit
        self.stop_loss = 8       # 8 points stop
        self.max_hold_candles = 6  # Max 6 candles (30 minutes)
        
        # Results
        self.scalps = []
        self.daily_results = {}
        
    def run_aggressive_scalping(self, symbol: str = "NSE:NIFTY50-INDEX", days: int = 15):
        """Run super aggressive scalping"""
        
        print(f"\n💥 STARTING AGGRESSIVE SCALPING")
        print("=" * 35)
        print(f"📊 Symbol: {symbol}")
        print(f"📅 Period: {days} days")
        print(f"💰 Target: {self.profit_target} points")
        print(f"⛔ Stop: {self.stop_loss} points")
        print(f"📦 Quantity: {self.quantity} lots")
        
        # Get data
        df = self.get_scalping_data(symbol, days)
        if df is None or len(df) < 50:
            print("❌ Insufficient data")
            return
            
        # Add simple indicators
        df = self.add_simple_indicators(df)
        
        # Run aggressive scalping
        self.execute_aggressive_scalping(df)
        
        # Show results
        self.show_aggressive_results()
        
    def get_scalping_data(self, symbol: str, days: int):
        """Get real NIFTY data"""
        
        try:
            end_date = datetime.now() 
            start_date = end_date - timedelta(days=days)
            
            data_request = {
                "symbol": symbol,
                "resolution": "5",  # 5-minute data
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
                df['time_only'] = df['datetime'].dt.time
                df['date_only'] = df['datetime'].dt.date
                
                # Keep only market hours
                df = df[(df['time_only'] >= time(9, 15)) & 
                       (df['time_only'] <= time(15, 30))]
                
                print(f"✅ Real NIFTY data: {len(df):,} candles")
                print(f"📈 Price range: Rs.{df['low'].min():.0f} - Rs.{df['high'].max():.0f}")
                print(f"📅 Period: {df['date_only'].min()} to {df['date_only'].max()}")
                
                return df.reset_index(drop=True)
                
            else:
                print(f"❌ Data fetch failed")
                return None
                
        except Exception as e:
            print(f"❌ Data error: {e}")
            return None
    
    def add_simple_indicators(self, df):
        """Add very simple indicators for aggressive scalping"""
        
        print("🔧 Adding simple scalping indicators...")
        
        # Very simple moving average
        df['ma_fast'] = df['close'].rolling(3).mean()
        df['ma_slow'] = df['close'].rolling(8).mean()
        
        # Price change from previous candle
        df['price_change'] = df['close'].diff()
        df['up_move'] = df['price_change'] > 2  # 2+ points up
        df['down_move'] = df['price_change'] < -2  # 2+ points down
        
        # High/Low breaks
        df['high_5'] = df['high'].rolling(5).max()
        df['low_5'] = df['low'].rolling(5).min()
        df['breakout_high'] = df['close'] > df['high_5'].shift(1)
        df['breakdown_low'] = df['close'] < df['low_5'].shift(1)
        
        # Volume (simple)
        df['vol_above_avg'] = df['volume'] > df['volume'].rolling(10).mean()
        
        print("✅ Simple indicators ready")
        return df
        
    def execute_aggressive_scalping(self, df):
        """Execute super aggressive scalping strategy"""
        
        print(f"\n🔥 EXECUTING AGGRESSIVE SCALPING")
        print("=" * 35)
        
        scalp_id = 1
        
        # Go through each candle aggressively
        for i in range(10, len(df) - 5):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Only trade during active hours
            if not (time(9, 20) <= current['time_only'] <= time(15, 20)):
                continue
            
            # AGGRESSIVE LONG SIGNALS (very simple)
            long_signals = []
            
            # 1. Simple upward price movement
            if (current['close'] > prev['close'] and 
                current['price_change'] > 3):
                long_signals.append('price_momentum_up')
            
            # 2. MA crossover
            if (pd.notna(current['ma_fast']) and pd.notna(current['ma_slow']) and
                current['ma_fast'] > current['ma_slow'] and
                prev['ma_fast'] <= prev['ma_slow']):
                long_signals.append('ma_cross_up')
            
            # 3. Breakout above recent high
            if (current['breakout_high'] and current['vol_above_avg']):
                long_signals.append('breakout_up')
            
            # AGGRESSIVE SHORT SIGNALS
            short_signals = []
            
            # 1. Simple downward price movement  
            if (current['close'] < prev['close'] and
                current['price_change'] < -3):
                short_signals.append('price_momentum_down')
            
            # 2. MA crossover down
            if (pd.notna(current['ma_fast']) and pd.notna(current['ma_slow']) and
                current['ma_fast'] < current['ma_slow'] and
                prev['ma_fast'] >= prev['ma_slow']):
                short_signals.append('ma_cross_down')
            
            # 3. Breakdown below recent low
            if (current['breakdown_low'] and current['vol_above_avg']):
                short_signals.append('breakdown_down')
            
            # Execute trades
            for signal in long_signals:
                scalp = self.execute_single_scalp(df, i, 'BUY', signal, scalp_id)
                if scalp:
                    self.scalps.append(scalp)
                    scalp_id += 1
                    break  # One trade per candle max
                    
            for signal in short_signals:
                if len(long_signals) == 0:  # Only if no long signal
                    scalp = self.execute_single_scalp(df, i, 'SELL', signal, scalp_id)
                    if scalp:
                        self.scalps.append(scalp)
                        scalp_id += 1
                        break
        
        print(f"✅ Aggressive scalping complete: {len(self.scalps)} trades")
    
    def execute_single_scalp(self, df, entry_idx, side, signal, scalp_id):
        """Execute a single scalp trade"""
        
        entry_candle = df.iloc[entry_idx]
        entry_price = entry_candle['close']
        entry_time = entry_candle['datetime']
        
        # Set targets
        if side == 'BUY':
            target_price = entry_price + self.profit_target
            stop_price = entry_price - self.stop_loss
        else:
            target_price = entry_price - self.profit_target
            stop_price = entry_price + self.stop_loss
        
        # Look for exit in next candles
        exit_found = False
        exit_price = entry_price
        exit_time = entry_time
        exit_reason = 'TIME'
        
        for j in range(1, min(self.max_hold_candles + 1, len(df) - entry_idx)):
            candle = df.iloc[entry_idx + j]
            
            if side == 'BUY':
                # Check target hit
                if candle['high'] >= target_price:
                    exit_price = target_price
                    exit_time = candle['datetime']
                    exit_reason = 'TARGET'
                    exit_found = True
                    break
                # Check stop hit
                elif candle['low'] <= stop_price:
                    exit_price = stop_price  
                    exit_time = candle['datetime']
                    exit_reason = 'STOP'
                    exit_found = True
                    break
            else:  # SELL
                # Check target hit
                if candle['low'] <= target_price:
                    exit_price = target_price
                    exit_time = candle['datetime']
                    exit_reason = 'TARGET'
                    exit_found = True
                    break
                # Check stop hit
                elif candle['high'] >= stop_price:
                    exit_price = stop_price
                    exit_time = candle['datetime']
                    exit_reason = 'STOP'
                    exit_found = True
                    break
        
        # If no exit found, time-based exit
        if not exit_found and entry_idx + self.max_hold_candles < len(df):
            exit_candle = df.iloc[entry_idx + self.max_hold_candles]
            exit_price = exit_candle['close']
            exit_time = exit_candle['datetime']
            exit_reason = 'TIME'
        
        # Calculate P&L
        if side == 'BUY':
            points = exit_price - entry_price
        else:
            points = entry_price - exit_price
            
        gross_pnl = points * self.quantity
        net_pnl = gross_pnl - self.commission
        
        duration_minutes = (exit_time - entry_time).total_seconds() / 60
        
        result = 'WIN' if net_pnl > 0 else 'LOSS'
        
        return {
            'id': scalp_id,
            'date': entry_time.strftime('%m-%d'),
            'entry_time': entry_time.strftime('%H:%M'),
            'exit_time': exit_time.strftime('%H:%M'),
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'points': points,
            'pnl': net_pnl,
            'duration_min': duration_minutes,
            'signal': signal,
            'exit_reason': exit_reason,
            'result': result
        }
    
    def show_aggressive_results(self):
        """Show aggressive scalping results"""
        
        print(f"\n💥 AGGRESSIVE SCALPING RESULTS 💥")
        print("=" * 45)
        
        if not self.scalps:
            print("❌ No scalp trades found")
            return
        
        # Overall stats
        total_scalps = len(self.scalps)
        winners = len([s for s in self.scalps if s['pnl'] > 0])
        losers = total_scalps - winners
        win_rate = (winners / total_scalps) * 100 if total_scalps > 0 else 0
        
        total_pnl = sum(s['pnl'] for s in self.scalps)
        roi = (total_pnl / self.capital) * 100
        
        avg_win = np.mean([s['pnl'] for s in self.scalps if s['pnl'] > 0]) if winners > 0 else 0
        avg_loss = np.mean([s['pnl'] for s in self.scalps if s['pnl'] < 0]) if losers > 0 else 0
        avg_duration = np.mean([s['duration_min'] for s in self.scalps])
        
        print(f"📊 SCALPING PERFORMANCE:")
        print(f"   🔥 Total Scalps:           {total_scalps:5d}")
        print(f"   🏆 Win Rate:               {win_rate:5.1f}%") 
        print(f"   ✅ Winners:                {winners:5d}")
        print(f"   ❌ Losers:                 {losers:5d}")
        print(f"   💚 Avg Winner:             Rs.{avg_win:+4.0f}")
        print(f"   💔 Avg Loser:              Rs.{avg_loss:+4.0f}")
        print(f"   ⏱️ Avg Duration:           {avg_duration:5.1f} min")
        print(f"   💰 Total P&L:              Rs.{total_pnl:+6,.0f}")
        print(f"   📈 ROI:                    {roi:+5.1f}%")
        
        # Show first 10 trades as examples
        print(f"\n📋 SAMPLE SCALP TRADES:")
        print("-" * 60)
        print(f"{'ID':<3} {'Date':<5} {'In':<5} {'Out':<5} {'Side':<4} {'Entry':<6} {'Exit':<6} {'Pts':<4} {'P&L':<6} {'Result':<4}")
        print("-" * 60)
        
        for i, scalp in enumerate(self.scalps[:20]):  # Show first 20
            print(f"{scalp['id']:<3d} {scalp['date']:<5} {scalp['entry_time']:<5} "
                  f"{scalp['exit_time']:<5} {scalp['side']:<4} {scalp['entry_price']:<6.0f} "
                  f"{scalp['exit_price']:<6.0f} {scalp['points']:+4.0f} "
                  f"Rs.{scalp['pnl']:+4.0f} {scalp['result']:<4}")
        
        if len(self.scalps) > 20:
            print(f"   ... and {len(self.scalps)-20} more scalps")
        
        # Daily breakdown
        daily_pnl = {}
        for scalp in self.scalps:
            date = scalp['date']
            if date not in daily_pnl:
                daily_pnl[date] = {'count': 0, 'pnl': 0, 'wins': 0}
            daily_pnl[date]['count'] += 1
            daily_pnl[date]['pnl'] += scalp['pnl']
            if scalp['pnl'] > 0:
                daily_pnl[date]['wins'] += 1
        
        print(f"\n📅 DAILY SCALPING BREAKDOWN:")
        print("-" * 30)
        
        for date, stats in list(daily_pnl.items())[:10]:  # First 10 days
            day_win_rate = (stats['wins'] / stats['count']) * 100
            print(f"   {date}: {stats['count']:2d} scalps, {day_win_rate:4.0f}% win, Rs.{stats['pnl']:+5.0f}")
        
        # Signal analysis
        signal_stats = {}
        for scalp in self.scalps:
            signal = scalp['signal']
            if signal not in signal_stats:
                signal_stats[signal] = {'count': 0, 'wins': 0, 'pnl': 0}
            signal_stats[signal]['count'] += 1
            signal_stats[signal]['pnl'] += scalp['pnl']
            if scalp['pnl'] > 0:
                signal_stats[signal]['wins'] += 1
        
        print(f"\n🎯 SIGNAL TYPE PERFORMANCE:")
        print("-" * 30)
        
        for signal, stats in signal_stats.items():
            sig_win_rate = (stats['wins'] / stats['count']) * 100
            avg_pnl = stats['pnl'] / stats['count']
            print(f"   {signal}:")
            print(f"      {stats['count']:2d} trades, {sig_win_rate:4.0f}% win, Rs.{avg_pnl:+4.0f} avg")
        
        # Final assessment
        print(f"\n💡 SCALPING ASSESSMENT:")
        if roi > 10:
            print(f"   🚀 EXCELLENT: {roi:+.1f}% ROI - Aggressive scalping works!")
        elif roi > 2:
            print(f"   ✅ GOOD: {roi:+.1f}% ROI - Profitable scalping")
        elif roi > 0:
            print(f"   ⚠️ MARGINAL: {roi:+.1f}% ROI - Barely profitable")
        else:
            print(f"   ❌ LOSS: {roi:+.1f}% ROI - Strategy needs work")
        
        print(f"   ⚡ {total_scalps} intraday scalps executed")
        print(f"   🎯 {self.profit_target} point targets, {self.stop_loss} point stops")
        print(f"   📈 100% REAL NIFTY data from Fyers API")

if __name__ == "__main__":
    print("💥 Starting Super Aggressive Intraday Scalping...")
    
    try:
        scalper = AggressiveScalpingSystem()
        
        scalper.run_aggressive_scalping(
            symbol="NSE:NIFTY50-INDEX", 
            days=15
        )
        
        print(f"\n✅ AGGRESSIVE SCALPING COMPLETE")
        print(f"🔥 Real intraday scalping results shown")
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()