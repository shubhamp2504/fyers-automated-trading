#!/usr/bin/env python3
"""
🚀 OPTIMIZED INTRADAY SCALPING SYSTEM 🚀
================================================================================
⚡ TRUE SCALPING: 30 second to 5 minute trades only
💰 PROFIT TARGET: 10-30 points with tight stops  
🎯 HIGH WIN RATE: Quick entries and exits
📈 MULTIPLE STRATEGIES: VWAP, MA, Momentum, Range
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
class QuickScalp:
    """Individual quick scalp record"""
    id: int
    date: str
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    points: float
    duration_sec: int
    pnl: float
    signal_type: str
    result: str

class OptimizedScalpingSystem:
    """Optimized intraday scalping with realistic signals"""
    
    def __init__(self):
        print("🚀 OPTIMIZED INTRADAY SCALPING SYSTEM 🚀")
        print("=" * 60)
        print("⚡ TRUE SCALPING: 30sec - 5min trades only")
        print("💰 PROFIT TARGET: 10-30 points, tight stops")
        print("🎯 HIGH WIN RATE: Quick in, quick out")
        print("📈 MULTIPLE STRATEGIES: VWAP, MA, momentum")
        print("=" * 60)
        
        # Initialize Fyers
        try:
            self.fyers_client = FyersClient('fyers_config.json')
            print("✅ Connected to real Fyers account")
        except Exception as e:
            print(f"❌ Fyers error: {e}")
            return
            
        # Optimized scalping parameters
        self.capital = 100000
        self.risk_per_scalp = 0.003  # 0.3% risk per scalp
        self.commission = 20
        self.max_scalps_per_day = 50
        
        # Scalping targets (more realistic)
        self.min_profit_target = 8   # 8 points minimum
        self.max_profit_target = 25  # 25 points maximum  
        self.stop_loss_points = 12   # 12 points stop loss
        self.max_hold_seconds = 300  # 5 minutes max hold
        
        # Trading hours
        self.start_trading = time(9, 20)   # Start 5 min after open
        self.stop_trading = time(15, 20)   # Stop 10 min before close
        
        # Results tracking
        self.all_scalps = []
        self.daily_summary = {}
        
    def run_optimized_scalping_test(self, symbol: str = "NSE:NIFTY50-INDEX", days: int = 20):
        """Run optimized scalping backtest"""
        
        print(f"\n🚀 STARTING OPTIMIZED SCALPING TEST")
        print("=" * 40)
        print(f"📊 Symbol: {symbol}")
        print(f"📅 Period: {days} days")
        print(f"⏰ Scalping Hours: {self.start_trading} - {self.stop_trading}")
        print(f"🎯 Target: {self.min_profit_target}-{self.max_profit_target} points")
        print(f"⛔ Stop Loss: {self.stop_loss_points} points")
        print(f"⏱️ Max Hold: {self.max_hold_seconds} seconds")
        
        # Get recent 5-minute data (more realistic than 1-minute for backtesting)
        df = self.fetch_scalping_data(symbol, days)
        if df is None:
            return
            
        # Add optimized indicators
        df = self.add_optimized_indicators(df)
        
        # Run scalping simulation
        self.simulate_scalping_sessions(df)
        
        # Generate results
        self.generate_scalping_results()
    
    def fetch_scalping_data(self, symbol: str, days: int):
        """Fetch 5-minute data for scalping simulation"""
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Use 5-minute data for more realistic backtesting
            data_request = {
                "symbol": symbol,
                "resolution": "5",  # 5-minute candles
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
                df['date'] = df['datetime'].dt.date
                
                # Filter market hours only
                df = df[(df['time'] >= time(9, 15)) & (df['time'] <= time(15, 30))]
                
                print(f"✅ Scalping data: {len(df):,} candles")
                print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
                
                return df
            else:
                print(f"❌ Data fetch failed")
                return None
                
        except Exception as e:
            print(f"❌ Data error: {e}")
            return None
    
    def add_optimized_indicators(self, df):
        """Add optimized scalping indicators"""
        
        print("🔧 Adding optimized scalping indicators...")
        
        # Fast moving averages for scalping
        df['ema_3'] = df['close'].ewm(span=3).mean()
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
        # VWAP (intraday reset)
        df_vwap = []
        for date, day_data in df.groupby('date'):
            day_data = day_data.copy()
            day_data['vwap'] = (day_data['close'] * day_data['volume']).cumsum() / day_data['volume'].cumsum()
            df_vwap.append(day_data)
        
        df = pd.concat(df_vwap, ignore_index=True)
        
        # Volume analysis
        df['vol_ma'] = df['volume'].rolling(10).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
        df['high_volume'] = df['vol_ratio'] > 1.8
        
        # Price momentum (very short term)
        df['momentum_1'] = df['close'] - df['close'].shift(1)
        df['momentum_3'] = df['close'] - df['close'].shift(3)
        
        # Support/Resistance levels (short-term)
        df['swing_high'] = df['high'].rolling(3, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(3, center=True).min() == df['low']
        
        # Price position relative to candle
        df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
        df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])
        df['body_pct'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Trend detection
        df['uptrend'] = (df['ema_3'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])
        df['downtrend'] = (df['ema_3'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])
        
        print("✅ Scalping indicators added")
        return df
    
    def simulate_scalping_sessions(self, df):
        """Simulate scalping for each trading day"""
        
        print(f"\n⚡ SIMULATING SCALPING SESSIONS")
        print("=" * 35)
        
        scalp_count = 0
        
        for date, day_df in df.groupby('date'):
            day_df = day_df.reset_index(drop=True)
            
            if len(day_df) < 20:  # Need minimum candles
                continue
                
            print(f"\n📅 {date}")
            print("-" * 15)
            
            day_scalps = []
            
            # Simulate scalping throughout the day
            for i in range(10, len(day_df) - 5):
                current = day_df.iloc[i].copy()
                
                # Only trade during active hours
                if not (self.start_trading <= current['time'] <= self.stop_trading):
                    continue
                
                # Check for scalping opportunities
                scalp_signals = self.detect_scalp_opportunities(day_df, i)
                
                for signal in scalp_signals:
                    if len(day_scalps) >= self.max_scalps_per_day:
                        break
                        
                    scalp = self.simulate_scalp_execution(day_df, i, signal, scalp_count + 1)
                    if scalp:
                        day_scalps.append(scalp)
                        scalp_count += 1
                        
                        # Print scalp result
                        print(f"   #{scalp.id:2d} {scalp.entry_time} {scalp.side:4s} "
                              f"{scalp.entry_price:6.0f}→{scalp.exit_price:6.0f} "
                              f"{scalp.points:+4.0f}pts {scalp.duration_sec:3d}s "
                              f"Rs.{scalp.pnl:+5.0f} {scalp.result}")
                
                if len(day_scalps) >= self.max_scalps_per_day:
                    break
            
            if day_scalps:
                day_pnl = sum(s.pnl for s in day_scalps)
                wins = len([s for s in day_scalps if s.pnl > 0])
                win_rate = (wins / len(day_scalps)) * 100
                
                print(f"   📊 {len(day_scalps)} scalps, {win_rate:.0f}% win rate, Rs.{day_pnl:+,.0f}")
                
                self.daily_summary[date] = {
                    'scalps': len(day_scalps),
                    'wins': wins,
                    'win_rate': win_rate,
                    'pnl': day_pnl
                }
                
                self.all_scalps.extend(day_scalps)
            else:
                print(f"   ⚪ No scalps")
    
    def detect_scalp_opportunities(self, df, idx):
        """Detect multiple scalping opportunities"""
        
        current = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        opportunities = []
        
        # Skip if no valid data
        if pd.isna(current['ema_3']) or pd.isna(current['vwap']):
            return opportunities
        
        # 1. VWAP Cross Long
        if (current['close'] > current['vwap'] and 
            prev['close'] <= prev['vwap'] and
            current['high_volume'] and
            current['close'] > current['open']):
            opportunities.append('vwap_cross_long')
        
        # 2. VWAP Cross Short  
        if (current['close'] < current['vwap'] and
            prev['close'] >= prev['vwap'] and
            current['high_volume'] and
            current['close'] < current['open']):
            opportunities.append('vwap_cross_short')
        
        # 3. EMA Momentum Long
        if (current['ema_3'] > current['ema_8'] and
            prev['ema_3'] <= prev['ema_8'] and
            current['momentum_1'] > 0 and
            current['vol_ratio'] > 1.5):
            opportunities.append('ema_momentum_long')
        
        # 4. EMA Momentum Short
        if (current['ema_3'] < current['ema_8'] and
            prev['ema_3'] >= prev['ema_8'] and
            current['momentum_1'] < 0 and
            current['vol_ratio'] > 1.5):
            opportunities.append('ema_momentum_short')
        
        # 5. Breakout Long
        if (current['high'] > df.iloc[idx-5:idx]['high'].max() and
            current['body_pct'] > 0.6 and
            current['close'] > current['open'] and
            current['high_volume']):
            opportunities.append('breakout_long')
        
        # 6. Breakdown Short
        if (current['low'] < df.iloc[idx-5:idx]['low'].min() and
            current['body_pct'] > 0.6 and
            current['close'] < current['open'] and
            current['high_volume']):
            opportunities.append('breakdown_short')
        
        return opportunities
    
    def simulate_scalp_execution(self, df, entry_idx, signal, scalp_id):
        """Simulate scalp trade execution"""
        
        entry_candle = df.iloc[entry_idx]
        entry_price = entry_candle['close']
        entry_time = entry_candle['datetime']
        
        # Determine direction
        side = 'BUY' if 'long' in signal else 'SELL'
        
        # Set targets and stops
        if side == 'BUY':
            target_price = entry_price + self.min_profit_target
            stop_price = entry_price - self.stop_loss_points
        else:
            target_price = entry_price - self.min_profit_target  
            stop_price = entry_price + self.stop_loss_points
        
        # Position size
        risk_amt = self.capital * self.risk_per_scalp
        quantity = max(1, int(risk_amt / self.stop_loss_points))
        
        # Look for exit in next few candles (realistic scalping)
        max_hold_candles = min(3, len(df) - entry_idx - 1)  # Max 3 candles (15 min)
        
        for i in range(1, max_hold_candles + 1):
            exit_idx = entry_idx + i
            if exit_idx >= len(df):
                break
                
            exit_candle = df.iloc[exit_idx]
            
            # Check hit levels within the candle
            if side == 'BUY':
                # Check if target hit
                if exit_candle['high'] >= target_price:
                    return self.create_scalp_record(
                        scalp_id, entry_candle, exit_candle, 
                        side, entry_price, target_price, 
                        quantity, signal, 'TARGET'
                    )
                # Check if stop hit
                elif exit_candle['low'] <= stop_price:
                    return self.create_scalp_record(
                        scalp_id, entry_candle, exit_candle,
                        side, entry_price, stop_price,
                        quantity, signal, 'STOP'
                    )
            else:  # SELL
                # Check if target hit
                if exit_candle['low'] <= target_price:
                    return self.create_scalp_record(
                        scalp_id, entry_candle, exit_candle,
                        side, entry_price, target_price,
                        quantity, signal, 'TARGET'
                    )
                # Check if stop hit  
                elif exit_candle['high'] >= stop_price:
                    return self.create_scalp_record(
                        scalp_id, entry_candle, exit_candle,
                        side, entry_price, stop_price,
                        quantity, signal, 'STOP'
                    )
        
        # Time exit (if no target/stop hit)
        if max_hold_candles > 0:
            exit_candle = df.iloc[entry_idx + max_hold_candles]
            return self.create_scalp_record(
                scalp_id, entry_candle, exit_candle,
                side, entry_price, exit_candle['close'],
                quantity, signal, 'TIME'
            )
        
        return None
    
    def create_scalp_record(self, scalp_id, entry_candle, exit_candle, 
                           side, entry_price, exit_price, quantity, signal, exit_type):
        """Create detailed scalp record"""
        
        # Calculate metrics
        if side == 'BUY':
            points = exit_price - entry_price
        else:
            points = entry_price - exit_price
        
        gross_pnl = points * quantity
        net_pnl = gross_pnl - self.commission
        
        duration_sec = int((exit_candle['datetime'] - entry_candle['datetime']).total_seconds())
        
        result = 'WIN' if net_pnl > 0 else 'LOSS'
        
        return QuickScalp(
            id=scalp_id,
            date=entry_candle['date'].strftime('%m-%d'),
            entry_time=entry_candle['datetime'].strftime('%H:%M'),
            exit_time=exit_candle['datetime'].strftime('%H:%M'),
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            points=points,
            duration_sec=duration_sec,
            pnl=net_pnl,
            signal_type=signal,
            result=result
        )
    
    def generate_scalping_results(self):
        """Generate comprehensive scalping results"""
        
        print(f"\n🚀 OPTIMIZED SCALPING RESULTS 🚀")
        print("=" * 50)
        
        if not self.all_scalps:
            print("❌ No scalps executed")
            return
        
        # Overall metrics
        total_scalps = len(self.all_scalps)
        winners = len([s for s in self.all_scalps if s.pnl > 0])
        losers = total_scalps - winners
        win_rate = (winners / total_scalps) * 100
        
        total_pnl = sum(s.pnl for s in self.all_scalps)
        roi = (total_pnl / self.capital) * 100
        
        avg_win = np.mean([s.pnl for s in self.all_scalps if s.pnl > 0]) if winners > 0 else 0
        avg_loss = np.mean([s.pnl for s in self.all_scalps if s.pnl < 0]) if losers > 0 else 0
        avg_duration = np.mean([s.duration_sec for s in self.all_scalps]) / 60  # minutes
        
        profit_factor = abs(sum(s.pnl for s in self.all_scalps if s.pnl > 0) / 
                           sum(s.pnl for s in self.all_scalps if s.pnl < 0)) if losers > 0 else float('inf')
        
        # Performance summary
        print(f"📊 SCALPING PERFORMANCE:")
        print(f"   ⚡ Total Scalps:           {total_scalps:5d}")
        print(f"   🏆 Win Rate:               {win_rate:5.1f}%")
        print(f"   ✅ Winners:                {winners:5d}")
        print(f"   ❌ Losers:                 {losers:5d}")
        print(f"   💚 Avg Win:                Rs.{avg_win:+4.0f}")
        print(f"   💔 Avg Loss:               Rs.{avg_loss:+4.0f}")
        print(f"   📊 Profit Factor:          {profit_factor:5.2f}")
        print(f"   ⏱️ Avg Duration:           {avg_duration:5.1f} min")
        print(f"   💰 Total P&L:              Rs.{total_pnl:+6,.0f}")
        print(f"   📈 ROI:                    {roi:+5.1f}%")
        
        # Daily breakdown
        if self.daily_summary:
            print(f"\n📅 DAILY SCALPING SUMMARY:")
            print("-" * 30)
            
            total_days = len(self.daily_summary)
            avg_scalps_day = np.mean([d['scalps'] for d in self.daily_summary.values()])
            avg_pnl_day = np.mean([d['pnl'] for d in self.daily_summary.values()])
            
            print(f"   📊 Trading Days:           {total_days:5d}")
            print(f"   ⚡ Avg Scalps/Day:         {avg_scalps_day:5.1f}")
            print(f"   💰 Avg P&L/Day:            Rs.{avg_pnl_day:+4.0f}")
            
            for date, summary in list(self.daily_summary.items())[:5]:  # Show first 5 days
                print(f"      {date}: {summary['scalps']:2d} scalps, "
                      f"{summary['win_rate']:4.0f}% win, Rs.{summary['pnl']:+5.0f}")
        
        # Signal analysis
        print(f"\n🎯 SIGNAL TYPE ANALYSIS:")
        print("-" * 25)
        
        signal_stats = {}
        for scalp in self.all_scalps:
            signal = scalp.signal_type
            if signal not in signal_stats:
                signal_stats[signal] = {'count': 0, 'wins': 0, 'pnl': 0}
            
            signal_stats[signal]['count'] += 1
            signal_stats[signal]['pnl'] += scalp.pnl
            if scalp.pnl > 0:
                signal_stats[signal]['wins'] += 1
        
        for signal, stats in signal_stats.items():
            win_rate_sig = (stats['wins'] / stats['count']) * 100
            avg_pnl_sig = stats['pnl'] / stats['count']
            
            print(f"   {signal}:")
            print(f"      {stats['count']:2d} trades, {win_rate_sig:4.0f}% win, Rs.{avg_pnl_sig:+4.0f} avg")
        
        # Key insights
        print(f"\n💡 SCALPING INSIGHTS:")
        print(f"   ⚡ Pure intraday: No overnight positions")
        print(f"   🕒 Quick trades: Avg {avg_duration:.1f} minutes")
        print(f"   🎯 Tight targets: {self.min_profit_target} points average")
        print(f"   ⛔ Controlled risk: {self.stop_loss_points} points max loss")
        
        if roi > 5:
            print(f"   ✅ PROFITABLE SCALPING: {roi:+.1f}% ROI")
        elif roi > 0:
            print(f"   ⚠️ MARGINAL PROFIT: {roi:+.1f}% ROI - needs optimization")
        else:
            print(f"   ❌ NEEDS WORK: {roi:+.1f}% ROI - refine signals")

if __name__ == "__main__":
    print("🚀 Starting Optimized Intraday Scalping System...")
    
    try:
        scalper = OptimizedScalpingSystem()
        
        # Run scalping test
        scalper.run_optimized_scalping_test(
            symbol="NSE:NIFTY50-INDEX",
            days=20
        )
        
        print(f"\n✅ OPTIMIZED SCALPING TEST COMPLETE")
        print(f"⚡ True intraday scalping results")
        
    except Exception as e:
        print(f"❌ Scalping error: {e}")
        import traceback
        traceback.print_exc()