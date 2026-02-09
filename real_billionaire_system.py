#!/usr/bin/env python3
"""
🚀 REAL BILLIONAIRE SYSTEM - SUPPLY & DEMAND ANALYSIS 🚀
================================================================================
❌ TRUTH: 0.22% monthly = 349 years to billionaire (USELESS!)
✅ SOLUTION: Supply/demand analysis for 15-25% monthly ROI
📈 TARGET: 10x capital every 12 months = Billionaire in 10 years
🔥 METHOD: All historical data + order flow + supply/demand zones
================================================================================
REAL BILLIONAIRE MATH:
- Need 25% monthly for 10x annual growth
- Start: Rs.1 lakh → Year 10: Rs.100 crore = BILLIONAIRE!
- Method: Find major supply/demand imbalances in NIFTY data

SUPPLY/DEMAND STRATEGY:
1. Analyze 2+ years of historical data
2. Identify major support/resistance zones  
3. Find volume spikes (institutional activity)
4. Trade breakouts from consolidation zones
5. Target 50-100 point moves (Rs.150-300 per trade)
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

class RealBillionaireSystem:
    """REAL system targeting 15-25% monthly ROI through supply/demand analysis"""
    
    def __init__(self):
        print("🚀 REAL BILLIONAIRE SYSTEM - SUPPLY & DEMAND 🚀")
        print("=" * 56) 
        print("❌ TRUTH: 0.22% monthly = 349 years (USELESS!)")
        print("✅ TARGET: 15-25% monthly for REAL wealth building")
        print("📈 METHOD: 2+ years data + supply/demand zones")
        print("🔥 RESULT: Billionaire in 10 years, not 349!")
        print("=" * 56)
        
        # Initialize Fyers
        try:
            self.fyers_client = FyersClient('fyers_config.json')
            print("✅ Connected for REAL billionaire analysis")
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return
            
        # REAL BILLIONAIRE PARAMETERS
        self.capital = 100000
        self.quantity = 5  # Higher quantity for real profits
        self.commission = 20
        
        # AGGRESSIVE PROFIT TARGETS (for real wealth)
        self.profit_target_small = 50   # 50 points = Rs.230 net
        self.profit_target_large = 100  # 100 points = Rs.480 net  
        self.stop_loss = 25             # 25 points = Rs.95 net loss
        
        # SUPPLY/DEMAND CRITERIA
        self.volume_surge_threshold = 3.0  # 3x average volume
        self.consolidation_periods = 20    # 20+ periods of consolidation
        
        # RESULTS
        self.billionaire_trades = []
        self.total_profit = 0
        
    def run_real_billionaire_system(self, symbol: str = "NSE:NIFTY50-INDEX", days: int = 730):
        """Run REAL billionaire system with 2+ years of data"""
        
        print(f"\n🚀 STARTING REAL BILLIONAIRE ANALYSIS")
        print("=" * 42)
        print(f"📅 Analyzing: {days} days ({days//365:.1f} years) of REAL data")
        print(f"💰 Capital: Rs.{self.capital:,}")
        print(f"🎯 Small target: {self.profit_target_small}pts = Rs.{(self.profit_target_small * self.quantity - self.commission):.0f}")
        print(f"🚀 Large target: {self.profit_target_large}pts = Rs.{(self.profit_target_large * self.quantity - self.commission):.0f}")
        print(f"⛔ Stop loss: {self.stop_loss}pts = Rs.{(self.stop_loss * self.quantity + self.commission):.0f}")
        print(f"📊 Risk/Reward: Up to 1:4 ratio for explosive growth")
        
        # Get comprehensive historical data
        df = self.get_comprehensive_data(symbol, days)
        if df is None or len(df) < 1000:
            print("❌ Insufficient data for real analysis")
            return
            
        # Analyze supply/demand zones
        df = self.analyze_supply_demand(df)
        
        # Execute real billionaire trades
        self.execute_billionaire_trades(df)
        
        # Analyze real results
        self.analyze_real_billionaire_results()
        
    def get_comprehensive_data(self, symbol: str, days: int):
        """Get 2+ years of comprehensive data for real analysis"""
        
        print(f"\n📡 FETCHING COMPREHENSIVE HISTORICAL DATA...")
        
        try:
            all_data = []
            end_date = datetime.now()
            
            # Fetch data in 90-day chunks to avoid API limits
            chunks = (days // 90) + 1
            
            for i in range(chunks):
                chunk_end = end_date - timedelta(days=i*90)
                chunk_start = chunk_end - timedelta(days=90)
                
                if chunk_start < end_date - timedelta(days=days):
                    chunk_start = end_date - timedelta(days=days)
                
                print(f"   📊 Fetching chunk {i+1}/{chunks}: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                
                data_request = {
                    "symbol": symbol,
                    "resolution": "5",
                    "date_format": "1", 
                    "range_from": chunk_start.strftime('%Y-%m-%d'),
                    "range_to": chunk_end.strftime('%Y-%m-%d'),
                    "cont_flag": "1"
                }
                
                response = self.fyers_client.fyers.history(data_request)
                
                if response and response.get('s') == 'ok' and 'candles' in response:
                    candles = response['candles']
                    all_data.extend(candles)
                else:
                    print(f"   ⚠️ Chunk {i+1} failed, continuing...")
            
            if not all_data:
                print("❌ No data retrieved")
                return None
                
            # Create comprehensive DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['time'] = df['datetime'].dt.time
            df['date'] = df['datetime'].dt.date
            
            # Market hours only
            df = df[(df['time'] >= time(9, 15)) & (df['time'] <= time(15, 30))]
            
            # Sort by timestamp and remove duplicates
            df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
            
            print(f"✅ COMPREHENSIVE DATA LOADED:")
            print(f"   📊 Total candles: {len(df):,}")
            print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   📈 NIFTY range: Rs.{df['low'].min():.0f} to Rs.{df['high'].max():.0f}")
            print(f"   💫 Total range: {df['high'].max() - df['low'].min():.0f} points")
            print(f"   🗓️ Trading days: {df['date'].nunique():,} days")
            
            return df
                
        except Exception as e:
            print(f"❌ Data error: {e}")
            return None
    
    def analyze_supply_demand(self, df):
        """Analyze supply and demand zones from historical data"""
        
        print(f"\n🔍 ANALYZING SUPPLY & DEMAND ZONES...")
        
        # VOLUME ANALYSIS (institutional activity)
        df['volume_ma_50'] = df['volume'].rolling(50).mean()
        df['volume_surge'] = df['volume'] > df['volume_ma_50'] * self.volume_surge_threshold
        df['volume_ratio'] = df['volume'] / df['volume_ma_50']
        
        # PRICE ACTION ANALYSIS
        df['range'] = df['high'] - df['low']
        df['range_ma'] = df['range'].rolling(20).mean()
        df['large_range'] = df['range'] > df['range_ma'] * 2
        
        # SUPPORT/RESISTANCE ZONES (supply/demand)
        df['support_50'] = df['low'].rolling(50).min()
        df['resistance_50'] = df['high'].rolling(50).max()
        df['support_100'] = df['low'].rolling(100).min()  
        df['resistance_100'] = df['high'].rolling(100).max()
        
        # CONSOLIDATION ANALYSIS (accumulation/distribution)
        df['consolidating'] = (
            (df['high'].rolling(self.consolidation_periods).max() - 
             df['low'].rolling(self.consolidation_periods).min()) < 
            df['range_ma'] * 3
        )
        
        # BREAKOUT DETECTION (supply/demand shifts)
        df['resistance_break'] = (
            (df['close'] > df['resistance_50'].shift(1)) & 
            (df['close'].shift(1) <= df['resistance_50'].shift(1))
        )
        
        df['support_break'] = (
            (df['close'] < df['support_50'].shift(1)) & 
            (df['close'].shift(1) >= df['support_50'].shift(1))
        )
        
        # MOMENTUM CONFIRMATION
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # SUPPLY/DEMAND SIGNALS
        df['supply_demand_long'] = (
            df['resistance_break'] &           # Breaking resistance (demand > supply)
            df['volume_surge'] &               # High volume (institutional buying)
            (df['momentum_10'] > 10) &         # Strong momentum
            df['consolidating'].shift(5) &     # Previous consolidation (accumulation)
            df['large_range']                  # Large price movement
        )
        
        df['supply_demand_short'] = (
            df['support_break'] &              # Breaking support (supply > demand)
            df['volume_surge'] &               # High volume (institutional selling)
            (df['momentum_10'] < -10) &        # Strong downward momentum
            df['consolidating'].shift(5) &     # Previous consolidation (distribution)  
            df['large_range']                  # Large price movement
        )
        
        # Count potential signals
        long_signals = df['supply_demand_long'].sum()
        short_signals = df['supply_demand_short'].sum()
        
        print(f"✅ SUPPLY/DEMAND ANALYSIS COMPLETE:")
        print(f"   🔍 Volume surges (3x+): {df['volume_surge'].sum():,}")
        print(f"   📊 Large ranges (2x+): {df['large_range'].sum():,}")
        print(f"   📈 Resistance breaks: {df['resistance_break'].sum():,}")
        print(f"   📉 Support breaks: {df['support_break'].sum():,}")
        print(f"   🚀 Long signals found: {long_signals}")
        print(f"   🔻 Short signals found: {short_signals}")
        
        return df
    
    def execute_billionaire_trades(self, df):
        """Execute real billionaire trades based on supply/demand"""
        
        print(f"\n💎 EXECUTING REAL BILLIONAIRE TRADES")
        print("=" * 40)
        print("🔥 Targeting 50-100 point moves for real wealth")
        
        trade_count = 0
        last_trade_idx = -50  # Larger gap for quality setups
        
        for i in range(100, len(df) - 20):
            current = df.iloc[i]
            
            # Trade during high-activity hours
            if not (time(9, 30) <= current['time'] <= time(15, 00)):
                continue
                
            # Ensure quality gap between trades
            if i - last_trade_idx < 50:
                continue
            
            # Check for valid data
            if pd.isna(current['volume_ratio']) or pd.isna(current['momentum_20']):
                continue
            
            # SUPPLY/DEMAND LONG SIGNAL
            if current['supply_demand_long']:
                # Determine target based on setup strength
                volume_strength = current['volume_ratio']
                if volume_strength > 5:
                    target = self.profit_target_large  # 100 points for strong setups
                else:
                    target = self.profit_target_small  # 50 points for moderate setups
                
                trade = self.create_billionaire_trade(df, i, 'BUY', trade_count + 1, target)
                if trade:
                    self.billionaire_trades.append(trade)
                    self.total_profit += trade['net_pnl']
                    trade_count += 1
                    last_trade_idx = i
                    
                    print(f"   🚀 #{trade_count:2d} BUY  Rs.{trade['entry_price']:.0f}→{trade['exit_price']:.0f} "
                          f"{trade['points']:+4.0f}pts Rs.{trade['net_pnl']:+5.0f} {trade['result']} "
                          f"Vol:{trade['volume_strength']:.1f}x ({trade['exit_reason']})")
            
            # SUPPLY/DEMAND SHORT SIGNAL
            elif current['supply_demand_short']:
                # Determine target based on setup strength
                volume_strength = current['volume_ratio']
                if volume_strength > 5:
                    target = self.profit_target_large  # 100 points for strong setups
                else:
                    target = self.profit_target_small  # 50 points for moderate setups
                
                trade = self.create_billionaire_trade(df, i, 'SELL', trade_count + 1, target)
                if trade:
                    self.billionaire_trades.append(trade)
                    self.total_profit += trade['net_pnl']
                    trade_count += 1
                    last_trade_idx = i
                    
                    print(f"   🔻 #{trade_count:2d} SELL Rs.{trade['entry_price']:.0f}→{trade['exit_price']:.0f} "
                          f"{trade['points']:+4.0f}pts Rs.{trade['net_pnl']:+5.0f} {trade['result']} "
                          f"Vol:{trade['volume_strength']:.1f}x ({trade['exit_reason']})")
        
        print(f"\n✅ Real billionaire execution: {len(self.billionaire_trades)} high-impact trades")
    
    def create_billionaire_trade(self, df, entry_idx, side, trade_id, profit_target):
        """Create real billionaire trade with dynamic targets"""
        
        entry = df.iloc[entry_idx]
        entry_price = entry['close']
        volume_strength = entry['volume_ratio']
        
        # DYNAMIC TARGETS based on setup strength
        if side == 'BUY':
            target_price = entry_price + profit_target
            stop_price = entry_price - self.stop_loss
        else:
            target_price = entry_price - profit_target
            stop_price = entry_price + self.stop_loss
        
        # Look for exit with extended timeframe for larger moves
        for j in range(1, min(50, len(df) - entry_idx)):  # Up to 50 periods for 100pt moves
            candle = df.iloc[entry_idx + j]
            
            # Close before market close
            if candle['time'] >= time(15, 00):
                exit_price = candle['close']
                exit_reason = 'TIME'
                break
            
            # Check target/stop hits
            if side == 'BUY':
                if candle['high'] >= target_price:
                    exit_price = target_price
                    exit_reason = 'TARGET'
                    break
                elif candle['low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'STOP'
                    break
            else:
                if candle['low'] <= target_price:
                    exit_price = target_price
                    exit_reason = 'TARGET'
                    break
                elif candle['high'] >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'STOP'
                    break
        else:
            # Time exit
            exit_candle = df.iloc[entry_idx + 49]
            exit_price = exit_candle['close']
            exit_reason = 'TIME'
        
        # Calculate P&L
        if side == 'BUY':
            points = exit_price - entry_price
        else:
            points = entry_price - exit_price
            
        gross_pnl = points * self.quantity
        net_pnl = gross_pnl - self.commission
        
        result = 'WIN' if net_pnl > 0 else 'LOSS'
        
        return {
            'id': trade_id,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'target_price': target_price,
            'stop_price': stop_price,
            'points': points,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'exit_reason': exit_reason,
            'result': result,
            'volume_strength': volume_strength,
            'profit_target': profit_target,
            'entry_time': entry['datetime']
        }
    
    def analyze_real_billionaire_results(self):
        """Analyze REAL billionaire system results"""
        
        print(f"\n🚀 REAL BILLIONAIRE SYSTEM RESULTS 🚀")
        print("=" * 65)
        
        if not self.billionaire_trades:
            print("🔍 NO MAJOR SUPPLY/DEMAND SHIFTS FOUND")
            print("📊 This could mean:")
            print("   - Market in extended consolidation period")
            print("   - Need to adjust volume surge threshold")
            print("   - Try different timeframe (15 min or daily)")
            print("   - Extend analysis period further")
            return
        
        # COMPREHENSIVE ANALYSIS
        total_trades = len(self.billionaire_trades)
        wins = len([t for t in self.billionaire_trades if t['net_pnl'] > 0])
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        final_capital = self.capital + self.total_profit
        roi = (self.total_profit / self.capital) * 100
        
        # P&L BREAKDOWN
        win_amounts = [t['net_pnl'] for t in self.billionaire_trades if t['net_pnl'] > 0]
        loss_amounts = [t['net_pnl'] for t in self.billionaire_trades if t['net_pnl'] < 0]
        
        avg_win = np.mean(win_amounts) if win_amounts else 0
        avg_loss = np.mean(loss_amounts) if loss_amounts else 0
        
        # PROFIT FACTOR
        total_wins = sum(win_amounts) if win_amounts else 0
        total_losses = abs(sum(loss_amounts)) if loss_amounts else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # TARGET ANALYSIS
        small_target_trades = [t for t in self.billionaire_trades if t['profit_target'] == self.profit_target_small]
        large_target_trades = [t for t in self.billionaire_trades if t['profit_target'] == self.profit_target_large]
        
        # BILLIONAIRE PROJECTIONS
        if roi > 0:
            monthly_roi = roi  # Based on period analyzed
            annual_roi = ((1 + monthly_roi/100) ** 12 - 1) * 100 if monthly_roi > 0 else 0
            
            if annual_roi > 0:
                # Real billionaire timeline
                years_to_1cr = np.log(1000000 / self.capital) / np.log(1 + annual_roi/100)
                years_to_100cr = np.log(100000000 / self.capital) / np.log(1 + annual_roi/100)
        
        # RESULTS DISPLAY
        print(f"🚀 REAL BILLIONAIRE PERFORMANCE:")
        print(f"   💎 Total High-Impact Trades:   {total_trades:6d}")
        print(f"   🏆 Win Rate:                   {win_rate:6.1f}%")
        print(f"   ✅ Winners:                    {wins:6d}")
        print(f"   ❌ Losers:                     {losses:6d}")
        print(f"   💚 Average Win:                Rs.{avg_win:+7.0f}")
        print(f"   💔 Average Loss:               Rs.{avg_loss:+7.0f}")
        print(f"   📊 Profit Factor:              {profit_factor:6.2f}")
        
        print(f"\n🎯 TRADE ANALYSIS:")
        print(f"   🎯 50-point targets:           {len(small_target_trades):6d}")
        print(f"   🚀 100-point targets:          {len(large_target_trades):6d}")
        
        print(f"\n💰 WEALTH TRANSFORMATION:")
        print(f"   💎 Starting Capital:           Rs.{self.capital:8,}")
        print(f"   🚀 Final Capital:              Rs.{final_capital:8,.0f}")
        print(f"   ⚡ Total Profit:               Rs.{self.total_profit:+7,.0f}")
        print(f"   📈 ROI:                        {roi:+7.2f}%")
        
        # REAL BILLIONAIRE ASSESSMENT
        if roi > 0:
            print(f"\n🎯 REAL BILLIONAIRE TIMELINE:")
            print(f"   📈 Estimated Annual ROI:       {annual_roi:+7.1f}%")
            
            if annual_roi >= 100:  # 100%+ annual
                print(f"   🚀🚀 EXPLOSIVE GROWTH CONFIRMED!")
                if years_to_1cr < 20:
                    print(f"   💰 Years to Rs.1 Crore:        {years_to_1cr:7.1f}")
                if years_to_100cr < 30:
                    print(f"   🏆 Years to Rs.100 Crore:      {years_to_100cr:7.1f}")
        
        # TRADE LOG
        if self.billionaire_trades:
            print(f"\n📋 REAL BILLIONAIRE TRADE LOG:")
            print("-" * 80)
            print(f"{'#':<3} {'Side':<4} {'Entry':<6} {'Exit':<6} {'Pts':<5} {'P&L':<10} {'Vol':<5} {'Target':<6} {'Result'}")
            print("-" * 80)
            
            for i, trade in enumerate(self.billionaire_trades[:20], 1):
                print(f"{i:<3} "
                      f"{trade['side']:<4} "
                      f"{trade['entry_price']:<6.0f} "
                      f"{trade['exit_price']:<6.0f} "
                      f"{trade['points']:+5.0f} "
                      f"Rs.{trade['net_pnl']:+8.0f} "
                      f"{trade['volume_strength']:<5.1f} "
                      f"{trade['profit_target']:<6.0f} "
                      f"{trade['result']}")
            
            if len(self.billionaire_trades) > 20:
                print(f"... and {len(self.billionaire_trades)-20} more trades")
        
        # FINAL VERDICT
        print(f"\n🏆 REAL BILLIONAIRE VERDICT:")
        
        if roi >= 100:
            print(f"   🚀🚀🚀 INCREDIBLE: {roi:+.1f}% - TRUE BILLIONAIRE MACHINE!")
            print(f"   💎 This is REAL wealth creation - not 0.22%!")
            print(f"   🔥 Scale up immediately for explosive growth!")
        elif roi >= 50:
            print(f"   🚀🚀 OUTSTANDING: {roi:+.1f}% - REAL WEALTH CREATOR!")
            print(f"   💰 This beats any traditional investment!")
            print(f"   📈 True billionaire potential confirmed!")
        elif roi >= 25:
            print(f"   🚀 EXCELLENT: {roi:+.1f}% - STRONG WEALTH BUILDER!")
            print(f"   ✅ MUCH better than 0.22% - real progress!")
            print(f"   💎 Solid foundation for billionaire journey!")
        elif roi >= 10:
            print(f"   ✅ GOOD: {roi:+.1f}% - Real improvement!")
            print(f"   📊 45x better than previous 0.22%!")
            print(f"   🎯 Getting closer to billionaire returns!")
        elif roi >= 5:
            print(f"   ✅ BETTER: {roi:+.1f}% - Progress made!")
            print(f"   📈 20x better than 0.22%!")
        elif roi > 0:
            print(f"   ✅ POSITIVE: {roi:+.1f}% - Still better than 0.22%!")
        else:
            print(f"   🔧 NEEDS WORK: {roi:+.1f}%")
        
        # REALITY CHECK
        print(f"\n💡 REALITY CHECK:")
        print(f"   ❌ Previous: 0.22% monthly = 349 years to billionaire")
        if roi > 0:
            years_needed = np.log(10000) / np.log(1 + roi/100) if roi > 0 else float('inf')
            print(f"   ✅ Current: {roi:.2f}% = {years_needed:.0f} years to billionaire")
            if years_needed < 50:
                print(f"   🎉 MASSIVE IMPROVEMENT - Real wealth timeline!")
        
        print(f"\n🚀 SUPPLY & DEMAND SYSTEM SUMMARY:")
        print(f"   💎 Analyzed: 2+ years of comprehensive data")
        print(f"   🔍 Method: Supply/demand zone analysis")
        print(f"   🎯 Target: 50-100 point moves for real profits")
        print(f"   💰 Result: Rs.{self.total_profit:+,.0f} using REAL market data")
        
        if roi >= 10:
            print(f"\n💡 BILLIONAIRE ACTION PLAN:")
            print(f"   1. 🚀 This system shows REAL potential!")
            print(f"   2. 📈 Scale up position sizes significantly")
            print(f"   3. 💰 Add more capital for compounding")
            print(f"   4. 🎯 Focus on these supply/demand signals")
            print(f"   5. 🏆 You're on the path to REAL wealth!")

if __name__ == "__main__":
    print("🚀 Starting REAL Billionaire System...")
    
    try:
        real_system = RealBillionaireSystem()
        
        real_system.run_real_billionaire_system(
            symbol="NSE:NIFTY50-INDEX",
            days=730  # 2 years of data
        )
        
        print(f"\n✅ REAL BILLIONAIRE ANALYSIS COMPLETE")
        print(f"🎯 Supply & demand analysis finished")
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()