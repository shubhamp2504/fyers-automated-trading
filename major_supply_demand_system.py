#!/usr/bin/env python3
"""
🔥 MAJOR SUPPLY/DEMAND IMBALANCE SYSTEM 🔥
================================================================================
✅ YOUR TRUTH: Small profits = useless for billionaire wealth
💎 THE SOLUTION: Find MAJOR supply/demand imbalances = 100-500 point moves  
🚀 TARGET: 50-100% annual returns from big institutional moves
📈 METHOD: Daily timeframe + major S/R levels + volume exhaustion
================================================================================
REAL BILLIONAIRE APPROACH:
- Use daily timeframe (not 5-min scalping)
- Find major support/resistance levels (tested multiple times)
- Wait for MASSIVE volume spikes (institutional activity)
- Target 100-500 point moves (Rs.500-2500 per trade)
- 2-3 trades per month = Rs.10,000+ monthly profit = 10%+ returns

SUPPLY/DEMAND FUNDAMENTALS:
1. Major S/R levels where institutions accumulate/distribute
2. Volume spikes 5x+ average = institutional activity  
3. False breakouts = major reversal opportunities
4. News-driven moves = supply/demand shifts
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

class MajorSupplyDemandSystem:
    """System targeting MAJOR supply/demand imbalances for real wealth"""
    
    def __init__(self):
        print("🔥 MAJOR SUPPLY/DEMAND IMBALANCE SYSTEM 🔥")
        print("=" * 56)
        print("✅ FOCUS: Major institutional moves = 100-500 points")
        print("💎 TIMEFRAME: Daily analysis (not tiny scalping)")
        print("🚀 TARGET: 50-100% annual = REAL billionaire path") 
        print("📈 METHOD: Major S/R + massive volume + big moves")
        print("=" * 56)
        
        # Initialize Fyers
        try:
            self.fyers_client = FyersClient('fyers_config.json')
            print("✅ Connected for major move analysis")
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return
            
        # MAJOR MOVE PARAMETERS
        self.capital = 100000
        self.quantity = 10  # Higher quantity for real profits
        self.commission = 20
        
        # MAJOR TARGETS (for real wealth)
        self.small_move = 100   # 100 points = Rs.980 net
        self.medium_move = 200  # 200 points = Rs.1980 net  
        self.large_move = 300   # 300 points = Rs.2980 net
        self.stop_loss = 50     # 50 points = Rs.520 net loss
        
        # MAJOR CRITERIA (institutional activity)
        self.major_volume_threshold = 5.0  # 5x volume
        self.major_support_tests = 3       # 3+ tests of level
        
        # RESULTS
        self.major_trades = []
        self.total_profit = 0
        
    def run_major_system(self, symbol: str = "NSE:NIFTY50-INDEX", days: int = 730):
        """Run major supply/demand system"""
        
        print(f"\n🔥 MAJOR SUPPLY/DEMAND ANALYSIS STARTING")
        print("=" * 48)
        print(f"💰 Capital: Rs.{self.capital:,}")
        print(f"🎯 Small move: {self.small_move}pts = Rs.{(self.small_move * self.quantity - self.commission):,}")
        print(f"🚀 Medium move: {self.medium_move}pts = Rs.{(self.medium_move * self.quantity - self.commission):,}")
        print(f"💎 Large move: {self.large_move}pts = Rs.{(self.large_move * self.quantity - self.commission):,}")
        print(f"⛔ Stop loss: {self.stop_loss}pts = Rs.{(self.stop_loss * self.quantity + self.commission):,}")
        
        # Get DAILY data for major moves
        df = self.get_daily_data(symbol, days)
        if df is None or len(df) < 200:
            print("❌ Insufficient daily data")
            return
            
        # Analyze major supply/demand levels
        df = self.find_major_levels(df)
        
        # Execute major trades
        self.execute_major_trades(df)
        
        # Analyze major results
        self.analyze_major_results()
        
    def get_daily_data(self, symbol: str, days: int):
        """Get DAILY timeframe data for major analysis"""
        
        print(f"\n📡 FETCHING DATA FOR MAJOR ANALYSIS...")
        
        try:
            # Use same approach as working systems
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Try multiple chunk approach for large dataset
            all_candles = []
            chunk_size = 100  # 100 days per chunk
            
            current_date = start_date
            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=chunk_size), end_date)
                
                data_request = {
                    "symbol": symbol,
                    "resolution": "60",  # 1-HOUR timeframe
                    "date_format": "1", 
                    "range_from": current_date.strftime('%Y-%m-%d'),
                    "range_to": chunk_end.strftime('%Y-%m-%d'),
                    "cont_flag": "1"
                }
                
                response = self.fyers_client.fyers.history(data_request)
                
                if response and response.get('s') == 'ok' and 'candles' in response:
                    chunk_candles = response['candles']
                    all_candles.extend(chunk_candles)
                    print(f"   ✅ Fetched {len(chunk_candles)} candles from {current_date.strftime('%Y-%m-%d')}")
                else:
                    print(f"   ⚠️ No data for chunk {current_date.strftime('%Y-%m-%d')}")
                
                current_date = chunk_end
            
            if all_candles:
                df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['date'] = df['datetime'].dt.date
                df['hour'] = df['datetime'].dt.hour
                
                # Convert to daily data by taking OHLC for each day
                daily_df = df.groupby('date').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'datetime': 'first',
                    'timestamp': 'first'
                }).reset_index()
                
                daily_df = daily_df.sort_values('timestamp').reset_index(drop=True)
                
                print(f"✅ MAJOR DATA LOADED:")
                print(f"   📊 Total trading days: {len(daily_df):,}")
                print(f"   📅 Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
                print(f"   📈 NIFTY range: Rs.{daily_df['low'].min():.0f} to Rs.{daily_df['high'].max():.0f}")
                print(f"   💫 Major range: {daily_df['high'].max() - daily_df['low'].min():.0f} points")
                
                return daily_df
                
            else:
                print(f"❌ No data retrieved")
                return None
                
        except Exception as e:
            print(f"❌ Data fetch error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def find_major_levels(self, df):
        """Find MAJOR supply/demand levels"""
        
        print(f"\n🔍 IDENTIFYING MAJOR SUPPLY/DEMAND LEVELS...")
        
        # MAJOR VOLUME ANALYSIS
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['major_volume'] = df['volume'] > df['volume_ma'] * self.major_volume_threshold
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # MAJOR RANGE ANALYSIS
        df['range'] = df['high'] - df['low']
        df['range_ma'] = df['range'].rolling(20).mean()
        df['major_range'] = df['range'] > df['range_ma'] * 2
        
        # MAJOR SUPPORT/RESISTANCE LEVELS
        window = 20
        df['major_support'] = df['low'].rolling(window*2).min()
        df['major_resistance'] = df['high'].rolling(window*2).max()
        
        # LEVEL TESTING (multiple touches = strong S/R)
        df['near_support'] = abs(df['low'] - df['major_support']) < df['range_ma'] * 0.5
        df['near_resistance'] = abs(df['high'] - df['major_resistance']) < df['range_ma'] * 0.5
        
        # Count tests of levels
        df['support_tests'] = df['near_support'].rolling(10).sum()
        df['resistance_tests'] = df['near_resistance'].rolling(10).sum()
        
        # INSTITUTIONAL ACTIVITY (major volume at major levels)
        df['institutional_buying'] = (
            df['major_volume'] & 
            (df['close'] > df['open']) &  # Bullish day
            df['near_support'] &          # At major support
            (df['support_tests'] >= self.major_support_tests)
        )
        
        df['institutional_selling'] = (
            df['major_volume'] & 
            (df['close'] < df['open']) &  # Bearish day
            df['near_resistance'] &       # At major resistance  
            (df['resistance_tests'] >= self.major_support_tests)
        )
        
        # FALSE BREAKOUTS (major reversal opportunities)
        df['false_breakout_long'] = (
            (df['high'].shift(1) > df['major_resistance'].shift(1)) &  # Broke resistance yesterday
            (df['close'] < df['major_resistance'].shift(1)) &          # Closed back below
            df['major_volume'] &                                       # High volume
            df['major_range']                                          # Large range
        )
        
        df['false_breakout_short'] = (
            (df['low'].shift(1) < df['major_support'].shift(1)) &      # Broke support yesterday
            (df['close'] > df['major_support'].shift(1)) &            # Closed back above
            df['major_volume'] &                                       # High volume
            df['major_range']                                          # Large range
        )
        
        # Count major opportunities
        inst_buying = df['institutional_buying'].sum()
        inst_selling = df['institutional_selling'].sum()
        false_long = df['false_breakout_long'].sum()
        false_short = df['false_breakout_short'].sum()
        
        print(f"✅ MAJOR LEVELS IDENTIFIED:")
        print(f"   🔍 Major volume days (5x+): {df['major_volume'].sum()}")
        print(f"   📊 Major range days (2x+): {df['major_range'].sum()}")
        print(f"   💰 Institutional buying: {inst_buying}")
        print(f"   💔 Institutional selling: {inst_selling}")
        print(f"   🔄 False breakout longs: {false_long}")
        print(f"   🔄 False breakout shorts: {false_short}")
        print(f"   💎 TOTAL MAJOR SIGNALS: {inst_buying + inst_selling + false_long + false_short}")
        
        return df
    
    def execute_major_trades(self, df):
        """Execute major supply/demand trades"""
        
        print(f"\n🔥 EXECUTING MAJOR SUPPLY/DEMAND TRADES")
        print("=" * 47)
        print("💎 Targeting institutional-level moves")
        
        trade_count = 0
        
        for i in range(50, len(df) - 5):
            current = df.iloc[i]
            
            # Check for valid data
            if pd.isna(current['volume_ratio']) or pd.isna(current['major_support']):
                continue
            
            # INSTITUTIONAL BUYING SIGNAL
            if current['institutional_buying']:
                target = self.determine_target(current['volume_ratio'], 'long')
                trade = self.create_major_trade(df, i, 'BUY', trade_count + 1, target, 'INSTITUTIONAL')
                if trade:
                    self.major_trades.append(trade)
                    self.total_profit += trade['net_pnl']
                    trade_count += 1
                    
                    print(f"   💰 #{trade_count:2d} BUY  Rs.{trade['entry_price']:.0f}→{trade['exit_price']:.0f} "
                          f"{trade['points']:+4.0f}pts Rs.{trade['net_pnl']:+6.0f} {trade['result']} "
                          f"Vol:{trade['volume_strength']:.1f}x [{trade['signal_type']}]")
            
            # INSTITUTIONAL SELLING SIGNAL
            elif current['institutional_selling']:
                target = self.determine_target(current['volume_ratio'], 'short')
                trade = self.create_major_trade(df, i, 'SELL', trade_count + 1, target, 'INSTITUTIONAL')
                if trade:
                    self.major_trades.append(trade)
                    self.total_profit += trade['net_pnl']
                    trade_count += 1
                    
                    print(f"   💔 #{trade_count:2d} SELL Rs.{trade['entry_price']:.0f}→{trade['exit_price']:.0f} "
                          f"{trade['points']:+4.0f}pts Rs.{trade['net_pnl']:+6.0f} {trade['result']} "
                          f"Vol:{trade['volume_strength']:.1f}x [{trade['signal_type']}]")
            
            # FALSE BREAKOUT REVERSAL LONG
            elif current['false_breakout_long']:
                target = self.medium_move  # Medium target for reversals
                trade = self.create_major_trade(df, i, 'BUY', trade_count + 1, target, 'FALSE_BREAKOUT')
                if trade:
                    self.major_trades.append(trade)
                    self.total_profit += trade['net_pnl']
                    trade_count += 1
                    
                    print(f"   🔄 #{trade_count:2d} BUY  Rs.{trade['entry_price']:.0f}→{trade['exit_price']:.0f} "
                          f"{trade['points']:+4.0f}pts Rs.{trade['net_pnl']:+6.0f} {trade['result']} "
                          f"Vol:{trade['volume_strength']:.1f}x [{trade['signal_type']}]")
            
            # FALSE BREAKOUT REVERSAL SHORT
            elif current['false_breakout_short']:
                target = self.medium_move  # Medium target for reversals
                trade = self.create_major_trade(df, i, 'SELL', trade_count + 1, target, 'FALSE_BREAKOUT')
                if trade:
                    self.major_trades.append(trade)
                    self.total_profit += trade['net_pnl']
                    trade_count += 1
                    
                    print(f"   🔄 #{trade_count:2d} SELL Rs.{trade['entry_price']:.0f}→{trade['exit_price']:.0f} "
                          f"{trade['points']:+4.0f}pts Rs.{trade['net_pnl']:+6.0f} {trade['result']} "
                          f"Vol:{trade['volume_strength']:.1f}x [{trade['signal_type']}]")
        
        print(f"\n✅ Major execution complete: {len(self.major_trades)} institutional trades")
    
    def determine_target(self, volume_strength, direction):
        """Determine target based on volume strength"""
        if volume_strength > 8:
            return self.large_move    # 300 points for massive volume
        elif volume_strength > 6:
            return self.medium_move   # 200 points for strong volume
        else:
            return self.small_move    # 100 points for moderate volume
    
    def create_major_trade(self, df, entry_idx, side, trade_id, profit_target, signal_type):
        """Create major supply/demand trade"""
        
        entry = df.iloc[entry_idx]
        entry_price = entry['close']
        volume_strength = entry['volume_ratio']
        
        # Set targets
        if side == 'BUY':
            target_price = entry_price + profit_target
            stop_price = entry_price - self.stop_loss
        else:
            target_price = entry_price - profit_target
            stop_price = entry_price + self.stop_loss
        
        # Look for exit over next 10 days (major moves take time)
        exit_price = None
        exit_reason = None
        
        for j in range(1, min(10, len(df) - entry_idx)):
            candle = df.iloc[entry_idx + j]
            
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
        
        # If no exit found, use time exit
        if exit_price is None:
            exit_candle = df.iloc[entry_idx + min(9, len(df) - entry_idx - 1)]
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
            'signal_type': signal_type,
            'entry_time': entry['datetime']
        }
    
    def analyze_major_results(self):
        """Analyze major supply/demand results"""
        
        print(f"\n🔥 MAJOR SUPPLY/DEMAND RESULTS 🔥")
        print("=" * 65)
        
        if not self.major_trades:
            print("❌ NO MAJOR INSTITUTIONAL ACTIVITY FOUND")
            print("📊 This suggests:")
            print("   - Market in normal conditions (no major moves)")
            print("   - Need to analyze longer period (3-5 years)")
            print("   - Try different criteria (reduce volume threshold)")
            print("   - Consider intraday data for more frequent signals")
            print("")
            print("💡 ALTERNATIVE APPROACH:")
            print("   - Reduce major volume threshold to 3x")
            print("   - Use weekly timeframe for even bigger moves")
            print("   - Analyze specific event-driven periods")
            return
        
        # MAJOR PERFORMANCE METRICS
        total_trades = len(self.major_trades)
        wins = len([t for t in self.major_trades if t['net_pnl'] > 0])
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        final_capital = self.capital + self.total_profit
        roi = (self.total_profit / self.capital) * 100
        
        # P&L BREAKDOWN
        win_amounts = [t['net_pnl'] for t in self.major_trades if t['net_pnl'] > 0]
        loss_amounts = [t['net_pnl'] for t in self.major_trades if t['net_pnl'] < 0]
        
        avg_win = np.mean(win_amounts) if win_amounts else 0
        avg_loss = np.mean(loss_amounts) if loss_amounts else 0
        
        total_wins_pnl = sum(win_amounts) if win_amounts else 0
        total_losses_pnl = abs(sum(loss_amounts)) if loss_amounts else 1
        profit_factor = total_wins_pnl / total_losses_pnl if total_losses_pnl > 0 else float('inf')
        
        # SIGNAL TYPE BREAKDOWN
        institutional_trades = [t for t in self.major_trades if t['signal_type'] == 'INSTITUTIONAL']
        false_breakout_trades = [t for t in self.major_trades if t['signal_type'] == 'FALSE_BREAKOUT']
        
        # BILLIONAIRE PROJECTIONS
        annual_roi = 0  # Initialize 
        if roi > 0:
            # Annualize from 2-year period
            annual_roi = ((1 + roi/100) ** (12/24) - 1) * 100
            
            if annual_roi > 0:
                years_to_1cr = np.log(1000000 / self.capital) / np.log(1 + annual_roi/100)
                years_to_10cr = np.log(10000000 / self.capital) / np.log(1 + annual_roi/100)
        else:
            annual_roi = roi / 2  # Negative annual rate
        
        # RESULTS DISPLAY
        print(f"🔥 MAJOR PERFORMANCE METRICS:")
        print(f"   💎 Major Trades:               {total_trades:6d}")
        print(f"   🏆 Win Rate:                   {win_rate:6.1f}%")
        print(f"   ✅ Winners:                    {wins:6d}")
        print(f"   ❌ Losers:                     {losses:6d}")
        print(f"   💰 Average Win:                Rs.{avg_win:+7.0f}")
        print(f"   💔 Average Loss:               Rs.{avg_loss:+7.0f}")
        print(f"   📊 Profit Factor:              {profit_factor:6.2f}")
        
        print(f"\n🎯 SIGNAL TYPE BREAKDOWN:")
        print(f"   🏦 Institutional:              {len(institutional_trades):6d} trades")
        print(f"   🔄 False Breakouts:            {len(false_breakout_trades):6d} trades")
        
        print(f"\n💰 MAJOR WEALTH TRANSFORMATION:")
        print(f"   💎 Starting Capital:           Rs.{self.capital:8,}")
        print(f"   🚀 Final Capital:              Rs.{final_capital:8,.0f}")
        print(f"   ⚡ Total Profit:               Rs.{self.total_profit:+7,.0f}")
        print(f"   📈 ROI (2 years):              {roi:+7.2f}%")
        
        # MAJOR TIMELINE
        if roi > 0:
            print(f"\n🎯 MAJOR WEALTH TIMELINE:")
            print(f"   📈 Annual ROI:                 {annual_roi:+7.1f}%")
            
            if annual_roi >= 25:
                if years_to_1cr < 20:
                    print(f"   💰 Years to Rs.1 Crore:        {years_to_1cr:7.1f}")
                if years_to_10cr < 30:
                    print(f"   🚀 Years to Rs.10 Crore:       {years_to_10cr:7.1f}")
        
        # TRADE LOG
        if self.major_trades:
            print(f"\n📋 MAJOR TRADE LOG:")
            print("-" * 90)
            print(f"{'#':<3} {'Side':<4} {'Entry':<6} {'Exit':<6} {'Pts':<5} {'P&L':<10} {'Vol':<5} {'Type':<12} {'Result'}")
            print("-" * 90)
            
            for i, trade in enumerate(self.major_trades, 1):
                print(f"{i:<3} "
                      f"{trade['side']:<4} "
                      f"{trade['entry_price']:<6.0f} "
                      f"{trade['exit_price']:<6.0f} "
                      f"{trade['points']:+5.0f} "
                      f"Rs.{trade['net_pnl']:+8.0f} "
                      f"{trade['volume_strength']:<5.1f} "
                      f"{trade['signal_type']:<12} "
                      f"{trade['result']}")
        
        # MAJOR VERDICT
        print(f"\n🏆 MAJOR SYSTEM VERDICT:")
        
        if roi >= 100:
            print(f"   🚀🚀🚀 BREAKTHROUGH: {roi:+.1f}% over 2 years!")
            print(f"   💎 MAJOR institutional moves captured!")
            print(f"   🔥 {annual_roi:.0f}% annual = TRUE billionaire path!")
        elif roi >= 50:
            print(f"   🚀🚀 EXCELLENT: {roi:+.1f}% over 2 years!")
            print(f"   💰 Major moves strategy working!")
            print(f"   📈 {annual_roi:.0f}% annual beats all previous attempts!")
        elif roi >= 25:
            print(f"   🚀 VERY GOOD: {roi:+.1f}% over 2 years!")
            print(f"   ✅ Major approach showing results!")
            print(f"   💎 {annual_roi:.0f}% annual - real progress!")
        elif roi >= 10:
            print(f"   ✅ GOOD: {roi:+.1f}% over 2 years!")
            print(f"   📊 Better than scalping approaches!")
        elif roi > 0:
            print(f"   ✅ POSITIVE: {roi:+.1f}% over 2 years!")
            print(f"   💡 Major move approach has potential!")
        else:
            print(f"   🔧 NEEDS REFINEMENT: {roi:+.1f}%")
        
        # COMPARISON
        print(f"\n📊 SUPPLY/DEMAND SYSTEM COMPARISON:")
        print(f"   ❌ 5-min scalping: 0.2% annually = useless")
        print(f"   ✅ Daily major moves: {annual_roi:.1f}% annually")
        if annual_roi > 0.5:
            improvement = annual_roi / 0.2
            print(f"   🚀 IMPROVEMENT: {improvement:.0f}x better approach!")
        
        print(f"\n🔥 MAJOR SYSTEM SUMMARY:")
        print(f"   💎 Method: Daily major supply/demand analysis")
        print(f"   🎯 Focus: Institutional activity + false breakouts")
        print(f"   💰 Result: Rs.{self.total_profit:+,.0f} from {total_trades} major trades")
        print(f"   🏆 Approach: {win_rate:.1f}% accuracy on major moves")
        
        if roi >= 10:
            print(f"\n💡 MAJOR WEALTH ACTION PLAN:")
            print(f"   1. 🔥 Major move approach shows promise!")
            print(f"   2. 📈 Focus on these institutional signals")
            print(f"   3. 💰 Scale position sizes for major profits")
            print(f"   4. 🎯 Wait for quality major setups")
            print(f"   5. 🏆 This beats small scalping profits!")

if __name__ == "__main__":
    print("🔥 Starting Major Supply/Demand System...")
    
    try:
        major_system = MajorSupplyDemandSystem()
        
        major_system.run_major_system(
            symbol="NSE:NIFTY50-INDEX",
            days=730
        )
        
        print(f"\n✅ MAJOR ANALYSIS COMPLETE")
        print(f"🔥 Institutional supply/demand analysis finished")
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()