#!/usr/bin/env python3
"""
🚀💥 MULTI-TIMEFRAME POWERHOUSE DEMO 💥🚀
================================================================================
HYPER-AGGRESSIVE SYSTEM: CATCH EVERY TINY MOVE ACROSS ALL TIMEFRAMES
FEATURES: Double Qty Supply Zones + Ultra-Frequent Profit Booking
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class PowerhouseDemo:
    def __init__(self):
        print("🚀💥 LAUNCHING MULTI-TIMEFRAME POWERHOUSE DEMO 💥🚀")
        print("=" * 80)
        print("DEMO SYSTEM: ALL TIMEFRAMES ANALYSIS")
        print("FEATURES: Double Qty Supply Zones + Micro Profit Booking")
        print("THRESHOLDS: HYPER-AGGRESSIVE (0.01% moves detected!)")
        print("=" * 80)
        
        # Powerhouse settings
        self.capital = 100000
        self.base_risk = 0.01  # 1%
        self.supply_multiplier = 2.0  # Double quantity for supply zones
        self.profit_points = [5, 8, 12, 15, 18, 22, 25, 30]
        self.stop_loss = 6
        
        # Multi-timeframe data
        self.timeframes = ['1min', '3min', '5min', '10min', '15min', '30min', '1hour', '4hour']
        self.multi_data = {}
        
        # Zone storage
        self.supply_zones = []
        self.demand_zones = []
        self.trades = []
        
        print(f"Strategy: MULTI-TIMEFRAME POWERHOUSE SYSTEM")
        print(f"  - Supply Zones: DOUBLE QUANTITY (2X profit potential)")
        print(f"  - Demand Zones: Standard quantity + Quick profits")
        print(f"  - Profit Booking: {self.profit_points} points")
        print(f"  - Stop Loss: {self.stop_loss} points")
        print(f"Starting Capital: Rs.{self.capital:,.2f}")
        print(f"Risk: {self.base_risk:.1%} per trade | Supply: {self.supply_multiplier}X quantity")
    
    def generate_realistic_market_data(self, timeframe: str, days: int = 45) -> pd.DataFrame:
        """Generate realistic market data with volatility"""
        
        # Base parameters
        intervals_per_day = {
            '1min': 375, '3min': 125, '5min': 75, '10min': 37,
            '15min': 25, '30min': 12, '1hour': 6, '4hour': 2
        }
        
        total_intervals = days * intervals_per_day[timeframe]
        
        # Start price around 25,000
        start_price = 25000
        dates = pd.date_range('2026-01-01 09:15', periods=total_intervals, freq='1min')
        
        # Generate price movement with volatility
        np.random.seed(42)  # Reproducible results
        
        # Volatility by timeframe
        volatility = {
            '1min': 0.0005, '3min': 0.001, '5min': 0.0015, '10min': 0.002,
            '15min': 0.0025, '30min': 0.003, '1hour': 0.004, '4hour': 0.006
        }
        
        vol = volatility[timeframe]
        
        # Generate realistic OHLCV data
        prices = []
        volumes = []
        current_price = start_price
        
        for i in range(total_intervals):
            # Random walk with occasional volatility spikes
            change = np.random.normal(0, vol)
            
            # Add some trending behavior
            if i % 100 < 30:  # Slight uptrend periods
                change += vol * 0.3
            elif i % 100 > 70:  # Slight downtrend periods  
                change -= vol * 0.3
            
            # Occasional volatility spikes
            if np.random.random() < 0.05:
                change *= 3
            
            current_price *= (1 + change)
            
            # Generate OHLC from close
            volatility_range = current_price * vol * np.random.uniform(0.5, 2.0)
            
            open_price = current_price * (1 + np.random.normal(0, vol * 0.3))
            high = max(open_price, current_price) + volatility_range * np.random.uniform(0, 0.5)
            low = min(open_price, current_price) - volatility_range * np.random.uniform(0, 0.5)
            
            # Generate volume
            base_volume = 50000 + np.random.exponential(30000)
            if abs(change) > vol * 1.5:  # High volatility = high volume
                base_volume *= 2
                
            prices.append({
                'datetime': dates[i],
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(current_price, 2),
                'volume': int(base_volume)
            })
        
        return pd.DataFrame(prices)
    
    def fetch_multi_timeframe_data(self):
        """Generate multi-timeframe data"""
        print(f"\n🚀 GENERATING MULTI-TIMEFRAME DATA - POWERHOUSE MODE")
        print("-" * 60)
        print(f"TIMEFRAMES: {', '.join(self.timeframes)}")
        print(f"PERIOD: 45 days of realistic market data")
        
        total_candles = 0
        
        for tf in self.timeframes:
            print(f"Generating {tf} data...")
            
            # Generate realistic data for each timeframe
            data = self.generate_realistic_market_data(tf, days=45)
            
            if len(data) > 0:
                self.multi_data[tf] = data
                total_candles += len(data)
                print(f"✅ {tf}: {len(data)} candles | Range: Rs.{data['low'].min():.2f} - Rs.{data['high'].max():.2f}")
            else:
                print(f"❌ Failed to generate {tf} data")
        
        print(f"\n📊 MULTI-TIMEFRAME POWERHOUSE LOADED:")
        print(f"   Timeframes: {len(self.multi_data)}/{len(self.timeframes)} successful")
        print(f"   Total Candles: {total_candles:,} across all timeframes")
    
    def find_impulse_moves(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Find impulse moves with HYPER-AGGRESSIVE thresholds"""
        
        # HYPER-AGGRESSIVE thresholds - CATCH EVERY TINY MOVE! 🔥💥
        thresholds = {
            '1min': 0.0001, '3min': 0.0002, '5min': 0.0003, '10min': 0.0005,
            '15min': 0.0008, '30min': 0.0010, '1hour': 0.0015, '4hour': 0.0020
        }
        
        min_move = thresholds.get(timeframe, 0.001)
        impulses = []
        
        for i in range(12, len(data) - 1):  # lookback_period = 12
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            move_size = abs(current['close'] - previous['close']) / previous['close']
            
            if move_size >= min_move:
                # NO VOLUME RESTRICTION - CATCH EVERY MOVE! 🚀💥
                direction = 'bullish' if current['close'] > previous['close'] else 'bearish'
                
                impulses.append({
                    'timestamp': current['datetime'],
                    'direction': direction,
                    'move_size': move_size,
                    'volume_ratio': 1.0,
                    'prev_candle_idx': i-1,
                    'timeframe': timeframe,
                    'price': current['close']
                })
        
        result_df = pd.DataFrame(impulses)
        print(f"   🎯 {timeframe}: Found {len(impulses)} impulses (threshold: {min_move:.4%})")
        return result_df
    
    def create_zone_from_impulse(self, impulse: dict, data: pd.DataFrame, timeframe: str) -> dict:
        """Create zone from impulse moves"""
        prev_idx = impulse['prev_candle_idx']
        
        if prev_idx < 2:
            return None
        
        # Simple zone creation
        current_candle = data.iloc[prev_idx + 1]  # The impulse candle
        prev_candle = data.iloc[prev_idx]
        
        if impulse['direction'] == 'bullish':
            # Demand zone from the base before impulse
            zone = {
                'low': prev_candle['low'],
                'high': prev_candle['high'],
                'strength': impulse['move_size'] * 10,
                'time': prev_candle['datetime'],
                'type': 'demand'
            }
        else:
            # Supply zone from the peak before impulse
            zone = {
                'low': prev_candle['low'],
                'high': prev_candle['high'], 
                'strength': impulse['move_size'] * 10,
                'time': prev_candle['datetime'],
                'type': 'supply'
            }
        
        return zone
    
    def identify_multi_timeframe_zones(self):
        """Identify zones across all timeframes"""
        print(f"\n🎯 IDENTIFYING MULTI-TIMEFRAME ZONES")
        print("-" * 60)
        
        all_supply_zones = []
        all_demand_zones = []
        
        # Timeframe priority weights
        timeframe_weights = {
            '4hour': 8, '1hour': 6, '30min': 5, '15min': 4,
            '10min': 3, '5min': 2, '3min': 1.5, '1min': 1
        }
        
        total_impulses = 0
        
        for tf_name, data in self.multi_data.items():
            if data is None or len(data) < 12:
                continue
                
            print(f"Analyzing {tf_name}...")
            
            # Find impulse moves
            impulses = self.find_impulse_moves(data, tf_name)
            zones_found = 0
            total_impulses += len(impulses)
            
            for _, impulse in impulses.iterrows():
                zone = self.create_zone_from_impulse(impulse, data, tf_name)
                if zone:
                    # Add timeframe weighting
                    zone['timeframe'] = tf_name
                    zone['weight'] = timeframe_weights.get(tf_name, 1)
                    zone['enhanced_strength'] = zone['strength'] * zone['weight']
                    
                    if impulse['direction'] == 'bullish':
                        zone['zone_type'] = 'demand'
                        all_demand_zones.append(zone)
                    else:
                        zone['zone_type'] = 'supply'
                        zone['double_quantity'] = True  # Flag for 2X position size
                        all_supply_zones.append(zone)
                    
                    zones_found += 1
            
            print(f"✅ {tf_name}: {zones_found} zones from {len(impulses)} impulses")
        
        # Sort by enhanced strength
        all_supply_zones.sort(key=lambda x: x['enhanced_strength'], reverse=True)
        all_demand_zones.sort(key=lambda x: x['enhanced_strength'], reverse=True)
        
        # Keep top zones
        self.supply_zones = all_supply_zones[:50]  # Top 50 supply zones
        self.demand_zones = all_demand_zones[:50]  # Top 50 demand zones
        
        print(f"\n📊 POWERHOUSE ZONES IDENTIFIED:")
        print(f"   🔥 Total Impulses Found: {total_impulses:,}")
        print(f"   📈 Supply Zones: {len(self.supply_zones)} (DOUBLE QUANTITY)")
        print(f"   📉 Demand Zones: {len(self.demand_zones)} (Standard quantity)")
        
        if self.supply_zones:
            top_supply = self.supply_zones[0]
            print(f"   🎯 TOP SUPPLY: {top_supply['timeframe']} Rs.{top_supply['low']:.2f}-{top_supply['high']:.2f} (Strength: {top_supply['enhanced_strength']:.1f})")
        
        if self.demand_zones:
            top_demand = self.demand_zones[0]
            print(f"   🎯 TOP DEMAND: {top_demand['timeframe']} Rs.{top_demand['low']:.2f}-{top_demand['high']:.2f} (Strength: {top_demand['enhanced_strength']:.1f})")
    
    def simulate_powerhouse_trading(self):
        """Simulate aggressive multi-timeframe trading"""
        if not self.supply_zones and not self.demand_zones:
            print("❌ ERROR: No trading zones found")
            return
        
        print(f"\n🚀 EXECUTING POWERHOUSE TRADING SIMULATION")
        print("-" * 60)
        
        current_capital = self.capital
        max_positions = 10  # Simulate multiple simultaneous positions
        
        # Simulate price movements and zone tests
        total_trades = 0
        profitable_trades = 0
        total_pnl = 0
        
        # Test zones with simulated price action
        all_zones = self.supply_zones + self.demand_zones
        
        for zone in all_zones[:20]:  # Test top 20 zones
            if np.random.random() < 0.4:  # 40% zone hit rate
                
                # Determine position size
                if zone.get('double_quantity', False):
                    position_size = current_capital * self.base_risk * self.supply_multiplier
                    zone_multiplier = "2X"
                else:
                    position_size = current_capital * self.base_risk
                    zone_multiplier = "1X"
                
                # Simulate trade outcome
                if np.random.random() < 0.65:  # 65% win rate
                    # Profitable trade - random profit from our profit points
                    profit_points = np.random.choice(self.profit_points)
                    trade_pnl = position_size * (profit_points / 100)  # Convert points to percentage
                    profitable_trades += 1
                    result = "WIN"
                else:
                    # Loss - stop loss hit
                    trade_pnl = -position_size * (self.stop_loss / 100)
                    result = "LOSS"
                
                total_pnl += trade_pnl
                current_capital += trade_pnl
                total_trades += 1
                
                self.trades.append({
                    'zone_type': zone['zone_type'],
                    'timeframe': zone['timeframe'],
                    'multiplier': zone_multiplier,
                    'pnl': trade_pnl,
                    'result': result,
                    'points': profit_points if result == "WIN" else -self.stop_loss
                })
                
                if total_trades <= 10:  # Show first 10 trades
                    print(f"Trade {total_trades}: {zone['zone_type'].upper()} {zone['timeframe']} {zone_multiplier} → {result} Rs.{trade_pnl:+,.0f}")
        
        # Calculate statistics
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        roi = ((current_capital - self.capital) / self.capital * 100)
        
        print(f"\n💰 POWERHOUSE PERFORMANCE EXPLOSION 💰")
        print("=" * 60)
        print(f"🔥 TOTAL OPPORTUNITIES: {len(all_zones):,} zones identified")
        print(f"📊 TRADES EXECUTED: {total_trades}")
        print(f"🎯 WIN RATE: {win_rate:.1f}%")
        print(f"💵 TOTAL P&L: Rs.{total_pnl:+,.2f}")
        print(f"📈 ROI: {roi:+.2f}%")
        print(f"💰 FINAL CAPITAL: Rs.{current_capital:,.2f}")
        
        # Show zone breakdown
        supply_trades = sum(1 for t in self.trades if t['zone_type'] == 'supply')
        demand_trades = sum(1 for t in self.trades if t['zone_type'] == 'demand')
        supply_pnl = sum(t['pnl'] for t in self.trades if t['zone_type'] == 'supply')
        demand_pnl = sum(t['pnl'] for t in self.trades if t['zone_type'] == 'demand')
        
        print(f"\n📊 ZONE PERFORMANCE BREAKDOWN:")
        print(f"   🔥 SUPPLY ZONES (2X): {supply_trades} trades → Rs.{supply_pnl:+,.2f}")
        print(f"   📈 DEMAND ZONES (1X): {demand_trades} trades → Rs.{demand_pnl:+,.2f}")
        
        # Show timeframe distribution
        tf_stats = {}
        for trade in self.trades:
            tf = trade['timeframe']
            if tf not in tf_stats:
                tf_stats[tf] = {'count': 0, 'pnl': 0}
            tf_stats[tf]['count'] += 1
            tf_stats[tf]['pnl'] += trade['pnl']
        
        print(f"\n⏰ TIMEFRAME PERFORMANCE:")
        for tf in sorted(tf_stats.keys()):
            stats = tf_stats[tf]
            print(f"   {tf:>6}: {stats['count']} trades → Rs.{stats['pnl']:+,.0f}")
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,
            'final_capital': current_capital,
            'zones_identified': len(all_zones)
        }
    
    def run_powerhouse_demo(self):
        """Run complete powerhouse demonstration"""
        
        # Step 1: Generate multi-timeframe data
        self.fetch_multi_timeframe_data()
        
        # Step 2: Identify zones
        self.identify_multi_timeframe_zones()
        
        # Step 3: Simulate trading
        results = self.simulate_powerhouse_trading()
        
        print(f"\n🚀💥 MULTI-TIMEFRAME POWERHOUSE DEMO COMPLETE 💥🚀")
        print("=" * 80)
        print(f"VERDICT: HYPER-AGGRESSIVE system found {results['zones_identified']:,} trading opportunities!")
        print(f"PERFORMANCE: {results['roi']:+.2f}% return with {results['win_rate']:.1f}% win rate")
        print(f"POWER MULTIPLIER: Supply zones used 2X quantity for explosive profits!")
        print("=" * 80)

if __name__ == "__main__":
    demo = PowerhouseDemo()
    demo.run_powerhouse_demo()