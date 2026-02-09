#!/usr/bin/env python3
"""
🚀💥 EXPLOSIVE MONEY MACHINE - REAL DATA VERSION 💥🚀
================================================================================
TESTING EXPLOSIVE PROFITS WITH ACTUAL FYERS API DATA
NO SIMULATION - REAL HISTORICAL MARKET MOVEMENTS ONLY!
================================================================================
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fyers_client import FyersClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class RealDataExplosiveMachine:
    def __init__(self):
        print("🚀💥 EXPLOSIVE MONEY MACHINE - REAL DATA VERSION 💥🚀")
        print("=" * 80)
        print("TESTING EXPLOSIVE PROFITS WITH ACTUAL FYERS API DATA")
        print("NO SIMULATION - REAL HISTORICAL MARKET MOVEMENTS ONLY!")
        print("=" * 80)
        
        # Initialize Fyers client
        self.client = FyersClient()
        print("Initializing Fyers API Client...")
        print(f"   Client ID: {self.client.client_id[:10]}...")
        print("   Using LIVE FYERS API with REAL DATA")
        
        # Test connection
        try:
            profile = self.client.get_profile()
            if profile:
                print(f"✅ Profile retrieved successfully")
                print(f"   📝 Name: {profile.get('name', 'N/A')}")
                print(f"   📧 Email: {profile.get('email_id', 'N/A')}")
                print(f"Connected to REAL Fyers account")
                self.connected = True
            else:
                print("❌ Error getting profile: Your token has expired. Please generate a token")
                print("Connected but unable to fetch profile")
                self.connected = False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            self.connected = False
        
        # EXPLOSIVE SETTINGS (same as simulated version)
        self.capital = 100000
        self.aggressive_risk = 0.05  # 5% per trade
        self.supply_multiplier = 4.0  # 4X on supply zones
        self.demand_multiplier = 2.0  # 2X on demand zones
        self.profit_targets = [15, 25, 35, 50, 75, 100, 150, 200]
        self.stop_loss = 8
        
        print(f"\n🔥 EXPLOSIVE CONFIGURATION:")
        print(f"💰 Capital: Rs.{self.capital:,.2f}")
        print(f"⚡ Risk per Trade: {self.aggressive_risk:.0%}")
        print(f"🎯 Supply Multiplier: {self.supply_multiplier}X")
        print(f"📈 Demand Multiplier: {self.demand_multiplier}X")
        print(f"💥 Profit Targets: {self.profit_targets}")
    
    def fetch_real_market_data(self, start_date: str, end_date: str):
        """Fetch REAL historical data from Fyers API"""
        print(f"\n📊 FETCHING REAL MARKET DATA FROM FYERS API")
        print("-" * 60)
        print(f"SYMBOL: NSE:NIFTY50-INDEX")
        print(f"PERIOD: {start_date} to {end_date}")
        print(f"RESOLUTION: 5 minutes (for detailed analysis)")
        
        if not self.connected:
            print("❌ No API connection - cannot fetch real data")
            return None
        
        try:
            # Fetch real historical data
            print("Fetching REAL historical data from Fyers API:")
            print(f"   Symbol: NSE:NIFTY50-INDEX | Resolution: 5")
            print(f"   Period: {start_date} to {end_date}")
            
            data = self.client.get_historical_data(
                symbol="NSE:NIFTY50-INDEX",
                resolution="5",
                from_date=start_date,
                to_date=end_date
            )
            
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                df['datetime'] = pd.to_datetime(df['datetime'])
                
                print(f"✅ Real data retrieved: {len(df)} candles")
                print(f"📊 Price Range: Rs.{df['low'].min():.2f} - Rs.{df['high'].max():.2f}")
                print(f"📅 Date Range: {df['datetime'].min()} to {df['datetime'].max()}")
                
                # Calculate daily volatility from real data
                df['daily_return'] = df['close'].pct_change()
                avg_volatility = df['daily_return'].std() * 100
                print(f"📈 Average Volatility: {avg_volatility:.2f}%")
                
                return df
            else:
                print("❌ No data received from API")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching real data: {e}")
            return None
    
    def identify_real_supply_demand_zones(self, real_data: pd.DataFrame):
        """Identify supply/demand zones from REAL market data"""
        print(f"\n🎯 ANALYZING REAL DATA FOR SUPPLY & DEMAND ZONES")
        print("-" * 60)
        
        if real_data is None or len(real_data) < 50:
            print("❌ Insufficient real data for analysis")
            return [], []
        
        supply_zones = []
        demand_zones = []
        
        # Calculate technical indicators on REAL data
        real_data['volume_sma'] = real_data['volume'].rolling(20).mean()
        real_data['price_change'] = real_data['close'].pct_change()
        real_data['volatility'] = real_data['price_change'].rolling(10).std()
        
        print(f"📊 Analyzing {len(real_data)} real market candles...")
        
        # Identify zones from REAL price action
        for i in range(20, len(real_data) - 5):
            current = real_data.iloc[i]
            
            # Real move size
            move_size = abs(current['price_change'])
            
            # Real volume analysis
            volume_ratio = current['volume'] / current['volume_sma'] if current['volume_sma'] > 0 else 1
            
            # AGGRESSIVE zone detection (lower thresholds for more opportunities)
            if move_size > 0.008 and volume_ratio > 1.3:  # 0.8% moves with volume
                
                # Determine zone type based on REAL price action
                if current['price_change'] > 0.008:  # Strong up move
                    zone_type = 'supply'
                    zone_strength = move_size * volume_ratio
                elif current['price_change'] < -0.008:  # Strong down move
                    zone_type = 'demand'
                    zone_strength = move_size * volume_ratio
                else:
                    continue
                
                # Create zone from REAL data
                zone = {
                    'type': zone_type,
                    'datetime': current['datetime'],
                    'low': min(current['low'], current['open']),
                    'high': max(current['high'], current['close']),
                    'strength': zone_strength,
                    'volume_ratio': volume_ratio,
                    'move_size': move_size,
                    'price_level': current['close']
                }
                
                if zone_type == 'supply':
                    supply_zones.append(zone)
                else:
                    demand_zones.append(zone)
        
        # Sort by strength
        supply_zones.sort(key=lambda x: x['strength'], reverse=True)
        demand_zones.sort(key=lambda x: x['strength'], reverse=True)
        
        print(f"🔥 REAL DATA ZONE ANALYSIS COMPLETE:")
        print(f"   📈 Supply Zones Found: {len(supply_zones)}")
        print(f"   📉 Demand Zones Found: {len(demand_zones)}")
        
        if supply_zones:
            top_supply = supply_zones[0]
            print(f"   🎯 Top Supply Zone: Rs.{top_supply['low']:.2f}-{top_supply['high']:.2f} (Strength: {top_supply['strength']:.3f})")
        
        if demand_zones:
            top_demand = demand_zones[0]
            print(f"   🎯 Top Demand Zone: Rs.{top_demand['low']:.2f}-{top_demand['high']:.2f} (Strength: {top_demand['strength']:.3f})")
        
        return supply_zones[:20], demand_zones[:20]  # Top 20 each
    
    def simulate_explosive_trading_on_real_data(self, real_data: pd.DataFrame, supply_zones: list, demand_zones: list):
        """Simulate explosive trading strategy on REAL market movements"""
        print(f"\n💥 SIMULATING EXPLOSIVE TRADING ON REAL DATA")
        print("-" * 60)
        
        if not supply_zones and not demand_zones:
            print("❌ No zones found - cannot simulate trading")
            return []
        
        trades = []
        current_capital = self.capital
        
        all_zones = supply_zones + demand_zones
        print(f"🎯 Total Zones to Test: {len(all_zones)}")
        
        # Test each zone against subsequent REAL price action
        for zone in all_zones:
            zone_time = zone['datetime']
            
            # Find data points after this zone was formed
            future_data = real_data[real_data['datetime'] > zone_time]
            
            if len(future_data) < 5:
                continue
            
            # Check if price hits the zone in future REAL data
            zone_hit = False
            entry_price = None
            
            for _, candle in future_data.head(50).iterrows():  # Look ahead 50 candles
                if zone['low'] <= candle['close'] <= zone['high']:
                    zone_hit = True
                    entry_price = candle['close']
                    entry_time = candle['datetime']
                    break
            
            if zone_hit:
                # Calculate position size based on EXPLOSIVE settings
                base_risk = current_capital * self.aggressive_risk
                
                if zone['type'] == 'supply':
                    position_risk = base_risk * self.supply_multiplier * zone['strength']
                    multiplier = f"{self.supply_multiplier}X"
                else:
                    position_risk = base_risk * self.demand_multiplier * zone['strength']
                    multiplier = f"{self.demand_multiplier}X"
                
                # Simulate trade outcome based on REAL subsequent price action
                trade_data = future_data[future_data['datetime'] > entry_time]
                
                if len(trade_data) > 0:
                    # Look for profit targets or stop loss in REAL data
                    profit_hit = False
                    loss_hit = False
                    
                    for _, price_candle in trade_data.head(20).iterrows():
                        # Calculate points move from entry
                        if zone['type'] == 'supply':
                            # Short position - profit when price goes down
                            points_move = (entry_price - price_candle['low']) / entry_price * 100
                        else:
                            # Long position - profit when price goes up
                            points_move = (price_candle['high'] - entry_price) / entry_price * 100
                        
                        # Check profit targets
                        for target in self.profit_targets:
                            if points_move >= target:
                                trade_pnl = position_risk * (target / 100)
                                profit_hit = True
                                break
                        
                        # Check stop loss
                        if points_move <= -self.stop_loss:
                            trade_pnl = -position_risk * (self.stop_loss / 100)
                            loss_hit = True
                            break
                        
                        if profit_hit or loss_hit:
                            break
                    
                    # Record trade if outcome determined
                    if profit_hit or loss_hit:
                        current_capital += trade_pnl
                        
                        trade = {
                            'zone_datetime': zone_time,
                            'entry_datetime': entry_time,
                            'zone_type': zone['type'],
                            'multiplier': multiplier,
                            'entry_price': entry_price,
                            'pnl': trade_pnl,
                            'result': "WIN 🚀" if profit_hit else "LOSS",
                            'points': points_move if profit_hit else -self.stop_loss,
                            'strength': zone['strength'],
                            'capital_after': current_capital
                        }
                        
                        trades.append(trade)
        
        # Calculate performance on REAL data
        total_pnl = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if 'WIN' in t['result'])
        win_rate = (wins / len(trades) * 100) if trades else 0
        roi = ((current_capital - self.capital) / self.capital * 100)
        
        print(f"💰 REAL DATA TRADING RESULTS:")
        print(f"   📊 Total Trades: {len(trades)}")
        print(f"   🎯 Win Rate: {win_rate:.1f}%")
        print(f"   💵 Total P&L: Rs.{total_pnl:+,.2f}")
        print(f"   📈 ROI: {roi:+.1f}%")
        print(f"   💰 Final Capital: Rs.{current_capital:,.2f}")
        
        return trades
    
    def generate_real_data_report(self, trades: list):
        """Generate performance report based on REAL data results"""
        print(f"\n🚀💰 EXPLOSIVE MACHINE - REAL DATA PERFORMANCE REPORT 💰🚀")
        print("=" * 80)
        
        if not trades:
            print("❌ NO TRADES EXECUTED ON REAL DATA")
            print("💡 This could mean:")
            print("   - Market was too stable (no strong moves)")
            print("   - Zone criteria too strict")
            print("   - Limited data period")
            print("   - Market conditions not suitable for aggressive strategy")
            return
        
        # Calculate metrics
        total_pnl = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if 'WIN' in t['result'])
        losses = len(trades) - wins
        win_rate = (wins / len(trades) * 100)
        
        final_capital = self.capital + total_pnl
        roi = (total_pnl / self.capital * 100)
        
        # Zone breakdown
        supply_trades = [t for t in trades if t['zone_type'] == 'supply']
        demand_trades = [t for t in trades if t['zone_type'] == 'demand']
        supply_pnl = sum(t['pnl'] for t in supply_trades)
        demand_pnl = sum(t['pnl'] for t in demand_trades)
        
        print(f"📊 REAL DATA PERFORMANCE SUMMARY:")
        print(f"💰 STARTING CAPITAL:    Rs.{self.capital:10,.2f}")
        print(f"🎯 FINAL CAPITAL:       Rs.{final_capital:10,.2f}")
        print(f"📈 TOTAL P&L:           Rs.{total_pnl:+9,.2f}")
        print(f"🚀 ROI (REAL DATA):     {roi:+8.1f}%")
        print(f"📋 TOTAL TRADES:        {len(trades):10,d}")
        print(f"🏆 WIN RATE:            {win_rate:9.1f}%")
        print(f"✅ WINNING TRADES:      {wins:10,d}")
        print(f"❌ LOSING TRADES:       {losses:10,d}")
        
        print(f"\n🎯 ZONE TYPE PERFORMANCE (REAL DATA):")
        print(f"   🔥 SUPPLY ZONES ({self.supply_multiplier}X): {len(supply_trades):2d} trades → Rs.{supply_pnl:+8,.0f}")
        print(f"   📈 DEMAND ZONES ({self.demand_multiplier}X): {len(demand_trades):2d} trades → Rs.{demand_pnl:+8,.0f}")
        
        # Show sample trades
        print(f"\n📋 SAMPLE REAL DATA TRADES:")
        for i, trade in enumerate(trades[:5]):
            result_emoji = "🚀" if "WIN" in trade['result'] else "❌"
            print(f"   {i+1}. {trade['zone_type'].upper()} {trade['multiplier']} → {trade['result']} Rs.{trade['pnl']:+6,.0f} {result_emoji}")
        
        print("=" * 80)
        
        # Verdict
        if roi > 20:
            print("🎯 EXPLOSIVE SUCCESS: REAL DATA CONFIRMS EXPLOSIVE PROFITS! 🚀💰")
        elif roi > 5:
            print("✅ SOLID PERFORMANCE: REAL DATA SHOWS GOOD RETURNS!")
        elif roi > 0:
            print("📈 MODEST GAINS: REAL DATA SHOWS POSITIVE RESULTS")
        else:
            print("⚠️ REAL DATA REALITY CHECK: Market conditions challenging")
            print("💡 Consider adjusting strategy for current market conditions")
        
        print("🔍 REAL DATA VALIDATION COMPLETE - NO SIMULATION USED!")
        print("=" * 80)
    
    def run_real_data_explosive_test(self):
        """Run complete explosive test on REAL data"""
        
        # Step 1: Fetch real data
        real_data = self.fetch_real_market_data("2026-01-01", "2026-02-08")
        
        if real_data is None:
            print("❌ Cannot proceed without real data")
            print("🔧 Please ensure:")
            print("   1. Valid Fyers API token")
            print("   2. Internet connection")
            print("   3. Valid date range")
            return
        
        # Step 2: Identify zones from real data
        supply_zones, demand_zones = self.identify_real_supply_demand_zones(real_data)
        
        # Step 3: Simulate trading on real data
        trades = self.simulate_explosive_trading_on_real_data(real_data, supply_zones, demand_zones)
        
        # Step 4: Generate real data report
        self.generate_real_data_report(trades)

if __name__ == "__main__":
    machine = RealDataExplosiveMachine()
    machine.run_real_data_explosive_test()