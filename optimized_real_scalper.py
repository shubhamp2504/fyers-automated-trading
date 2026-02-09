#!/usr/bin/env python3
"""
🔥 REAL MARKET SCALPING BACKTESTER - OPTIMIZED FOR LIVE DATA! 🔥
================================================================================
Using REAL NIFTY data with REALISTIC zone identification
Optimized for actual market conditions and live price movements!
================================================================================
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
import time

class OptimizedRealScalper:
    def __init__(self):
        print("🔥 REAL MARKET SCALPING BACKTESTER - OPTIMIZED FOR LIVE DATA! 🔥")
        print("=" * 80)
        print("Using REAL NIFTY data with REALISTIC zone identification")
        print("Optimized for actual market conditions and live price movements!")
        print("=" * 80)
        
        # Load API credentials
        self.load_config()
        
        # Initialize Fyers API
        self.initialize_fyers()
        
    def load_config(self):
        """Load real API config"""
        with open('fyers_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.client_id = self.config['fyers']['client_id']
        self.access_token = self.config['fyers']['access_token']
        
        print(f"✅ Using account: {self.client_id}")
    
    def initialize_fyers(self):
        """Initialize Fyers API"""
        self.fyers = fyersModel.FyersModel(
            client_id=self.client_id,
            is_async=False,
            token=self.access_token,
            log_path=""
        )
        
        # Test connection
        profile = self.fyers.get_profile()
        if profile['s'] == 'ok':
            print(f"✅ API Connected: {profile['data']['fy_id']}")
            self.api_working = True
        else:
            print(f"❌ API Error: {profile}")
            self.api_working = False
    
    def get_real_market_data(self):
        """Fetch REAL 2026 NIFTY data"""
        print(f"\n📊 FETCHING REAL 2026 MARKET DATA")
        print("-" * 45)
        
        # Date range for 2026 data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=40)
        
        print(f"   📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"   🎯 Symbol: NSE:NIFTY50-INDEX")
        
        try:
            data = {
                "symbol": "NSE:NIFTY50-INDEX",
                "resolution": "5",  # 5-minute for scalping
                "date_format": "1",
                "range_from": start_date.strftime('%Y-%m-%d'),
                "range_to": end_date.strftime('%Y-%m-%d'),
                "cont_flag": "1"
            }
            
            response = self.fyers.history(data)
            
            if response['s'] == 'ok':
                candles = response['candles']
                print(f"   ✅ Real data fetched: {len(candles)} candles")
                
                # Convert to DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                print(f"   📈 Latest price: Rs.{df['close'].iloc[-1]:.2f}")
                print(f"   📊 Range: Rs.{df['low'].min():.2f} - Rs.{df['high'].max():.2f}")
                
                return df
            else:
                print(f"   ❌ Data fetch failed: {response}")
                return None
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def identify_realistic_zones(self, df):
        """Identify supply/demand zones - REALISTIC for live market"""
        print(f"\n🎯 IDENTIFYING REALISTIC SUPPLY/DEMAND ZONES")
        print("-" * 55)
        
        zones = []
        
        # More realistic parameters for actual market data
        lookback_period = 10  # Shorter lookback for scalping
        min_strength = 1      # Lower minimum strength requirement
        
        print(f"   ⚙️ Lookback period: {lookback_period} candles")
        print(f"   ⚙️ Min strength: {min_strength} rejection")
        
        # Find supply zones (resistance)
        for i in range(lookback_period, len(df) - 5):
            current = df.iloc[i]
            
            # Check for local high
            prev = df.iloc[i-lookback_period:i]
            next_few = df.iloc[i+1:i+6]
            
            # Less strict criteria for real market
            if (current['high'] >= prev['high'].max() * 0.998 and  # 0.2% tolerance
                current['high'] >= next_few['high'].max() * 0.995):  # 0.5% tolerance
                
                # Check for rejections (price coming back down)
                rejections = 0
                for j in range(i+1, min(i+11, len(df))):
                    test_candle = df.iloc[j]
                    
                    # If price traded near the high but closed lower
                    if (test_candle['high'] >= current['high'] * 0.999 and  # Within 0.1%
                        test_candle['close'] < current['high'] * 0.997):      # Closed 0.3% lower
                        rejections += 1
                
                if rejections >= min_strength:
                    zones.append({
                        'type': 'supply',
                        'price': current['high'],
                        'datetime': current['datetime'],
                        'strength': rejections,
                        'index': i
                    })
        
        # Find demand zones (support)
        for i in range(lookback_period, len(df) - 5):
            current = df.iloc[i]
            
            # Check for local low 
            prev = df.iloc[i-lookback_period:i]
            next_few = df.iloc[i+1:i+6]
            
            # Less strict criteria for real market
            if (current['low'] <= prev['low'].min() * 1.002 and   # 0.2% tolerance
                current['low'] <= next_few['low'].min() * 1.005):  # 0.5% tolerance
                
                # Check for bounces (price coming back up)
                bounces = 0
                for j in range(i+1, min(i+11, len(df))):
                    test_candle = df.iloc[j]
                    
                    # If price traded near the low but closed higher
                    if (test_candle['low'] <= current['low'] * 1.001 and   # Within 0.1%
                        test_candle['close'] > current['low'] * 1.003):     # Closed 0.3% higher
                        bounces += 1
                
                if bounces >= min_strength:
                    zones.append({
                        'type': 'demand',
                        'price': current['low'],
                        'datetime': current['datetime'],
                        'strength': bounces,
                        'index': i
                    })
        
        # Remove overlapping zones
        zones = sorted(zones, key=lambda x: x['strength'], reverse=True)
        filtered_zones = []
        
        for zone in zones:
            overlap = False
            for existing in filtered_zones:
                price_diff = abs(zone['price'] - existing['price'])
                if price_diff < 50:  # 50 points minimum gap
                    overlap = True
                    break
            if not overlap:
                filtered_zones.append(zone)
        
        supply_zones = [z for z in filtered_zones if z['type'] == 'supply']
        demand_zones = [z for z in filtered_zones if z['type'] == 'demand']
        
        print(f"   🔥 SUPPLY ZONES: {len(supply_zones)} identified")
        print(f"   📈 DEMAND ZONES: {len(demand_zones)} identified")
        print(f"   ✅ TOTAL ZONES: {len(filtered_zones)} from real data")
        
        # Show top zones
        if supply_zones:
            top_supply = sorted(supply_zones, key=lambda x: x['strength'], reverse=True)[:3]
            print(f"   🏆 Top Supply Zones:")
            for i, zone in enumerate(top_supply):
                print(f"      {i+1}. Rs.{zone['price']:.2f} (strength: {zone['strength']})")
        
        if demand_zones:
            top_demand = sorted(demand_zones, key=lambda x: x['strength'], reverse=True)[:3]
            print(f"   🏆 Top Demand Zones:")
            for i, zone in enumerate(top_demand):
                print(f"      {i+1}. Rs.{zone['price']:.2f} (strength: {zone['strength']})")
        
        return filtered_zones
    
    def execute_realistic_scalping(self, df, zones):
        """Execute scalping with REAL market parameters"""
        print(f"\n⚡ EXECUTING REALISTIC SCALPING STRATEGY")
        print("-" * 50)
        
        trades = []
        capital = 100000
        current_capital = capital
        
        # REALISTIC scalping parameters
        risk_per_trade = 0.01   # 1% risk per trade (realistic)
        target_points = [8, 12, 15, 18, 20]  # Realistic targets for NIFTY
        stop_loss_points = 6    # Tight stops for scalping
        zone_tolerance = 8      # 8 points entry tolerance
        
        print(f"   💰 Capital: Rs.{capital:,.0f}")
        print(f"   🎯 Risk per trade: {risk_per_trade:.1%}")
        print(f"   📊 Targets: {target_points} points")
        print(f"   🛡️ Stop loss: {stop_loss_points} points")
        print(f"   📍 Zone tolerance: {zone_tolerance} points")
        
        # Execute scalping trades
        for i in range(20, len(df) - 1):  # Start after initial period
            current = df.iloc[i]
            next_candle = df.iloc[i + 1]
            
            # Check each zone for scalping opportunities
            for zone in zones:
                if zone['index'] < i - 5:  # Zone must be established
                    
                    # SUPPLY zone scalping (SHORT)
                    if zone['type'] == 'supply':
                        # Price approaching supply zone
                        if (current['high'] >= zone['price'] - zone_tolerance and
                            current['high'] <= zone['price'] + 2 and
                            current['close'] < zone['price']):
                            
                            # Execute SHORT scalp
                            entry_price = zone['price'] - 1
                            target_price = entry_price - np.random.choice(target_points)
                            stop_price = entry_price + stop_loss_points
                            
                            # Position size
                            trade_risk = current_capital * risk_per_trade
                            
                            # Check next candle for outcome
                            if next_candle['low'] <= target_price:
                                # Target hit - WIN
                                points_gained = entry_price - target_price
                                trade_pnl = trade_risk * (points_gained / stop_loss_points)
                                result = "WIN ⚡"
                            elif next_candle['high'] >= stop_price:
                                # Stop hit - LOSS
                                trade_pnl = -trade_risk
                                points_gained = -stop_loss_points
                                result = "LOSS"
                            else:
                                continue  # No clear outcome
                            
                            current_capital += trade_pnl
                            
                            trades.append({
                                'datetime': current['datetime'],
                                'type': 'supply_scalp',
                                'zone_price': zone['price'],
                                'entry_price': entry_price,
                                'target': target_price,
                                'stop': stop_price,
                                'points': points_gained,
                                'pnl': trade_pnl,
                                'result': result,
                                'capital': current_capital,
                                'zone_strength': zone['strength']
                            })
                            
                            break  # One trade per candle
                    
                    # DEMAND zone scalping (LONG)
                    elif zone['type'] == 'demand':
                        # Price approaching demand zone
                        if (current['low'] <= zone['price'] + zone_tolerance and
                            current['low'] >= zone['price'] - 2 and
                            current['close'] > zone['price']):
                            
                            # Execute LONG scalp
                            entry_price = zone['price'] + 1
                            target_price = entry_price + np.random.choice(target_points)
                            stop_price = entry_price - stop_loss_points
                            
                            # Position size
                            trade_risk = current_capital * risk_per_trade
                            
                            # Check next candle for outcome
                            if next_candle['high'] >= target_price:
                                # Target hit - WIN
                                points_gained = target_price - entry_price
                                trade_pnl = trade_risk * (points_gained / stop_loss_points)
                                result = "WIN ⚡"
                            elif next_candle['low'] <= stop_price:
                                # Stop hit - LOSS
                                trade_pnl = -trade_risk
                                points_gained = -stop_loss_points
                                result = "LOSS"
                            else:
                                continue  # No clear outcome
                            
                            current_capital += trade_pnl
                            
                            trades.append({
                                'datetime': current['datetime'],
                                'type': 'demand_scalp',
                                'zone_price': zone['price'],
                                'entry_price': entry_price,
                                'target': target_price,
                                'stop': stop_price,
                                'points': points_gained,
                                'pnl': trade_pnl,
                                'result': result,
                                'capital': current_capital,
                                'zone_strength': zone['strength']
                            })
                            
                            break  # One trade per candle
        
        return trades, current_capital
    
    def generate_real_scalping_report(self, trades, final_capital, df):
        """Generate REAL data scalping performance report"""
        print(f"\n🔥💰 REAL 2026 NIFTY SCALPING RESULTS 💰🔥")
        print("=" * 80)
        
        if not trades:
            print("⚠️ NO SCALPING OPPORTUNITIES FOUND")
            print("💡 Market may be trending without clear zones")
            return
        
        # Calculate metrics
        initial_capital = 100000
        total_pnl = final_capital - initial_capital
        num_trades = len(trades)
        wins = len([t for t in trades if 'WIN' in t['result']])
        losses = num_trades - wins
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
        roi = (total_pnl / initial_capital * 100)
        
        # Period info
        start_date = df['datetime'].iloc[0].strftime('%Y-%m-%d')
        end_date = df['datetime'].iloc[-1].strftime('%Y-%m-%d')
        
        print(f"📊 REAL DATA SCALPING:")
        print(f"   🔌 Source: LIVE FYERS API")
        print(f"   📅 Period: {start_date} to {end_date}")
        print(f"   📈 Candles: {len(df)} (5-minute NIFTY)")
        print(f"   💯 Authenticity: 100% REAL MARKET DATA")
        
        print(f"\n⚡ SCALPING PERFORMANCE:")
        print(f"   💰 Starting Capital:     Rs.{initial_capital:10,.0f}")
        print(f"   🎯 Final Capital:        Rs.{final_capital:10,.0f}")
        print(f"   🚀 Total P&L:            Rs.{total_pnl:+9,.0f}")
        print(f"   📈 ROI:                  {roi:+8.1f}%")
        print(f"   ⚡ Total Scalps:         {num_trades:10d}")
        print(f"   🏆 Win Rate:             {win_rate:9.1f}%")
        print(f"   ✅ Winning Scalps:       {wins:10d}")
        print(f"   ❌ Losing Scalps:        {losses:10d}")
        
        # Show trades
        print(f"\n📋 REAL SCALPING TRADES:")
        for i, trade in enumerate(trades[:8]):
            time_str = trade['datetime'].strftime('%m-%d %H:%M')
            zone_type = trade['type'].replace('_scalp', '').upper()
            print(f"   {i+1}. {time_str} {zone_type:6} Rs.{trade['zone_price']:7.0f} → {trade['points']:+3.0f}pts Rs.{trade['pnl']:+6,.0f} {trade['result']}")
        
        if len(trades) > 8:
            print(f"   ... (and {len(trades)-8} more real trades)")
        
        # Zone performance
        supply_trades = [t for t in trades if t['type'] == 'supply_scalp']
        demand_trades = [t for t in trades if t['type'] == 'demand_scalp']
        
        if supply_trades:
            supply_pnl = sum(t['pnl'] for t in supply_trades)
            print(f"\n🎯 ZONE PERFORMANCE:")
            print(f"   🔥 Supply scalps: {len(supply_trades)} → Rs.{supply_pnl:+7,.0f}")
        
        if demand_trades:
            demand_pnl = sum(t['pnl'] for t in demand_trades)
            print(f"   📈 Demand scalps: {len(demand_trades)} → Rs.{demand_pnl:+7,.0f}")
        
        print("\n" + "=" * 80)
        
        # Final verdict
        if roi > 15:
            print("🚀🔥 SCALPING SUCCESS: Real NIFTY data produces profitable results! 🔥🚀")
        elif roi > 5:
            print("✅ SCALPING VIABLE: Positive returns with actual market conditions!")
        elif roi > 0:
            print("📈 MODEST SCALPING: Small profit with real market friction")
        else:
            print("⚠️ SCALPING CHALLENGE: Real markets are tougher than simulations")
        
        print(f"💯 VERIFIED: {num_trades} scalps on 100% REAL 2026 NIFTY DATA!")
        print(f"🎯 Period: {(df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days} days of live market")
        
        return {
            'total_pnl': total_pnl,
            'roi': roi,
            'trades': num_trades,
            'win_rate': win_rate
        }
    
    def run_real_market_scalping(self):
        """Complete real market scalping backtest"""
        if not self.api_working:
            print("❌ API not working - cannot proceed")
            return
        
        # Get real market data  
        df = self.get_real_market_data()
        if df is None:
            return
        
        # Identify realistic zones
        zones = self.identify_realistic_zones(df)
        
        # Execute scalping
        trades, final_capital = self.execute_realistic_scalping(df, zones)
        
        # Generate report
        results = self.generate_real_scalping_report(trades, final_capital, df)
        
        return results

if __name__ == "__main__":
    try:
        scalper = OptimizedRealScalper()
        scalper.run_real_market_scalping()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("🔧 Check API credentials and connection")