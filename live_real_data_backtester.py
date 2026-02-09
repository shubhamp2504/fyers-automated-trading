#!/usr/bin/env python3
"""
🔥 LIVE FYERS API BACKTESTING - REAL 2026 DATA 🔥
================================================================================
Using REAL Fyers API credentials from config
Fetching ACTUAL market data for genuine backtesting
NO MORE DUMMY DATA - REAL SCALPING WITH REAL PRICES!
================================================================================
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
import requests
import time

class LiveFyersBacktester:
    def __init__(self):
        print("🔥 LIVE FYERS API BACKTESTING - REAL 2026 DATA 🔥")
        print("=" * 80)
        print("Using REAL Fyers API credentials from config")
        print("Fetching ACTUAL market data for genuine backtesting")  
        print("NO MORE DUMMY DATA - REAL SCALPING WITH REAL PRICES!")
        print("=" * 80)
        
        # Load REAL API credentials
        self.load_real_config()
        
        # Initialize Fyers API
        self.initialize_fyers_api()
        
        # Test API connection
        self.test_api_connection()
    
    def load_real_config(self):
        """Load actual Fyers API credentials"""
        print(f"🔌 LOADING REAL API CREDENTIALS")
        print("-" * 40)
        
        try:
            with open('fyers_config.json', 'r') as f:
                self.config = json.load(f)
            
            self.client_id = self.config['fyers']['client_id']
            self.access_token = self.config['fyers']['access_token']
            
            print(f"   ✅ Client ID: {self.client_id}")
            print(f"   ✅ Token loaded: {len(self.access_token)} characters")
            print(f"   ✅ Config loaded successfully")
            
        except Exception as e:
            print(f"   ❌ ERROR loading config: {e}")
            raise
    
    def initialize_fyers_api(self):
        """Initialize Fyers API with real credentials"""
        print(f"\n📡 INITIALIZING FYERS API CONNECTION")
        print("-" * 40)
        
        try:
            self.fyers = fyersModel.FyersModel(
                client_id=self.client_id,
                is_async=False,
                token=self.access_token,
                log_path=""
            )
            
            print(f"   ✅ API model initialized")
            print(f"   ✅ Using live credentials")
            
        except Exception as e:
            print(f"   ❌ ERROR initializing API: {e}")
            raise
    
    def test_api_connection(self):
        """Test live API connection"""
        print(f"\n🔍 TESTING LIVE API CONNECTION")
        print("-" * 40)
        
        try:
            # Test profile endpoint
            profile = self.fyers.get_profile()
            
            if profile['s'] == 'ok':
                print(f"   ✅ API Connection: SUCCESS")
                print(f"   ✅ User: {profile['data'].get('display_name', 'Not set')}")
                print(f"   ✅ Account: {profile['data']['fy_id']}")
                self.api_working = True
            else:
                print(f"   ❌ API Connection failed: {profile}")
                self.api_working = False
                
        except Exception as e:
            print(f"   ❌ Connection error: {e}")
            self.api_working = False
    
    def get_real_nifty_data(self, days=40):
        """Fetch REAL NIFTY data for 2026"""
        print(f"\n📊 FETCHING REAL NIFTY DATA FOR 2026")
        print("-" * 50)
        
        if not self.api_working:
            print("   ❌ API not working - cannot fetch real data")
            return None
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"   📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"   🎯 Symbol: NSE:NIFTY50-INDEX")
        
        try:
            # Fetch historical data
            data = {
                "symbol": "NSE:NIFTY50-INDEX",
                "resolution": "5",  # 5-minute candles for scalping
                "date_format": "1",
                "range_from": start_date.strftime('%Y-%m-%d'),
                "range_to": end_date.strftime('%Y-%m-%d'),
                "cont_flag": "1"
            }
            
            response = self.fyers.history(data)
            
            if response['s'] == 'ok':
                candles = response['candles']
                print(f"   ✅ Data fetched: {len(candles)} candles")
                print(f"   📊 Timeframe: 5-minute")
                print(f"   🔥 REAL MARKET DATA AVAILABLE!")
                
                # Convert to DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                print(f"   📈 Latest price: {df['close'].iloc[-1]:.2f}")
                print(f"   📊 Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
                
                return df
            else:
                print(f"   ❌ Data fetch failed: {response}")
                return None
                
        except Exception as e:
            print(f"   ❌ Error fetching data: {e}")
            return None
    
    def identify_real_supply_demand_zones(self, df):
        """Identify supply/demand zones from REAL data"""
        print(f"\n🎯 IDENTIFYING SUPPLY/DEMAND ZONES FROM REAL DATA")
        print("-" * 55)
        
        zones = []
        
        # Look for supply zones (resistance levels)
        for i in range(20, len(df)-20):
            current = df.iloc[i]
            
            # Check if this is a significant high
            prev_candles = df.iloc[i-20:i]
            next_candles = df.iloc[i+1:i+21]
            
            if (current['high'] > prev_candles['high'].max() and 
                current['high'] > next_candles['high'].max()):
                
                # Validate supply zone strength
                rejection_count = 0
                for j in range(i+1, min(i+21, len(df))):
                    if df.iloc[j]['high'] >= current['high'] * 0.998:  # Touched zone
                        if df.iloc[j]['close'] < current['high'] * 0.995:  # Rejected
                            rejection_count += 1
                
                if rejection_count >= 2:  # Strong supply zone
                    zones.append({
                        'type': 'supply',
                        'price': current['high'],
                        'datetime': current['datetime'],
                        'strength': rejection_count,
                        'candle_index': i
                    })
        
        # Look for demand zones (support levels)
        for i in range(20, len(df)-20):
            current = df.iloc[i]
            
            # Check if this is a significant low
            prev_candles = df.iloc[i-20:i]
            next_candles = df.iloc[i+1:i+21]
            
            if (current['low'] < prev_candles['low'].min() and 
                current['low'] < next_candles['low'].min()):
                
                # Validate demand zone strength
                bounce_count = 0
                for j in range(i+1, min(i+21, len(df))):
                    if df.iloc[j]['low'] <= current['low'] * 1.002:  # Touched zone
                        if df.iloc[j]['close'] > current['low'] * 1.005:  # Bounced
                            bounce_count += 1
                
                if bounce_count >= 2:  # Strong demand zone
                    zones.append({
                        'type': 'demand',
                        'price': current['low'],
                        'datetime': current['datetime'],
                        'strength': bounce_count,
                        'candle_index': i
                    })
        
        supply_zones = [z for z in zones if z['type'] == 'supply']
        demand_zones = [z for z in zones if z['type'] == 'demand']
        
        print(f"   🔥 SUPPLY ZONES: {len(supply_zones)} identified")
        print(f"   📈 DEMAND ZONES: {len(demand_zones)} identified") 
        print(f"   ✅ TOTAL ZONES: {len(zones)} from REAL data")
        
        # Show strongest zones
        if supply_zones:
            strongest_supply = max(supply_zones, key=lambda x: x['strength'])
            print(f"   🏆 Strongest Supply: {strongest_supply['price']:.2f} (strength: {strongest_supply['strength']})")
        
        if demand_zones:
            strongest_demand = max(demand_zones, key=lambda x: x['strength'])
            print(f"   🏆 Strongest Demand: {strongest_demand['price']:.2f} (strength: {strongest_demand['strength']})")
        
        return zones
    
    def execute_real_scalping_strategy(self, df, zones):
        """Execute scalping strategy on real data"""
        print(f"\n⚡ EXECUTING SCALPING STRATEGY ON REAL DATA")
        print("-" * 50)
        
        trades = []
        capital = 100000
        current_capital = capital
        
        zone_tolerance = 5  # 5 points tolerance for zone entry
        
        # Scalping parameters (more conservative for real data)
        risk_per_trade = 0.015  # 1.5% risk per trade
        target_points = [10, 15, 18, 22, 25]  # Conservative targets
        stop_loss_points = 8  # Tight stop loss
        
        print(f"   💰 Starting capital: Rs.{capital:,.0f}")
        print(f"   🎯 Risk per trade: {risk_per_trade:.1%}")
        print(f"   🎯 Target points: {target_points}")
        print(f"   🛡️ Stop loss: {stop_loss_points} points")
        
        # Process each candle for scalping opportunities
        for i in range(100, len(df)-1):  # Start after zone identification period
            current_candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Check for supply zone scalping opportunities
            for zone in zones:
                if zone['type'] == 'supply' and zone['candle_index'] < i:
                    # Price approaching supply zone
                    if (abs(current_candle['high'] - zone['price']) <= zone_tolerance and
                        current_candle['close'] < zone['price']):
                        
                        # Execute short scalp
                        entry_price = zone['price'] - 2  # Enter slightly below zone
                        target_price = entry_price - np.random.choice(target_points)
                        stop_price = entry_price + stop_loss_points
                        
                        # Position sizing
                        trade_risk = current_capital * risk_per_trade
                        
                        # Simulate trade outcome (using next candle)
                        if next_candle['low'] <= target_price:
                            # Target hit
                            points_gained = abs(entry_price - target_price)
                            trade_pnl = trade_risk * (points_gained / stop_loss_points)
                            result = "WIN ⚡"
                        elif next_candle['high'] >= stop_price:
                            # Stop loss hit
                            trade_pnl = -trade_risk
                            points_gained = -stop_loss_points
                            result = "LOSS"
                        else:
                            continue  # No clear outcome
                        
                        current_capital += trade_pnl
                        
                        trade = {
                            'datetime': current_candle['datetime'],
                            'type': 'supply_scalp',
                            'entry_price': entry_price,
                            'target_price': target_price,
                            'stop_price': stop_price,
                            'points': points_gained,
                            'pnl': trade_pnl,
                            'result': result,
                            'capital_after': current_capital,
                            'zone_strength': zone['strength']
                        }
                        trades.append(trade)
                        
                        break  # One trade per candle max
                
                elif zone['type'] == 'demand' and zone['candle_index'] < i:
                    # Price approaching demand zone
                    if (abs(current_candle['low'] - zone['price']) <= zone_tolerance and
                        current_candle['close'] > zone['price']):
                        
                        # Execute long scalp
                        entry_price = zone['price'] + 2  # Enter slightly above zone
                        target_price = entry_price + np.random.choice(target_points)
                        stop_price = entry_price - stop_loss_points
                        
                        # Position sizing
                        trade_risk = current_capital * risk_per_trade
                        
                        # Simulate trade outcome
                        if next_candle['high'] >= target_price:
                            # Target hit
                            points_gained = abs(target_price - entry_price)
                            trade_pnl = trade_risk * (points_gained / stop_loss_points)
                            result = "WIN ⚡"
                        elif next_candle['low'] <= stop_price:
                            # Stop loss hit
                            trade_pnl = -trade_risk
                            points_gained = -stop_loss_points
                            result = "LOSS"
                        else:
                            continue  # No clear outcome
                        
                        current_capital += trade_pnl
                        
                        trade = {
                            'datetime': current_candle['datetime'],
                            'type': 'demand_scalp',
                            'entry_price': entry_price,
                            'target_price': target_price,
                            'stop_price': stop_price,
                            'points': points_gained,
                            'pnl': trade_pnl,
                            'result': result,
                            'capital_after': current_capital,
                            'zone_strength': zone['strength']
                        }
                        trades.append(trade)
                        
                        break  # One trade per candle max
        
        return trades, current_capital
    
    def generate_real_data_report(self, trades, final_capital, df):
        """Generate performance report with REAL data"""
        print(f"\n🔥📊 REAL DATA SCALPING RESULTS - 2026 📊🔥")
        print("=" * 80)
        
        if not trades:
            print("❌ NO TRADES EXECUTED - Check zone identification or entry criteria")
            return
        
        # Calculate metrics
        total_pnl = final_capital - 100000
        num_trades = len(trades)
        wins = len([t for t in trades if 'WIN' in t['result']])
        losses = num_trades - wins
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
        roi = (total_pnl / 100000 * 100)
        
        # Time period
        start_date = df['datetime'].iloc[0].strftime('%Y-%m-%d')
        end_date = df['datetime'].iloc[-1].strftime('%Y-%m-%d')
        
        print(f"📊 REAL DATA SOURCE:")
        print(f"   🔌 API: LIVE FYERS CONNECTION")
        print(f"   📅 Period: {start_date} to {end_date}")
        print(f"   📈 Candles: {len(df)} (5-minute)")
        print(f"   💯 Data: 100% REAL MARKET PRICES")
        
        print(f"\n⚡ SCALPING PERFORMANCE:")
        print(f"   💰 Starting Capital:     Rs.{100000:10,.0f}")
        print(f"   🎯 Final Capital:        Rs.{final_capital:10,.0f}")
        print(f"   🚀 Total P&L:            Rs.{total_pnl:+9,.0f}")
        print(f"   📈 ROI:                  {roi:+8.1f}%")
        print(f"   ⚡ Total Scalps:         {num_trades:10d}")
        print(f"   🏆 Win Rate:             {win_rate:9.1f}%")
        print(f"   ✅ Winning Scalps:       {wins:10d}")
        print(f"   ❌ Losing Scalps:        {losses:10d}")
        
        # Show sample trades
        print(f"\n📋 SAMPLE REAL TRADES:")
        for i, trade in enumerate(trades[:5]):
            time_str = trade['datetime'].strftime('%m-%d %H:%M')
            print(f"   {i+1}. {time_str} {trade['type']:12} → {trade['points']:+3.0f}pts Rs.{trade['pnl']:+6,.0f} {trade['result']}")
        
        if len(trades) > 5:
            print(f"   ... (and {len(trades)-5} more real trades)")
        
        # Zone breakdown
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
        if roi > 20:
            print("🔥🚀 REAL DATA SUCCESS: Scalping strategy works with LIVE market data! 🚀🔥")
        elif roi > 5:
            print("✅ REAL DATA POSITIVE: Strategy shows promise with actual prices!")
        elif roi > 0:
            print("📈 REAL DATA BREAK-EVEN: Small profit with real market conditions")
        else:
            print("⚠️ REAL DATA CHALLENGE: Strategy needs optimization for live trading")
        
        print(f"💯 VERIFIED: {num_trades} trades executed with 100% REAL FYERS DATA!")
        
        return {
            'total_pnl': total_pnl,
            'roi': roi,
            'total_trades': num_trades,
            'win_rate': win_rate
        }
    
    def run_live_data_backtest(self):
        """Run complete backtest with live Fyers data"""
        print(f"🔥 STARTING LIVE DATA BACKTESTING")
        print(f"🎯 Using YOUR real Fyers API credentials")
        print(f"📊 Fetching actual 2026 market data")
        
        # Step 1: Get real market data
        df = self.get_real_nifty_data(days=40)
        if df is None:
            print("❌ Cannot proceed without real data")
            return
        
        # Step 2: Identify real supply/demand zones
        zones = self.identify_real_supply_demand_zones(df)
        
        # Step 3: Execute scalping strategy
        trades, final_capital = self.execute_real_scalping_strategy(df, zones)
        
        # Step 4: Generate real data report
        results = self.generate_real_data_report(trades, final_capital, df)
        
        return results

if __name__ == "__main__":
    try:
        backtester = LiveFyersBacktester()
        backtester.run_live_data_backtest()
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        print("🔧 Check your fyers_config.json and API credentials!")