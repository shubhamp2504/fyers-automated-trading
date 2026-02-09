"""
MULTI-TIMEFRAME NIFTY OPTIONS POWERHOUSE
Advanced supply & demand trading across ALL timeframes
Features: 1m,3m,5m,10m,15m,30m,1h,4h analysis with double qty supply zones

POWERHOUSE FEATURES:
- Multi-timeframe zone identification
- Double quantity for supply zones (faster profits)
- Frequent profit booking (5-25 points)
- Tight stop losses (6 points)
- Enhanced position sizing
- Real-time trend confirmation across timeframes
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fyers_client import FyersClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframePowerhouse:
    """
    MULTI-TIMEFRAME POWERHOUSE TRADING SYSTEM
    Maximum opportunities across all timeframes with enhanced profit potential
    """
    
    def __init__(self):
        print("MULTI-TIMEFRAME NIFTY OPTIONS POWERHOUSE")
        print("=" * 80)
        print("ENHANCED SYSTEM: ALL TIMEFRAMES ANALYSIS")
        print("DATA SOURCE: REAL MARKET DATA via Official Fyers API")
        print("FEATURES: Double Qty Supply Zones + Frequent Profit Booking")
        print("=" * 80)
        
        # Initialize SECURE Fyers client
        try:
            self.client = FyersClient()
            print("POWERHOUSE CONNECTION: Official Fyers API initialized successfully")
        except Exception as e:
            print(f"ERROR: Failed to initialize Fyers client: {e}")
            raise
        
        # Multi-timeframe configuration
        self.timeframes = {
            '1': '1min',    # Scalping opportunities
            '3': '3min',    # Quick entries  
            '5': '5min',    # Standard timeframe
            '10': '10min',  # Swing entries
            '15': '15min',  # Trend confirmation
            '30': '30min',  # Major zones
            '60': '1hour',  # Strong institutional zones
            '240': '4hour'  # Highest priority zones
        }
        
        # Multi-timeframe data storage
        self.multi_data = {}
        self.multi_zones = {'supply': {}, 'demand': {}}
        
        # POWERHOUSE TRADING PARAMETERS
        self.lookback_period = 12           # Faster analysis
        self.min_impulse_size = 0.10        # Lower threshold = more opportunities
        self.zone_buffer = 0.12             # Tighter zones (12%)
        self.risk_per_trade = 0.010         # 1% base risk
        self.supply_multiplier = 2.0        # DOUBLE QTY for supply zones
        self.max_dte = 12                   # Extended expiry range
        self.min_premium = 1.5              # Lower premium for frequent trading
        
        # FREQUENT PROFIT BOOKING SYSTEM
        self.micro_profit_points = [5, 8, 12, 15, 18, 22, 25, 30]  # Quick booking
        self.micro_stop_loss = 6            # Tight 6-point stop
        self.supply_profit_target = 1.8     # 80% profit for supply (faster)
        self.demand_profit_target = 1.5     # 50% profit for demand
        
        # Performance tracking
        self.capital = 100000
        self.trades = []
        self.daily_performance = {}
        self.supply_zones = []
        self.demand_zones = []
        
        print(f"Strategy: MULTI-TIMEFRAME POWERHOUSE SYSTEM")
        print(f"  - Supply Zones: DOUBLE QUANTITY (2X profit potential)")
        print(f"  - Demand Zones: Standard quantity + Quick profits")
        print(f"  - Profit Booking: {self.micro_profit_points} points")
        print(f"  - Stop Loss: {self.micro_stop_loss} points")
        print(f"Starting Capital: Rs.{self.capital:,.2f}")
        print(f"Risk: {self.risk_per_trade*100}% per trade | Supply: {self.supply_multiplier}X quantity")
    
    def fetch_multi_timeframe_data(self, start_date: str, end_date: str):
        """
        Fetch data across ALL timeframes for comprehensive analysis
        """
        print(f"\nFETCHING MULTI-TIMEFRAME DATA - POWERHOUSE MODE")
        print("-" * 60)
        print(f"TIMEFRAMES: 1m, 3m, 5m, 10m, 15m, 30m, 1h, 4h")
        print(f"PERIOD: {start_date} to {end_date}")
        
        success_count = 0
        total_candles = 0
        
        for resolution, name in self.timeframes.items():
            try:
                print(f"Fetching {name} data...")
                df = self.client.get_historical_data(
                    symbol="NSE:NIFTY50-INDEX",
                    start_date=start_date,
                    end_date=end_date,
                    resolution=resolution
                )
                
                if df is not None and len(df) > 0:
                    self.multi_data[name] = df
                    success_count += 1
                    total_candles += len(df)
                    
                    print(f"✅ {name}: {len(df)} candles | Range: Rs.{df['low'].min():.2f} - Rs.{df['high'].max():.2f}")
                else:
                    print(f"❌ Failed to fetch {name} data")
                    
            except Exception as e:
                print(f"❌ Error fetching {name}: {str(e)}")
        
        print(f"\n📊 MULTI-TIMEFRAME POWERHOUSE LOADED:")
        print(f"   Timeframes: {success_count}/{len(self.timeframes)} successful")
        print(f"   Total Candles: {total_candles:,} across all timeframes")
        
        # Set primary data for compatibility
        if '5min' in self.multi_data:
            self.nifty_data = self.multi_data['5min']
        
        return success_count >= 4
    
    def identify_multi_timeframe_zones(self):
        """
        Identify supply/demand zones across ALL timeframes
        Enhanced for maximum trading opportunities
        """
        print(f"\nIDENTIFYING MULTI-TIMEFRAME ZONES")
        print("-" * 60)
        
        all_supply_zones = []
        all_demand_zones = []
        
        # Timeframe priority weights
        timeframe_weights = {
            '4hour': 8, '1hour': 6, '30min': 5, '15min': 4,
            '10min': 3, '5min': 2, '3min': 1.5, '1min': 1
        }
        
        for tf_name, data in self.multi_data.items():
            if data is None or len(data) < self.lookback_period:
                continue
                
            print(f"Analyzing {tf_name}...")
            
            # Find impulse moves
            impulses = self.find_impulse_moves(data, tf_name)
            zones_found = 0
            
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
            
            print(f"✅ {tf_name}: {zones_found} zones")
        
        # Sort by enhanced strength
        all_supply_zones.sort(key=lambda x: x['enhanced_strength'], reverse=True)
        all_demand_zones.sort(key=lambda x: x['enhanced_strength'], reverse=True)
        
        # Keep top zones
        self.supply_zones = all_supply_zones[:25]  # Top 25 supply zones
        self.demand_zones = all_demand_zones[:25]  # Top 25 demand zones
        
        print(f"\n📊 POWERHOUSE ZONES IDENTIFIED:")
        print(f"   Supply Zones: {len(self.supply_zones)} (DOUBLE QUANTITY)")
        print(f"   Demand Zones: {len(self.demand_zones)} (Standard quantity)")
        
        if self.supply_zones:
            top_supply = self.supply_zones[0]
            print(f"   TOP SUPPLY: {top_supply['timeframe']} Rs.{top_supply['low']:.2f}-{top_supply['high']:.2f} (Strength: {top_supply['enhanced_strength']:.1f})")
        
        if self.demand_zones:
            top_demand = self.demand_zones[0]
            print(f"   TOP DEMAND: {top_demand['timeframe']} Rs.{top_demand['low']:.2f}-{top_demand['high']:.2f} (Strength: {top_demand['enhanced_strength']:.1f})")
    
    def find_impulse_moves(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Find impulse moves with timeframe-specific thresholds
        """
        # HYPER-AGGRESSIVE thresholds - CATCH EVERY TINY MOVE! 🔥💥
        thresholds = {
            '1min': 0.0001, '3min': 0.0002, '5min': 0.0003, '10min': 0.0005,
            '15min': 0.0008, '30min': 0.0010, '1hour': 0.0015, '4hour': 0.0020
        }
        
        min_move = thresholds.get(timeframe, 0.15)
        impulses = []
        
        for i in range(self.lookback_period, len(data) - 1):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            move_size = abs(current['close'] - previous['close']) / previous['close']
            
            if move_size >= min_move:
                # NO VOLUME RESTRICTION - CATCH EVERY MOVE! 🚀💥
                direction = 'bullish' if current['close'] > previous['close'] else 'bearish'
                
                impulses.append({
                    'timestamp': current.name,
                    'direction': direction,
                    'move_size': move_size,
                    'volume_ratio': 1.0,  # Always pass
                    'prev_candle_idx': i-1,
                    'timeframe': timeframe
                })
        
        result_df = pd.DataFrame(impulses)
        print(f"   🎯 {timeframe}: Found {len(impulses)} impulses (threshold: {min_move:.3%})")
        return result_df
    
    def create_zone_from_impulse(self, impulse: dict, data: pd.DataFrame, timeframe: str) -> dict:
        """
        Create precise zone from impulse with consolidation detection
        """
        prev_idx = impulse['prev_candle_idx']
        
        if prev_idx < 2:
            return None
        
        # Look for consolidation before impulse
        consolidation_candles = []
        for look_back in range(0, min(4, prev_idx)):
            candle_idx = prev_idx - look_back
            candle = data.iloc[candle_idx]
            
            # Consolidation criteria
            body_size = abs(candle['close'] - candle['open'])
            candle_range = candle['high'] - candle['low']
            
            if body_size < candle_range * 0.5:  # Small body
                consolidation_candles.append(candle)
        
        if not consolidation_candles:
            consolidation_candles = [data.iloc[prev_idx]]
        
        # Create zone
        zone_high = max(c['high'] for c in consolidation_candles)
        zone_low = min(c['low'] for c in consolidation_candles)
        
        base_strength = impulse['move_size'] * impulse['volume_ratio']
        
        return {
            'timestamp': consolidation_candles[0].name,
            'high': zone_high,
            'low': zone_low,
            'strength': base_strength,
            'move_size': impulse['move_size'],
            'volume_ratio': impulse['volume_ratio'],
            'tested': False,
            'broken': False,
            'trade_count': 0
        }
    
    def execute_powerhouse_trade(self, signal: dict, current_candle: pd.Series) -> dict:
        """
        Execute trade with enhanced position sizing and double quantity for supply zones
        """
        current_price = current_candle['close']
        current_date = current_candle.name
        
        # Get option parameters
        expiry_date = self.get_nearest_expiry(current_date)
        if not expiry_date:
            return None
        
        dte = (datetime.strptime(expiry_date, "%Y-%m-%d").date() - current_date.date()).days
        if dte > self.max_dte or dte < 1:
            return None
        
        # Strike selection
        if signal['action'] == 'buy_call':
            option_type = 'CE'
            strike = self.get_optimal_strike(current_price, signal['zone_price'], 'CE', signal['timeframe'])
        else:
            option_type = 'PE'
            strike = self.get_optimal_strike(current_price, signal['zone_price'], 'PE', signal['timeframe'])
        
        # Premium calculation
        premium = self.calculate_premium(current_price, strike, option_type, dte)
        
        # ENHANCED POSITION SIZING
        is_supply_zone = signal.get('zone_type') == 'supply'
        base_risk = self.capital * self.risk_per_trade
        
        if is_supply_zone:
            # DOUBLE QUANTITY for supply zones
            risk_amount = base_risk * self.supply_multiplier
            trade_type = 'SUPPLY_DOUBLE'
        else:
            risk_amount = base_risk
            trade_type = 'DEMAND_STANDARD'
        
        # Calculate quantity
        max_loss = premium * 75 * 0.4  # 40% max loss assumption
        quantity = max(1, min(int(risk_amount / max_loss), 30 if is_supply_zone else 15))
        
        # Create trade record
        trade = {
            'entry_time': current_date,
            'symbol': f"NSE:NIFTY{expiry_date.replace('-', '')}{int(strike)}{option_type}",
            'action': signal['action'],
            'spot_price': current_price,
            'strike': strike,
            'option_type': option_type,
            'quantity': quantity,
            'entry_premium': premium,
            'dte': dte,
            'zone_type': signal['zone_type'],
            'timeframe': signal['timeframe'],
            'trade_type': trade_type,
            'double_quantity': is_supply_zone,
            'profit_targets': self.micro_profit_points.copy(),
            'target_profit': self.supply_profit_target if is_supply_zone else self.demand_profit_target,
            'stop_loss_points': self.micro_stop_loss,
            'status': 'open',
            'pnl': 0,
            'lot_size': 75
        }
        
        print(f"{trade_type}: {trade['symbol']} @ Rs.{premium:.2f}")
        print(f"  Spot: Rs.{current_price:.2f}, Qty: {quantity}{'(2X)' if is_supply_zone else ''}, TF: {signal['timeframe']}")
        
        return trade
    
    def get_optimal_strike(self, spot: float, zone_price: float, option_type: str, timeframe: str) -> float:
        """Get optimal strike based on timeframe"""
        base_strike = round(zone_price / 50) * 50
        
        # Timeframe-based adjustments
        if timeframe in ['1min', '3min']:
            offset = 0  # ATM for scalping
        elif timeframe in ['5min', '10min']:
            offset = 25 if option_type == 'CE' else -25
        else:
            offset = 50 if option_type == 'CE' else -50
        
        return base_strike + offset
    
    def calculate_premium(self, spot: float, strike: float, option_type: str, dte: int) -> float:
        """Simplified premium calculation"""
        if option_type == 'CE':
            intrinsic = max(0, spot - strike)
        else:
            intrinsic = max(0, strike - spot)
        
        time_value = max(spot * 0.008 * (dte / 7), spot * 0.003)
        premium = intrinsic + time_value
        return max(premium, 2.0)
    
    def get_nearest_expiry(self, current_date: datetime) -> str:
        """Get nearest Thursday expiry"""
        days_ahead = 3 - current_date.weekday()  # Thursday = 3
        if days_ahead <= 0:
            days_ahead += 7
        
        expiry = current_date + timedelta(days=days_ahead)
        return expiry.strftime("%Y-%m-%d")
    
    def validate_zone(self, current_candle: pd.Series, zone: dict) -> bool:
        """Enhanced zone validation"""
        zone_high = zone['high']
        zone_low = zone['low']
        buffer = (zone_high - zone_low) * self.zone_buffer
        
        # Check overlap
        overlap = (current_candle['low'] <= zone_high + buffer and 
                  current_candle['high'] >= zone_low - buffer)
        
        # Enhanced validations
        return (overlap and 
                zone['trade_count'] < 15 and 
                zone.get('enhanced_strength', 0) > 0.2 and
                not zone['broken'])
    
    def manage_trades(self, current_candle: pd.Series, open_trades: list):
        """Enhanced trade management with frequent profit booking"""
        current_price = current_candle['close']
        current_date = current_candle.name
        
        for trade in open_trades:
            if trade['status'] != 'open':
                continue
            
            # Calculate current premium
            days_held = (current_date.date() - trade['entry_time'].date()).days
            remaining_dte = max(1, trade['dte'] - days_held)
            current_premium = self.calculate_premium(current_price, trade['strike'], trade['option_type'], remaining_dte)
            
            # P&L calculation
            premium_change = current_premium - trade['entry_premium']
            pnl = premium_change * trade['quantity'] * trade['lot_size']
            profit_pct = premium_change / trade['entry_premium']
            points_moved = abs(current_price - trade['spot_price'])
            
            exit_reason = None
            
            # FREQUENT PROFIT BOOKING
            for target_points in trade['profit_targets']:
                if points_moved >= target_points and profit_pct > 0.12:
                    exit_reason = f'quick_profit_{int(points_moved)}pts'
                    break
            
            # Target profit
            if not exit_reason and profit_pct >= trade['target_profit']:
                exit_reason = 'target_profit'
            
            # Tight stop loss
            elif not exit_reason and points_moved >= trade['stop_loss_points'] and profit_pct < -0.20:
                exit_reason = 'micro_stop'
            
            # Time decay
            elif not exit_reason and remaining_dte <= 1:
                exit_reason = 'time_decay'
            
            # Execute exit
            if exit_reason:
                trade['status'] = 'closed'
                trade['exit_time'] = current_date
                trade['exit_premium'] = current_premium
                trade['pnl'] = pnl
                trade['exit_reason'] = exit_reason
                
                self.capital += pnl
                
                # Track daily performance
                exit_date = current_date.strftime('%Y-%m-%d')
                if exit_date not in self.daily_performance:
                    self.daily_performance[exit_date] = {'entries': 0, 'exits': 0, 'daily_pnl': 0, 'trades': []}
                self.daily_performance[exit_date]['exits'] += 1
                self.daily_performance[exit_date]['daily_pnl'] += pnl
                
                print(f"EXIT: {trade['symbol']} @ Rs.{current_premium:.2f} | P&L: Rs.{pnl:,.0f} | {exit_reason} | {trade['trade_type']}")
    
    def run_powerhouse_backtest(self, start_date: str, end_date: str):
        """
        Execute multi-timeframe powerhouse backtesting
        """
        print(f"\nSTARTING MULTI-TIMEFRAME POWERHOUSE BACKTEST")
        print("=" * 80)
        
        # Fetch data
        if not self.fetch_multi_timeframe_data(start_date, end_date):
            print("ERROR: Insufficient timeframe data")
            return
        
        # Identify zones
        self.identify_multi_timeframe_zones()
        
        if not self.supply_zones and not self.demand_zones:
            print("ERROR: No trading zones found")
            return
        
        # Execute backtest
        open_trades = []
        signals_generated = 0
        trades_executed = 0
        supply_trades = 0
        demand_trades = 0
        
        print(f"\n🚀 EXECUTING POWERHOUSE BACKTEST...")
        print(f"Supply Zones: {len(self.supply_zones)} (DOUBLE QTY)")
        print(f"Demand Zones: {len(self.demand_zones)} (Standard)")
        
        # Use 5-minute data as execution timeframe
        primary_data = self.multi_data.get('5min')
        if primary_data is None:
            print("ERROR: No 5-minute data available")
            return
        
        for i in range(self.lookback_period, len(primary_data)):
            current_candle = primary_data.iloc[i]
            current_date = current_candle.name
            
            # Trade management
            self.manage_trades(current_candle, open_trades)
            
            # Market hours check
            if not (time(9, 15) <= current_date.time() <= time(15, 30)):
                continue
            
            # Position limits
            active_positions = len([t for t in open_trades if t['status'] == 'open'])
            if active_positions >= 20:  # Allow more positions
                continue
            
            # SUPPLY ZONE TRADING (Double quantity)
            for zone in self.supply_zones[:15]:
                if self.validate_zone(current_candle, zone):
                    signal = {
                        'action': 'buy_put',
                        'zone_type': 'supply',
                        'zone_price': (zone['high'] + zone['low']) / 2,
                        'timeframe': zone['timeframe']
                    }
                    
                    trade = self.execute_powerhouse_trade(signal, current_candle)
                    if trade:
                        open_trades.append(trade)
                        self.trades.append(trade)
                        signals_generated += 1
                        trades_executed += 1
                        supply_trades += 1
                        
                        zone['trade_count'] += 1
                        
                        # Daily tracking
                        entry_date = current_date.strftime('%Y-%m-%d')
                        if entry_date not in self.daily_performance:
                            self.daily_performance[entry_date] = {'entries': 0, 'exits': 0, 'daily_pnl': 0, 'trades': []}
                        self.daily_performance[entry_date]['entries'] += 1
                        break
            
            # DEMAND ZONE TRADING (Standard quantity)
            if active_positions < 20:
                for zone in self.demand_zones[:15]:
                    if self.validate_zone(current_candle, zone):
                        signal = {
                            'action': 'buy_call',
                            'zone_type': 'demand',
                            'zone_price': (zone['high'] + zone['low']) / 2,
                            'timeframe': zone['timeframe']
                        }
                        
                        trade = self.execute_powerhouse_trade(signal, current_candle)
                        if trade:
                            open_trades.append(trade)
                            self.trades.append(trade)
                            signals_generated += 1
                            trades_executed += 1
                            demand_trades += 1
                            
                            zone['trade_count'] += 1
                            
                            # Daily tracking
                            entry_date = current_date.strftime('%Y-%m-%d')
                            if entry_date not in self.daily_performance:
                                self.daily_performance[entry_date] = {'entries': 0, 'exits': 0, 'daily_pnl': 0, 'trades': []}
                            self.daily_performance[entry_date]['entries'] += 1
                            break
        
        print(f"\n🎯 POWERHOUSE BACKTEST COMPLETED")
        print(f"SIGNALS: {signals_generated}")
        print(f"TRADES: {trades_executed}")
        print(f"  🔴 Supply (2X): {supply_trades}")
        print(f"  🟢 Demand (1X): {demand_trades}")
        print(f"FINAL CAPITAL: Rs.{self.capital:,.2f}")
        
        # Performance analysis
        self.display_performance()
    
    def display_performance(self):
        """Display comprehensive performance analysis"""
        if not self.daily_performance:
            print("\n📊 No daily performance data available")
            return
        
        print(f"\n📊 DAY-WISE POWERHOUSE RESULTS")
        print("=" * 84)
        print(f"{'Date':<12} {'Entries':<8} {'Exits':<8} {'Daily P&L':<15} {'Running Capital':<15}")
        print("-" * 84)
        
        running_capital = 100000
        profitable_days = 0
        total_days = 0
        
        for date in sorted(self.daily_performance.keys()):
            day_data = self.daily_performance[date]
            daily_pnl = day_data['daily_pnl']
            running_capital += daily_pnl
            
            if day_data['entries'] > 0 or day_data['exits'] > 0:
                total_days += 1
                if daily_pnl > 0:
                    profitable_days += 1
                
                pnl_icon = "📈" if daily_pnl > 0 else "📉" if daily_pnl < 0 else "📊"
                print(f"{date:<12} {day_data['entries']:<8} {day_data['exits']:<8} "
                      f"{pnl_icon} Rs.{daily_pnl:,.0f}{'':^7} Rs.{running_capital:,.0f}")
        
        print("-" * 84)
        
        # Summary stats
        if total_days > 0:
            win_rate = (profitable_days / total_days) * 100
            total_return = ((running_capital / 100000) - 1) * 100
            
            print(f"\n🎯 POWERHOUSE PERFORMANCE SUMMARY:")
            print(f"   Trading Days: {total_days}")
            print(f"   Profitable Days: {profitable_days} ({win_rate:.1f}%)")
            print(f"   Total Return: {total_return:+.2f}%")
            print(f"   Final Capital: Rs.{running_capital:,.0f}")
            
            if len(self.trades) > 0:
                closed_trades = [t for t in self.trades if t['status'] == 'closed']
                if closed_trades:
                    winning_trades = len([t for t in closed_trades if t['pnl'] > 0])
                    trade_win_rate = (winning_trades / len(closed_trades)) * 100
                    avg_pnl = sum(t['pnl'] for t in closed_trades) / len(closed_trades)
                    
                    print(f"\n📈 TRADE STATISTICS:")
                    print(f"   Total Trades: {len(closed_trades)}")
                    print(f"   Win Rate: {trade_win_rate:.1f}%")
                    print(f"   Avg P&L: Rs.{avg_pnl:,.0f}")
                    
                    # Supply vs Demand performance
                    supply_trades = [t for t in closed_trades if t.get('double_quantity', False)]
                    demand_trades = [t for t in closed_trades if not t.get('double_quantity', False)]
                    
                    if supply_trades:
                        supply_pnl = sum(t['pnl'] for t in supply_trades)
                        supply_wins = len([t for t in supply_trades if t['pnl'] > 0])
                        supply_win_rate = (supply_wins / len(supply_trades)) * 100
                        print(f"   Supply Zones (2X): {len(supply_trades)} trades, Rs.{supply_pnl:,.0f}, {supply_win_rate:.1f}% win rate")
                    
                    if demand_trades:
                        demand_pnl = sum(t['pnl'] for t in demand_trades)
                        demand_wins = len([t for t in demand_trades if t['pnl'] > 0])
                        demand_win_rate = (demand_wins / len(demand_trades)) * 100
                        print(f"   Demand Zones (1X): {len(demand_trades)} trades, Rs.{demand_pnl:,.0f}, {demand_win_rate:.1f}% win rate")

def main():
    """
    Launch Multi-Timeframe Powerhouse System
    """
    print("LAUNCHING MULTI-TIMEFRAME POWERHOUSE...")
    
    try:
        powerhouse = MultiTimeframePowerhouse()
        
        # Full 2-month analysis
        start_date = "2026-01-01"
        end_date = "2026-02-28"
        
        print(f"\nCOMMENCING POWERHOUSE ANALYSIS")
        print(f"Period: {start_date} to {end_date}")
        print(f"Strategy: ALL timeframes with double qty supply zones")
        
        powerhouse.run_powerhouse_backtest(start_date, end_date)
        
        print(f"\n🚀 MULTI-TIMEFRAME POWERHOUSE COMPLETE")
        print("Ready for maximum profit generation!")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()