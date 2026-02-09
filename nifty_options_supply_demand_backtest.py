"""
🎯 NIFTY OPTIONS SUPPLY & DEMAND STRATEGY BACKTESTER
Advanced options trading system using institutional supply and demand zones

Based on Supply & Demand Theory:
- Demand zones = consolidation before upward moves (buy levels)
- Supply zones = consolidation before downward moves (sell levels)  
- Only trade WITH the trend direction
- Use last candle before impulse to identify zones
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

class NiftyOptionsSupplyDemandBacktester:
    """Advanced NIFTY Options backtesting using Supply & Demand strategy"""
    
    def __init__(self):
        self.client = FyersClient()
        
        # Strategy parameters
        self.lookback_period = 20  # Lookback for S&D zone identification
        self.min_impulse_size = 0.5  # Minimum impulse move percentage
        self.zone_buffer = 0.2  # Zone buffer percentage
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.max_dte = 7  # Maximum days to expiry for options
        self.min_premium = 5.0  # Minimum option premium
        
        # Data storage
        self.nifty_data = None
        self.supply_zones = []
        self.demand_zones = []
        self.trades = []
        self.capital = 100000  # Starting capital
        
        print("🎯 NIFTY OPTIONS SUPPLY & DEMAND BACKTESTER")
        print("=" * 80)
        print("📈 Strategy: Institutional Supply & Demand Zones")
        print("🎯 Market: NIFTY Index Options")
        print("💰 Starting Capital: ₹1,00,000")
        print("⚠️  Risk per Trade: 1%")
        print("=" * 80)
    
    def fetch_nifty_data(self, start_date: str, end_date: str, resolution: str = "5"):
        """Fetch NIFTY index data for analysis"""
        print(f"\n📊 FETCHING NIFTY DATA")
        print("-" * 50)
        
        try:
            params = {
                "symbol": "NSE:NIFTY50-INDEX",
                "resolution": resolution,
                "date_format": "1",
                "range_from": start_date,
                "range_to": end_date,
                "cont_flag": "1"
            }
            
            response = self.client.fyers.history(data=params)
            
            if response['s'] == 'ok':
                candles = response['candles']
                
                # Create DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')
                df.set_index('datetime', inplace=True)
                
                # Calculate additional indicators
                df['sma_20'] = df['close'].rolling(20).mean()
                df['ema_9'] = df['close'].ewm(span=9).mean()
                df['atr'] = self.calculate_atr(df)
                df['body_size'] = abs(df['close'] - df['open'])
                df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
                df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
                df['is_indecision'] = self.identify_indecision_candles(df)
                
                self.nifty_data = df
                print(f"✅ NIFTY data loaded: {len(df)} candles")
                print(f"📅 Period: {df.index[0]} to {df.index[-1]}")
                print(f"📊 Price range: ₹{df['low'].min():.2f} - ₹{df['high'].max():.2f}")
                
                return True
            else:
                print(f"❌ Error fetching data: {response.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Data fetch error: {str(e)}")
            return False
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def identify_indecision_candles(self, df: pd.DataFrame) -> pd.Series:
        """Identify indecision candles (last candles before impulse)"""
        conditions = [
            # Small body relative to range
            df['body_size'] <= (df['high'] - df['low']) * 0.3,
            # Has both upper and lower wicks
            df['upper_wick'] > 0,
            df['lower_wick'] > 0,
            # Wicks are significant
            (df['upper_wick'] + df['lower_wick']) >= df['body_size']
        ]
        
        return pd.Series(np.logical_and.reduce(conditions), index=df.index)
    
    def identify_impulse_moves(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify impulsive moves (strong directional moves)"""
        impulse_data = []
        
        for i in range(1, len(df)):
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            
            # Calculate move size
            move_size = abs(current_candle['close'] - prev_candle['close']) / prev_candle['close'] * 100
            
            # Check for impulse conditions
            is_impulse = (
                move_size >= self.min_impulse_size and
                current_candle['body_size'] >= (current_candle['high'] - current_candle['low']) * 0.7 and
                (current_candle['upper_wick'] + current_candle['lower_wick']) <= current_candle['body_size'] * 0.5
            )
            
            if is_impulse:
                direction = 'bullish' if current_candle['close'] > current_candle['open'] else 'bearish'
                
                impulse_data.append({
                    'timestamp': current_candle.name,
                    'direction': direction,
                    'move_size': move_size,
                    'open': current_candle['open'],
                    'high': current_candle['high'],
                    'low': current_candle['low'],
                    'close': current_candle['close'],
                    'prev_candle_idx': i-1
                })
        
        return pd.DataFrame(impulse_data)
    
    def identify_supply_demand_zones(self):
        """Identify supply and demand zones based on impulse moves"""
        print(f"\n🎯 IDENTIFYING SUPPLY & DEMAND ZONES")
        print("-" * 50)
        
        if self.nifty_data is None:
            print("❌ No NIFTY data available")
            return
        
        # Find impulse moves
        impulses = self.identify_impulse_moves(self.nifty_data)
        
        supply_zones = []
        demand_zones = []
        
        for _, impulse in impulses.iterrows():
            prev_candle_idx = impulse['prev_candle_idx']
            
            # Check if previous candle is indecision candle
            if prev_candle_idx >= 0 and prev_candle_idx < len(self.nifty_data):
                prev_candle = self.nifty_data.iloc[prev_candle_idx]
                
                if self.nifty_data['is_indecision'].iloc[prev_candle_idx]:
                    
                    if impulse['direction'] == 'bullish':
                        # Create demand zone from last candle before bullish impulse
                        zone = {
                            'timestamp': prev_candle.name,
                            'type': 'demand',
                            'high': prev_candle['high'],
                            'low': prev_candle['low'],
                            'strength': impulse['move_size'],
                            'tested': False,
                            'broken': False
                        }
                        demand_zones.append(zone)
                        
                    elif impulse['direction'] == 'bearish':
                        # Create supply zone from last candle before bearish impulse  
                        zone = {
                            'timestamp': prev_candle.name,
                            'type': 'supply',
                            'high': prev_candle['high'],
                            'low': prev_candle['low'],
                            'strength': impulse['move_size'],
                            'tested': False,
                            'broken': False
                        }
                        supply_zones.append(zone)
        
        self.supply_zones = supply_zones
        self.demand_zones = demand_zones
        
        print(f"✅ Supply zones identified: {len(supply_zones)}")
        print(f"✅ Demand zones identified: {len(demand_zones)}")
        
        # Display strongest zones
        if supply_zones:
            strongest_supply = max(supply_zones, key=lambda x: x['strength'])
            print(f"💪 Strongest supply zone: ₹{strongest_supply['low']:.2f}-₹{strongest_supply['high']:.2f} ({strongest_supply['strength']:.2f}%)")
        
        if demand_zones:
            strongest_demand = max(demand_zones, key=lambda x: x['strength'])
            print(f"💪 Strongest demand zone: ₹{strongest_demand['low']:.2f}-₹{strongest_demand['high']:.2f} ({strongest_demand['strength']:.2f}%)")
    
    def determine_market_trend(self, current_idx: int) -> str:
        """Determine market trend using swing highs and lows"""
        if current_idx < 20:
            return 'sideways'
        
        # Look at last 20 candles for trend analysis
        recent_data = self.nifty_data.iloc[current_idx-19:current_idx+1]
        
        # Find swing high and lows
        highs = recent_data['high'].rolling(5, center=True).max()
        lows = recent_data['low'].rolling(5, center=True).min()
        
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_data)-2):
            if recent_data['high'].iloc[i] == highs.iloc[i]:
                swing_highs.append(recent_data['high'].iloc[i])
            if recent_data['low'].iloc[i] == lows.iloc[i]:
                swing_lows.append(recent_data['low'].iloc[i])
        
        # Determine trend
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            higher_highs = swing_highs[-1] > swing_highs[-2] if len(swing_highs) >= 2 else False
            higher_lows = swing_lows[-1] > swing_lows[-2] if len(swing_lows) >= 2 else False
            lower_highs = swing_highs[-1] < swing_highs[-2] if len(swing_highs) >= 2 else False  
            lower_lows = swing_lows[-1] < swing_lows[-2] if len(swing_lows) >= 2 else False
            
            if higher_highs and higher_lows:
                return 'uptrend'
            elif lower_highs and lower_lows:
                return 'downtrend'
        
        return 'sideways'
    
    def get_option_strike(self, current_price: float, zone_price: float, option_type: str) -> float:
        """Calculate appropriate option strike price"""
        
        # Round to nearest 50 (NIFTY option strikes are in multiples of 50)
        base_strike = round(zone_price / 50) * 50
        
        if option_type == 'CE':
            # For calls, use ATM or slightly OTM
            strike_offset = 0 if current_price <= zone_price else 50
            return base_strike + strike_offset
        else:  # PE
            # For puts, use ATM or slightly OTM  
            strike_offset = 0 if current_price >= zone_price else -50
            return base_strike + strike_offset
    
    def construct_option_symbol(self, strike: float, option_type: str, expiry_date: str) -> str:
        """Construct NIFTY option symbol using Fyers format"""
        
        # Convert expiry date to required format
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        
        # Weekly expiry format: NSE:NIFTY{YY}{M}{dd}{Strike}{CE/PE}
        year = expiry_dt.strftime("%y")
        month_code = expiry_dt.strftime("%b").upper()[:3]  # JAN, FEB, MAR etc
        day = expiry_dt.strftime("%d")
        
        # Format: NSE:NIFTY23JAN11000CE
        symbol = f"NSE:NIFTY{year}{month_code}{int(strike)}{option_type}"
        
        return symbol
    
    def get_nearest_expiry(self, current_date: datetime) -> str:
        """Get nearest Thursday expiry for NIFTY options"""
        
        # Find next Thursday
        days_ahead = 3 - current_date.weekday()  # Thursday is 3
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        expiry_date = current_date + timedelta(days=days_ahead)
        
        # Check if expiry is too far
        if days_ahead > self.max_dte:
            # Use current week Thursday if within max DTE
            days_ahead = 3 - current_date.weekday()
            if days_ahead >= 0:
                expiry_date = current_date + timedelta(days=days_ahead)
        
        return expiry_date.strftime("%Y-%m-%d")
    
    def simulate_option_premium(self, spot_price: float, strike: float, option_type: str, dte: int) -> float:
        """Simulate option premium (simplified Black-Scholes approximation)"""
        
        # Simplified premium calculation
        moneyness = abs(spot_price - strike) / spot_price
        time_value = max(0.1, dte / 365 * 0.3)  # Rough time decay
        
        if option_type == 'CE':
            intrinsic = max(0, spot_price - strike)
        else:  # PE
            intrinsic = max(0, strike - spot_price)
        
        # Add volatility component
        volatility_component = spot_price * 0.01 * np.sqrt(dte / 365)
        
        premium = intrinsic + time_value * spot_price * 0.02 + volatility_component
        
        return max(self.min_premium, premium)
    
    def check_zone_trigger(self, current_candle: pd.Series, zone: dict) -> bool:
        """Check if price has entered a supply/demand zone"""
        
        zone_high = zone['high']
        zone_low = zone['low']
        
        # Add buffer to zone
        buffer = (zone_high - zone_low) * self.zone_buffer
        zone_high_buffer = zone_high + buffer
        zone_low_buffer = zone_low - buffer
        
        # Check if current candle overlaps with zone
        candle_high = current_candle['high']
        candle_low = current_candle['low']
        
        return (candle_low <= zone_high_buffer and candle_high >= zone_low_buffer)
    
    def execute_trade(self, signal: dict, current_candle: pd.Series) -> dict:
        """Execute options trade based on signal"""
        
        current_price = current_candle['close']
        current_date = current_candle.name
        
        # Get expiry date
        expiry_date = self.get_nearest_expiry(current_date)
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        dte = (expiry_dt.date() - current_date.date()).days
        
        if dte > self.max_dte:
            return None  # Skip if expiry too far
        
        # Determine option type and strike
        if signal['action'] == 'buy_call':
            option_type = 'CE'
            strike = self.get_option_strike(current_price, signal['zone_price'], 'CE')
        elif signal['action'] == 'buy_put':
            option_type = 'PE'
            strike = self.get_option_strike(current_price, signal['zone_price'], 'PE')
        else:
            return None
        
        # Construct option symbol
        option_symbol = self.construct_option_symbol(strike, option_type, expiry_date)
        
        # Calculate premium
        premium = self.simulate_option_premium(current_price, strike, option_type, dte)
        
        # Calculate position size based on risk
        risk_amount = self.capital * self.risk_per_trade
        max_loss = premium * 0.5  # Assume 50% max loss on premium
        
        if max_loss > 0:
            quantity = min(int(risk_amount / max_loss), 50)  # Max 50 lots
            quantity = max(quantity, 1)  # At least 1 lot
        else:
            quantity = 1
        
        # Create trade
        trade = {
            'entry_time': current_date,
            'symbol': option_symbol,
            'action': signal['action'],
            'spot_price': current_price,
            'strike': strike,
            'option_type': option_type,
            'quantity': quantity,
            'entry_premium': premium,
            'dte': dte,
            'zone_type': signal['zone_type'],
            'zone_price': signal['zone_price'],
            'trend': signal['trend'],
            'status': 'open',
            'exit_time': None,
            'exit_premium': None,
            'pnl': 0,
            'exit_reason': None
        }
        
        return trade
    
    def manage_open_trades(self, current_candle: pd.Series, open_trades: list):
        """Manage open trades - check for exit conditions"""
        
        current_price = current_candle['close']
        current_date = current_candle.name
        
        for trade in open_trades:
            if trade['status'] != 'open':
                continue
            
            # Calculate current DTE
            expiry_dt = datetime.strptime(trade['entry_time'].strftime("%Y-%m-%d"), "%Y-%m-%d")
            dte = (expiry_dt.date() - current_date.date()).days
            
            # Get current premium
            current_premium = self.simulate_option_premium(
                current_price, trade['strike'], trade['option_type'], max(0, dte)
            )
            
            # Exit conditions
            exit_reason = None
            
            # 1. Profit target (100% gain)
            if current_premium >= trade['entry_premium'] * 2:
                exit_reason = 'profit_target'
            
            # 2. Stop loss (50% loss)
            elif current_premium <= trade['entry_premium'] * 0.5:
                exit_reason = 'stop_loss'
            
            # 3. Time decay (1 day to expiry)  
            elif dte <= 1:
                exit_reason = 'expiry'
            
            # 4. Trend reversal
            elif self.determine_market_trend(self.nifty_data.index.get_loc(current_date)) != trade['trend']:
                exit_reason = 'trend_reversal'
            
            # Execute exit
            if exit_reason:
                trade['exit_time'] = current_date
                trade['exit_premium'] = current_premium
                trade['status'] = 'closed'
                trade['exit_reason'] = exit_reason
                
                # Calculate P&L
                if trade['action'] in ['buy_call', 'buy_put']:
                    trade['pnl'] = (current_premium - trade['entry_premium']) * trade['quantity'] * 75  # Lot size
                else:
                    trade['pnl'] = (trade['entry_premium'] - current_premium) * trade['quantity'] * 75
                
                self.capital += trade['pnl']
    
    def run_backtest(self, start_date: str, end_date: str):
        """Run comprehensive supply & demand options backtesting"""
        
        print(f"\n🚀 STARTING SUPPLY & DEMAND OPTIONS BACKTEST")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"💰 Starting Capital: ₹{self.capital:,.2f}")
        print("=" * 80)
        
        # Fetch data
        if not self.fetch_nifty_data(start_date, end_date, "5"):
            return
        
        # Identify zones
        self.identify_supply_demand_zones()
        
        # Backtest simulation
        open_trades = []
        signals_count = 0
        
        for i in range(self.lookback_period, len(self.nifty_data)):
            current_candle = self.nifty_data.iloc[i]
            current_date = current_candle.name
            current_price = current_candle['close']
            
            # Manage existing trades
            self.manage_open_trades(current_candle, open_trades)
            
            # Check for new signals
            trend = self.determine_market_trend(i)
            
            # Only trade during market hours (9:15 AM to 3:30 PM IST)
            current_time = current_date.time()
            market_open = time(9, 15)
            market_close = time(15, 30)
            
            if not (market_open <= current_time <= market_close):
                continue
            
            # Check demand zones (for buying calls in uptrend)
            if trend == 'uptrend':
                for zone in self.demand_zones:
                    if not zone['broken'] and self.check_zone_trigger(current_candle, zone):
                        
                        signal = {
                            'action': 'buy_call',
                            'zone_type': 'demand',
                            'zone_price': (zone['high'] + zone['low']) / 2,
                            'trend': trend
                        }
                        
                        trade = self.execute_trade(signal, current_candle)
                        if trade:
                            open_trades.append(trade)
                            self.trades.append(trade)
                            signals_count += 1
                            
                            print(f"📈 CALL BUY: {trade['symbol']} @ ₹{trade['entry_premium']:.2f} "
                                  f"(Spot: ₹{current_price:.2f})")
                        
                        # Mark zone as tested
                        zone['tested'] = True
                        break
            
            # Check supply zones (for buying puts in downtrend)
            elif trend == 'downtrend':
                for zone in self.supply_zones:
                    if not zone['broken'] and self.check_zone_trigger(current_candle, zone):
                        
                        signal = {
                            'action': 'buy_put',
                            'zone_type': 'supply',
                            'zone_price': (zone['high'] + zone['low']) / 2,
                            'trend': trend
                        }
                        
                        trade = self.execute_trade(signal, current_candle)
                        if trade:
                            open_trades.append(trade)
                            self.trades.append(trade)
                            signals_count += 1
                            
                            print(f"📉 PUT BUY: {trade['symbol']} @ ₹{trade['entry_premium']:.2f} "
                                  f"(Spot: ₹{current_price:.2f})")
                        
                        # Mark zone as tested
                        zone['tested'] = True
                        break
        
        print(f"\n✅ BACKTEST COMPLETED")
        print(f"📊 Total Signals Generated: {signals_count}")
        print(f"💼 Total Trades Executed: {len(self.trades)}")
    
    def generate_performance_report(self):
        """Generate comprehensive performance analysis"""
        
        if not self.trades:
            print("❌ No trades to analyze")
            return
        
        print(f"\n📊 SUPPLY & DEMAND OPTIONS STRATEGY PERFORMANCE")
        print("=" * 80)
        
        # Basic statistics
        total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        
        if closed_trades:
            win_rate = len(winning_trades) / len(closed_trades) * 100
            total_pnl = sum(t['pnl'] for t in closed_trades)
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            print(f"📈 STRATEGY PERFORMANCE:")
            print(f"   • Total Trades: {total_trades}")
            print(f"   • Closed Trades: {len(closed_trades)}")
            print(f"   • Win Rate: {win_rate:.1f}%")
            print(f"   • Total P&L: ₹{total_pnl:,.2f}")
            print(f"   • Average Win: ₹{avg_win:,.2f}")
            print(f"   • Average Loss: ₹{avg_loss:,.2f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else avg_win / abs(avg_loss)
                print(f"   • Profit Factor: {profit_factor:.2f}")
            
            # Returns
            initial_capital = 100000
            final_capital = self.capital
            total_return = (final_capital - initial_capital) / initial_capital * 100
            
            print(f"\n💰 CAPITAL ANALYSIS:")
            print(f"   • Starting Capital: ₹{initial_capital:,.2f}")
            print(f"   • Ending Capital: ₹{final_capital:,.2f}")
            print(f"   • Total Return: {total_return:.2f}%")
            
            # Trade analysis by zone type
            demand_trades = [t for t in closed_trades if t['zone_type'] == 'demand']
            supply_trades = [t for t in closed_trades if t['zone_type'] == 'supply']
            
            print(f"\n🎯 ZONE TYPE ANALYSIS:")
            if demand_trades:
                demand_pnl = sum(t['pnl'] for t in demand_trades)
                demand_win_rate = len([t for t in demand_trades if t['pnl'] > 0]) / len(demand_trades) * 100
                print(f"   • Demand Zones: {len(demand_trades)} trades, ₹{demand_pnl:,.2f} P&L, {demand_win_rate:.1f}% win rate")
            
            if supply_trades:
                supply_pnl = sum(t['pnl'] for t in supply_trades)
                supply_win_rate = len([t for t in supply_trades if t['pnl'] > 0]) / len(supply_trades) * 100
                print(f"   • Supply Zones: {len(supply_trades)} trades, ₹{supply_pnl:,.2f} P&L, {supply_win_rate:.1f}% win rate")
            
            # Exit reason analysis
            print(f"\n🚪 EXIT REASON ANALYSIS:")
            exit_reasons = {}
            for trade in closed_trades:
                reason = trade.get('exit_reason', 'unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            for reason, count in exit_reasons.items():
                percentage = count / len(closed_trades) * 100
                print(f"   • {reason.replace('_', ' ').title()}: {count} trades ({percentage:.1f}%)")
            
            # Best and worst trades
            if closed_trades:
                best_trade = max(closed_trades, key=lambda x: x['pnl'])
                worst_trade = min(closed_trades, key=lambda x: x['pnl'])
                
                print(f"\n🏆 BEST TRADE:")
                print(f"   • {best_trade['symbol']} - ₹{best_trade['pnl']:,.2f}")
                print(f"   • Entry: {best_trade['entry_time'].strftime('%Y-%m-%d %H:%M')}")
                print(f"   • Zone: {best_trade['zone_type']}, Trend: {best_trade['trend']}")
                
                print(f"\n💔 WORST TRADE:")
                print(f"   • {worst_trade['symbol']} - ₹{worst_trade['pnl']:,.2f}")
                print(f"   • Entry: {worst_trade['entry_time'].strftime('%Y-%m-%d %H:%M')}")
                print(f"   • Zone: {worst_trade['zone_type']}, Trend: {worst_trade['trend']}")
        
        # Zone effectiveness
        print(f"\n📍 SUPPLY & DEMAND ZONE EFFECTIVENESS:")
        tested_demand = len([z for z in self.demand_zones if z['tested']])
        tested_supply = len([z for z in self.supply_zones if z['tested']])
        
        print(f"   • Total Demand Zones: {len(self.demand_zones)}")
        print(f"   • Tested Demand Zones: {tested_demand}")
        print(f"   • Total Supply Zones: {len(self.supply_zones)}")
        print(f"   • Tested Supply Zones: {tested_supply}")
        
        print("=" * 80)

def main():
    """Run NIFTY Options Supply & Demand Strategy Backtest"""
    
    # Initialize backtester
    backtester = NiftyOptionsSupplyDemandBacktester()
    
    # Run backtest for January 2026
    start_date = "2026-01-01"
    end_date = "2026-02-05"
    
    backtester.run_backtest(start_date, end_date)
    
    # Generate performance report
    backtester.generate_performance_report()
    
    print(f"\n🎯 NIFTY OPTIONS SUPPLY & DEMAND BACKTEST COMPLETE!")

if __name__ == "__main__":
    main()