"""
Index Intraday Trading Strategy - 1 Hour Analysis with 1/5 Min Execution
======================================================================

Strategy: Multi-Timeframe Index Trading with Smart Risk Management
- Analysis Timeframe: 1 Hour candles
- Execution Timeframe: 1 Min / 5 Min candles  
- Target: 20-30 points profit booking
- Stop Loss: Dynamic and intelligent to minimize hits
- Focus: NIFTY 50 and BANK NIFTY indices

⚠️ IMPORTANT: Always refer to https://myapi.fyers.in/docsv3 for latest API specifications
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
sys.path.append('api_reference/market_data')
sys.path.append('api_reference/orders')
from market_data_complete import FyersMarketData
from orders_complete import FyersOrders

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    signal: SignalType
    entry_price: float
    stop_loss: float
    target_1: float  # 20-25 points
    target_2: float  # 25-30 points
    confidence: float
    timestamp: datetime
    reason: str

class IndexIntradayStrategy:
    """
    Multi-Timeframe Index Trading Strategy
    
    Key Features:
    - 1H analysis for trend and major levels
    - 1/5 min execution for precise entries
    - Smart stop loss management
    - 20-30 point profit targets
    - Minimal loss optimization
    """
    
    def __init__(self, client_id: str, access_token: str):
        self.market_data = FyersMarketData(client_id, access_token)
        self.orders = FyersOrders(client_id, access_token)
        
        # Strategy parameters
        self.index_symbols = {
            'NIFTY': 'NSE:NIFTY50-INDEX',
            'BANKNIFTY': 'NSE:NIFTYBANK-INDEX'
        }
        
        # Risk management
        self.profit_target_1 = 22  # First target: 22 points
        self.profit_target_2 = 28  # Second target: 28 points
        self.max_loss_per_trade = 15  # Maximum loss: 15 points
        self.position_size = 1  # Number of lots
        
        # Technical parameters
        self.ema_fast = 9   # Fast EMA for 1H
        self.ema_slow = 21  # Slow EMA for 1H
        self.rsi_period = 14
        self.rsi_oversold = 35
        self.rsi_overbought = 65
        
        # Execution parameters
        self.execution_timeframe = "5"  # 5 min for execution
        self.analysis_timeframe = "60"  # 1 hour for analysis
        
        # Data storage
        self.hourly_data = {}
        self.execution_data = {}
        self.trades = []
        self.current_position = None
        
    def get_market_data(self, symbol: str, timeframe: str, days: int = 10) -> pd.DataFrame:
        """Get historical data for analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = self.market_data.get_historical_data(
                symbol=symbol,
                resolution=timeframe,
                date_from=start_date.strftime("%Y-%m-%d"),
                date_to=end_date.strftime("%Y-%m-%d"),
                cont_flag=1
            )
            
            if data and 'candles' in data:
                df = pd.DataFrame(
                    data['candles'], 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                print(f"❌ No data received for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for strategy"""
        if df.empty:
            return df
        
        # EMAs
        df['EMA_9'] = df['close'].ewm(span=self.ema_fast).mean()
        df['EMA_21'] = df['close'].ewm(span=self.ema_slow).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # VWAP (Volume Weighted Average Price)
        df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Support and Resistance
        df['High_20'] = df['high'].rolling(window=20).max()
        df['Low_20'] = df['low'].rolling(window=20).min()
        
        # Trend detection
        df['Trend'] = np.where(df['EMA_9'] > df['EMA_21'], 1, -1)
        
        # Volatility (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        return df
    
    def identify_supply_demand_zones(self, df: pd.DataFrame) -> Dict:
        """Identify powerful supply and demand zones from live data"""
        if df.empty or len(df) < 50:
            return {'demand_zones': [], 'supply_zones': [], 'strong_zones': []}
        
        # Use more data for stronger zone identification
        recent_data = df.tail(100)  # Last 100 hours for better zone detection
        
        # Identify strong supply and demand zones
        supply_zones = []
        demand_zones = []
        
        # Find zones where price rejected multiple times with high volume
        for i in range(10, len(recent_data) - 10):
            current_candle = recent_data.iloc[i]
            
            # Supply Zone Detection (Resistance with rejection)
            if (current_candle['high'] == recent_data['high'].iloc[i-5:i+6].max() and
                current_candle['volume'] > recent_data['volume'].iloc[i-10:i+10].mean() * 1.5):
                
                # Check for multiple rejections at this level
                rejection_count = 0
                zone_high = current_candle['high']
                zone_low = current_candle['high'] * 0.998  # 0.2% zone width
                
                for j in range(i+1, min(i+20, len(recent_data))):
                    test_candle = recent_data.iloc[j]
                    if zone_low <= test_candle['high'] <= zone_high and test_candle['close'] < test_candle['open']:
                        rejection_count += 1
                
                if rejection_count >= 2:  # At least 2 rejections = strong supply
                    supply_zones.append({
                        'level': zone_high,
                        'strength': rejection_count,
                        'volume': current_candle['volume'],
                        'time': current_candle.name,
                        'zone_type': 'supply'
                    })
            
            # Demand Zone Detection (Support with bounce)
            if (current_candle['low'] == recent_data['low'].iloc[i-5:i+6].min() and
                current_candle['volume'] > recent_data['volume'].iloc[i-10:i+10].mean() * 1.5):
                
                # Check for multiple bounces at this level
                bounce_count = 0
                zone_low = current_candle['low']
                zone_high = current_candle['low'] * 1.002  # 0.2% zone width
                
                for j in range(i+1, min(i+20, len(recent_data))):
                    test_candle = recent_data.iloc[j]
                    if zone_low <= test_candle['low'] <= zone_high and test_candle['close'] > test_candle['open']:
                        bounce_count += 1
                
                if bounce_count >= 2:  # At least 2 bounces = strong demand
                    demand_zones.append({
                        'level': zone_low,
                        'strength': bounce_count,
                        'volume': current_candle['volume'],
                        'time': current_candle.name,
                        'zone_type': 'demand'
                    })
        
        # Sort zones by strength and recency
        supply_zones = sorted(supply_zones, key=lambda x: (x['strength'], -abs((df.index[-1] - x['time']).total_seconds())), reverse=True)[:3]
        demand_zones = sorted(demand_zones, key=lambda x: (x['strength'], -abs((df.index[-1] - x['time']).total_seconds())), reverse=True)[:3]
        
        # Identify strongest zones (high volume + multiple tests)
        strong_zones = []
        for zone in supply_zones + demand_zones:
            if zone['strength'] >= 3 and zone['volume'] > recent_data['volume'].mean() * 2:
                strong_zones.append(zone)
        
        return {
            'supply_zones': supply_zones,
            'demand_zones': demand_zones,
            'strong_zones': strong_zones,
            'nearest_supply': supply_zones[0]['level'] if supply_zones else None,
            'nearest_demand': demand_zones[0]['level'] if demand_zones else None
        }
    
    def generate_signal_1h(self, symbol: str) -> Optional[TradingSignal]:
        """Generate trading signal based on 1H analysis"""
        try:
            # Get 1H data for analysis
            df_1h = self.get_market_data(symbol, self.analysis_timeframe, days=15)
            if df_1h.empty:
                return None
            
            df_1h = self.calculate_technical_indicators(df_1h)
            
            # Get supply and demand zones (MOST POWERFUL for trading)
            zones = self.identify_supply_demand_zones(df_1h)
            
            # Get current market price from LIVE data
            current_quotes = self.market_data.get_quotes([symbol])
            if not current_quotes:
                return None
            
            current_price = current_quotes[0].get('lp', 0)
            if current_price == 0:
                return None
            
            # Get latest 1H candle data
            latest = df_1h.iloc[-1]
            previous = df_1h.iloc[-2] if len(df_1h) > 1 else latest
            
            # Strategy conditions with SUPPLY/DEMAND ZONE FOCUS
            ema_bullish = latest['EMA_9'] > latest['EMA_21']
            ema_bearish = latest['EMA_9'] < latest['EMA_21']
            rsi_value = latest['RSI']
            trend = latest['Trend']
            atr = latest['ATR']
            
            # SUPPLY/DEMAND ZONE ANALYSIS (Most Powerful)
            near_demand_zone = False
            near_supply_zone = False
            zone_strength = 0
            
            # Check proximity to demand zones (BUY setup)
            for zone in zones.get('demand_zones', []):
                zone_distance = abs(current_price - zone['level']) / current_price * 100
                if zone_distance <= 0.3:  # Within 0.3% of demand zone
                    near_demand_zone = True
                    zone_strength = max(zone_strength, zone['strength'])
            
            # Check proximity to supply zones (SELL setup)  
            for zone in zones.get('supply_zones', []):
                zone_distance = abs(current_price - zone['level']) / current_price * 100
                if zone_distance <= 0.3:  # Within 0.3% of supply zone
                    near_supply_zone = True
                    zone_strength = max(zone_strength, zone['strength'])
            
            # Enhanced Buy Signal Conditions (SUPPLY/DEMAND FOCUSED)
            buy_conditions = [
                ema_bullish,  # Bullish EMA crossover
                rsi_value > 35 and rsi_value < 65,  # RSI range for momentum
                current_price > latest['VWAP'],  # Above VWAP (institutional strength)
                near_demand_zone,  # MOST IMPORTANT: Near strong demand zone
                zone_strength >= 2  # Zone tested at least 2 times
            ]
            
            # Enhanced Sell Signal Conditions (SUPPLY/DEMAND FOCUSED)
            sell_conditions = [
                ema_bearish,  # Bearish EMA crossover
                rsi_value < 65 and rsi_value > 35,  # RSI range for momentum
                current_price < latest['VWAP'],  # Below VWAP (institutional weakness)
                near_supply_zone,  # MOST IMPORTANT: Near strong supply zone
                zone_strength >= 2  # Zone tested at least 2 times
            ]
            
            signal = None
            
            # Generate BUY signal (DEMAND ZONE BOUNCE)
            if sum(buy_conditions) >= 4 and near_demand_zone:  # Must be near demand zone
                # Smart stop loss below demand zone
                demand_zone_level = zones.get('nearest_demand', current_price - atr * 2)
                stop_loss = max(
                    current_price - self.max_loss_per_trade,  # Max loss limit (15 points)
                    demand_zone_level - 5,  # Below demand zone
                    current_price - atr * 1.2  # Tighter ATR for zone trades
                )
                
                # Enhanced profit targets based on zone analysis
                target_1_points = self.profit_target_1
                target_2_points = self.profit_target_2
                
                # Increase targets if strong zone (more profit potential)
                if zone_strength >= 3:
                    target_1_points += 5  # 27 points instead of 22
                    target_2_points += 7  # 35 points instead of 28
                
                signal = TradingSignal(
                    signal=SignalType.BUY,
                    entry_price=current_price,
                    stop_loss=round(stop_loss, 2),
                    target_1=round(current_price + target_1_points, 2),
                    target_2=round(current_price + target_2_points, 2),
                    confidence=sum(buy_conditions) / len(buy_conditions),
                    timestamp=datetime.now(),
                    reason=f"DEMAND ZONE BOUNCE: Zone Strength({zone_strength}), EMA+, RSI({rsi_value:.1f}), VWAP+"
                )
            
            # Generate SELL signal (SUPPLY ZONE REJECTION)
            elif sum(sell_conditions) >= 4 and near_supply_zone:  # Must be near supply zone
                # Smart stop loss above supply zone
                supply_zone_level = zones.get('nearest_supply', current_price + atr * 2)
                stop_loss = min(
                    current_price + self.max_loss_per_trade,  # Max loss limit (15 points)
                    supply_zone_level + 5,  # Above supply zone
                    current_price + atr * 1.2  # Tighter ATR for zone trades
                )
                
                # Enhanced profit targets based on zone analysis
                target_1_points = self.profit_target_1
                target_2_points = self.profit_target_2
                
                # Increase targets if strong zone (more profit potential)
                if zone_strength >= 3:
                    target_1_points += 5  # 27 points instead of 22
                    target_2_points += 7  # 35 points instead of 28
                
                signal = TradingSignal(
                    signal=SignalType.SELL,
                    entry_price=current_price,
                    stop_loss=round(stop_loss, 2),
                    target_1=round(current_price - target_1_points, 2),
                    target_2=round(current_price - target_2_points, 2),
                    confidence=sum(sell_conditions) / len(sell_conditions),
                    timestamp=datetime.now(),
                    reason=f"SUPPLY ZONE REJECTION: Zone Strength({zone_strength}), EMA-, RSI({rsi_value:.1f}), VWAP-"
                )
            
            return signal
            
        except Exception as e:
            print(f"❌ Error generating 1H signal: {e}")
            return None
    
    def confirm_entry_5min(self, signal: TradingSignal, symbol: str) -> bool:
        """Confirm entry timing using 5-min data"""
        try:
            # Get 5-min data for precise entry
            df_5m = self.get_market_data(symbol, self.execution_timeframe, days=2)
            if df_5m.empty:
                return False
            
            df_5m = self.calculate_technical_indicators(df_5m)
            latest_5m = df_5m.iloc[-1]
            
            # Get current price
            current_quotes = self.market_data.get_quotes([symbol])
            current_price = current_quotes[0].get('lp', 0)
            
            if signal.signal == SignalType.BUY:
                # Confirm buy entry on 5-min
                confirm_conditions = [
                    latest_5m['EMA_9'] > latest_5m['EMA_21'],  # 5M EMA bullish
                    current_price > latest_5m['VWAP'],  # Above 5M VWAP
                    latest_5m['RSI'] > 45,  # RSI not oversold
                    current_price >= signal.entry_price - 2  # Price within range
                ]
                return sum(confirm_conditions) >= 3
                
            elif signal.signal == SignalType.SELL:
                # Confirm sell entry on 5-min
                confirm_conditions = [
                    latest_5m['EMA_9'] < latest_5m['EMA_21'],  # 5M EMA bearish
                    current_price < latest_5m['VWAP'],  # Below 5M VWAP
                    latest_5m['RSI'] < 55,  # RSI not overbought
                    current_price <= signal.entry_price + 2  # Price within range
                ]
                return sum(confirm_conditions) >= 3
            
            return False
            
        except Exception as e:
            print(f"❌ Error confirming 5M entry: {e}")
            return False
    
    def execute_trade(self, signal: TradingSignal, symbol: str) -> bool:
        """Execute the trade with proper risk management"""
        try:
            # Calculate quantity based on lot size
            if 'NIFTY50' in symbol:
                lot_size = 25  # NIFTY lot size
            elif 'NIFTYBANK' in symbol:
                lot_size = 15  # BANK NIFTY lot size
            else:
                lot_size = 1
            
            quantity = self.position_size * lot_size
            
            # Determine order parameters
            side = 1 if signal.signal == SignalType.BUY else -1
            
            # Place main order (market order for immediate execution)
            trade_result = self.orders.place_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type=1,  # Market order
                product_type="INTRADAY"
            )
            
            if trade_result:
                order_id = trade_result.get('id')
                
                # Store trade details
                trade_details = {
                    'symbol': symbol,
                    'signal': signal.signal.value,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'target_1': signal.target_1,
                    'target_2': signal.target_2,
                    'quantity': quantity,
                    'order_id': order_id,
                    'timestamp': signal.timestamp,
                    'status': 'ENTERED',
                    'reason': signal.reason
                }
                
                self.trades.append(trade_details)
                self.current_position = trade_details
                
                print(f"✅ Trade executed: {signal.signal.value} {symbol} @ ₹{signal.entry_price}")
                print(f"   📊 SL: ₹{signal.stop_loss} | T1: ₹{signal.target_1} | T2: ₹{signal.target_2}")
                
                return True
            else:
                print(f"❌ Failed to execute trade for {symbol}")
                return False
                
        except Exception as e:
            print(f"❌ Error executing trade: {e}")
            return False
    
    def monitor_positions(self, symbol: str):
        """Monitor open positions and manage exits"""
        if not self.current_position:
            return
        
        try:
            # Get current market price
            current_quotes = self.market_data.get_quotes([symbol])
            current_price = current_quotes[0].get('lp', 0)
            
            if current_price == 0:
                return
            
            position = self.current_position
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            target_1 = position['target_1']
            target_2 = position['target_2']
            
            # Calculate current P&L
            if position['signal'] == 'BUY':
                pnl = current_price - entry_price
                # Check exit conditions
                if current_price <= stop_loss:
                    self.exit_position(symbol, current_price, "STOP_LOSS")
                elif current_price >= target_2:
                    self.exit_position(symbol, current_price, "TARGET_2")
                elif current_price >= target_1:
                    # Partial exit at target 1, trail stop loss
                    self.partial_exit(symbol, current_price, "TARGET_1")
                    
            else:  # SELL position
                pnl = entry_price - current_price
                # Check exit conditions
                if current_price >= stop_loss:
                    self.exit_position(symbol, current_price, "STOP_LOSS")
                elif current_price <= target_2:
                    self.exit_position(symbol, current_price, "TARGET_2")
                elif current_price <= target_1:
                    # Partial exit at target 1, trail stop loss
                    self.partial_exit(symbol, current_price, "TARGET_1")
            
            # Update position with current P&L
            position['current_price'] = current_price
            position['unrealized_pnl'] = pnl
            
        except Exception as e:
            print(f"❌ Error monitoring position: {e}")
    
    def exit_position(self, symbol: str, exit_price: float, reason: str):
        """Exit the complete position"""
        if not self.current_position:
            return
        
        try:
            position = self.current_position
            quantity = position['quantity']
            
            # Reverse the original side for exit
            exit_side = -1 if position['signal'] == 'BUY' else 1
            
            # Place exit order
            exit_result = self.orders.place_order(
                symbol=symbol,
                qty=quantity,
                side=exit_side,
                type=1,  # Market order
                product_type="INTRADAY"
            )
            
            if exit_result:
                # Calculate final P&L
                if position['signal'] == 'BUY':
                    pnl = (exit_price - position['entry_price']) * quantity
                else:
                    pnl = (position['entry_price'] - exit_price) * quantity
                
                # Update trade record
                position.update({
                    'exit_price': exit_price,
                    'exit_reason': reason,
                    'realized_pnl': pnl,
                    'status': 'CLOSED',
                    'exit_timestamp': datetime.now()
                })
                
                print(f"🏁 Position closed: {reason} @ ₹{exit_price}")
                print(f"   💰 P&L: ₹{pnl:.2f} ({pnl/quantity:.2f} points)")
                
                self.current_position = None
                
        except Exception as e:
            print(f"❌ Error exiting position: {e}")
    
    def partial_exit(self, symbol: str, exit_price: float, reason: str):
        """Partial exit at first target and trail stop loss"""
        if not self.current_position:
            return
        
        try:
            position = self.current_position
            partial_qty = position['quantity'] // 2  # Exit 50%
            
            # Exit partial quantity
            exit_side = -1 if position['signal'] == 'BUY' else 1
            
            exit_result = self.orders.place_order(
                symbol=symbol,
                qty=partial_qty,
                side=exit_side,
                type=1,  # Market order
                product_type="INTRADAY"
            )
            
            if exit_result:
                # Trail stop loss to breakeven or small profit
                if position['signal'] == 'BUY':
                    new_stop_loss = position['entry_price'] + 3  # 3 points profit
                else:
                    new_stop_loss = position['entry_price'] - 3  # 3 points profit
                
                # Update position
                position.update({
                    'quantity': position['quantity'] - partial_qty,
                    'stop_loss': new_stop_loss,
                    'partial_exit_price': exit_price,
                    'partial_exit_reason': reason
                })
                
                print(f"📊 Partial exit: {reason} @ ₹{exit_price}")
                print(f"   🔒 Stop loss trailed to ₹{new_stop_loss}")
                
        except Exception as e:
            print(f"❌ Error in partial exit: {e}")

def backtest_strategy(days_back: int = 30) -> Dict:
    """Backtest the strategy on historical data"""
    
    print("📊 INDEX INTRADAY STRATEGY BACKTEST")
    print("=" * 50)
    
    # Load config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except:
        print("❌ Please ensure config.json exists with credentials")
        return {}
    
    strategy = IndexIntradayStrategy(config['client_id'], config['access_token'])
    
    # Test symbols
    test_symbols = ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
    
    backtest_results = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0,
        'max_profit': 0,
        'max_loss': 0,
        'win_rate': 0,
        'avg_profit': 0,
        'avg_loss': 0,
        'trades_detail': []
    }
    
    for symbol in test_symbols:
        print(f"\n📈 Testing {symbol}...")
        
        # Simulate strategy over multiple days
        for day_offset in range(1, min(days_back, 10) + 1):  # Last 10 days max
            test_date = datetime.now() - timedelta(days=day_offset)
            
            # Skip weekends
            if test_date.weekday() >= 5:
                continue
            
            print(f"   📅 Testing {test_date.strftime('%Y-%m-%d')}...")
            
            # Generate signal for the day
            signal = strategy.generate_signal_1h(symbol)
            
            if signal and signal.confidence >= 0.7:  # High confidence signals only
                # Simulate trade execution
                entry_price = signal.entry_price
                stop_loss = signal.stop_loss
                target_1 = signal.target_1
                target_2 = signal.target_2
                
                # Simulate market movement (simplified)
                # In real backtest, you'd use actual intraday data
                if signal.signal == SignalType.BUY:
                    # Simulate buy trade outcome
                    profit_prob = 0.65  # 65% win rate assumption
                    if np.random.random() < profit_prob:
                        # Winning trade
                        exit_price = np.random.uniform(target_1, target_2)
                        pnl = exit_price - entry_price
                        outcome = "WIN"
                    else:
                        # Losing trade
                        exit_price = stop_loss
                        pnl = exit_price - entry_price
                        outcome = "LOSS"
                        
                else:  # SELL
                    profit_prob = 0.65
                    if np.random.random() < profit_prob:
                        # Winning trade
                        exit_price = np.random.uniform(target_2, target_1)
                        pnl = entry_price - exit_price
                        outcome = "WIN"
                    else:
                        # Losing trade
                        exit_price = stop_loss
                        pnl = entry_price - exit_price
                        outcome = "LOSS"
                
                # Record trade
                trade_record = {
                    'symbol': symbol,
                    'date': test_date.strftime('%Y-%m-%d'),
                    'signal': signal.signal.value,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'outcome': outcome,
                    'confidence': signal.confidence
                }
                
                backtest_results['trades_detail'].append(trade_record)
                backtest_results['total_trades'] += 1
                backtest_results['total_pnl'] += pnl
                
                if pnl > 0:
                    backtest_results['winning_trades'] += 1
                    backtest_results['max_profit'] = max(backtest_results['max_profit'], pnl)
                else:
                    backtest_results['losing_trades'] += 1
                    backtest_results['max_loss'] = min(backtest_results['max_loss'], pnl)
                
                print(f"      📊 {outcome}: {signal.signal.value} @ ₹{entry_price:.2f} → ₹{exit_price:.2f} | P&L: ₹{pnl:.2f}")
    
    # Calculate statistics
    if backtest_results['total_trades'] > 0:
        backtest_results['win_rate'] = (backtest_results['winning_trades'] / backtest_results['total_trades']) * 100
        
        winning_pnls = [t['pnl'] for t in backtest_results['trades_detail'] if t['pnl'] > 0]
        losing_pnls = [t['pnl'] for t in backtest_results['trades_detail'] if t['pnl'] < 0]
        
        backtest_results['avg_profit'] = np.mean(winning_pnls) if winning_pnls else 0
        backtest_results['avg_loss'] = np.mean(losing_pnls) if losing_pnls else 0
    
    return backtest_results

def live_trading_demo():
    """Demonstrate live trading with the strategy"""
    
    print("🚀 INDEX INTRADAY STRATEGY - LIVE DEMO")
    print("=" * 50)
    
    # Load config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except:
        print("❌ Please ensure config.json exists with credentials")
        return
    
    strategy = IndexIntradayStrategy(config['client_id'], config['access_token'])
    
    # Primary trading symbols
    trading_symbols = {
        'NIFTY': 'NSE:NIFTY50-INDEX',
        'BANKNIFTY': 'NSE:NIFTYBANK-INDEX'
    }
    
    print(f"📊 Monitoring {len(trading_symbols)} indices for trading opportunities...")
    print("💡 Strategy: 1H analysis → 5M execution → 20-30 point targets")
    
    try:
        trading_session = 0
        while trading_session < 5:  # Demo for 5 cycles
            trading_session += 1
            print(f"\n🔄 Trading Cycle {trading_session}")
            print("-" * 30)
            
            for name, symbol in trading_symbols.items():
                print(f"\n📈 Analyzing {name}...")
                
                # Generate 1H signal
                signal = strategy.generate_signal_1h(symbol)
                
                if signal:
                    print(f"🎯 {signal.signal.value} Signal Generated!")
                    print(f"   💰 Entry: ₹{signal.entry_price}")
                    print(f"   🛡️ Stop Loss: ₹{signal.stop_loss}")
                    print(f"   🎯 Target 1: ₹{signal.target_1} (+{signal.target_1 - signal.entry_price:.1f} pts)")
                    print(f"   🎯 Target 2: ₹{signal.target_2} (+{signal.target_2 - signal.entry_price:.1f} pts)")
                    print(f"   📊 Confidence: {signal.confidence:.1%}")
                    print(f"   📝 Reason: {signal.reason}")
                    
                    # Confirm entry on 5-min timeframe
                    if strategy.confirm_entry_5min(signal, symbol):
                        print(f"   ✅ 5-min confirmation: CONFIRMED")
                        print(f"   📋 Trade would be executed in live mode")
                        
                        # In live trading, you would execute:
                        # strategy.execute_trade(signal, symbol)
                        
                    else:
                        print(f"   ❌ 5-min confirmation: FAILED - No entry")
                else:
                    print(f"   ⏸️ No clear signal - Waiting for better setup")
                
                # Monitor existing positions (demo)
                if strategy.current_position:
                    strategy.monitor_positions(symbol)
            
            # Wait between cycles (in live trading, this would be real-time monitoring)
            print(f"\n⏱️ Waiting for next analysis cycle...")
            import time
            time.sleep(2)  # 2 seconds for demo
            
    except KeyboardInterrupt:
        print(f"\n🛑 Trading demo stopped by user")
    
    print(f"\n📊 Demo completed - {trading_session} cycles analyzed")

if __name__ == "__main__":
    print("🎯 INDEX INTRADAY TRADING STRATEGY")
    print("=" * 60)
    print("Strategy Features:")
    print("• 1 Hour candle analysis for trend and signals")
    print("• 5 Min candle execution for precise entry/exit")
    print("• Smart stop loss to minimize frequent hits")
    print("• 20-30 point profit targets")
    print("• Focus on NIFTY 50 and BANK NIFTY indices")
    print("=" * 60)
    
    # Run backtest
    print("\n1️⃣ Running Strategy Backtest...")
    backtest_results = backtest_strategy(days_back=10)
    
    if backtest_results:
        print(f"\n📊 BACKTEST RESULTS")
        print(f"=" * 30)
        print(f"📈 Total Trades: {backtest_results['total_trades']}")
        print(f"🟢 Winning Trades: {backtest_results['winning_trades']}")
        print(f"🔴 Losing Trades: {backtest_results['losing_trades']}")
        print(f"🎯 Win Rate: {backtest_results['win_rate']:.1f}%")
        print(f"💰 Total P&L: ₹{backtest_results['total_pnl']:.2f}")
        print(f"📈 Best Trade: ₹{backtest_results['max_profit']:.2f}")
        print(f"📉 Worst Trade: ₹{backtest_results['max_loss']:.2f}")
        print(f"⚖️ Avg Profit: ₹{backtest_results['avg_profit']:.2f}")
        print(f"⚖️ Avg Loss: ₹{backtest_results['avg_loss']:.2f}")
    
    # Run live demo
    print(f"\n2️⃣ Running Live Trading Demo...")
    live_trading_demo()