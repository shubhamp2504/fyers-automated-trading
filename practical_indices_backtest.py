"""
🎯 PRACTICAL INDICES INTRADAY BACKTEST
Realistic backtesting using available data and tradeable instruments
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fyers_client import FyersClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import random

class PracticalIndicesBacktester:
    """Practical backtester using available data and realistic simulations"""
    
    def __init__(self, initial_capital: float = 200000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.positions = []
        
        # Use futures symbols which should have better data availability
        self.trading_symbols = {
            "NIFTY_FUT": "NSE:NIFTY50-INDEX",  # We'll use index data but simulate futures trading
            "BANKNIFTY_FUT": "NSE:NIFTYBANK-INDEX",
            # Add some major stocks for comparison
            "RELIANCE": "NSE:RELIANCE-EQ",
            "INFY": "NSE:INFY-EQ",
            "TCS": "NSE:TCS-EQ"
        }
        
        # Practical trading parameters
        self.config = {
            "risk_per_trade": 0.02,      # 2% risk per trade
            "max_positions": 2,          # Max 2 positions
            "stop_loss_pct": 2.0,        # 2% stop loss
            "target_pct_1": 2.5,         # 2.5% first target
            "target_pct_2": 4.0,         # 4% second target
            "min_confidence": 65,        # Min 65% confidence
            "commission_per_trade": 50,  # ₹50 commission
            "slippage_pct": 0.1,         # 0.1% slippage
            "futures_multiplier": {      # Multipliers for P&L calculation
                "NIFTY_FUT": 50,
                "BANKNIFTY_FUT": 15,
                "RELIANCE": 1,
                "INFY": 1,
                "TCS": 1
            }
        }
        
        print("🎯 PRACTICAL INDICES INTRADAY BACKTESTER INITIALIZED")
        print(f"💰 Initial Capital: ₹{self.initial_capital:,.2f}")
        print(f"📊 Symbols: {list(self.trading_symbols.keys())}")
    
    def simulate_intraday_from_daily(self, daily_df: pd.DataFrame):
        """Simulate intraday price movements from daily data"""
        
        intraday_data = []
        
        for i, row in daily_df.iterrows():
            # Simulate 6 hourly candles (10 AM to 3 PM)
            hourly_prices = []
            
            # Calculate daily range and volatility
            daily_range = row['high'] - row['low']
            open_price = row['open']
            close_price = row['close']
            
            # Generate realistic intraday progression
            price_progression = [open_price]
            
            # Simulate 5 hourly closes leading to daily close
            for hour in range(5):
                # Add some randomness but trend towards close
                trend_factor = (close_price - open_price) / 5
                random_factor = (random.random() - 0.5) * daily_range * 0.3
                
                next_price = price_progression[-1] + trend_factor + random_factor
                
                # Ensure price stays within daily high/low
                next_price = max(row['low'], min(row['high'], next_price))
                price_progression.append(next_price)
            
            # Create hourly candles
            for hour in range(5):
                hour_open = price_progression[hour]
                hour_close = price_progression[hour + 1]
                
                # Calculate hour high/low based on volatility
                hour_range = daily_range * 0.2 * random.uniform(0.5, 1.5)
                hour_high = max(hour_open, hour_close) + hour_range * 0.5
                hour_low = min(hour_open, hour_close) - hour_range * 0.5
                
                # Ensure within daily limits
                hour_high = min(row['high'], hour_high)
                hour_low = max(row['low'], hour_low)
                
                # Create hourly timestamp
                hour_timestamp = pd.Timestamp(i) + pd.Timedelta(hours=10 + hour)
                
                intraday_data.append({
                    'timestamp': hour_timestamp,
                    'open': hour_open,
                    'high': hour_high,
                    'low': hour_low,
                    'close': hour_close,
                    'volume': row.get('volume', 100000) / 5  # Distribute volume
                })
        
        return pd.DataFrame(intraday_data).set_index('timestamp')
    
    def generate_practical_signals(self, intraday_df: pd.DataFrame, symbol: str):
        """Generate practical trading signals from simulated intraday data"""
        
        if len(intraday_df) < 10:  # Reduce requirement
            return []
        
        signals = []
        
        # Add simple technical indicators
        intraday_df['SMA_5'] = intraday_df['close'].rolling(window=5).mean()
        intraday_df['SMA_10'] = intraday_df['close'].rolling(window=10).mean()
        intraday_df['RSI'] = self.calculate_rsi(intraday_df['close'], 14)
        
        for i in range(5, len(intraday_df)):  # Start earlier
            current = intraday_df.iloc[i]
            
            # Only trade during market hours (10 AM to 2 PM for safety)
            if current.name.hour < 10 or current.name.hour > 14:
                continue
            
            # Calculate momentum
            price_change = (current['close'] - intraday_df.iloc[i-5]['close']) / intraday_df.iloc[i-5]['close'] * 100
            
            # BUY Signal: Price above SMA_5, SMA_5 > SMA_10, RSI between 40-70, positive momentum
            if (current['close'] > current['SMA_5'] and 
                current['SMA_5'] > current['SMA_10'] and
                40 < current['RSI'] < 70 and
                price_change > 0.5):
                
                confidence = min(85, 60 + abs(price_change) * 3)
                
                if confidence >= self.config['min_confidence']:
                    entry_price = current['close'] * (1 + self.config['slippage_pct'] / 100)
                    
                    signals.append({
                        'timestamp': current.name,
                        'symbol': symbol,
                        'signal': 'BUY',
                        'entry_price': entry_price,
                        'stop_loss': entry_price * (1 - self.config['stop_loss_pct'] / 100),
                        'target_1': entry_price * (1 + self.config['target_pct_1'] / 100),
                        'target_2': entry_price * (1 + self.config['target_pct_2'] / 100),
                        'confidence': confidence,
                        'rsi': current['RSI'],
                        'reason': f"BUY: Momentum {price_change:.1f}%, RSI {current['RSI']:.0f}"
                    })
            
            # SELL Signal: Price below SMA_5, SMA_5 < SMA_10, RSI between 30-60, negative momentum
            elif (current['close'] < current['SMA_5'] and 
                  current['SMA_5'] < current['SMA_10'] and
                  30 < current['RSI'] < 60 and
                  price_change < -0.5):
                
                confidence = min(85, 60 + abs(price_change) * 3)
                
                if confidence >= self.config['min_confidence']:
                    entry_price = current['close'] * (1 - self.config['slippage_pct'] / 100)
                    
                    signals.append({
                        'timestamp': current.name,
                        'symbol': symbol,
                        'signal': 'SELL',
                        'entry_price': entry_price,
                        'stop_loss': entry_price * (1 + self.config['stop_loss_pct'] / 100),
                        'target_1': entry_price * (1 - self.config['target_pct_1'] / 100),
                        'target_2': entry_price * (1 - self.config['target_pct_2'] / 100),
                        'confidence': confidence,
                        'rsi': current['RSI'],
                        'reason': f"SELL: Momentum {price_change:.1f}%, RSI {current['RSI']:.0f}"
                    })
        
        return signals
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def execute_practical_trade(self, signal: dict):
        """Execute trade with practical position sizing"""
        
        symbol = signal['symbol']
        multiplier = self.config['futures_multiplier'].get(symbol, 1)
        
        # Calculate position size based on risk
        if signal['signal'] == 'BUY':
            risk_per_unit = signal['entry_price'] - signal['stop_loss']
        else:
            risk_per_unit = signal['stop_loss'] - signal['entry_price']
        
        if risk_per_unit <= 0:
            return None
        
        risk_amount = self.current_capital * self.config['risk_per_trade']
        
        # For futures, calculate lot size
        if symbol in ['NIFTY_FUT', 'BANKNIFTY_FUT']:
            lots = max(1, min(3, int(risk_amount / (risk_per_unit * multiplier))))
            quantity = lots * multiplier
        else:  # For stocks
            quantity = max(1, int(risk_amount / risk_per_unit))
            lots = 1
        
        # Ensure we don't use more than 80% of capital
        required_margin = signal['entry_price'] * quantity * 0.2  # 20% margin
        if required_margin > self.current_capital * 0.8:
            return None
        
        trade = {
            'id': len(self.trades) + 1,
            'timestamp': signal['timestamp'],
            'symbol': symbol,
            'signal': signal['signal'],
            'entry_price': signal['entry_price'],
            'quantity': quantity,
            'lots': lots if symbol in ['NIFTY_FUT', 'BANKNIFTY_FUT'] else 1,
            'multiplier': multiplier,
            'stop_loss': signal['stop_loss'],
            'target_1': signal['target_1'],
            'target_2': signal['target_2'],
            'confidence': signal['confidence'],
            'rsi': signal['rsi'],
            'reason': signal['reason'],
            'status': 'OPEN',
            'pnl': 0,
            'commission': self.config['commission_per_trade']
        }
        
        self.trades.append(trade)
        self.positions.append(trade.copy())
        
        print(f"📈 Trade #{trade['id']}: {trade['signal']} {trade['symbol']} @ ₹{trade['entry_price']:.2f}")
        if symbol in ['NIFTY_FUT', 'BANKNIFTY_FUT']:
            print(f"   📊 Lots: {lots} | Qty: {quantity} | Confidence: {trade['confidence']:.0f}%")
        else:
            print(f"   📊 Qty: {quantity} | Confidence: {trade['confidence']:.0f}%")
        
        return trade
    
    def simulate_trade_exit(self, trade: dict, intraday_df: pd.DataFrame):
        """Simulate trade exit using intraday data"""
        
        if trade['status'] != 'OPEN':
            return
        
        trade_start_idx = intraday_df.index.get_loc(trade['timestamp'])
        
        # Check subsequent candles for exit conditions
        for i in range(trade_start_idx + 1, min(trade_start_idx + 20, len(intraday_df))):  # Max 4 hours
            candle = intraday_df.iloc[i]
            
            exit_price = None
            exit_reason = ""
            
            if trade['signal'] == 'BUY':
                if candle['low'] <= trade['stop_loss']:
                    exit_price = trade['stop_loss']
                    exit_reason = "Stop Loss"
                elif candle['high'] >= trade['target_2']:
                    exit_price = trade['target_2']
                    exit_reason = "Target 2"
                elif candle['high'] >= trade['target_1']:
                    exit_price = trade['target_1']
                    exit_reason = "Target 1"
            else:  # SELL
                if candle['high'] >= trade['stop_loss']:
                    exit_price = trade['stop_loss']
                    exit_reason = "Stop Loss"
                elif candle['low'] <= trade['target_2']:
                    exit_price = trade['target_2']
                    exit_reason = "Target 2"
                elif candle['low'] <= trade['target_1']:
                    exit_price = trade['target_1']
                    exit_reason = "Target 1"
            
            if exit_price:
                self.close_practical_position(trade, exit_price, exit_reason)
                return
        
        # If no exit condition met, close at end of day
        final_price = intraday_df.iloc[-1]['close']
        self.close_practical_position(trade, final_price, "EOD Exit")
    
    def close_practical_position(self, trade: dict, exit_price: float, exit_reason: str):
        """Close position and calculate P&L"""
        
        # Calculate P&L with multiplier
        if trade['signal'] == 'BUY':
            pnl = (exit_price - trade['entry_price']) * trade['quantity']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['quantity']
        
        pnl -= trade['commission']
        
        trade['exit_price'] = exit_price
        trade['exit_reason'] = exit_reason
        trade['pnl'] = pnl
        trade['status'] = 'CLOSED'
        
        self.current_capital += pnl
        
        # Remove from positions
        self.positions = [p for p in self.positions if p['id'] != trade['id']]
        
        pnl_emoji = "💚" if pnl > 0 else "❤️"
        print(f"{pnl_emoji} Trade #{trade['id']} CLOSED: {exit_reason} @ ₹{exit_price:.2f} | P&L: ₹{pnl:,.0f}")
    
    def run_practical_backtest(self, start_date: str, end_date: str):
        """Run practical backtest with available data"""
        
        print(f"\n🚀 RUNNING PRACTICAL INDICES INTRADAY BACKTEST")
        print(f"📅 Period: {start_date} to {end_date}")
        print("=" * 70)
        
        all_signals = []
        
        # Test each symbol
        for symbol_name, symbol_code in self.trading_symbols.items():
            print(f"\n📊 Testing {symbol_name}...")
            
            # Get daily data
            try:
                fyers = FyersClient()
                df = fyers.get_historical_data(symbol_code, "1D", start_date, end_date)
                
                if df is not None and len(df) > 5:  # Reduce requirement to 5 days
                    print(f"✅ Loaded {len(df)} daily candles for {symbol_name}")
                    
                    # Simulate intraday data
                    intraday_df = self.simulate_intraday_from_daily(df)
                    print(f"✅ Generated {len(intraday_df)} simulated intraday candles")
                    
                    # Generate signals
                    signals = self.generate_practical_signals(intraday_df, symbol_name)
                    print(f"✅ Generated {len(signals)} trading signals")
                    
                    # Execute trades for this symbol
                    for signal in signals:
                        if len(self.positions) < self.config['max_positions']:
                            trade = self.execute_practical_trade(signal)
                            if trade:
                                self.simulate_trade_exit(trade, intraday_df)
                else:
                    print(f"❌ Insufficient data for {symbol_name} - got {len(df) if df is not None else 0} candles")
                    
            except Exception as e:
                print(f"❌ Error processing {symbol_name}: {e}")
        
        print(f"\n📊 Backtest completed. Generating results...")
        self.generate_practical_results()
    
    def generate_practical_results(self):
        """Generate practical backtest results"""
        
        print("\n" + "=" * 70)
        print("📊 PRACTICAL INDICES INTRADAY BACKTEST RESULTS")
        print("=" * 70)
        
        if not self.trades:
            print("❌ No trades executed")
            return
        
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        total_trades = len(closed_trades)
        
        if total_trades == 0:
            print("❌ No trades completed")
            return
        
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in closed_trades)
        total_return = (total_pnl / self.initial_capital) * 100
        win_rate = (len(winning_trades) / total_trades * 100)
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        print(f"💰 PERFORMANCE SUMMARY:")
        print(f"   Initial Capital:     ₹{self.initial_capital:,.0f}")
        print(f"   Final Capital:       ₹{self.current_capital:,.0f}")
        print(f"   Total P&L:           ₹{total_pnl:,.0f}")
        print(f"   Return:              {total_return:.2f}%")
        
        print(f"\n📊 TRADING METRICS:")
        print(f"   Total Trades:        {total_trades}")
        print(f"   Winning Trades:      {len(winning_trades)}")
        print(f"   Losing Trades:       {len(losing_trades)}")
        print(f"   Win Rate:            {win_rate:.1f}%")
        print(f"   Average Win:         ₹{avg_win:,.0f}")
        print(f"   Average Loss:        ₹{avg_loss:,.0f}")
        
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            print(f"   Profit Factor:       {profit_factor:.2f}")
        
        # Symbol performance
        print(f"\n📈 SYMBOL PERFORMANCE:")
        print("-" * 50)
        
        for symbol in self.trading_symbols.keys():
            symbol_trades = [t for t in closed_trades if t['symbol'] == symbol]
            if symbol_trades:
                symbol_pnl = sum(t['pnl'] for t in symbol_trades)
                symbol_wins = len([t for t in symbol_trades if t['pnl'] > 0])
                symbol_win_rate = (symbol_wins / len(symbol_trades)) * 100
                
                print(f"   {symbol:12} | Trades: {len(symbol_trades):2d} | P&L: ₹{symbol_pnl:8,.0f} | Win: {symbol_win_rate:.1f}%")
        
        # Recent trades
        print(f"\n📋 ALL TRADES:")
        print("-" * 70)
        
        for i, trade in enumerate(closed_trades, 1):
            pnl_emoji = "💚" if trade['pnl'] > 0 else "❤️"
            print(f"{pnl_emoji} #{i:2d} {trade['timestamp'].strftime('%m-%d %H:%M')} | {trade['symbol']:12} {trade['signal']:4}")
            print(f"     Entry: ₹{trade['entry_price']:7.2f} → Exit: ₹{trade['exit_price']:7.2f} | P&L: ₹{trade['pnl']:7,.0f} | {trade['exit_reason']}")
        
        print("=" * 70)
        
        if total_return > 10:
            print("🎉 EXCELLENT: HIGHLY PROFITABLE STRATEGY!")
        elif total_return > 0:
            print("✅ GOOD: PROFITABLE STRATEGY")
        else:
            print("⚠️ NEEDS WORK: STRATEGY REQUIRES OPTIMIZATION")
        
        print("=" * 70)

def main():
    """Run practical indices backtesting"""
    
    print("🎯 PRACTICAL FYERS INDICES INTRADAY BACKTEST")
    print("Testing with available data and realistic simulations")
    print("=" * 70)
    
    backtester = PracticalIndicesBacktester(initial_capital=200000)
    
    # Test recent period
    backtester.run_practical_backtest("2024-01-15", "2024-01-25")

if __name__ == "__main__":
    main()