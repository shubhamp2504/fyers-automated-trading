"""
🎯 ENHANCED INDICES INTRADAY TRADING BACKTEST
Advanced backtesting for NIFTY and BANKNIFTY with 1H data and realistic intraday signals
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fyers_client import FyersClient
from index_intraday_strategy import IndexIntradayStrategy
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

class EnhancedIndicesBacktester:
    """Enhanced backtester for indices with 1H data and realistic intraday signals"""
    
    def __init__(self, initial_capital: float = 200000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.positions = []
        self.daily_pnl = []
        
        # Indices symbols for backtesting
        self.indices_symbols = {
            "NIFTY": "NSE:NIFTY50-INDEX",
            "BANKNIFTY": "NSE:NIFTYBANK-INDEX"
        }
        
        # Enhanced intraday trading parameters
        self.config = {
            "risk_per_trade": 0.015,     # 1.5% risk per trade (conservative)
            "max_positions": 3,          # Max 3 positions at once
            "stop_loss_pct": 1.8,        # 1.8% stop loss
            "target_pct_1": 2.5,         # First target: 2.5%
            "target_pct_2": 4.0,         # Second target: 4.0%
            "min_confidence": 60,        # Minimum confidence for entry
            "commission_per_lot": 40,    # Commission per lot
            "slippage_pct": 0.05,        # 0.05% slippage
            "lot_sizes": {
                "NIFTY": 50,             # NIFTY lot size
                "BANKNIFTY": 15          # BANKNIFTY lot size
            }
        }
        
        print("🚀 ENHANCED INDICES INTRADAY BACKTESTER INITIALIZED")
        print(f"💰 Initial Capital: ₹{self.initial_capital:,.2f}")
        print(f"📊 Symbols: {list(self.indices_symbols.keys())}")
        print(f"📝 Strategy: Multi-timeframe with 1H analysis")
    
    def get_hourly_data_for_backtest(self, symbol: str, start_date: str, end_date: str):
        """Get hourly historical data for more realistic backtesting"""
        try:
            fyers = FyersClient()
            
            # Get hourly data for the backtest period
            df = fyers.get_historical_data(
                symbol=symbol,
                resolution="1H",
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and len(df) > 0:
                print(f"✅ Hourly data loaded: {symbol} ({len(df)} candles)")
                return df
            else:
                print(f"❌ No hourly data available for {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading hourly data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame):
        """Calculate technical indicators for signal generation"""
        
        # RSI calculation
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        # Moving averages
        df['EMA_9'] = df['close'].ewm(span=9).mean()
        df['EMA_21'] = df['close'].ewm(span=21).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        df['RSI'] = calculate_rsi(df['close'])
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume analysis (using high-low range as proxy)
        df['Volume_Proxy'] = (df['high'] - df['low']) / df['close'] * 1000000
        df['Volume_SMA'] = df['Volume_Proxy'].rolling(window=20).mean()
        
        return df
    
    def generate_enhanced_signals(self, df: pd.DataFrame, symbol: str):
        """Generate enhanced intraday signals based on technical analysis"""
        
        if len(df) < 60:  # Need sufficient data for indicators
            return []
        
        # Add technical indicators
        df = self.calculate_technical_indicators(df)
        
        signals = []
        
        for i in range(50, len(df)):  # Start after sufficient data for indicators
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Only consider trading hours (9:15 AM to 3:00 PM IST)
            current_hour = current.name.hour
            if current_hour < 9 or current_hour > 15:
                continue
            
            # Calculate price momentum
            price_change_pct = ((current['close'] - previous['close']) / previous['close']) * 100
            
            # BUY Signal Conditions
            buy_conditions = [
                current['EMA_9'] > current['EMA_21'],           # Short MA > Long MA
                current['close'] > current['BB_Middle'],         # Price above BB middle
                current['RSI'] > 45 and current['RSI'] < 70,   # RSI in good range
                current['MACD'] > current['MACD_Signal'],       # MACD bullish
                current['Volume_Proxy'] > current['Volume_SMA'], # High volume
                price_change_pct > 0.3                         # Positive momentum
            ]
            
            # SELL Signal Conditions  
            sell_conditions = [
                current['EMA_9'] < current['EMA_21'],           # Short MA < Long MA
                current['close'] < current['BB_Middle'],         # Price below BB middle
                current['RSI'] < 55 and current['RSI'] > 30,   # RSI in good range
                current['MACD'] < current['MACD_Signal'],       # MACD bearish
                current['Volume_Proxy'] > current['Volume_SMA'], # High volume
                price_change_pct < -0.3                        # Negative momentum
            ]
            
            # Generate BUY signal
            if sum(buy_conditions) >= 4:  # At least 4 conditions met
                confidence = min(90, 50 + sum(buy_conditions) * 6)
                
                entry_price = current['close'] + (current['close'] * self.config['slippage_pct'] / 100)
                stop_loss = entry_price * (1 - self.config['stop_loss_pct'] / 100)
                target_1 = entry_price * (1 + self.config['target_pct_1'] / 100)
                target_2 = entry_price * (1 + self.config['target_pct_2'] / 100)
                
                signal = {
                    'timestamp': current.name,
                    'symbol': symbol,
                    'signal': 'BUY',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target_1': target_1,
                    'target_2': target_2,
                    'confidence': confidence,
                    'rsi': current['RSI'],
                    'reason': f"Tech BUY: {sum(buy_conditions)}/6 conditions, RSI:{current['RSI']:.1f}"
                }
                
                if confidence >= self.config['min_confidence']:
                    signals.append(signal)
            
            # Generate SELL signal
            elif sum(sell_conditions) >= 4:  # At least 4 conditions met
                confidence = min(90, 50 + sum(sell_conditions) * 6)
                
                entry_price = current['close'] - (current['close'] * self.config['slippage_pct'] / 100)
                stop_loss = entry_price * (1 + self.config['stop_loss_pct'] / 100)
                target_1 = entry_price * (1 - self.config['target_pct_1'] / 100)
                target_2 = entry_price * (1 - self.config['target_pct_2'] / 100)
                
                signal = {
                    'timestamp': current.name,
                    'symbol': symbol,
                    'signal': 'SELL',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target_1': target_1,
                    'target_2': target_2,
                    'confidence': confidence,
                    'rsi': current['RSI'],
                    'reason': f"Tech SELL: {sum(sell_conditions)}/6 conditions, RSI:{current['RSI']:.1f}"
                }
                
                if confidence >= self.config['min_confidence']:
                    signals.append(signal)
        
        return signals
    
    def calculate_position_size(self, signal: dict):
        """Calculate optimal position size based on risk management"""
        
        symbol = signal['symbol']
        lot_size = self.config['lot_sizes'][symbol]
        
        # Calculate risk per share/point
        if signal['signal'] == 'BUY':
            risk_per_point = signal['entry_price'] - signal['stop_loss']
        else:  # SELL
            risk_per_point = signal['stop_loss'] - signal['entry_price']
        
        if risk_per_point <= 0:
            return 0, 0
        
        # Risk amount per trade
        risk_amount = self.current_capital * self.config['risk_per_trade']
        
        # Calculate number of lots
        risk_per_lot = risk_per_point * lot_size
        num_lots = max(1, int(risk_amount / risk_per_lot))
        
        # Limit max position size to 30% of capital
        max_lots_by_capital = int((self.current_capital * 0.3) / (signal['entry_price'] * lot_size))
        num_lots = min(num_lots, max_lots_by_capital, 3)  # Max 3 lots
        
        quantity = num_lots * lot_size
        
        return num_lots, quantity
    
    def execute_enhanced_trade(self, signal: dict):
        """Execute a trade with enhanced position sizing"""
        
        num_lots, quantity = self.calculate_position_size(signal)
        
        if quantity <= 0:
            return None
        
        # Calculate required margin (approximate)
        required_margin = signal['entry_price'] * quantity * 0.15  # 15% margin for indices
        
        if required_margin > self.current_capital * 0.8:  # Don't use more than 80% capital
            return None
        
        # Create trade
        trade = {
            'id': len(self.trades) + 1,
            'timestamp': signal['timestamp'],
            'symbol': signal['symbol'],
            'signal': signal['signal'],
            'entry_price': signal['entry_price'],
            'quantity': quantity,
            'lots': num_lots,
            'stop_loss': signal['stop_loss'],
            'target_1': signal['target_1'],
            'target_2': signal['target_2'],
            'confidence': signal['confidence'],
            'rsi': signal['rsi'],
            'reason': signal['reason'],
            'status': 'OPEN',
            'pnl': 0,
            'commission': self.config['commission_per_lot'] * num_lots,
            'exit_price': None,
            'exit_reason': None
        }
        
        self.trades.append(trade)
        self.positions.append(trade.copy())
        
        print(f"📈 Trade #{trade['id']}: {trade['signal']} {trade['symbol']} @ ₹{trade['entry_price']:.2f}")
        print(f"   📊 Lots: {num_lots} | Qty: {quantity} | Confidence: {trade['confidence']:.0f}% | RSI: {trade['rsi']:.1f}")
        
        return trade
    
    def manage_open_positions(self, df: pd.DataFrame, symbol: str):
        """Manage open positions with current market data"""
        
        for trade in self.positions[:]:  # Use slice to avoid modification during iteration
            if trade['symbol'] == symbol and trade['status'] == 'OPEN':
                
                current_candle = df.iloc[-1]  # Latest candle
                current_price = current_candle['close']
                
                exit_price = None
                exit_reason = ""
                
                if trade['signal'] == 'BUY':
                    # Check stop loss
                    if current_candle['low'] <= trade['stop_loss']:
                        exit_price = trade['stop_loss']
                        exit_reason = "Stop Loss Hit"
                    # Check targets
                    elif current_candle['high'] >= trade['target_2']:
                        exit_price = trade['target_2']
                        exit_reason = "Target 2 Hit"
                    elif current_candle['high'] >= trade['target_1']:
                        exit_price = trade['target_1'] 
                        exit_reason = "Target 1 Hit"
                    # Check time-based exit (after 4 hours)
                    elif (current_candle.name - trade['timestamp']).total_seconds() > 14400:  # 4 hours
                        exit_price = current_price
                        exit_reason = "Time-based Exit"
                
                else:  # SELL
                    # Check stop loss
                    if current_candle['high'] >= trade['stop_loss']:
                        exit_price = trade['stop_loss']
                        exit_reason = "Stop Loss Hit"
                    # Check targets
                    elif current_candle['low'] <= trade['target_2']:
                        exit_price = trade['target_2']
                        exit_reason = "Target 2 Hit"
                    elif current_candle['low'] <= trade['target_1']:
                        exit_price = trade['target_1']
                        exit_reason = "Target 1 Hit"
                    # Check time-based exit (after 4 hours)
                    elif (current_candle.name - trade['timestamp']).total_seconds() > 14400:  # 4 hours
                        exit_price = current_price
                        exit_reason = "Time-based Exit"
                
                # Execute exit if conditions met
                if exit_price is not None:
                    self.close_position(trade, exit_price, exit_reason)
    
    def close_position(self, trade: dict, exit_price: float, exit_reason: str):
        """Close a position and calculate P&L"""
        
        # Calculate P&L
        if trade['signal'] == 'BUY':
            pnl = (exit_price - trade['entry_price']) * trade['quantity']
        else:  # SELL
            pnl = (trade['entry_price'] - exit_price) * trade['quantity']
        
        pnl -= trade['commission']  # Deduct commission
        
        # Update trade
        trade['exit_price'] = exit_price
        trade['exit_reason'] = exit_reason
        trade['pnl'] = pnl
        trade['status'] = 'CLOSED'
        
        # Update capital
        self.current_capital += pnl
        
        # Remove from positions
        self.positions = [p for p in self.positions if p['id'] != trade['id']]
        
        pnl_emoji = "💚" if pnl > 0 else "❤️"
        print(f"{pnl_emoji} Trade #{trade['id']} CLOSED: {exit_reason} @ ₹{exit_price:.2f} | P&L: ₹{pnl:,.2f}")
    
    def run_enhanced_backtest(self, start_date: str, end_date: str):
        """Run enhanced indices backtest with hourly data"""
        
        print(f"\n🚀 RUNNING ENHANCED INDICES INTRADAY BACKTEST")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"📊 Using 1H timeframe for realistic intraday simulation")
        print("=" * 70)
        
        all_data = {}
        
        # Get hourly data for both indices
        for symbol_name, symbol_code in self.indices_symbols.items():
            print(f"\n📊 Loading hourly data for {symbol_name}...")
            
            df = self.get_hourly_data_for_backtest(symbol_code, start_date, end_date)
            
            if df is not None and len(df) > 50:  # Need sufficient data
                all_data[symbol_name] = df
                print(f"✅ Loaded {len(df)} hourly candles for {symbol_name}")
            else:
                print(f"❌ Insufficient data for {symbol_name}")
        
        if not all_data:
            print("❌ No sufficient data available for backtesting")
            return
        
        # Generate signals for each symbol
        all_signals = []
        for symbol_name, df in all_data.items():
            print(f"\n🔍 Analyzing {symbol_name} for trading signals...")
            signals = self.generate_enhanced_signals(df, symbol_name)
            all_signals.extend(signals)
            print(f"✅ Generated {len(signals)} signals for {symbol_name}")
        
        # Sort signals by timestamp
        all_signals.sort(key=lambda x: x['timestamp'])
        
        print(f"\n📈 TOTAL SIGNALS GENERATED: {len(all_signals)}")
        print("=" * 70)
        
        if not all_signals:
            print("❌ No trading signals generated for the given period")
            return
        
        # Process signals chronologically
        for signal in all_signals:
            # Check if we can take more positions
            if len(self.positions) < self.config['max_positions']:
                # Execute trade
                trade = self.execute_enhanced_trade(signal)
                
                if trade:
                    # Manage existing positions with updated market data
                    symbol_df = all_data[signal['symbol']]
                    signal_index = symbol_df.index.get_loc(signal['timestamp'])
                    
                    # Process subsequent candles for position management
                    for i in range(signal_index + 1, len(symbol_df)):
                        current_df = symbol_df.iloc[:i+1]
                        self.manage_open_positions(current_df, signal['symbol'])
                        
                        # If position was closed, break
                        if not any(p['id'] == trade['id'] for p in self.positions):
                            break
        
        # Close any remaining open positions
        for trade in self.positions[:]:
            if trade['status'] == 'OPEN':
                # Close at last available price
                symbol_df = all_data[trade['symbol']]
                last_price = symbol_df.iloc[-1]['close']
                self.close_position(trade, last_price, "Backtest End")
        
        # Generate results
        self.generate_enhanced_results()
    
    def generate_enhanced_results(self):
        """Generate comprehensive enhanced backtest results"""
        
        print("\n" + "=" * 70)
        print("📊 ENHANCED INDICES INTRADAY BACKTEST RESULTS")
        print("=" * 70)
        
        if not self.trades:
            print("❌ No trades executed during backtest period")
            return
        
        # Calculate comprehensive metrics
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        total_trades = len(closed_trades)
        
        if total_trades == 0:
            print("❌ No trades completed during backtest period")
            return
        
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in closed_trades)
        total_return = (total_pnl / self.initial_capital) * 100
        
        win_rate = (len(winning_trades) / total_trades * 100)
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        max_win = max([t['pnl'] for t in closed_trades]) if closed_trades else 0
        max_loss = min([t['pnl'] for t in closed_trades]) if closed_trades else 0
        
        # Calculate drawdown
        running_pnl = 0
        peak = self.initial_capital
        max_drawdown = 0
        
        for trade in closed_trades:
            running_pnl += trade['pnl']
            current_capital = self.initial_capital + running_pnl
            
            if current_capital > peak:
                peak = current_capital
            
            drawdown = (peak - current_capital) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Display comprehensive results
        print(f"💰 FINANCIAL PERFORMANCE:")
        print(f"   Initial Capital:     ₹{self.initial_capital:,.2f}")
        print(f"   Final Capital:       ₹{self.current_capital:,.2f}")
        print(f"   Total P&L:           ₹{total_pnl:,.2f}")
        print(f"   Total Return:        {total_return:.2f}%")
        print(f"   Max Drawdown:        {max_drawdown:.2f}%")
        
        print(f"\n📊 TRADING STATISTICS:")
        print(f"   Total Trades:        {total_trades}")
        print(f"   Winning Trades:      {len(winning_trades)} ({len(winning_trades)/total_trades*100:.1f}%)")
        print(f"   Losing Trades:       {len(losing_trades)} ({len(losing_trades)/total_trades*100:.1f}%)")
        print(f"   Win Rate:            {win_rate:.1f}%")
        print(f"   Average Win:         ₹{avg_win:,.2f}")
        print(f"   Average Loss:        ₹{avg_loss:,.2f}")
        print(f"   Best Trade:          ₹{max_win:,.2f}")
        print(f"   Worst Trade:         ₹{max_loss:,.2f}")
        
        if avg_loss != 0:
            profit_factor = abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades]))
            print(f"   Profit Factor:       {profit_factor:.2f}")
        
        # Risk metrics
        total_lots_traded = sum(t['lots'] for t in closed_trades)
        avg_confidence = np.mean([t['confidence'] for t in closed_trades])
        
        print(f"\n📈 RISK & PERFORMANCE METRICS:")
        print(f"   Total Lots Traded:   {total_lots_traded}")
        print(f"   Avg Confidence:      {avg_confidence:.1f}%")
        print(f"   Capital Utilization: {((self.initial_capital - self.current_capital + total_pnl) / self.initial_capital * 100):.1f}%")
        
        # Symbol-wise performance
        print(f"\n📊 SYMBOL-WISE PERFORMANCE:")
        print("-" * 50)
        
        for symbol in self.indices_symbols.keys():
            symbol_trades = [t for t in closed_trades if t['symbol'] == symbol]
            if symbol_trades:
                symbol_pnl = sum(t['pnl'] for t in symbol_trades)
                symbol_wins = len([t for t in symbol_trades if t['pnl'] > 0])
                symbol_win_rate = (symbol_wins / len(symbol_trades)) * 100
                symbol_lots = sum(t['lots'] for t in symbol_trades)
                
                print(f"   {symbol:10} | Trades: {len(symbol_trades):2d} | Lots: {symbol_lots:2d} | P&L: ₹{symbol_pnl:8,.0f} | Win: {symbol_win_rate:.1f}%")
        
        # Recent trades
        print(f"\n📈 RECENT TRADES (Last 10):")
        print("-" * 70)
        
        recent_trades = closed_trades[-10:] if len(closed_trades) >= 10 else closed_trades
        
        for trade in recent_trades:
            pnl_emoji = "💚" if trade['pnl'] > 0 else "❤️"
            print(f"{pnl_emoji} {trade['timestamp'].strftime('%Y-%m-%d %H:%M')} | {trade['symbol']} {trade['signal']}")
            print(f"   Entry: ₹{trade['entry_price']:.2f} → Exit: ₹{trade['exit_price']:.2f} | P&L: ₹{trade['pnl']:,.2f}")
            print(f"   {trade['exit_reason']} | Confidence: {trade['confidence']:.0f}% | Lots: {trade['lots']}")
            print()
        
        print("=" * 70)
        
        if total_return > 5:
            print("🎉 BACKTEST RESULT: HIGHLY PROFITABLE STRATEGY")
        elif total_return > 0:
            print("✅ BACKTEST RESULT: PROFITABLE STRATEGY") 
        else:
            print("⚠️  BACKTEST RESULT: STRATEGY NEEDS OPTIMIZATION")
        
        print("=" * 70)

def main():
    """Run enhanced indices intraday backtesting"""
    
    print("🎯 ENHANCED FYERS INDICES INTRADAY TRADING BACKTEST")
    print("Advanced Multi-Timeframe Strategy for NIFTY & BANKNIFTY")
    print("=" * 70)
    
    # Initialize enhanced backtester
    backtester = EnhancedIndicesBacktester(initial_capital=200000)
    
    # Define test periods (recent periods with good market activity)
    test_periods = [
        {
            "name": "Recent Bull Run",
            "start": "2024-01-01",
            "end": "2024-01-31"
        }
    ]
    
    # Run enhanced backtests
    for period in test_periods:
        print(f"\n🔍 TESTING PERIOD: {period['name']}")
        print(f"📅 {period['start']} to {period['end']}")
        
        # Reset for new test
        backtester.current_capital = backtester.initial_capital
        backtester.trades = []
        backtester.positions = []
        
        # Run enhanced backtest
        backtester.run_enhanced_backtest(period['start'], period['end'])

if __name__ == "__main__":
    main()