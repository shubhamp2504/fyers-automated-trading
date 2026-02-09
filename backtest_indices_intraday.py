"""
🎯 INDICES INTRADAY TRADING BACKTEST
Run backtests specifically for NIFTY and BANKNIFTY intraday strategies
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

class IndicesIntradayBacktester:
    """Backtester specifically for indices intraday trading"""
    
    def __init__(self, initial_capital: float = 100000):
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
        
        # Intraday trading parameters
        self.config = {
            "risk_per_trade": 0.02,  # 2% risk per trade
            "max_positions": 2,      # Max 2 positions (NIFTY + BANKNIFTY)
            "stop_loss_points": 25,  # 25 points stop loss
            "target_points_1": 35,   # First target: 35 points
            "target_points_2": 50,   # Second target: 50 points
            "min_confidence": 65,    # Minimum confidence for entry
            "commission_per_trade": 50,  # Commission per index trade
            "slippage_points": 2     # 2 points slippage
        }
        
        print("🎯 INDICES INTRADAY BACKTESTER INITIALIZED")
        print(f"💰 Initial Capital: ₹{self.initial_capital:,.2f}")
        print(f"📊 Symbols: {list(self.indices_symbols.keys())}")
    
    def get_historical_data_for_backtest(self, symbol: str, start_date: str, end_date: str):
        """Get historical data for backtesting"""
        try:
            fyers = FyersClient()
            
            # Get daily data for the backtest period
            df = fyers.get_historical_data(
                symbol=symbol,
                resolution="1D",
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and len(df) > 0:
                print(f"✅ Historical data loaded: {symbol} ({len(df)} days)")
                return df
            else:
                print(f"❌ No data available for {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading data for {symbol}: {e}")
            return None
    
    def simulate_intraday_signals(self, df: pd.DataFrame, symbol: str):
        """Generate simulated intraday signals based on daily data"""
        signals = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Simple momentum strategy for indices
            price_change_percent = ((current['close'] - previous['close']) / previous['close']) * 100
            volatility = ((current['high'] - current['low']) / current['open']) * 100
            
            # Generate BUY signal conditions
            if (price_change_percent > 0.5 and volatility > 1.0 and volatility < 4.0):
                confidence = min(85, 50 + abs(price_change_percent) * 5)
                
                signal = {
                    'timestamp': current.name,
                    'symbol': symbol,
                    'signal': 'BUY',
                    'entry_price': current['open'] + 5,  # Simulate buy after gap up
                    'stop_loss': current['open'] - self.config['stop_loss_points'],
                    'target_1': current['open'] + self.config['target_points_1'],
                    'target_2': current['open'] + self.config['target_points_2'],
                    'confidence': confidence,
                    'reason': f"Momentum up {price_change_percent:.1f}%, Vol: {volatility:.1f}%"
                }
                
                if confidence >= self.config['min_confidence']:
                    signals.append(signal)
            
            # Generate SELL signal conditions  
            elif (price_change_percent < -0.5 and volatility > 1.0 and volatility < 4.0):
                confidence = min(85, 50 + abs(price_change_percent) * 5)
                
                signal = {
                    'timestamp': current.name,
                    'symbol': symbol,
                    'signal': 'SELL',
                    'entry_price': current['open'] - 5,  # Simulate sell after gap down
                    'stop_loss': current['open'] + self.config['stop_loss_points'],
                    'target_1': current['open'] - self.config['target_points_1'],
                    'target_2': current['open'] - self.config['target_points_2'],
                    'confidence': confidence,
                    'reason': f"Momentum down {price_change_percent:.1f}%, Vol: {volatility:.1f}%"
                }
                
                if confidence >= self.config['min_confidence']:
                    signals.append(signal)
        
        return signals
    
    def execute_trade(self, signal: dict):
        """Execute a simulated trade based on signal"""
        
        # Calculate position size based on risk
        risk_amount = self.current_capital * self.config['risk_per_trade']
        
        if signal['signal'] == 'BUY':
            risk_per_share = signal['entry_price'] - signal['stop_loss']
        else:  # SELL
            risk_per_share = signal['stop_loss'] - signal['entry_price']
        
        if risk_per_share <= 0:
            return None
        
        quantity = int(risk_amount / risk_per_share)
        if quantity <= 0:
            return None
        
        # Calculate required capital
        required_capital = quantity * signal['entry_price']
        
        if required_capital > self.current_capital * 0.8:  # Use max 80% capital per trade
            quantity = int((self.current_capital * 0.8) / signal['entry_price'])
        
        if quantity <= 0:
            return None
        
        # Simulate trade execution
        trade = {
            'id': len(self.trades) + 1,
            'timestamp': signal['timestamp'],
            'symbol': signal['symbol'],
            'signal': signal['signal'],
            'entry_price': signal['entry_price'] + (self.config['slippage_points'] if signal['signal'] == 'BUY' else -self.config['slippage_points']),
            'quantity': quantity,
            'stop_loss': signal['stop_loss'],
            'target_1': signal['target_1'],
            'target_2': signal['target_2'],
            'confidence': signal['confidence'],
            'reason': signal['reason'],
            'status': 'OPEN',
            'pnl': 0,
            'commission': self.config['commission_per_trade']
        }
        
        self.trades.append(trade)
        self.positions.append(trade.copy())
        
        print(f"📈 Trade #{trade['id']}: {trade['signal']} {trade['symbol']} @ ₹{trade['entry_price']:.2f} (Qty: {quantity})")
        
        return trade
    
    def simulate_trade_exit(self, trade: dict, current_price: float, high_price: float, low_price: float):
        """Simulate trade exit based on targets or stop loss"""
        
        if trade['status'] != 'OPEN':
            return trade
        
        exit_price = None
        exit_reason = ""
        
        if trade['signal'] == 'BUY':
            # Check stop loss
            if low_price <= trade['stop_loss']:
                exit_price = trade['stop_loss']
                exit_reason = "Stop Loss Hit"
            # Check targets
            elif high_price >= trade['target_2']:
                exit_price = trade['target_2']
                exit_reason = "Target 2 Hit"
            elif high_price >= trade['target_1']:
                exit_price = trade['target_1'] 
                exit_reason = "Target 1 Hit"
        
        else:  # SELL
            # Check stop loss
            if high_price >= trade['stop_loss']:
                exit_price = trade['stop_loss']
                exit_reason = "Stop Loss Hit"
            # Check targets
            elif low_price <= trade['target_2']:
                exit_price = trade['target_2']
                exit_reason = "Target 2 Hit"
            elif low_price <= trade['target_1']:
                exit_price = trade['target_1']
                exit_reason = "Target 1 Hit"
        
        # If no exit condition met, simulate partial booking at market close
        if exit_price is None:
            exit_price = current_price
            exit_reason = "Market Close (Intraday)"
        
        # Calculate P&L
        if trade['signal'] == 'BUY':
            pnl = (exit_price - trade['entry_price']) * trade['quantity']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['quantity']
        
        pnl -= trade['commission']  # Deduct commission
        
        trade['exit_price'] = exit_price
        trade['exit_reason'] = exit_reason
        trade['pnl'] = pnl
        trade['status'] = 'CLOSED'
        
        # Update capital
        self.current_capital += pnl
        
        # Remove from positions
        self.positions = [p for p in self.positions if p['id'] != trade['id']]
        
        print(f"📊 Trade #{trade['id']} CLOSED: {exit_reason} @ ₹{exit_price:.2f} | P&L: ₹{pnl:.2f}")
        
        return trade
    
    def run_backtest(self, start_date: str, end_date: str):
        """Run comprehensive indices backtest"""
        
        print(f"\n🚀 RUNNING INDICES INTRADAY BACKTEST")
        print(f"📅 Period: {start_date} to {end_date}")
        print("=" * 60)
        
        all_signals = []
        
        # Get historical data for both indices
        for symbol_name, symbol_code in self.indices_symbols.items():
            print(f"\n📊 Loading data for {symbol_name}...")
            
            df = self.get_historical_data_for_backtest(symbol_code, start_date, end_date)
            
            if df is not None:
                # Generate signals
                signals = self.simulate_intraday_signals(df, symbol_name)
                all_signals.extend(signals)
                
                print(f"✅ Generated {len(signals)} signals for {symbol_name}")
        
        # Sort signals by timestamp
        all_signals.sort(key=lambda x: x['timestamp'])
        
        print(f"\n📈 TOTAL SIGNALS GENERATED: {len(all_signals)}")
        print("=" * 60)
        
        # Execute trades
        for signal in all_signals:
            # Check if we can take more positions
            if len(self.positions) < self.config['max_positions']:
                trade = self.execute_trade(signal)
                
                # Simulate same-day exit (intraday)
                if trade:
                    # Get the day's price data for exit simulation
                    symbol_code = self.indices_symbols[trade['symbol']]
                    df_day = self.get_historical_data_for_backtest(
                        symbol_code, 
                        signal['timestamp'].strftime('%Y-%m-%d'),
                        signal['timestamp'].strftime('%Y-%m-%d')
                    )
                    
                    if df_day is not None and len(df_day) > 0:
                        day_data = df_day.iloc[0]
                        self.simulate_trade_exit(
                            trade,
                            day_data['close'],
                            day_data['high'], 
                            day_data['low']
                        )
        
        # Generate results
        self.generate_backtest_results()
    
    def generate_backtest_results(self):
        """Generate comprehensive backtest results"""
        
        print("\n" + "=" * 60)
        print("📊 INDICES INTRADAY BACKTEST RESULTS")
        print("=" * 60)
        
        if not self.trades:
            print("❌ No trades executed during backtest period")
            return
        
        # Calculate metrics
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in closed_trades)
        total_return = (total_pnl / self.initial_capital) * 100
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Display results
        print(f"💰 FINANCIAL PERFORMANCE:")
        print(f"   Initial Capital:     ₹{self.initial_capital:,.2f}")
        print(f"   Final Capital:       ₹{self.current_capital:,.2f}")
        print(f"   Total P&L:           ₹{total_pnl:,.2f}")
        print(f"   Total Return:        {total_return:.2f}%")
        
        print(f"\n📊 TRADING STATISTICS:")
        print(f"   Total Trades:        {total_trades}")
        print(f"   Winning Trades:      {len(winning_trades)}")
        print(f"   Losing Trades:       {len(losing_trades)}")
        print(f"   Win Rate:            {win_rate:.1f}%")
        print(f"   Average Win:         ₹{avg_win:,.2f}")
        print(f"   Average Loss:        ₹{avg_loss:,.2f}")
        
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            print(f"   Profit Factor:       {profit_factor:.2f}")
        
        # Show recent trades
        print(f"\n📈 RECENT TRADES (Last 5):")
        print("-" * 60)
        
        recent_trades = closed_trades[-5:] if len(closed_trades) >= 5 else closed_trades
        
        for trade in recent_trades:
            pnl_emoji = "💚" if trade['pnl'] > 0 else "❤️"
            print(f"{pnl_emoji} {trade['timestamp'].strftime('%Y-%m-%d')} | {trade['symbol']} {trade['signal']}")
            print(f"   Entry: ₹{trade['entry_price']:.2f} → Exit: ₹{trade['exit_price']:.2f}")
            print(f"   P&L: ₹{trade['pnl']:.2f} | {trade['exit_reason']}")
            print()
        
        # Symbol-wise performance
        print(f"📊 SYMBOL-WISE PERFORMANCE:")
        print("-" * 40)
        
        for symbol in self.indices_symbols.keys():
            symbol_trades = [t for t in closed_trades if t['symbol'] == symbol]
            if symbol_trades:
                symbol_pnl = sum(t['pnl'] for t in symbol_trades)
                symbol_wins = len([t for t in symbol_trades if t['pnl'] > 0])
                symbol_win_rate = (symbol_wins / len(symbol_trades)) * 100
                
                print(f"   {symbol:12} | Trades: {len(symbol_trades):2d} | P&L: ₹{symbol_pnl:7,.0f} | Win Rate: {symbol_win_rate:.1f}%")
        
        print("\n" + "=" * 60)
        
        if total_return > 0:
            print("🎉 BACKTEST RESULT: PROFITABLE STRATEGY")
        else:
            print("⚠️  BACKTEST RESULT: STRATEGY NEEDS OPTIMIZATION")
        
        print("=" * 60)

def main():
    """Run indices intraday backtesting"""
    
    print("🎯 FYERS INDICES INTRADAY TRADING BACKTEST")
    print("Backtesting NIFTY & BANKNIFTY intraday strategies")
    print("=" * 60)
    
    # Initialize backtester
    backtester = IndicesIntradayBacktester(initial_capital=100000)
    
    # Define test periods
    test_periods = [
        {
            "name": "Recent 3 Months",
            "start": "2024-01-15",
            "end": "2024-01-19"  # Using working date range
        },
        {
            "name": "Extended Period",
            "start": "2024-01-10", 
            "end": "2024-01-25"
        }
    ]
    
    # Run backtests for different periods
    for period in test_periods:
        print(f"\n🔍 TESTING PERIOD: {period['name']}")
        
        # Reset for new test
        backtester.current_capital = backtester.initial_capital
        backtester.trades = []
        backtester.positions = []
        
        # Run backtest
        backtester.run_backtest(period['start'], period['end'])

if __name__ == "__main__":
    main()