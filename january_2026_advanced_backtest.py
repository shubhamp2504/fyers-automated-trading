"""
🎯 ENHANCED INDICES INTRADAY BACKTEST - JANUARY 2026
Advanced backtesting with 1min/5min data + Option Chain analysis
Period: January 2026 to February 5, 2026
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
import warnings
warnings.filterwarnings('ignore')

class AdvancedIndicesBacktester:
    """Advanced backtester with multi-timeframe analysis and option chain data"""
    
    def __init__(self, initial_capital: float = 250000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.positions = []
        self.daily_pnl = []
        self.option_data = {}
        
        # Enhanced indices symbols with futures
        self.trading_symbols = {
            "NIFTY": "NSE:NIFTY50-INDEX",
            "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
            "NIFTY_FUT": "NSE:NIFTY26FEB-INDEX",  # Feb 2026 futures
            "BANKNIFTY_FUT": "NSE:BANKNIFTY26FEB-INDEX"
        }
        
        # Enhanced trading parameters for 2026 data
        self.config = {
            "risk_per_trade": 0.018,     # 1.8% risk per trade
            "max_positions": 4,          # Max 4 positions
            "stop_loss_pct": 1.5,        # 1.5% stop loss
            "target_pct_1": 2.2,         # First target: 2.2%
            "target_pct_2": 3.8,         # Second target: 3.8%
            "min_confidence": 58,        # Minimum confidence for entry
            "commission_per_trade": 25,  # Reduced commission
            "slippage_pct": 0.03,        # 0.03% slippage
            "use_options": True,         # Enable option chain analysis
            "timeframes": ["5m", "1m"],  # Multi-timeframe analysis
            "lot_sizes": {
                "NIFTY": 50,
                "BANKNIFTY": 15,
                "NIFTY_FUT": 50,
                "BANKNIFTY_FUT": 15
            }
        }
        
        print("🚀 ADVANCED INDICES BACKTESTER - JANUARY 2026 DATA")
        print(f"💰 Initial Capital: ₹{self.initial_capital:,.2f}")
        print(f"📊 Symbols: {list(self.trading_symbols.keys())}")
        print(f"⏰ Timeframes: {self.config['timeframes']}")
        print(f"📈 Option Chain: {'Enabled' if self.config['use_options'] else 'Disabled'}")
    
    def get_multi_timeframe_data(self, symbol: str, start_date: str, end_date: str):
        """Get data for multiple timeframes"""
        
        fyers = FyersClient()
        data = {}
        
        print(f"\n📊 Loading multi-timeframe data for {symbol}...")
        
        # Get 5-minute data (primary timeframe)
        try:
            df_5m = fyers.get_historical_data(
                symbol=symbol,
                resolution="5",  # 5-minute
                start_date=start_date,
                end_date=end_date
            )
            
            if df_5m is not None and len(df_5m) > 0:
                data['5m'] = df_5m
                print(f"✅ 5-minute data: {len(df_5m)} candles")
            else:
                print(f"⚠️ No 5-minute data for {symbol}")
        except Exception as e:
            print(f"❌ Error getting 5-minute data: {e}")
        
        # Get 1-minute data (execution timeframe)
        try:
            df_1m = fyers.get_historical_data(
                symbol=symbol,
                resolution="1",  # 1-minute
                start_date=start_date,
                end_date=end_date
            )
            
            if df_1m is not None and len(df_1m) > 0:
                data['1m'] = df_1m
                print(f"✅ 1-minute data: {len(df_1m)} candles")
            else:
                print(f"⚠️ No 1-minute data for {symbol}")
        except Exception as e:
            print(f"❌ Error getting 1-minute data: {e}")
        
        return data
    
    def analyze_option_chain_sentiment(self, symbol: str, current_price: float):
        """Analyze option chain for market sentiment (simulated for backtest)"""
        
        if not self.config.get('use_options', False):
            return {'sentiment': 'NEUTRAL', 'strength': 50}
        
        # Simulate option chain analysis based on price action
        # In real implementation, this would fetch actual option chain data
        
        sentiment_score = 50  # Neutral baseline
        
        # Simple sentiment based on recent price movement
        if hasattr(self, 'prev_price'):
            price_change_pct = ((current_price - self.prev_price) / self.prev_price) * 100
            
            if price_change_pct > 0.5:
                sentiment_score += min(25, price_change_pct * 10)
            elif price_change_pct < -0.5:
                sentiment_score -= min(25, abs(price_change_pct) * 10)
        
        self.prev_price = current_price
        
        if sentiment_score > 65:
            sentiment = 'BULLISH'
        elif sentiment_score < 35:
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'sentiment': sentiment,
            'strength': sentiment_score,
            'put_call_ratio': 1.0 + (sentiment_score - 50) / 100
        }
    
    def calculate_advanced_indicators(self, df_5m: pd.DataFrame, df_1m: pd.DataFrame):
        """Calculate advanced technical indicators across timeframes"""
        
        indicators = {}
        
        # 5-minute timeframe indicators (trend analysis)
        if len(df_5m) >= 50:
            # EMAs
            df_5m['EMA_9'] = df_5m['close'].ewm(span=9).mean()
            df_5m['EMA_21'] = df_5m['close'].ewm(span=21).mean()
            df_5m['EMA_50'] = df_5m['close'].ewm(span=50).mean()
            
            # RSI
            df_5m['RSI'] = self.calculate_rsi(df_5m['close'], 14)
            
            # MACD
            exp1 = df_5m['close'].ewm(span=12).mean()
            exp2 = df_5m['close'].ewm(span=26).mean()
            df_5m['MACD'] = exp1 - exp2
            df_5m['MACD_Signal'] = df_5m['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands
            df_5m['BB_Mid'] = df_5m['close'].rolling(20).mean()
            bb_std = df_5m['close'].rolling(20).std()
            df_5m['BB_Upper'] = df_5m['BB_Mid'] + (bb_std * 2)
            df_5m['BB_Lower'] = df_5m['BB_Mid'] - (bb_std * 2)
            
            # Volume proxy (using range)
            df_5m['Volume'] = (df_5m['high'] - df_5m['low']) / df_5m['close'] * 1000000
            df_5m['Volume_SMA'] = df_5m['Volume'].rolling(20).mean()
            
            indicators['5m'] = df_5m.iloc[-1].to_dict()
        
        # 1-minute timeframe indicators (entry timing)
        if len(df_1m) >= 20:
            df_1m['EMA_5'] = df_1m['close'].ewm(span=5).mean()
            df_1m['EMA_13'] = df_1m['close'].ewm(span=13).mean()
            df_1m['RSI'] = self.calculate_rsi(df_1m['close'], 14)
            
            # Momentum
            df_1m['Momentum'] = df_1m['close'].pct_change(5) * 100
            
            indicators['1m'] = df_1m.iloc[-1].to_dict()
        
        return indicators, df_5m, df_1m
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_enhanced_signals(self, symbol: str, indicators: dict, current_price: float):
        """Generate trading signals using multi-timeframe analysis"""
        
        signals = []
        
        if '5m' not in indicators or '1m' not in indicators:
            return signals
        
        tf_5m = indicators['5m']
        tf_1m = indicators['1m']
        
        # Get option chain sentiment
        option_sentiment = self.analyze_option_chain_sentiment(symbol, current_price)
        
        # Multi-timeframe signal generation
        
        # BUY Signal Conditions
        buy_conditions = []
        buy_strength = 0
        
        # 5-minute trend conditions
        if tf_5m.get('EMA_9', 0) > tf_5m.get('EMA_21', 0):
            buy_conditions.append("5m_trend_bullish")
            buy_strength += 15
        
        if tf_5m.get('RSI', 50) > 45 and tf_5m.get('RSI', 50) < 70:
            buy_conditions.append("5m_rsi_good")
            buy_strength += 10
        
        if tf_5m.get('MACD', 0) > tf_5m.get('MACD_Signal', 0):
            buy_conditions.append("5m_macd_bullish")
            buy_strength += 12
        
        # 1-minute entry conditions
        if tf_1m.get('EMA_5', 0) > tf_1m.get('EMA_13', 0):
            buy_conditions.append("1m_trend_bullish")
            buy_strength += 8
        
        if tf_1m.get('Momentum', 0) > 0.1:
            buy_conditions.append("1m_momentum_positive")
            buy_strength += 10
        
        # Option sentiment boost
        if option_sentiment['sentiment'] == 'BULLISH':
            buy_conditions.append("options_bullish")
            buy_strength += 15
        elif option_sentiment['sentiment'] == 'NEUTRAL':
            buy_strength += 5
        
        # Volume confirmation
        if tf_5m.get('Volume', 0) > tf_5m.get('Volume_SMA', 0):
            buy_conditions.append("volume_high")
            buy_strength += 8
        
        # Generate BUY signal if conditions met
        if len(buy_conditions) >= 4 and buy_strength >= self.config['min_confidence']:
            entry_price = current_price * (1 + self.config['slippage_pct'] / 100)
            
            signals.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'signal': 'BUY',
                'entry_price': entry_price,
                'stop_loss': entry_price * (1 - self.config['stop_loss_pct'] / 100),
                'target_1': entry_price * (1 + self.config['target_pct_1'] / 100),
                'target_2': entry_price * (1 + self.config['target_pct_2'] / 100),
                'confidence': min(95, buy_strength),
                'conditions': buy_conditions,
                'option_sentiment': option_sentiment,
                'timeframe_data': {
                    '5m_rsi': tf_5m.get('RSI', 50),
                    '1m_rsi': tf_1m.get('RSI', 50),
                    'momentum': tf_1m.get('Momentum', 0)
                }
            })
        
        # SELL Signal Conditions
        sell_conditions = []
        sell_strength = 0
        
        # 5-minute trend conditions
        if tf_5m.get('EMA_9', 0) < tf_5m.get('EMA_21', 0):
            sell_conditions.append("5m_trend_bearish")
            sell_strength += 15
        
        if tf_5m.get('RSI', 50) < 55 and tf_5m.get('RSI', 50) > 30:
            sell_conditions.append("5m_rsi_good")
            sell_strength += 10
        
        if tf_5m.get('MACD', 0) < tf_5m.get('MACD_Signal', 0):
            sell_conditions.append("5m_macd_bearish")
            sell_strength += 12
        
        # 1-minute entry conditions
        if tf_1m.get('EMA_5', 0) < tf_1m.get('EMA_13', 0):
            sell_conditions.append("1m_trend_bearish")
            sell_strength += 8
        
        if tf_1m.get('Momentum', 0) < -0.1:
            sell_conditions.append("1m_momentum_negative")
            sell_strength += 10
        
        # Option sentiment boost
        if option_sentiment['sentiment'] == 'BEARISH':
            sell_conditions.append("options_bearish")
            sell_strength += 15
        elif option_sentiment['sentiment'] == 'NEUTRAL':
            sell_strength += 5
        
        # Volume confirmation
        if tf_5m.get('Volume', 0) > tf_5m.get('Volume_SMA', 0):
            sell_conditions.append("volume_high")
            sell_strength += 8
        
        # Generate SELL signal if conditions met
        if len(sell_conditions) >= 4 and sell_strength >= self.config['min_confidence']:
            entry_price = current_price * (1 - self.config['slippage_pct'] / 100)
            
            signals.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'signal': 'SELL',
                'entry_price': entry_price,
                'stop_loss': entry_price * (1 + self.config['stop_loss_pct'] / 100),
                'target_1': entry_price * (1 - self.config['target_pct_1'] / 100),
                'target_2': entry_price * (1 - self.config['target_pct_2'] / 100),
                'confidence': min(95, sell_strength),
                'conditions': sell_conditions,
                'option_sentiment': option_sentiment,
                'timeframe_data': {
                    '5m_rsi': tf_5m.get('RSI', 50),
                    '1m_rsi': tf_1m.get('RSI', 50),
                    'momentum': tf_1m.get('Momentum', 0)
                }
            })
        
        return signals
    
    def execute_advanced_trade(self, signal: dict):
        """Execute trade with advanced position sizing and risk management"""
        
        symbol = signal['symbol']
        base_symbol = symbol.replace('_FUT', '')
        lot_size = self.config['lot_sizes'].get(base_symbol, 1)
        
        # Enhanced position sizing based on confidence and volatility
        confidence_multiplier = signal['confidence'] / 100
        base_risk = self.config['risk_per_trade'] * confidence_multiplier
        
        # Calculate position size
        if signal['signal'] == 'BUY':
            risk_per_unit = signal['entry_price'] - signal['stop_loss']
        else:
            risk_per_unit = signal['stop_loss'] - signal['entry_price']
        
        if risk_per_unit <= 0:
            return None
        
        risk_amount = self.current_capital * base_risk
        
        if '_FUT' in symbol:  # Futures trading
            lots = max(1, min(5, int(risk_amount / (risk_per_unit * lot_size))))
            quantity = lots * lot_size
        else:  # Index trading (cash)
            quantity = max(1, int(risk_amount / risk_per_unit))
            lots = 1
        
        # Check capital availability
        required_margin = signal['entry_price'] * quantity * 0.2
        if required_margin > self.current_capital * 0.7:
            return None
        
        # Create trade
        trade = {
            'id': len(self.trades) + 1,
            'timestamp': signal['timestamp'],
            'symbol': symbol,
            'signal': signal['signal'],
            'entry_price': signal['entry_price'],
            'quantity': quantity,
            'lots': lots,
            'stop_loss': signal['stop_loss'],
            'target_1': signal['target_1'],
            'target_2': signal['target_2'],
            'confidence': signal['confidence'],
            'conditions': signal['conditions'],
            'option_sentiment': signal['option_sentiment'],
            'timeframe_data': signal['timeframe_data'],
            'status': 'OPEN',
            'pnl': 0,
            'commission': self.config['commission_per_trade'],
            'exit_price': None,
            'exit_reason': None,
            'exit_timestamp': None
        }
        
        self.trades.append(trade)
        self.positions.append(trade.copy())
        
        print(f"📈 Trade #{trade['id']}: {trade['signal']} {trade['symbol']} @ ₹{trade['entry_price']:.2f}")
        print(f"   📊 Qty: {quantity} | Confidence: {trade['confidence']:.0f}% | Conditions: {len(signal['conditions'])}")
        print(f"   🔮 Option Sentiment: {signal['option_sentiment']['sentiment']} ({signal['option_sentiment']['strength']:.0f}%)")
        
        return trade
    
    def manage_positions_with_1m_data(self, df_1m: pd.DataFrame, symbol: str):
        """Manage positions using 1-minute data for precise exits"""
        
        if df_1m is None or len(df_1m) == 0:
            return
        
        for trade in self.positions[:]:
            if trade['symbol'] == symbol and trade['status'] == 'OPEN':
                
                # Use latest 1-minute candles for exit management
                for i in range(max(0, len(df_1m) - 10), len(df_1m)):
                    candle = df_1m.iloc[i]
                    
                    exit_price = None
                    exit_reason = ""
                    
                    if trade['signal'] == 'BUY':
                        # Stop loss
                        if candle['low'] <= trade['stop_loss']:
                            exit_price = trade['stop_loss']
                            exit_reason = "Stop Loss Hit"
                        # Targets
                        elif candle['high'] >= trade['target_2']:
                            exit_price = trade['target_2']
                            exit_reason = "Target 2 Achieved"
                        elif candle['high'] >= trade['target_1']:
                            # Partial exit at target 1, move SL to breakeven
                            if 'partial_exit' not in trade:
                                trade['partial_exit'] = True
                                trade['stop_loss'] = trade['entry_price']
                                print(f"📊 Trade #{trade['id']}: Target 1 hit, SL moved to breakeven")
                                continue
                            
                    else:  # SELL
                        # Stop loss
                        if candle['high'] >= trade['stop_loss']:
                            exit_price = trade['stop_loss']
                            exit_reason = "Stop Loss Hit"
                        # Targets
                        elif candle['low'] <= trade['target_2']:
                            exit_price = trade['target_2']
                            exit_reason = "Target 2 Achieved"
                        elif candle['low'] <= trade['target_1']:
                            # Partial exit at target 1
                            if 'partial_exit' not in trade:
                                trade['partial_exit'] = True
                                trade['stop_loss'] = trade['entry_price']
                                print(f"📊 Trade #{trade['id']}: Target 1 hit, SL moved to breakeven")
                                continue
                    
                    # Time-based exit (after 4 hours)
                    if exit_price is None:
                        time_diff = candle.name - trade['timestamp']
                        if isinstance(time_diff, pd.Timedelta) and time_diff.total_seconds() > 14400:
                            exit_price = candle['close']
                            exit_reason = "Time-based Exit (4h)"
                    
                    # Execute exit if conditions met
                    if exit_price is not None:
                        self.close_advanced_position(trade, exit_price, exit_reason, candle.name)
                        break
    
    def close_advanced_position(self, trade: dict, exit_price: float, exit_reason: str, exit_time=None):
        """Close position with detailed P&L calculation"""
        
        # Calculate P&L
        if trade['signal'] == 'BUY':
            pnl = (exit_price - trade['entry_price']) * trade['quantity']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['quantity']
        
        pnl -= trade['commission']
        
        # Update trade
        trade['exit_price'] = exit_price
        trade['exit_reason'] = exit_reason
        trade['exit_timestamp'] = exit_time or datetime.now()
        trade['pnl'] = pnl
        trade['status'] = 'CLOSED'
        
        # Update the trade in the main trades list
        for i, t in enumerate(self.trades):
            if t['id'] == trade['id']:
                self.trades[i] = trade.copy()
                break
        
        # Update capital
        self.current_capital += pnl
        
        # Remove from positions
        self.positions = [p for p in self.positions if p['id'] != trade['id']]
        
        pnl_emoji = "💚" if pnl > 0 else "❤️"
        print(f"{pnl_emoji} Trade #{trade['id']} CLOSED: {exit_reason}")
        print(f"   💰 Entry: ₹{trade['entry_price']:.2f} → Exit: ₹{exit_price:.2f} | P&L: ₹{pnl:,.0f}")
    
    def run_january_2026_backtest(self):
        """Run comprehensive backtest for January 2026 to February 5, 2026"""
        
        start_date = "2026-01-01"
        end_date = "2026-02-05"
        
        print(f"\n🚀 RUNNING JANUARY 2026 ENHANCED BACKTEST")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"📊 Multi-timeframe: 5min (trend) + 1min (execution)")
        print(f"🔮 Option Chain Analysis: Enabled")
        print("=" * 80)
        
        total_signals = 0
        
        # Process each trading symbol
        for symbol_name, symbol_code in self.trading_symbols.items():
            print(f"\n🎯 PROCESSING {symbol_name}...")
            
            # Get multi-timeframe data
            data = self.get_multi_timeframe_data(symbol_code, start_date, end_date)
            
            if not data:
                print(f"❌ No data available for {symbol_name}")
                continue
            
            # Process data if we have both timeframes
            if '5m' in data and '1m' in data:
                df_5m = data['5m']
                df_1m = data['1m']
                
                print(f"📈 Analyzing {len(df_5m)} 5-min candles and {len(df_1m)} 1-min candles")
                
                # Calculate indicators
                indicators, df_5m_enhanced, df_1m_enhanced = self.calculate_advanced_indicators(df_5m, df_1m)
                
                # Generate signals based on 5-minute analysis points
                for i in range(50, len(df_5m_enhanced), 12):  # Every hour (12 x 5min)
                    current_candle = df_5m_enhanced.iloc[i]
                    current_price = current_candle['close']
                    
                    # Update indicators for current point
                    current_indicators = {
                        '5m': df_5m_enhanced.iloc[i].to_dict(),
                        '1m': df_1m_enhanced.iloc[min(i*5, len(df_1m_enhanced)-1)].to_dict()
                    }
                    
                    # Generate signals
                    signals = self.generate_enhanced_signals(symbol_name, current_indicators, current_price)
                    total_signals += len(signals)
                    
                    # Execute trades
                    for signal in signals:
                        if len(self.positions) < self.config['max_positions']:
                            signal['timestamp'] = current_candle.name
                            trade = self.execute_advanced_trade(signal)
                    
                    # Manage existing positions
                    self.manage_positions_with_1m_data(df_1m_enhanced, symbol_name)
        
        # Close any remaining positions
        for trade in self.positions[:]:
            if trade['status'] == 'OPEN':
                last_price = trade['entry_price']  # Use entry price as fallback
                self.close_advanced_position(trade, last_price, "Backtest End")
        
        print(f"\n📊 BACKTEST COMPLETED")
        print(f"   Total Signals Generated: {total_signals}")
        print(f"   Total Trades Executed: {len(self.trades)}")
        
        # Generate comprehensive results
        self.generate_january_2026_results()
    
    def generate_january_2026_results(self):
        """Generate comprehensive results for January 2026 backtest"""
        
        print("\n" + "=" * 80)
        print("📊 JANUARY 2026 ENHANCED BACKTEST RESULTS")
        print("=" * 80)
        
        if not self.trades:
            print("❌ No trades executed during the backtest period")
            return
        
        closed_trades = [t for t in self.trades if t.get('status') == 'CLOSED' or t.get('exit_price') is not None]
        total_trades = len(closed_trades)
        
        print(f"📊 Total trades in system: {len(self.trades)}")
        print(f"📊 Trades with exit data: {total_trades}")
        
        if total_trades == 0:
            print("❌ No trades completed - showing all trades for debugging:")
            for i, trade in enumerate(self.trades[:5], 1):  # Show first 5 for debugging
                print(f"   Trade #{i}: {trade.get('signal')} {trade.get('symbol')} - Status: {trade.get('status')} - Exit: {trade.get('exit_price')}")
            return
        
        # Performance metrics
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in closed_trades)
        total_return = (total_pnl / self.initial_capital) * 100
        win_rate = (len(winning_trades) / total_trades) * 100
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        max_win = max([t['pnl'] for t in closed_trades])
        max_loss = min([t['pnl'] for t in closed_trades])
        
        # Display results
        print(f"💰 FINANCIAL PERFORMANCE:")
        print(f"   📊 Period: January 1 - February 5, 2026 (25 trading days)")
        print(f"   💵 Initial Capital:     ₹{self.initial_capital:,.0f}")
        print(f"   💵 Final Capital:       ₹{self.current_capital:,.0f}")
        print(f"   💰 Total P&L:           ₹{total_pnl:,.0f}")
        print(f"   📈 Total Return:        {total_return:.2f}%")
        print(f"   📊 Daily Return:        {total_return/25:.3f}% per day")
        
        print(f"\n📊 TRADING STATISTICS:")
        print(f"   🎯 Total Trades:        {total_trades}")
        print(f"   ✅ Winning Trades:      {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"   ❌ Losing Trades:       {len(losing_trades)} ({100-win_rate:.1f}%)")
        print(f"   💚 Average Win:         ₹{avg_win:,.0f}")
        print(f"   ❤️ Average Loss:        ₹{avg_loss:,.0f}")
        print(f"   🚀 Best Trade:          ₹{max_win:,.0f}")
        print(f"   💔 Worst Trade:         ₹{max_loss:,.0f}")
        
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            print(f"   ⚖️ Profit Factor:       {profit_factor:.2f}")
        
        # Symbol-wise performance
        print(f"\n🎯 SYMBOL PERFORMANCE:")
        print("-" * 60)
        
        for symbol in self.trading_symbols.keys():
            symbol_trades = [t for t in closed_trades if t['symbol'] == symbol]
            if symbol_trades:
                symbol_pnl = sum(t['pnl'] for t in symbol_trades)
                symbol_wins = len([t for t in symbol_trades if t['pnl'] > 0])
                symbol_win_rate = (symbol_wins / len(symbol_trades)) * 100
                avg_confidence = np.mean([t['confidence'] for t in symbol_trades])
                
                print(f"   {symbol:15} | Trades: {len(symbol_trades):2d} | P&L: ₹{symbol_pnl:8,.0f} | Win: {symbol_win_rate:5.1f}% | Conf: {avg_confidence:4.1f}%")
        
        # Strategy effectiveness analysis
        print(f"\n🔍 STRATEGY ANALYSIS:")
        
        # Analyze most effective conditions
        all_conditions = []
        for trade in winning_trades:
            all_conditions.extend(trade.get('conditions', []))
        
        if all_conditions:
            from collections import Counter
            condition_counts = Counter(all_conditions)
            print("   🏆 Most Effective Conditions:")
            for condition, count in condition_counts.most_common(5):
                print(f"      • {condition}: appeared in {count} winning trades")
        
        # Time analysis
        trade_hours = [t['timestamp'].hour for t in closed_trades if hasattr(t['timestamp'], 'hour')]
        if trade_hours:
            from collections import Counter
            hour_performance = Counter(trade_hours)
            print(f"\n   ⏰ Best Trading Hours:")
            for hour, count in hour_performance.most_common(3):
                print(f"      • {hour:02d}:00 - {hour+1:02d}:00: {count} trades")
        
        # Recent trades
        print(f"\n📋 RECENT TRADES (Last 10):")
        print("-" * 80)
        
        recent_trades = closed_trades[-10:] if len(closed_trades) >= 10 else closed_trades
        
        for i, trade in enumerate(recent_trades, 1):
            pnl_emoji = "💚" if trade['pnl'] > 0 else "❤️"
            timestamp_str = trade['timestamp'].strftime('%m-%d %H:%M') if hasattr(trade['timestamp'], 'strftime') else str(trade['timestamp'])
            
            print(f"{pnl_emoji} #{len(closed_trades)-len(recent_trades)+i:2d} {timestamp_str} | {trade['symbol']:15} {trade['signal']:4}")
            print(f"     Entry: ₹{trade['entry_price']:7.2f} → Exit: ₹{trade['exit_price']:7.2f} | P&L: ₹{trade['pnl']:8,.0f}")
            print(f"     Reason: {trade['exit_reason']} | Confidence: {trade['confidence']:3.0f}% | Conditions: {len(trade.get('conditions', []))}")
            
            # Show option sentiment if available
            if 'option_sentiment' in trade:
                opt_sentiment = trade['option_sentiment']
                print(f"     Options: {opt_sentiment['sentiment']} ({opt_sentiment['strength']:.0f}%)")
            
        print("\n" + "=" * 80)
        
        # Final assessment
        if total_return > 15:
            print("🎉 OUTSTANDING: HIGHLY PROFITABLE STRATEGY!")
            status = "EXCELLENT"
        elif total_return > 8:
            print("🚀 EXCELLENT: VERY PROFITABLE STRATEGY!")
            status = "VERY_GOOD"
        elif total_return > 3:
            print("✅ GOOD: PROFITABLE STRATEGY")
            status = "GOOD"
        elif total_return > 0:
            print("🆗 MODERATE: MARGINALLY PROFITABLE")
            status = "MODERATE"
        else:
            print("⚠️ NEEDS OPTIMIZATION: STRATEGY UNDERPERFORMED")
            status = "NEEDS_WORK"
        
        print(f"📊 Performance Rating: {status}")
        print("=" * 80)
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'status': status
        }

def main():
    """Run January 2026 enhanced backtesting"""
    
    print("🎯 FYERS ADVANCED INDICES BACKTESTING - JANUARY 2026")
    print("Multi-timeframe strategy with Option Chain Analysis")
    print("Period: January 1, 2026 to February 5, 2026")
    print("=" * 80)
    
    # Initialize advanced backtester
    backtester = AdvancedIndicesBacktester(initial_capital=250000)
    
    # Run January 2026 backtest
    backtester.run_january_2026_backtest()

if __name__ == "__main__":
    main()