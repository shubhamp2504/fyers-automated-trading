"""
Advanced Trading Strategy Framework
Utilizing FYERS API v3 Advanced Features
⚠️  CRITICAL: Always consult https://myapi.fyers.in/docsv3 FIRST before implementing features"""

import json
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

from fyers_client import FyersClient
from advanced_features import AdvancedFyersFeatures
from websocket_stream import FyersWebSocketStream, TradingAlerts

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderType(Enum):
    MARKET = 1
    LIMIT = 2
    STOP_MARKET = 3
    STOP_LIMIT = 4

@dataclass
class TradingSignal:
    symbol: str
    signal: SignalType
    price: float
    confidence: float  # 0.0 to 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: int = 0
    reason: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class StrategyBase:
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.client = FyersClient()
        self.advanced = AdvancedFyersFeatures()
        self.stream = None
        self.alerts = None
        
        # Strategy parameters
        self.enabled = True
        self.max_positions = 5
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.min_confidence = 0.7
        
        # Data storage
        self.live_data = {}
        self.historical_data = {}
        self.signals = []
        self.positions = []
        
    def setup_streaming(self, access_token: str, client_id: str):
        """Setup real-time data streaming"""
        self.stream = FyersWebSocketStream(access_token, client_id)
        self.alerts = TradingAlerts(self.stream)
        
    def add_symbol(self, symbol: str, callback: Optional[Callable] = None):
        """Add symbol for monitoring"""
        if self.stream:
            self.stream.subscribe([symbol], callback or self._on_data_update)
            
    def _on_data_update(self, symbol: str, data: Dict):
        """Handle real-time data updates"""
        self.live_data[symbol] = data
        
        # Generate signals
        signal = self.generate_signal(symbol, data)
        if signal and signal.confidence >= self.min_confidence:
            self.signals.append(signal)
            self.process_signal(signal)
    
    def generate_signal(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Override this method in strategy implementations"""
        raise NotImplementedError("Subclasses must implement generate_signal")
    
    def process_signal(self, signal: TradingSignal):
        """Process trading signal"""
        if not self.enabled:
            return
            
        if len(self.positions) >= self.max_positions:
            print(f"⚠️ Max positions reached ({self.max_positions}), skipping {signal.symbol}")
            return
        
        print(f"🎯 {self.name} Signal: {signal.signal.value} {signal.symbol} at ₹{signal.price}")
        print(f"   Confidence: {signal.confidence:.2%} | Reason: {signal.reason}")
        
        if signal.signal in [SignalType.BUY, SignalType.SELL]:
            self.execute_trade(signal)
    
    def execute_trade(self, signal: TradingSignal):
        """Execute trade based on signal"""
        try:
            # Calculate position size based on risk
            account_value = self.get_account_value()
            risk_amount = account_value * self.risk_per_trade
            
            if signal.stop_loss:
                price_diff = abs(signal.price - signal.stop_loss)
                quantity = max(1, int(risk_amount / price_diff))
            else:
                # Default 2% stop loss
                quantity = max(1, int(risk_amount / (signal.price * 0.02)))
            
            # Place order
            side = 1 if signal.signal == SignalType.BUY else -1
            
            result = self.advanced.place_advanced_order(
                symbol=signal.symbol,
                side=side,
                qty=quantity,
                order_type=OrderType.MARKET.value,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if result:
                position = {
                    "symbol": signal.symbol,
                    "side": signal.signal.value,
                    "quantity": quantity,
                    "entry_price": signal.price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "order_id": result["order_id"],
                    "timestamp": datetime.now(),
                    "strategy": self.name
                }
                
                self.positions.append(position)
                print(f"✅ Trade executed: {signal.signal.value} {quantity} {signal.symbol}")
                
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
    
    def get_account_value(self) -> float:
        """Get total account value"""
        try:
            funds = self.client.get_funds()
            if funds:
                return funds.get('fund_limit', 100000)  # Default 1 lakh
            return 100000
        except:
            return 100000
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data for analysis"""
        if symbol in self.historical_data:
            return self.historical_data[symbol]
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = self.client.get_historical_data(
                symbol=symbol,
                resolution="D",  # Daily data
                date_from=start_date.strftime("%Y-%m-%d"),
                date_to=end_date.strftime("%Y-%m-%d"),
                cont_flag=1
            )
            
            if data and 'candles' in data:
                df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                
                self.historical_data[symbol] = df
                return df
                
        except Exception as e:
            print(f"❌ Historical data error for {symbol}: {e}")
            
        return None

class MomentumStrategy(StrategyBase):
    """Momentum-based trading strategy"""
    
    def __init__(self):
        super().__init__("Momentum Strategy")
        self.momentum_threshold = 0.02  # 2% price move
        self.volume_multiplier = 2.0    # 2x average volume
        
    def generate_signal(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Generate momentum-based signals"""
        try:
            change_percent = data.get('change_percent', 0) / 100
            current_volume = data.get('volume', 0)
            current_price = data.get('ltp', 0)
            
            # Get historical data for volume comparison
            hist_data = self.get_historical_data(symbol, 20)
            if hist_data is None or len(hist_data) < 5:
                return None
            
            avg_volume = hist_data['volume'].tail(10).mean()
            
            # Momentum conditions
            strong_price_move = abs(change_percent) >= self.momentum_threshold
            high_volume = current_volume >= (avg_volume * self.volume_multiplier)
            
            if strong_price_move and high_volume:
                signal_type = SignalType.BUY if change_percent > 0 else SignalType.SELL
                
                # Calculate stop loss and take profit
                if signal_type == SignalType.BUY:
                    stop_loss = current_price * 0.98  # 2% below entry
                    take_profit = current_price * 1.04  # 4% above entry
                else:
                    stop_loss = current_price * 1.02  # 2% above entry
                    take_profit = current_price * 0.96  # 4% below entry
                
                confidence = min(0.95, 0.5 + abs(change_percent) + (current_volume / avg_volume - 1) * 0.1)
                
                return TradingSignal(
                    symbol=symbol,
                    signal=signal_type,
                    price=current_price,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Strong momentum: {change_percent:.1%} move, {current_volume/avg_volume:.1f}x volume"
                )
                
        except Exception as e:
            print(f"❌ Momentum signal error for {symbol}: {e}")
            
        return None

class MeanReversionStrategy(StrategyBase):
    """Mean reversion trading strategy"""
    
    def __init__(self):
        super().__init__("Mean Reversion Strategy")
        self.oversold_threshold = -0.03  # -3% from open
        self.overbought_threshold = 0.03   # +3% from open
        
    def generate_signal(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Generate mean reversion signals"""
        try:
            current_price = data.get('ltp', 0)
            open_price = data.get('open', 0)
            high_price = data.get('high', 0)
            low_price = data.get('low', 0)
            
            if open_price == 0:
                return None
            
            # Calculate intraday move
            intraday_change = (current_price - open_price) / open_price
            
            # Check for extreme moves (potential reversal points)
            if intraday_change <= self.oversold_threshold:
                # Oversold - potential buy signal
                signal_type = SignalType.BUY
                stop_loss = low_price * 0.99  # 1% below day's low
                take_profit = open_price  # Target return to open
                reason = f"Oversold: {intraday_change:.1%} below open"
                
            elif intraday_change >= self.overbought_threshold:
                # Overbought - potential sell signal
                signal_type = SignalType.SELL
                stop_loss = high_price * 1.01  # 1% above day's high
                take_profit = open_price  # Target return to open
                reason = f"Overbought: {intraday_change:.1%} above open"
                
            else:
                return None
            
            # Calculate confidence based on extremity of move
            confidence = min(0.9, 0.5 + abs(intraday_change) * 10)
            
            return TradingSignal(
                symbol=symbol,
                signal=signal_type,
                price=current_price,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason
            )
            
        except Exception as e:
            print(f"❌ Mean reversion signal error for {symbol}: {e}")
            
        return None

class StrategyManager:
    """Manage multiple trading strategies"""
    
    def __init__(self, config_file: str = "config.json"):
        self.strategies = []
        self.config = self.load_config(config_file)
        self.running = False
        
    def load_config(self, config_file: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Config loading error: {e}")
            return {}
    
    def add_strategy(self, strategy: StrategyBase):
        """Add trading strategy"""
        self.strategies.append(strategy)
        
        # Setup streaming for strategy
        if self.config:
            strategy.setup_streaming(
                self.config.get("access_token"),
                self.config.get("client_id")
            )
            
        print(f"✅ Added strategy: {strategy.name}")
    
    def start_trading(self, symbols: List[str]):
        """Start automated trading"""
        print("🚀 Starting Automated Trading System")
        print("=" * 50)
        
        # Connect all strategies to streaming
        for strategy in self.strategies:
            if strategy.stream and not strategy.stream.is_connected:
                if strategy.stream.connect():
                    for symbol in symbols:
                        strategy.add_symbol(symbol)
                        
        print(f"📊 Monitoring {len(symbols)} symbols with {len(self.strategies)} strategies")
        
        self.running = True
        
        try:
            while self.running:
                time.sleep(10)
                self.print_status()
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping automated trading...")
            self.stop_trading()
    
    def stop_trading(self):
        """Stop automated trading"""
        self.running = False
        
        for strategy in self.strategies:
            if strategy.stream:
                strategy.stream.disconnect()
        
        print("✅ Automated trading stopped")
    
    def print_status(self):
        """Print current status"""
        total_signals = sum(len(s.signals) for s in self.strategies)
        total_positions = sum(len(s.positions) for s in self.strategies)
        
        print(f"📊 Status: {total_signals} signals generated, {total_positions} active positions")
        
        for strategy in self.strategies:
            if strategy.signals:
                latest_signal = strategy.signals[-1]
                print(f"   {strategy.name}: Latest - {latest_signal.signal.value} {latest_signal.symbol}")

def demo_advanced_trading():
    """Demonstrate advanced trading system"""
    
    print("🚀 FYERS Advanced Trading System Demo")
    print("=" * 60)
    
    # Create strategy manager
    manager = StrategyManager()
    
    # Add strategies
    momentum = MomentumStrategy()
    mean_reversion = MeanReversionStrategy()
    
    # Configure strategies
    momentum.min_confidence = 0.8
    momentum.max_positions = 3
    
    mean_reversion.min_confidence = 0.7
    mean_reversion.max_positions = 2
    
    manager.add_strategy(momentum)
    manager.add_strategy(mean_reversion)
    
    # Define symbols to trade
    nifty_50_symbols = [
        "NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:INFY-EQ", "NSE:HDFCBANK-EQ",
        "NSE:ITC-EQ", "NSE:HINDUNILVR-EQ", "NSE:SBIN-EQ", "NSE:ICICIBANK-EQ",
        "NSE:BHARTIARTL-EQ", "NSE:KOTAKBANK-EQ"
    ]
    
    print("💡 Starting with paper trading mode...")
    print("💡 Press Ctrl+C to stop")
    
    # Start trading (in demo mode)
    manager.start_trading(nifty_50_symbols)

if __name__ == "__main__":
    demo_advanced_trading()