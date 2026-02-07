"""
Live Index Trading Execution System
===================================

Real-time implementation of optimized index intraday strategy
- Live market data monitoring
- Automated signal generation and execution
- Real-time risk management
- Position monitoring and profit/loss tracking
- Emergency stop mechanisms

⚠️ IMPORTANT: Always refer to https://myapi.fyers.in/docsv3 for latest API specifications
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
import warnings
warnings.filterwarnings('ignore')

# Import our modules
sys.path.append('api_reference/market_data')
sys.path.append('api_reference/orders')
sys.path.append('api_reference/portfolio')
sys.path.append('api_reference/websocket')

from market_data_complete import FyersMarketData
from orders_complete import FyersOrders
from portfolio_complete import FyersPortfolio
from index_intraday_strategy import IndexIntradayStrategy, SignalType, TradingSignal

class TradeStatus(Enum):
    PENDING = "PENDING"
    ENTERED = "ENTERED"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

@dataclass
class LiveTrade:
    """Live trade tracking"""
    trade_id: str
    symbol: str
    signal: SignalType
    entry_time: datetime
    entry_price: float
    quantity: int
    stop_loss: float
    target_1: float
    target_2: float
    current_price: float
    unrealized_pnl: float
    status: TradeStatus
    order_ids: List[str]
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None

class LiveTradingSystem:
    """
    Live trading system for index intraday strategy
    """
    
    def __init__(self, config_file: str = 'config.json'):
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Initialize FYERS APIs
        self.market_data = FyersMarketData(self.config['client_id'], self.config['access_token'])
        self.orders = FyersOrders(self.config['client_id'], self.config['access_token'])
        self.portfolio = FyersPortfolio(self.config['client_id'], self.config['access_token'])
        
        # Load optimized strategy parameters
        self.strategy = self.load_optimized_strategy()
        
        # Trading configuration
        self.trading_symbols = {
            'NIFTY': 'NSE:NIFTY50-INDEX',
            'BANKNIFTY': 'NSE:NIFTYBANK-INDEX'
        }
        
        # Risk management
        self.max_daily_loss = 5000  # Maximum daily loss in rupees
        self.max_positions = 2  # Maximum concurrent positions
        self.position_size_per_trade = 1  # Lot size multiplier
        
        # Live trading state
        self.active_trades: Dict[str, LiveTrade] = {}
        self.daily_pnl = 0
        self.total_trades_today = 0
        self.is_trading_active = False
        self.last_signal_time = {}
        
        # Market hours
        self.market_open_time = "09:15"
        self.market_close_time = "15:15"
        self.strategy_stop_time = "14:30"  # Stop new trades 45 min before close
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        print("🚀 Live Trading System Initialized")
        print(f"📊 Trading Symbols: {list(self.trading_symbols.keys())}")
        print(f"💰 Max Daily Loss: ₹{self.max_daily_loss:,}")
        print(f"🎯 Max Positions: {self.max_positions}")
    
    def load_optimized_strategy(self) -> IndexIntradayStrategy:
        """Load strategy with optimized parameters"""
        
        strategy = IndexIntradayStrategy(self.config['client_id'], self.config['access_token'])
        
        # Try to load optimized parameters
        try:
            for symbol_key in ['nifty50-index', 'niftybank-index']:
                param_file = f"optimized_params_{symbol_key}.json"
                try:
                    with open(param_file, 'r') as f:
                        params = json.load(f)
                        
                    # Apply best parameters
                    best_params = params.get('best_parameters', {})
                    if best_params:
                        strategy.profit_target_1 = best_params.get('profit_target_1', 22)
                        strategy.profit_target_2 = best_params.get('profit_target_2', 28)
                        strategy.max_loss_per_trade = best_params.get('max_loss_per_trade', 15)
                        strategy.ema_fast = best_params.get('ema_fast', 9)
                        strategy.ema_slow = best_params.get('ema_slow', 21)
                        
                        print(f"✅ Loaded optimized parameters for {symbol_key}")
                        break
                        
                except FileNotFoundError:
                    continue
            else:
                print("⚠️ No optimized parameters found, using default values")
                
        except Exception as e:
            print(f"⚠️ Error loading optimized parameters: {e}")
            print("📊 Using default strategy parameters")
        
        return strategy
    
    def is_market_open(self) -> bool:
        """Check if market is open for trading"""
        
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_weekday = now.weekday()
        
        # Check if it's a weekday
        if current_weekday >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check trading hours
        if self.market_open_time <= current_time <= self.market_close_time:
            return True
        
        return False
    
    def can_place_new_trades(self) -> bool:
        """Check if new trades can be placed"""
        
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        # Check market hours
        if not self.is_market_open():
            return False
        
        # Stop new trades before market close
        if current_time > self.strategy_stop_time:
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            print(f"🛑 Daily loss limit reached: ₹{self.daily_pnl:,.2f}")
            return False
        
        # Check maximum positions
        if len(self.active_trades) >= self.max_positions:
            return False
        
        return True
    
    def get_current_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        
        try:
            quotes = self.market_data.get_quotes([symbol])
            if quotes and len(quotes) > 0:
                return quotes[0].get('lp', 0)  # Last price
            return None
        except Exception as e:
            print(f"❌ Error getting price for {symbol}: {e}")
            return None
    
    def generate_live_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate trading signal for live trading"""
        
        try:
            # Avoid generating signals too frequently
            symbol_key = symbol.split(':')[1]  # Get symbol name
            last_signal = self.last_signal_time.get(symbol_key)
            
            if last_signal and (datetime.now() - last_signal).seconds < 300:  # 5 minutes
                return None
            
            # Generate signal using our strategy
            signal = self.strategy.generate_signal_1h(symbol)
            
            if signal:
                # Update last signal time
                self.last_signal_time[symbol_key] = datetime.now()
                
                # Additional live trading filters
                current_price = self.get_current_market_price(symbol)
                if not current_price:
                    return None
                
                # Confirm signal is still valid with current price
                price_diff = abs(current_price - signal.entry_price)
                if price_diff > 5:  # Price moved too much since signal generation
                    return None
                
                # Update signal with current price
                signal.entry_price = current_price
                
                # Confirm with 5-minute data
                if not self.strategy.confirm_entry_5min(signal, symbol):
                    print(f"❌ 5-min confirmation failed for {symbol}")
                    return None
                
                print(f"🎯 Live signal generated for {symbol}: {signal.signal.value}")
                print(f"   💰 Entry: ₹{signal.entry_price:.2f}")
                print(f"   🎯 Targets: ₹{signal.target_1:.2f} | ₹{signal.target_2:.2f}")
                print(f"   🛡️ Stop Loss: ₹{signal.stop_loss:.2f}")
                
                return signal
            
            return None
            
        except Exception as e:
            print(f"❌ Error generating live signal for {symbol}: {e}")
            return None
    
    def execute_live_trade(self, signal: TradingSignal, symbol: str) -> bool:
        """Execute live trade based on signal"""
        
        try:
            # Calculate position size
            if 'NIFTY50' in symbol:
                lot_size = 25
            elif 'NIFTYBANK' in symbol:
                lot_size = 15
            else:
                lot_size = 25  # Default
            
            quantity = self.position_size_per_trade * lot_size
            
            # Place market order
            side = 1 if signal.signal == SignalType.BUY else -1
            
            print(f"📋 Placing order: {signal.signal.value} {quantity} units of {symbol}")
            
            order_result = self.orders.place_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type=1,  # Market order
                product_type="INTRADAY"
            )
            
            if order_result and 'id' in order_result:
                order_id = order_result['id']
                
                # Create trade record
                trade_id = f"{symbol}_{int(datetime.now().timestamp())}"
                
                live_trade = LiveTrade(
                    trade_id=trade_id,
                    symbol=symbol,
                    signal=signal.signal,
                    entry_time=datetime.now(),
                    entry_price=signal.entry_price,
                    quantity=quantity,
                    stop_loss=signal.stop_loss,
                    target_1=signal.target_1,
                    target_2=signal.target_2,
                    current_price=signal.entry_price,
                    unrealized_pnl=0,
                    status=TradeStatus.ENTERED,
                    order_ids=[order_id]
                )
                
                self.active_trades[trade_id] = live_trade
                self.total_trades_today += 1
                
                print(f"✅ Trade executed successfully!")
                print(f"   📊 Trade ID: {trade_id}")
                print(f"   📈 Order ID: {order_id}")
                print(f"   📊 Quantity: {quantity}")
                
                return True
            else:
                print(f"❌ Failed to place order: {order_result}")
                return False
                
        except Exception as e:
            print(f"❌ Error executing live trade: {e}")
            return False
    
    def monitor_active_positions(self):
        """Monitor active positions and manage exits"""
        
        if not self.active_trades:
            return
        
        for trade_id, trade in list(self.active_trades.items()):
            try:
                # Get current market price
                current_price = self.get_current_market_price(trade.symbol)
                if not current_price:
                    continue
                
                # Update current price and unrealized P&L
                trade.current_price = current_price
                
                if trade.signal == SignalType.BUY:
                    unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
                else:
                    unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
                
                trade.unrealized_pnl = unrealized_pnl
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                if trade.signal == SignalType.BUY:
                    # Buy position exit conditions
                    if current_price <= trade.stop_loss:
                        should_exit = True
                        exit_reason = "STOP_LOSS"
                    elif current_price >= trade.target_2:
                        should_exit = True
                        exit_reason = "TARGET_2"
                    elif current_price >= trade.target_1 and trade.status == TradeStatus.ENTERED:
                        # Partial exit at target 1
                        self.partial_exit_position(trade_id, "TARGET_1")
                        continue
                        
                else:  # SELL position
                    if current_price >= trade.stop_loss:
                        should_exit = True
                        exit_reason = "STOP_LOSS"
                    elif current_price <= trade.target_2:
                        should_exit = True
                        exit_reason = "TARGET_2"
                    elif current_price <= trade.target_1 and trade.status == TradeStatus.ENTERED:
                        # Partial exit at target 1
                        self.partial_exit_position(trade_id, "TARGET_1")
                        continue
                
                # Force exit near market close
                current_time = datetime.now().strftime("%H:%M")
                if current_time > "15:00":
                    should_exit = True
                    exit_reason = "MARKET_CLOSE"
                
                if should_exit:
                    self.exit_position(trade_id, exit_reason)
                
            except Exception as e:
                print(f"❌ Error monitoring position {trade_id}: {e}")
                continue
    
    def partial_exit_position(self, trade_id: str, reason: str):
        """Partially exit position at first target"""
        
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                return
            
            # Exit 50% of position
            exit_quantity = trade.quantity // 2
            remaining_quantity = trade.quantity - exit_quantity
            
            # Place exit order
            exit_side = -1 if trade.signal == SignalType.BUY else 1
            
            exit_result = self.orders.place_order(
                symbol=trade.symbol,
                qty=exit_quantity,
                side=exit_side,
                type=1,  # Market order
                product_type="INTRADAY"
            )
            
            if exit_result and 'id' in exit_result:
                # Update trade
                trade.quantity = remaining_quantity
                trade.status = TradeStatus.PARTIAL_EXIT
                
                # Trail stop loss
                if trade.signal == SignalType.BUY:
                    trade.stop_loss = trade.entry_price + 3  # 3 points profit
                else:
                    trade.stop_loss = trade.entry_price - 3  # 3 points profit
                
                print(f"📊 Partial exit executed for {trade_id}")
                print(f"   📈 Reason: {reason}")
                print(f"   🔒 Stop loss trailed to ₹{trade.stop_loss:.2f}")
                print(f"   📊 Remaining quantity: {remaining_quantity}")
                
        except Exception as e:
            print(f"❌ Error in partial exit for {trade_id}: {e}")
    
    def exit_position(self, trade_id: str, reason: str):
        """Exit complete position"""
        
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                return
            
            # Place exit order for remaining quantity
            exit_side = -1 if trade.signal == SignalType.BUY else 1
            
            exit_result = self.orders.place_order(
                symbol=trade.symbol,
                qty=trade.quantity,
                side=exit_side,
                type=1,  # Market order
                product_type="INTRADAY"
            )
            
            if exit_result and 'id' in exit_result:
                # Calculate realized P&L
                current_price = self.get_current_market_price(trade.symbol)
                
                if current_price:
                    if trade.signal == SignalType.BUY:
                        realized_pnl = (current_price - trade.entry_price) * trade.quantity
                    else:
                        realized_pnl = (trade.entry_price - current_price) * trade.quantity
                    
                    # Update trade record
                    trade.exit_time = datetime.now()
                    trade.exit_price = current_price
                    trade.realized_pnl = realized_pnl
                    trade.status = TradeStatus.CLOSED
                    
                    # Update daily P&L
                    self.daily_pnl += realized_pnl
                    
                    print(f"🏁 Position closed: {trade_id}")
                    print(f"   📊 Reason: {reason}")
                    print(f"   💰 Entry: ₹{trade.entry_price:.2f} → Exit: ₹{current_price:.2f}")
                    print(f"   💰 Realized P&L: ₹{realized_pnl:+.2f}")
                    print(f"   📊 Daily P&L: ₹{self.daily_pnl:+.2f}")
                    
                    # Remove from active trades
                    del self.active_trades[trade_id]
                
        except Exception as e:
            print(f"❌ Error exiting position {trade_id}: {e}")
    
    def display_trading_dashboard(self):
        """Display live trading dashboard"""
        
        print(f"\n" + "="*60)
        print(f"🚀 LIVE TRADING DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"="*60)
        
        # Market status
        market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
        trading_status = "✅ ACTIVE" if self.is_trading_active else "⏸️ PAUSED"
        
        print(f"📊 Market: {market_status} | Trading: {trading_status}")
        print(f"💰 Daily P&L: ₹{self.daily_pnl:+,.2f}")
        print(f"📈 Trades Today: {self.total_trades_today}")
        print(f"🎯 Active Positions: {len(self.active_trades)}")
        
        # Active positions
        if self.active_trades:
            print(f"\n🎯 ACTIVE POSITIONS:")
            print("-" * 60)
            for trade_id, trade in self.active_trades.items():
                status_icon = "🟢" if trade.unrealized_pnl > 0 else "🔴"
                print(f"{status_icon} {trade.symbol} {trade.signal.value}")
                print(f"   💰 Entry: ₹{trade.entry_price:.2f} | Current: ₹{trade.current_price:.2f}")
                print(f"   📊 Unrealized P&L: ₹{trade.unrealized_pnl:+.2f}")
                print(f"   🎯 Targets: ₹{trade.target_1:.2f} | ₹{trade.target_2:.2f}")
                print(f"   🛡️ Stop Loss: ₹{trade.stop_loss:.2f}")
        else:
            print(f"\n⏸️ No active positions")
        
        print(f"="*60)
    
    def start_live_trading(self):
        """Start live trading system"""
        
        print(f"🚀 Starting Live Trading System...")
        print(f"⏰ Market Hours: {self.market_open_time} - {self.market_close_time}")
        print(f"🛑 Strategy Stop Time: {self.strategy_stop_time}")
        
        self.is_trading_active = True
        self.stop_monitoring = False
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.monitoring_thread.start()
        
        print(f"✅ Live trading system started!")
        print(f"💡 Press Ctrl+C to stop trading")
        
        try:
            # Main dashboard loop
            while self.is_trading_active:
                self.display_trading_dashboard()
                time.sleep(30)  # Update dashboard every 30 seconds
                
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping live trading system...")
            self.stop_live_trading()
    
    def trading_loop(self):
        """Main trading loop"""
        
        while not self.stop_monitoring:
            try:
                # Check if we can trade
                if not self.can_place_new_trades():
                    time.sleep(60)  # Wait 1 minute
                    continue
                
                # Scan for signals
                for name, symbol in self.trading_symbols.items():
                    try:
                        # Generate signal
                        signal = self.generate_live_signal(symbol)
                        
                        if signal and signal.confidence >= 0.7:  # High confidence only
                            print(f"🎯 High confidence signal for {name}: {signal.signal.value}")
                            
                            # Execute trade
                            if self.execute_live_trade(signal, symbol):
                                print(f"✅ Trade executed for {name}")
                            else:
                                print(f"❌ Failed to execute trade for {name}")
                        
                        time.sleep(5)  # Small delay between symbols
                        
                    except Exception as e:
                        print(f"❌ Error processing {name}: {e}")
                        continue
                
                # Monitor active positions
                self.monitor_active_positions()
                
                # Wait before next scan
                time.sleep(60)  # Scan every minute
                
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                time.sleep(30)
    
    def stop_live_trading(self):
        """Stop live trading system"""
        
        print(f"🛑 Stopping live trading system...")
        
        self.is_trading_active = False
        self.stop_monitoring = True
        
        # Close all active positions
        if self.active_trades:
            print(f"📊 Closing {len(self.active_trades)} active positions...")
            for trade_id in list(self.active_trades.keys()):
                self.exit_position(trade_id, "SYSTEM_STOP")
        
        print(f"✅ Live trading system stopped")
        print(f"📊 Final Daily P&L: ₹{self.daily_pnl:+,.2f}")
        print(f"📈 Total Trades Today: {self.total_trades_today}")

def main():
    """Main function to run live trading system"""
    
    print(f"🎯 INDEX LIVE TRADING SYSTEM")
    print(f"="*50)
    
    try:
        # Initialize trading system
        trading_system = LiveTradingSystem()
        
        print(f"\n📊 System Checks:")
        print(f"   Market Open: {'✅' if trading_system.is_market_open() else '❌'}")
        print(f"   Can Trade: {'✅' if trading_system.can_place_new_trades() else '❌'}")
        
        if not trading_system.is_market_open():
            print(f"\n⏰ Market is closed. Trading will resume during market hours.")
            print(f"📅 Market Hours: {trading_system.market_open_time} - {trading_system.market_close_time}")
            return
        
        # Start live trading
        user_input = input(f"\n🚀 Start live trading? (y/N): ").lower()
        
        if user_input == 'y':
            trading_system.start_live_trading()
        else:
            print(f"📊 Live trading cancelled by user")
            
    except Exception as e:
        print(f"❌ Error in live trading system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()