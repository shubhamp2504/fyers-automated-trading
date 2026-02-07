"""
Live Index Trading Execution System - REAL MONEY TRADING
========================================================

⚠️ CRITICAL: This system uses REAL Fyers API with LIVE money
⚠️ Source: https://myapi.fyers.in/docsv3 

Features:
- ✅ Real-time market data from Fyers API
- ✅ Live order placement with real money
- ✅ Automated signal generation and execution
- ✅ Real-time risk management
- ✅ Position monitoring and P&L tracking
- ❌ NO DEMO/DUMMY DATA - LIVE TRADING ONLY

RISK WARNING: This system places real orders with real money.
Always use proper risk management and position sizing.
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

# Import Fyers client and strategy
from fyers_client import FyersClient
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

class LiveIndexTradingSystem:
    """
    LIVE Index Trading System - Real Money Trading with Fyers API
    
    ⚠️ WARNING: This system trades with REAL MONEY on live markets
    """
    
    def __init__(self, fyers_client: FyersClient, config_file: str = 'fyers_config.json'):
        """Initialize live trading system with real Fyers API client"""
        
        print("🚨 INITIALIZING LIVE TRADING SYSTEM")
        print("   ⚠️  Trading with REAL MONEY via Fyers API")
        print("   📊 Live market data streaming")
        print("   💰 Real order placement enabled")
        
        # Use provided Fyers client (already verified)
        self.fyers_client = fyers_client
        
        # Load trading configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Verify live trading is enabled
        if not self.config['trading']['live_trading']:
            raise ValueError("❌ Live trading disabled in config. Enable 'live_trading': true")
        
        # Initialize strategy with optimized parameters
        self.strategy = IndexIntradayStrategy(
            client_id=self.fyers_client.client_id,
            access_token=self.fyers_client.access_token
        )
        
        # Trading symbols for live trading
        self.trading_symbols = {
            'NIFTY': self.config['symbols']['NIFTY_INDEX'],
            'BANKNIFTY': self.config['symbols']['BANKNIFTY_INDEX']
        }
        
        # Risk management from config
        self.max_daily_loss = self.config['trading']['max_daily_loss']
        self.max_positions = self.config['trading']['max_open_positions']
        self.risk_per_trade = self.config['trading']['risk_per_trade']
        self.stop_loss_pct = self.config['trading']['stop_loss_percentage']
        
        # Live trading state
        self.active_trades: Dict[str, LiveTrade] = {}
        self.daily_pnl = 0
        self.total_trades_today = 0
        self.is_trading_active = False
        self.last_signal_time = {}
        
        # Market hours check
        self.market_start = self.config['market_hours']['start_time']
        self.market_end = self.config['market_hours']['end_time']
        
        print(f"✅ Live trading system initialized")
        print(f"   🎯 Max daily loss: ₹{self.max_daily_loss:,.2f}")
        print(f"   📈 Max positions: {self.max_positions}")
        print(f"   ⏰ Market hours: {self.market_start} - {self.market_end}")
        
        # Initial portfolio status
        self._display_portfolio_status()
    
    def _display_portfolio_status(self):
        """Display current portfolio status from live Fyers account"""
        try:
            print("\n📊 LIVE PORTFOLIO STATUS:")
            
            # Get real account funds
            funds = self.fyers_client.get_funds()
            if funds:
                print(f"   💰 Available funds: ₹{funds.get('availableAmount', 0):,.2f}")
                print(f"   📊 Used margin: ₹{funds.get('utilisedAmount', 0):,.2f}")
            
            # Get live positions
            positions = self.fyers_client.get_positions()
            if positions and len(positions) > 0:
                print(f"   📈 Open positions: {len(positions)}")
                for pos in positions[:3]:  # Show first 3
                    symbol = pos.get('symbol', 'Unknown')
                    qty = pos.get('qty', 0)
                    pnl = pos.get('pl', 0)
                    print(f"      {symbol}: {qty} qty, P&L: ₹{pnl:.2f}")
            else:
                print("   📈 No open positions")
            
            print("   ✅ Portfolio status updated from live Fyers API\n")
            
        except Exception as e:
            print(f"   ⚠️ Unable to fetch live portfolio: {e}\n")
    
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
        """Get current market price using live Fyers API"""
        
        try:
            quotes = self.fyers_client.get_live_quotes([symbol])
            if quotes and 'd' in quotes and quotes['d']:
                data = quotes['d']
                if symbol in data and 'v' in data[symbol] and 'lp' in data[symbol]['v']:
                    return data[symbol]['v']['lp']  # Last price
            return None
        except Exception as e:
            print(f"❌ Error getting live price for {symbol}: {e}")
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
            
            print(f"📋 Placing LIVE order via Fyers API: {signal.signal.value} {quantity} units of {symbol}")
            
            # Place order using live Fyers API
            order_result = self.fyers_client.place_order(
                symbol=symbol,
                qty=quantity,
                side=signal.signal.value,  # "BUY" or "SELL"
                order_type="MARKET",
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
            
            # Place exit order using live Fyers API
            exit_side = "SELL" if trade.signal == SignalType.BUY else "BUY"
            
            exit_result = self.fyers_client.place_order(
                symbol=trade.symbol,
                qty=exit_quantity,
                side=exit_side,
                order_type="MARKET",
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
            
            # Place exit order for remaining quantity using live Fyers API
            exit_side = "SELL" if trade.signal == SignalType.BUY else "BUY"
            
            exit_result = self.fyers_client.place_order(
                symbol=trade.symbol,
                qty=trade.quantity,
                side=exit_side,
                order_type="MARKET",
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
    
    def run(self):
        """Main method to start the live trading system"""
        
        print("🚀 STARTING LIVE FYERS TRADING SYSTEM")
        print("=" * 60)
        
        try:
            # Pre-flight checks
            if not self.is_market_open():
                print(f"⏰ Market is closed. System will wait for market hours.")
                print(f"📅 Trading hours: {self.market_start} - {self.market_end}")
                
                # Wait for market to open (optional)
                while not self.is_market_open():
                    print("⏳ Waiting for market to open...")
                    time.sleep(300)  # Check every 5 minutes
            
            # Confirm ready to trade
            print("\n📊 LIVE TRADING SYSTEM READY")
            print("⚠️  WARNING: This will place REAL orders with REAL money")
            print("💰 Ensure adequate funds and risk management")
            
            # Start trading
            self.is_trading_active = True
            self.start_trading_loop()
            
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
            self.stop_trading()
            
        except Exception as e:
            print(f"\n❌ Error in trading system: {e}")
            print("🚨 Trading stopped due to error")
            self.stop_trading()
    
    def start_trading_loop(self):
        """Main trading loop with real market data"""
        
        print("🔄 Starting live trading loop...")
        print("📊 Monitoring symbols:", list(self.trading_symbols.keys()))
        print("⏰ Press Ctrl+C to stop\n")
        
        while self.is_trading_active and self.is_market_open():
            try:
                # Display current status
                self._display_live_status()
                
                # Check for new trading signals
                for name, symbol in self.trading_symbols.items():
                    if not self.can_place_new_trades():
                        break
                        
                    try:
                        signal = self.generate_live_signal(symbol)
                        if signal and signal.confidence >= 0.7:
                            print(f"🎯 HIGH CONFIDENCE SIGNAL for {name}:")
                            print(f"   📈 Direction: {signal.signal.value}")
                            print(f"   💰 Entry: ₹{signal.entry_price:.2f}")
                            print(f"   🎯 Confidence: {signal.confidence:.1%}")
                            
                            if self.execute_live_trade(symbol, signal):
                                print(f"✅ Trade executed successfully for {name}")
                            else:
                                print(f"❌ Trade execution failed for {name}")
                    
                    except Exception as e:
                        print(f"⚠️ Error processing {name}: {e}")
                        continue
                
                # Monitor existing positions
                self.monitor_active_positions()
                
                # Brief pause before next iteration
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
        
        # Trading stopped
        if not self.is_market_open():
            print("⏰ Market closed - trading stopped")
        
        self.stop_trading()
    
    def _display_live_status(self):
        """Display current live trading status"""
        
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n⏰ {now} - LIVE TRADING STATUS:")
        print(f"   📈 Active positions: {len(self.active_trades)}")
        print(f"   💰 Daily P&L: ₹{self.daily_pnl:+,.2f}")
        print(f"   📊 Trades today: {self.total_trades_today}")
        
        if self.active_trades:
            print("   📋 Active trades:")
            for trade_id, trade in list(self.active_trades.items())[:3]:  # Show up to 3
                pnl_color = "📈" if trade.unrealized_pnl >= 0 else "📉"
                print(f"      {pnl_color} {trade.symbol}: {trade.signal.value} | P&L: ₹{trade.unrealized_pnl:+.2f}")
    
    def stop_trading(self):
        """Stop trading and clean up"""
        
        print("\n🛑 STOPPING LIVE TRADING SYSTEM")
        self.is_trading_active = False
        
        # Close any open positions
        if self.active_trades:
            print("📊 Closing remaining positions...")
            for trade_id in list(self.active_trades.keys()):
                self.exit_position(trade_id, "SYSTEM_SHUTDOWN")
        
        # Final portfolio update
        self._display_portfolio_status()
        
        print("✅ Trading system stopped safely")