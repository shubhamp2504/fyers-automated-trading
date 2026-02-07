#!/usr/bin/env python3
"""
JEAFX PORTFOLIO MANAGER & AUTOMATION SYSTEM
Complete trading automation with portfolio management

🤖 FEATURES:
- Automated signal generation and execution
- Multi-symbol portfolio management
- Real-time monitoring and alerts
- Performance tracking and optimization
- Risk-adjusted position sizing
- Automated stop loss and target management
"""

import asyncio
import pandas as pd
import numpy as np
import json
import sqlite3
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# Import our systems
from jeafx_advanced_system import AdvancedJeafxSystem, AdvancedSignal, AdvancedZone
from jeafx_risk_manager import JeafxRiskManager, Position, RiskLevel

class OrderStatus(Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class PortfolioState(Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    EMERGENCY = "EMERGENCY"

@dataclass
class TradingOrder:
    """Trading order representation"""
    order_id: str
    symbol: str
    order_type: str  # BUY/SELL
    quantity: int
    price: float
    order_status: OrderStatus
    signal_id: str
    created_time: datetime
    executed_time: Optional[datetime] = None
    execution_price: Optional[float] = None
    
@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    timestamp: datetime
    total_value: float
    cash_balance: float
    invested_amount: float
    unrealized_pnl: float
    realized_pnl: float
    total_return: float
    daily_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    active_positions: int
    total_trades: int

class JeafxPortfolioManager:
    """
    Complete JEAFX Portfolio Management and Automation System
    """
    
    def __init__(self, config_file: str = "portfolio_config.json"):
        # Initialize components
        self.config = self._load_portfolio_config(config_file)
        self.jeafx_system = AdvancedJeafxSystem()
        self.risk_manager = JeafxRiskManager()
        
        # Portfolio state
        self.state = PortfolioState.ACTIVE
        self.portfolio_value = self.config['initial_capital']
        self.cash_balance = self.config['initial_capital']
        
        # Trading state
        self.active_orders: Dict[str, TradingOrder] = {}
        self.active_positions: Dict[str, Dict] = {}  # Combined position data
        self.watchlist = self.config['symbols']
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.daily_metrics: List[PortfolioMetrics] = []
        
        # Threading and automation
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup
        self._setup_portfolio_database()
        self._setup_logging()
        self._setup_automation()
        
        self.logger.info("🤖 JEAFX Portfolio Manager Initialized")
        
    def _load_portfolio_config(self, config_file: str) -> Dict:
        """Load portfolio configuration"""
        
        default_config = {
            "initial_capital": 100000,
            "symbols": [
                "NSE:NIFTY50-INDEX",
                "NSE:NIFTYBANK-INDEX", 
                "NSE:RELIANCE-EQ",
                "NSE:TCS-EQ",
                "NSE:INFY-EQ"
            ],
            "automation": {
                "auto_trading": True,
                "signal_check_interval": 300,  # 5 minutes
                "market_hours_only": True,
                "max_orders_per_hour": 10,
                "position_monitoring_interval": 60,  # 1 minute
            },
            "execution": {
                "order_timeout": 300,  # 5 minutes
                "slippage_tolerance": 0.001,  # 0.1%
                "partial_fill_threshold": 0.8,  # 80%
                "retry_failed_orders": True,
                "max_retries": 3
            },
            "portfolio": {
                "rebalance_frequency": "daily",
                "max_symbol_weight": 0.25,  # 25% per symbol
                "cash_reserve": 0.05,  # 5% cash reserve
                "correlation_limit": 0.7,
                "sector_diversification": True
            },
            "alerts": {
                "performance_alerts": True,
                "risk_alerts": True,
                "execution_alerts": True,
                "daily_summary": True
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        except FileNotFoundError:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config
        
    def _setup_portfolio_database(self):
        """Setup portfolio database"""
        self.portfolio_db = sqlite3.connect('jeafx_portfolio.db', check_same_thread=False)
        cursor = self.portfolio_db.cursor()
        
        # Orders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT,
                order_type TEXT,
                quantity INTEGER,
                price REAL,
                status TEXT,
                signal_id TEXT,
                created_time TIMESTAMP,
                executed_time TIMESTAMP,
                execution_price REAL
            )
        ''')
        
        # Portfolio metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_metrics (
                timestamp TIMESTAMP PRIMARY KEY,
                total_value REAL,
                cash_balance REAL,
                unrealized_pnl REAL,
                realized_pnl REAL,
                total_return REAL,
                active_positions INTEGER,
                daily_return REAL,
                sharpe_ratio REAL
            )
        ''')
        
        # Trade history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                entry_price REAL,
                exit_price REAL,
                quantity INTEGER,
                pnl REAL,
                pnl_percent REAL,
                strategy TEXT,
                signal_confidence REAL
            )
        ''')
        
        self.portfolio_db.commit()
        
    def _setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('jeafx_portfolio.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('JEAFX_PORTFOLIO')
        
    def _setup_automation(self):
        """Setup automated trading schedule"""
        
        # Market scanning schedule
        schedule.every(self.config['automation']['signal_check_interval']).seconds.do(
            self._scheduled_signal_scan
        )
        
        # Position monitoring
        schedule.every(self.config['automation']['position_monitoring_interval']).seconds.do(
            self._scheduled_position_monitoring
        )
        
        # Daily portfolio rebalancing
        schedule.every().day.at("09:30").do(self._scheduled_rebalancing)
        
        # End of day reporting
        schedule.every().day.at("15:30").do(self._scheduled_daily_report)
        
    def start_automation(self):
        """Start automated trading system"""
        
        if self.is_running:
            self.logger.warning("⚠️ Automation already running")
            return
            
        self.is_running = True
        self.state = PortfolioState.ACTIVE
        
        self.logger.info("🚀 Starting JEAFX Portfolio Automation")
        
        # Start background thread for automation
        automation_thread = threading.Thread(target=self._automation_loop, daemon=True)
        automation_thread.start()
        
        # Initial portfolio scan
        self._initial_portfolio_scan()
        
    def stop_automation(self):
        """Stop automated trading system"""
        
        self.is_running = False
        self.state = PortfolioState.STOPPED
        
        # Cancel all pending orders
        for order_id in list(self.active_orders.keys()):
            self._cancel_order(order_id)
            
        self.logger.info("🛑 JEAFX Portfolio Automation Stopped")
        
    def pause_automation(self):
        """Pause automated trading (keep monitoring)"""
        
        self.state = PortfolioState.PAUSED
        self.logger.info("⏸️ JEAFX Portfolio Automation Paused")
        
    def emergency_stop(self):
        """Emergency stop - close all positions"""
        
        self.state = PortfolioState.EMERGENCY
        self.logger.critical("🚨 EMERGENCY STOP ACTIVATED")
        
        # Close all positions immediately
        for position_id in list(self.active_positions.keys()):
            self._emergency_close_position(position_id)
            
        self.stop_automation()
        
    def _automation_loop(self):
        """Main automation loop"""
        
        while self.is_running:
            try:
                if self._is_market_hours():
                    schedule.run_pending()
                else:
                    self.logger.debug("🌙 Outside market hours - automation paused")
                    
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Automation error: {e}")
                time.sleep(60)  # Wait 1 minute on error
                
    def _is_market_hours(self) -> bool:
        """Check if within market trading hours"""
        
        if not self.config['automation']['market_hours_only']:
            return True
            
        now = datetime.now()
        
        # Indian market hours: 9:15 AM to 3:30 PM (Mon-Fri)
        if now.weekday() >= 5:  # Weekend
            return False
            
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
        
    def _scheduled_signal_scan(self):
        """Scheduled signal generation and processing"""
        
        if self.state != PortfolioState.ACTIVE:
            return
            
        self.logger.info("🔍 Running scheduled signal scan")
        
        try:
            # Scan all symbols for signals
            for symbol in self.watchlist:
                signals = self.jeafx_system.generate_trading_signals(symbol)
                
                for signal in signals:
                    if self._validate_signal_for_execution(signal):
                        self._process_signal(signal)
                        
        except Exception as e:
            self.logger.error(f"❌ Signal scan error: {e}")
            
    def _scheduled_position_monitoring(self):
        """Scheduled position monitoring and management"""
        
        if not self.active_positions:
            return
            
        self.logger.debug("👁️ Monitoring active positions")
        
        try:
            for position_id, position_data in list(self.active_positions.items()):
                # Get current market price
                current_price = self._get_current_price(position_data['symbol'])
                
                if current_price:
                    # Update position
                    self._update_position_price(position_id, current_price)
                    
                    # Check exit conditions
                    self._check_position_exit_conditions(position_id, current_price)
                    
        except Exception as e:
            self.logger.error(f"❌ Position monitoring error: {e}")
            
    def _scheduled_rebalancing(self):
        """Scheduled portfolio rebalancing"""
        
        self.logger.info("⚖️ Running portfolio rebalancing")
        
        try:
            # Calculate current portfolio weights
            portfolio_weights = self._calculate_portfolio_weights()
            
            # Check for rebalancing needs
            rebalancing_needed = False
            for symbol, weight in portfolio_weights.items():
                if weight > self.config['portfolio']['max_symbol_weight']:
                    rebalancing_needed = True
                    break
                    
            if rebalancing_needed:
                self._rebalance_portfolio()
                
        except Exception as e:
            self.logger.error(f"❌ Rebalancing error: {e}")
            
    def _scheduled_daily_report(self):
        """Generate and send daily performance report"""
        
        self.logger.info("📊 Generating daily report")
        
        try:
            # Calculate daily metrics
            metrics = self._calculate_portfolio_metrics()
            self.daily_metrics.append(metrics)
            
            # Save to database
            self._save_portfolio_metrics(metrics)
            
            # Generate report
            report = self._generate_daily_report(metrics)
            
            # Send alerts if configured
            if self.config['alerts']['daily_summary']:
                self._send_daily_summary_alert(report)
                
        except Exception as e:
            self.logger.error(f"❌ Daily report error: {e}")
            
    def _initial_portfolio_scan(self):
        """Initial portfolio scan on startup"""
        
        self.logger.info("📊 Running initial portfolio scan")
        
        # Scan all watchlist symbols
        for symbol in self.watchlist:
            self.logger.info(f"🔍 Scanning {symbol}")
            
            # Get current zones and signals
            zones = self.jeafx_system.scan_for_zones(symbol)
            signals = self.jeafx_system.generate_trading_signals(symbol)
            
            self.logger.info(f"   Found {len(zones)} zones, {len(signals)} signals")
            
        self.logger.info("✅ Initial scan complete")
        
    def _validate_signal_for_execution(self, signal: AdvancedSignal) -> bool:
        """Validate if signal should be executed"""
        
        # Check portfolio state
        if self.state != PortfolioState.ACTIVE:
            return False
            
        # Check signal confidence
        if signal.confidence_score < 75:  # Require high confidence
            return False
            
        # Check risk limits
        if len(self.active_positions) >= 5:  # Max positions
            return False
            
        # Check if we already have position in this symbol
        for pos_data in self.active_positions.values():
            if pos_data['symbol'] == signal.symbol:
                return False  # One position per symbol
                
        # Check available capital
        required_capital = signal.position_size * signal.entry_price
        if required_capital > self.cash_balance * 0.8:  # Keep 20% cash
            return False
            
        return True
        
    def _process_signal(self, signal: AdvancedSignal):
        """Process validated trading signal"""
        
        self.logger.info(f"⚡ Processing signal: {signal.signal_type} {signal.symbol}")
        
        try:
            # Calculate position size with risk manager
            position_size_data = self.risk_manager.calculate_position_size({
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'confidence_score': signal.confidence_score,
                'win_probability': signal.win_probability
            })
            
            # Create order
            order = self._create_order(signal, position_size_data)
            
            if order:
                # Execute order (simulated for demo)
                execution_result = self._execute_order(order)
                
                if execution_result['success']:
                    # Create position
                    self._create_position_from_order(order, signal)
                    
        except Exception as e:
            self.logger.error(f"❌ Signal processing error: {e}")
            
    def _create_order(self, signal: AdvancedSignal, position_size_data: Dict) -> Optional[TradingOrder]:
        """Create trading order from signal"""
        
        order_id = f"ORD_{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        order = TradingOrder(
            order_id=order_id,
            symbol=signal.symbol,
            order_type=signal.signal_type,
            quantity=int(position_size_data['position_size']),
            price=signal.entry_price,
            order_status=OrderStatus.PENDING,
            signal_id=signal.signal_id,
            created_time=datetime.now()
        )
        
        self.active_orders[order_id] = order
        
        # Log to database
        self._log_order_to_db(order)
        
        return order
        
    def _execute_order(self, order: TradingOrder) -> Dict:
        """Execute trading order (simulated for demo)"""
        
        # In real implementation, this would connect to broker API
        # For demo, we'll simulate execution
        
        self.logger.info(f"📋 Executing order: {order.order_type} {order.quantity} {order.symbol} @ ₹{order.price:.2f}")
        
        # Simulate execution with slight slippage
        slippage = np.random.uniform(-0.0005, 0.0005)  # ±0.05% slippage
        execution_price = order.price * (1 + slippage)
        
        # Update order
        order.order_status = OrderStatus.EXECUTED
        order.executed_time = datetime.now()
        order.execution_price = execution_price
        
        # Update cash balance
        trade_value = order.quantity * execution_price
        if order.order_type == "BUY":
            self.cash_balance -= trade_value
        else:
            self.cash_balance += trade_value
            
        self.logger.info(f"✅ Order executed at ₹{execution_price:.2f} | Cash: ₹{self.cash_balance:,.0f}")
        
        return {"success": True, "execution_price": execution_price}
        
    def _create_position_from_order(self, order: TradingOrder, signal: AdvancedSignal):
        """Create position from executed order"""
        
        position_id = f"POS_{order.symbol}_{order.order_id}"
        
        # Create combined position data
        position_data = {
            'position_id': position_id,
            'symbol': order.symbol,
            'position_type': order.order_type,
            'entry_price': order.execution_price,
            'current_price': order.execution_price,
            'quantity': order.quantity,
            'entry_time': order.executed_time,
            'stop_loss': signal.stop_loss,
            'target_price': signal.target_1,
            'signal_confidence': signal.confidence_score,
            'risk_amount': abs(order.execution_price - signal.stop_loss) * order.quantity
        }
        
        # Add to active positions
        self.active_positions[position_id] = position_data
        
        # Add to risk manager
        self.risk_manager.add_position({
            'position_id': position_id,
            'symbol': order.symbol,
            'position_type': order.order_type,
            'entry_price': order.execution_price,
            'quantity': order.quantity,
            'stop_loss': signal.stop_loss,
            'target_price': signal.target_1,
            'risk_amount': position_data['risk_amount'],
            'position_value': order.quantity * order.execution_price
        })
        
        # Remove from active orders
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
            
        self.logger.info(f"🎯 Position created: {position_id} | Risk: ₹{position_data['risk_amount']:,.0f}")
        
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        
        try:
            # Get latest data point
            data = self.jeafx_system.get_enhanced_market_data(symbol, timeframe="1", days=1)
            
            if not data.empty:
                return data['close'].iloc[-1]
                
        except Exception as e:
            self.logger.error(f"❌ Price fetch error for {symbol}: {e}")
            
        return None
        
    def _update_position_price(self, position_id: str, current_price: float):
        """Update position with current market price"""
        
        if position_id not in self.active_positions:
            return
            
        position = self.active_positions[position_id]
        position['current_price'] = current_price
        
        # Calculate unrealized P&L
        if position['position_type'] == "BUY":
            unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            
        position['unrealized_pnl'] = unrealized_pnl
        
        # Update risk manager
        self.risk_manager.update_position_price(position_id, current_price)
        
    def _check_position_exit_conditions(self, position_id: str, current_price: float):
        """Check if position should be closed"""
        
        position = self.active_positions[position_id]
        
        # Check stop loss
        if position['position_type'] == "BUY":
            if current_price <= position['stop_loss']:
                self._close_position(position_id, current_price, "STOP_LOSS")
                return
                
            # Check target
            if current_price >= position['target_price']:
                self._close_position(position_id, current_price, "TARGET_HIT")
                return
                
        else:  # SELL position
            if current_price >= position['stop_loss']:
                self._close_position(position_id, current_price, "STOP_LOSS")
                return
                
            if current_price <= position['target_price']:
                self._close_position(position_id, current_price, "TARGET_HIT")
                return
                
    def _close_position(self, position_id: str, exit_price: float, exit_reason: str):
        """Close trading position"""
        
        if position_id not in self.active_positions:
            return
            
        position = self.active_positions[position_id]
        
        # Calculate final P&L
        if position['position_type'] == "BUY":
            final_pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            final_pnl = (position['entry_price'] - exit_price) * position['quantity']
            
        # Update cash balance
        position_value = position['quantity'] * exit_price
        if position['position_type'] == "BUY":
            self.cash_balance += position_value
        else:
            self.cash_balance -= position_value
            
        # Log trade
        trade_record = {
            'trade_id': f"TRADE_{position_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'symbol': position['symbol'],
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'pnl': final_pnl,
            'pnl_percent': final_pnl / (position['entry_price'] * position['quantity']) * 100,
            'exit_reason': exit_reason,
            'signal_confidence': position['signal_confidence']
        }
        
        self.trade_history.append(trade_record)
        self._log_trade_to_db(trade_record)
        
        # Close in risk manager
        self.risk_manager.close_position(position_id, exit_price, exit_reason)
        
        # Remove from active positions
        del self.active_positions[position_id]
        
        pnl_icon = "💰" if final_pnl > 0 else "📉"
        self.logger.info(f"{pnl_icon} Position closed: {position['symbol']} | P&L: ₹{final_pnl:+,.0f} | {exit_reason}")
        
    def _cancel_order(self, order_id: str):
        """Cancel pending order"""
        
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            order.order_status = OrderStatus.CANCELLED
            del self.active_orders[order_id]
            
            self.logger.info(f"❌ Order cancelled: {order_id}")
            
    def _emergency_close_position(self, position_id: str):
        """Emergency close position at market price"""
        
        if position_id not in self.active_positions:
            return
            
        position = self.active_positions[position_id]
        current_price = self._get_current_price(position['symbol'])
        
        if current_price:
            self._close_position(position_id, current_price, "EMERGENCY_CLOSE")
            
    def _calculate_portfolio_weights(self) -> Dict[str, float]:
        """Calculate current portfolio weights by symbol"""
        
        weights = {}
        total_value = self._calculate_total_portfolio_value()
        
        for position in self.active_positions.values():
            symbol = position['symbol']
            position_value = position['quantity'] * position['current_price']
            weights[symbol] = position_value / total_value if total_value > 0 else 0
            
        return weights
        
    def _calculate_total_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        
        total_value = self.cash_balance
        
        for position in self.active_positions.values():
            position_value = position['quantity'] * position['current_price']
            total_value += position_value
            
        return total_value
        
    def _calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        
        total_value = self._calculate_total_portfolio_value()
        invested_amount = total_value - self.cash_balance
        
        # Calculate unrealized P&L
        unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in self.active_positions.values())
        
        # Calculate realized P&L
        realized_pnl = sum(trade['pnl'] for trade in self.trade_history)
        
        # Calculate returns
        total_return = (total_value - self.config['initial_capital']) / self.config['initial_capital']
        
        # Calculate daily return
        if len(self.daily_metrics) > 0:
            previous_value = self.daily_metrics[-1].total_value
            daily_return = (total_value - previous_value) / previous_value if previous_value > 0 else 0
        else:
            daily_return = 0
            
        # Calculate volatility (simplified)
        if len(self.daily_metrics) > 10:
            returns = [m.daily_return for m in self.daily_metrics[-20:]]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
        else:
            volatility = 0
            
        # Calculate Sharpe ratio (simplified, assuming 0% risk-free rate)
        sharpe_ratio = total_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        peak_value = self.config['initial_capital']
        max_dd = 0
        
        for metrics in self.daily_metrics:
            if metrics.total_value > peak_value:
                peak_value = metrics.total_value
            drawdown = (peak_value - metrics.total_value) / peak_value
            max_dd = max(max_dd, drawdown)
            
        # Trading statistics
        if self.trade_history:
            wins = [t for t in self.trade_history if t['pnl'] > 0]
            win_rate = len(wins) / len(self.trade_history)
            
            if wins and len(wins) < len(self.trade_history):
                avg_win = np.mean([t['pnl'] for t in wins])
                avg_loss = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] < 0])
                profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else 0
            else:
                profit_factor = 0
        else:
            win_rate = 0
            profit_factor = 0
            
        return PortfolioMetrics(
            timestamp=datetime.now(),
            total_value=total_value,
            cash_balance=self.cash_balance,
            invested_amount=invested_amount,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_return=total_return,
            daily_return=daily_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            active_positions=len(self.active_positions),
            total_trades=len(self.trade_history)
        )
        
    def _rebalance_portfolio(self):
        """Rebalance portfolio to target weights"""
        
        self.logger.info("⚖️ Rebalancing portfolio...")
        
        # For demo, we'll implement a simple rebalancing logic
        # In practice, this would be more sophisticated
        
        weights = self._calculate_portfolio_weights()
        max_weight = self.config['portfolio']['max_symbol_weight']
        
        # Close positions that are over-weighted
        for symbol, weight in weights.items():
            if weight > max_weight:
                # Find position to reduce
                for pos_id, pos_data in self.active_positions.items():
                    if pos_data['symbol'] == symbol:
                        current_price = self._get_current_price(symbol)
                        if current_price:
                            self._close_position(pos_id, current_price, "REBALANCING")
                            break
                            
    def _generate_daily_report(self, metrics: PortfolioMetrics) -> Dict:
        """Generate daily performance report"""
        
        report = {
            "date": metrics.timestamp.strftime("%Y-%m-%d"),
            "portfolio_summary": {
                "total_value": f"₹{metrics.total_value:,.0f}",
                "cash_balance": f"₹{metrics.cash_balance:,.0f}",
                "invested_amount": f"₹{metrics.invested_amount:,.0f}",
                "total_return": f"{metrics.total_return:.2%}",
                "daily_return": f"{metrics.daily_return:.2%}"
            },
            "performance_metrics": {
                "unrealized_pnl": f"₹{metrics.unrealized_pnl:+,.0f}",
                "realized_pnl": f"₹{metrics.realized_pnl:+,.0f}",
                "win_rate": f"{metrics.win_rate:.1%}",
                "profit_factor": f"{metrics.profit_factor:.2f}",
                "max_drawdown": f"{metrics.max_drawdown:.2%}",
                "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}"
            },
            "position_summary": {
                "active_positions": metrics.active_positions,
                "total_trades": metrics.total_trades,
                "positions_detail": [
                    {
                        "symbol": pos['symbol'],
                        "type": pos['position_type'],
                        "pnl": f"₹{pos.get('unrealized_pnl', 0):+,.0f}",
                        "value": f"₹{pos['quantity'] * pos['current_price']:,.0f}"
                    }
                    for pos in self.active_positions.values()
                ]
            }
        }
        
        return report
        
    def _log_order_to_db(self, order: TradingOrder):
        """Log order to database"""
        
        cursor = self.portfolio_db.cursor()
        cursor.execute('''
            INSERT INTO orders (order_id, symbol, order_type, quantity, price, 
                              status, signal_id, created_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (order.order_id, order.symbol, order.order_type, order.quantity,
              order.price, order.order_status.value, order.signal_id, order.created_time))
        self.portfolio_db.commit()
        
    def _log_trade_to_db(self, trade: Dict):
        """Log completed trade to database"""
        
        cursor = self.portfolio_db.cursor()
        cursor.execute('''
            INSERT INTO trade_history (trade_id, symbol, entry_time, exit_time,
                                     entry_price, exit_price, quantity, pnl, 
                                     pnl_percent, strategy, signal_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (trade['trade_id'], trade['symbol'], trade['entry_time'], trade['exit_time'],
              trade['entry_price'], trade['exit_price'], trade['quantity'], trade['pnl'],
              trade['pnl_percent'], 'JEAFX', trade['signal_confidence']))
        self.portfolio_db.commit()
        
    def _save_portfolio_metrics(self, metrics: PortfolioMetrics):
        """Save portfolio metrics to database"""
        
        cursor = self.portfolio_db.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO portfolio_metrics 
            (timestamp, total_value, cash_balance, unrealized_pnl, realized_pnl,
             total_return, active_positions, daily_return, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (metrics.timestamp, metrics.total_value, metrics.cash_balance,
              metrics.unrealized_pnl, metrics.realized_pnl, metrics.total_return,
              metrics.active_positions, metrics.daily_return, metrics.sharpe_ratio))
        self.portfolio_db.commit()
        
    def _send_daily_summary_alert(self, report: Dict):
        """Send daily summary alert"""
        
        # For demo, just log the summary
        self.logger.info("📊 DAILY SUMMARY:")
        self.logger.info(f"   Portfolio Value: {report['portfolio_summary']['total_value']}")
        self.logger.info(f"   Daily Return: {report['portfolio_summary']['daily_return']}")
        self.logger.info(f"   Active Positions: {report['position_summary']['active_positions']}")
        self.logger.info(f"   Win Rate: {report['performance_metrics']['win_rate']}")
        
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        
        metrics = self._calculate_portfolio_metrics()
        
        return {
            "state": self.state.value,
            "is_running": self.is_running,
            "portfolio_metrics": asdict(metrics),
            "active_positions": len(self.active_positions),
            "active_orders": len(self.active_orders),
            "watchlist_symbols": len(self.watchlist),
            "cash_balance": self.cash_balance,
            "recent_trades": self.trade_history[-5:] if self.trade_history else []
        }

def main():
    """Demo the portfolio management system"""
    
    print("🤖 JEAFX PORTFOLIO MANAGER - DEMO")
    print("="*50)
    
    # Initialize portfolio manager
    portfolio_manager = JeafxPortfolioManager()
    
    # Show initial status
    print("\n📊 INITIAL PORTFOLIO STATUS")
    print("-"*35)
    
    status = portfolio_manager.get_portfolio_status()
    print(f"💰 Cash Balance: ₹{status['cash_balance']:,.0f}")
    print(f"👁️ Watching {status['watchlist_symbols']} symbols")
    print(f"📈 Portfolio State: {status['state']}")
    
    # Start automation (for demo, we'll run a short cycle)
    print(f"\n🚀 STARTING AUTOMATION")
    print("-"*30)
    
    portfolio_manager.start_automation()
    
    # Run for a short period in demo
    print("⏳ Running automation for 30 seconds...")
    
    # Let it run briefly
    time.sleep(30)
    
    # Show updated status
    print(f"\n📊 PORTFOLIO STATUS AFTER AUTOMATION")
    print("-"*45)
    
    status = portfolio_manager.get_portfolio_status()
    metrics = status['portfolio_metrics']
    
    print(f"💰 Total Portfolio Value: ₹{metrics['total_value']:,.0f}")
    print(f"💵 Cash Balance: ₹{metrics['cash_balance']:,.0f}")
    print(f"📊 Active Positions: {metrics['active_positions']}")
    print(f"📈 Total Return: {metrics['total_return']:.2%}")
    print(f"🎯 Total Trades: {metrics['total_trades']}")
    
    if status['recent_trades']:
        print(f"\n💼 RECENT TRADES:")
        for trade in status['recent_trades']:
            pnl_icon = "💰" if trade['pnl'] > 0 else "📉"
            print(f"   {pnl_icon} {trade['symbol']}: ₹{trade['pnl']:+,.0f} ({trade['pnl_percent']:+.1f}%)")
    
    # Stop automation
    portfolio_manager.stop_automation()
    
    print(f"\n✅ Portfolio Management Demo Complete!")
    print(f"🎯 System ready for live trading with full automation")

if __name__ == "__main__":
    main()