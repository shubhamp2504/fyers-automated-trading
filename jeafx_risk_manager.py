#!/usr/bin/env python3
"""
JEAFX RISK MANAGEMENT SYSTEM - Professional Risk Control
Advanced risk management for JEAFX trading system

🔒 FEATURES:
- Position sizing algorithms
- Dynamic stop loss management
- Portfolio heat mapping
- Drawdown protection
- Real-time risk monitoring
- Alert system for risk breaches
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

class RiskLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class Position:
    """Individual trading position"""
    position_id: str
    symbol: str
    position_type: PositionType
    entry_price: float
    current_price: float
    quantity: int
    entry_time: datetime
    stop_loss: float
    target_price: float
    unrealized_pnl: float
    risk_amount: float
    position_value: float
    
@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    total_capital: float
    total_risk_exposure: float
    current_drawdown: float
    max_drawdown: float
    portfolio_heat: float  # Percentage of capital at risk
    sharpe_ratio: float
    var_95: float  # Value at Risk (95% confidence)
    expected_shortfall: float
    risk_adjusted_return: float
    
@dataclass
class RiskAlert:
    """Risk management alert"""
    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: RiskLevel
    message: str
    recommended_action: str
    position_id: Optional[str] = None

class JeafxRiskManager:
    """
    Professional Risk Management System for JEAFX
    """
    
    def __init__(self, config_file: str = "risk_config.json"):
        self.config = self._load_risk_config(config_file)
        self.positions: Dict[str, Position] = {}
        self.risk_alerts: List[RiskAlert] = []
        
        # Risk tracking
        self.daily_pnl_history: List[float] = []
        self.peak_capital = self.config['initial_capital']
        self.current_capital = self.config['initial_capital']
        
        # Setup logging
        self.logger = logging.getLogger('JEAFX_RISK')
        
        # Database for risk logging
        self._setup_risk_database()
        
        self.logger.info("🔒 JEAFX Risk Management System Initialized")
        
    def _load_risk_config(self, config_file: str) -> Dict:
        """Load risk management configuration"""
        
        default_config = {
            "initial_capital": 100000,
            "risk_limits": {
                "max_risk_per_trade": 0.02,  # 2% per trade
                "max_portfolio_risk": 0.06,   # 6% total portfolio risk
                "max_positions": 5,           # Maximum open positions
                "max_single_position_size": 0.20,  # 20% of capital per position
                "max_correlation_exposure": 0.40,  # 40% in correlated positions
                "max_sector_exposure": 0.30,       # 30% in single sector
                "max_daily_loss": 0.05,           # 5% daily loss limit
                "max_drawdown": 0.15,             # 15% maximum drawdown
            },
            "position_sizing": {
                "method": "FIXED_FRACTIONAL",  # FIXED_FRACTIONAL, KELLY, VOLATILITY_ADJUSTED
                "base_risk_percent": 0.02,
                "kelly_multiplier": 0.25,      # Conservative Kelly
                "volatility_lookback": 20,
                "min_position_size": 1000,     # Minimum ₹1000 position
                "max_position_size": 50000     # Maximum ₹50000 position
            },
            "stop_loss": {
                "method": "ATR_BASED",  # ATR_BASED, PERCENTAGE, VOLATILITY_ADJUSTED
                "atr_multiplier": 2.0,
                "min_stop_percent": 0.01,  # Minimum 1% stop
                "max_stop_percent": 0.05,  # Maximum 5% stop
                "trailing_stop": True,
                "breakeven_threshold": 1.5  # Move to breakeven at 1.5R profit
            },
            "alerts": {
                "risk_breach_email": False,
                "risk_breach_sms": False,
                "console_alerts": True,
                "log_all_risks": True
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                default_config.update(loaded_config)
        except FileNotFoundError:
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.logger.info(f"📝 Created default risk config: {config_file}")
        
        return default_config
    
    def _setup_risk_database(self):
        """Setup risk management database"""
        self.risk_db = sqlite3.connect('jeafx_risk.db')
        cursor = self.risk_db.cursor()
        
        # Risk metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics (
                date DATE PRIMARY KEY,
                total_capital REAL,
                risk_exposure REAL,
                portfolio_heat REAL,
                current_drawdown REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                var_95 REAL
            )
        ''')
        
        # Risk alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                position_id TEXT
            )
        ''')
        
        # Position risk table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_risk (
                position_id TEXT,
                timestamp TIMESTAMP,
                symbol TEXT,
                risk_amount REAL,
                unrealized_pnl REAL,
                stop_distance REAL,
                heat_contribution REAL
            )
        ''')
        
        self.risk_db.commit()
        
    def calculate_position_size(self, signal_data: Dict) -> Dict:
        """Calculate optimal position size using configured method"""
        
        entry_price = signal_data['entry_price']
        stop_loss = signal_data['stop_loss']
        confidence = signal_data.get('confidence_score', 70)
        
        # Base risk amount
        base_risk = self.current_capital * self.config['risk_limits']['max_risk_per_trade']
        
        # Adjust risk based on confidence
        confidence_multiplier = min(confidence / 70, 1.5)  # Scale up to 1.5x for high confidence
        adjusted_risk = base_risk * confidence_multiplier
        
        # Risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit <= 0:
            return {"error": "Invalid stop loss level"}
        
        # Calculate position size based on method
        method = self.config['position_sizing']['method']
        
        if method == "FIXED_FRACTIONAL":
            position_size = adjusted_risk / risk_per_unit
            
        elif method == "KELLY":
            # Simplified Kelly criterion
            win_rate = signal_data.get('win_probability', 0.55)
            avg_win = signal_data.get('avg_win', 0.02)
            avg_loss = risk_per_unit / entry_price
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, kelly_fraction * self.config['position_sizing']['kelly_multiplier'])
            
            position_size = (self.current_capital * kelly_fraction) / entry_price
            
        elif method == "VOLATILITY_ADJUSTED":
            # Adjust position size based on volatility
            volatility = signal_data.get('volatility', 0.02)
            base_volatility = 0.02  # 2% base volatility
            
            volatility_adjustment = base_volatility / volatility
            adjusted_size = adjusted_risk * volatility_adjustment / risk_per_unit
            position_size = adjusted_size
            
        else:
            position_size = adjusted_risk / risk_per_unit
        
        # Apply position size limits
        min_size = self.config['position_sizing']['min_position_size'] / entry_price
        max_size = self.config['position_sizing']['max_position_size'] / entry_price
        
        position_size = max(min_size, min(position_size, max_size))
        
        # Check portfolio limits
        position_value = position_size * entry_price
        max_position_value = self.current_capital * self.config['risk_limits']['max_single_position_size']
        
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            
        return {
            "position_size": round(position_size, 2),
            "position_value": position_size * entry_price,
            "risk_amount": position_size * risk_per_unit,
            "risk_percent": (position_size * risk_per_unit) / self.current_capital * 100,
            "method_used": method
        }
    
    def add_position(self, position_data: Dict) -> bool:
        """Add new position with risk validation"""
        
        # Check position limits
        if len(self.positions) >= self.config['risk_limits']['max_positions']:
            self._create_alert("MAX_POSITIONS", RiskLevel.HIGH, 
                             f"Maximum positions ({self.config['risk_limits']['max_positions']}) reached")
            return False
        
        # Check portfolio heat
        new_risk = position_data['risk_amount']
        current_heat = self.calculate_portfolio_heat()
        new_heat = (current_heat * self.current_capital + new_risk) / self.current_capital
        
        if new_heat > self.config['risk_limits']['max_portfolio_risk']:
            self._create_alert("PORTFOLIO_HEAT", RiskLevel.HIGH,
                             f"New position would exceed portfolio heat limit ({new_heat:.1%} > {self.config['risk_limits']['max_portfolio_risk']:.1%})")
            return False
        
        # Create position
        position = Position(
            position_id=position_data['position_id'],
            symbol=position_data['symbol'],
            position_type=PositionType[position_data['position_type']],
            entry_price=position_data['entry_price'],
            current_price=position_data['entry_price'],  # Initial
            quantity=position_data['quantity'],
            entry_time=datetime.now(),
            stop_loss=position_data['stop_loss'],
            target_price=position_data['target_price'],
            unrealized_pnl=0.0,
            risk_amount=position_data['risk_amount'],
            position_value=position_data['position_value']
        )
        
        self.positions[position.position_id] = position
        
        # Log position
        self._log_position_risk(position)
        
        self.logger.info(f"✅ Position added: {position.symbol} | Risk: ₹{position.risk_amount:,.0f}")
        
        return True
    
    def update_position_price(self, position_id: str, current_price: float):
        """Update position with current market price"""
        
        if position_id not in self.positions:
            return
        
        position = self.positions[position_id]
        position.current_price = current_price
        
        # Calculate unrealized P&L
        if position.position_type == PositionType.LONG:
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
        
        # Check for stop loss breach
        if self._check_stop_loss_breach(position):
            self._create_alert("STOP_LOSS_BREACH", RiskLevel.CRITICAL,
                             f"Stop loss breached for {position.symbol} at ₹{current_price:.2f}",
                             position_id)
        
        # Update trailing stop if enabled
        if self.config['stop_loss']['trailing_stop']:
            self._update_trailing_stop(position)
        
        # Log updated position risk
        self._log_position_risk(position)
    
    def _check_stop_loss_breach(self, position: Position) -> bool:
        """Check if stop loss has been breached"""
        
        if position.position_type == PositionType.LONG:
            return position.current_price <= position.stop_loss
        else:
            return position.current_price >= position.stop_loss
    
    def _update_trailing_stop(self, position: Position):
        """Update trailing stop loss"""
        
        profit_multiple = abs(position.unrealized_pnl / position.risk_amount)
        breakeven_threshold = self.config['stop_loss']['breakeven_threshold']
        
        if profit_multiple >= breakeven_threshold:
            # Move stop to breakeven or better
            if position.position_type == PositionType.LONG:
                new_stop = max(position.stop_loss, position.entry_price)
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    self.logger.info(f"📈 Trailing stop updated for {position.symbol}: ₹{new_stop:.2f}")
            else:
                new_stop = min(position.stop_loss, position.entry_price)
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    self.logger.info(f"📉 Trailing stop updated for {position.symbol}: ₹{new_stop:.2f}")
    
    def close_position(self, position_id: str, exit_price: float, exit_reason: str) -> Dict:
        """Close position and calculate final P&L"""
        
        if position_id not in self.positions:
            return {"error": "Position not found"}
        
        position = self.positions[position_id]
        
        # Calculate final P&L
        if position.position_type == PositionType.LONG:
            final_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            final_pnl = (position.entry_price - exit_price) * position.quantity
        
        # Update capital
        self.current_capital += final_pnl
        
        # Update peak capital and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Record daily P&L
        self.daily_pnl_history.append(final_pnl)
        
        # Remove position
        del self.positions[position_id]
        
        self.logger.info(f"🏁 Position closed: {position.symbol} | P&L: ₹{final_pnl:+,.0f} | Reason: {exit_reason}")
        
        # Check for daily loss limit breach
        daily_pnl = sum(self.daily_pnl_history)
        daily_loss_limit = self.config['initial_capital'] * self.config['risk_limits']['max_daily_loss']
        
        if daily_pnl < -daily_loss_limit:
            self._create_alert("DAILY_LOSS_LIMIT", RiskLevel.CRITICAL,
                             f"Daily loss limit breached: ₹{daily_pnl:+,.0f}")
        
        return {
            "final_pnl": final_pnl,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "position_duration": datetime.now() - position.entry_time
        }
    
    def calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (percentage of capital at risk)"""
        
        total_risk = sum(pos.risk_amount for pos in self.positions.values())
        return total_risk / self.current_capital if self.current_capital > 0 else 0
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        # Current metrics
        total_risk_exposure = sum(abs(pos.unrealized_pnl) for pos in self.positions.values())
        current_drawdown = max(0, (self.peak_capital - self.current_capital) / self.peak_capital)
        portfolio_heat = self.calculate_portfolio_heat()
        
        # Calculate max drawdown
        running_peak = self.config['initial_capital']
        max_dd = 0
        running_capital = self.config['initial_capital']
        
        for pnl in self.daily_pnl_history:
            running_capital += pnl
            if running_capital > running_peak:
                running_peak = running_capital
            drawdown = (running_peak - running_capital) / running_peak
            max_dd = max(max_dd, drawdown)
        
        # Sharpe ratio (simplified)
        if len(self.daily_pnl_history) > 5:
            returns = pd.Series(self.daily_pnl_history) / self.config['initial_capital']
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # VaR calculation (simplified)
        if len(self.daily_pnl_history) > 10:
            var_95 = np.percentile(self.daily_pnl_history, 5)  # 5th percentile
            expected_shortfall = np.mean([pnl for pnl in self.daily_pnl_history if pnl <= var_95])
        else:
            var_95 = 0
            expected_shortfall = 0
        
        # Risk-adjusted return
        total_return = (self.current_capital - self.config['initial_capital']) / self.config['initial_capital']
        risk_adjusted_return = total_return / max(max_dd, 0.01)  # Avoid division by zero
        
        return RiskMetrics(
            total_capital=self.current_capital,
            total_risk_exposure=total_risk_exposure,
            current_drawdown=current_drawdown,
            max_drawdown=max_dd,
            portfolio_heat=portfolio_heat,
            sharpe_ratio=sharpe_ratio,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            risk_adjusted_return=risk_adjusted_return
        )
    
    def _create_alert(self, alert_type: str, severity: RiskLevel, message: str, position_id: Optional[str] = None):
        """Create risk management alert"""
        
        alert = RiskAlert(
            alert_id=f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            recommended_action=self._get_recommended_action(alert_type, severity),
            position_id=position_id
        )
        
        self.risk_alerts.append(alert)
        
        # Log alert to database
        cursor = self.risk_db.cursor()
        cursor.execute('''
            INSERT INTO risk_alerts (alert_id, timestamp, alert_type, severity, message, position_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (alert.alert_id, alert.timestamp, alert.alert_type, alert.severity.value, 
              alert.message, alert.position_id))
        self.risk_db.commit()
        
        # Console alert
        if self.config['alerts']['console_alerts']:
            severity_icon = {"LOW": "ℹ️", "MODERATE": "⚠️", "HIGH": "🚨", "CRITICAL": "🔥"}
            icon = severity_icon.get(severity.value, "⚠️")
            self.logger.warning(f"{icon} RISK ALERT ({severity.value}): {message}")
    
    def _get_recommended_action(self, alert_type: str, severity: RiskLevel) -> str:
        """Get recommended action for alert type"""
        
        actions = {
            "MAX_POSITIONS": "Consider closing weakest position before opening new ones",
            "PORTFOLIO_HEAT": "Reduce position sizes or close positions to lower portfolio heat",
            "STOP_LOSS_BREACH": "Close position immediately",
            "DAILY_LOSS_LIMIT": "Stop trading for today and review strategy",
            "DRAWDOWN_LIMIT": "Reduce risk per trade and review trading approach",
            "CORRELATION_RISK": "Close correlated positions to reduce portfolio concentration"
        }
        
        return actions.get(alert_type, "Review position and take appropriate action")
    
    def _log_position_risk(self, position: Position):
        """Log position risk to database"""
        
        cursor = self.risk_db.cursor()
        cursor.execute('''
            INSERT INTO position_risk (position_id, timestamp, symbol, risk_amount, 
                                     unrealized_pnl, stop_distance, heat_contribution)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            position.position_id,
            datetime.now(),
            position.symbol,
            position.risk_amount,
            position.unrealized_pnl,
            abs(position.current_price - position.stop_loss),
            position.risk_amount / self.current_capital
        ))
        self.risk_db.commit()
    
    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        
        metrics = self.calculate_risk_metrics()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "capital_status": {
                "current_capital": metrics.total_capital,
                "initial_capital": self.config['initial_capital'],
                "total_return": (metrics.total_capital - self.config['initial_capital']) / self.config['initial_capital'],
                "peak_capital": self.peak_capital
            },
            "risk_metrics": {
                "portfolio_heat": metrics.portfolio_heat,
                "current_drawdown": metrics.current_drawdown,
                "max_drawdown": metrics.max_drawdown,
                "risk_exposure": metrics.total_risk_exposure,
                "var_95": metrics.var_95,
                "sharpe_ratio": metrics.sharpe_ratio
            },
            "positions": {
                "total_positions": len(self.positions),
                "position_details": [
                    {
                        "symbol": pos.symbol,
                        "type": pos.position_type.value,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "risk_amount": pos.risk_amount,
                        "current_price": pos.current_price,
                        "stop_loss": pos.stop_loss
                    }
                    for pos in self.positions.values()
                ]
            },
            "recent_alerts": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "type": alert.alert_type,
                    "severity": alert.severity.value,
                    "message": alert.message
                }
                for alert in self.risk_alerts[-10:]  # Last 10 alerts
            ],
            "risk_limits": {
                "max_portfolio_heat": self.config['risk_limits']['max_portfolio_risk'],
                "max_positions": self.config['risk_limits']['max_positions'],
                "max_daily_loss": self.config['risk_limits']['max_daily_loss'],
                "max_drawdown": self.config['risk_limits']['max_drawdown']
            }
        }
        
        return report
    
    def export_risk_report(self, filename: Optional[str] = None) -> str:
        """Export risk report to file"""
        
        if filename is None:
            filename = f"jeafx_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.generate_risk_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📊 Risk report exported: {filename}")
        return filename

def main():
    """Demonstrate risk management system"""
    
    print("🔒 JEAFX RISK MANAGEMENT SYSTEM - DEMO")
    print("="*50)
    
    # Initialize risk manager
    risk_manager = JeafxRiskManager()
    
    # Demo position sizing
    print("\n📊 POSITION SIZING DEMO")
    print("-"*30)
    
    signal_data = {
        'entry_price': 21500,
        'stop_loss': 21400,
        'confidence_score': 85,
        'win_probability': 0.65,
        'volatility': 0.018
    }
    
    position_size = risk_manager.calculate_position_size(signal_data)
    print(f"Signal: BUY @ ₹{signal_data['entry_price']}, Stop @ ₹{signal_data['stop_loss']}")
    print(f"Confidence: {signal_data['confidence_score']}%")
    print(f"Recommended Position Size: ₹{position_size['position_value']:,.0f}")
    print(f"Risk Amount: ₹{position_size['risk_amount']:,.0f} ({position_size['risk_percent']:.1f}%)")
    print(f"Method: {position_size['method_used']}")
    
    # Demo position management
    print("\n🎯 POSITION MANAGEMENT DEMO")
    print("-"*35)
    
    # Add demo position
    position_data = {
        'position_id': 'DEMO_001',
        'symbol': 'NSE:NIFTY50-INDEX',
        'position_type': 'LONG',
        'entry_price': 21500,
        'quantity': 25,
        'stop_loss': 21400,
        'target_price': 21700,
        'risk_amount': position_size['risk_amount'],
        'position_value': position_size['position_value']
    }
    
    success = risk_manager.add_position(position_data)
    print(f"Position Added: {'✅ Success' if success else '❌ Failed'}")
    
    # Update position price
    risk_manager.update_position_price('DEMO_001', 21550)
    print(f"Position Updated: Price moved to ₹21,550")
    
    # Calculate risk metrics
    metrics = risk_manager.calculate_risk_metrics()
    print(f"\n📈 RISK METRICS:")
    print(f"Portfolio Heat: {metrics.portfolio_heat:.1%}")
    print(f"Current Drawdown: {metrics.current_drawdown:.1%}")
    print(f"Unrealized P&L: ₹{sum(pos.unrealized_pnl for pos in risk_manager.positions.values()):+,.0f}")
    
    # Generate and export report
    print(f"\n📊 RISK REPORT GENERATION")
    print("-"*30)
    
    report_file = risk_manager.export_risk_report()
    print(f"Risk Report: {report_file}")
    
    # Show recent alerts
    if risk_manager.risk_alerts:
        print(f"\n🚨 RECENT ALERTS:")
        for alert in risk_manager.risk_alerts[-3:]:
            severity_icon = {"LOW": "ℹ️", "MODERATE": "⚠️", "HIGH": "🚨", "CRITICAL": "🔥"}
            icon = severity_icon.get(alert.severity.value, "⚠️")
            print(f"   {icon} {alert.alert_type}: {alert.message}")
    
    print(f"\n✅ Risk Management Demo Complete!")

if __name__ == "__main__":
    main()