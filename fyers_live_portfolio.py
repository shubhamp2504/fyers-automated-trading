"""
FYERS Live Portfolio Manager for Real Money Automated Trading
Integrates JEAFX analysis with FYERS API for automated trading
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

# Import all required modules
from fyers_live_trading_clean import FyersLiveTrader
from jeafx_advanced_system import AdvancedJeafxSystem, AdvancedSignal
from jeafx_risk_manager import JeafxRiskManager
from jeafx_alert_system import JeafxAlertSystem, AlertLevel, send_trading_alert, send_risk_alert


class FyersLivePortfolioManager:
    """
    FYERS Live Portfolio Manager for Real Money Automated Trading
    """
    
    def __init__(self, config_file: str = "fyers_portfolio_config.json"):
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize all systems
        self.fyers_trader = FyersLiveTrader()
        self.jeafx_system = AdvancedJeafxSystem()
        self.risk_manager = JeafxRiskManager()
        self.alert_system = JeafxAlertSystem()
        
        # Trading state
        self.is_running = False
        self.live_positions = {}
        self.pending_orders = {}
        self.trade_history = []
        
        # Performance tracking
        self.session_stats = {
            'trades_executed': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0
        }
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("🚀 FYERS Live Portfolio Manager Initialized")
        
        # Check if ready for live trading
        self._validate_live_trading_setup()
        
    def _load_config(self, config_file: str) -> Dict:
        """Load portfolio configuration"""
        
        default_config = {
            "live_trading": {
                "enabled": False,  # CRITICAL: Set to True for real money trading
                "max_positions": 5,
                "max_risk_per_trade": 0.02,  # 2% per trade
                "max_portfolio_risk": 0.1,   # 10% total portfolio risk
                "min_signal_confidence": 75, # Minimum signal confidence
                "position_sizing_method": "FIXED_RISK"  # FIXED_RISK, VOLATILITY, KELLY
            },
            "symbols": {
                "watchlist": [
                    "NSE:RELIANCE-EQ",
                    "NSE:TCS-EQ", 
                    "NSE:INFY-EQ",
                    "NSE:HDFCBANK-EQ",
                    "NSE:ICICIBANK-EQ"
                ],
                "trading_enabled": {
                    "NSE:RELIANCE-EQ": True,
                    "NSE:TCS-EQ": True,
                    "NSE:INFY-EQ": True,
                    "NSE:HDFCBANK-EQ": True,
                    "NSE:ICICIBANK-EQ": True
                }
            },
            "automation": {
                "signal_scan_interval": 300,  # 5 minutes
                "position_monitor_interval": 60,  # 1 minute
                "market_hours_only": True,
                "auto_stop_loss": True,
                "auto_target_booking": True
            },
            "safety": {
                "max_orders_per_hour": 20,
                "emergency_stop_loss_percent": -5,  # -5% emergency stop
                "max_daily_loss": -10000,  # ₹10,000 max daily loss
                "require_manual_approval": False
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        except FileNotFoundError:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"📝 Created portfolio config: {config_file}")
            
        return default_config
        
    def _setup_logging(self):
        """Setup logging"""
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fyers_portfolio_live.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FYERS_LIVE_PORTFOLIO')
        
    def _validate_live_trading_setup(self):
        """Validate setup for live trading"""
        
        if not self.fyers_trader.fyers:
            self.logger.error("❌ FYERS API not connected - Live trading disabled")
            return False
            
        if not self.config['live_trading']['enabled']:
            self.logger.warning("⚠️ Live trading disabled in config")
            self.logger.warning("💡 Set live_trading.enabled: true to enable real money trading")
            return False
            
        # Get account balance
        account_info = self.fyers_trader._get_account_balance()
        
        if not account_info:
            self.logger.error("❌ Cannot get account balance - Live trading may not work")
            return False
            
        self.logger.info("✅ LIVE TRADING VALIDATION PASSED")
        self.logger.info(f"💰 Available Balance: ₹{account_info['available_balance']:,.2f}")
        
        return True
        
    def start_live_trading(self):
        """Start live automated trading with real money"""
        
        if self.is_running:
            self.logger.warning("⚠️ Live trading already running")
            return
            
        if not self.config['live_trading']['enabled']:
            self.logger.error("❌ Live trading disabled in configuration")
            self.logger.error("💡 Enable live_trading in config to trade with real money")
            return
            
        if not self.fyers_trader.fyers:
            self.logger.error("❌ FYERS API not connected")
            return
            
        self.logger.info("🚀 STARTING LIVE AUTOMATED TRADING")
        self.logger.info("💰 WARNING: TRADING WITH REAL MONEY!")
        
        self.is_running = True
        
        self.logger.info("✅ Live automated trading started successfully")
        
    def stop_live_trading(self):
        """Stop live trading"""
        
        self.is_running = False
        
        self.logger.info("🛑 STOPPING LIVE AUTOMATED TRADING")
        
    def get_live_portfolio_status(self) -> Dict:
        """Get current portfolio status with live data"""
        
        # Get live account info
        account_info = self.fyers_trader._get_account_balance()
        live_positions = self.fyers_trader.get_live_positions()
        
        total_pnl = sum(pos.get('pnl', 0) for pos in live_positions.values())
        total_unrealized = sum(pos.get('unrealized_pnl', 0) for pos in live_positions.values())
        
        return {
            'account_balance': account_info.get('available_balance', 0) if account_info else 0,
            'total_balance': account_info.get('total_balance', 0) if account_info else 0,
            'active_positions': len(live_positions),
            'total_pnl': total_pnl,
            'unrealized_pnl': total_unrealized,
            'realized_pnl': total_pnl - total_unrealized,
            'is_running': self.is_running,
            'market_open': self.fyers_trader.is_market_open(),
            'live_trading_enabled': self.config['live_trading']['enabled'],
            'session_stats': self.session_stats
        }

def main():
    """Demo FYERS Live Portfolio Manager"""
    
    print("🚀 FYERS LIVE PORTFOLIO MANAGER DEMO")
    print("="*50)
    
    try:
        # Initialize portfolio manager
        portfolio_manager = FyersLivePortfolioManager()
        
        if not portfolio_manager.fyers_trader.fyers:
            print("\n⚠️ FYERS API not connected")
            print("📝 Please configure FYERS API credentials first")
            return
            
        print("\n✅ FYERS Live Portfolio Manager Initialized")
        
        # Get current status
        status = portfolio_manager.get_live_portfolio_status()
        
        print("\n💰 LIVE ACCOUNT STATUS:")
        print(f"   📊 Available Balance: ₹{status['account_balance']:,.2f}")
        print(f"   🎯 Active Positions: {status['active_positions']}")
        print(f"   💵 Total P&L: ₹{status['total_pnl']:+,.2f}")
        print(f"   📈 Market Open: {'Yes' if status['market_open'] else 'No'}")
        print(f"   🚀 Live Trading: {'Enabled' if status['live_trading_enabled'] else 'Disabled'}")
        
        print("\n⚠️ WARNING: This system trades with REAL MONEY when enabled!")
        print("💡 Set live_trading.enabled: true in config to activate")
        print("\n✅ FYERS Live Portfolio Manager Ready!")
        
    except Exception as e:
        print(f"\n❌ Error initializing portfolio manager: {e}")
        print("💡 Please ensure all dependencies are installed and configured")

if __name__ == "__main__":
    main()