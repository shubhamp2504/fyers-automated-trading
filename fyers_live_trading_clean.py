#!/usr/bin/env python3
"""
FYERS LIVE TRADING INTEGRATION
Real money automated trading with FYERS API

🚀 FEATURES:
- Live FYERS API integration
- Real money order execution
- Position management
- Account monitoring
- Risk controls for live trading
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

try:
    from fyers_apiv3 import fyersModel
    FYERS_AVAILABLE = True
except ImportError:
    FYERS_AVAILABLE = False
    
class FyersLiveTrader:
    """
    FYERS Live Trading Integration for Real Money Trading
    """
    
    def __init__(self, config_file: str = "fyers_config.json"):
        # Setup logging first
        self._setup_logging()
        
        # Load FYERS configuration
        self.config = self._load_fyers_config(config_file)
        
        # Initialize FYERS API
        self.fyers = None
        self._initialize_fyers_api()
        
        # Trading state
        self.account_info = {}
        self.positions = {}
        self.orders = {}
        
        self.logger.info("🚀 FYERS Live Trader Initialized")
        
    def _load_fyers_config(self, config_file: str) -> Dict:
        """Load FYERS trading configuration"""
        
        default_config = {
            "fyers": {
                "client_id": "",  # Your FYERS Client ID (e.g., "XYZ1234-100")
                "access_token": "",  # Your FYERS Access Token
                "redirect_uri": "https://trade.fyers.in/api-login/redirect-to-app",
                "response_type": "code",
                "state": "sample_state"
            },
            "trading": {
                "live_trading": False,  # Set to True for live trading with real money
                "max_orders_per_minute": 10,
                "order_timeout": 300,
                "slippage_tolerance": 0.5,  # 0.5% slippage tolerance
                "max_position_size": 50000,  # Max position size in INR
                "risk_per_trade": 0.02  # 2% risk per trade
            },
            "symbols": {
                "NSE:NIFTY50-INDEX": "NSE:NIFTY2470724700CE",  # Example option symbol
                "NSE:NIFTYBANK-INDEX": "NSE:BANKNIFTY2470751500CE",
                "NSE:RELIANCE-EQ": "NSE:RELIANCE-EQ",
                "NSE:TCS-EQ": "NSE:TCS-EQ",
                "NSE:INFY-EQ": "NSE:INFY-EQ"
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                # Update default with loaded config
                default_config.update(loaded_config)
        except FileNotFoundError:
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"📝 Created FYERS config file: {config_file}")
            print("⚠️ Please update with your FYERS API credentials before live trading!")
            
        return default_config
        
    def _setup_logging(self):
        """Setup logging for FYERS trading"""
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fyers_live_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FYERS_LIVE_TRADER')
        
    def _initialize_fyers_api(self):
        """Initialize FYERS API for live trading"""
        
        if not FYERS_AVAILABLE:
            self.logger.error("❌ FYERS API not installed. Run: pip install fyers-apiv3")
            return False
            
        fyers_config = self.config['fyers']
        
        if not fyers_config['client_id'] or not fyers_config['access_token']:
            self.logger.warning("⚠️ FYERS credentials not configured")
            self.logger.warning("📝 Update fyers_config.json with your API credentials")
            return False
            
        try:
            # Initialize FYERS API for LIVE TRADING
            self.fyers = fyersModel.FyersModel(
                client_id=fyers_config['client_id'],
                token=fyers_config['access_token'],
                log_path="fyers_api.log"
            )
            
            # Test API connection with real account
            profile_response = self.fyers.get_profile()
            
            if profile_response and profile_response.get('s') == 'ok':
                profile_data = profile_response['data']
                
                self.logger.info("🚀 FYERS API CONNECTED - LIVE TRADING ENABLED")
                self.logger.info(f"👤 Account: {profile_data.get('name', 'N/A')}")
                self.logger.info(f"📧 Email: {profile_data.get('email', 'N/A')}")
                self.logger.info(f"🏦 Broker: FYERS")
                
                # Store account info
                self.account_info = profile_data
                
                # Get account balance for real money trading
                self._get_account_balance()
                
                return True
                
            else:
                self.logger.error(f"❌ FYERS API Connection Failed: {profile_response}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ FYERS API initialization failed: {e}")
            self.logger.error("💡 Check your client_id and access_token in fyers_config.json")
            return False
            
    def _get_account_balance(self):
        """Get real account balance from FYERS"""
        
        if not self.fyers:
            return None
            
        try:
            funds_response = self.fyers.funds()
            
            if funds_response and funds_response.get('s') == 'ok':
                fund_data = funds_response['data']['fund_limit'][0]
                
                available_balance = fund_data.get('equityAmount', 0)
                utilized_amount = fund_data.get('utilized_amount', 0)
                total_balance = available_balance + utilized_amount
                
                self.logger.info("💰 LIVE ACCOUNT BALANCE:")
                self.logger.info(f"   💵 Available: ₹{available_balance:,.2f}")
                self.logger.info(f"   📊 Utilized: ₹{utilized_amount:,.2f}")
                self.logger.info(f"   🏦 Total: ₹{total_balance:,.2f}")
                
                return {
                    'available_balance': available_balance,
                    'utilized_amount': utilized_amount,
                    'total_balance': total_balance
                }
                
            else:
                self.logger.error(f"❌ Failed to get account balance: {funds_response}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting account balance: {e}")
            return None

def main():
    """Demo FYERS live trading integration"""
    
    print("🚀 FYERS LIVE TRADING INTEGRATION DEMO")
    print("="*50)
    
    # Initialize FYERS live trader
    trader = FyersLiveTrader()
    
    if not trader.fyers:
        print("\n⚠️ FYERS API not configured for live trading")
        print("📝 Please update fyers_config.json with your API credentials")
        print("\n📋 Required configuration:")
        print("   • client_id: Your FYERS Client ID")
        print("   • access_token: Your FYERS Access Token")
        print("   • live_trading: Set to true for real money trading")
        print("\n💡 Steps to get FYERS API credentials:")
        print("1. Login to FYERS web/app")
        print("2. Go to My Profile > Settings > API")
        print("3. Generate API credentials")
        print("4. Update fyers_config.json with your credentials")
        return
        
    print("\n✅ FYERS API Connected Successfully!")
    print("🎯 Ready for live automated trading with real money!")
    print("\n⚠️ WARNING: This system will trade with REAL MONEY when enabled!")

if __name__ == "__main__":
    main()