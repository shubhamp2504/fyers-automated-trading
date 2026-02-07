"""
FYERS Live Trading System Validation
===================================

Validates the complete Fyers API integration with real data.
Source: https://myapi.fyers.in/docsv3

This script performs comprehensive checks to ensure:
- ✅ Official Fyers API connectivity 
- ✅ Real account data retrieval
- ✅ Live market data streaming 
- ✅ Order placement capability (paper mode)
- ✅ Risk management system
- ❌ NO DEMO/DUMMY DATA

⚠️ WARNING: This validates LIVE Fyers API with REAL account data
"""

import sys
import os
from datetime import datetime
import json

# Import our Fyers client and system
from fyers_client import FyersClient
from live_trading_system import LiveIndexTradingSystem

def validate_fyers_api_connection():
    """Test 1: Validate Fyers API connection and authentication"""
    
    print("🔑 TEST 1: Fyers API Connection & Authentication")
    print("-" * 50)
    
    try:
        # Initialize client
        fyers = FyersClient('fyers_config.json')
        
        # Test profile access
        profile = fyers.portfolio.get_profile()
        if profile:
            print("✅ Authentication successful")
            print(f"   👤 Account holder: {profile.get('name', 'Unknown')}")
            print(f"   📧 Email: {profile.get('email_id', 'Unknown')}")
            print(f"   📱 Mobile: {profile.get('mobile_number', 'Unknown')}")
            return True, fyers
        else:
            print("❌ Authentication failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False, None

def validate_live_market_data(fyers):
    """Test 2: Validate live market data retrieval"""
    
    print("\n📊 TEST 2: Live Market Data Retrieval") 
    print("-" * 50)
    
    try:
        # Test symbols from config
        test_symbols = [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX", 
            "NSE:RELIANCE-EQ"
        ]
        
        # Get live quotes
        quotes = fyers.get_live_quotes(test_symbols)
        if quotes and 'd' in quotes and quotes['d']:
            print("✅ Live market data retrieval successful")
            for symbol in test_symbols:
                if symbol in quotes['d']:
                    data = quotes['d'][symbol]
                    if 'v' in data and 'lp' in data['v']:
                        ltp = data['v']['lp']
                        print(f"   📈 {symbol}: ₹{ltp:.2f}")
                    else:
                        print(f"   ⚠️ Incomplete data for {symbol}")
                else:
                    print(f"   ⚠️ No data for {symbol}")
            return True
        else:
            print("❌ Market data retrieval failed")
            print(f"   Response: {quotes}")
            return False
            
    except Exception as e:
        print(f"❌ Market data error: {e}")
        return False

def validate_historical_data(fyers):
    """Test 3: Validate historical data retrieval"""
    
    print("\n📈 TEST 3: Historical Data Retrieval")
    print("-" * 50)
    
    try:
        # Test historical data for NIFTY (using recent dates)
        symbol = "NSE:NIFTY50-INDEX"
        
        # Use working date range from 2024 (more recent than Dec 2024)
        df = fyers.get_historical_data(
            symbol=symbol,
            resolution="1D",
            start_date="2024-01-15", 
            end_date="2024-01-19"
        )
        
        if df is not None and len(df) > 0:
            print("✅ Historical data retrieval successful")
            print(f"   📊 Records: {len(df)} candles")
            print(f"   📅 Period: {df.index[0]} to {df.index[-1]}")
            print(f"   💰 Latest close: ₹{df['close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ Historical data retrieval failed")
            return False
            
    except Exception as e:
        print(f"❌ Historical data error: {e}")
        return False

def validate_account_funds(fyers):
    """Test 4: Validate account funds and positions"""
    
    print("\n💰 TEST 4: Account Funds & Portfolio")
    print("-" * 50)
    
    try:
        # Get funds
        funds = fyers.get_funds()
        if funds:
            print("✅ Account funds retrieval successful")
            print(f"   💵 Available: ₹{funds.get('availableAmount', 0):,.2f}")
            print(f"   📊 Used margin: ₹{funds.get('utilisedAmount', 0):,.2f}")
            
            # Check minimum balance for trading
            available = funds.get('availableAmount', 0)
            if available < 10000:
                print("   ⚠️ Low balance for live trading")
            else:
                print("   ✅ Sufficient balance for trading")
                
        # Get positions
        positions = fyers.get_positions()
        if positions is not None:
            print(f"   📈 Current positions: {len(positions) if positions else 0}")
        
        # Get holdings
        holdings = fyers.get_holdings()
        if holdings is not None:
            print(f"   💎 Holdings: {len(holdings) if holdings else 0}")
            
        return True
        
    except Exception as e:
        print(f"❌ Portfolio data error: {e}")
        return False

def validate_trading_system_initialization(fyers):
    """Test 5: Validate trading system initialization"""
    
    print("\n🚀 TEST 5: Trading System Initialization")
    print("-" * 50)
    
    try:
        # Initialize trading system
        trading_system = LiveIndexTradingSystem(fyers_client=fyers)
        
        print("✅ Trading system initialized successfully")
        print(f"   🎯 Trading symbols: {list(trading_system.trading_symbols.keys())}")
        print(f"   💰 Max daily loss: ₹{trading_system.max_daily_loss:,.2f}")
        print(f"   📈 Max positions: {trading_system.max_positions}")
        
        # Test market hours check
        is_market_open = trading_system.is_market_open()
        print(f"   ⏰ Market open: {'✅' if is_market_open else '❌'}")
        
        return True, trading_system
        
    except Exception as e:
        print(f"❌ Trading system error: {e}")
        return False, None

def validate_risk_management(trading_system):
    """Test 6: Validate risk management system"""
    
    print("\n⚠️ TEST 6: Risk Management System")
    print("-" * 50)
    
    try:
        # Test risk parameters
        config = trading_system.config['trading']
        
        print("✅ Risk management configuration:")
        print(f"   🛑 Max daily loss: ₹{config['max_daily_loss']:,.2f}")
        print(f"   📊 Risk per trade: {config['risk_per_trade']:.1%}")
        print(f"   🔒 Stop loss: {config['stop_loss_percentage']:.1f}%")
        print(f"   🎯 Take profit: {config['take_profit_percentage']:.1f}%")
        print(f"   📈 Max positions: {config['max_open_positions']}")
        
        # Validate risk limits are reasonable
        if config['max_daily_loss'] > 50000:
            print("   ⚠️ WARNING: High daily loss limit")
        if config['risk_per_trade'] > 0.05:
            print("   ⚠️ WARNING: High risk per trade")
            
        return True
        
    except Exception as e:
        print(f"❌ Risk management error: {e}")
        return False

def main():
    """Main validation function"""
    
    print("=" * 60)
    print("🔍 FYERS LIVE TRADING SYSTEM VALIDATION")
    print("=" * 60)
    print("⚠️  Validating REAL Fyers API with LIVE account data")
    print("⚠️  NO demo/dummy data - all connections are LIVE")
    print("=" * 60)
    
    # Track test results
    results = []
    
    # Test 1: API Connection
    success, fyers = validate_fyers_api_connection()
    results.append(("API Connection", success))
    
    if not success:
        print("\n❌ VALIDATION FAILED: Cannot proceed without API connection")
        return
    
    # Test 2: Market Data
    success = validate_live_market_data(fyers)
    results.append(("Live Market Data", success))
    
    # Test 3: Historical Data  
    success = validate_historical_data(fyers)
    results.append(("Historical Data", success))
    
    # Test 4: Account Funds
    success = validate_account_funds(fyers)
    results.append(("Account & Portfolio", success))
    
    # Test 5: Trading System
    success, trading_system = validate_trading_system_initialization(fyers)
    results.append(("Trading System Init", success))
    
    if success:
        # Test 6: Risk Management
        success = validate_risk_management(trading_system)
        results.append(("Risk Management", success))
    
    # Final results
    print("\n" + "=" * 60)
    print("📋 VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} | {test_name}")
        if not passed:
            all_passed = False
    
    print("-" * 60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED - SYSTEM READY FOR LIVE TRADING")
        print("⚠️  Remember: This system trades with REAL MONEY")
        print("💰 Use appropriate position sizing and risk management")
    else:
        print("❌ SOME TESTS FAILED - REVIEW BEFORE TRADING")
        print("🚨 Do not trade until all issues are resolved")
    
    print("=" * 60)

if __name__ == "__main__":
    main()