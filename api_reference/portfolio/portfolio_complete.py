"""
FYERS API v3 - Portfolio Reference Implementation
===============================================

Source: https://myapi.fyers.in/docsv3/#tag/Portfolio

All portfolio management API calls with proper implementation examples.
"""

from fyers_apiv3 import fyersModel
import json
from typing import Dict, Optional, List

class FyersPortfolio:
    """Complete portfolio reference implementation"""
    
    def __init__(self, client_id: str, access_token: str):
        self.fyers = fyersModel.FyersModel(client_id=client_id, token=access_token)
    
    def get_profile(self) -> Optional[Dict]:
        """
        Get user profile information
        API Doc: https://myapi.fyers.in/docsv3/#operation/profile
        """
        try:
            response = self.fyers.get_profile()
            
            if response['s'] == 'ok':
                profile = response['data']
                print(f"✅ Profile retrieved successfully")
                print(f"   📝 Name: {profile.get('name', 'Unknown')}")
                print(f"   📧 Email: {profile.get('email_id', 'Unknown')}")
                print(f"   📱 Mobile: {profile.get('mobile_number', 'Unknown')}")
                return profile
            else:
                print(f"❌ Error getting profile: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_profile: {str(e)}")
            return None
    
    def get_funds(self) -> Optional[Dict]:
        """
        Get fund details (available cash, margins, etc.)
        API Doc: https://myapi.fyers.in/docsv3/#operation/funds
        """
        try:
            response = self.fyers.funds()
            
            if response['s'] == 'ok':
                funds = response['fund_limit']
                print(f"✅ Funds retrieved successfully")
                print(f"   💰 Available Cash: ₹{funds.get('equityAmount', 0):,.2f}")
                print(f"   📊 Used Margin: ₹{funds.get('utilisedAmount', 0):,.2f}")
                print(f"   🔄 Available Margin: ₹{funds.get('availableAmount', 0):,.2f}")
                return funds
            else:
                print(f"❌ Error getting funds: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_funds: {str(e)}")
            return None
    
    def get_holdings(self) -> Optional[List[Dict]]:
        """
        Get current stock holdings (delivery positions)
        API Doc: https://myapi.fyers.in/docsv3/#operation/holdings
        """
        try:
            response = self.fyers.holdings()
            
            if response['s'] == 'ok':
                holdings = response.get('holdings', [])
                print(f"✅ Holdings retrieved: {len(holdings)} positions")
                
                total_current_value = 0
                total_investment = 0
                
                for holding in holdings:
                    symbol = holding.get('symbol', 'Unknown')
                    qty = holding.get('qty', 0)
                    avg_price = holding.get('costPrice', 0)
                    current_price = holding.get('ltp', 0)
                    current_value = qty * current_price
                    investment = qty * avg_price
                    pnl = current_value - investment
                    pnl_pct = (pnl / investment * 100) if investment > 0 else 0
                    
                    total_current_value += current_value
                    total_investment += investment
                    
                    print(f"   📈 {symbol}: {qty} @ ₹{avg_price:.2f} | Current: ₹{current_price:.2f} | P&L: ₹{pnl:.2f} ({pnl_pct:.2f}%)")
                
                total_pnl = total_current_value - total_investment
                total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
                
                print(f"   🎯 Total Investment: ₹{total_investment:,.2f}")
                print(f"   💎 Current Value: ₹{total_current_value:,.2f}")
                print(f"   {'🟢' if total_pnl >= 0 else '🔴'} Total P&L: ₹{total_pnl:,.2f} ({total_pnl_pct:.2f}%)")
                
                return holdings
            else:
                print(f"❌ Error getting holdings: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_holdings: {str(e)}")
            return None
    
    def get_positions(self) -> Optional[Dict]:
        """
        Get current trading positions (intraday + overnight)
        API Doc: https://myapi.fyers.in/docsv3/#operation/positions
        """
        try:
            response = self.fyers.positions()
            
            if response['s'] == 'ok':
                net_positions = response.get('netPositions', [])
                overall = response.get('overall', {})
                
                print(f"✅ Positions retrieved: {len(net_positions)} net positions")
                
                total_pnl = overall.get('pl_total', 0)
                print(f"   {'🟢' if total_pnl >= 0 else '🔴'} Overall P&L: ₹{total_pnl:,.2f}")
                
                for position in net_positions:
                    symbol = position.get('symbol', 'Unknown')
                    net_qty = position.get('netQty', 0)
                    avg_price = position.get('avgPrice', 0)
                    current_price = position.get('ltp', 0)
                    pnl = position.get('pl', 0)
                    
                    if net_qty != 0:
                        side = "LONG" if net_qty > 0 else "SHORT"
                        print(f"   📊 {symbol}: {side} {abs(net_qty)} @ ₹{avg_price:.2f} | LTP: ₹{current_price:.2f} | P&L: ₹{pnl:.2f}")
                
                return response
            else:
                print(f"❌ Error getting positions: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_positions: {str(e)}")
            return None
    
    def convert_position(self, symbol: str, qty: int, 
                        from_product: str, to_product: str) -> bool:
        """
        Convert position from one product type to another
        API Doc: https://myapi.fyers.in/docsv3/#operation/convertPosition
        
        Args:
            symbol: Trading symbol
            qty: Quantity to convert
            from_product: Source product type (INTRADAY, CNC, MARGIN)
            to_product: Target product type (INTRADAY, CNC, MARGIN)
        """
        try:
            data = {
                "symbol": symbol,
                "qty": qty,
                "fromProductType": from_product,
                "toProductType": to_product
            }
            
            response = self.fyers.convert_position(data=data)
            
            if response['s'] == 'ok':
                print(f"✅ Position converted: {qty} {symbol} from {from_product} to {to_product}")
                return True
            else:
                print(f"❌ Error converting position: {response.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Exception in convert_position: {str(e)}")
            return False
    
    def get_ledger(self) -> Optional[List[Dict]]:
        """
        Get ledger (transaction history)
        API Doc: https://myapi.fyers.in/docsv3/#operation/ledger
        """
        try:
            response = self.fyers.ledger()
            
            if response['s'] == 'ok':
                ledger = response.get('ledger', [])
                print(f"✅ Ledger retrieved: {len(ledger)} transactions")
                
                for transaction in ledger[:5]:  # Show first 5 transactions
                    date = transaction.get('date', 'Unknown')
                    particulars = transaction.get('particulars', 'Unknown')
                    amount = transaction.get('amount', 0)
                    balance = transaction.get('balance', 0)
                    
                    print(f"   💳 {date}: {particulars} | Amount: ₹{amount:,.2f} | Balance: ₹{balance:,.2f}")
                
                return ledger
            else:
                print(f"❌ Error getting ledger: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_ledger: {str(e)}")
            return None

def demo_portfolio_management():
    """Demonstrate all portfolio management APIs"""
    
    print("💼 FYERS API v3 - Portfolio Management Demo")
    print("=" * 60)
    
    # Load config
    try:
        with open("../config.json", "r") as f:
            config = json.load(f)
        
        client_id = config["client_id"]
        access_token = config["access_token"]
    except:
        print("❌ Please ensure config.json exists with credentials")
        return
    
    portfolio = FyersPortfolio(client_id, access_token)
    
    # 1. Get Profile
    print("\n📋 1. Testing Profile Information")
    profile = portfolio.get_profile()
    
    # 2. Get Funds
    print("\n📋 2. Testing Fund Details")
    funds = portfolio.get_funds()
    
    # 3. Get Holdings
    print("\n📋 3. Testing Holdings")
    holdings = portfolio.get_holdings()
    
    # 4. Get Positions
    print("\n📋 4. Testing Positions")
    positions = portfolio.get_positions()
    
    # 5. Get Ledger
    print("\n📋 5. Testing Ledger (Recent Transactions)")
    ledger = portfolio.get_ledger()
    
    # Portfolio Summary
    print("\n📊 PORTFOLIO SUMMARY")
    print("=" * 30)
    
    if funds:
        available = funds.get('availableAmount', 0)
        used = funds.get('utilisedAmount', 0)
        print(f"💰 Available Funds: ₹{available:,.2f}")
        print(f"📊 Used Margin: ₹{used:,.2f}")
    
    if holdings:
        print(f"📈 Holdings: {len(holdings)} stocks")
    
    if positions:
        net_positions = positions.get('netPositions', [])
        active_positions = [p for p in net_positions if p.get('netQty', 0) != 0]
        print(f"📊 Active Positions: {len(active_positions)}")
        
        overall_pnl = positions.get('overall', {}).get('pl_total', 0)
        print(f"{'🟢' if overall_pnl >= 0 else '🔴'} Today's P&L: ₹{overall_pnl:,.2f}")
    
    print("\n✅ Portfolio Management API demonstration completed!")

if __name__ == "__main__":
    demo_portfolio_management()