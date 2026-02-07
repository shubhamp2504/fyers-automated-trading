"""
FYERS API v3 - Orders Reference Implementation
============================================

Source: https://myapi.fyers.in/docsv3/#tag/Orders

All order management API calls with proper implementation examples.
"""

from fyers_apiv3 import fyersModel
import json
from typing import Dict, Optional, List
from datetime import datetime

class FyersOrders:
    """Complete orders reference implementation"""
    
    def __init__(self, client_id: str, access_token: str):
        self.fyers = fyersModel.FyersModel(client_id=client_id, token=access_token)
    
    def place_order(self, symbol: str, qty: int, side: int, type: int,
                   product_type: str = "CNC", limit_price: float = 0,
                   stop_price: float = 0, disclosed_qty: int = 0,
                   validity: str = "DAY", offline_order: str = "False") -> Optional[Dict]:
        """
        Place a new order
        API Doc: https://myapi.fyers.in/docsv3/#operation/placeOrder
        
        Args:
            symbol: Trading symbol (e.g., "NSE:RELIANCE-EQ")
            qty: Quantity to trade
            side: 1 for BUY, -1 for SELL
            type: 1=MARKET, 2=LIMIT, 3=STOP_MARKET, 4=STOP_LIMIT
            product_type: "CNC", "INTRADAY", "MARGIN", "CO", "BO"
            limit_price: Limit price for LIMIT orders
            stop_price: Stop price for STOP orders
            disclosed_qty: Disclosed quantity for iceberg orders
            validity: "DAY", "IOC", "GTD"
            offline_order: "True" or "False"
        """
        try:
            data = {
                "symbol": symbol,
                "qty": qty,
                "type": type,
                "side": side,
                "productType": product_type,
                "limitPrice": limit_price,
                "stopPrice": stop_price,
                "validity": validity,
                "disclosedQty": disclosed_qty,
                "offlineOrder": offline_order
            }
            
            response = self.fyers.place_order(data=data)
            
            if response['s'] == 'ok':
                order_id = response.get('id', 'Unknown')
                print(f"✅ Order placed successfully: {order_id}")
                return response
            else:
                print(f"❌ Error placing order: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in place_order: {str(e)}")
            return None
    
    def modify_order(self, order_id: str, qty: Optional[int] = None,
                    limit_price: Optional[float] = None, 
                    stop_price: Optional[float] = None,
                    type: Optional[int] = None) -> Optional[Dict]:
        """
        Modify an existing order
        API Doc: https://myapi.fyers.in/docsv3/#operation/modifyOrder
        
        Args:
            order_id: Order ID to modify
            qty: New quantity (optional)
            limit_price: New limit price (optional)
            stop_price: New stop price (optional)
            type: New order type (optional)
        """
        try:
            data = {"id": order_id}
            
            # Add only provided parameters
            if qty is not None:
                data["qty"] = qty
            if limit_price is not None:
                data["limitPrice"] = limit_price
            if stop_price is not None:
                data["stopPrice"] = stop_price
            if type is not None:
                data["type"] = type
            
            response = self.fyers.modify_order(data=data)
            
            if response['s'] == 'ok':
                print(f"✅ Order modified successfully: {order_id}")
                return response
            else:
                print(f"❌ Error modifying order: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in modify_order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order
        API Doc: https://myapi.fyers.in/docsv3/#operation/cancelOrder
        
        Args:
            order_id: Order ID to cancel
        """
        try:
            data = {"id": order_id}
            
            response = self.fyers.cancel_order(data=data)
            
            if response['s'] == 'ok':
                print(f"✅ Order cancelled successfully: {order_id}")
                return True
            else:
                print(f"❌ Error cancelling order: {response.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Exception in cancel_order: {str(e)}")
            return False
    
    def get_orderbook(self) -> Optional[List[Dict]]:
        """
        Get all orders (orderbook)
        API Doc: https://myapi.fyers.in/docsv3/#operation/orderbook
        """
        try:
            response = self.fyers.orderbook()
            
            if response['s'] == 'ok':
                orders = response.get('orderBook', [])
                print(f"✅ Orderbook retrieved: {len(orders)} orders")
                return orders
            else:
                print(f"❌ Error getting orderbook: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_orderbook: {str(e)}")
            return None
    
    def get_tradebook(self) -> Optional[List[Dict]]:
        """
        Get all executed trades (tradebook)
        API Doc: https://myapi.fyers.in/docsv3/#operation/tradebook
        """
        try:
            response = self.fyers.tradebook()
            
            if response['s'] == 'ok':
                trades = response.get('tradeBook', [])
                print(f"✅ Tradebook retrieved: {len(trades)} trades")
                return trades
            else:
                print(f"❌ Error getting tradebook: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_tradebook: {str(e)}")
            return None
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Get status of a specific order
        API Doc: https://myapi.fyers.in/docsv3/#operation/orderStatus
        
        Args:
            order_id: Order ID to check
        """
        try:
            data = {"id": order_id}
            
            response = self.fyers.order_status(data=data)
            
            if response['s'] == 'ok':
                print(f"✅ Order status retrieved for {order_id}")
                return response['d']
            else:
                print(f"❌ Error getting order status: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_order_status: {str(e)}")
            return None
    
    def place_bracket_order(self, symbol: str, qty: int, side: int,
                          entry_price: float, stop_loss: float, 
                          target: float, trailing_stop_loss: int = 0) -> Optional[Dict]:
        """
        Place bracket order (BO)
        API Doc: https://myapi.fyers.in/docsv3/#operation/placeBracketOrder
        
        Args:
            symbol: Trading symbol
            qty: Quantity
            side: 1 for BUY, -1 for SELL
            entry_price: Entry price
            stop_loss: Stop loss price
            target: Target price
            trailing_stop_loss: Trailing stop loss points
        """
        try:
            data = {
                "symbol": symbol,
                "qty": qty,
                "type": 2,  # LIMIT order for entry
                "side": side,
                "productType": "BO",
                "limitPrice": entry_price,
                "stopPrice": 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": "False",
                "stopLoss": stop_loss,
                "takeProfit": target,
                "trailingStopLoss": trailing_stop_loss
            }
            
            response = self.fyers.place_order(data=data)
            
            if response['s'] == 'ok':
                order_id = response.get('id', 'Unknown')
                print(f"✅ Bracket order placed successfully: {order_id}")
                return response
            else:
                print(f"❌ Error placing bracket order: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in place_bracket_order: {str(e)}")
            return None
    
    def place_cover_order(self, symbol: str, qty: int, side: int,
                         entry_price: float, trigger_price: float) -> Optional[Dict]:
        """
        Place cover order (CO)
        API Doc: https://myapi.fyers.in/docsv3/#operation/placeCoverOrder
        
        Args:
            symbol: Trading symbol
            qty: Quantity
            side: 1 for BUY, -1 for SELL
            entry_price: Entry price
            trigger_price: Stop loss trigger price
        """
        try:
            data = {
                "symbol": symbol,
                "qty": qty,
                "type": 2,  # LIMIT order for entry
                "side": side,
                "productType": "CO",
                "limitPrice": entry_price,
                "stopPrice": trigger_price,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": "False"
            }
            
            response = self.fyers.place_order(data=data)
            
            if response['s'] == 'ok':
                order_id = response.get('id', 'Unknown')
                print(f"✅ Cover order placed successfully: {order_id}")
                return response
            else:
                print(f"❌ Error placing cover order: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception in place_cover_order: {str(e)}")
            return None

def demo_order_management():
    """Demonstrate all order management APIs"""
    
    print("📝 FYERS API v3 - Orders Management Demo")
    print("=" * 55)
    
    # Load config
    try:
        with open("../config.json", "r") as f:
            config = json.load(f)
        
        client_id = config["client_id"]
        access_token = config["access_token"]
    except:
        print("❌ Please ensure config.json exists with credentials")
        return
    
    orders = FyersOrders(client_id, access_token)
    
    # 1. Get Orderbook
    print("\n📋 1. Testing Orderbook")
    orderbook = orders.get_orderbook()
    if orderbook:
        print(f"   📊 Found {len(orderbook)} orders in orderbook")
        for order in orderbook[:3]:  # Show first 3
            symbol = order.get('symbol', 'Unknown')
            status = order.get('status', 'Unknown')
            print(f"   • {symbol}: {status}")
    
    # 2. Get Tradebook
    print("\n📋 2. Testing Tradebook")
    tradebook = orders.get_tradebook()
    if tradebook:
        print(f"   📊 Found {len(tradebook)} trades in tradebook")
        for trade in tradebook[:3]:  # Show first 3
            symbol = trade.get('symbol', 'Unknown')
            qty = trade.get('qty', 0)
            price = trade.get('tradedPrice', 0)
            print(f"   • {symbol}: {qty} @ ₹{price}")
    
    # Note: Actual order placement is commented out for safety
    # In a real implementation, you would uncomment these for testing
    
    # # 3. Place a sample order (COMMENTED FOR SAFETY)
    # print("\n📋 3. Testing Order Placement")
    # sample_order = orders.place_order(
    #     symbol="NSE:RELIANCE-EQ",
    #     qty=1,
    #     side=1,  # BUY
    #     type=2,  # LIMIT
    #     limit_price=2500.0,
    #     product_type="CNC"
    # )
    # 
    # if sample_order:
    #     order_id = sample_order.get('id')
    #     
    #     # 4. Check order status
    #     print("\n📋 4. Testing Order Status")
    #     status = orders.get_order_status(order_id)
    #     
    #     # 5. Modify order
    #     print("\n📋 5. Testing Order Modification")
    #     orders.modify_order(order_id, limit_price=2450.0)
    #     
    #     # 6. Cancel order
    #     print("\n📋 6. Testing Order Cancellation")
    #     orders.cancel_order(order_id)
    
    print("\n✅ Order Management API demonstration completed!")
    print("💡 Order placement/modification/cancellation are commented for safety")

if __name__ == "__main__":
    demo_order_management()