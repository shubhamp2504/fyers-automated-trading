from fyers_client import FyersClient

class SimpleStrategy:
    def __init__(self, fyers_client):
        self.fyers = fyers_client

    def run(self):
        try:
            # Example: Buy if price is below a threshold
            symbol = "NSE:SBIN-EQ"
            print(f"Getting market quote for {symbol}...")
            
            quote = self.fyers.get_market_quote(symbol)
            print(f"Quote response: {quote}")
            
            if quote.get('s') == 'ok' and quote.get('d'):
                price = quote['d'][0]['v']['lp']
                print(f"Current price of {symbol}: ₹{price}")
                
                # Example trading logic - modify as needed
                if price < 1060:  # Example threshold
                    print("🟢 Price is below threshold, placing buy order...")
                    order = self.fyers.place_order(
                        symbol=symbol,
                        qty=1,
                        side=1,  # 1=Buy, -1=Sell
                        type_=2,  # 2=Market order, 1=Limit order
                        productType="INTRADAY"
                    )
                    print(f"Order response: {order}")
                else:
                    print(f"🔵 Price ₹{price} is above threshold ₹1060. No trade executed.")
            else:
                print(f"❌ Failed to get market quote: {quote.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error in strategy execution: {e}")
            import traceback
            traceback.print_exc()
