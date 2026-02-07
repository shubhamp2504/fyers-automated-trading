import json
from fyers_apiv3 import fyersModel

class FyersClient:
    def __init__(self, config_path='config.json'):
        with open(config_path) as f:
            config = json.load(f)
        self.client_id = config['client_id']
        self.secret_key = config['secret_key']
        self.redirect_uri = config['redirect_uri']
        self.access_token = config['access_token']
        
        # Initialize FYERS Model with access token
        self.fyers = fyersModel.FyersModel(client_id=self.client_id, is_async=False, token=self.access_token, log_path=".")

    def place_order(self, symbol, qty, side, type_, productType, limitPrice=0, stopPrice=0, disclosedQty=0, validity="DAY", offlineOrder="False", stopLoss=0, takeProfit=0):
        order_data = {
            "symbol": symbol,
            "qty": qty,
            "type": type_,
            "side": side,
            "productType": productType,
            "limitPrice": limitPrice,
            "stopPrice": stopPrice,
            "disclosedQty": disclosedQty,
            "validity": validity,
            "offlineOrder": offlineOrder,
            "stopLoss": stopLoss,
            "takeProfit": takeProfit
        }
        response = self.fyers.place_order(order_data)
        return response

    def get_market_quote(self, symbols):
        return self.fyers.quotes({"symbols": symbols})
    
    def get_profile(self):
        return self.fyers.get_profile()
        
    def get_funds(self):
        return self.fyers.funds()
