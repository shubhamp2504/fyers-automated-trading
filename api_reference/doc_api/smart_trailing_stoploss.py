from fyers_apiv3 import fyersModel

client_id = "AAAAAAAA-100"
access_token = "YOUR_ACCESS_TOKEN"

fyers = fyersModel.FyersModel(
    client_id=client_id,
    is_async=False,
    token=access_token,
    log_path=""
)

data = {
    "symbol": "NSE:SBIN-EQ",
    "side": -1,                 # Sell
    "qty": 1,
    "productType": "CNC",
    "stopPrice": 740,
    "jump_diff": 5,
    "orderType": 2,
    "mpp": 1
}

response = fyers.create_smart_order_trail(data=data)
print(response)
