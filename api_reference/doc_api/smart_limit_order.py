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
    "side": 1,                  # Buy
    "qty": 1,
    "productType": "CNC",
    "limitPrice": 1250,
    "stopPrice": 1200,
    "orderType": 1,
    "endTime": 1768987800,
    "hpr": 1300,
    "lpr": 700,
    "mpp": 1,
    "onExp": 2
}

response = fyers.create_smart_order_limit(data=data)
print(response)

