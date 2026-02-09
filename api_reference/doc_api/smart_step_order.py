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
    "side": 1,
    "qty": 10,
    "productType": "CNC",
    "initQty": 2,
    "avgqty": 2,
    "avgdiff": 5,
    "direction": 1,
    "limitPrice": 750,
    "orderType": 1,
    "startTime": 1768987800,
    "endTime": 1768989000,
    "hpr": 800,
    "lpr": 700,
    "mpp": 1
}

response = fyers.create_smart_order_step(data=data)
print(response)
