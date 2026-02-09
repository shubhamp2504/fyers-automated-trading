from fyers_apiv3 import fyersModel

client_id = "XC4XXXXM-100"
access_token = "eyJ0eXXXXXXXX2c5-Y3RgS8wR14g"
# Initialize the FyersModel instance with your client_id, access_token, and enable async mode
fyers = fyersModel.FyersModel(client_id=client_id, token=access_token,is_async=False, log_path="")

data = {
    "orderTag": "tag1",
    "productType": "MARGIN",
    "offlineOrder": False,
    "orderType": "3L",
    "validity": "IOC",
    "legs": {
        "leg1": {
          "symbol": "NSE:SBIN24JUNFUT",
          "qty": 750,
          "side": 1,
          "type": 1,
          "limitPrice": 800
        },
        "leg2": {
            "symbol": "NSE:SBIN24JULFUT",
            "qty": 750,
            "side": 1,
            "type": 1,
            "limitPrice": 800
        },
        "leg3": {
            "symbol": "NSE:SBIN24JUN900CE",
            "qty": 750,
            "side": 1,
            "type": 1,
            "limitPrice": 3
        }
    }
}
response = fyers.place_multileg_order(data=data)
print(response)
