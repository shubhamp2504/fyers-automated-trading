from fyers_apiv3 import fyersModel

client_id = "XC4XXXXM-100"
access_token = "eyJ0eXXXXXXXX2c5-Y3RgS8wR14g"
# Initialize the FyersModel instance with your client_id, access_token, and enable async mode
fyers = fyersModel.FyersModel(client_id=client_id, token=access_token,is_async=False, log_path="")

data = {
      "side": 1,
      "symbol": "NSE:SBIN-EQ",
      "productType":"CNC",
      "orderInfo": {
          "leg1": {
              "price": 1000,
              "triggerPrice": 1000,
              "qty": 3
          },
          "leg2": {
              "price": 600,
              "triggerPrice": 600,
              "qty": 3
          }
      }
  }
response = fyers.place_gtt_order(data=data)
print(response)
