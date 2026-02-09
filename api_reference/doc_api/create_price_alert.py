from fyers_apiv3 import fyersModel

client_id = "XC4XXXXM-100"
access_token = "eyJ0eXXXXXXXX2c5-Y3RgS8wR14g"

fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, is_async=False, log_path="")

data = {
    "agent": "fyers-api",
    "alert-type": 1,
    "name": "SBIN Alert",
    "symbol": "NSE:SILVERMIC25DECFUT",
    "comparisonType": "CLOSE",
    "condition": "LT",
    "value": 45
}

response = fyers.create_alert(data=data)
print(response)

# ------------------------------------------------------------------------------------------------------------------------------------------
# Sample Success Response 
# ------------------------------------------------------------------------------------------------------------------------------------------
# {
#   "code": 120,
#   "message": "A price alert for NSE:SILVERMIC25DECFUT at ₹45 is created.",
#   "s": "ok"
# }
