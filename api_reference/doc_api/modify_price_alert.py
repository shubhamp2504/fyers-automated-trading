from fyers_apiv3 import fyersModel

client_id = "XC4XXXXM-100"
access_token = "eyJ0eXXXXXXXX2c5-Y3RgS8wR14g"

fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, is_async=False, log_path="")

data = {
    "alertId": "3870995",
    "agent": "fyers-api",
    "alert-type": 1,
    "name": "Updated Alert",
    "symbol": "NSE:SILVERMIC25DECFUT",
    "comparisonType": "CLOSE",
    "condition": "LT",
    "value": 50
}

response = fyers.update_alert(data=data)
print(response)

# ------------------------------------------------------------------------------------------------------------------------------------------
# Sample Success Response 
# ------------------------------------------------------------------------------------------------------------------------------------------
# {
#   "code": 123,
#   "message": "A price alert for NSE:SILVERMIC25DECFUT at ₹50 is updated.",
#   "s": "ok"
# }
