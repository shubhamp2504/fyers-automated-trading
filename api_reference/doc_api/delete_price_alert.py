from fyers_apiv3 import fyersModel

client_id = "XC4XXXXM-100"
access_token = "eyJ0eXXXXXXXX2c5-Y3RgS8wR14g"

fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, is_async=False, log_path="")

data = {
    "alertId": "3870995",
    "agent": "fyers-api"
}

response = fyers.delete_alert(data=data)
print(response)

# ------------------------------------------------------------------------------------------------------------------------------------------
# Sample Success Response 
# ------------------------------------------------------------------------------------------------------------------------------------------
# {
#   "code": 121,
#   "message": "A price alert for NSE:GOLDBEES-EQ at ₹185 is deleted.",
#   "s": "ok"
# }
