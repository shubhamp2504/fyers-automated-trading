from fyers_apiv3 import fyersModel

client_id = "XC4XXXXM-100"
access_token = "eyJ0eXXXXXXXX2c5-Y3RgS8wR14g"

fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, is_async=False, log_path="")

response = fyers.get_alert()
print(response)

# ------------------------------------------------------------------------------------------------------------------------------------------
# Sample Success Response 
# ------------------------------------------------------------------------------------------------------------------------------------------
# {
#   "code": 200,
#   "message": "",
#   "data": {
#     "5682484": {
#       "fyToken": "10100000008080",
#       "alert": {
#         "comparisonType": "LTP",
#         "condition": "LT",
#         "name": "Silver alert ",
#         "type": "V",
#         "value": 130,
#         "triggeredAt": "",
#         "createdAt": "09-Dec-2025 11:49:06",
#         "modifiedAt": "",
#         "notes": "",
#         "status": 2,
#         "triggeredEpoch": 0,
#         "createdEpoch": 1765280946,
#         "modifiedEpoch": 0
#       },
#       "symbol": "NSE:SILVERBEES-EQ"
#     }
#   },
#   "s": "ok"
# }
