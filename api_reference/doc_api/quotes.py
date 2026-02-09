from fyers_apiv3 import fyersModel

client_id = "XC4XXXXM-100"
access_token = "eyJ0eXXXXXXXX2c5-Y3RgS8wR14g"
# Initialize the FyersModel instance with your client_id, access_token, and enable async mode
fyers = fyersModel.FyersModel(client_id=client_id, token=access_token,is_async=False, log_path="")

data = {
    "symbols":"NSE:SBIN-EQ,NSE:IDEA-EQ"
}

response = fyers.quotes(data=data)
print(response)
#  ------------------------------------------------------------------------------------------------------------------------------------------
#  Sample Success Response 
#  ------------------------------------------------------------------------------------------------------------------------------------------
#   {
#     "code":200,
#     "d":[
#         {
#           "n":"NSE:SBIN-EQ",
#           "s":"ok",
#           "v":{
#               "ask":0,
#               "bid":590.5,
#               "ch":-7.95,
#               "chp":-1.33,
#               "description":"NSE:SBIN-EQ",
#               "exchange":"NSE",
#               "fyToken":"10100000003045",
#               "high_price":600.85,
#               "low_price":585,
#               "lp":590.5,
#               "open_price":598.7,
#               "original_name":"NSE:SBIN-EQ",
#               "prev_close_price":598.45,
#               "atp": 428.07
#               "short_name":"SBIN-EQ",
#               "spread":-590.5,
#               "symbol":"NSE:SBIN-EQ",
#               "tt":"1691020800",
#               "volume":27774877
#           }
#         },
#         {
#           "n":"NSE:IDEA-EQ",
#           "s":"ok",
#           "v":{
#               "ask":7.85,
#               "bid":0,
#               "ch":-0.05,
#               "chp":-0.63,
#               "description":"NSE:IDEA-EQ",
#               "exchange":"NSE",
#               "fyToken":"101000000014366",
#               "high_price":8.05,
#               "low_price":7.8,
#               "lp":7.85,
#               "open_price":7.9,
#               "original_name":"NSE:IDEA-EQ",
#               "prev_close_price":7.9,
#               "atp": 7.5
#               "short_name":"IDEA-EQ",
#               "spread":7.85,
#               "symbol":"NSE:IDEA-EQ",
#               "tt":"1691020800",
#               "volume":78116800
#           }
#         }
#     ],
#     "message":"",
#     "s":"ok"
#   }
