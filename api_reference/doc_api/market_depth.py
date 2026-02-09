from fyers_apiv3 import fyersModel


client_id = "XC4XXXXM-100"
access_token = "eyJ0eXXXXXXXX2c5-Y3RgS8wR14g"

# Initialize the FyersModel instance with your client_id, access_token, and enable async mode
fyers = fyersModel.FyersModel(client_id=client_id, token=access_token,is_async=False, log_path="")

data = {
    "symbol":"NSE:SBIN-EQ",
    "ohlcv_flag":"1"
}

response = fyers.depth(data=data)
print(response)

#  ------------------------------------------------------------------------------------------------------------------------------------------
#  Sample Success Response 
#  ------------------------------------------------------------------------------------------------------------------------------------------
#  {
#  "s": "ok",
#  "d": {
#      "NSE:SBIN-EQ": {
#          "totalbuyqty": 2396063,
#          "totalsellqty": 4990001,
#          "bids": [
#              {
#                  "price": 427.25,
#                  "volume": 4738,
#                  "ord": 5
#              },
#              {
#                  "price": 427.2,
#                  "volume": 2631,
#                  "ord": 9
#              },
#              {
#                  "price": 427.15,
#                  "volume": 6366,
#                  "ord": 19
#              },
#              {
#                  "price": 427.1,
#                  "volume": 6344,
#                  "ord": 18
#              },
#              {
#                  "price": 427.05,
#                  "volume": 8136,
#                  "ord": 16
#              }
#          ],
#          "ask": [
#              {
#                  "price": 427.4,
#                  "volume": 2193,
#                  "ord": 4
#              },
#              {
#                  "price": 427.45,
#                  "volume": 5406,
#                  "ord": 19
#              },
#              {
#                  "price": 427.5,
#                  "volume": 15311,
#                  "ord": 57
#              },
#              {
#                  "price": 427.55,
#                  "volume": 11170,
#                  "ord": 17
#              },
#              {
#                  "price": 427.6,
#                  "volume": 7272,
#                  "ord": 25
#              }
#          ],
#          "o": 430.5,
#          "h": 433.65,
#          "l": 423.6,
#          "c": 425.2,
#          "chp": 0.48,
#          "tick_Size": 0.05,
#          "ch": 2.05,
#          "ltq": 20,
#          "ltt": 1622184920,
#          "ltp": 427.25,
#          "v": 39163870,
#          "atp": 428.07,
#          "lower_ckt": 382.7,
#          "upper_ckt": 467.7,
#          "expiry": "",
#          "oi": 0,
#          "oiflag": false,
#          "pdoi": 0,
#          "oipercent": 0.0
#      }
#  },
#   "message": ""
#   }
