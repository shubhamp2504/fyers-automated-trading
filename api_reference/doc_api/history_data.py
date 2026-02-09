from fyers_apiv3 import fyersModel

client_id = "XC4XXXXM-100"
access_token = "eyJ0eXXXXXXXX2c5-Y3RgS8wR14g"

# Initialize the FyersModel instance with your client_id, access_token, and enable async mode
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")

data = {
    "symbol":"NSE:SBIN-EQ",
    "resolution":"D",
    "date_format":"0",
    "range_from":"1690895316",
    "range_to":"1691068173",
    "cont_flag":"1"
}

response = fyers.history(data=data)
print(response)



#  ------------------------------------------------------------------------------------------------------------------------------------------
# Sample Success Response 
# ------------------------------------------------------------------------------------------------------------------------------------------
# {
#   "candles":[
#       [
#         1690934400,
#         609.85,
#         610.5,
#         594.1,
#         598.45,
#         14977497
#       ],
#       [
#         1691020800,
#         598.7,
#         600.85,
#         585,
#         590.5,
#         27774877
#       ]
#   ],
#   "code":200,
#   "message":"",
#   "s":"ok"
# }


# The historical API provides archived data (up to date) for the symbols. across various exchanges within the given range. A historical record is presented in the form of a candle and the data is available in different resolutions like - minute, 10 minutes, 60 minutes...240 minutes and daily.


# To Handle partial Candle

# To receive completed candle data, it is important to send a timestamp that comes before the current minute. If you send a timestamp for the current minute, you will receive partial data because the minute is not yet finished. Therefore, it is recommended to always use a "range_to" timestamp of the previous minute to ensure that you receive the completed candle data.
# Example:

# Current Time(seconds can be 1-59): 12:10:20 PM

# Input for history will be:

#     range_from: 12:08:00 PM
#     range_to: Current Time - 1 minute = 12:09:20 PM

# So you will get 2 candles - 12:08 PM and 12:09 PM candles. This example is for 1-minute candles; for other resolutions, you have to subtract the resolution time from "range_to" to get completed candles only.

# Limits for History

#     Unlimited number of stocks history data can be downloaded in a day.
#     Up to 100 days of data per request for resolutions of 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 120, 180, and 240 minutes. Data is available from July 3, 2017.
#     For 1D resolutions up to 366 days of data per request for 1D (1 day) resolutions.
#     For Seconds Charts the history will be available only for 30-Trading Days

# Request Attribute
# Attribute 	Data Type 	Description
# symbol* 	string 	Mandatory.
# Eg: NSE:SBIN-EQ
# resolution* 	string 	The candle resolution. Possible values are: Day : “D” or “1D”
# 5 seconds : “5S”
# 10 seconds : “10S”
# 15 seconds : “15S”
# 30 seconds : “30S”
# 45 seconds : “45S”
# 1 minute : “1”
# 2 minute : “2"
# 3 minute : "3"
# 5 minute : "5"
# 10 minute : "10"
# 15 minute : "15"
# 20 minute : "20"
# 30 minute : "30"
# 60 minute : "60"
# 120 minute : "120"
# 240 minute : "240"
# date_format* 	int 	date_format is a boolean flag. 0 to enter the epoch value. Eg:670073472. 1 to enter the date format as yyyy-mm-dd.
# range_from* 	string 	Indicating the start date of records. Accepts epoch value if date_format flag is set to 0. Eg: range_from: 670073472
# Accepts yyyy-mm-dd format if date_format flag is set to 1. Eg: 2021-01-01
# range_to* 	string 	Indicating the end date of records. Accepts epoch value if date_format flag is set to 0. Eg: range_to: 1622028732
# Accepts yyyy-mm-dd format if date_format flag is set to 1. Eg:2021-03-01
# cont_flag* 	int 	set cont flag 1 for continues data and future options.
# oi_flag 	int 	set flag to "1" enable oi as a part of candle
