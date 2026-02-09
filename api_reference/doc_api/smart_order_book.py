from fyers_apiv3 import fyersModel

client_id = "AAAAAAAA-100"
access_token = ""

# Initialize the FyersModel instance with your clientid, access_token, and enable async mode_
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="") 

response = fyers.smart_orderbook_with_filter()
print(response)

