from fyers_apiv3 import fyersModel

client_id = "AAAAAAAA-100"
access_token = ""

# Initialize the FyersModel instance with your clientid, access_token, and enable async mode_
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="") 

data ={
  "flowId": "356b4c5c-411b-46e9-9af3-edc2147c98ba"
}
pause_order = fyers.pause_smart_order(data=data)
print(pause_order)
