from fyers_apiv3 import fyersModel

client_id = "AAAAAAAA-100"
access_token = "XXXXXXXXXXX"

# Initialize the FyersModel instance with your clientid, access_token, and enable async mode_
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")

data ={
  "flowId": "b86ee5e4-df36-47b3-a08f-dfb58e94472c"
}
cancel_order = fyers.cancel_smart_order(data=data)
print(cancel_order)
