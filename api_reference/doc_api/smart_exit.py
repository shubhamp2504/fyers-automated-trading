from fyers_apiv3 import fyersModel

client_id = "AAAAAAAA-100"
access_token = ""

# Initialize the FyersModel instance with your clientid, access_token, and enable async mode_
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="") 

data ={
    "name": "Alert Only Strategy",
  "type": 1,
    "profitRate": 5000,
    "lossRate": -2000
}
create_smartexit_trigger = fyers.create_smartexit_trigger(data=data)
print(create_smartexit_trigger)
