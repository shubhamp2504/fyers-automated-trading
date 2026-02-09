from fyers_apiv3 import fyersModel

client_id = "AAAAAAAA-100"
access_token = ""

# Initialize the FyersModel instance with your clientid, access_token, and enable async mode_
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="") 

data ={
  "flowId": "74fd68df-b5e0-44fc-a705-a1338c453525",
    "profitRate": 8000,
    "lossRate": -3000,
  "type": 1,
  "name": "My Daily Exit"
}

update_smartexit_trigger= fyers.update_smartexit_trigger(data=data)
print(update_smartexit_trigger)

---------------------------------------------------------------------------------------------------------------------------------------------
Sample Success Response 
---------------------------------------------------------------------------------------------------------------------------------------------
Response structure:
{
  "code": 200,
  "message": "Updated the smart exit successfully.",
  "s": "ok",
  "data": {
    "flowId": "74fd68df-b5e0-44fc-a705-a1338c453525",
    "name": "My Daily Exit",
    "profitRate": 8000,
    "lossRate": -3000,
    "type": 1,
    "waitTime": 0
  }
}
---------------------------------------------------------------------------------------------------------------------------------------------
Activate/Deactivate Smart Exit Trigger
---------------------------------------------------------------------------------------------------------------------------------------------
const FyersAPI = require("fyers-api-v3").fyersModel

var fyers = new FyersAPI()
fyers.setAppId("AAAAAAAA-100")
fyers.setRedirectUrl("https://url.xyz")
fyers.setAccessToken("YOUR_ACCESS_TOKEN")

const activateReqBody = {
    "flowId": "74fd68df-b5e0-44fc-a705-a1338c453525"
}

fyers.activate_deactivate_smartexit_trigger(activateReqBody).then((response) => {
    console.log(response)
}).catch((error) => {
    console.log(error)
})
