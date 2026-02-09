from fyers_apiv3 import fyersModel

client_id = "XC4XXXXM-100"
access_token = "eyJ0eXXXXXXXX2c5-Y3RgS8wR14g"

# Initialize the FyersModel instance with your client_id, access_token, and enable async mode
fyers = fyersModel.FyersModel(client_id=client_id, token=access_token,is_async=False, log_path="")

data = {
    "id": "25012400002074",
    "orderInfo": {
        "leg1": {
            "price": 1020,
            "triggerPrice": 1020,
            "qty": 5
        },
        "leg2": {
            "price": 620,
            "triggerPrice": 620,
            "qty": 5
        }
    }
}
response = fyers.modify_gtt_order(data=data)
print(response)
