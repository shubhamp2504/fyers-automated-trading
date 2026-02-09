from fyers_apiv3.FyersWebsocket import data_ws

def onmessage(message):
    """
    Callback function to handle incoming messages from the FyersDataSocket WebSocket.

    Parameters:
        message (dict): The received message from the WebSocket.

    """
    print("Response:", message)


def onerror(message):
    """
    Callback function to handle WebSocket errors.

    Parameters:
        message (dict): The error message received from the WebSocket.


    """
    print("Error:", message)


def onclose(message):
    """
    Callback function to handle WebSocket connection close events.
    """
    print("Connection closed:", message)


def onopen():
    """
    Callback function to subscribe to data type and symbols upon WebSocket connection.

    """
    # Specify the data type and symbols you want to subscribe to
    data_type = "DepthUpdate"

    # Subscribe to the specified symbols and data type
    symbols = ['NSE:SBIN-EQ', 'NSE:ADANIENT-EQ']
    fyers.subscribe(symbols=symbols, data_type=data_type)

    # Keep the socket running to receive real-time data
    fyers.keep_running()


# Replace the sample access token with your actual access token obtained from Fyers
access_token = "XXDHHIOKS-100:eajoljXXXXXXXXXX"

# Create a FyersDataSocket instance with the provided parameters
fyers = data_ws.FyersDataSocket(
    access_token=access_token,       # Access token in the format "appid:accesstoken"
    log_path="",                     # Path to save logs. Leave empty to auto-create logs in the current directory.
    litemode=False,                  # Lite mode disabled. Set to True if you want a lite response.
    write_to_file=False,              # Save response in a log file instead of printing it.
    reconnect=True,                  # Enable auto-reconnection to WebSocket on disconnection.
    on_connect=onopen,               # Callback function to subscribe to data upon connection.
    on_close=onclose,                # Callback function to handle WebSocket connection close events.
    on_error=onerror,                # Callback function to handle WebSocket errors.
    on_message=onmessage             # Callback function to handle incoming messages from the WebSocket.
)

# Establish a connection to the Fyers WebSocket
fyers.connect()


#   ------------------------------------------------------------------------------------------------------------------------------------------
#  Sample Success Response 
#  ------------------------------------------------------------------------------------------------------------------------------------------
           
#  {
#   "bid_price1":606.25,
#   "bid_price2":606.2,
#   "bid_price3":606.15,
#   "bid_price4":606.1,
#   "bid_price5":606.05,
#   "ask_price1":606.3,
#   "ask_price2":606.35,
#   "ask_price3":606.4,
#   "ask_price4":606.45,
#   "ask_price5":606.5,
#   "bid_size1":20,
#   "bid_size2":902,
#   "bid_size3":111,
#   "bid_size4":110,
#   "bid_size5":979,
#   "ask_size1":282,
#   "ask_size2":568,
#   "ask_size3":2910,
#   "ask_size4":1676,
#   "ask_size5":2981,
#   "bid_order1":1,
#   "bid_order2":3,
#   "bid_order3":2,
#   "bid_order4":2,
#   "bid_order5":9,
#   "ask_order1":4,
#   "ask_order2":2,
#   "ask_order3":12,
#   "ask_order4":9,
#   "ask_order5":17,
#   "type":"dp",
#   "symbol":"NSE:SBIN-EQ"
# }
