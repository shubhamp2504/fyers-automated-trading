from fyers_apiv3.FyersWebsocket import order_ws

def onPosition(message):
    """
    Callback function to handle incoming messages from the FyersDataSocket WebSocket.

    Parameters:
        message (dict): The received message from the WebSocket.

    """
    print("Position Response:", message)


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
    data_type = "OnPositions"

    # data_type = "OnOrders"
    # data_type = "OnTrades"
    # data_type = "OnGeneral"
    # data_type = "OnOrders,OnTrades,OnPositions,OnGeneral"

    fyers.subscribe(data_type=data_type)

    # Keep the socket running to receive real-time data
    fyers.keep_running()


# Replace the sample access token with your actual access token obtained from Fyers
access_token = "Xxxxx67IM-100:eyJxxxxxxIUzI1NiJ9.eyJpc3MiOiJhcGkuZnllcnMuaW4iLCJpYXQiOjE2OTA5NTMyNjAsImV4cCI6MTY5MTAyMjYyMCwibmJmIjoxNjkwOTUzMjYwLCJhdWQiOlsieDowIiwieDoxIiwieDoyIiwiZDoxIiwiZDoyIiwieDoxIiwieDowIl0sInN1YiI6ImFjY2Vzc190b2tlbiIsImF0X2hhc2giOiJnQUFBQUFCa3llWXMtX1FSSVdRLUItcUxENUdodFMxZ3JjeTZvREhSbjJhNE5EUEN2TWJLSEc1UEtaM3NqNWE5YWNzNDVueDQ1dHhkeWpqS1lNU1VTTmp2eDhfMlppXzNHQ2pJVkxDcE14U1BLUlhRTkFYV29xYz0iLCJkaXNwbGF5X25hbWUiOiJWSU5BWSBLVU1BUiBNQVVSWUEiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiIwYTU1NjU0ZGE5ZGM4NTMxMTFjN2FjMjQ2MzZlOTI1NDExMTBiN2M1MmNjNWNiZDE0MDA2OTA1NyIsImZ5X2lkIjoiWFYyMDk4NiIsImFwcFR5cGUiOjEwMCwicG9hX2ZsYWciOiJOIn0.vDp4UV_JIPxxxxGXLQOT7NnGxxxxNSBoLo"

# Create a FyersDataSocket instance with the provided parameters
fyers = order_ws.FyersOrderSocket(
    access_token=access_token,  # Your access token for authenticating with the Fyers API.
    write_to_file=False,        # A boolean flag indicating whether to write data to a log file or not.
    log_path="",                # The path to the log file if write_to_file is set to True (empty string means current directory).
    on_connect=onopen,          # Callback function to be executed upon successful WebSocket connection.
    on_close=onclose,           # Callback function to be executed when the WebSocket connection is closed.
    on_error=onerror,           # Callback function to handle any WebSocket errors that may occur.
    on_positions=onPosition,    # Callback function to handle position-related events from the WebSocket.
)


# Establish a connection to the Fyers WebSocket
fyers.connect()


------------------------------------------------------------------------------------------------------------------------------------------
Sample Success Response 
------------------------------------------------------------------------------------------------------------------------------------------
          
  {
    "s":"ok",
    "positions":{
        "symbol":"NSE:IDEA-EQ",
        "id":"NSE:IDEA-EQ-INTRADAY",
        "buyAvg":8,
        "buyQty":1,
        "buyVal":8,
        "sellAvg":7.95,
        "sellQty":1,
        "sellVal":7.95,
        "netAvg":0,
        "netQty":0,
        "side":0,
        "qty":0,
        "productType":"INTRADAY",
        "realized_profit":-0.04999999999999982,
        "rbiRefRate":1,
        "fyToken":"101000000014366",
        "exchange":10,
        "segment":10,
        "dayBuyQty":1,
        "daySellQty":1,
        "cfBuyQty":0,
        "cfSellQty":0,
        "qtyMulti_com":1
    }
  }
