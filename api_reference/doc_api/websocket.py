from fyers_apiv3.FyersWebsocket import data_ws

fyers = data_ws.FyersDataSocket(
    access_token=access_token,       # Access token in the format "appid:accesstoken"
    log_path="",                     # Path to save logs. Leave empty to auto-create logs in the current directory.
    litemode=False,                  # Lite mode disabled. Set to True if you want a lite response.
    write_to_file=False,              # Save response in a log file instead of printing it.
    reconnect=True,                  # Enable auto-reconnection to WebSocket on disconnection.
    on_connect=onopen,               # Callback function to subscribe to data upon connection.
    on_close=onclose,                # Callback function to handle WebSocket connection close events.
    on_error=onerror,                # Callback function to handle WebSocket errors.
    on_message=onmessage,            # Callback function to handle incoming messages from the WebSocket.
    reconnect_retry=10               # Number of times reconnection will be attepmted in case
)


# The WebSocket provides a robust method for accessing real-time data or order updates seamlessly and with low latency. It enables developers and users to establish a persistent, bidirectional connection with the server, allowing them to receive continuous updates, such as symbol updates, depth updates or orderupdate. To enhance your experience with our WebSocket, here are some helpful tips and best practices

#     Subscription Limit: You have the flexibility to subscribe up to 5000 data subscriptions simultaneously via WebSocket with latest SDK versions, please refer Change Log. Staying within this limit ensures smooth subscription management without errors.
#     Single Instance: Keep in mind that you can create only one WebSocket connection instance at a time. This approach ensures stability and prevents issues that might arise from multiple concurrent connections.
#     Efficient Thread Management: WebSocket operates on a dedicated thread, allowing it to run independently of your main application thread. This design guarantees that your primary tasks can continue without interruptions from WebSocket operations.
#     Customizable Callback Functions: Tailor your application's behavior using callback functions provided by the WebSocket. These functions empower you to define specific actions for events like data updates or error occurrences.
#     Auto-Reconnect: Enjoy uninterrupted connectivity by enabling automatic reconnection in case of disconnection. Simply set the reconnect parameter to true during WebSocket initialization, ensuring your application can recover without manual intervention.You can set max reconnection count upto 50.
#     Logging to File: If you want to log data to a file, you can set the write_to_file parameter to true. This feature allows you to efficiently save received data to a log file for analysis or archival purposes. The write_to_file function will only work without callback functions.
#     Reconnect Retry: If you want to define dynamic retry count(max 50), you can set the reconnect_retry parameter to int value of number of times you want it to try reconnection.(In case of node fyersdata.autoreconnect(trycount))
#     Disable Logging(node JS): In case you want to disable logging use disable logging flag to disable logging sample format:
#     new FyersSocket("token","logpath",true/*flag to enable disable logging*/)

