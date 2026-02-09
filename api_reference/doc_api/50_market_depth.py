from fyers_apiv3.FyersWebsocket.tbt_ws import FyersTbtSocket, SubscriptionModes

def onopen():
    """
    Callback function to subscribe to data type and symbols upon WebSocket connection.

    """
    print("Connection opened")
    # Specify the data type and symbols you want to subscribe to
    mode = SubscriptionModes.DEPTH
    Channel = '1'
    # Subscribe to the specified symbols and data type
    symbols = ['NSE:NIFTY25MARFUT', 'NSE:BANKNIFTY25MARFUT']
    
    fyers.subscribe(symbol_tickers=symbols, channelNo=Channel, mode=mode)
    fyers.switchChannel(resume_channels=[Channel], pause_channels=[])

    # Keep the socket running to receive real-time data
    fyers.keep_running()

def on_depth_update(ticker, message):
    """
    Callback function to handle incoming messages from the FyersDataSocket WebSocket.

    Parameters:
        ticker (str): The ticker symbol of the received message.
        message (Depth): The received message from the WebSocket.

    """
    print("ticker", ticker)
    print("depth response:", message)
    print("total buy qty:", message.tbq)
    print("total sell qty:", message.tsq)
    print("bids:", message.bidprice)
    print("asks:", message.askprice)
    print("bidqty:", message.bidqty)
    print("askqty:", message.askqty)
    print("bids ord numbers:", message.bidordn)
    print("asks ord numbers:", message.askordn)
    print("issnapshot:", message.snapshot)
    print("tick timestamp:", message.timestamp)


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

def onerror_message(message):
    """
    Callback function for error message events from the server

    Parameters:
        message (dict): The error message received from the Server.

    """
    print("Error Message:", message)

# Replace the sample access token with your actual access token obtained from Fyers
access_token = "XCXXXXXXM-100:eyJ0tHfZNSBoLo"


fyers = FyersTbtSocket(
    access_token=access_token,  # Your access token for authenticating with the Fyers API.
    write_to_file=False,        # A boolean flag indicating whether to write data to a log file or not.
    log_path="",                # The path to the log file if write_to_file is set to True (empty string means current directory).
    on_open=onopen,          # Callback function to be executed upon successful WebSocket connection.
    on_close=onclose,           # Callback function to be executed when the WebSocket connection is closed.
    on_error=onerror,           # Callback function to handle any WebSocket errors that may occur.
    on_depth_update=on_depth_update, # Callback function to handle depth-related events from the WebSocket
    on_error_message=onerror_message         # Callback function to handle server-related erros from the WebSocket.
)
# Establish a connection to the Fyers WebSocket
fyers.connect()



# Tick-by-Tick (TBT) Websocket Usage Guide
# Introduction

# Tick-by-tick data is the most detailed market data, recording every trade and order book update in real-time. Each "tick" includes the price, volume, and timestamp of individual trades, as well as changes to buy and sell orders. This granular data is crucial for analyzing market microstructure, tracking order flow, and developing high-frequency trading strategies.

# Key Points:

#     Available for NFO and NSE Instruments Only
#         Tick-by-tick data is exclusively available only for NFO (NSE Futures & Options) and NSE (Equity) instruments.
#     Data Formats
#         Requests are sent in JSON format.
#         Responses are received in protobuf format (a compact, efficient data format).
#     Incremental Data Updates
#         Instead of sending the full market data repeatedly, the server only sends changes (differences) between the last data packet and the current one.
#         To get the complete picture, users must maintain previous data and apply these changes.
#         The official SDKs provided by Fyers will handle this process automatically.
#     Snapshot Data on New Subscriptions
#         When a user subscribes to tick-by-tick data, the first packet received is a snapshot, containing the full market data at that moment.
#         After this, all subsequent packets only contain updates (differences) that need to be applied to the snapshot for a real-time view.

# TBT WebSocket Connection [50 Market Depth]

# Currently, these are the features available on the socket
# Feature 	Description 	Status
# TBT 50 Market Depth 	TBT 50 Market Depth provides users wtih 50 levels of market depth. Learn more 	Available
# To connect to tbt websocket, the below input params are mandated

#     Websocket endpoint : wss://rtsocket-api.fyers.in/versova
#     Header:
#     Key: Authorization
#         Format: < appId:accessToken >
#         Sample header format : 7ABXUX38S-100:eyJ0eXAi**********qiTnzd2lGwS17s

# Concept of channels

# With the Tick-by-Tick (TBT) WebSocket, we are introducing the concept of channels. A channel acts as a logical grouping for different subscribed symbols, making it easier to manage data streams efficiently.
# How Channels Work

#     When subscribing to market data, you need to specify both the symbols and a channel number.
#     Simply subscribing to a channel does not start the data stream—you must also resume the channel to begin receiving updates.

# Example Usage
# Let’s say you organize your subscriptions as follows:

#     Channel 1: All Nifty-related symbols
#     Channel 2: All BankNifty-related symbols

# Now, depending on what data you need, you can control the channels dynamically:

#     To receive only Nifty data → Pause Channel 2 and Resume Channel 1
#     To receive only BankNifty data → Pause Channel 1 and Resume Channel 2
#     To receive both Nifty and BankNifty data → Resume both channels

# This approach provides greater flexibility and control over market data streaming, allowing you to filter and manage real-time data efficiently.

# Request Message Types
# Type 	Purpose 	Encoding Format 	Data Format
# Ping 	Ping message to keep connection alive 	String 	

#       "ping"
      

# Subscribe 	Subscribe to the symbols for which data is required 	Json string 	

#       {
#         "type": 1,
#         "data": {
#             "subs": 1,
#             "symbols": ["NSE:IOC25FEBFUT", "NSE:NIFTY25FEBFUT", 
#             "NSE:BANKNIFTY25FEBFUT", "NSE:FINNIFTY25FEBFUT", "NSE:TCS25FEBFUT"],
#             "mode": "depth",
#             "channel": "1"
#         }
#       }
      

# Unsubscribe 	Unsubscribe to the symbols for which data is NOT required 	Json string 	

#       {
#         "type": 1,
#         "data": {
#             "subs": -1,
#             "symbols": ["NSE:IOC25FEBFUT", "NSE:NIFTY25FEBFUT", 
#             "NSE:BANKNIFTY25FEBFUT", "NSE:FINNIFTY25FEBFUT", "NSE:TCS25FEBFUT"],
#             "mode": "depth",
#             "channel": "1"
#         }
#       }
      

# Switch Channel 	Switch between active and paused channels 	Json string 	

#       {
#           "type": 2,
#           "data": {
#               "resumeChannels": ["1"],
#               "pauseChannels": []
#           }
#       }
      

# Response Message Types

# We use Protocol Buffers (protobuf) as the response format for all market data. The .proto file, which defines the data structure, is available at:

# 📌 Proto URL: https://public.fyers.in/tbtproto/1.0.0/msg.proto

# Protobuf is a widely used data format, and compilers are available to generate code in different programming languages.

# How to Install and Use Protobuf

#     Protobuf Compiler Installation: https://protobuf.dev/getting-started/
#     Using Protobuf with Python: https://protobuf.dev/reference/python/python-generated/
#     Using Protobuf with Node.js: https://www.npmjs.com/package/protobufjs

# We have uploaded the compiled files also for python, nodejs, and golang. You can download the files from the below link: https://public.fyers.in/tbtproto/1.0.0/protogencode.zip
# Copy the required file directly in your project and use it.
# Proto Versions and Links:
# Proto Version 	Proto URL 	Compiled files URL 	Updated on
# 1.0.0 	https://public.fyers.in/tbtproto/1.0.0/msg.proto 	https://public.fyers.in/tbtproto/1.0.0/protogencode.zip 	20-02-2025
# Type 	Data Format 	Field Explanation
# SocketMessage 	

#     type: value will always be MessageType.depth
#     feeds: string (key) will be the symbol ticker and value will be the MarketFeed datastructure. This value will have the actual 50 depth values
#     snapshot: true if its a snapshot and false if its a diff packet
#     msg: will mostly contain error msgs when error = true
#     error: true if it is an error, else false. In case of true, feeds will be nil/empty

	

# message SocketMessage {
#       MessageType type = 1;
#       map<string, MarketFeed> feeds = 2;
#       bool snapshot = 3;
#       string msg = 4;
#       bool error = 5;
# }
      

# MarketFeed 	

#     depth: Depth datastructure. This field will have the actual market depth value
#     feed_time: time of the packet in unix epoch
#     send_time: time of the packet at the time of sending from server
#     token: fytoken for the symbol
#     snapshot: true if its a snapshot else false for a diff packet
#     ticker: symbol ticker
#     Note: other fields can be ignored 

	

# message MarketFeed {
#     Quote quote = 1;
#     ExtendedQuote eq = 2;
#     DailyQuote dq = 3;
#     OHLCV ohlcv = 4;
#     Depth depth = 5;
#     google.protobuf.UInt64Value feed_time = 6; 
#     google.protobuf.UInt64Value send_time = 7;
#     string token = 8;
#     uint64 sequence_no = 9;
#     bool snapshot = 10;
#     string ticker = 11;
#     SymDetail symdetail = 12;
# }
      

# Depth 	

#     tbq: total bid qty
#     tsq: total sell qty
#     asks: array of asks of type Marketlevel datastructure
#     bids: array of bids of type Marketlevel datastructure

	

# message Depth {
#     google.protobuf.UInt64Value tbq = 1;                
#     google.protobuf.UInt64Value tsq = 2;        
#     repeated MarketLevel asks = 3;                 
#     repeated MarketLevel bids = 4;                       
# }
      

# MarketLevel 	

#     price: price
#     qty: qty in the market depth for the price
#     nord: number of orders in the market depth for the price
#     num: depth number, for 50 depth will be between 0 and 49 [0 based array indexing]

	

# message MarketLevel {
#     google.protobuf.Int64Value price = 1;
#     google.protobuf.UInt32Value qty  = 2; 
#     google.protobuf.UInt32Value nord = 3; 
#     google.protobuf.UInt32Value num = 4;
# }
      

# Ratelimits

# Following ratelimits apply for TBT Websocket:
# Description 	Limit
# Active Connections Per App Per User 	3
# Symbols per connection [Market Depth] 	5
# Channels per connection 	50 (1-50)