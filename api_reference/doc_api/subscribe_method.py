def subscribe(self, data_type: str) -> None:
"""
Subscribes to real-time updates of a specific data type.

Args:
    data_type (str): The type of data to subscribe to, such as orders, position, or holdings.


"""

try:
    if self.__ws_object is not None:
        self.data_type = []
        for elem in data_type.split(","):
            if isinstance(self.socket_type[elem], list):
                self.data_type.extend(self.socket_type[elem])
            else:
                self.data_type.append(self.socket_type[elem])
        
        print("Data type is ", self.data_type)
        print("Data type is ", type(self.data_type))
        message = json.dumps(
            {"T": "SUB_ORD", "SLIST": self.data_type, "SUB_T": 1}
        )
        self.__ws_object.send(message)

except Exception as e:
    self.order_logger.error(e) 



# Subscribe Method

# Once the connection is established, it is required to subscribe for required actions to get the data. To subscribe for different actions, create a message data, which would be the string format for json node.

# message = json.dumps( {"T": "SUB_ORD", "SLIST": action_data, "SUB_T": 1} )

# Json node Params:

# T: Type: String
# value: “SUB_ORD” (Fixed)

# action_data: Type: List/Array
# Value: ['orders', 'trades', 'positions', 'edis', 'pricealerts', 'login'] Note: Based on the list passed in action_data web_socket data will be received

# SUB_T: Integer
# Value: 1 (value 1 is for subscribing and -1 for unsubscribe)

# Convert the json to string and send this message to socket to subscribe for action_data mentioned

# Sample Response:

# {'code': 1605, 'message': 'Successfully subscribed', 's': 'ok'}

# Response from socket on any action triggered

# Once the subscribed successfully, for any action triggered, data will be received through socket, if any callback function is defined, would receive on function.

# Response would be string, In node we get as array buffer and in python we string, then it parsed to required format. In another programming language you might get in another format you just have to change in string.

# Type: string
# Value: {"orders":{"client_id":"XP0001","exchange":10,"fy_token":"10100000001628","id":"23121800292158","id_fyers":"df013f50-6925-4e2d-ba0f-0becf1229298","instrument":0,"lot_size":1,"offline_flag":false,"oms_flag":"K:1","ord_source":"W","ord_status":20,"ord_type":2,"ordertag":"2:Untagged","org_ord_status":4,"pan":"LVJPS3998E","precision":2,"price_multiplier":1,"product_type":"CNC","qty":1,"qty_multiplier":1,"qty_remaining":1,"report_type":"New Ack","segment":10,"status_msg":"New Ack","symbol":"NSE:BECTORFOOD-EQ","symbol_desc":"MRS BECTORS FOOD SPE LTD","symbol_exch":"BECTORFOOD", "tick_size":0.05,"time_epoch_oms":1702887690,"time_exch":"NA","time_oms":"18-Dec-2023 13:51:30","tran_side":1,"update_time_epoch_oms":1702887690,"update_time_exch":"01-Jan-1970 05:30:00","validity":"DAY"},"s":"ok"}

# Note:

#     Above response would be in the string format (In Python ) and arraybuffer (In Node). You have to check in which format you are getting data from websocket as it is dependent on the websocket library for the particular language.
#     Once you get data, you have to change into the string.(if already in string format then no need to change). After that, you have to change this string to JSON. Here you find the following keys as a JSON Key (One of them ) : orders, trades, positions.
#     Now Based on the Key the data is identified, if key is orders it means it is order update message, if key is trades then this message is for trades updates and same for positions updates.
#     In an Fyers SDK the socket raw response is parsed to generate data with required keys and remove the unnecessary keys

#     Parsed Data: {'s': 'ok', 'orders': {'clientId': 'XP03754', 'id': '23121800388066', 'qty': 1, 'remainingQuantity': 1, 'type': 2, 'fyToken': '10100000002705', 'exchange': 10, 'segment': 10, 'symbol': 'NSE:PRAJIND-EQ', 'instrument': 0, 'offlineOrder': False, 'orderDateTime': '18-Dec-2023 16:33:24', 'orderValidity': 'DAY', 'productType': 'CNC', 'side': 1, 'status': 4, 'source': 'W', 'ex_sym': 'PRAJIND', 'description': 'PRAJ INDUSTRIES LTD', 'orderNumStatus': '23121800388066:4'}}

#     All the keys information for orders updates are available there : Link
#     All the keys information for Trades updates are available there : Link
#     All the keys information for positions updates are available there : Link

#     Also attaching our internal mapping for your reference, how we are changing the keys from raw data to final data.

#             "position_mapper" :
#              {
#                     "symbol": "symbol",
#                     "id": "id",
#                     "buy_avg": "buyAvg",
#                     "buy_qty": "buyQty",
#                     "buy_val": "buyVal",
#                     "sell_avg": "sellAvg",
#                     "sell_qty": "sellQty",
#                     "sell_val": "sellVal",
#                     "net_avg": "netAvg",
#                     "net_qty": "netQty",
#                     "tran_side": "side",
#                     "qty": "qty",
#                     "product_type": "productType",
#                     "pl_realized": "realized_profit",
#                     "rbirefrate": "rbiRefRate",
#                     "fy_token": "fyToken",
#                     "exchange": "exchange",
#                     "segment": "segment",
#                     "day_buy_qty": "dayBuyQty",
#                     "day_sell_qty": "daySellQty",
#                     "cf_buy_qty": "cfBuyQty",
#                     "cf_sell_qty": "cfSellQty",
#                     "qty_multiplier": "qtyMulti_com",
#                     "pl_total": "pl",
#                     "cross_curr_flag": "crossCurrency",
#                     "pl_unrealized": "unrealized_profit"
#             },
#               "order_mapper" : 
#             {
#                     "client_id":"clientId",
#                     "id":"id",
#                     "id_parent":"parentId",
#                     "id_exchange":"exchOrdId",
#                     "qty":"qty",
#                     "qty_remaining":"remainingQuantity",
#                     "qty_filled":"filledQty",
#                     "price_limit":"limitPrice",
#                     "price_stop":"stopPrice",
#                     "tradedPrice":"price_traded",
#                     "ord_type":"type",
#                     "fy_token":"fyToken",
#                     "exchange":"exchange",
#                     "segment":"segment",
#                     "symbol":"symbol",
#                     "instrument":"instrument",
#                     "oms_msg":"message",
#                     "offline_flag":"offlineOrder",
#                     "time_oms":"orderDateTime",
#                     "validity":"orderValidity",
#                     "product_type":"productType",
#                     "tran_side":"side",
#                     "org_ord_status":"status",
#                     "ord_source":"source",
#                     "symbol_exch":"ex_sym",
#                     "symbol_desc":"description"
#               },
#                 "trade_mapper" : 
#               {
#                     "id_fill": "tradeNumber",
#                     "id": "orderNumber",
#                     "qty_traded": "tradedQty",
#                     "price_traded": "tradePrice",
#                     "traded_val": "tradeValue",
#                     "product_type": "productType",
#                     "client_id": "clientId",
#                     "id_exchange": "exchangeOrderNo",
#                     "ord_type": "orderType",
#                     "tran_side": "side",
#                     "symbol": "symbol",
#                     "fill_time": "orderDateTime",
#                     "fy_token": "fyToken",
#                     "exchange": "exchange",
#                     "segment": "segment"
#               }
              

