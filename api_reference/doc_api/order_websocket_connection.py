try:
if self.__ws_object is None:
    if self.write_to_file:
        self.background_flag = True
    header = {"authorization": self.__access_token}
    ws = websocket.WebSocketApp(
        self.__url,
        header=header,
        on_message=lambda ws, msg: self.__on_message(msg),
        on_error=lambda ws, msg: self.On_error(msg),
        on_close=lambda ws, close_code, close_reason: self.__on_close(
            ws, close_code, close_reason
        ),
        on_open=lambda ws: self.__on_open(ws),
    )
    self.t = Thread(target=ws.run_forever)
    self.t.daemon = self.background_flag
    self.t.start()

  #Sample callback function:

    def __on_open(self, ws):
    try:
        if self.__ws_object is None:
        self.__ws_object = ws
        self.ping_thread = threading.Thread(target=self.__ping)
        self.ping_thread.start()
    except Exception as e:
        self.order_logger.error(e)
        self.On_error(e)   


# Order WebSocket Connection

# To connect to order websocket, the below two input params are mandate

#     Websocket endpoint : wss://socket.fyers.in/trade/v3
#     Header:
#         Format: < appId:accessToken >
#         Sample header format : 7ABXUX38S-100:eyJ0eXAi**********qiTnzd2lGwS17s


# Based on the programming language chosen, respective socket connection libraries can be used.
# For the reference please find the sample code for socket connection written in Python.

# Here, we are making a connection with Fyers order websocket with parameters and callback functions on_message, on_error, on_close, on_open, which are required in the Python socket connection library used and would change as per the other programming library used. Sample callback function is shared below.

# Note : Handle accordingly in your Programming language

# Connection Response:

# On Success: Returns the socket object

# On Failure:

# Possible Error : Status code 403

#     Error: Handshake status 403 Forbidden -+-+- {'date': 'Tue, 19 Dec 2023 04:46:45 GMT', 'content-length': '0', 'connection': 'keep-alive', 'cf-cache-status': 'DYNAMIC', 'set-cookie': '__cf_bm=BOE16LGB7NHpNqW0AJOuFN1rcL3Q9TgnhmtpBfb3.Wk-1702961205-1- AfmECmK9cbVA2XGvkpx+jFuXyRsJET/ZOQYmw3LyZJ68pYLZTgtpbalvNs09ECZZ4GpPiogeYGhhFo+3PCp20nE=; path=/; expires=Tue, 19-Dec-23 05:16:45 GMT; domain=.fyers.in; HttpOnly; Secure', 'server': 'cloudflare', 'cf-ray': '837d00eec8ed17b6-MAA'} -+-+- b''
#     Reason : This error will come when your accessToken is wrong
#     How to solve : Provide correct accessToken
#     AccessToken format: < appID:accesstoken >

#     Error: Handshake status 404 Not Found -+-+- {'date': 'Tue, 19 Dec 2023 10:04:35 GMT', 'content-type': 'text/plain; charset=utf-8', 'content-length': '0', 'connection': 'keep-alive', 'cf-cache-status': 'DYNAMIC', 'set-cookie': '__cf_bm=I5oN6zdeKjfGsicqFiXZ57J5SX2IjsFDspaLIEGKPDE-1702980275-1-Ae+ldjrb3WfSuM7yNOzo3ykOIBQ1m+50QqcIqU26A+wqPIHhIEIGSy9kT2OG3OWNI0hwcmh7U+PnJ/aWWhz6fOA=; path=/; expires=Tue, 19-Dec-23 10:34:35 GMT; domain=.fyers.in; HttpOnly; Secure', 'server': 'cloudflare', 'cf-ray': '837ed28169c817ae-MAA'} -+-+- b''
#     Reason : Socket Connection URL would be wrong
#     How to solve : Provide valid URL.

# Sample callback function:

# For more reference, please find our on_open callback function code for more reference
