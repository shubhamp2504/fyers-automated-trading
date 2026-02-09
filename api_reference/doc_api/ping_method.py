def __ping(self) -> None:
  """
  Sends periodic ping messages to the server to maintain the WebSocket connection.

  The method continuously sends "__ping" messages to the server at a regular interval
  as long as the WebSocket connection is active.

  """

  while (
      self.__ws_object is not None
      and self.__ws_object.sock
      and self.__ws_object.sock.connected
  ):
      self.__ws_object.send("ping")
      time.sleep(10)


# To check whether the websocket connection is alive or not, we have to send a periodically “ping” message. If we get a pong from websocket, it means it is alive else dead. Find how we are doing in Python Code.

# Here there is one while loop with sleep of 10 seconds and we send a “ping” message to websocket to know that websocket is alive or not.
