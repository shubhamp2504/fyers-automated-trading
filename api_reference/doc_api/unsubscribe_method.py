def unsubscribe(self, data_type: str) -> None:
  """
    Unsubscribes from real-time updates of a specific data type.

  Args:
  data_type (str): The type of data to unsubscribe from, such as orders, position, holdings or general.

  """

  try:
    if self.__ws_object is not None:
      self.data_type = [
        self.socket_type[(type)] for type in data_type.split(",")
      ]
      message = json.dumps(
        {"T": "SUB_ORD", "SLIST": self.data_type, "SUB_T": -1}
      )
      self.__ws_object.send(message)

except Exception as e:
  self.order_logger.error(e)



# UnSubscribe Method

# To Unsubscribe for different actions, create a message data, which would be the string format for json node.

# message = json.dumps( {"T": "SUB_ORD", "SLIST": action_data, "SUB_T": -1} )

# Json node Params:

# T: Type: String
# value: “SUB_ORD” (Fixed)

# action_data: Type: List/Array
# Value: ['orders', 'trades', 'positions', 'edis', 'pricealerts', 'login'] Note: Based on the list passed in action_data web_socket data will be received

# SUB_T: Integer
# Value: -1 (value -1 is for unsubscribing and 1 for subscribe)

# Convert the json to string and send this message to socket to subscribe for action_data mentioned
