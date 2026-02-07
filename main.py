from fyers_client import FyersClient
from strategy import SimpleStrategy

def main():
    fyers = FyersClient()
    strategy = SimpleStrategy(fyers)
    strategy.run()

if __name__ == "__main__":
    main()
