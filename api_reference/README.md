# FYERS API v3 Complete Reference Guide

## 📚 Official Documentation
**⚠️ CRITICAL: Always consult https://myapi.fyers.in/docsv3 FIRST before implementing any feature**

This reference folder contains complete implementations of all FYERS API v3 endpoints with proper examples and documentation.

## 📁 Reference Structure

### 1. Authentication (`/authentication/`)
Complete authentication flow implementation:
- **[auth_complete.py](authentication/auth_complete.py)** - Full auth workflow
- Step 1: Authorization URL generation
- Step 2: Access token generation  
- Step 3: Token refresh
- Token validation and logout

**Key Endpoints:**
- `POST /generate-authcode` - Generate authorization URL
- `POST /validate-authcode` - Generate access token
- `POST /refresh-accesstoken` - Refresh token
- `DELETE /logout` - Logout user

### 2. Market Data (`/market_data/`)
All market data retrieval endpoints:
- **[market_data_complete.py](market_data/market_data_complete.py)** - Complete market data API
- Real-time quotes (up to 50 symbols per request)
- Historical data with multiple resolutions
- Market depth (Level 2 data)
- Option chains for derivatives
- Symbol master download
- Market status checking

**Key Endpoints:**
- `GET /quotes` - Get market quotes
- `GET /history` - Historical data  
- `GET /depth` - Market depth
- `GET /optionchain` - Option chain data
- `GET /symbolmaster` - Symbol master
- `GET /market-status` - Market status

### 3. Orders (`/orders/`)
Complete order management system:
- **[orders_complete.py](orders/orders_complete.py)** - Full order management
- Order placement (Market, Limit, Stop orders)
- Order modification and cancellation
- Orderbook and tradebook retrieval
- Advanced orders (Bracket, Cover orders)
- Order status tracking

**Key Endpoints:**
- `POST /orders` - Place new order
- `PUT /orders` - Modify existing order
- `DELETE /orders` - Cancel order
- `GET /orderbook` - Get all orders
- `GET /tradebook` - Get executed trades
- `GET /orders/{order_id}` - Get order status

### 4. Portfolio (`/portfolio/`)
Portfolio and account management:
- **[portfolio_complete.py](portfolio/portfolio_complete.py)** - Complete portfolio API
- User profile information
- Fund details and margins
- Holdings (delivery positions)
- Positions (intraday + overnight)
- Position conversion
- Transaction ledger

**Key Endpoints:**
- `GET /profile` - User profile
- `GET /funds` - Fund details
- `GET /holdings` - Stock holdings
- `GET /positions` - Trading positions
- `POST /positions/convert` - Convert position
- `GET /ledger` - Transaction history

### 5. WebSocket Streaming (`/websocket/`)
Real-time data streaming:
- **[websocket_complete.py](websocket/websocket_complete.py)** - Complete WebSocket implementation
- Real-time price data
- Market depth updates
- Symbol subscription management
- Connection handling and error recovery
- Data logging and callbacks

**WebSocket Features:**
- Symbol data streaming
- Market depth streaming
- Connection management
- Subscription/Unsubscription
- Error handling and reconnection

## 🔧 Usage Guidelines

### Before Implementing ANY Feature:
1. **Read the official docs first**: https://myapi.fyers.in/docsv3
2. **Check the reference implementation** in this folder
3. **Verify API parameters** and response formats
4. **Test with minimal examples** first
5. **Implement proper error handling**
6. **Add rate limiting** if needed

### Quick Reference Commands:

```python
# Authentication
from api_reference.authentication.auth_complete import FyersAuthentication
auth = FyersAuthentication(client_id, secret_key)
tokens = auth.step2_generate_access_token(auth_code)

# Market Data  
from api_reference.market_data.market_data_complete import FyersMarketData
market = FyersMarketData(client_id, access_token)
quotes = market.get_quotes(["NSE:RELIANCE-EQ"])

# Orders
from api_reference.orders.orders_complete import FyersOrders
orders = FyersOrders(client_id, access_token)
order = orders.place_order("NSE:RELIANCE-EQ", 1, 1, 2, limit_price=2500)

# Portfolio
from api_reference.portfolio.portfolio_complete import FyersPortfolio  
portfolio = FyersPortfolio(client_id, access_token)
holdings = portfolio.get_holdings()

# WebSocket
from api_reference.websocket.websocket_complete import FyersWebSocketReference
ws = FyersWebSocketReference(client_id, access_token)
ws.connect()
ws.subscribe_symbols(["NSE:RELIANCE-EQ"])
```

## 🎯 API Rate Limits & Best Practices

### Rate Limiting:
- **Quotes API**: Max 50 symbols per request
- **Historical Data**: 1 request per second recommended
- **Orders**: 10 orders per second max
- **WebSocket**: 100 symbol subscriptions max

### Best Practices:
- Always handle API errors gracefully
- Implement exponential backoff for retries
- Use batch requests when possible (quotes, etc.)
- Cache frequently accessed data
- Log all API calls for debugging
- Validate inputs before API calls

## 🔍 Error Handling Reference

Common FYERS API error codes:
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (invalid/expired token)
- `403` - Forbidden (insufficient permissions) 
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error

## 📞 Support Resources

- **Official Docs**: https://myapi.fyers.in/docsv3
- **API Status**: Check FYERS API status page
- **Rate Limits**: Documented per endpoint
- **Error Codes**: Complete list in official docs

## ⚠️ Important Notes

1. **Always test in paper trading mode first**
2. **Never hardcode credentials in source code**
3. **Use environment variables for sensitive data**
4. **Implement proper logging for all API calls**
5. **Handle network timeouts and failures**
6. **Keep tokens secure and refresh as needed**

---

**Last Updated**: February 7, 2026  
**FYERS API Version**: v3  
**Status**: Production Ready ✅

Remember: **Documentation First, Implementation Second!** 📚