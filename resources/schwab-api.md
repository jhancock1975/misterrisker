# Schwab API Documentation

**Based on the `schwab-py` unofficial Python wrapper**  
*Repository: [alexgolec/schwab-py](https://github.com/alexgolec/schwab-py)*

> **Disclaimer:** `schwab-py` is an unofficial API wrapper. It is in no way endorsed by or affiliated with Charles Schwab or any associated organization. Make sure to read and understand the terms of service of the underlying API before using it.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Authentication](#authentication)
4. [HTTP Client](#http-client)
   - [Account Information](#account-information)
   - [Orders](#orders)
   - [Quotes](#quotes)
   - [Option Chains](#option-chains)
   - [Price History](#price-history)
   - [Market Movers](#market-movers)
   - [Market Hours](#market-hours)
   - [Instruments](#instruments)
   - [Transactions](#transactions)
5. [Order Templates](#order-templates)
   - [Equity Orders](#equity-orders)
   - [Option Orders](#option-orders)
   - [Complex Order Strategies](#complex-order-strategies)
6. [Order Builder](#order-builder)
7. [Streaming Client](#streaming-client)
   - [Level One Data](#level-one-data)
   - [Level Two Order Books](#level-two-order-books)
   - [Charts](#charts)
   - [Account Activity](#account-activity)
   - [Screeners](#screeners)
8. [Utilities](#utilities)
9. [Enumerations Reference](#enumerations-reference)

---

## Overview

`schwab-py` is an unofficial wrapper around the Charles Schwab Consumer APIs. It provides:

- **Safe Authentication**: Handles OAuth token fetch and refreshing automatically
- **Minimal API Wrapping**: Returns raw responses, allowing you to interpret complex API responses as you see fit
- **Login and authentication**
- **Quotes, fundamentals, and historical pricing data**
- **Options chains**
- **Streaming quotes and order book depth data**
- **Trades and trade management**
- **Account info**

### Limitations

- Does not provide access to thinkorswim (TOS) API
- No paper trading/sandbox environment
- Schwab API does not provide price history for options, futures, or instruments other than equities/ETFs

---

## Installation

```bash
pip install schwab-py
```

### Quick Start Example

```python
from schwab import auth, client
import json

api_key = 'YOUR_API_KEY'
app_secret = 'YOUR_APP_SECRET'
callback_url = 'https://127.0.0.1:8182/'
token_path = '/path/to/token.json'

c = auth.easy_client(api_key, app_secret, callback_url, token_path)

r = c.get_price_history_every_day('AAPL')
r.raise_for_status()
print(json.dumps(r.json(), indent=4))
```

---

## Authentication

The Schwab API uses OAuth 2.0 authentication. `schwab-py` provides several methods to obtain and manage tokens.

### Setting Up Your Schwab Developer Account

1. **Create a Schwab Developer Account** at [beta-developer.schwab.com](https://beta-developer.schwab.com/)
2. **Create an Application** with:
   - **API Product**: "Accounts and Trading Production" (recommended)
   - **Order Limit**: 120 orders per minute (recommended max)
   - **Callback URL**: Must use HTTPS (e.g., `https://127.0.0.1:8182/`)

### Authentication Methods

#### `easy_client()` - Recommended for Most Users

```python
from schwab.auth import easy_client

c = easy_client(
    api_key='YOUR_API_KEY',
    app_secret='YOUR_APP_SECRET',
    callback_url='https://127.0.0.1:8182/',
    token_path='/path/to/token.json',
    asyncio=False,              # Set True for async support
    enforce_enums=True,         # Validate enum inputs
    max_token_age=60*60*24*6.5, # Token age in seconds (6.5 days default)
    callback_timeout=300.0,     # Timeout waiting for callback
    interactive=True            # Require user input before browser
)
```

If a valid token exists at `token_path`, it loads from file. Otherwise, it initiates a login flow.

#### `client_from_login_flow()` - Browser-Based Authentication

```python
from schwab.auth import client_from_login_flow

c = client_from_login_flow(
    api_key='YOUR_API_KEY',
    app_secret='YOUR_APP_SECRET',
    callback_url='https://127.0.0.1:8182/',
    token_path='/path/to/token.json',
    asyncio=False,
    enforce_enums=True,
    callback_timeout=300.0,
    interactive=True,
    requested_browser=None  # Use specific browser (e.g., 'chrome', 'firefox')
)
```

This opens a browser, starts a local HTTPS server, and captures the OAuth callback.

#### `client_from_manual_flow()` - Manual Copy-Paste Authentication

```python
from schwab.auth import client_from_manual_flow

c = client_from_manual_flow(
    api_key='YOUR_API_KEY',
    app_secret='YOUR_APP_SECRET',
    callback_url='https://127.0.0.1:8182/',
    token_path='/path/to/token.json',
    asyncio=False,
    enforce_enums=True
)
```

Useful for Jupyter notebooks or environments without browser access. Prints a URL for manual login.

#### `client_from_token_file()` - Load Existing Token

```python
from schwab.auth import client_from_token_file

c = client_from_token_file(
    token_path='/path/to/token.json',
    api_key='YOUR_API_KEY',
    app_secret='YOUR_APP_SECRET',
    asyncio=False,
    enforce_enums=True
)
```

#### `client_from_access_functions()` - Custom Token Storage

```python
from schwab.auth import client_from_access_functions

def read_token():
    # Read and return token from your storage
    pass

def write_token(token, *args, **kwargs):
    # Write token to your storage
    pass

c = client_from_access_functions(
    api_key='YOUR_API_KEY',
    app_secret='YOUR_APP_SECRET',
    token_read_func=read_token,
    token_write_func=write_token,
    asyncio=False,
    enforce_enums=True
)
```

### Web Application Authentication

For integrating into web applications:

```python
from schwab.auth import get_auth_context, client_from_received_url

# Step 1: Get authorization URL
auth_context = get_auth_context(api_key, callback_url, state=None)
# Redirect user to: auth_context.authorization_url

# Step 2: After user logs in and is redirected back
c = client_from_received_url(
    api_key, app_secret, auth_context, received_url, 
    token_write_func, asyncio=False, enforce_enums=True
)
```

---

## HTTP Client

The HTTP client provides direct access to all Schwab API endpoints.

### Client Configuration

```python
# Set timeout for all requests (default: 30 seconds)
c.set_timeout(30.0)

# Get token age in seconds
age = c.token_age()
```

### Account Information

#### Get Account Numbers (Account Hashes)

```python
# Returns mapping of account IDs to account hashes
# Account hashes are required for most account-specific operations
response = c.get_account_numbers()
accounts = response.json()
# Example: [{'accountNumber': '12345678', 'hashValue': 'ABC123...'}]
```

#### Get Account Details

```python
from schwab.client import Client

# Get single account
response = c.get_account(
    account_hash='ABC123...',
    fields=[Client.Account.Fields.POSITIONS]  # Optional: include positions
)

# Get all linked accounts
response = c.get_accounts(
    fields=[Client.Account.Fields.POSITIONS]
)
```

**Account Fields:**
- `Client.Account.Fields.POSITIONS` - Include position information

### Orders

#### Get Orders

```python
from schwab.client import Client
import datetime

# Get orders for specific account
response = c.get_orders_for_account(
    account_hash='ABC123...',
    max_results=100,
    from_entered_datetime=datetime.datetime.now() - datetime.timedelta(days=30),
    to_entered_datetime=datetime.datetime.now(),
    status=Client.Order.Status.FILLED
)

# Get orders for all linked accounts
response = c.get_orders_for_all_linked_accounts(
    max_results=100,
    from_entered_datetime=datetime.datetime.now() - datetime.timedelta(days=30),
    to_entered_datetime=datetime.datetime.now(),
    status=Client.Order.Status.WORKING
)

# Get specific order
response = c.get_order(order_id=123456789, account_hash='ABC123...')
```

**Order Status Values:**
| Status | Description |
|--------|-------------|
| `AWAITING_PARENT_ORDER` | Waiting for parent order |
| `AWAITING_CONDITION` | Waiting for condition |
| `AWAITING_STOP_CONDITION` | Waiting for stop condition |
| `AWAITING_MANUAL_REVIEW` | Under manual review |
| `ACCEPTED` | Order accepted |
| `AWAITING_UR_OUT` | Awaiting UR out |
| `PENDING_ACTIVATION` | Pending activation |
| `QUEUED` | Order queued |
| `WORKING` | Order is working |
| `REJECTED` | Order rejected |
| `PENDING_CANCEL` | Cancel pending |
| `CANCELED` | Order canceled |
| `PENDING_REPLACE` | Replace pending |
| `REPLACED` | Order replaced |
| `FILLED` | Order filled |
| `EXPIRED` | Order expired |
| `NEW` | New order |
| `AWAITING_RELEASE_TIME` | Waiting for release time |
| `PENDING_ACKNOWLEDGEMENT` | Pending acknowledgement |
| `PENDING_RECALL` | Pending recall |
| `UNKNOWN` | Unknown status |

#### Place Order

```python
from schwab.orders.equities import equity_buy_limit

order = equity_buy_limit('AAPL', 10, '150.00')
response = c.place_order(account_hash='ABC123...', order_spec=order)

# Extract order ID from response
order_id = response.headers.get('location').split('/')[-1]
```

#### Replace Order

```python
new_order = equity_buy_limit('AAPL', 10, '155.00')
response = c.replace_order(
    account_hash='ABC123...',
    order_id=123456789,
    order_spec=new_order
)
```

#### Cancel Order

```python
response = c.cancel_order(order_id=123456789, account_hash='ABC123...')
```

#### Preview Order

```python
order = equity_buy_limit('AAPL', 10, '150.00')
response = c.preview_order(account_hash='ABC123...', order_spec=order)
```

### Quotes

#### Get Single Quote

```python
from schwab.client import Client

response = c.get_quote(
    symbol='AAPL',
    fields=[
        Client.Quote.Fields.QUOTE,
        Client.Quote.Fields.FUNDAMENTAL,
        Client.Quote.Fields.EXTENDED,
        Client.Quote.Fields.REFERENCE,
        Client.Quote.Fields.REGULAR
    ]
)
```

#### Get Multiple Quotes

```python
response = c.get_quotes(
    symbols=['AAPL', 'MSFT', 'GOOGL', '/ES'],  # Supports futures like /ES
    fields=[Client.Quote.Fields.QUOTE],
    indicative=False  # Optional: indicative quotes
)
```

**Quote Fields:**
- `QUOTE` - Current quote data
- `FUNDAMENTAL` - Fundamental data (P/E, dividend, etc.)
- `EXTENDED` - Extended hours data
- `REFERENCE` - Reference data
- `REGULAR` - Regular market data

### Option Chains

```python
from schwab.client import Client
import datetime

response = c.get_option_chain(
    symbol='AAPL',
    contract_type=Client.Options.ContractType.CALL,  # CALL, PUT, or ALL
    strike_count=10,  # Number of strikes above/below ATM
    include_underlying_quote=True,
    strategy=Client.Options.Strategy.SINGLE,
    interval=None,  # Strike interval for spreads
    strike=150.0,  # Specific strike price
    strike_range=Client.Options.StrikeRange.NEAR_THE_MONEY,
    from_date=datetime.date.today(),
    to_date=datetime.date.today() + datetime.timedelta(days=30),
    volatility=None,  # For ANALYTICAL strategy
    underlying_price=None,  # For ANALYTICAL strategy
    interest_rate=None,  # For ANALYTICAL strategy
    days_to_expiration=None,  # For ANALYTICAL strategy
    exp_month=Client.Options.ExpirationMonth.JANUARY,
    option_type=Client.Options.Type.STANDARD,
    entitlement=Client.Options.Entitlement.NON_PRO
)

# Get option expiration chain
response = c.get_option_expiration_chain(symbol='AAPL')
```

**Contract Types:**
- `CALL`, `PUT`, `ALL`

**Strategies:**
- `SINGLE`, `ANALYTICAL`, `COVERED`, `VERTICAL`, `CALENDAR`, `STRANGLE`, `STRADDLE`, `BUTTERFLY`, `CONDOR`, `DIAGONAL`, `COLLAR`, `ROLL`

**Strike Ranges:**
- `IN_THE_MONEY` (ITM), `NEAR_THE_MONEY` (NTM), `OUT_OF_THE_MONEY` (OTM)
- `STRIKES_ABOVE_MARKET` (SAK), `STRIKES_BELOW_MARKET` (SBK), `STRIKES_NEAR_MARKET` (SNK), `ALL`

**Option Types:**
- `STANDARD` (S), `NON_STANDARD` (NS), `ALL`

**Expiration Months:**
- `JANUARY`, `FEBRUARY`, `MARCH`, `APRIL`, `MAY`, `JUNE`, `JULY`, `AUGUST`, `SEPTEMBER`, `OCTOBER`, `NOVEMBER`, `DECEMBER`, `ALL`

**Entitlements:**
- `PAYING_PRO` (PP), `NON_PRO` (NP), `NON_PAYING_PRO` (PN)

### Price History

Schwab provides price history for equities and ETFs only (not options or futures).

#### Generic Price History

```python
from schwab.client import Client
import datetime

response = c.get_price_history(
    symbol='AAPL',
    period_type=Client.PriceHistory.PeriodType.YEAR,
    period=Client.PriceHistory.Period.ONE_YEAR,
    frequency_type=Client.PriceHistory.FrequencyType.DAILY,
    frequency=Client.PriceHistory.Frequency.DAILY,
    start_datetime=datetime.datetime(2023, 1, 1),
    end_datetime=datetime.datetime(2024, 1, 1),
    need_extended_hours_data=True,
    need_previous_close=True
)
```

#### Convenience Methods

```python
# Per-minute data (up to ~48 days)
response = c.get_price_history_every_minute(
    symbol='AAPL',
    start_datetime=datetime.datetime(2024, 1, 1),
    end_datetime=datetime.datetime(2024, 1, 15),
    need_extended_hours_data=True,
    need_previous_close=True
)

# Per-5-minutes data (~9 months)
response = c.get_price_history_every_five_minutes('AAPL')

# Per-10-minutes data (~9 months)
response = c.get_price_history_every_ten_minutes('AAPL')

# Per-15-minutes data (~9 months)
response = c.get_price_history_every_fifteen_minutes('AAPL')

# Per-30-minutes data (~9 months)
response = c.get_price_history_every_thirty_minutes('AAPL')

# Daily data (20+ years for some symbols like AAPL)
response = c.get_price_history_every_day('AAPL')

# Weekly data (20+ years)
response = c.get_price_history_every_week('AAPL')
```

**Period Types:**
- `DAY`, `MONTH`, `YEAR`, `YEAR_TO_DATE`

**Periods:**
- Daily: `ONE_DAY`, `TWO_DAYS`, `THREE_DAYS`, `FOUR_DAYS`, `FIVE_DAYS`, `TEN_DAYS`
- Monthly: `ONE_MONTH`, `TWO_MONTHS`, `THREE_MONTHS`, `SIX_MONTHS`
- Yearly: `ONE_YEAR`, `TWO_YEARS`, `THREE_YEARS`, `FIVE_YEARS`, `TEN_YEARS`, `FIFTEEN_YEARS`, `TWENTY_YEARS`
- YTD: `YEAR_TO_DATE`

**Frequency Types:**
- `MINUTE`, `DAILY`, `WEEKLY`, `MONTHLY`

**Frequencies:**
- Minute: `EVERY_MINUTE`, `EVERY_FIVE_MINUTES`, `EVERY_TEN_MINUTES`, `EVERY_FIFTEEN_MINUTES`, `EVERY_THIRTY_MINUTES`
- Other: `DAILY`, `WEEKLY`, `MONTHLY`

### Market Movers

```python
from schwab.client import Client

response = c.get_movers(
    index=Client.Movers.Index.SPX,
    sort_order=Client.Movers.SortOrder.PERCENT_CHANGE_UP,
    frequency=Client.Movers.Frequency.ZERO
)
```

**Indexes:**
- `DJI` ($DJI), `COMPX` ($COMPX), `SPX` ($SPX)
- `NYSE`, `NASDAQ`, `OTCBB`
- `INDEX_ALL`, `EQUITY_ALL`, `OPTION_ALL`, `OPTION_PUT`, `OPTION_CALL`

**Sort Orders:**
- `VOLUME`, `TRADES`, `PERCENT_CHANGE_UP`, `PERCENT_CHANGE_DOWN`

**Frequencies:**
- `ZERO`, `ONE`, `FIVE`, `TEN`, `THIRTY`, `SIXTY`

### Market Hours

```python
from schwab.client import Client
import datetime

response = c.get_market_hours(
    markets=[
        Client.MarketHours.Market.EQUITY,
        Client.MarketHours.Market.OPTION,
        Client.MarketHours.Market.FUTURE,
        Client.MarketHours.Market.FOREX,
        Client.MarketHours.Market.BOND
    ],
    date=datetime.date.today()
)
```

### Instruments

#### Search Instruments

```python
from schwab.client import Client

# Symbol search
response = c.get_instruments(
    symbols='AAPL',
    projection=Client.Instrument.Projection.SYMBOL_SEARCH
)

# Symbol regex
response = c.get_instruments(
    symbols='AAP.*',
    projection=Client.Instrument.Projection.SYMBOL_REGEX
)

# Description search
response = c.get_instruments(
    symbols='Apple',
    projection=Client.Instrument.Projection.DESCRIPTION_SEARCH
)

# Get fundamentals
response = c.get_instruments(
    symbols=['AAPL', 'MSFT'],
    projection=Client.Instrument.Projection.FUNDAMENTAL
)

# Get by CUSIP
response = c.get_instrument_by_cusip(cusip='037833100')  # AAPL CUSIP
```

**Projection Types:**
| Projection | Symbol Format | Description |
|------------|---------------|-------------|
| `SYMBOL_SEARCH` | String or array | Search by symbols |
| `SYMBOL_REGEX` | Single string | Regex match symbols |
| `DESCRIPTION_SEARCH` | Single string | Search descriptions |
| `DESCRIPTION_REGEX` | Single string | Regex match descriptions |
| `SEARCH` | Single string | General search |
| `FUNDAMENTAL` | String or array | Get fundamental data |

### Transactions

```python
from schwab.client import Client
import datetime

# Get transactions for account
response = c.get_transactions(
    account_hash='ABC123...',
    start_date=datetime.datetime.now() - datetime.timedelta(days=30),
    end_date=datetime.datetime.now(),
    transaction_types=[
        Client.Transactions.TransactionType.TRADE,
        Client.Transactions.TransactionType.DIVIDEND_OR_INTEREST
    ],
    symbol='AAPL'  # Optional: filter by symbol
)

# Get specific transaction
response = c.get_transaction(
    account_hash='ABC123...',
    transaction_id=123456789
)
```

**Transaction Types:**
- `TRADE`, `RECEIVE_AND_DELIVER`, `DIVIDEND_OR_INTEREST`
- `ACH_RECEIPT`, `ACH_DISBURSEMENT`, `CASH_RECEIPT`, `CASH_DISBURSEMENT`
- `ELECTRONIC_FUND`, `WIRE_OUT`, `WIRE_IN`
- `JOURNAL`, `MEMORANDUM`, `MARGIN_CALL`, `MONEY_MARKET`, `SMA_ADJUSTMENT`

### User Preferences

```python
response = c.get_user_preferences()
```

---

## Order Templates

Pre-built order templates for common order types.

### Equity Orders

```python
from schwab.orders.equities import (
    equity_buy_market,
    equity_buy_limit,
    equity_sell_market,
    equity_sell_limit,
    equity_sell_short_market,
    equity_sell_short_limit,
    equity_buy_to_cover_market,
    equity_buy_to_cover_limit
)

# Buy orders
order = equity_buy_market('AAPL', 10)            # Market buy 10 shares
order = equity_buy_limit('AAPL', 10, '150.00')   # Limit buy at $150.00

# Sell orders
order = equity_sell_market('AAPL', 10)           # Market sell
order = equity_sell_limit('AAPL', 10, '160.00')  # Limit sell at $160.00

# Short sell orders
order = equity_sell_short_market('AAPL', 10)
order = equity_sell_short_limit('AAPL', 10, '160.00')

# Buy to cover orders
order = equity_buy_to_cover_market('AAPL', 10)
order = equity_buy_to_cover_limit('AAPL', 10, '145.00')

# Place the order
c.place_order(account_hash='ABC123...', order_spec=order)
```

### Option Orders

```python
from schwab.orders.options import (
    # Single leg options
    option_buy_to_open_market,
    option_buy_to_open_limit,
    option_sell_to_open_market,
    option_sell_to_open_limit,
    option_buy_to_close_market,
    option_buy_to_close_limit,
    option_sell_to_close_market,
    option_sell_to_close_limit,
    # Verticals
    bull_call_vertical_open,
    bull_call_vertical_close,
    bear_call_vertical_open,
    bear_call_vertical_close,
    bull_put_vertical_open,
    bull_put_vertical_close,
    bear_put_vertical_open,
    bear_put_vertical_close,
    # Option symbol helper
    OptionSymbol
)

# Build option symbol
option_symbol = OptionSymbol(
    underlying_symbol='AAPL',
    expiration_date='240420',  # YYMMDD or datetime.date
    contract_type='C',         # 'C'/'CALL' or 'P'/'PUT'
    strike_price_as_string='150'
).build()
# Result: 'AAPL  240420C00150000'

# Parse existing option symbol
parsed = OptionSymbol.parse_symbol('AAPL  240420C00150000')

# Single leg option orders
order = option_buy_to_open_market(option_symbol, 1)
order = option_buy_to_open_limit(option_symbol, 1, '5.00')
order = option_sell_to_close_limit(option_symbol, 1, '6.00')

# Vertical spreads
order = bull_call_vertical_open(
    long_call_symbol='AAPL  240420C00150000',
    short_call_symbol='AAPL  240420C00160000',
    quantity=1,
    net_debit='3.00'
)

order = bear_put_vertical_open(
    short_put_symbol='AAPL  240420P00140000',
    long_put_symbol='AAPL  240420P00130000',
    quantity=1,
    net_debit='2.50'
)
```

### Complex Order Strategies

```python
from schwab.orders.common import one_cancels_other, first_triggers_second
from schwab.orders.equities import equity_buy_limit, equity_sell_limit

# One-Cancels-Other (OCO)
order = one_cancels_other(
    equity_buy_limit('AAPL', 10, '145.00'),
    equity_buy_limit('MSFT', 5, '350.00')
)

# First-Triggers-Second (bracket order)
order = first_triggers_second(
    equity_buy_limit('AAPL', 10, '150.00'),  # Entry
    equity_sell_limit('AAPL', 10, '165.00')  # Exit
)
```

---

## Order Builder

For complete control over order construction:

```python
from schwab.orders.generic import OrderBuilder
from schwab.orders.common import (
    Session, Duration, OrderType, OrderStrategyType,
    EquityInstruction, OptionInstruction,
    ComplexOrderStrategyType, SpecialInstruction, Destination
)

order = (OrderBuilder()
    # Basic order properties
    .set_session(Session.NORMAL)
    .set_duration(Duration.DAY)
    .set_order_type(OrderType.LIMIT)
    .set_price('150.00')  # Use strings for prices
    
    # Order strategy
    .set_order_strategy_type(OrderStrategyType.SINGLE)
    
    # For complex option strategies
    .set_complex_order_strategy_type(ComplexOrderStrategyType.VERTICAL)
    .set_quantity(1)  # For multi-leg orders
    
    # Add equity leg
    .add_equity_leg(EquityInstruction.BUY, 'AAPL', 10)
    
    # Or add option leg
    .add_option_leg(OptionInstruction.BUY_TO_OPEN, 'AAPL  240420C00150000', 1)
    
    # Stop orders
    .set_stop_price('145.00')
    .set_stop_type(StopType.STANDARD)
    
    # Special instructions
    .set_special_instruction(SpecialInstruction.ALL_OR_NONE)
    
    # Destination routing
    .set_destination_link_name(Destination.NASDAQ)
    
    # Activation price (for trailing stops)
    .set_activation_price(155.00)
    
    # Child orders for complex strategies
    .add_child_order_strategy(another_order)
)

# Build the order dict
order_dict = order.build()
```

### Sessions

| Session | Hours | Description |
|---------|-------|-------------|
| `NORMAL` | 9:30 AM - 4:00 PM ET | Regular market hours |
| `AM` | 8:00 AM - 9:30 AM ET | Pre-market session |
| `PM` | 4:00 PM - 8:00 PM ET | After-hours session |
| `SEAMLESS` | All except overnight | AM + NORMAL + PM |

### Durations

| Duration | Description |
|----------|-------------|
| `DAY` | Cancel at end of trading day |
| `GOOD_TILL_CANCEL` | Active for 6 months or until cancel date |
| `FILL_OR_KILL` | Execute immediately at specified price or cancel |
| `IMMEDIATE_OR_CANCEL` | Execute immediately, cancel unfilled portion |
| `END_OF_WEEK` | Active until end of week |
| `END_OF_MONTH` | Active until end of month |
| `NEXT_END_OF_MONTH` | Active until next end of month |

### Order Types

| Order Type | Description |
|------------|-------------|
| `MARKET` | Execute at best available price |
| `LIMIT` | Execute at your price or better |
| `STOP` | Trigger market order when stop price reached |
| `STOP_LIMIT` | Trigger limit order when stop price reached |
| `TRAILING_STOP` | Adjusting stop that follows price movement |
| `TRAILING_STOP_LIMIT` | Trailing stop that triggers limit order |
| `MARKET_ON_CLOSE` | Execute at closing price |
| `LIMIT_ON_CLOSE` | Limit order at close |
| `EXERCISE` | Exercise an option |
| `NET_DEBIT` | Options spread with net debit |
| `NET_CREDIT` | Options spread with net credit |
| `NET_ZERO` | Options spread with no net cost |
| `CABINET` | Cabinet trade |
| `NON_MARKETABLE` | Non-marketable order |

### Special Instructions

| Instruction | Description |
|-------------|-------------|
| `ALL_OR_NONE` | Don't allow partial fills |
| `DO_NOT_REDUCE` | Don't adjust for dividends |
| `ALL_OR_NONE_DO_NOT_REDUCE` | Both of the above |

### Equity Instructions

| Instruction | Description |
|-------------|-------------|
| `BUY` | Open long position |
| `SELL` | Close long position |
| `SELL_SHORT` | Open short position |
| `BUY_TO_COVER` | Close short position |

### Option Instructions

| Instruction | Description |
|-------------|-------------|
| `BUY_TO_OPEN` | Enter new long position |
| `SELL_TO_CLOSE` | Exit existing long position |
| `SELL_TO_OPEN` | Enter new short position |
| `BUY_TO_CLOSE` | Exit existing short position |

### Complex Order Strategy Types

| Strategy | Description |
|----------|-------------|
| `NONE` | No complex strategy (default) |
| `COVERED` | Covered call |
| `VERTICAL` | Vertical spread |
| `BACK_RATIO` | Ratio backspread |
| `CALENDAR` | Calendar spread |
| `DIAGONAL` | Diagonal spread |
| `STRADDLE` | Straddle spread |
| `STRANGLE` | Strangle spread |
| `BUTTERFLY` | Butterfly spread |
| `CONDOR` | Condor spread |
| `IRON_CONDOR` | Iron condor |
| `COLLAR_WITH_STOCK` | Collar strategy |
| `DOUBLE_DIAGONAL` | Double diagonal |
| `VERTICAL_ROLL` | Roll a vertical |
| `UNBALANCED_BUTTERFLY` | Unbalanced butterfly |
| `UNBALANCED_CONDOR` | Unbalanced condor |
| `UNBALANCED_IRON_CONDOR` | Unbalanced iron condor |
| `UNBALANCED_VERTICAL_ROLL` | Unbalanced vertical roll |
| `MUTUAL_FUND_SWAP` | Mutual fund swap |
| `CUSTOM` | Custom multi-leg |

---

## Streaming Client

Real-time streaming data via WebSocket connection.

### Basic Setup

```python
from schwab.streaming import StreamClient
from schwab.auth import easy_client
import asyncio
import json

# Create HTTP client first
client = easy_client(
    api_key='YOUR_API_KEY',
    app_secret='YOUR_APP_SECRET',
    callback_url='https://127.0.0.1:8182/',
    token_path='/path/to/token.json'
)

# Create stream client
stream_client = StreamClient(
    client,
    account_id=1234567890,  # Optional
    enforce_enums=True,
    ssl_context=None  # Optional custom SSL context
)

async def read_stream():
    await stream_client.login()
    
    def print_message(message):
        print(json.dumps(message, indent=4))
    
    # Add handler BEFORE subscribing
    stream_client.add_level_one_equity_handler(print_message)
    await stream_client.level_one_equity_subs(['AAPL', 'MSFT'])
    
    while True:
        await stream_client.handle_message()

asyncio.run(read_stream())
```

### Custom JSON Decoder

```python
from schwab.streaming import StreamJsonDecoder

class MyJsonDecoder(StreamJsonDecoder):
    def decode_json_string(self, raw):
        # Custom parsing logic
        return json.loads(raw)

stream_client.set_json_decoder(MyJsonDecoder())
```

### Level One Data

#### Equities

```python
from schwab.streaming import StreamClient

# Subscribe
await stream_client.level_one_equity_subs(
    symbols=['AAPL', 'MSFT'],
    fields=[
        StreamClient.LevelOneEquityFields.SYMBOL,
        StreamClient.LevelOneEquityFields.BID_PRICE,
        StreamClient.LevelOneEquityFields.ASK_PRICE,
        StreamClient.LevelOneEquityFields.LAST_PRICE,
        StreamClient.LevelOneEquityFields.TOTAL_VOLUME
    ]
)

# Add more symbols
await stream_client.level_one_equity_add(['GOOGL'])

# Unsubscribe
await stream_client.level_one_equity_unsubs(['AAPL'])

# Register handler
stream_client.add_level_one_equity_handler(handler_func)
```

**Level One Equity Fields:**

| Field | Description |
|-------|-------------|
| `SYMBOL` | Ticker symbol |
| `BID_PRICE` | Highest bid |
| `ASK_PRICE` | Lowest ask |
| `LAST_PRICE` | Last trade price |
| `BID_SIZE` | Size of highest bid |
| `ASK_SIZE` | Size of lowest ask |
| `TOTAL_VOLUME` | Total volume traded |
| `LAST_SIZE` | Size of last trade |
| `HIGH_PRICE` | Daily high |
| `LOW_PRICE` | Daily low |
| `CLOSE_PRICE` | Previous close |
| `OPEN_PRICE` | Today's open |
| `NET_CHANGE` | Net change |
| `NET_CHANGE_PERCENT` | Net change percent |
| `HIGH_PRICE_52_WEEK` | 52-week high |
| `LOW_PRICE_52_WEEK` | 52-week low |
| `PE_RATIO` | P/E ratio |
| `DIVIDEND_AMOUNT` | Dividend amount |
| `DIVIDEND_YIELD` | Dividend yield |
| `DIVIDEND_DATE` | Dividend date |
| `NAV` | ETF net asset value |
| `EXCHANGE_NAME` | Exchange name |
| `MARGINABLE` | Is marginable? |
| `MARK` | Mark price |
| `MARK_CHANGE` | Mark change |
| `MARK_CHANGE_PERCENT` | Mark change percent |
| `QUOTE_TIME_MILLIS` | Quote timestamp |
| `TRADE_TIME_MILLIS` | Trade timestamp |
| `SECURITY_STATUS` | Security status |
| `HARD_TO_BORROW` | Hard to borrow? |
| `IS_SHORTABLE` | Is shortable? |
| `HTB_QUANTITY` | HTB quantity |
| `HTB_RATE` | HTB rate |
| `POST_MARKET_NET_CHANGE` | Post-market change |
| `POST_MARKET_NET_CHANGE_PERCENT` | Post-market change % |

#### Options

```python
await stream_client.level_one_option_subs(
    symbols=['AAPL  240420C00150000'],
    fields=[
        StreamClient.LevelOneOptionFields.SYMBOL,
        StreamClient.LevelOneOptionFields.BID_PRICE,
        StreamClient.LevelOneOptionFields.ASK_PRICE,
        StreamClient.LevelOneOptionFields.DELTA,
        StreamClient.LevelOneOptionFields.GAMMA,
        StreamClient.LevelOneOptionFields.THETA,
        StreamClient.LevelOneOptionFields.VEGA,
        StreamClient.LevelOneOptionFields.IMPLIED_YIELD
    ]
)
stream_client.add_level_one_option_handler(handler_func)
```

**Level One Option Fields:**
- Basic: `SYMBOL`, `DESCRIPTION`, `BID_PRICE`, `ASK_PRICE`, `LAST_PRICE`, `HIGH_PRICE`, `LOW_PRICE`, `CLOSE_PRICE`
- Volume: `TOTAL_VOLUME`, `OPEN_INTEREST`, `BID_SIZE`, `ASK_SIZE`, `LAST_SIZE`
- Greeks: `DELTA`, `GAMMA`, `THETA`, `VEGA`, `RHO`, `VOLATILITY`
- Contract: `UNDERLYING`, `STRIKE_TYPE`, `CONTRACT_TYPE`, `EXPIRATION_YEAR`, `EXPIRATION_MONTH`, `EXPIRATION_DAY`, `DAYS_TO_EXPIRATION`, `MULTIPLIER`, `EXERCISE_TYPE`, `SETTLEMENT_TYPE`
- Pricing: `MARK`, `NET_CHANGE`, `NET_PERCENT_CHANGE`, `MARK_CHANGE`, `MARK_CHANGE_PERCENT`, `THEORETICAL_OPTION_VALUE`, `TIME_VALUE`, `MONEY_INTRINSIC_VALUE`, `UNDERLYING_PRICE`
- Other: `DELIVERABLES`, `OPTION_ROOT`, `HIGH_PRICE_52_WEEK`, `LOW_PRICE_52_WEEK`, `IS_PENNY`, `INDICATIVE_ASKING_PRICE`, `INDICATIVE_BID_PRICE`

#### Futures

```python
await stream_client.level_one_futures_subs(
    symbols=['/ES', '/NQ'],
    fields=[
        StreamClient.LevelOneFuturesFields.SYMBOL,
        StreamClient.LevelOneFuturesFields.BID_PRICE,
        StreamClient.LevelOneFuturesFields.ASK_PRICE,
        StreamClient.LevelOneFuturesFields.LAST_PRICE
    ]
)
stream_client.add_level_one_futures_handler(handler_func)
```

**Level One Futures Fields:**
- Quote: `SYMBOL`, `BID_PRICE`, `ASK_PRICE`, `LAST_PRICE`, `BID_SIZE`, `ASK_SIZE`, `LAST_SIZE`
- Times: `QUOTE_TIME_MILLIS`, `TRADE_TIME_MILLIS`, `ASK_TIME_MILLIS`, `BID_TIME_MILLIS`
- Prices: `HIGH_PRICE`, `LOW_PRICE`, `CLOSE_PRICE`, `OPEN_PRICE`, `MARK`
- Contract: `DESCRIPTION`, `PRODUCT`, `FUTURE_MULTIPLIER`, `FUTURE_EXPIRATION_DATE`, `FUTURE_SETTLEMENT_PRICE`, `FUTURE_ACTIVE_SYMBOL`, `FUTURE_IS_ACTIVE`, `FUTURE_IS_TRADABLE`
- Other: `TOTAL_VOLUME`, `OPEN_INTEREST`, `NET_CHANGE`, `FUTURE_CHANGE_PERCENT`, `TICK`, `TICK_AMOUNT`, `EXCHANGE_ID`, `EXCHANGE_NAME`, `SECURITY_STATUS`, `FUTURE_PRICE_FORMAT`, `FUTURE_TRADING_HOURS`, `EXPIRATION_STYLE`, `SETTLEMENT_DATE`, `QUOTED_IN_SESSION`

#### Forex

```python
await stream_client.level_one_forex_subs(
    symbols=['EUR/USD', 'GBP/USD'],
    fields=None  # None = all fields
)
stream_client.add_level_one_forex_handler(handler_func)
```

**Level One Forex Fields:**
- `SYMBOL`, `BID_PRICE`, `ASK_PRICE`, `LAST_PRICE`, `BID_SIZE`, `ASK_SIZE`, `TOTAL_VOLUME`, `LAST_SIZE`
- `QUOTE_TIME_MILLIS`, `TRADE_TIME_MILLIS`
- `HIGH_PRICE`, `LOW_PRICE`, `CLOSE_PRICE`, `OPEN_PRICE`, `MARK`
- `NET_CHANGE`, `CHANGE_PERCENT`, `HIGH_PRICE_52_WEEK`, `LOW_PRICE_52_WEEK`
- `DESCRIPTION`, `EXCHANGE_ID`, `EXCHANGE_NAME`, `DIGITS`, `TICK`, `TICK_AMOUNT`, `PRODUCT`, `TRADING_HOURS`, `IS_TRADABLE`, `MARKET_MAKER`, `SECURITY_STATUS`

#### Futures Options

```python
await stream_client.level_one_futures_options_subs(
    symbols=['./ESH24C5000'],
    fields=None
)
stream_client.add_level_one_futures_options_handler(handler_func)
```

### Level Two Order Books

```python
# NYSE Book
await stream_client.nyse_book_subs(['IBM', 'GE'])
await stream_client.nyse_book_unsubs(['IBM'])
await stream_client.nyse_book_add(['F'])
stream_client.add_nyse_book_handler(handler_func)

# NASDAQ Book
await stream_client.nasdaq_book_subs(['AAPL', 'MSFT'])
await stream_client.nasdaq_book_unsubs(['AAPL'])
await stream_client.nasdaq_book_add(['GOOGL'])
stream_client.add_nasdaq_book_handler(handler_func)

# Options Book
await stream_client.options_book_subs(['AAPL  240420C00150000'])
await stream_client.options_book_unsubs(['AAPL  240420C00150000'])
await stream_client.options_book_add(['MSFT  240420C00400000'])
stream_client.add_options_book_handler(handler_func)
```

**Book Fields:**
- `SYMBOL`, `BOOK_TIME`, `BIDS`, `ASKS`

**Bid/Ask Fields:**
- `BID_PRICE`/`ASK_PRICE`, `TOTAL_VOLUME`, `NUM_BIDS`/`NUM_ASKS`, `BIDS`/`ASKS`

**Per-Exchange Bid/Ask Fields:**
- `EXCHANGE`, `BID_VOLUME`/`ASK_VOLUME`, `SEQUENCE`

### Charts

#### Equity Charts

```python
await stream_client.chart_equity_subs(['AAPL', 'MSFT'])
await stream_client.chart_equity_unsubs(['AAPL'])
await stream_client.chart_equity_add(['GOOGL'])
stream_client.add_chart_equity_handler(handler_func)
```

**Chart Equity Fields:**
- `SYMBOL`, `SEQUENCE`
- `OPEN_PRICE`, `HIGH_PRICE`, `LOW_PRICE`, `CLOSE_PRICE`
- `VOLUME`, `CHART_TIME_MILLIS`, `CHART_DAY`

#### Futures Charts

```python
await stream_client.chart_futures_subs(['/ES', '/NQ'])
await stream_client.chart_futures_unsubs(['/ES'])
await stream_client.chart_futures_add(['/CL'])
stream_client.add_chart_futures_handler(handler_func)
```

**Chart Futures Fields:**
- `SYMBOL`, `CHART_TIME_MILLIS`
- `OPEN_PRICE`, `HIGH_PRICE`, `LOW_PRICE`, `CLOSE_PRICE`
- `VOLUME`

### Account Activity

```python
# Subscribe to account activity
await stream_client.account_activity_sub()
await stream_client.account_activity_unsubs()
stream_client.add_account_activity_handler(handler_func)
```

**Account Activity Fields:**
- `SUBSCRIPTION_KEY` - Identifies the subscription
- `ACCOUNT` - Account number
- `MESSAGE_TYPE` - Dictates message data format
- `MESSAGE_DATA` - JSON data, NULL, or error text

### Screeners

```python
# Equity screener
await stream_client.screener_equity_subs(['$COMPX', '$SPX'])
await stream_client.screener_equity_unsubs(['$COMPX'])
await stream_client.screener_equity_add(['$DJI'])
stream_client.add_screener_equity_handler(handler_func)

# Option screener
await stream_client.screener_option_subs(['OPTION_PUT', 'OPTION_CALL'])
await stream_client.screener_option_unsubs(['OPTION_PUT'])
await stream_client.screener_option_add(['OPTION_ALL'])
stream_client.add_screener_option_handler(handler_func)
```

**Screener Fields:**
- `SYMBOL` - Symbol for lookup
- `TIMESTAMP` - Market snapshot timestamp (ms since epoch)
- `SORT_FIELD` - Field used for sorting
- `FREQUENCY` - Data frequency
- `ITEMS` - Array of screener items

### Login/Logout

```python
# Login with custom websocket args
await stream_client.login(websocket_connect_args={
    'ping_timeout': 30,
    'close_timeout': 10
})

# Logout (closes connection)
await stream_client.logout()
```

---

## Utilities

### Extract Order ID

```python
from schwab.utils import Utils

utils = Utils(client, account_hash='ABC123...')

# Place order and extract ID
response = client.place_order(account_hash, order_spec)
order_id = utils.extract_order_id(response)

# Change account hash
utils.set_account_hash('NEW_HASH...')
```

---

## Enumerations Reference

### Quick Reference Table

| Category | Location | Values |
|----------|----------|--------|
| Account Fields | `Client.Account.Fields` | `POSITIONS` |
| Order Status | `Client.Order.Status` | See [Orders](#orders) section |
| Quote Fields | `Client.Quote.Fields` | `QUOTE`, `FUNDAMENTAL`, `EXTENDED`, `REFERENCE`, `REGULAR` |
| Option Contract Type | `Client.Options.ContractType` | `CALL`, `PUT`, `ALL` |
| Option Strategy | `Client.Options.Strategy` | `SINGLE`, `ANALYTICAL`, `COVERED`, `VERTICAL`, `CALENDAR`, `STRANGLE`, `STRADDLE`, `BUTTERFLY`, `CONDOR`, `DIAGONAL`, `COLLAR`, `ROLL` |
| Option Strike Range | `Client.Options.StrikeRange` | `ITM`, `NTM`, `OTM`, `SAK`, `SBK`, `SNK`, `ALL` |
| Option Type | `Client.Options.Type` | `S`, `NS`, `ALL` |
| Option Expiration | `Client.Options.ExpirationMonth` | `JAN`-`DEC`, `ALL` |
| Option Entitlement | `Client.Options.Entitlement` | `PP`, `NP`, `PN` |
| Price History Period Type | `Client.PriceHistory.PeriodType` | `DAY`, `MONTH`, `YEAR`, `YTD` |
| Price History Period | `Client.PriceHistory.Period` | Various (see section) |
| Price History Frequency Type | `Client.PriceHistory.FrequencyType` | `MINUTE`, `DAILY`, `WEEKLY`, `MONTHLY` |
| Price History Frequency | `Client.PriceHistory.Frequency` | Various (see section) |
| Movers Index | `Client.Movers.Index` | `$DJI`, `$COMPX`, `$SPX`, `NYSE`, `NASDAQ`, etc. |
| Movers Sort | `Client.Movers.SortOrder` | `VOLUME`, `TRADES`, `PERCENT_CHANGE_UP`, `PERCENT_CHANGE_DOWN` |
| Movers Frequency | `Client.Movers.Frequency` | `0`, `1`, `5`, `10`, `30`, `60` |
| Market Hours | `Client.MarketHours.Market` | `EQUITY`, `OPTION`, `BOND`, `FUTURE`, `FOREX` |
| Instrument Projection | `Client.Instrument.Projection` | `SYMBOL_SEARCH`, `SYMBOL_REGEX`, `DESCRIPTION_SEARCH`, `DESCRIPTION_REGEX`, `SEARCH`, `FUNDAMENTAL` |
| Transaction Type | `Client.Transactions.TransactionType` | Various (see section) |
| Session | `Session` | `NORMAL`, `AM`, `PM`, `SEAMLESS` |
| Duration | `Duration` | `DAY`, `GOOD_TILL_CANCEL`, `FILL_OR_KILL`, etc. |
| Order Type | `OrderType` | `MARKET`, `LIMIT`, `STOP`, `STOP_LIMIT`, etc. |
| Order Strategy Type | `OrderStrategyType` | `SINGLE`, `OCO`, `TRIGGER`, etc. |
| Equity Instruction | `EquityInstruction` | `BUY`, `SELL`, `SELL_SHORT`, `BUY_TO_COVER` |
| Option Instruction | `OptionInstruction` | `BUY_TO_OPEN`, `SELL_TO_CLOSE`, `SELL_TO_OPEN`, `BUY_TO_CLOSE` |
| Complex Order Strategy | `ComplexOrderStrategyType` | Various (see section) |
| Special Instruction | `SpecialInstruction` | `ALL_OR_NONE`, `DO_NOT_REDUCE`, `ALL_OR_NONE_DO_NOT_REDUCE` |
| Destination | `Destination` | `INET`, `ECN_ARCA`, `CBOE`, `AMEX`, `PHLX`, `ISE`, `BOX`, `NYSE`, `NASDAQ`, `BATS`, `C2`, `AUTO` |

---

## Error Handling

```python
import httpx

try:
    response = c.get_quote('INVALID_SYMBOL')
    response.raise_for_status()
    data = response.json()
except httpx.HTTPStatusError as e:
    print(f"HTTP error: {e.response.status_code}")
except Exception as e:
    print(f"Error: {e}")
```

### Streaming Errors

```python
from schwab.streaming import (
    UnexpectedResponse,
    UnexpectedResponseCode,
    UnparsableMessage
)

try:
    await stream_client.handle_message()
except UnexpectedResponse as e:
    print(f"Unexpected response: {e.response}")
except UnexpectedResponseCode as e:
    print(f"Unexpected response code: {e.response}")
except UnparsableMessage as e:
    print(f"Could not parse message: {e.raw_msg}")
    print(f"Parse error: {e.json_parse_exception}")
```

---

## API Endpoints Reference

| Endpoint | HTTP Method | Client Method |
|----------|-------------|---------------|
| `/trader/v1/accounts/accountNumbers` | GET | `get_account_numbers()` |
| `/trader/v1/accounts/{accountHash}` | GET | `get_account()` |
| `/trader/v1/accounts` | GET | `get_accounts()` |
| `/trader/v1/accounts/{accountHash}/orders` | GET | `get_orders_for_account()` |
| `/trader/v1/accounts/{accountHash}/orders` | POST | `place_order()` |
| `/trader/v1/accounts/{accountHash}/orders/{orderId}` | GET | `get_order()` |
| `/trader/v1/accounts/{accountHash}/orders/{orderId}` | PUT | `replace_order()` |
| `/trader/v1/accounts/{accountHash}/orders/{orderId}` | DELETE | `cancel_order()` |
| `/trader/v1/accounts/{accountHash}/previewOrder` | POST | `preview_order()` |
| `/trader/v1/orders` | GET | `get_orders_for_all_linked_accounts()` |
| `/trader/v1/accounts/{accountHash}/transactions` | GET | `get_transactions()` |
| `/trader/v1/accounts/{accountHash}/transactions/{transactionId}` | GET | `get_transaction()` |
| `/trader/v1/userPreference` | GET | `get_user_preferences()` |
| `/marketdata/v1/{symbol}/quotes` | GET | `get_quote()` |
| `/marketdata/v1/quotes` | GET | `get_quotes()` |
| `/marketdata/v1/chains` | GET | `get_option_chain()` |
| `/marketdata/v1/expirationchain` | GET | `get_option_expiration_chain()` |
| `/marketdata/v1/pricehistory` | GET | `get_price_history()` |
| `/marketdata/v1/movers/{index}` | GET | `get_movers()` |
| `/marketdata/v1/markets` | GET | `get_market_hours()` |
| `/marketdata/v1/instruments` | GET | `get_instruments()` |
| `/marketdata/v1/instruments/{cusip}` | GET | `get_instrument_by_cusip()` |

---

## Resources

- **Official Documentation**: [developer.schwab.com](https://developer.schwab.com/)
- **schwab-py Documentation**: [schwab-py.readthedocs.io](https://schwab-py.readthedocs.io/)
- **GitHub Repository**: [github.com/alexgolec/schwab-py](https://github.com/alexgolec/schwab-py)
- **Discord Community**: [discord.gg/M3vjtHj](https://discord.gg/M3vjtHj)
- **Patreon**: [patreon.com/schwabpy](https://www.patreon.com/schwabpy)

---

*Last updated: November 2025*
