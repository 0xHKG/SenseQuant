API Usage
from breeze_connect import BreezeConnect

# Initialize SDK
breeze = BreezeConnect(api_key="your_api_key")

# Obtain your session key from https://api.icicidirect.com/apiuser/login?api_key=YOUR_API_KEY
# Incase your api-key has special characters(like +,=,!) then encode the api key before using in the url as shown below.
import urllib
print("https://api.icicidirect.com/apiuser/login?api_key="+urllib.parse.quote_plus("your_api_key"))

# Generate Session
breeze.generate_session(api_secret="your_secret_key",
                        session_token="your_api_session")

# Generate ISO8601 Date/DateTime String
import datetime
iso_date_string = datetime.datetime.strptime("28/02/2021","%d/%m/%Y").isoformat()[:10] + 'T05:30:00.000Z'
iso_date_time_string = datetime.datetime.strptime("28/02/2021 23:59:59","%d/%m/%Y %H:%M:%S").isoformat()[:19] + '.000Z'

Websocket Usage
from breeze_connect import BreezeConnect

# Initialize SDK
breeze = BreezeConnect(api_key="your_api_key")

# Obtain your session key from https://api.icicidirect.com/apiuser/login?api_key=YOUR_API_KEY
# Incase your api-key has special characters(like +,=,!) then encode the api key before using in the url as shown below.
import urllib
print("https://api.icicidirect.com/apiuser/login?api_key="+urllib.parse.quote_plus("your_api_key"))

# Generate Session
breeze.generate_session(api_secret="your_secret_key",
                        session_token="your_api_session")

# Connect to websocket(it will connect to tick-by-tick data server)
breeze.ws_connect()

# Callback to receive ticks.
def on_ticks(ticks):
    print("Ticks: {}".format(ticks))

# Assign the callbacks.
breeze.on_ticks = on_ticks

# ws_disconnect (it will disconnect from all actively connected servers)
breeze.ws_disconnect()
Subscribing to Real Time Streaming OHLCV Data of stocks by stock-token
breeze.subscribe_feeds(stock_token="4.1!2885", 
                      interval="1minute")

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(stock_token="4.1!2885", interval="1minute")

Subscribe equity stocks by stock-token (Exchange Quotes)
    breeze.subscribe_feeds(stock_token="4.1!2885")

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(stock_token="4.1!2885")

Subscribe to Real Time Streaming of NSE stock
breeze.subscribe_feeds(exchange_code="NSE",
                        stock_code="NIFTY",
                        product_type="cash",
                        get_market_depth=False,
                        get_exchange_quotes=True)

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(exchange_code="NSE", stock_code="NIFTY", product_type="cash", get_market_depth=False, get_exchange_quotes=True)

Subscribe to Real Time Streaming OHLCV Data of NFO stocks
breeze.subscribe_feeds(exchange_code= "NFO", 
                  stock_code="NIFTY", 
                  expiry_date="13-Feb-2025", 
                  strike_price="23550", 
                  right="call", 
                  product_type="options", 
                  get_market_depth=False ,
                  get_exchange_quotes=True,
                  interval="1minute")

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(exchange_code= "NFO", stock_code="NIFTY", expiry_date="13-Feb-2025", strike_price="23550", right="call", product_type="options", get_market_depth=False , get_exchange_quotes=True, interval="1minute")

Subscribe stocks feeds (NFO Exchange Quotes)
breeze.subscribe_feeds(exchange_code= "NFO", 
                  stock_code="NIFTY", 
                  expiry_date="13-Feb-2025", 
                  strike_price="23550", 
                  right="call", 
                  product_type="options", 
                  get_market_depth=False ,
                  get_exchange_quotes=True)

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(exchange_code= "NFO", stock_code="NIFTY", expiry_date="13-Feb-2025", strike_price="23550", right="call", product_type="options", get_market_depth=False , get_exchange_quotes=True)

Subscribe stocks feeds (NFO Market Depth)
breeze.subscribe_feeds(exchange_code= "NFO", 
                  stock_code="NIFTY", 
                  expiry_date="13-Feb-2025", 
                  strike_price="23550", 
                  right="call", 
                  product_type="options", 
                  get_market_depth=True ,
                  get_exchange_quotes=False)

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(exchange_code= "NFO", stock_code="NIFTY", expiry_date="13-Feb-2025", strike_price="23550", right="call", product_type="options", get_market_depth=True , get_exchange_quotes=False)

Subscribe to Real Time Streaming OHLCV Data of BFO stocks
breeze.subscribe_feeds(exchange_code= "BFO", 
                  stock_code="BSESEN", 
                  expiry_date="18-Feb-2025", 
                  strike_price="78200", 
                  right="call", 
                  product_type="options", 
                  get_market_depth=False,
                  get_exchange_quotes=True,
                  interval="1minute")

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(exchange_code= "BFO", stock_code="BSESEN", expiry_date="18-Feb-2025", strike_price="78200", right="call", product_type="options", get_market_depth=False, get_exchange_quotes=True, interval="1minute)

Subscribe stocks feeds (BFO Exchange Quotes)
breeze.subscribe_feeds(exchange_code= "BFO", 
                  stock_code="BSESEN", 
                  expiry_date="18-Feb-2025", 
                  strike_price="78200", 
                  right="call", 
                  product_type="options", 
                  get_market_depth=False,
                  get_exchange_quotes=True)

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(exchange_code= "BFO", stock_code="BSESEN", expiry_date="18-Feb-2025", strike_price="78200", right="call", product_type="options", get_market_depth=False, get_exchange_quotes=True)

Subscribe stocks feeds (BFO Market Depth)
breeze.subscribe_feeds(exchange_code= "BFO", 
                  stock_code="BSESEN", 
                  expiry_date="18-Feb-2025", 
                  strike_price="78200", 
                  right="call", 
                  product_type="options", 
                  get_market_depth=True ,
                  get_exchange_quotes=False )

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(exchange_code= "BFO", stock_code="BSESEN", expiry_date="18-Feb-2025", strike_price="78200", right="call", product_type="options", get_market_depth=True , get_exchange_quotes=False )

Subscribe oneclick strategy stream
breeze.subscribe_feeds(stock_token="one_click_fno")

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(stock_token = "one_click_fno")

Subscribe oneclick equity strategy stream(i_click_2_gain)
breeze.subscribe_feeds(stock_token="i_click_2_gain") 

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(stock_token = "i_click_2_gain")

Subscribe to multiple stock tokens
breeze.subscribe_feeds(stock_token=['4.1!3499','4.1!2885'])

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(stock_token=['4.1!3499','4.1!2885'])

Subscribe to order notifications
breeze.subscribe_feeds(get_order_notification=True)

View Response
NOTE :
For unsubscribe : breeze.unsubscribe_feeds(get_order_notification=True)


ADDITIONAL NOTES
Examples for stock_token are "4.1!38071" or "1.1!500780".
Template for stock_token : X.Y!
X : exchange code
Y : Market Level data
Token : ISEC stock code
Value of X can be :
1 for BSE(equity)
2 for BFO OHLC Data
4 for NSE
4 for NFO
8 for BFO live Data
Value of Y can be :
1 for Exchange Quote data
2 for Market Depth data
Token number can be obtained via get_names() function or downloading master security file via https://api.icicidirect.com/breezeapi/documents/index.html#instruments
Exchange_code must be
BSE
NSE
BFO
NFO
Stock Code Validation: The stock_code field cannot be left empty. Valid examples include "WIPRO" or "ZEEENT".
Product_type Requirements: Acceptable values are 'Futures', 'Options', or a non-empty string. For exchanges NFO, this field must not be left empty.
Expiry_date Format: Should be in DD-MMM-YYYY format (e.g., 01-Jan-2022), and cannot be empty for NFO exchanges.
Strike_price Format: Must be a float value represented as a string or remain empty. For Options under product_type, this field must not be empty.
Right Field Requirements: Acceptable values are 'Put', 'Call', or an empty string. For Options, this field cannot be left empty.
get_exchange_quotes and get_market_depth Validation: At least one must be set to True. Both can be True, but both cannot be False.
OHLCV Streaming Interval: The interval field cannot be empty and must be one of the following values: "1second", "1minute", "5minute", or "30minute".

List of other APIs:
Index
get_customer_details
get_demat_holdings
get_funds
set_funds
get_historical_data
get_historical_data_v2
get_margin
place_order
order_detail
order_list
cancel_order
modify_order
get_portfolio_holding
get_portfolio_position
get_quotes
get_option_chain_quotes
square_off
get_trade_list
get_trade_detail
get_names
preview_order
limit_calculator
margin_calculator
gtt_three_leg_place_order
gtt_three_leg_modify_order
gtt_three_leg_cancel_order
gtt_single_leg_place_order
gtt_single_leg_modify_order
gtt_single_leg_cancel_order
gtt_order_book

Get Customer details by api-session value
breeze.get_customer_details(api_session="your_api_session") 

View API Response

Back to Index
Get Demat Holding details
breeze.get_demat_holdings()

View API Response

Back to Index
Get Funds
breeze.get_funds()

View API Response

Back to Index
Set Funds
breeze.set_funds(transaction_type="debit", 
                    amount="200",
                    segment="Equity")

View API Response
NOTE:

For adding fund, transaction_type="credit", amount="200", segment="Equity".
Segment can be Equity, FNO, Commodity
Back to Index
Historical Data : Futures
breeze.get_historical_data(interval="1minute",
                  from_date= "2025-02-03T09:21:00.000Z",
                  to_date= "2025-02-03T09:21:00.000Z",
                  stock_code="NIFTY",
                  exchange_code="NFO",
                  product_type="futures",
                  expiry_date="2025-02-27T07:00:00.000Z",
                  right="others",
                  strike_price="0")                    

View API Response

Back to Index
Historical Data : Equity
breeze.get_historical_data(interval="1minute",
                  from_date= "2025-02-03T09:20:00.000Z",
                  to_date= "2025-02-03T09:22:00.000Z",
                  stock_code="RELIND",
                  exchange_code="NSE",
                  product_type="cash")

View API Response

Back to Index
Historical Data : Options
breeze.get_historical_data(interval="1minute",
                  from_date= "2025-02-03T09:20:00.000Z",
                  to_date= "2025-02-03T09:22:00.000Z",
                  stock_code="NIFTY",
                  exchange_code="NFO",
                  product_type="options",
                  expiry_date="2025-02-06T07:00:00.000Z",
                  right="call",
                  strike_price="23200")

View API Response
NOTE:
Get Historical Data for specific stock-code by mentioned interval either as "1minute", "5minute", "30minute" or as "1day"

Back to Index
Historical Data V2 : FUTURES
breeze.get_historical_data_v2(interval="1minute",
                  from_date= "2025-02-03T09:21:00.000Z",
                  to_date= "2025-02-03T09:21:00.000Z",
                  stock_code="NIFTY",
                  exchange_code="NFO",
                  product_type="futures",
                  expiry_date="2025-02-27T07:00:00.000Z",
                  right="others",
                  strike_price="0")                      

View API Response
NOTE:
Product Type historical data v2 should be "futures", "options","cash"
Interval should be "1minute", "5minute", "30minute" or "1day"
Back to Index

Histroical Data V2 : EQUITY
breeze.get_historical_data_v2(interval="1minute",
                    from_date= "2025-02-03T09:20:00.000Z",
                    to_date= "2025-02-03T09:22:00.000Z",
                    stock_code="RELIND",
                    exchange_code="NSE",
                    product_type="cash")

View API Response

Back to Index
Histroical Data V2 : OPTIONS
breeze.get_historical_data_v2(interval="1minute",
                    from_date= "2025-02-03T09:20:00.000Z",
                    to_date= "2025-02-03T09:21:00.000Z",
                    stock_code="NIFTY",
                    exchange_code="NFO",
                    product_type="options",
                    expiry_date="2025-02-06T07:00:00.000Z",
                    right="call",
                    strike_price="23200")

View API Response
NOTE:

Get Historical Data (version 2) for specific stock-code by mentioning interval either as "1second","1minute", "5minute", "30minute" or as "1day".
Maximum candle intervals in one single request is 1000

Back to Index
Get Margin of your account.
breeze.get_margin(exchange_code="NSE")

View API Response
NOTE:

Please change exchange_code=“NFO” to get F&O margin details

Back to Index
Place Order : FUTURES
breeze.place_order(stock_code="NIFTY",
                  exchange_code="NFO",
                  product="futures",
                  action="buy",
                  order_type="limit",
                  stoploss="0",
                  quantity="75",
                  price="23700",
                  validity="day",
                  validity_date="2022-08-22T06:00:00.000Z",
                  disclosed_quantity="0",
                  expiry_date="2025-02-27T06:00:00.000Z",
                  right="others",
                  strike_price="0",
                  user_remark="Test")

View API Response
NOTE:
Order Type should be either "limit" or "market"
Back to Index
Place Order : OPTIONS
breeze.place_order(stock_code="NIFTY",
                  exchange_code="NFO",
                  product="options",
                  action="buy",
                  order_type="limit",
                  stoploss="",
                  quantity="75",
                  price="0.20",
                  validity="day",
                  validity_date="2025-02-05T06:00:00.000Z",
                  disclosed_quantity="0",
                  expiry_date="2025-02-27T06:00:00.000Z",
                  right="call",
                  strike_price="24800")

View API Response
NOTE:
Order Type should be either "limit" or "market"

Back to Index
Place Order : EQUITY
breeze.place_order(stock_code="ITC",
                    exchange_code="NSE",
                    product="cash",
                    action="buy",
                    order_type="limit",
                    stoploss="",
                    quantity="1",
                    price="420",
                    validity="day"
                )

View API Response

Back to Index
Get order detail
breeze.get_order_detail(exchange_code="NSE",
                        order_id="20250205N300001234")

View API Response
NOTE:

Please change exchange_code=“NFO” to get details about F&O
Back to Index
Get order list
breeze.get_order_list(exchange_code="NSE",
                      from_date="2025-02-05T10:00:00.000Z",
                      to_date="2025-02-05T10:00:00.000Z")

View API Response
NOTE:

Please change exchange_code=“NFO” to get details about F&O
Back to Index

Cancel order
breeze.cancel_order(exchange_code="NSE",
                    order_id="20250205N300001234")

View API Response

Back to Index
Modify order
breeze.modify_order(order_id="202502051400012345",
                    exchange_code="NFO",
                    order_type="limit",
                    stoploss="0",
                    quantity="75",
                    price="0.30",
                    validity="day",
                    disclosed_quantity="0",
                    validity_date="2025-08-22T06:00:00.000Z")

View API Response

Back to Index
Get Portfolio Holdings
breeze.get_portfolio_holdings(exchange_code="NFO",
                    from_date="2024-08-01T06:00:00.000Z", 
                    to_date="2024-09-19T06:00:00.000Z", 
                    stock_code="", 
                    portfolio_type="")

View API Response
NOTE:

Please change exchange_code=“NSE” to get Equity Portfolio Holdings
Back to Index
Get Portfolio Positions
breeze.get_portfolio_positions()

View API Response

Back to Index
Get quotes
breeze.get_quotes(stock_code="NIFTY",
                    exchange_code="NFO",
                    expiry_date="2025-02-27T06:00:00.000Z",
                    product_type="futures",
                    right="others",
                    strike_price="0")

View API Response
NOTE:
For equity, exchange_code = "NSE", expiry_date = "", product_type = "cash", right="", strike_price=""
For options, exchange_code = "NFO", expiry_date = "27-Feb-2025", product_type = "options", right="call/put", strike_price="24000"
Back to Index
Get option chain quotes
breeze.get_option_chain_quotes(stock_code="ICIBAN",
                    exchange_code="NFO",
                    product_type="futures",
                    expiry_date="2025-01-25T06:00:00.000Z")

View API Response
NOTE:
Get option-chain of mentioned stock-code for product-type Options where atleast 2 input is required out of expiry-date, right and strike-price
Get option-chain of mentioned stock-code for product-type Futures where input of expiry-date is not compulsory
Back to Index

Sqaure Off: OPTIONS
breeze.square_off(exchange_code="NFO",
                  product="options",
                  stock_code="NIFTY",
                  expiry_date="2025-02-27T06:00:00.000Z",
                  right="Call",
                  strike_price="24000",
                  action="sell",
                  order_type="market",
                  validity="day",
                  stoploss="0",
                  quantity="75",
                  price="0",
                  validity_date="2025-02-05T06:00:00.000Z",
                  trade_password="",
                  disclosed_quantity="0")

View API Response

Back to Index
Sqaure Off: FUTURES
breeze.square_off(exchange_code="NFO",
                  product="futures",
                  stock_code="NIFTY",
                  expiry_date="2025-02-27T06:00:00.000Z",
                  action="sell",
                  order_type="market",
                  validity="day",
                  stoploss="0",
                  quantity="75",
                  price="0",
                  validity_date="2025-02-27T06:00:00.000Z",
                  trade_password="",
                  disclosed_quantity="0")

View API Response

Back to Index
Get trade list
breeze.get_trade_list(from_date="2025-02-05T06:00:00.000Z",
                        to_date="2025-02-05T06:00:00.000Z",
                        exchange_code="NSE",
                        product_type="",
                        action="",
                        stock_code="")

View API Response
NOTE:

Please change exchange_code=“NFO” to get details about F&O
Back to Index
Get trade detail
breeze.get_trade_detail(exchange_code="NSE",
                        order_id="20250205N300012345")

View API Response
NOTE:

Please change exchange_code=“NFO” to get details about F&O
Back to Index
Get Names
breeze.get_names(exchange_code = 'NSE',stock_code = 'TATASTEEL')

View API Response
NOTE:

Use this method to find ICICI specific stock codes / token
Back to Index

Preview Order
breeze.preview_order(stock_code = "ITC",
            exchange_code = "NSE",
            product = "margin",
            order_type = "limit",
            price = "440",
            action = "buy",
            quantity = "1",
            specialflag = "N")

View API Response

Back to Index
Limit Calculator
breeze.limit_calculator(strike_price="24000",                                    
            product_type = "optionplus",                 
            expiry_date  = "06-Feb-2025",
            underlying = "NIFTY",
            exchange_code = "NFO",
            order_flow = "Buy",
            stop_loss_trigger = "8",
            option_type = "Call",
            source_flag = "P",
            limit_rate = "7.5",
            order_reference = "",
            available_quantity = "",
            market_type = "limit",
            fresh_order_limit = "10.95")

View API Response

Back to Index
Margin Calculator
breeze.margin_calculator([{
            "strike_price": "0",
            "quantity": "30",
            "right": "others",
            "product": "futures",
            "action": "buy",
            "price": "49500",
            "expiry_date": "27-Feb-2025",
            "stock_code": "CNXBAN",
            "cover_order_flow": "N",
            "fresh_order_type": "N",
            "cover_limit_rate": "0",
            "cover_sltp_price": "0",
            "fresh_limit_rate": "0",
            "open_quantity": "0"
        },
        {
            "strike_price": "50000",
            "quantity": "30",
            "right": "Call",
            "product": "options",
            "action": "buy",
            "price": "1150",
            "expiry_date": "27-Feb-2025",
            "stock_code": "CNXBAN",
            "cover_order_flow": "N",
            "fresh_order_type": "N",
            "cover_limit_rate": "0",
            "cover_sltp_price": "0",
            "fresh_limit_rate": "0",
            "open_quantity": "0"
        },
        {
            "strike_price": "0",
            "quantity": "75",
            "right": "others",
            "product": "futures",
            "action": "buy",
            "price": "23400",
            "expiry_date": "27-Feb-2025",
            "stock_code": "NIFTY",
            "cover_order_flow": "N",
            "fresh_order_type": "N",
            "cover_limit_rate": "0",
            "cover_sltp_price": "0",
            "fresh_limit_rate": "0",
            "open_quantity": "0"
        },
        {
            "strike_price": "23400",
            "quantity": "75",
            "right": "call",
            "product": "options",
            "action": "buy",
            "price": "577",
            "expiry_date": "27-Feb-2025",
            "stock_code": "NIFTY",
            "cover_order_flow": "sell",
            "fresh_order_type": "limit",
            "cover_limit_rate": "0",
            "cover_sltp_price": "0",
            "fresh_limit_rate": "0",
            "open_quantity": "0"
        }],exchange_code = "NFO")

View API Response

Back to Index
GTT(Good Till Trigger)
GTT Three Leg OCO(One Cancels Other) Place order
breeze.gtt_three_leg_place_order(exchange_code ="NFO",
                  stock_code="NIFTY",
                  product="options",
                  quantity = "75",
                  expiry_date="2025-02-06T06:00:00.00Z",
                  right = "call",
                  strike_price = "24000",
                  gtt_type="cover_oco",
                  fresh_order_action="buy",
                  fresh_order_price="8",
                  fresh_order_type="limit",
                  index_or_stock="index",
                  trade_date="2025-02-05T06:00:00.00Z",
                  order_details=[
                    {
                      "gtt_leg_type" : "target",
                      "action" : "sell",
                      "limit_price" : "15",
                      "trigger_price" : "14.50"
                    },
                    {
                      "gtt_leg_type" : "stoploss",
                      "action" : "sell",
                      "limit_price" : "7",
                      "trigger_price" : "7.5"
                    },
                    ])

View API Response

Back to Index
GTT Three Leg Modify order
breeze.gtt_three_leg_modify_order(exchange_code = "NFO",
                      gtt_order_id = "2025020500001234",
                      gtt_type ="oco",
                      order_details = [
                        {
                          "gtt_leg_type" : "target",
                          "action" : "sell",
                          "limit_price" : "12",
                          "trigger_price" : "11.50"
                        },
                        {
                          "gtt_leg_type" : "stoploss",
                          "action" : "sell",
                          "limit_price" : "4",
                          "trigger_price" : "5"
                        }])

View API Response

Back to Index
GTT Three Leg Cancel order
breeze.gtt_three_leg_cancel_order(exchange_code = "NFO",
                        gtt_order_id = "2025020500001234")

View API Response

Back to Index
GTT Single Leg Place order
breeze.gtt_single_leg_place_order(exchange_code ="NFO",
                    stock_code="NIFTY",
                    product="options",
                    quantity = "75",
                    expiry_date="2025-02-06T06:00:00.00Z",
                    right = "call",
                    strike_price = "24000",
                    gtt_type="single",
                    index_or_stock="index",
                    trade_date="2025-02-05T06:00:00.00Z",
                    order_details=[
                    {
                        "action" : "buy",
                        "limit_price" : "7",
                        "trigger_price" : "8"
                    }])

View API Response

Back to Index
GTT Single Leg Modify order
breeze.gtt_single_leg_modify_order(exchange_code="NFO",
                      gtt_order_id="2025020500001234",
                      gtt_type="single",
                      order_details=[
                        {
                          "action": "buy",
                          "limit_price": "6",
                          "trigger_price": "7"
                        }])

View API Response

Back to Index
GTT Single Leg Cancel order
breeze.gtt_single_leg_cancel_order(exchange_code = "NFO",
                                   gtt_order_id = "2025011500003608")

View API Response

Back to Index
OCO and Single GTT order book
breeze.gtt_order_book(exchange_code ="NFO",
            from_date = "2025-02-05T06:00:00.00Z",
            to_date = "2025-02-05T06:00:00.00Z")
