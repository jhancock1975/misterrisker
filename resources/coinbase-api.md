# 📚 Coinbase API Documentation
# 🌐 Source: https://coinbase.github.io/coinbase-advanced-py/
# 📅 Scraped on: 2025-11-30 08:19:29
# 📊 Total pages scraped: 16

## 📑 Table of Contents

- https://coinbase.github.io/coinbase-advanced-py/
- https://coinbase.github.io/coinbase-advanced-py/_modules/coinbase/jwt_generator.html
- https://coinbase.github.io/coinbase-advanced-py/_modules/coinbase/rest.html
- https://coinbase.github.io/coinbase-advanced-py/_modules/coinbase/websocket.html
- https://coinbase.github.io/coinbase-advanced-py/_modules/coinbase/websocket/websocket_base.html
- https://coinbase.github.io/coinbase-advanced-py/_modules/index.html
- https://coinbase.github.io/coinbase-advanced-py/_sources/coinbase.rest.rst.txt
- https://coinbase.github.io/coinbase-advanced-py/_sources/coinbase.websocket.rst.txt
- https://coinbase.github.io/coinbase-advanced-py/_sources/coinbase.websocket.user.rst.txt
- https://coinbase.github.io/coinbase-advanced-py/_sources/index.rst.txt
- https://coinbase.github.io/coinbase-advanced-py/_sources/jwt_generator.rst.txt
- https://coinbase.github.io/coinbase-advanced-py/coinbase.rest.html
- https://coinbase.github.io/coinbase-advanced-py/coinbase.websocket.html
- https://coinbase.github.io/coinbase-advanced-py/coinbase.websocket.user.html
- https://coinbase.github.io/coinbase-advanced-py/index.html
- https://coinbase.github.io/coinbase-advanced-py/jwt_generator.html

================================================================================


================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/
================================================================================

# Coinbase Advanced API Python SDK

* * *

# Getting Started

[](https://pypi.org/project/coinbase-advanced-py/) [](https://opensource.org/licenses/Apache-2.0)

Welcome to the official Coinbase Advanced API Python SDK. This python project was created to allow coders to easily plug into the [Coinbase Advanced API](https://docs.cdp.coinbase.com/advanced-trade/docs/welcome/)

  * Docs: <https://docs.cdp.coinbase.com/advanced-trade/docs/welcome/>

  * Python SDK: <https://github.com/coinbase/coinbase-advanced-py>




For detailed exercises on how to get started using the SDK look at our SDK Overview: <https://docs.cdp.coinbase.com/advanced-trade/docs/sdk-overview>

* * *

Contents:

  * [REST API Client](coinbase.rest.html)
    * [RESTClient Constructor](coinbase.rest.html#restclient-constructor)
      * [`RESTClient`](coinbase.rest.html#coinbase.rest.RESTClient)
    * [REST Utils](coinbase.rest.html#rest-utils)
      * [`get()`](coinbase.rest.html#coinbase.rest.RESTClient.get)
      * [`post()`](coinbase.rest.html#coinbase.rest.RESTClient.post)
      * [`put()`](coinbase.rest.html#coinbase.rest.RESTClient.put)
      * [`delete()`](coinbase.rest.html#coinbase.rest.RESTClient.delete)
    * [Accounts](coinbase.rest.html#accounts)
      * [`get_accounts()`](coinbase.rest.html#coinbase.rest.RESTClient.get_accounts)
      * [`get_account()`](coinbase.rest.html#coinbase.rest.RESTClient.get_account)
    * [Products](coinbase.rest.html#products)
      * [`get_products()`](coinbase.rest.html#coinbase.rest.RESTClient.get_products)
      * [`get_product()`](coinbase.rest.html#coinbase.rest.RESTClient.get_product)
      * [`get_product_book()`](coinbase.rest.html#coinbase.rest.RESTClient.get_product_book)
      * [`get_best_bid_ask()`](coinbase.rest.html#coinbase.rest.RESTClient.get_best_bid_ask)
    * [Market Data](coinbase.rest.html#market-data)
      * [`get_candles()`](coinbase.rest.html#coinbase.rest.RESTClient.get_candles)
      * [`get_market_trades()`](coinbase.rest.html#coinbase.rest.RESTClient.get_market_trades)
    * [Orders](coinbase.rest.html#orders)
      * [`create_order()`](coinbase.rest.html#coinbase.rest.RESTClient.create_order)
      * [`market_order()`](coinbase.rest.html#coinbase.rest.RESTClient.market_order)
      * [`market_order_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.market_order_buy)
      * [`market_order_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.market_order_sell)
      * [`limit_order_ioc()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_ioc)
      * [`limit_order_ioc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_ioc_buy)
      * [`limit_order_ioc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_ioc_sell)
      * [`limit_order_gtc()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_gtc)
      * [`limit_order_gtc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_gtc_buy)
      * [`limit_order_gtc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_gtc_sell)
      * [`limit_order_gtd()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_gtd)
      * [`limit_order_gtd_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_gtd_buy)
      * [`limit_order_gtd_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_gtd_sell)
      * [`limit_order_fok()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_fok)
      * [`limit_order_fok_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_fok_buy)
      * [`limit_order_fok_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_fok_sell)
      * [`stop_limit_order_gtc()`](coinbase.rest.html#coinbase.rest.RESTClient.stop_limit_order_gtc)
      * [`stop_limit_order_gtc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.stop_limit_order_gtc_buy)
      * [`stop_limit_order_gtc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.stop_limit_order_gtc_sell)
      * [`stop_limit_order_gtd()`](coinbase.rest.html#coinbase.rest.RESTClient.stop_limit_order_gtd)
      * [`stop_limit_order_gtd_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.stop_limit_order_gtd_buy)
      * [`stop_limit_order_gtd_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.stop_limit_order_gtd_sell)
      * [`trigger_bracket_order_gtc()`](coinbase.rest.html#coinbase.rest.RESTClient.trigger_bracket_order_gtc)
      * [`trigger_bracket_order_gtc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.trigger_bracket_order_gtc_buy)
      * [`trigger_bracket_order_gtc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.trigger_bracket_order_gtc_sell)
      * [`trigger_bracket_order_gtd()`](coinbase.rest.html#coinbase.rest.RESTClient.trigger_bracket_order_gtd)
      * [`trigger_bracket_order_gtd_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.trigger_bracket_order_gtd_buy)
      * [`trigger_bracket_order_gtd_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.trigger_bracket_order_gtd_sell)
      * [`get_order()`](coinbase.rest.html#coinbase.rest.RESTClient.get_order)
      * [`list_orders()`](coinbase.rest.html#coinbase.rest.RESTClient.list_orders)
      * [`get_fills()`](coinbase.rest.html#coinbase.rest.RESTClient.get_fills)
      * [`edit_order()`](coinbase.rest.html#coinbase.rest.RESTClient.edit_order)
      * [`preview_edit_order()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_edit_order)
      * [`cancel_orders()`](coinbase.rest.html#coinbase.rest.RESTClient.cancel_orders)
      * [`preview_order()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_order)
      * [`preview_market_order()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_market_order)
      * [`preview_market_order_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_market_order_buy)
      * [`preview_market_order_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_market_order_sell)
      * [`preview_limit_order_ioc()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_ioc)
      * [`preview_limit_order_ioc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_ioc_buy)
      * [`preview_limit_order_ioc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_ioc_sell)
      * [`preview_limit_order_gtc()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_gtc)
      * [`preview_limit_order_gtc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_gtc_buy)
      * [`preview_limit_order_gtc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_gtc_sell)
      * [`preview_limit_order_gtd()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_gtd)
      * [`preview_limit_order_gtd_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_gtd_buy)
      * [`preview_limit_order_gtd_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_gtd_sell)
      * [`preview_limit_order_fok()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_fok)
      * [`preview_limit_order_fok_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_fok_buy)
      * [`preview_limit_order_fok_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_fok_sell)
      * [`preview_stop_limit_order_gtc()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_stop_limit_order_gtc)
      * [`preview_stop_limit_order_gtc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_stop_limit_order_gtc_buy)
      * [`preview_stop_limit_order_gtc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_stop_limit_order_gtc_sell)
      * [`preview_stop_limit_order_gtd()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_stop_limit_order_gtd)
      * [`preview_stop_limit_order_gtd_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_stop_limit_order_gtd_buy)
      * [`preview_stop_limit_order_gtd_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_stop_limit_order_gtd_sell)
      * [`preview_trigger_bracket_order_gtc()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_trigger_bracket_order_gtc)
      * [`preview_trigger_bracket_order_gtc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_trigger_bracket_order_gtc_buy)
      * [`preview_trigger_bracket_order_gtc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_trigger_bracket_order_gtc_sell)
      * [`preview_trigger_bracket_order_gtd()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_trigger_bracket_order_gtd)
      * [`preview_trigger_bracket_order_gtd_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_trigger_bracket_order_gtd_buy)
      * [`preview_trigger_bracket_order_gtd_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_trigger_bracket_order_gtd_sell)
      * [`close_position()`](coinbase.rest.html#coinbase.rest.RESTClient.close_position)
    * [Portfolios](coinbase.rest.html#portfolios)
      * [`get_portfolios()`](coinbase.rest.html#coinbase.rest.RESTClient.get_portfolios)
      * [`create_portfolio()`](coinbase.rest.html#coinbase.rest.RESTClient.create_portfolio)
      * [`get_portfolio_breakdown()`](coinbase.rest.html#coinbase.rest.RESTClient.get_portfolio_breakdown)
      * [`move_portfolio_funds()`](coinbase.rest.html#coinbase.rest.RESTClient.move_portfolio_funds)
      * [`edit_portfolio()`](coinbase.rest.html#coinbase.rest.RESTClient.edit_portfolio)
      * [`delete_portfolio()`](coinbase.rest.html#coinbase.rest.RESTClient.delete_portfolio)
    * [Futures](coinbase.rest.html#futures)
      * [`get_futures_balance_summary()`](coinbase.rest.html#coinbase.rest.RESTClient.get_futures_balance_summary)
      * [`list_futures_positions()`](coinbase.rest.html#coinbase.rest.RESTClient.list_futures_positions)
      * [`get_futures_position()`](coinbase.rest.html#coinbase.rest.RESTClient.get_futures_position)
      * [`schedule_futures_sweep()`](coinbase.rest.html#coinbase.rest.RESTClient.schedule_futures_sweep)
      * [`list_futures_sweeps()`](coinbase.rest.html#coinbase.rest.RESTClient.list_futures_sweeps)
      * [`cancel_pending_futures_sweep()`](coinbase.rest.html#coinbase.rest.RESTClient.cancel_pending_futures_sweep)
      * [`get_intraday_margin_setting()`](coinbase.rest.html#coinbase.rest.RESTClient.get_intraday_margin_setting)
      * [`get_current_margin_window()`](coinbase.rest.html#coinbase.rest.RESTClient.get_current_margin_window)
      * [`set_intraday_margin_setting()`](coinbase.rest.html#coinbase.rest.RESTClient.set_intraday_margin_setting)
    * [Perpetuals](coinbase.rest.html#perpetuals)
      * [`allocate_portfolio()`](coinbase.rest.html#coinbase.rest.RESTClient.allocate_portfolio)
      * [`get_perps_portfolio_summary()`](coinbase.rest.html#coinbase.rest.RESTClient.get_perps_portfolio_summary)
      * [`list_perps_positions()`](coinbase.rest.html#coinbase.rest.RESTClient.list_perps_positions)
      * [`get_perps_position()`](coinbase.rest.html#coinbase.rest.RESTClient.get_perps_position)
      * [`get_perps_portfolio_balances()`](coinbase.rest.html#coinbase.rest.RESTClient.get_perps_portfolio_balances)
      * [`opt_in_or_out_multi_asset_collateral()`](coinbase.rest.html#coinbase.rest.RESTClient.opt_in_or_out_multi_asset_collateral)
    * [Fees](coinbase.rest.html#fees)
      * [`get_transaction_summary()`](coinbase.rest.html#coinbase.rest.RESTClient.get_transaction_summary)
    * [Converts](coinbase.rest.html#converts)
      * [`create_convert_quote()`](coinbase.rest.html#coinbase.rest.RESTClient.create_convert_quote)
      * [`get_convert_trade()`](coinbase.rest.html#coinbase.rest.RESTClient.get_convert_trade)
      * [`commit_convert_trade()`](coinbase.rest.html#coinbase.rest.RESTClient.commit_convert_trade)
    * [Public](coinbase.rest.html#public)
      * [`get_unix_time()`](coinbase.rest.html#coinbase.rest.RESTClient.get_unix_time)
      * [`get_public_product_book()`](coinbase.rest.html#coinbase.rest.RESTClient.get_public_product_book)
      * [`get_public_products()`](coinbase.rest.html#coinbase.rest.RESTClient.get_public_products)
      * [`get_public_product()`](coinbase.rest.html#coinbase.rest.RESTClient.get_public_product)
      * [`get_public_candles()`](coinbase.rest.html#coinbase.rest.RESTClient.get_public_candles)
      * [`get_public_market_trades()`](coinbase.rest.html#coinbase.rest.RESTClient.get_public_market_trades)
    * [Payments](coinbase.rest.html#payments)
      * [`list_payment_methods()`](coinbase.rest.html#coinbase.rest.RESTClient.list_payment_methods)
      * [`get_payment_method()`](coinbase.rest.html#coinbase.rest.RESTClient.get_payment_method)
    * [Data API](coinbase.rest.html#data-api)
      * [`get_api_key_permissions()`](coinbase.rest.html#coinbase.rest.RESTClient.get_api_key_permissions)
  * [Websocket API Client](coinbase.websocket.html)
    * [WSClient Constructor](coinbase.websocket.html#wsclient-constructor)
      * [`WSClient`](coinbase.websocket.html#coinbase.websocket.WSClient)
    * [WebSocket Utils](coinbase.websocket.html#websocket-utils)
      * [`open()`](coinbase.websocket.html#coinbase.websocket.WSClient.open)
      * [`open_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.open_async)
      * [`close()`](coinbase.websocket.html#coinbase.websocket.WSClient.close)
      * [`close_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.close_async)
      * [`subscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.subscribe)
      * [`subscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.subscribe_async)
      * [`unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.unsubscribe)
      * [`unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.unsubscribe_async)
      * [`unsubscribe_all()`](coinbase.websocket.html#coinbase.websocket.WSClient.unsubscribe_all)
      * [`unsubscribe_all_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.unsubscribe_all_async)
      * [`sleep_with_exception_check()`](coinbase.websocket.html#coinbase.websocket.WSClient.sleep_with_exception_check)
      * [`sleep_with_exception_check_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.sleep_with_exception_check_async)
      * [`run_forever_with_exception_check()`](coinbase.websocket.html#coinbase.websocket.WSClient.run_forever_with_exception_check)
      * [`run_forever_with_exception_check_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.run_forever_with_exception_check_async)
      * [`raise_background_exception()`](coinbase.websocket.html#coinbase.websocket.WSClient.raise_background_exception)
    * [Channels](coinbase.websocket.html#channels)
      * [`heartbeats()`](coinbase.websocket.html#coinbase.websocket.WSClient.heartbeats)
      * [`heartbeats_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.heartbeats_async)
      * [`heartbeats_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.heartbeats_unsubscribe)
      * [`heartbeats_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.heartbeats_unsubscribe_async)
      * [`candles()`](coinbase.websocket.html#coinbase.websocket.WSClient.candles)
      * [`candles_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.candles_async)
      * [`candles_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.candles_unsubscribe)
      * [`candles_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.candles_unsubscribe_async)
      * [`market_trades()`](coinbase.websocket.html#coinbase.websocket.WSClient.market_trades)
      * [`market_trades_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.market_trades_async)
      * [`market_trades_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.market_trades_unsubscribe)
      * [`market_trades_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.market_trades_unsubscribe_async)
      * [`status()`](coinbase.websocket.html#coinbase.websocket.WSClient.status)
      * [`status_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.status_async)
      * [`status_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.status_unsubscribe)
      * [`status_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.status_unsubscribe_async)
      * [`ticker()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker)
      * [`ticker_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_async)
      * [`ticker_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_unsubscribe)
      * [`ticker_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_unsubscribe_async)
      * [`ticker_batch()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_batch)
      * [`ticker_batch_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_batch_async)
      * [`ticker_batch_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_batch_unsubscribe)
      * [`ticker_batch_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_batch_unsubscribe_async)
      * [`level2()`](coinbase.websocket.html#coinbase.websocket.WSClient.level2)
      * [`level2_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.level2_async)
      * [`level2_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.level2_unsubscribe)
      * [`level2_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.level2_unsubscribe_async)
      * [`user()`](coinbase.websocket.html#coinbase.websocket.WSClient.user)
      * [`user_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.user_async)
      * [`user_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.user_unsubscribe)
      * [`user_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.user_unsubscribe_async)
      * [`futures_balance_summary()`](coinbase.websocket.html#coinbase.websocket.WSClient.futures_balance_summary)
      * [`futures_balance_summary_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.futures_balance_summary_async)
      * [`futures_balance_summary_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.futures_balance_summary_unsubscribe)
      * [`futures_balance_summary_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.futures_balance_summary_unsubscribe_async)
    * [Exceptions](coinbase.websocket.html#exceptions)
      * [`WSClientException()`](coinbase.websocket.html#coinbase.websocket.WSClientException)
      * [`WSClientConnectionClosedException()`](coinbase.websocket.html#coinbase.websocket.WSClientConnectionClosedException)
  * [Websocket User API Client](coinbase.websocket.user.html)
    * [WSUserClient Constructor](coinbase.websocket.user.html#wsuserclient-constructor)
      * [`WSUserClient`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient)
    * [WebSocket Utils](coinbase.websocket.user.html#websocket-utils)
      * [`open()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.open)
      * [`open_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.open_async)
      * [`close()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.close)
      * [`close_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.close_async)
      * [`subscribe()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.subscribe)
      * [`subscribe_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.subscribe_async)
      * [`unsubscribe()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.unsubscribe)
      * [`unsubscribe_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.unsubscribe_async)
      * [`unsubscribe_all()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.unsubscribe_all)
      * [`unsubscribe_all_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.unsubscribe_all_async)
      * [`sleep_with_exception_check()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.sleep_with_exception_check)
      * [`sleep_with_exception_check_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.sleep_with_exception_check_async)
      * [`run_forever_with_exception_check()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.run_forever_with_exception_check)
      * [`run_forever_with_exception_check_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.run_forever_with_exception_check_async)
      * [`raise_background_exception()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.raise_background_exception)
    * [Channels](coinbase.websocket.user.html#channels)
      * [`heartbeats()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.heartbeats)
      * [`heartbeats_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.heartbeats_async)
      * [`heartbeats_unsubscribe()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.heartbeats_unsubscribe)
      * [`heartbeats_unsubscribe_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.heartbeats_unsubscribe_async)
      * [`user()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.user)
      * [`user_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.user_async)
      * [`user_unsubscribe()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.user_unsubscribe)
      * [`user_unsubscribe_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.user_unsubscribe_async)
      * [`futures_balance_summary()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.futures_balance_summary)
      * [`futures_balance_summary_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.futures_balance_summary_async)
      * [`futures_balance_summary_unsubscribe()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.futures_balance_summary_unsubscribe)
      * [`futures_balance_summary_unsubscribe_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.futures_balance_summary_unsubscribe_async)
    * [Exceptions](coinbase.websocket.user.html#exceptions)
  * [Authentication](jwt_generator.html)
    * [`build_rest_jwt()`](jwt_generator.html#coinbase.jwt_generator.build_rest_jwt)
    * [`build_ws_jwt()`](jwt_generator.html#coinbase.jwt_generator.build_ws_jwt)
    * [`format_jwt_uri()`](jwt_generator.html#coinbase.jwt_generator.format_jwt_uri)





================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/coinbase.websocket.html
================================================================================

# Websocket API Client

## WSClient Constructor

_class _coinbase.websocket.WSClient(_api_key : str | None = None_, _api_secret : str | None = None_, _key_file : IO | str | None = None_, _base_url ='wss://advanced-trade-ws.coinbase.com'_, _timeout : int | None = None_, _max_size : int | None = 10485760_, _on_message : Callable[[str], None] | None = None_, _on_open : Callable[[], None] | None = None_, _on_close : Callable[[], None] | None = None_, _retry : bool | None = True_, _verbose : bool | None = False_)[[source]](_modules/coinbase/websocket.html#WSClient)
    

### **WSClient**

Initialize using WSClient

* * *

**Parameters** :

  * **api_key | Optional (str)** \- The API key

  * **api_secret | Optional (str)** \- The API key secret

  * **key_file | Optional (IO | str)** \- Path to API key file or file-like object

  * **base_url | (str)** \- The websocket base url. Default set to “wss://advanced-trade-ws.coinbase.com”

  * **timeout | Optional (int)** \- Set timeout in seconds for REST requests

  * **max_size | Optional (int)** \- Max size in bytes for messages received. Default set to (10 * 1024 * 1024)

  * **on_message | Optional (Callable[[str], None])** \- Function called when a message is received

  * **on_open | Optional ([Callable[[], None]])** \- Function called when a connection is opened

  * **on_close | Optional ([Callable[[], None]])** \- Function called when a connection is closed

  * **retry | Optional (bool)** \- Enables automatic reconnections. Default set to True

  * **verbose | Optional (bool)** \- Enables debug logging. Default set to False




## WebSocket Utils

coinbase.websocket.WSClient.open(_self_) → None
    

### **Open Websocket**

* * *

Open the websocket client connection.

_async _coinbase.websocket.WSClient.open_async(_self_) → None
    

### **Open Websocket Async**

* * *

Open the websocket client connection asynchronously.

coinbase.websocket.WSClient.close(_self_) → None
    

### **Close Websocket**

* * *

Close the websocket client connection.

_async _coinbase.websocket.WSClient.close_async(_self_) → None
    

### **Close Websocket Async**

* * *

Close the websocket client connection asynchronously.

coinbase.websocket.WSClient.subscribe(_self_ , _product_ids : List[str]_, _channels : List[str]_) → None
    

### **Subscribe**

* * *

Subscribe to a list of channels for a list of product ids.

  * **product_ids** \- product ids to subscribe to

  * **channels** \- channels to subscribe to




_async _coinbase.websocket.WSClient.subscribe_async(_self_ , _product_ids : List[str]_, _channels : List[str]_) → None
    

### **Subscribe Async**

* * *

Async subscribe to a list of channels for a list of product ids.

  * **product_ids** \- product ids to subscribe to

  * **channels** \- channels to subscribe to




coinbase.websocket.WSClient.unsubscribe(_self_ , _product_ids : List[str]_, _channels : List[str]_) → None
    

### **Unsubscribe**

* * *

Unsubscribe to a list of channels for a list of product ids.

  * **product_ids** \- product ids to unsubscribe from

  * **channels** \- channels to unsubscribe from




_async _coinbase.websocket.WSClient.unsubscribe_async(_self_ , _product_ids : List[str]_, _channels : List[str]_) → None
    

### **Unsubscribe Async**

* * *

Async unsubscribe to a list of channels for a list of product ids.

  * **product_ids** \- product ids to unsubscribe from

  * **channels** \- channels to unsubscribe from




coinbase.websocket.WSClient.unsubscribe_all(_self_) → None
    

### **Unsubscribe All**

* * *

Unsubscribe from all channels you are currently subscribed to.

_async _coinbase.websocket.WSClient.unsubscribe_all_async(_self_) → None
    

### **Unsubscribe All Async**

* * *

Async unsubscribe from all channels you are currently subscribed to.

coinbase.websocket.WSClient.sleep_with_exception_check(_self_ , _sleep : int_) → None
    

### **Sleep with Exception Check**

* * *

Sleep for a specified number of seconds and check for background exceptions.

  * **sleep** \- number of seconds to sleep.




_async _coinbase.websocket.WSClient.sleep_with_exception_check_async(_self_ , _sleep : int_) → None
    

### **Sleep with Exception Check Async**

* * *

Async sleep for a specified number of seconds and check for background exceptions.

  * **sleep** \- number of seconds to sleep.




coinbase.websocket.WSClient.run_forever_with_exception_check(_self_) → None
    

### **Run Forever with Exception Check**

* * *

Runs an endless loop, checking for background exceptions every second.

_async _coinbase.websocket.WSClient.run_forever_with_exception_check_async(_self_) → None
    

### **Run Forever with Exception Check Async**

* * *

Async runs an endless loop, checking for background exceptions every second.

coinbase.websocket.WSClient.raise_background_exception(_self_) → None
    

### **Raise Background Exception**

* * *

Raise any background exceptions that occurred in the message handler.

## Channels

coinbase.websocket.WSClient.heartbeats(_self_) → None
    

### **Heartbeats Subscribe**

* * *

**Description:**

Subscribe to heartbeats channel.

* * *

**Read more on the official documentation:** [Heartbeats Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#heartbeats-channel)

_async _coinbase.websocket.WSClient.heartbeats_async(_self_) → None
    

### **Heartbeats Subscribe Async**

* * *

**Description:**

Async subscribe to heartbeats channel.

* * *

**Read more on the official documentation:** [Heartbeats Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#heartbeats-channel)

coinbase.websocket.WSClient.heartbeats_unsubscribe(_self_) → None
    

### **Heartbeats Unsubscribe**

* * *

**Description:**

Unsubscribe to heartbeats channel.

* * *

**Read more on the official documentation:** [Heartbeats Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#heartbeats-channel)

_async _coinbase.websocket.WSClient.heartbeats_unsubscribe_async(_self_) → None
    

### **Heartbeats Unsubscribe Async**

* * *

**Description:**

Async unsubscribe to heartbeats channel.

* * *

**Read more on the official documentation:** [Heartbeats Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#heartbeats-channel)

coinbase.websocket.WSClient.candles(_self_ , _product_ids : List[str]_) → None
    

### **Candles Subscribe**

* * *

**Description:**

Subscribe to candles channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Candles Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#candles-channel)

_async _coinbase.websocket.WSClient.candles_async(_self_ , _product_ids : List[str]_) → None
    

### **Candles Subscribe Async**

* * *

**Description:**

Async subscribe to candles channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Candles Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#candles-channel)

coinbase.websocket.WSClient.candles_unsubscribe(_self_ , _product_ids : List[str]_) → None
    

### **Candles Unsubscribe**

* * *

**Description:**

Unsubscribe to candles channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Candles Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#candles-channel)

_async _coinbase.websocket.WSClient.candles_unsubscribe_async(_self_ , _product_ids : List[str]_) → None
    

### **Candles Unsubscribe Async**

* * *

**Description:**

Async unsubscribe to candles channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Candles Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#candles-channel)

coinbase.websocket.WSClient.market_trades(_self_ , _product_ids : List[str]_) → None
    

### **Market Trades Subscribe**

* * *

**Description:**

Subscribe to market_trades channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Market Trades Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#market-trades-channel)

_async _coinbase.websocket.WSClient.market_trades_async(_self_ , _product_ids : List[str]_) → None
    

### **Market Trades Subscribe Async**

* * *

**Description:**

Async subscribe to market_trades channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Market Trades Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#market-trades-channel)

coinbase.websocket.WSClient.market_trades_unsubscribe(_self_ , _product_ids : List[str]_) → None
    

### **Market Trades Unsubscribe**

* * *

**Description:**

Unsubscribe to market_trades channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Market Trades Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#market-trades-channel)

_async _coinbase.websocket.WSClient.market_trades_unsubscribe_async(_self_ , _product_ids : List[str]_) → None
    

### **Market Trades Unsubscribe Async**

* * *

**Description:**

Async unsubscribe to market_trades channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Market Trades Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#market-trades-channel)

coinbase.websocket.WSClient.status(_self_ , _product_ids : List[str]_) → None
    

### **Status Subscribe**

* * *

**Description:**

Subscribe to status channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Status Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#status-channel)

_async _coinbase.websocket.WSClient.status_async(_self_ , _product_ids : List[str]_) → None
    

### **Status Subscribe Async**

* * *

**Description:**

Async subscribe to status channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Status Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#status-channel)

coinbase.websocket.WSClient.status_unsubscribe(_self_ , _product_ids : List[str]_) → None
    

### **Status Unsubscribe**

* * *

**Description:**

Unsubscribe to status channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Status Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#status-channel)

_async _coinbase.websocket.WSClient.status_unsubscribe_async(_self_ , _product_ids : List[str]_) → None
    

### **Status Unsubscribe Async**

* * *

**Description:**

Async unsubscribe to status channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Status Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#status-channel)

coinbase.websocket.WSClient.ticker(_self_ , _product_ids : List[str]_) → None
    

### **Ticker Subscribe**

* * *

**Description:**

Subscribe to ticker channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Ticker Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#ticker-channel)

_async _coinbase.websocket.WSClient.ticker_async(_self_ , _product_ids : List[str]_) → None
    

### **Ticker Subscribe Async**

* * *

**Description:**

Async subscribe to ticker channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Ticker Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#ticker-channel)

coinbase.websocket.WSClient.ticker_unsubscribe(_self_ , _product_ids : List[str]_) → None
    

### **Ticker Unsubscribe**

* * *

**Description:**

Unsubscribe to ticker channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Ticker Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#ticker-channel)

_async _coinbase.websocket.WSClient.ticker_unsubscribe_async(_self_ , _product_ids : List[str]_) → None
    

### **Ticker Unsubscribe Async**

* * *

**Description:**

Async unsubscribe to ticker channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Ticker Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#ticker-channel)

coinbase.websocket.WSClient.ticker_batch(_self_ , _product_ids : List[str]_) → None
    

### **Ticker Batch Subscribe**

* * *

**Description:**

Subscribe to ticker_batch channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Ticker Batch Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#ticker-batch-channel)

_async _coinbase.websocket.WSClient.ticker_batch_async(_self_ , _product_ids : List[str]_) → None
    

### **Ticker Batch Subscribe Async**

* * *

**Description:**

Async subscribe to ticker_batch channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Ticker Batch Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#ticker-batch-channel)

coinbase.websocket.WSClient.ticker_batch_unsubscribe(_self_ , _product_ids : List[str]_) → None
    

### **Ticker Batch Unsubscribe**

* * *

**Description:**

Unsubscribe to ticker_batch channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Ticker Batch Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#ticker-batch-channel)

_async _coinbase.websocket.WSClient.ticker_batch_unsubscribe_async(_self_ , _product_ids : List[str]_) → None
    

### **Ticker Batch Unsubscribe Async**

* * *

**Description:**

Async unsubscribe to ticker_batch channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Ticker Batch Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#ticker-batch-channel)

coinbase.websocket.WSClient.level2(_self_ , _product_ids : List[str]_) → None
    

### **Level2 Subscribe**

* * *

**Description:**

Subscribe to level2 channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Level2 Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#level2-channel)

_async _coinbase.websocket.WSClient.level2_async(_self_ , _product_ids : List[str]_) → None
    

### **Level2 Subscribe Async**

* * *

**Description:**

Async subscribe to level2 channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Level2 Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#level2-channel)

coinbase.websocket.WSClient.level2_unsubscribe(_self_ , _product_ids : List[str]_) → None
    

### **Level2 Unsubscribe**

* * *

**Description:**

Unsubscribe to level2 channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Level2 Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#level2-channel)

_async _coinbase.websocket.WSClient.level2_unsubscribe_async(_self_ , _product_ids : List[str]_) → None
    

### **Level2 Unsubscribe Async**

* * *

**Description:**

Async unsubscribe to level2 channel for a list of products_ids.

* * *

**Read more on the official documentation:** [Level2 Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#level2-channel)

coinbase.websocket.WSClient.user(_self_ , _product_ids : List[str]_) → None
    

### **User Subscribe**

* * *

**Description:**

Subscribe to user channel for a list of products_ids.

* * *

**Read more on the official documentation:** [User Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#user-channel)

_async _coinbase.websocket.WSClient.user_async(_self_ , _product_ids : List[str]_) → None
    

### **User Subscribe Async**

* * *

**Description:**

Async subscribe to user channel for a list of products_ids.

* * *

**Read more on the official documentation:** [User Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#user-channel)

coinbase.websocket.WSClient.user_unsubscribe(_self_ , _product_ids : List[str]_) → None
    

### **User Unsubscribe**

* * *

**Description:**

Unsubscribe to user channel for a list of products_ids.

* * *

**Read more on the official documentation:** [User Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#user-channel)

_async _coinbase.websocket.WSClient.user_unsubscribe_async(_self_ , _product_ids : List[str]_) → None
    

### **User Unsubscribe Async**

* * *

**Description:**

Async unsubscribe to user channel for a list of products_ids.

* * *

**Read more on the official documentation:** [User Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#user-channel)

coinbase.websocket.WSClient.futures_balance_summary(_self_) → None
    

### **Futures Balance Summary Subscribe**

* * *

**Description:**

Subscribe to futures_balance_summary channel.

* * *

**Read more on the official documentation:** [Futures Balance Summary Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#futures-balance-summary-channel)

_async _coinbase.websocket.WSClient.futures_balance_summary_async(_self_) → None
    

### **Futures Balance Summary Subscribe Async**

* * *

**Description:**

Async subscribe to futures_balance_summary channel.

* * *

**Read more on the official documentation:** [Futures Balance Summary Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#futures-balance-summary-channel)

coinbase.websocket.WSClient.futures_balance_summary_unsubscribe(_self_) → None
    

### **Futures Balance Summary Unsubscribe**

* * *

**Description:**

Unsubscribe to futures_balance_summary channel.

* * *

**Read more on the official documentation:** [Futures Balance Summary Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#futures-balance-summary-channel)

_async _coinbase.websocket.WSClient.futures_balance_summary_unsubscribe_async(_self_) → None
    

### **Futures Balance Summary Unsubscribe Async**

* * *

**Description:**

Async unsubscribe to futures_balance_summary channel.

* * *

**Read more on the official documentation:** [Futures Balance Summary Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#futures-balance-summary-channel)

## Exceptions

coinbase.websocket.WSClientException()[[source]](_modules/coinbase/websocket/websocket_base.html#WSClientException)
    

### **WSClientException**

* * *

Exception raised for errors in the WebSocket client.

coinbase.websocket.WSClientConnectionClosedException()[[source]](_modules/coinbase/websocket/websocket_base.html#WSClientConnectionClosedException)
    

### **WSClientConnectionClosedException**

* * *

Exception raised for unexpected closure in the WebSocket client.



================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/jwt_generator.html
================================================================================

# Authentication

coinbase.jwt_generator.build_rest_jwt(_uri_ , _key_var_ , _secret_var_) → str[[source]](_modules/coinbase/jwt_generator.html#build_rest_jwt)
    

## **Build REST JWT**

**Description:**

Builds and returns a JWT token for connecting to the REST API.

* * *

Parameters:

  * **uri (str)** \- Formatted URI for the endpoint (e.g. “GET api.coinbase.com/api/v3/brokerage/accounts”) Can be generated using `format_jwt_uri`

  * **key_var (str)** \- The API key

  * **secret_var (str)** \- The API key secret




coinbase.jwt_generator.build_ws_jwt(_key_var_ , _secret_var_) → str[[source]](_modules/coinbase/jwt_generator.html#build_ws_jwt)
    

## **Build WebSocket JWT**

**Description:**

Builds and returns a JWT token for connecting to the WebSocket API.

* * *

Parameters:

  * **key_var (str)** \- The API key

  * **secret_var (str)** \- The API key secret




coinbase.jwt_generator.format_jwt_uri(_method_ , _path_) → str[[source]](_modules/coinbase/jwt_generator.html#format_jwt_uri)
    

## **Format JWT URI**

**Description:**

Formats method and path into valid URI for JWT generation.

* * *

Parameters:

  * **method (str)** \- The REST request method. E.g. GET, POST, PUT, DELETE

  * **path (str)** \- The path of the endpoint. E.g. “/api/v3/brokerage/accounts”






================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/coinbase.websocket.user.html
================================================================================

# Websocket User API Client

## WSUserClient Constructor

_class _coinbase.websocket.WSUserClient(_api_key : str | None = None_, _api_secret : str | None = None_, _key_file : IO | str | None = None_, _base_url ='wss://advanced-trade-ws-user.coinbase.com'_, _timeout : int | None = None_, _max_size : int | None = 10485760_, _on_message : Callable[[str], None] | None = None_, _on_open : Callable[[], None] | None = None_, _on_close : Callable[[], None] | None = None_, _retry : bool | None = True_, _verbose : bool | None = False_)[[source]](_modules/coinbase/websocket.html#WSUserClient)
    

### **WSUserClient**

Initialize using WSUserClient

* * *

**Parameters** :

  * **api_key | Optional (str)** \- The API key

  * **api_secret | Optional (str)** \- The API key secret

  * **key_file | Optional (IO | str)** \- Path to API key file or file-like object

  * **base_url | (str)** \- The websocket base url. Default set to “wss://advanced-trade-ws.coinbase.com”

  * **timeout | Optional (int)** \- Set timeout in seconds for REST requests

  * **max_size | Optional (int)** \- Max size in bytes for messages received. Default set to (10 * 1024 * 1024)

  * **on_message | Optional (Callable[[str], None])** \- Function called when a message is received

  * **on_open | Optional ([Callable[[], None]])** \- Function called when a connection is opened

  * **on_close | Optional ([Callable[[], None]])** \- Function called when a connection is closed

  * **retry | Optional (bool)** \- Enables automatic reconnections. Default set to True

  * **verbose | Optional (bool)** \- Enables debug logging. Default set to False




## WebSocket Utils

coinbase.websocket.WSUserClient.open(_self_) → None
    

### **Open Websocket**

* * *

Open the websocket client connection.

_async _coinbase.websocket.WSUserClient.open_async(_self_) → None
    

### **Open Websocket Async**

* * *

Open the websocket client connection asynchronously.

coinbase.websocket.WSUserClient.close(_self_) → None
    

### **Close Websocket**

* * *

Close the websocket client connection.

_async _coinbase.websocket.WSUserClient.close_async(_self_) → None
    

### **Close Websocket Async**

* * *

Close the websocket client connection asynchronously.

coinbase.websocket.WSUserClient.subscribe(_self_ , _product_ids : List[str]_, _channels : List[str]_) → None
    

### **Subscribe**

* * *

Subscribe to a list of channels for a list of product ids.

  * **product_ids** \- product ids to subscribe to

  * **channels** \- channels to subscribe to




_async _coinbase.websocket.WSUserClient.subscribe_async(_self_ , _product_ids : List[str]_, _channels : List[str]_) → None
    

### **Subscribe Async**

* * *

Async subscribe to a list of channels for a list of product ids.

  * **product_ids** \- product ids to subscribe to

  * **channels** \- channels to subscribe to




coinbase.websocket.WSUserClient.unsubscribe(_self_ , _product_ids : List[str]_, _channels : List[str]_) → None
    

### **Unsubscribe**

* * *

Unsubscribe to a list of channels for a list of product ids.

  * **product_ids** \- product ids to unsubscribe from

  * **channels** \- channels to unsubscribe from




_async _coinbase.websocket.WSUserClient.unsubscribe_async(_self_ , _product_ids : List[str]_, _channels : List[str]_) → None
    

### **Unsubscribe Async**

* * *

Async unsubscribe to a list of channels for a list of product ids.

  * **product_ids** \- product ids to unsubscribe from

  * **channels** \- channels to unsubscribe from




coinbase.websocket.WSUserClient.unsubscribe_all(_self_) → None
    

### **Unsubscribe All**

* * *

Unsubscribe from all channels you are currently subscribed to.

_async _coinbase.websocket.WSUserClient.unsubscribe_all_async(_self_) → None
    

### **Unsubscribe All Async**

* * *

Async unsubscribe from all channels you are currently subscribed to.

coinbase.websocket.WSUserClient.sleep_with_exception_check(_self_ , _sleep : int_) → None
    

### **Sleep with Exception Check**

* * *

Sleep for a specified number of seconds and check for background exceptions.

  * **sleep** \- number of seconds to sleep.




_async _coinbase.websocket.WSUserClient.sleep_with_exception_check_async(_self_ , _sleep : int_) → None
    

### **Sleep with Exception Check Async**

* * *

Async sleep for a specified number of seconds and check for background exceptions.

  * **sleep** \- number of seconds to sleep.




coinbase.websocket.WSUserClient.run_forever_with_exception_check(_self_) → None
    

### **Run Forever with Exception Check**

* * *

Runs an endless loop, checking for background exceptions every second.

_async _coinbase.websocket.WSUserClient.run_forever_with_exception_check_async(_self_) → None
    

### **Run Forever with Exception Check Async**

* * *

Async runs an endless loop, checking for background exceptions every second.

coinbase.websocket.WSUserClient.raise_background_exception(_self_) → None
    

### **Raise Background Exception**

* * *

Raise any background exceptions that occurred in the message handler.

## Channels

coinbase.websocket.WSUserClient.heartbeats(_self_) → None
    

### **Heartbeats Subscribe**

* * *

**Description:**

Subscribe to heartbeats channel.

* * *

**Read more on the official documentation:** [Heartbeats Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#heartbeats-channel)

_async _coinbase.websocket.WSUserClient.heartbeats_async(_self_) → None
    

### **Heartbeats Subscribe Async**

* * *

**Description:**

Async subscribe to heartbeats channel.

* * *

**Read more on the official documentation:** [Heartbeats Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#heartbeats-channel)

coinbase.websocket.WSUserClient.heartbeats_unsubscribe(_self_) → None
    

### **Heartbeats Unsubscribe**

* * *

**Description:**

Unsubscribe to heartbeats channel.

* * *

**Read more on the official documentation:** [Heartbeats Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#heartbeats-channel)

_async _coinbase.websocket.WSUserClient.heartbeats_unsubscribe_async(_self_) → None
    

### **Heartbeats Unsubscribe Async**

* * *

**Description:**

Async unsubscribe to heartbeats channel.

* * *

**Read more on the official documentation:** [Heartbeats Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#heartbeats-channel)

coinbase.websocket.WSUserClient.user(_self_ , _product_ids : List[str]_) → None
    

### **User Subscribe**

* * *

**Description:**

Subscribe to user channel for a list of products_ids.

* * *

**Read more on the official documentation:** [User Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#user-channel)

_async _coinbase.websocket.WSUserClient.user_async(_self_ , _product_ids : List[str]_) → None
    

### **User Subscribe Async**

* * *

**Description:**

Async subscribe to user channel for a list of products_ids.

* * *

**Read more on the official documentation:** [User Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#user-channel)

coinbase.websocket.WSUserClient.user_unsubscribe(_self_ , _product_ids : List[str]_) → None
    

### **User Unsubscribe**

* * *

**Description:**

Unsubscribe to user channel for a list of products_ids.

* * *

**Read more on the official documentation:** [User Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#user-channel)

_async _coinbase.websocket.WSUserClient.user_unsubscribe_async(_self_ , _product_ids : List[str]_) → None
    

### **User Unsubscribe Async**

* * *

**Description:**

Async unsubscribe to user channel for a list of products_ids.

* * *

**Read more on the official documentation:** [User Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#user-channel)

coinbase.websocket.WSUserClient.futures_balance_summary(_self_) → None
    

### **Futures Balance Summary Subscribe**

* * *

**Description:**

Subscribe to futures_balance_summary channel.

* * *

**Read more on the official documentation:** [Futures Balance Summary Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#futures-balance-summary-channel)

_async _coinbase.websocket.WSUserClient.futures_balance_summary_async(_self_) → None
    

### **Futures Balance Summary Subscribe Async**

* * *

**Description:**

Async subscribe to futures_balance_summary channel.

* * *

**Read more on the official documentation:** [Futures Balance Summary Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#futures-balance-summary-channel)

coinbase.websocket.WSUserClient.futures_balance_summary_unsubscribe(_self_) → None
    

### **Futures Balance Summary Unsubscribe**

* * *

**Description:**

Unsubscribe to futures_balance_summary channel.

* * *

**Read more on the official documentation:** [Futures Balance Summary Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#futures-balance-summary-channel)

_async _coinbase.websocket.WSUserClient.futures_balance_summary_unsubscribe_async(_self_) → None
    

### **Futures Balance Summary Unsubscribe Async**

* * *

**Description:**

Async unsubscribe to futures_balance_summary channel.

* * *

**Read more on the official documentation:** [Futures Balance Summary Channel](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels#futures-balance-summary-channel)

## Exceptions



================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/_sources/index.rst.txt
================================================================================


    .. Docs documentation master file, created by
       sphinx-quickstart on Wed Jan 17 16:53:39 2024.
       You can adapt this file completely to your liking, but it should at least
       contain the root `toctree` directive.
    
    Coinbase Advanced API Python SDK
    ================================
    
    -------------
    
    Getting Started
    ================================
    .. image:: https://badge.fury.io/py/coinbase-advanced-py.svg
       :target: https://pypi.org/project/coinbase-advanced-py/
       :alt: PyPI Version
    
    .. image:: https://img.shields.io/badge/License-Apache%202.0-green.svg
       :target: https://opensource.org/licenses/Apache-2.0
       :alt: Apache License 2.0
    
    Welcome to the official Coinbase Advanced API Python SDK. This python project was created to allow coders to easily plug
    into the `Coinbase Advanced API <https://docs.cdp.coinbase.com/advanced-trade/docs/welcome/>`_
    
    
    - Docs: https://docs.cdp.coinbase.com/advanced-trade/docs/welcome/
    - Python SDK: https://github.com/coinbase/coinbase-advanced-py
    
    For detailed exercises on how to get started using the SDK look at our SDK Overview:
    https://docs.cdp.coinbase.com/advanced-trade/docs/sdk-overview
    
    -------------
    
    .. toctree::
       :maxdepth: 3
       :caption: Contents:
    
       coinbase.rest
       coinbase.websocket
       coinbase.websocket.user
       jwt_generator



================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/coinbase.rest.html
================================================================================

# REST API Client

## RESTClient Constructor

_class _coinbase.rest.RESTClient(_api_key : str | None = None_, _api_secret : str | None = None_, _key_file : IO | str | None = None_, _base_url ='api.coinbase.com'_, _timeout : int | None = None_, _verbose : bool | None = False_, _rate_limit_headers : bool | None = False_)[[source]](_modules/coinbase/rest.html#RESTClient)
    

### **RESTClient**

Initialize using RESTClient

* * *

**Parameters** :

  * **api_key | Optional (str)** \- The API key

  * **api_secret | Optional (str)** \- The API key secret

  * **key_file | Optional (IO | str)** \- Path to API key file or file-like object

  * **base_url | (str)** \- The base URL for REST requests. Default set to “<https://api.coinbase.com>”

  * **timeout | Optional (int)** \- Set timeout in seconds for REST requests

  * **verbose | Optional (bool)** \- Enables debug logging. Default set to False

  * **rate_limit_headers | Optional (bool)** \- Enables rate limit headers. Default set to False




## REST Utils

coinbase.rest.RESTClient.get(_self_ , _url_path_ , _params : dict | None = None_, _public =False_, _** kwargs_) → Dict[str, Any]
    

### **GET Request**

* * *

**Parameters:**

  * **url_path | (str)** \- the URL path

  * **params | Optional ([dict])** \- the query parameters

  * **public | (bool)** \- flag indicating whether to treat endpoint as public




coinbase.rest.RESTClient.post(_self_ , _url_path_ , _params : dict | None = None_, _data : dict | None = None_, _** kwargs_) → Dict[str, Any]
    

### **Authenticated POST Request**

* * *

> **Parameters:**

  * **url_path | (str)** \- the URL path

  * **params | Optional ([dict])** \- the query parameters

  * **data | Optional ([dict])** \- the request body




coinbase.rest.RESTClient.put(_self_ , _url_path_ , _params : dict | None = None_, _data : dict | None = None_, _** kwargs_) → Dict[str, Any]
    

### **Authenticated PUT Request**

* * *

**Parameters:**

  * **url_path | (str)** \- the URL path

  * **params | Optional ([dict])** \- the query parameters

  * **data | Optional ([dict])** \- the request body




coinbase.rest.RESTClient.delete(_self_ , _url_path_ , _params : dict | None = None_, _data : dict | None = None_, _** kwargs_) → Dict[str, Any]
    

### **Authenticated DELETE Request**

* * *

**Parameters:**

  * **url_path | (str)** \- the URL path

  * **params | Optional ([dict])** \- the query parameters

  * **data | Optional ([dict])** \- the request body




## Accounts

coinbase.rest.RESTClient.get_accounts(_self_ , _limit : int | None = None_, _cursor : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → ListAccountsResponse
    

### **List Accounts**

[GET] <https://api.coinbase.com/api/v3/brokerage/accounts>

* * *

**Description:**

Get a list of authenticated accounts for the current user.

* * *

**Read more on the official documentation:** [List Accounts](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getaccounts)

coinbase.rest.RESTClient.get_account(_self_ , _account_uuid : str_, _** kwargs_) → GetAccountResponse
    

### **Get Account**

[GET] <https://api.coinbase.com/api/v3/brokerage/accounts>/{account_uuid}

* * *

**Description:**

Get a list of information about an account, given an account UUID.

* * *

**Read more on the official documentation:** [Get Account](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getaccount)

## Products

coinbase.rest.RESTClient.get_products(_self_ , _limit : int | None = None_, _offset : int | None = None_, _product_type : str | None = None_, _product_ids : List[str] | None = None_, _contract_expiry_type : str | None = None_, _expiring_contract_status : str | None = None_, _get_tradability_status : bool | None = False_, _get_all_products : bool | None = False_, _** kwargs_) → ListProductsResponse
    

### **List Products**

[GET] <https://api.coinbase.com/api/v3/brokerage/products>

* * *

**Description:**

Get a list of the available currency pairs for trading.

* * *

**Read more on the official documentation:** [List Products](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getproducts)

coinbase.rest.RESTClient.get_product(_self_ , _product_id : str_, _get_tradability_status : bool | None = False_, _** kwargs_) → GetProductResponse
    

### **Get Product**

[GET] <https://api.coinbase.com/api/v3/brokerage/products>/{product_id}

* * *

**Description:**

Get information on a single product by product ID.

* * *

**Read more on the official documentation:** [Get Product](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getproduct)

coinbase.rest.RESTClient.get_product_book(_self_ , _product_id : str_, _limit : int | None = None_, _aggregation_price_increment : str | None = None_, _** kwargs_) → GetProductBookResponse
    

### **Get Product Book**

[GET] <https://api.coinbase.com/api/v3/brokerage/product_book>

* * *

**Description:**

Get a list of bids/asks for a single product. The amount of detail shown can be customized with the limit parameter.

* * *

**Read more on the official documentation:** [Get Product Book](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getproductbook)

coinbase.rest.RESTClient.get_best_bid_ask(_self_ , _product_ids : List[str] | None = None_, _** kwargs_) → GetBestBidAskResponse
    

### **Get Best Bid/Ask**

[GET] <https://api.coinbase.com/api/v3/brokerage/best_bid_ask>

* * *

**Description:**

Get the best bid/ask for all products. A subset of all products can be returned instead by using the product_ids input.

* * *

**Read more on the official documentation:** [Get Best Bid/Ask](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getproductbook)

## Market Data

coinbase.rest.RESTClient.get_candles(_self_ , _product_id : str_, _start : str_, _end : str_, _granularity : str_, _limit : int | None = None_, _** kwargs_) → GetProductCandlesResponse
    

### **Get Product Candles**

[GET] <https://api.coinbase.com/api/v3/brokerage/products>/{product_id}/candles

* * *

**Description:**

Get rates for a single product by product ID, grouped in buckets.

* * *

**Read more on the official documentation:** [Get Product Candles](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getcandles)

coinbase.rest.RESTClient.get_market_trades(_self_ , _product_id : str_, _limit : int_, _start : str | None = None_, _end : str | None = None_, _** kwargs_) → GetMarketTradesResponse
    

### **Get Market Trades**

[GET] <https://api.coinbase.com/api/v3/brokerage/products>/{product_id}/ticker

* * *

**Description:**

Get snapshot information, by product ID, about the last trades (ticks), best bid/ask, and 24h volume.

* * *

**Read more on the official documentation:** [Get Market Trades](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getmarkettrades)

## Orders

coinbase.rest.RESTClient.create_order(_self_ , _client_order_id : str_, _product_id : str_, _side : str_, _order_configuration_ , _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Create Order**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Create an order with a specified `product_id` (asset-pair), `side` (buy/sell), etc.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.market_order(_self_ , _client_order_id : str_, _product_id : str_, _side : str_, _quote_size : str | None = None_, _base_size : str | None = None_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Market Order**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a market order to BUY or SELL the desired product at the given market price. If you wish to purchase the product, provide a quote_size and if you wish to sell the product, provide a base_size.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.market_order_buy(_self_ , _client_order_id : str_, _product_id : str_, _quote_size : str | None = None_, _base_size : str | None = None_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Create Market Order Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a market order to BUY the desired product at the given market price.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.market_order_sell(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Create Market Order Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a market order to SELL the desired product at the given market price.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.limit_order_ioc(_self_ , _client_order_id : str_, _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Limit IOC Order**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a Limit Order with a IOC time-in-force policy. Provide the base_size (quantity of your base currency to spend) as well as a limit_price that indicates the maximum price at which the order should be filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.limit_order_ioc_buy(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Limit IOC Order Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a Buy Limit Order with a IOC time-in-force policy. Provide the base_size (quantity of your base currency to spend) as well as a limit_price that indicates the maximum price at which the order should be filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.limit_order_ioc_sell(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Limit IOC Order Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a Sell Limit Order with a IOC time-in-force policy. Provide the base_size (quantity of your base currency to spend) as well as a limit_price that indicates the maximum price at which the order should be filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.limit_order_gtc(_self_ , _client_order_id : str_, _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _post_only : bool = False_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Limit Order GTC**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a Limit Order with a GTC time-in-force policy. Provide the base_size (quantity of your base currency to spend) as well as a limit_price that indicates the maximum price at which the order should be filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.limit_order_gtc_buy(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _post_only : bool = False_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Limit Order GTC Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a BUY Limit Order with a GTC time-in-force policy. Provide the base_size (quantity of your base currency to spend) as well as a limit_price that indicates the maximum price at which the order should be filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.limit_order_gtc_sell(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _post_only : bool = False_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Limit Order GTC Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a SELL Limit Order with a GTC time-in-force policy. Provide the base_size (quantity of your base currency to spend) as well as a limit_price that indicates the maximum price at which the order should be filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.limit_order_gtd(_self_ , _client_order_id : str_, _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _end_time : str_, _post_only : bool = False_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Limit Order GTD**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a Limit Order with a GTD time-in-force policy. Unlike a Limit Order with a GTC time-in-force policy, this order type requires an end-time that indicates when this order should expire.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.limit_order_gtd_buy(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _end_time : str_, _post_only : bool = False_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Limit Order GTD Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a BUY Limit Order with a GTD time-in-force policy. Unlike a Limit Order with a GTC time-in-force policy, this order type requires an end-time that indicates when this order should expire.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.limit_order_gtd_sell(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _end_time : str_, _post_only : bool = False_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Limit Order GTD Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a SELL Limit Order with a GTD time-in-force policy. Unlike a Limit Order with a GTC time-in-force policy, this order type requires an end-time that indicates when this order should expire.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.limit_order_fok(_self_ , _client_order_id : str_, _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Limit FOK Order**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a Limit Order with a FOK time-in-force policy. Provide the base_size (quantity of your base currency to spend) as well as a limit_price that indicates the maximum price at which the order should be filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.limit_order_fok_buy(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Limit FOK Order Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a Buy Limit Order with a FOK time-in-force policy. Provide the base_size (quantity of your base currency to spend) as well as a limit_price that indicates the maximum price at which the order should be filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.limit_order_fok_sell(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Limit FOK Order Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a Sell Limit Order with a FOK time-in-force policy. Provide the base_size (quantity of your base currency to spend) as well as a limit_price that indicates the maximum price at which the order should be filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.stop_limit_order_gtc(_self_ , _client_order_id : str_, _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _stop_price : str_, _stop_direction : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Stop-Limit Order GTC**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a Stop Limit order with a GTC time-in-force policy. Stop orders become active and wait to trigger based on the movement of the last trade price. The last trade price is the last price at which an order was filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.stop_limit_order_gtc_buy(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _stop_price : str_, _stop_direction : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Stop-Limit Order GTC Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a BUY Stop Limit order with a GTC time-in-force policy. Stop orders become active and wait to trigger based on the movement of the last trade price. The last trade price is the last price at which an order was filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.stop_limit_order_gtc_sell(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _stop_price : str_, _stop_direction : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Stop-Limit Order GTC Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a SELL Stop Limit order with a GTC time-in-force policy. Stop orders become active and wait to trigger based on the movement of the last trade price. The last trade price is the last price at which an order was filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.stop_limit_order_gtd(_self_ , _client_order_id : str_, _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _stop_price : str_, _end_time : str_, _stop_direction : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Stop-Limit Order GTD**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a Stop Limit order with a GTD time-in-force policy. Stop orders become active and wait to trigger based on the movement of the last trade price. The last trade price is the last price at which an order was filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.stop_limit_order_gtd_buy(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _stop_price : str_, _end_time : str_, _stop_direction : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Stop-Limit Order GTD Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a BUY Stop Limit order with a GTD time-in-force policy. Stop orders become active and wait to trigger based on the movement of the last trade price. The last trade price is the last price at which an order was filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.stop_limit_order_gtd_sell(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _stop_price : str_, _end_time : str_, _stop_direction : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Stop-Limit Order GTD Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a SELL Stop Limit order with a GTD time-in-force policy. Stop orders become active and wait to trigger based on the movement of the last trade price. The last trade price is the last price at which an order was filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.trigger_bracket_order_gtc(_self_ , _client_order_id : str_, _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _stop_trigger_price : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Trigger Bracket Order GTC**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a Trigger Bracket order with a GTC time-in-force policy. Trigger Bracket orders become active and wait to trigger based on the movement of the last trade price. The last trade price is the last price at which an order was filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.trigger_bracket_order_gtc_buy(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _stop_trigger_price : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Trigger Bracket Order GTC Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a BUY Trigger Bracket order with a GTC time-in-force policy. Trigger Bracket orders become active and wait to trigger based on the movement of the last trade price. The last trade price is the last price at which an order was filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.trigger_bracket_order_gtc_sell(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _stop_trigger_price : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Trigger Bracket Order GTC Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a SELL Trigger Bracket order with a GTC time-in-force policy. Trigger Bracket orders become active and wait to trigger based on the movement of the last trade price. The last trade price is the last price at which an order was filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.trigger_bracket_order_gtd(_self_ , _client_order_id : str_, _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _stop_trigger_price : str_, _end_time : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Trigger Bracket Order GTD**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a Trigger Bracket order with a GTD time-in-force policy. Trigger Bracket orders become active and wait to trigger based on the movement of the last trade price. The last trade price is the last price at which an order was filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.trigger_bracket_order_gtd_buy(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _stop_trigger_price : str_, _end_time : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Trigger Bracket Order GTD Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a BUY Trigger Bracket order with a GTD time-in-force policy. Trigger Bracket orders become active and wait to trigger based on the movement of the last trade price. The last trade price is the last price at which an order was filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.trigger_bracket_order_gtd_sell(_self_ , _client_order_id : str_, _product_id : str_, _base_size : str_, _limit_price : str_, _stop_trigger_price : str_, _end_time : str_, _self_trade_prevention_id : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → CreateOrderResponse
    

### **Trigger Bracket Order GTD Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders>

* * *

**Description:**

Place a SELL Trigger Bracket order with a GTD time-in-force policy. Trigger Bracket orders become active and wait to trigger based on the movement of the last trade price. The last trade price is the last price at which an order was filled.

* * *

**Read more on the official documentation:** [Create Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder)

coinbase.rest.RESTClient.get_order(_self_ , _order_id : str_, _** kwargs_) → GetOrderResponse
    

### **Get Order**

[GET] <https://api.coinbase.com/api/v3/brokerage/orders/historical>/{order_id}

* * *

**Description:**

Get a single order by order ID.

* * *

**Read more on the official documentation:** [Get Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_gethistoricalorder)

coinbase.rest.RESTClient.list_orders(_self_ , _order_ids : List[str] | None = None_, _product_ids : List[str] | None = None_, _order_status : List[str] | None = None_, _limit : int | None = None_, _start_date : str | None = None_, _end_date : str | None = None_, _order_types : str | None = None_, _order_side : str | None = None_, _cursor : str | None = None_, _product_type : str | None = None_, _order_placement_source : str | None = None_, _contract_expiry_type : str | None = None_, _asset_filters : List[str] | None = None_, _retail_portfolio_id : str | None = None_, _time_in_forces : str | None = None_, _sort_by : str | None = None_, _** kwargs_) → ListOrdersResponse
    

### **List Orders**

[GET] <https://api.coinbase.com/api/v3/brokerage/orders/historical/batch>

* * *

**Description:**

Get a list of orders filtered by optional query parameters (`product_id`, `order_status`, etc).

* * *

**Read more on the official documentation:** [List Orders](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_gethistoricalorders)

coinbase.rest.RESTClient.get_fills(_self_ , _order_ids : List[str] | None = None_, _trade_ids : List[str] | None = None_, _product_ids : List[str] | None = None_, _start_sequence_timestamp : str | None = None_, _end_sequence_timestamp : str | None = None_, _retail_portfolio_id : str | None = None_, _limit : int | None = None_, _cursor : str | None = None_, _sort_by : str | None = None_, _** kwargs_) → ListFillsResponse
    

### **List Fills**

[GET] <https://api.coinbase.com/api/v3/brokerage/orders/historical/fills>

* * *

**Description:**

Get a list of fills filtered by optional query parameters (`product_id`, `order_id`, etc).

* * *

**Read more on the official documentation:** [List Fills](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getfills)

coinbase.rest.RESTClient.edit_order(_self_ , _order_id : str_, _size : str | None = None_, _price : str | None = None_, _** kwargs_) → EditOrderResponse
    

### **Edit Order**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/edit>

* * *

**Description:**

Edit an order with a specified new `size`, or new `price`. Only limit order types, with time in force type of good-till-cancelled can be edited.

* * *

**Read more on the official documentation:** [Edit Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_editorder)

coinbase.rest.RESTClient.preview_edit_order(_self_ , _order_id : str_, _size : str | None = None_, _price : str | None = None_, _** kwargs_) → EditOrderPreviewResponse
    

### **Preview Edit Order**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/edit_preview>

* * *

**Description:**

Simulate an edit order request with a specified new `size`, or new `price`, to preview the result of an edit. Only limit order types, with time in force type of good-till-cancelled can be edited.

* * *

**Read more on the official documentation:** [Edit Order Preview](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeweditorder)

coinbase.rest.RESTClient.cancel_orders(_self_ , _order_ids : List[str]_, _** kwargs_) → CancelOrdersResponse
    

### **Cancel Orders**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/batch_cancel>

* * *

**Description:**

Initiate cancel requests for one or more orders.

* * *

**Read more on the official documentation:** [Cancel Orders](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_cancelorders)

coinbase.rest.RESTClient.preview_order(_self_ , _product_id : str_, _side : str_, _order_configuration_ , _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Order**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of an order request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_market_order(_self_ , _product_id : str_, _side : str_, _quote_size : str | None = None_, _base_size : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Market Order**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a market order request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_market_order_buy(_self_ , _product_id : str_, _quote_size : str | None = None_, _base_size : str | None = None_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Market Buy Order**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a market order buy request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_market_order_sell(_self_ , _product_id : str_, _base_size : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Market Sell Order**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a market order sell request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_limit_order_ioc(_self_ , _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Limit Order IOC**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a limit order IOC request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_limit_order_ioc_buy(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Limit Order IOC Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a limit order IOC buy request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_limit_order_ioc_sell(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Limit Order IOC Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a limit order IOC sell request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_limit_order_gtc(_self_ , _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _post_only : bool = False_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Limit Order GTC**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a limit order GTC request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_limit_order_gtc_buy(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _post_only : bool = False_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Limit Order GTC Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a limit order GTC buy request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_limit_order_gtc_sell(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _post_only : bool = False_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Limit Order GTC Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a limit order GTC sell request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_limit_order_gtd(_self_ , _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _end_time : str_, _post_only : bool = False_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Limit Order GTD**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a limit order GTD request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_limit_order_gtd_buy(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _end_time : str_, _post_only : bool = False_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Limit Order GTD Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a limit order GTD buy request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_limit_order_gtd_sell(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _end_time : str_, _post_only : bool = False_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Limit Order GTD Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a limit order GTD sell request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_limit_order_fok(_self_ , _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Limit Order FOK**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a limit order FOK request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_limit_order_fok_buy(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Limit Order FOK Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a limit order FOK buy request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_limit_order_fok_sell(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Limit Order FOK Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a limit order FOK sell request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_stop_limit_order_gtc(_self_ , _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _stop_price : str_, _stop_direction : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Stop-Limit Order GTC**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a stop limit GTC order request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_stop_limit_order_gtc_buy(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _stop_price : str_, _stop_direction : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Stop-Limit Order GTC Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a stop limit GTC order buy request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_stop_limit_order_gtc_sell(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _stop_price : str_, _stop_direction : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Stop-Limit Order GTC Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a stop limit GTC order sell request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_stop_limit_order_gtd(_self_ , _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _stop_price : str_, _end_time : str_, _stop_direction : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Stop-Limit Order GTD**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a stop limit GTD order request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_stop_limit_order_gtd_buy(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _stop_price : str_, _end_time : str_, _stop_direction : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Stop-Limit Order GTD Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a stop limit GTD order buy request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_stop_limit_order_gtd_sell(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _stop_price : str_, _end_time : str_, _stop_direction : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Stop-Limit Order GTD Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a stop limit GTD order sell request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_trigger_bracket_order_gtc(_self_ , _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _stop_trigger_price : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Trigger Bracket Order GTC**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a trigger bracket GTC order request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_trigger_bracket_order_gtc_buy(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _stop_trigger_price : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Trigger Bracket Order GTC Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a trigger bracket GTC order buy request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_trigger_bracket_order_gtc_sell(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _stop_trigger_price : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Trigger Bracket Order GTC Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a trigger bracket GTC order sell request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_trigger_bracket_order_gtd(_self_ , _product_id : str_, _side : str_, _base_size : str_, _limit_price : str_, _stop_trigger_price : str_, _end_time : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Trigger Bracket Order GTD**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a trigger bracket GTD order request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_trigger_bracket_order_gtd_buy(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _stop_trigger_price : str_, _end_time : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Trigger Bracket Order GTD Buy**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a trigger bracket GTD order buy request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.preview_trigger_bracket_order_gtd_sell(_self_ , _product_id : str_, _base_size : str_, _limit_price : str_, _stop_trigger_price : str_, _end_time : str_, _leverage : str | None = None_, _margin_type : str | None = None_, _retail_portfolio_id : str | None = None_, _** kwargs_) → PreviewOrderResponse
    

### **Preview Trigger Bracket Order GTD Sell**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/preview>

* * *

**Description:**

Preview the results of a trigger bracket GTD order sell request before sending.

* * *

**Read more on the official documentation:** [Preview Order](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_previeworder)

coinbase.rest.RESTClient.close_position(_self_ , _client_order_id : str_, _product_id : str_, _size : str | None = None_, _** kwargs_) → ClosePositionResponse
    

### **Close Position**

[POST] <https://api.coinbase.com/api/v3/brokerage/orders/close_position>

* * *

**Description:**

Places an order to close any open positions for a specified `product_id`.

* * *

**Read more on the official documentation:** [Close Position](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_closeposition)

## Portfolios

coinbase.rest.RESTClient.get_portfolios(_self_ , _portfolio_type : str | None = None_, _** kwargs_) → ListPortfoliosResponse
    

### **List Portfolios**

[GET] <https://api.coinbase.com/api/v3/brokerage/portfolios>

* * *

**Description:**

Get a list of all portfolios of a user.

* * *

**Read more on the official documentation:** [List Portfolios](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getportfolios)

coinbase.rest.RESTClient.create_portfolio(_self_ , _name : str_, _** kwargs_) → CreatePortfolioResponse
    

### **Create Portfolio**

[POST] <https://api.coinbase.com/api/v3/brokerage/portfolios>

* * *

**Description:**

Create a portfolio.

* * *

**Read more on the official documentation:** [Create Portfolio](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_createportfolio)

coinbase.rest.RESTClient.get_portfolio_breakdown(_self_ , _portfolio_uuid : str_, _currency : str | None = None_, _** kwargs_) → GetPortfolioBreakdownResponse
    

### **Get Portfolio Breakdown**

[GET] <https://api.coinbase.com/api/v3/brokerage/portfolios>/{portfolio_uuid}

* * *

**Description:**

Get the breakdown of a portfolio by portfolio ID.

* * *

**Read more on the official documentation:** [Get Portfolio Breakdown](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getportfoliobreakdown)

coinbase.rest.RESTClient.move_portfolio_funds(_self_ , _value : str_, _currency : str_, _source_portfolio_uuid : str_, _target_portfolio_uuid : str_, _** kwargs_) → MovePortfolioFundsResponse
    

### **Move Portfolio Funds**

[POST] <https://api.coinbase.com/api/v3/brokerage/portfolios/move_funds>

* * *

**Description:**

Transfer funds between portfolios.

* * *

**Read more on the official documentation:** [Move Portfolio Funds](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_moveportfoliofunds)

coinbase.rest.RESTClient.edit_portfolio(_self_ , _portfolio_uuid : str_, _name : str_, _** kwargs_) → EditPortfolioResponse
    

### **Edit Portfolio**

[PUT] <https://api.coinbase.com/api/v3/brokerage/portfolios>/{portfolio_uuid}

* * *

**Description:**

Modify a portfolio by portfolio ID.

* * *

**Read more on the official documentation:** [Edit Portfolio](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_editportfolio)

coinbase.rest.RESTClient.delete_portfolio(_self_ , _portfolio_uuid : str_, _** kwargs_) → DeletePortfolioResponse
    

### **Delete Portfolio**

[DELETE] <https://api.coinbase.com/api/v3/brokerage/portfolios>/{portfolio_uuid}

* * *

**Description:**

Delete a portfolio by portfolio ID.

* * *

**Read more on the official documentation:** [Delete Portfolio](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_deleteportfolio)

## Futures

coinbase.rest.RESTClient.get_futures_balance_summary(_self_ , _** kwargs_) → GetFuturesBalanceSummaryResponse
    

### **Get Futures Balance Summary**

[GET] <https://api.coinbase.com/api/v3/brokerage/cfm/balance_summary>

* * *

**Description:**

Get information on your balances related to [Coinbase Financial Markets](https://www.coinbase.com/fcm) (CFM) futures trading.

* * *

**Read more on the official documentation:** [Get Futures Balance Summary](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getfcmbalancesummary)

coinbase.rest.RESTClient.list_futures_positions(_self_ , _** kwargs_) → ListFuturesPositionsResponse
    

### **List Futures Positions**

[GET] <https://api.coinbase.com/api/v3/brokerage/cfm/positions>

* * *

**Description:**

Get a list of all open positions in CFM futures products.

* * *

**Read more on the official documentation:** [List Futures Positions](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getfcmpositions)

coinbase.rest.RESTClient.get_futures_position(_self_ , _product_id : str_, _** kwargs_) → GetFuturesPositionResponse
    

### **Get Futures Position**

[GET] <https://api.coinbase.com/api/v3/brokerage/cfm/positions>/{product_id}

* * *

**Description:**

Get the position of a specific CFM futures product.

* * *

**Read more on the official documentation:** [Get Futures Position](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getfcmposition)

coinbase.rest.RESTClient.schedule_futures_sweep(_self_ , _usd_amount : str_, _** kwargs_) → ScheduleFuturesSweepResponse
    

### **Schedule Futures Sweep**

[POST] <https://api.coinbase.com/api/v3/brokerage/cfm/sweeps/schedule>

* * *

**Description:**

Schedule a sweep of funds from your CFTC-regulated futures account to your Coinbase Inc. USD Spot wallet.

* * *

**Read more on the official documentation:** [Schedule Futures Sweep](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_schedulefcmsweep)

coinbase.rest.RESTClient.list_futures_sweeps(_self_ , _** kwargs_) → ListFuturesSweepsResponse
    

### **List Futures Sweeps**

[GET] <https://api.coinbase.com/api/v3/brokerage/cfm/sweeps>

* * *

**Description:**

Get information on your pending and/or processing requests to sweep funds from your CFTC-regulated futures account to your Coinbase Inc. USD Spot wallet.

* * *

**Read more on the official documentation:** [List Futures Sweeps](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getfcmsweeps)

coinbase.rest.RESTClient.cancel_pending_futures_sweep(_self_ , _** kwargs_) → CancelPendingFuturesSweepResponse
    

### **Cancel Pending Futures Sweep**

[DELETE] <https://api.coinbase.com/api/v3/brokerage/cfm/sweeps>

* * *

**Description:**

Cancel your pending sweep of funds from your CFTC-regulated futures account to your Coinbase Inc. USD Spot wallet.

* * *

**Read more on the official documentation:** [Cancel Pending Futures Sweep](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_cancelfcmsweep)

coinbase.rest.RESTClient.get_intraday_margin_setting(_self_ , _** kwargs_) → GetIntradayMarginSettingResponse
    

### **Get Intraday Margin Setting**

[GET] <https://api.coinbase.com/api/v3/brokerage/cfm/intraday/margin_setting>

* * *

**Description:**

Get the status of whether your account is opted in to receive increased leverage on futures trades on weekdays from 8am-4pm ET.

* * *

**Read more on the official documentation:** [Get Intraday Margin Setting](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getintradaymarginsetting)

coinbase.rest.RESTClient.get_current_margin_window(_self_ , _margin_profile_type : str_, _** kwargs_) → GetCurrentMarginWindowResponse
    

### **Get Current Margin Window**

[GET] <https://api.coinbase.com/api/v3/brokerage/cfm/intraday/current_margin_window>

* * *

**Description:**

Get the current margin window to determine whether intraday or overnight margin rates are in effect.

* * *

**Read more on the official documentation:** [Get Current Margin Window](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getcurrentmarginwindow)

coinbase.rest.RESTClient.set_intraday_margin_setting(_self_ , _setting : str_, _** kwargs_) → SetIntradayMarginSettingResponse
    

### **Set Intraday Margin Setting**

[POST] <https://api.coinbase.com/api/v3/brokerage/cfm/intraday/margin_setting>

* * *

**Description:**

Opt in to receive increased leverage on futures trades on weekdays from 8am-4pm ET.

* * *

**Read more on the official documentation:** [Set Intraday Margin Setting](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_setintradaymarginsetting)

## Perpetuals

coinbase.rest.RESTClient.allocate_portfolio(_self_ , _portfolio_uuid : str_, _symbol : str_, _amount : str_, _currency : str_, _** kwargs_) → AllocatePortfolioResponse
    

### **Allocate Portfolio**

[POST] <https://api.coinbase.com/api/v3/brokerage/intx/allocate>

* * *

**Description:**

Allocate more funds to an isolated position in your Perpetuals portfolio.

* * *

**Read more on the official documentation:** [Allocate Portfolio](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_allocateportfolio)

coinbase.rest.RESTClient.get_perps_portfolio_summary(_self_ , _portfolio_uuid : str_, _** kwargs_) → GetPerpetualsPortfolioSummaryResponse
    

### **Get Perpetuals Portfolio Summary**

[GET] <https://api.coinbase.com/api/v3/brokerage/intx/portfolio>/{portfolio_uuid}

* * *

**Description:**

Get a summary of your Perpetuals portfolio.

* * *

**Read more on the official documentation:** [Get Perpetuals Portfolio Summary](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getintxportfoliosummary)

coinbase.rest.RESTClient.list_perps_positions(_self_ , _portfolio_uuid : str_, _** kwargs_) → ListPerpetualsPositionsResponse
    

### **List Perpetuals Positions**

[GET] <https://api.coinbase.com/api/v3/brokerage/intx/positions>/{portfolio_uuid}

* * *

**Description:**

Get a list of open positions in your Perpetuals portfolio.

* * *

**Read more on the official documentation:** [List Perpetuals Positions](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getintxpositions)

coinbase.rest.RESTClient.get_perps_position(_self_ , _portfolio_uuid : str_, _symbol : str_, _** kwargs_) → GetPerpetualsPositionResponse
    

### **Get Perpetuals Position**

[GET] <https://api.coinbase.com/api/v3/brokerage/intx/positions>/{portfolio_uuid}/{symbol}

* * *

**Description:**

Get a specific open position in your Perpetuals portfolio

* * *

**Read more on the official documentation:** [Get Perpetuals Positions](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getintxposition)

coinbase.rest.RESTClient.get_perps_portfolio_balances(_self_ , _portfolio_uuid : str_, _** kwargs_) → GetPortfolioBalancesResponse
    

### **Get Portfolio Balances**

[GET] <https://api.coinbase.com/api/v3/brokerage/intx/balances>/{portfolio_uuid}

* * *

**Description:**

Get a list of asset balances on Intx for a given Portfolio

* * *

**Read more on the official documentation:** [Get Portfolio Balances](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getintxbalances)

coinbase.rest.RESTClient.opt_in_or_out_multi_asset_collateral(_self_ , _portfolio_uuid : str_, _multi_asset_collateral_enabled : bool_, _** kwargs_) → OptInOutMultiAssetCollateralResponse
    

### **Opt In or Out of Multi Asset Collateral**

[POST] <https://api.coinbase.com/api/v3/brokerage/intx/multi_asset_collateral>

* * *

**Description:**

Enable or Disable Multi Asset Collateral for a given Portfolio.

* * *

**Read more on the official documentation:** [Opt In or Out of Multi Asset Collateral](https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_intxmultiassetcollateral)

## Fees

coinbase.rest.RESTClient.get_transaction_summary(_self_ , _product_type : str | None = None_, _contract_expiry_type : str | None = None_, _product_venue : str | None = None_, _** kwargs_) → GetTransactionSummaryResponse
    

### **Get Transactions Summary**

[GET] <https://api.coinbase.com/api/v3/brokerage/transaction_summary>

* * *

**Description:**

Get a summary of transactions with fee tiers, total volume, and fees.

* * *

**Read more on the official documentation:** [Get Transaction Summary](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_gettransactionsummary)

## Converts

coinbase.rest.RESTClient.create_convert_quote(_self_ , _from_account : str_, _to_account : str_, _amount : str_, _user_incentive_id : str | None = None_, _code_val : str | None = None_, _** kwargs_) → CreateConvertQuoteResponse
    

### **Create Convert Quote**

[POST] <https://api.coinbase.com/api/v3/brokerage/convert/quote>

* * *

**Description:**

Create a convert quote with a specified source currency, target currency, and amount.

* * *

**Read more on the official documentation:** [Create Convert Quote](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_createconvertquote)

coinbase.rest.RESTClient.get_convert_trade(_self_ , _trade_id : str_, _from_account : str_, _to_account : str_, _** kwargs_) → GetConvertTradeResponse
    

### **Get Convert Trade**

[GET] <https://api.coinbase.com/api/v3/brokerage/convert/trade>/{trade_id}

* * *

**Description:**

Gets a list of information about a convert trade with a specified trade ID, source currency, and target currency.

* * *

**Read more on the official documentation:** [Get Convert Trade](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getconverttrade)

coinbase.rest.RESTClient.commit_convert_trade(_self_ , _trade_id : str_, _from_account : str_, _to_account : str_, _** kwargs_) → CommitConvertTradeResponse
    

### **Commit Convert Trade**

[POST] <https://api.coinbase.com/api/v3/brokerage/convert/trade>/{trade_id}

* * *

**Description:**

Commits a convert trade with a specified trade ID, source currency, and target currency.

* * *

**Read more on the official documentation:** [Commit Convert Trade](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_commitconverttrade)

## Public

coinbase.rest.RESTClient.get_unix_time(_self_ , _** kwargs_) → GetServerTimeResponse
    

### **Get Server Time**

[GET] <https://api.coinbase.com/api/v3/brokerage/time>

* * *

**Description:**

Get the current time from the Coinbase Advanced API. This is a public endpoint.

* * *

**Read more on the official documentation:** [Get Server Time](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getservertime)

coinbase.rest.RESTClient.get_public_product_book(_self_ , _product_id : str_, _limit : int | None = None_, _aggregation_price_increment : str | None = None_, _** kwargs_) → GetProductBookResponse
    

### **Get Public Product Book**

[GET] <https://api.coinbase.com/api/v3/brokerage/market/product_book>

* * *

**Description:**

Get a list of bids/asks for a single product. The amount of detail shown can be customized with the limit parameter.

* * *

**API Key Permissions:**

This endpoint is public and does not need authentication.

* * *

**Read more on the official documentation:** [Get Public Product Book](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getpublicproductbook)

coinbase.rest.RESTClient.get_public_products(_self_ , _limit : int | None = None_, _offset : int | None = None_, _product_type : str | None = None_, _product_ids : List[str] | None = None_, _contract_expiry_type : str | None = None_, _expiring_contract_status : str | None = None_, _get_all_products : bool = False_, _** kwargs_) → ListProductsResponse
    

### **List Public Products**

[GET] <https://api.coinbase.com/api/v3/brokerage/market/products>

* * *

**Description:**

Get a list of the available currency pairs for trading.

* * *

**API Key Permissions:**

This endpoint is public and does not need authentication.

* * *

**Read more on the official documentation:** [List Public Products](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getpublicproducts)

coinbase.rest.RESTClient.get_public_product(_self_ , _product_id : str_, _** kwargs_) → GetProductResponse
    

### **Public Get Product**

[GET] <https://api.coinbase.com/api/v3/brokerage/market/products>/{product_id}

* * *

**Description:**

Get information on a single product by product ID.

* * *

**API Key Permissions:**

This endpoint is public and does not need authentication.

* * *

**Read more on the official documentation:** [Get Public Product](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getpublicproduct)

coinbase.rest.RESTClient.get_public_candles(_self_ , _product_id : str_, _start : str_, _end : str_, _granularity : str_, _limit : int | None = None_, _** kwargs_) → GetProductCandlesResponse
    

### **Get Public Product Candles**

[GET] <https://api.coinbase.com/api/v3/brokerage/market/products>/{product_id}/candles

* * *

**Description:**

Get rates for a single product by product ID, grouped in buckets.

* * *

**API Key Permissions:**

This endpoint is public and does not need authentication.

* * *

**Read more on the official documentation:** [Get Public Product Candles](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getpubliccandles)

coinbase.rest.RESTClient.get_public_market_trades(_self_ , _product_id : str_, _limit : int_, _start : str | None = None_, _end : str | None = None_, _** kwargs_) → GetMarketTradesResponse
    

### **Get Public Market Trades**

[GET] <https://api.coinbase.com/api/v3/brokerage/market/products>/{product_id}/ticker

* * *

**Description:**

Get snapshot information, by product ID, about the last trades (ticks), best bid/ask, and 24h volume.

* * *

**API Key Permissions:**

This endpoint is public and does not need authentication.

* * *

**Read more on the official documentation:** [Get Public Market Trades](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getpublicmarkettrades)

## Payments

coinbase.rest.RESTClient.list_payment_methods(_self_ , _** kwargs_) → ListPaymentMethodsResponse
    

### **List Payment Methods**

[GET] <https://api.coinbase.com/api/v3/brokerage/payment_methods>

* * *

**Description:**

Get a list of payment methods for the current user.

* * *

**Read more on the official documentation:** [List Payment Methods](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getpaymentmethods)

coinbase.rest.RESTClient.get_payment_method(_self_ , _payment_method_id : str_, _** kwargs_) → GetPaymentMethodResponse
    

### **Get Payment Method**

[GET] <https://api.coinbase.com/api/v3/brokerage/payment_methods>/{payment_method_id}

* * *

**Description:**

Get information about a payment method for the current user.

* * *

**Read more on the official documentation:** [Get Payment Method](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getpaymentmethod)

## Data API

coinbase.rest.RESTClient.get_api_key_permissions(_self_ , _** kwargs_) → GetAPIKeyPermissionsResponse
    

### **Get Api Key Permissions**

[GET] <https://api.coinbase.com/api/v3/brokerage/key_permissions>

* * *

**Description:**

Get information about your CDP API key permissions

* * *

**Read more on the official documentation:** [Create Convert Quote](https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getapikeypermissions)



================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/_modules/coinbase/rest.html
================================================================================

# Source code for coinbase.rest
    
    
    from .rest_base import RESTBase
    
    
    
    
    
    [[docs]](../../coinbase.rest.html#coinbase.rest.RESTClient)
    class RESTClient(RESTBase):
        """
        **RESTClient**
        _____________________________
    
        Initialize using RESTClient
    
        __________
    
        **Parameters**:
    
        - **api_key | Optional (str)** - The API key
        - **api_secret | Optional (str)** - The API key secret
        - **key_file | Optional (IO | str)** - Path to API key file or file-like object
        - **base_url | (str)** - The base URL for REST requests. Default set to "https://api.coinbase.com"
        - **timeout | Optional (int)** - Set timeout in seconds for REST requests
        - **verbose | Optional (bool)** - Enables debug logging. Default set to False
        - **rate_limit_headers | Optional (bool)** - Enables rate limit headers. Default set to False
    
        """
    
        from .accounts import get_account, get_accounts
        from .convert import commit_convert_trade, create_convert_quote, get_convert_trade
        from .data_api import get_api_key_permissions
        from .fees import get_transaction_summary
        from .futures import (
            cancel_pending_futures_sweep,
            get_current_margin_window,
            get_futures_balance_summary,
            get_futures_position,
            get_intraday_margin_setting,
            list_futures_positions,
            list_futures_sweeps,
            schedule_futures_sweep,
            set_intraday_margin_setting,
        )
        from .market_data import get_candles, get_market_trades
        from .orders import (
            cancel_orders,
            close_position,
            create_order,
            edit_order,
            get_fills,
            get_order,
            limit_order_fok,
            limit_order_fok_buy,
            limit_order_fok_sell,
            limit_order_gtc,
            limit_order_gtc_buy,
            limit_order_gtc_sell,
            limit_order_gtd,
            limit_order_gtd_buy,
            limit_order_gtd_sell,
            limit_order_ioc,
            limit_order_ioc_buy,
            limit_order_ioc_sell,
            list_orders,
            market_order,
            market_order_buy,
            market_order_sell,
            preview_edit_order,
            preview_limit_order_fok,
            preview_limit_order_fok_buy,
            preview_limit_order_fok_sell,
            preview_limit_order_gtc,
            preview_limit_order_gtc_buy,
            preview_limit_order_gtc_sell,
            preview_limit_order_gtd,
            preview_limit_order_gtd_buy,
            preview_limit_order_gtd_sell,
            preview_limit_order_ioc,
            preview_limit_order_ioc_buy,
            preview_limit_order_ioc_sell,
            preview_market_order,
            preview_market_order_buy,
            preview_market_order_sell,
            preview_order,
            preview_stop_limit_order_gtc,
            preview_stop_limit_order_gtc_buy,
            preview_stop_limit_order_gtc_sell,
            preview_stop_limit_order_gtd,
            preview_stop_limit_order_gtd_buy,
            preview_stop_limit_order_gtd_sell,
            preview_trigger_bracket_order_gtc,
            preview_trigger_bracket_order_gtc_buy,
            preview_trigger_bracket_order_gtc_sell,
            preview_trigger_bracket_order_gtd,
            preview_trigger_bracket_order_gtd_buy,
            preview_trigger_bracket_order_gtd_sell,
            stop_limit_order_gtc,
            stop_limit_order_gtc_buy,
            stop_limit_order_gtc_sell,
            stop_limit_order_gtd,
            stop_limit_order_gtd_buy,
            stop_limit_order_gtd_sell,
            trigger_bracket_order_gtc,
            trigger_bracket_order_gtc_buy,
            trigger_bracket_order_gtc_sell,
            trigger_bracket_order_gtd,
            trigger_bracket_order_gtd_buy,
            trigger_bracket_order_gtd_sell,
        )
        from .payments import get_payment_method, list_payment_methods
        from .perpetuals import (
            allocate_portfolio,
            get_perps_portfolio_balances,
            get_perps_portfolio_summary,
            get_perps_position,
            list_perps_positions,
            opt_in_or_out_multi_asset_collateral,
        )
        from .portfolios import (
            create_portfolio,
            delete_portfolio,
            edit_portfolio,
            get_portfolio_breakdown,
            get_portfolios,
            move_portfolio_funds,
        )
        from .products import get_best_bid_ask, get_product, get_product_book, get_products
        from .public import (
            get_public_candles,
            get_public_market_trades,
            get_public_product,
            get_public_product_book,
            get_public_products,
            get_unix_time,
        )
    
    
    
    



================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/_modules/coinbase/websocket/websocket_base.html
================================================================================

# Source code for coinbase.websocket.websocket_base
    
    
    import asyncio
    import json
    import logging
    import os
    import ssl
    import threading
    import time
    from multiprocessing import AuthenticationError
    from typing import IO, Callable, List, Optional, Union
    
    import backoff
    import websockets
    
    from coinbase import jwt_generator
    from coinbase.api_base import APIBase, get_logger
    from coinbase.constants import (
        API_ENV_KEY,
        API_SECRET_ENV_KEY,
        SUBSCRIBE_MESSAGE_TYPE,
        UNSUBSCRIBE_MESSAGE_TYPE,
        USER_AGENT,
        WS_AUTH_CHANNELS,
        WS_BASE_URL,
        WS_RETRY_BASE,
        WS_RETRY_FACTOR,
        WS_RETRY_MAX,
    )
    
    logger = get_logger("coinbase.WSClient")
    
    
    
    
    
    [[docs]](../../../coinbase.websocket.html#coinbase.websocket.WSClientException)
    class WSClientException(Exception):
        """
        **WSClientException**
        ________________________________________
    
        -----------------------------------------
    
        Exception raised for errors in the WebSocket client.
        """
    
        pass
    
    
    
    
    
    
    
    
    [[docs]](../../../coinbase.websocket.html#coinbase.websocket.WSClientConnectionClosedException)
    class WSClientConnectionClosedException(Exception):
        """
        **WSClientConnectionClosedException**
        ________________________________________
    
        ----------------------------------------
    
        Exception raised for unexpected closure in the WebSocket client.
        """
    
        pass
    
    
    
    
    
    class WSBase(APIBase):
        """
        :meta private:
        """
    
        def __init__(
            self,
            api_key: Optional[str] = os.getenv(API_ENV_KEY),
            api_secret: Optional[str] = os.getenv(API_SECRET_ENV_KEY),
            key_file: Optional[Union[IO, str]] = None,
            base_url=WS_BASE_URL,
            timeout: Optional[int] = None,
            max_size: Optional[int] = 10 * 1024 * 1024,
            on_message: Optional[Callable[[str], None]] = None,
            on_open: Optional[Callable[[], None]] = None,
            on_close: Optional[Callable[[], None]] = None,
            retry: Optional[bool] = True,
            verbose: Optional[bool] = False,
        ):
            super().__init__(
                api_key=api_key,
                api_secret=api_secret,
                key_file=key_file,
                base_url=base_url,
                timeout=timeout,
                verbose=verbose,
            )
    
            if not on_message:
                raise WSClientException("on_message callback is required.")
    
            if verbose:
                logger.setLevel(logging.DEBUG)
    
            self.max_size = max_size
            self.on_message = on_message
            self.on_open = on_open
            self.on_close = on_close
    
            self.websocket = None
            self.loop = None
            self.thread = None
            self._task = None
    
            self.retry = retry
            self._retry_max_tries = WS_RETRY_MAX
            self._retry_base = WS_RETRY_BASE
            self._retry_factor = WS_RETRY_FACTOR
            self._retry_count = 0
    
            self.subscriptions = {}
            self._background_exception = None
            self._retrying = False
    
        def open(self) -> None:
            """
            **Open Websocket**
            __________________
    
            ------------------------
    
            Open the websocket client connection.
            """
            if not self.loop or self.loop.is_closed():
                self.loop = asyncio.new_event_loop()  # Create a new event loop
                self.thread = threading.Thread(target=self.loop.run_forever)
                self.thread.daemon = True
                self.thread.start()
    
            self._run_coroutine_threadsafe(self.open_async())
    
        async def open_async(self) -> None:
            """
            **Open Websocket Async**
            ________________________
    
            ------------------------
    
            Open the websocket client connection asynchronously.
            """
            self._ensure_websocket_not_open()
    
            headers = self._set_headers()
    
            logger.debug("Connecting to %s", self.base_url)
            try:
                self.websocket = await websockets.connect(
                    self.base_url,
                    open_timeout=self.timeout,
                    max_size=self.max_size,
                    user_agent_header=USER_AGENT,
                    extra_headers=headers,
                    ssl=ssl.SSLContext() if self.base_url.startswith("wss://") else None,
                )
                logger.debug("Successfully connected to %s", self.base_url)
    
                if self.on_open:
                    self.on_open()
    
                # Start the message handler coroutine after establishing connection
                if not self._retrying:
                    self._task = asyncio.create_task(self._message_handler())
    
            except asyncio.TimeoutError as toe:
                self.websocket = None
                logger.error("Connection attempt timed out: %s", toe)
                raise WSClientException("Connection attempt timed out") from toe
            except (websockets.exceptions.WebSocketException, OSError) as wse:
                self.websocket = None
                logger.error("Failed to establish WebSocket connection: %s", wse)
                raise WSClientException("Failed to establish WebSocket connection") from wse
    
        def close(self) -> None:
            """
            **Close Websocket**
            ___________________
    
            ------------------------
    
            Close the websocket client connection.
            """
            if self.loop and not self.loop.is_closed():
                # Schedule the asynchronous close
                self._run_coroutine_threadsafe(self.close_async())
                # Stop the event loop
                self.loop.call_soon_threadsafe(self.loop.stop)
                # Wait for the thread to finish
                self.thread.join()
                # Close the event loop
                self.loop.close()
            else:
                raise WSClientException("Event loop is not running.")
    
        async def close_async(self) -> None:
            """
            **Close Websocket Async**
            _________________________
    
            ------------------------
    
            Close the websocket client connection asynchronously.
            """
            self._ensure_websocket_open()
    
            logger.debug("Closing connection to %s", self.base_url)
            try:
                await self.websocket.close()
                self.websocket = None
                self.subscriptions = {}
    
                logger.debug("Connection closed to %s", self.base_url)
    
                if self.on_close:
                    self.on_close()
            except (websockets.exceptions.WebSocketException, OSError) as wse:
                logger.error("Failed to close WebSocket connection: %s", wse)
                raise WSClientException("Failed to close WebSocket connection.") from wse
    
        def subscribe(self, product_ids: List[str], channels: List[str]) -> None:
            """
            **Subscribe**
            _____________
    
            ------------------------
    
            Subscribe to a list of channels for a list of product ids.
    
            - **product_ids** - product ids to subscribe to
            - **channels** - channels to subscribe to
            """
            if self.loop and not self.loop.is_closed():
                self._run_coroutine_threadsafe(self.subscribe_async(product_ids, channels))
            else:
                raise WSClientException("Websocket Client is not open.")
    
        async def subscribe_async(
            self, product_ids: List[str], channels: List[str]
        ) -> None:
            """
            **Subscribe Async**
            ___________________
    
            ------------------------
    
            Async subscribe to a list of channels for a list of product ids.
    
            - **product_ids** - product ids to subscribe to
            - **channels** - channels to subscribe to
            """
            self._ensure_websocket_open()
            for channel in channels:
                try:
                    if not self.is_authenticated and channel in WS_AUTH_CHANNELS:
                        raise AuthenticationError(
                            "Unauthenticated request to private channel."
                        )
    
                    is_public = False if channel in WS_AUTH_CHANNELS else True
                    message = self._build_subscription_message(
                        product_ids, channel, SUBSCRIBE_MESSAGE_TYPE, is_public
                    )
                    json_message = json.dumps(message)
    
                    logger.debug(
                        "Subscribing to channel %s for product IDs: %s",
                        channel,
                        product_ids,
                    )
    
                    await self.websocket.send(json_message)
    
                    logger.debug("Successfully sent subscription message.")
    
                    # add to subscriptions map
                    if channel not in self.subscriptions:
                        self.subscriptions[channel] = set()
                    self.subscriptions[channel].update(product_ids)
                except websockets.exceptions.WebSocketException as wse:
                    logger.error(
                        "Failed to subscribe to %s channel for product IDs %s: %s",
                        channel,
                        product_ids,
                        wse,
                    )
                    raise WSClientException(
                        f"Failed to subscribe to {channel} channel for product ids {product_ids}."
                    ) from wse
    
        def unsubscribe(self, product_ids: List[str], channels: List[str]) -> None:
            """
            **Unsubscribe**
            _______________
    
            ------------------------
    
            Unsubscribe to a list of channels for a list of product ids.
    
            - **product_ids** - product ids to unsubscribe from
            - **channels** - channels to unsubscribe from
            """
            if self.loop and not self.loop.is_closed():
                self._run_coroutine_threadsafe(
                    self.unsubscribe_async(product_ids, channels)
                )
            else:
                raise WSClientException("Websocket Client is not open.")
    
        async def unsubscribe_async(
            self, product_ids: List[str], channels: List[str]
        ) -> None:
            """
            **Unsubscribe Async**
            _____________________
    
            ------------------------
    
            Async unsubscribe to a list of channels for a list of product ids.
    
            - **product_ids** - product ids to unsubscribe from
            - **channels** - channels to unsubscribe from
            """
            self._ensure_websocket_open()
            for channel in channels:
                try:
                    if not self.is_authenticated and channel in WS_AUTH_CHANNELS:
                        raise AuthenticationError(
                            "Unauthenticated request to private channel. If you wish to access private channels, you must provide your API key and secret when initializing the WSClient."
                        )
                    is_public = False if channel in WS_AUTH_CHANNELS else True
                    message = self._build_subscription_message(
                        product_ids, channel, UNSUBSCRIBE_MESSAGE_TYPE, is_public
                    )
                    json_message = json.dumps(message)
    
                    logger.debug(
                        "Unsubscribing from channel %s for product IDs: %s",
                        channel,
                        product_ids,
                    )
    
                    await self.websocket.send(json_message)
    
                    logger.debug("Successfully sent unsubscribe message.")
    
                    # remove from subscriptions map
                    if channel in self.subscriptions:
                        self.subscriptions[channel].difference_update(product_ids)
                except (websockets.exceptions.WebSocketException, OSError) as wse:
                    logger.error(
                        "Failed to unsubscribe to %s channel for product IDs %s: %s",
                        channel,
                        product_ids,
                        wse,
                    )
    
                    raise WSClientException(
                        f"Failed to unsubscribe to {channel} channel for product ids {product_ids}."
                    ) from wse
    
        def unsubscribe_all(self) -> None:
            """
            **Unsubscribe All**
            ________________________
    
            ------------------------
    
            Unsubscribe from all channels you are currently subscribed to.
            """
            if self.loop and not self.loop.is_closed():
                self._run_coroutine_threadsafe(self.unsubscribe_all_async())
            else:
                raise WSClientException("Websocket Client is not open.")
    
        async def unsubscribe_all_async(self) -> None:
            """
            **Unsubscribe All Async**
            _________________________
    
            ------------------------
    
            Async unsubscribe from all channels you are currently subscribed to.
            """
            for channel, product_ids in self.subscriptions.items():
                if product_ids:
                    await self.unsubscribe_async(list(product_ids), [channel])
    
        def sleep_with_exception_check(self, sleep: int) -> None:
            """
            **Sleep with Exception Check**
            ______________________________
    
            ------------------------
    
            Sleep for a specified number of seconds and check for background exceptions.
    
            - **sleep** - number of seconds to sleep.
            """
            time.sleep(sleep)
            self.raise_background_exception()
    
        async def sleep_with_exception_check_async(self, sleep: int) -> None:
            """
            **Sleep with Exception Check Async**
            ____________________________________
    
            ------------------------
    
            Async sleep for a specified number of seconds and check for background exceptions.
    
            - **sleep** - number of seconds to sleep.
            """
            await asyncio.sleep(sleep)
            self.raise_background_exception()
    
        def run_forever_with_exception_check(self) -> None:
            """
            **Run Forever with Exception Check**
            ____________________________________
    
            ------------------------
    
            Runs an endless loop, checking for background exceptions every second.
            """
            while True:
                time.sleep(1)
                self.raise_background_exception()
    
        async def run_forever_with_exception_check_async(self) -> None:
            """
            **Run Forever with Exception Check Async**
            __________________________________________
    
            ------------------------
    
            Async runs an endless loop, checking for background exceptions every second.
            """
            while True:
                await asyncio.sleep(1)
                self.raise_background_exception()
    
        def raise_background_exception(self) -> None:
            """
            **Raise Background Exception**
            ______________________________
    
            ------------------------
    
            Raise any background exceptions that occurred in the message handler.
            """
            if self._background_exception:
                exception_to_raise = self._background_exception
                self._background_exception = None
                raise exception_to_raise
    
        def _run_coroutine_threadsafe(self, coro):
            """
            :meta private:
            """
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            return future.result()
    
        def _is_websocket_open(self):
            """
            :meta private:
            """
            return self.websocket and self.websocket.open
    
        async def _resubscribe(self):
            """
            :meta private:
            """
            for channel, product_ids in self.subscriptions.items():
                if product_ids:
                    await self.subscribe_async(list(product_ids), [channel])
    
        async def _retry_connection(self):
            """
            :meta private:
            """
            self._retry_count = 0
    
            @backoff.on_exception(
                backoff.expo,
                WSClientException,
                max_tries=self._retry_max_tries,
                base=self._retry_base,
                factor=self._retry_factor,
            )
            async def _retry_connect_and_resubscribe():
                self._retry_count += 1
    
                logger.debug("Retrying connection attempt %s", self._retry_count)
                if not self._is_websocket_open():
                    await self.open_async()
    
                logger.debug("Resubscribing to channels")
                self._retry_count = 0
                await self._resubscribe()
    
            return await _retry_connect_and_resubscribe()
    
        async def _message_handler(self):
            """
            :meta private:
            """
            self.handler_open = True
            while self._is_websocket_open():
                try:
                    message = await self.websocket.recv()
                    if self.on_message:
                        self.on_message(message)
                except websockets.exceptions.ConnectionClosedOK as cco:
                    logger.debug("Connection closed (OK): %s", cco)
                    break
                except websockets.exceptions.ConnectionClosedError as cce:
                    logger.error("Connection closed (ERROR): %s", cce)
                    if self.retry:
                        self._retrying = True
                        try:
                            logger.debug("Retrying connection")
                            await self._retry_connection()
                            self._retrying = False
                        except WSClientException:
                            logger.error(
                                "Connection closed unexpectedly. Retry attempts failed."
                            )
                            self._background_exception = WSClientConnectionClosedException(
                                "Connection closed unexpectedly. Retry attempts failed."
                            )
                            self.subscriptions = {}
                            self._retrying = False
                            self._retry_count = 0
                            break
                    else:
                        logger.error("Connection closed unexpectedly with error: %s", cce)
                        self._background_exception = WSClientConnectionClosedException(
                            f"Connection closed unexpectedly with error: {cce}"
                        )
                        self.subscriptions = {}
                        break
                except (
                    websockets.exceptions.WebSocketException,
                    json.JSONDecodeError,
                    WSClientException,
                ) as e:
                    logger.error("Exception in message handler: %s", e)
                    self._background_exception = WSClientException(
                        f"Exception in message handler: {e}"
                    )
                    break
    
        def _build_subscription_message(
            self, product_ids: List[str], channel: str, message_type: str, public: bool
        ):
            """
            :meta private:
            """
            return {
                "type": message_type,
                "product_ids": product_ids,
                "channel": channel,
                **(
                    {
                        "jwt": jwt_generator.build_ws_jwt(self.api_key, self.api_secret),
                    }
                    if self.is_authenticated
                    else {}
                ),
            }
    
        def _ensure_websocket_not_open(self):
            """
            :meta private:
            """
            if self._is_websocket_open():
                raise WSClientException("WebSocket is already open.")
    
        def _ensure_websocket_open(self):
            """
            :meta private:
            """
            if not self._is_websocket_open():
                raise WSClientException("WebSocket is closed or was never opened.")
    
        def _set_headers(self):
            """
            :meta private:
            """
            if self._retry_count > 0:
                return {"x-cb-retry-counter": str(self._retry_count)}
            else:
                return {}
    



================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/_modules/coinbase/websocket.html
================================================================================

# Source code for coinbase.websocket
    
    
    import os
    from typing import IO, Callable, Optional, Union
    
    from coinbase.constants import API_ENV_KEY, API_SECRET_ENV_KEY, WS_USER_BASE_URL
    
    from .types.websocket_response import WebsocketResponse
    from .websocket_base import WSBase, WSClientConnectionClosedException, WSClientException
    
    
    
    
    
    [[docs]](../../coinbase.websocket.html#coinbase.websocket.WSClient)
    class WSClient(WSBase):
        """
        **WSClient**
        _____________________________
    
        Initialize using WSClient
    
        __________
    
        **Parameters**:
    
        - **api_key | Optional (str)** - The API key
        - **api_secret | Optional (str)** - The API key secret
        - **key_file | Optional (IO | str)** - Path to API key file or file-like object
        - **base_url | (str)** - The websocket base url. Default set to "wss://advanced-trade-ws.coinbase.com"
        - **timeout | Optional (int)** - Set timeout in seconds for REST requests
        - **max_size | Optional (int)** - Max size in bytes for messages received. Default set to (10 * 1024 * 1024)
        - **on_message | Optional (Callable[[str], None])** - Function called when a message is received
        - **on_open | Optional ([Callable[[], None]])** - Function called when a connection is opened
        - **on_close | Optional ([Callable[[], None]])** - Function called when a connection is closed
        - **retry | Optional (bool)** - Enables automatic reconnections. Default set to True
        - **verbose | Optional (bool)** - Enables debug logging. Default set to False
    
    
        """
    
        from .channels import (
            candles,
            candles_async,
            candles_unsubscribe,
            candles_unsubscribe_async,
            futures_balance_summary,
            futures_balance_summary_async,
            futures_balance_summary_unsubscribe,
            futures_balance_summary_unsubscribe_async,
            heartbeats,
            heartbeats_async,
            heartbeats_unsubscribe,
            heartbeats_unsubscribe_async,
            level2,
            level2_async,
            level2_unsubscribe,
            level2_unsubscribe_async,
            market_trades,
            market_trades_async,
            market_trades_unsubscribe,
            market_trades_unsubscribe_async,
            status,
            status_async,
            status_unsubscribe,
            status_unsubscribe_async,
            ticker,
            ticker_async,
            ticker_batch,
            ticker_batch_async,
            ticker_batch_unsubscribe,
            ticker_batch_unsubscribe_async,
            ticker_unsubscribe,
            ticker_unsubscribe_async,
            user,
            user_async,
            user_unsubscribe,
            user_unsubscribe_async,
        )
    
    
    
    
    
    
    
    
    [[docs]](../../coinbase.websocket.user.html#coinbase.websocket.WSUserClient)
    class WSUserClient(WSBase):
        """
        **WSUserClient**
        _____________________________
    
        Initialize using WSUserClient
    
        __________
    
        **Parameters**:
    
        - **api_key | Optional (str)** - The API key
        - **api_secret | Optional (str)** - The API key secret
        - **key_file | Optional (IO | str)** - Path to API key file or file-like object
        - **base_url | (str)** - The websocket base url. Default set to "wss://advanced-trade-ws.coinbase.com"
        - **timeout | Optional (int)** - Set timeout in seconds for REST requests
        - **max_size | Optional (int)** - Max size in bytes for messages received. Default set to (10 * 1024 * 1024)
        - **on_message | Optional (Callable[[str], None])** - Function called when a message is received
        - **on_open | Optional ([Callable[[], None]])** - Function called when a connection is opened
        - **on_close | Optional ([Callable[[], None]])** - Function called when a connection is closed
        - **retry | Optional (bool)** - Enables automatic reconnections. Default set to True
        - **verbose | Optional (bool)** - Enables debug logging. Default set to False
    
    
        """
    
        from .channels import (
            futures_balance_summary,
            futures_balance_summary_async,
            futures_balance_summary_unsubscribe,
            futures_balance_summary_unsubscribe_async,
            heartbeats,
            heartbeats_async,
            heartbeats_unsubscribe,
            heartbeats_unsubscribe_async,
            user,
            user_async,
            user_unsubscribe,
            user_unsubscribe_async,
        )
    
        def __init__(
            self,
            api_key: Optional[str] = os.getenv(API_ENV_KEY),
            api_secret: Optional[str] = os.getenv(API_SECRET_ENV_KEY),
            key_file: Optional[Union[IO, str]] = None,
            base_url=WS_USER_BASE_URL,
            timeout: Optional[int] = None,
            max_size: Optional[int] = 10 * 1024 * 1024,
            on_message: Optional[Callable[[str], None]] = None,
            on_open: Optional[Callable[[], None]] = None,
            on_close: Optional[Callable[[], None]] = None,
            retry: Optional[bool] = True,
            verbose: Optional[bool] = False,
        ):
            super().__init__(
                api_key=api_key,
                api_secret=api_secret,
                key_file=key_file,
                base_url=base_url,
                timeout=timeout,
                max_size=max_size,
                on_message=on_message,
                on_open=on_open,
                on_close=on_close,
                retry=retry,
                verbose=verbose,
            )
    
    
    
    



================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/index.html
================================================================================

# Coinbase Advanced API Python SDK

* * *

# Getting Started

[](https://pypi.org/project/coinbase-advanced-py/) [](https://opensource.org/licenses/Apache-2.0)

Welcome to the official Coinbase Advanced API Python SDK. This python project was created to allow coders to easily plug into the [Coinbase Advanced API](https://docs.cdp.coinbase.com/advanced-trade/docs/welcome/)

  * Docs: <https://docs.cdp.coinbase.com/advanced-trade/docs/welcome/>

  * Python SDK: <https://github.com/coinbase/coinbase-advanced-py>




For detailed exercises on how to get started using the SDK look at our SDK Overview: <https://docs.cdp.coinbase.com/advanced-trade/docs/sdk-overview>

* * *

Contents:

  * [REST API Client](coinbase.rest.html)
    * [RESTClient Constructor](coinbase.rest.html#restclient-constructor)
      * [`RESTClient`](coinbase.rest.html#coinbase.rest.RESTClient)
    * [REST Utils](coinbase.rest.html#rest-utils)
      * [`get()`](coinbase.rest.html#coinbase.rest.RESTClient.get)
      * [`post()`](coinbase.rest.html#coinbase.rest.RESTClient.post)
      * [`put()`](coinbase.rest.html#coinbase.rest.RESTClient.put)
      * [`delete()`](coinbase.rest.html#coinbase.rest.RESTClient.delete)
    * [Accounts](coinbase.rest.html#accounts)
      * [`get_accounts()`](coinbase.rest.html#coinbase.rest.RESTClient.get_accounts)
      * [`get_account()`](coinbase.rest.html#coinbase.rest.RESTClient.get_account)
    * [Products](coinbase.rest.html#products)
      * [`get_products()`](coinbase.rest.html#coinbase.rest.RESTClient.get_products)
      * [`get_product()`](coinbase.rest.html#coinbase.rest.RESTClient.get_product)
      * [`get_product_book()`](coinbase.rest.html#coinbase.rest.RESTClient.get_product_book)
      * [`get_best_bid_ask()`](coinbase.rest.html#coinbase.rest.RESTClient.get_best_bid_ask)
    * [Market Data](coinbase.rest.html#market-data)
      * [`get_candles()`](coinbase.rest.html#coinbase.rest.RESTClient.get_candles)
      * [`get_market_trades()`](coinbase.rest.html#coinbase.rest.RESTClient.get_market_trades)
    * [Orders](coinbase.rest.html#orders)
      * [`create_order()`](coinbase.rest.html#coinbase.rest.RESTClient.create_order)
      * [`market_order()`](coinbase.rest.html#coinbase.rest.RESTClient.market_order)
      * [`market_order_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.market_order_buy)
      * [`market_order_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.market_order_sell)
      * [`limit_order_ioc()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_ioc)
      * [`limit_order_ioc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_ioc_buy)
      * [`limit_order_ioc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_ioc_sell)
      * [`limit_order_gtc()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_gtc)
      * [`limit_order_gtc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_gtc_buy)
      * [`limit_order_gtc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_gtc_sell)
      * [`limit_order_gtd()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_gtd)
      * [`limit_order_gtd_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_gtd_buy)
      * [`limit_order_gtd_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_gtd_sell)
      * [`limit_order_fok()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_fok)
      * [`limit_order_fok_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_fok_buy)
      * [`limit_order_fok_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.limit_order_fok_sell)
      * [`stop_limit_order_gtc()`](coinbase.rest.html#coinbase.rest.RESTClient.stop_limit_order_gtc)
      * [`stop_limit_order_gtc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.stop_limit_order_gtc_buy)
      * [`stop_limit_order_gtc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.stop_limit_order_gtc_sell)
      * [`stop_limit_order_gtd()`](coinbase.rest.html#coinbase.rest.RESTClient.stop_limit_order_gtd)
      * [`stop_limit_order_gtd_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.stop_limit_order_gtd_buy)
      * [`stop_limit_order_gtd_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.stop_limit_order_gtd_sell)
      * [`trigger_bracket_order_gtc()`](coinbase.rest.html#coinbase.rest.RESTClient.trigger_bracket_order_gtc)
      * [`trigger_bracket_order_gtc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.trigger_bracket_order_gtc_buy)
      * [`trigger_bracket_order_gtc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.trigger_bracket_order_gtc_sell)
      * [`trigger_bracket_order_gtd()`](coinbase.rest.html#coinbase.rest.RESTClient.trigger_bracket_order_gtd)
      * [`trigger_bracket_order_gtd_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.trigger_bracket_order_gtd_buy)
      * [`trigger_bracket_order_gtd_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.trigger_bracket_order_gtd_sell)
      * [`get_order()`](coinbase.rest.html#coinbase.rest.RESTClient.get_order)
      * [`list_orders()`](coinbase.rest.html#coinbase.rest.RESTClient.list_orders)
      * [`get_fills()`](coinbase.rest.html#coinbase.rest.RESTClient.get_fills)
      * [`edit_order()`](coinbase.rest.html#coinbase.rest.RESTClient.edit_order)
      * [`preview_edit_order()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_edit_order)
      * [`cancel_orders()`](coinbase.rest.html#coinbase.rest.RESTClient.cancel_orders)
      * [`preview_order()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_order)
      * [`preview_market_order()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_market_order)
      * [`preview_market_order_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_market_order_buy)
      * [`preview_market_order_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_market_order_sell)
      * [`preview_limit_order_ioc()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_ioc)
      * [`preview_limit_order_ioc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_ioc_buy)
      * [`preview_limit_order_ioc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_ioc_sell)
      * [`preview_limit_order_gtc()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_gtc)
      * [`preview_limit_order_gtc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_gtc_buy)
      * [`preview_limit_order_gtc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_gtc_sell)
      * [`preview_limit_order_gtd()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_gtd)
      * [`preview_limit_order_gtd_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_gtd_buy)
      * [`preview_limit_order_gtd_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_gtd_sell)
      * [`preview_limit_order_fok()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_fok)
      * [`preview_limit_order_fok_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_fok_buy)
      * [`preview_limit_order_fok_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_limit_order_fok_sell)
      * [`preview_stop_limit_order_gtc()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_stop_limit_order_gtc)
      * [`preview_stop_limit_order_gtc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_stop_limit_order_gtc_buy)
      * [`preview_stop_limit_order_gtc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_stop_limit_order_gtc_sell)
      * [`preview_stop_limit_order_gtd()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_stop_limit_order_gtd)
      * [`preview_stop_limit_order_gtd_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_stop_limit_order_gtd_buy)
      * [`preview_stop_limit_order_gtd_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_stop_limit_order_gtd_sell)
      * [`preview_trigger_bracket_order_gtc()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_trigger_bracket_order_gtc)
      * [`preview_trigger_bracket_order_gtc_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_trigger_bracket_order_gtc_buy)
      * [`preview_trigger_bracket_order_gtc_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_trigger_bracket_order_gtc_sell)
      * [`preview_trigger_bracket_order_gtd()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_trigger_bracket_order_gtd)
      * [`preview_trigger_bracket_order_gtd_buy()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_trigger_bracket_order_gtd_buy)
      * [`preview_trigger_bracket_order_gtd_sell()`](coinbase.rest.html#coinbase.rest.RESTClient.preview_trigger_bracket_order_gtd_sell)
      * [`close_position()`](coinbase.rest.html#coinbase.rest.RESTClient.close_position)
    * [Portfolios](coinbase.rest.html#portfolios)
      * [`get_portfolios()`](coinbase.rest.html#coinbase.rest.RESTClient.get_portfolios)
      * [`create_portfolio()`](coinbase.rest.html#coinbase.rest.RESTClient.create_portfolio)
      * [`get_portfolio_breakdown()`](coinbase.rest.html#coinbase.rest.RESTClient.get_portfolio_breakdown)
      * [`move_portfolio_funds()`](coinbase.rest.html#coinbase.rest.RESTClient.move_portfolio_funds)
      * [`edit_portfolio()`](coinbase.rest.html#coinbase.rest.RESTClient.edit_portfolio)
      * [`delete_portfolio()`](coinbase.rest.html#coinbase.rest.RESTClient.delete_portfolio)
    * [Futures](coinbase.rest.html#futures)
      * [`get_futures_balance_summary()`](coinbase.rest.html#coinbase.rest.RESTClient.get_futures_balance_summary)
      * [`list_futures_positions()`](coinbase.rest.html#coinbase.rest.RESTClient.list_futures_positions)
      * [`get_futures_position()`](coinbase.rest.html#coinbase.rest.RESTClient.get_futures_position)
      * [`schedule_futures_sweep()`](coinbase.rest.html#coinbase.rest.RESTClient.schedule_futures_sweep)
      * [`list_futures_sweeps()`](coinbase.rest.html#coinbase.rest.RESTClient.list_futures_sweeps)
      * [`cancel_pending_futures_sweep()`](coinbase.rest.html#coinbase.rest.RESTClient.cancel_pending_futures_sweep)
      * [`get_intraday_margin_setting()`](coinbase.rest.html#coinbase.rest.RESTClient.get_intraday_margin_setting)
      * [`get_current_margin_window()`](coinbase.rest.html#coinbase.rest.RESTClient.get_current_margin_window)
      * [`set_intraday_margin_setting()`](coinbase.rest.html#coinbase.rest.RESTClient.set_intraday_margin_setting)
    * [Perpetuals](coinbase.rest.html#perpetuals)
      * [`allocate_portfolio()`](coinbase.rest.html#coinbase.rest.RESTClient.allocate_portfolio)
      * [`get_perps_portfolio_summary()`](coinbase.rest.html#coinbase.rest.RESTClient.get_perps_portfolio_summary)
      * [`list_perps_positions()`](coinbase.rest.html#coinbase.rest.RESTClient.list_perps_positions)
      * [`get_perps_position()`](coinbase.rest.html#coinbase.rest.RESTClient.get_perps_position)
      * [`get_perps_portfolio_balances()`](coinbase.rest.html#coinbase.rest.RESTClient.get_perps_portfolio_balances)
      * [`opt_in_or_out_multi_asset_collateral()`](coinbase.rest.html#coinbase.rest.RESTClient.opt_in_or_out_multi_asset_collateral)
    * [Fees](coinbase.rest.html#fees)
      * [`get_transaction_summary()`](coinbase.rest.html#coinbase.rest.RESTClient.get_transaction_summary)
    * [Converts](coinbase.rest.html#converts)
      * [`create_convert_quote()`](coinbase.rest.html#coinbase.rest.RESTClient.create_convert_quote)
      * [`get_convert_trade()`](coinbase.rest.html#coinbase.rest.RESTClient.get_convert_trade)
      * [`commit_convert_trade()`](coinbase.rest.html#coinbase.rest.RESTClient.commit_convert_trade)
    * [Public](coinbase.rest.html#public)
      * [`get_unix_time()`](coinbase.rest.html#coinbase.rest.RESTClient.get_unix_time)
      * [`get_public_product_book()`](coinbase.rest.html#coinbase.rest.RESTClient.get_public_product_book)
      * [`get_public_products()`](coinbase.rest.html#coinbase.rest.RESTClient.get_public_products)
      * [`get_public_product()`](coinbase.rest.html#coinbase.rest.RESTClient.get_public_product)
      * [`get_public_candles()`](coinbase.rest.html#coinbase.rest.RESTClient.get_public_candles)
      * [`get_public_market_trades()`](coinbase.rest.html#coinbase.rest.RESTClient.get_public_market_trades)
    * [Payments](coinbase.rest.html#payments)
      * [`list_payment_methods()`](coinbase.rest.html#coinbase.rest.RESTClient.list_payment_methods)
      * [`get_payment_method()`](coinbase.rest.html#coinbase.rest.RESTClient.get_payment_method)
    * [Data API](coinbase.rest.html#data-api)
      * [`get_api_key_permissions()`](coinbase.rest.html#coinbase.rest.RESTClient.get_api_key_permissions)
  * [Websocket API Client](coinbase.websocket.html)
    * [WSClient Constructor](coinbase.websocket.html#wsclient-constructor)
      * [`WSClient`](coinbase.websocket.html#coinbase.websocket.WSClient)
    * [WebSocket Utils](coinbase.websocket.html#websocket-utils)
      * [`open()`](coinbase.websocket.html#coinbase.websocket.WSClient.open)
      * [`open_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.open_async)
      * [`close()`](coinbase.websocket.html#coinbase.websocket.WSClient.close)
      * [`close_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.close_async)
      * [`subscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.subscribe)
      * [`subscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.subscribe_async)
      * [`unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.unsubscribe)
      * [`unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.unsubscribe_async)
      * [`unsubscribe_all()`](coinbase.websocket.html#coinbase.websocket.WSClient.unsubscribe_all)
      * [`unsubscribe_all_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.unsubscribe_all_async)
      * [`sleep_with_exception_check()`](coinbase.websocket.html#coinbase.websocket.WSClient.sleep_with_exception_check)
      * [`sleep_with_exception_check_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.sleep_with_exception_check_async)
      * [`run_forever_with_exception_check()`](coinbase.websocket.html#coinbase.websocket.WSClient.run_forever_with_exception_check)
      * [`run_forever_with_exception_check_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.run_forever_with_exception_check_async)
      * [`raise_background_exception()`](coinbase.websocket.html#coinbase.websocket.WSClient.raise_background_exception)
    * [Channels](coinbase.websocket.html#channels)
      * [`heartbeats()`](coinbase.websocket.html#coinbase.websocket.WSClient.heartbeats)
      * [`heartbeats_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.heartbeats_async)
      * [`heartbeats_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.heartbeats_unsubscribe)
      * [`heartbeats_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.heartbeats_unsubscribe_async)
      * [`candles()`](coinbase.websocket.html#coinbase.websocket.WSClient.candles)
      * [`candles_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.candles_async)
      * [`candles_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.candles_unsubscribe)
      * [`candles_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.candles_unsubscribe_async)
      * [`market_trades()`](coinbase.websocket.html#coinbase.websocket.WSClient.market_trades)
      * [`market_trades_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.market_trades_async)
      * [`market_trades_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.market_trades_unsubscribe)
      * [`market_trades_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.market_trades_unsubscribe_async)
      * [`status()`](coinbase.websocket.html#coinbase.websocket.WSClient.status)
      * [`status_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.status_async)
      * [`status_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.status_unsubscribe)
      * [`status_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.status_unsubscribe_async)
      * [`ticker()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker)
      * [`ticker_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_async)
      * [`ticker_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_unsubscribe)
      * [`ticker_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_unsubscribe_async)
      * [`ticker_batch()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_batch)
      * [`ticker_batch_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_batch_async)
      * [`ticker_batch_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_batch_unsubscribe)
      * [`ticker_batch_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.ticker_batch_unsubscribe_async)
      * [`level2()`](coinbase.websocket.html#coinbase.websocket.WSClient.level2)
      * [`level2_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.level2_async)
      * [`level2_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.level2_unsubscribe)
      * [`level2_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.level2_unsubscribe_async)
      * [`user()`](coinbase.websocket.html#coinbase.websocket.WSClient.user)
      * [`user_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.user_async)
      * [`user_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.user_unsubscribe)
      * [`user_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.user_unsubscribe_async)
      * [`futures_balance_summary()`](coinbase.websocket.html#coinbase.websocket.WSClient.futures_balance_summary)
      * [`futures_balance_summary_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.futures_balance_summary_async)
      * [`futures_balance_summary_unsubscribe()`](coinbase.websocket.html#coinbase.websocket.WSClient.futures_balance_summary_unsubscribe)
      * [`futures_balance_summary_unsubscribe_async()`](coinbase.websocket.html#coinbase.websocket.WSClient.futures_balance_summary_unsubscribe_async)
    * [Exceptions](coinbase.websocket.html#exceptions)
      * [`WSClientException()`](coinbase.websocket.html#coinbase.websocket.WSClientException)
      * [`WSClientConnectionClosedException()`](coinbase.websocket.html#coinbase.websocket.WSClientConnectionClosedException)
  * [Websocket User API Client](coinbase.websocket.user.html)
    * [WSUserClient Constructor](coinbase.websocket.user.html#wsuserclient-constructor)
      * [`WSUserClient`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient)
    * [WebSocket Utils](coinbase.websocket.user.html#websocket-utils)
      * [`open()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.open)
      * [`open_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.open_async)
      * [`close()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.close)
      * [`close_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.close_async)
      * [`subscribe()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.subscribe)
      * [`subscribe_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.subscribe_async)
      * [`unsubscribe()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.unsubscribe)
      * [`unsubscribe_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.unsubscribe_async)
      * [`unsubscribe_all()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.unsubscribe_all)
      * [`unsubscribe_all_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.unsubscribe_all_async)
      * [`sleep_with_exception_check()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.sleep_with_exception_check)
      * [`sleep_with_exception_check_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.sleep_with_exception_check_async)
      * [`run_forever_with_exception_check()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.run_forever_with_exception_check)
      * [`run_forever_with_exception_check_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.run_forever_with_exception_check_async)
      * [`raise_background_exception()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.raise_background_exception)
    * [Channels](coinbase.websocket.user.html#channels)
      * [`heartbeats()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.heartbeats)
      * [`heartbeats_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.heartbeats_async)
      * [`heartbeats_unsubscribe()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.heartbeats_unsubscribe)
      * [`heartbeats_unsubscribe_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.heartbeats_unsubscribe_async)
      * [`user()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.user)
      * [`user_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.user_async)
      * [`user_unsubscribe()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.user_unsubscribe)
      * [`user_unsubscribe_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.user_unsubscribe_async)
      * [`futures_balance_summary()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.futures_balance_summary)
      * [`futures_balance_summary_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.futures_balance_summary_async)
      * [`futures_balance_summary_unsubscribe()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.futures_balance_summary_unsubscribe)
      * [`futures_balance_summary_unsubscribe_async()`](coinbase.websocket.user.html#coinbase.websocket.WSUserClient.futures_balance_summary_unsubscribe_async)
    * [Exceptions](coinbase.websocket.user.html#exceptions)
  * [Authentication](jwt_generator.html)
    * [`build_rest_jwt()`](jwt_generator.html#coinbase.jwt_generator.build_rest_jwt)
    * [`build_ws_jwt()`](jwt_generator.html#coinbase.jwt_generator.build_ws_jwt)
    * [`format_jwt_uri()`](jwt_generator.html#coinbase.jwt_generator.format_jwt_uri)





================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/_sources/coinbase.websocket.rst.txt
================================================================================


    Websocket API Client
    =====================
    
    WSClient Constructor
    ---------------------------
    
    .. autoclass:: coinbase.websocket.WSClient
    
    WebSocket Utils
    ---------------------------
    
    .. autofunction:: coinbase.websocket.WSClient.open
    .. autofunction:: coinbase.websocket.WSClient.open_async
    .. autofunction:: coinbase.websocket.WSClient.close
    .. autofunction:: coinbase.websocket.WSClient.close_async
    .. autofunction:: coinbase.websocket.WSClient.subscribe
    .. autofunction:: coinbase.websocket.WSClient.subscribe_async
    .. autofunction:: coinbase.websocket.WSClient.unsubscribe
    .. autofunction:: coinbase.websocket.WSClient.unsubscribe_async
    .. autofunction:: coinbase.websocket.WSClient.unsubscribe_all
    .. autofunction:: coinbase.websocket.WSClient.unsubscribe_all_async
    .. autofunction:: coinbase.websocket.WSClient.sleep_with_exception_check
    .. autofunction:: coinbase.websocket.WSClient.sleep_with_exception_check_async
    .. autofunction:: coinbase.websocket.WSClient.run_forever_with_exception_check
    .. autofunction:: coinbase.websocket.WSClient.run_forever_with_exception_check_async
    .. autofunction:: coinbase.websocket.WSClient.raise_background_exception
    
    Channels
    -----------------------------
    
    .. autofunction:: coinbase.websocket.WSClient.heartbeats
    .. autofunction:: coinbase.websocket.WSClient.heartbeats_async
    .. autofunction:: coinbase.websocket.WSClient.heartbeats_unsubscribe
    .. autofunction:: coinbase.websocket.WSClient.heartbeats_unsubscribe_async
    .. autofunction:: coinbase.websocket.WSClient.candles
    .. autofunction:: coinbase.websocket.WSClient.candles_async
    .. autofunction:: coinbase.websocket.WSClient.candles_unsubscribe
    .. autofunction:: coinbase.websocket.WSClient.candles_unsubscribe_async
    .. autofunction:: coinbase.websocket.WSClient.market_trades
    .. autofunction:: coinbase.websocket.WSClient.market_trades_async
    .. autofunction:: coinbase.websocket.WSClient.market_trades_unsubscribe
    .. autofunction:: coinbase.websocket.WSClient.market_trades_unsubscribe_async
    .. autofunction:: coinbase.websocket.WSClient.status
    .. autofunction:: coinbase.websocket.WSClient.status_async
    .. autofunction:: coinbase.websocket.WSClient.status_unsubscribe
    .. autofunction:: coinbase.websocket.WSClient.status_unsubscribe_async
    .. autofunction:: coinbase.websocket.WSClient.ticker
    .. autofunction:: coinbase.websocket.WSClient.ticker_async
    .. autofunction:: coinbase.websocket.WSClient.ticker_unsubscribe
    .. autofunction:: coinbase.websocket.WSClient.ticker_unsubscribe_async
    .. autofunction:: coinbase.websocket.WSClient.ticker_batch
    .. autofunction:: coinbase.websocket.WSClient.ticker_batch_async
    .. autofunction:: coinbase.websocket.WSClient.ticker_batch_unsubscribe
    .. autofunction:: coinbase.websocket.WSClient.ticker_batch_unsubscribe_async
    .. autofunction:: coinbase.websocket.WSClient.level2
    .. autofunction:: coinbase.websocket.WSClient.level2_async
    .. autofunction:: coinbase.websocket.WSClient.level2_unsubscribe
    .. autofunction:: coinbase.websocket.WSClient.level2_unsubscribe_async
    .. autofunction:: coinbase.websocket.WSClient.user
    .. autofunction:: coinbase.websocket.WSClient.user_async
    .. autofunction:: coinbase.websocket.WSClient.user_unsubscribe
    .. autofunction:: coinbase.websocket.WSClient.user_unsubscribe_async
    .. autofunction:: coinbase.websocket.WSClient.futures_balance_summary
    .. autofunction:: coinbase.websocket.WSClient.futures_balance_summary_async
    .. autofunction:: coinbase.websocket.WSClient.futures_balance_summary_unsubscribe
    .. autofunction:: coinbase.websocket.WSClient.futures_balance_summary_unsubscribe_async
    
    Exceptions
    ---------------------------
    
    .. autofunction:: coinbase.websocket.WSClientException
    .. autofunction:: coinbase.websocket.WSClientConnectionClosedException
    



================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/_modules/coinbase/jwt_generator.html
================================================================================

# Source code for coinbase.jwt_generator
    
    
    import secrets
    import time
    
    import jwt
    from cryptography.hazmat.primitives import serialization
    
    from coinbase.constants import BASE_URL
    
    
    def build_jwt(key_var, secret_var, uri=None) -> str:
        """
        :meta private:
        """
        try:
            private_key_bytes = secret_var.encode("utf-8")
            private_key = serialization.load_pem_private_key(
                private_key_bytes, password=None
            )
        except ValueError as e:
            # This handles errors like incorrect key format
            raise Exception(
                f"{e}\n"
                "Are you sure you generated your key at https://cloud.coinbase.com/access/api ?"
            )
    
        jwt_data = {
            "sub": key_var,
            "iss": "cdp",
            "nbf": int(time.time()),
            "exp": int(time.time()) + 120,
        }
    
        if uri:
            jwt_data["uri"] = uri
    
        jwt_token = jwt.encode(
            jwt_data,
            private_key,
            algorithm="ES256",
            headers={"kid": key_var, "nonce": secrets.token_hex()},
        )
    
        return jwt_token
    
    
    
    
    
    [[docs]](../../jwt_generator.html#coinbase.jwt_generator.build_rest_jwt)
    def build_rest_jwt(uri, key_var, secret_var) -> str:
        """
        **Build REST JWT**
        __________
    
        **Description:**
    
        Builds and returns a JWT token for connecting to the REST API.
    
        __________
    
        Parameters:
    
        - **uri (str)** - Formatted URI for the endpoint (e.g. "GET api.coinbase.com/api/v3/brokerage/accounts") Can be generated using ``format_jwt_uri``
        - **key_var (str)** - The API key
        - **secret_var (str)** - The API key secret
        """
        return build_jwt(key_var, secret_var, uri=uri)
    
    
    
    
    
    
    
    
    [[docs]](../../jwt_generator.html#coinbase.jwt_generator.build_ws_jwt)
    def build_ws_jwt(key_var, secret_var) -> str:
        """
        **Build WebSocket JWT**
        __________
    
        **Description:**
    
        Builds and returns a JWT token for connecting to the WebSocket API.
    
        __________
    
        Parameters:
    
        - **key_var (str)** - The API key
        - **secret_var (str)** - The API key secret
        """
        return build_jwt(key_var, secret_var)
    
    
    
    
    
    
    
    
    [[docs]](../../jwt_generator.html#coinbase.jwt_generator.format_jwt_uri)
    def format_jwt_uri(method, path) -> str:
        """
        **Format JWT URI**
        __________
    
        **Description:**
    
        Formats method and path into valid URI for JWT generation.
    
        __________
    
        Parameters:
    
        - **method (str)** - The REST request method. E.g. GET, POST, PUT, DELETE
        - **path (str)** - The path of the endpoint. E.g. "/api/v3/brokerage/accounts"
    
        """
        return f"{method} {BASE_URL}{path}"
    
    
    
    



================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/_sources/coinbase.rest.rst.txt
================================================================================


    REST API Client
    =====================
    
    
    RESTClient Constructor
    -------------------------------
    
    .. autoclass:: coinbase.rest.RESTClient
    
    REST Utils
    -------------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.get
    .. autofunction:: coinbase.rest.RESTClient.post
    .. autofunction:: coinbase.rest.RESTClient.put
    .. autofunction:: coinbase.rest.RESTClient.delete
    
    Accounts
    -----------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.get_accounts
    .. autofunction:: coinbase.rest.RESTClient.get_account
    
    Products
    -----------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.get_products
    .. autofunction:: coinbase.rest.RESTClient.get_product
    .. autofunction:: coinbase.rest.RESTClient.get_product_book
    .. autofunction:: coinbase.rest.RESTClient.get_best_bid_ask
    
    Market Data
    ---------------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.get_candles
    .. autofunction:: coinbase.rest.RESTClient.get_market_trades
    
    Orders
    ---------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.create_order
    .. autofunction:: coinbase.rest.RESTClient.market_order
    .. autofunction:: coinbase.rest.RESTClient.market_order_buy
    .. autofunction:: coinbase.rest.RESTClient.market_order_sell
    .. autofunction:: coinbase.rest.RESTClient.limit_order_ioc
    .. autofunction:: coinbase.rest.RESTClient.limit_order_ioc_buy
    .. autofunction:: coinbase.rest.RESTClient.limit_order_ioc_sell
    .. autofunction:: coinbase.rest.RESTClient.limit_order_gtc
    .. autofunction:: coinbase.rest.RESTClient.limit_order_gtc_buy
    .. autofunction:: coinbase.rest.RESTClient.limit_order_gtc_sell
    .. autofunction:: coinbase.rest.RESTClient.limit_order_gtd
    .. autofunction:: coinbase.rest.RESTClient.limit_order_gtd_buy
    .. autofunction:: coinbase.rest.RESTClient.limit_order_gtd_sell
    .. autofunction:: coinbase.rest.RESTClient.limit_order_fok
    .. autofunction:: coinbase.rest.RESTClient.limit_order_fok_buy
    .. autofunction:: coinbase.rest.RESTClient.limit_order_fok_sell
    .. autofunction:: coinbase.rest.RESTClient.stop_limit_order_gtc
    .. autofunction:: coinbase.rest.RESTClient.stop_limit_order_gtc_buy
    .. autofunction:: coinbase.rest.RESTClient.stop_limit_order_gtc_sell
    .. autofunction:: coinbase.rest.RESTClient.stop_limit_order_gtd
    .. autofunction:: coinbase.rest.RESTClient.stop_limit_order_gtd_buy
    .. autofunction:: coinbase.rest.RESTClient.stop_limit_order_gtd_sell
    .. autofunction:: coinbase.rest.RESTClient.trigger_bracket_order_gtc
    .. autofunction:: coinbase.rest.RESTClient.trigger_bracket_order_gtc_buy
    .. autofunction:: coinbase.rest.RESTClient.trigger_bracket_order_gtc_sell
    .. autofunction:: coinbase.rest.RESTClient.trigger_bracket_order_gtd
    .. autofunction:: coinbase.rest.RESTClient.trigger_bracket_order_gtd_buy
    .. autofunction:: coinbase.rest.RESTClient.trigger_bracket_order_gtd_sell
    .. autofunction:: coinbase.rest.RESTClient.get_order
    .. autofunction:: coinbase.rest.RESTClient.list_orders
    .. autofunction:: coinbase.rest.RESTClient.get_fills
    .. autofunction:: coinbase.rest.RESTClient.edit_order
    .. autofunction:: coinbase.rest.RESTClient.preview_edit_order
    .. autofunction:: coinbase.rest.RESTClient.cancel_orders
    .. autofunction:: coinbase.rest.RESTClient.preview_order
    .. autofunction:: coinbase.rest.RESTClient.preview_market_order
    .. autofunction:: coinbase.rest.RESTClient.preview_market_order_buy
    .. autofunction:: coinbase.rest.RESTClient.preview_market_order_sell
    .. autofunction:: coinbase.rest.RESTClient.preview_limit_order_ioc
    .. autofunction:: coinbase.rest.RESTClient.preview_limit_order_ioc_buy
    .. autofunction:: coinbase.rest.RESTClient.preview_limit_order_ioc_sell
    .. autofunction:: coinbase.rest.RESTClient.preview_limit_order_gtc
    .. autofunction:: coinbase.rest.RESTClient.preview_limit_order_gtc_buy
    .. autofunction:: coinbase.rest.RESTClient.preview_limit_order_gtc_sell
    .. autofunction:: coinbase.rest.RESTClient.preview_limit_order_gtd
    .. autofunction:: coinbase.rest.RESTClient.preview_limit_order_gtd_buy
    .. autofunction:: coinbase.rest.RESTClient.preview_limit_order_gtd_sell
    .. autofunction:: coinbase.rest.RESTClient.preview_limit_order_fok
    .. autofunction:: coinbase.rest.RESTClient.preview_limit_order_fok_buy
    .. autofunction:: coinbase.rest.RESTClient.preview_limit_order_fok_sell
    .. autofunction:: coinbase.rest.RESTClient.preview_stop_limit_order_gtc
    .. autofunction:: coinbase.rest.RESTClient.preview_stop_limit_order_gtc_buy
    .. autofunction:: coinbase.rest.RESTClient.preview_stop_limit_order_gtc_sell
    .. autofunction:: coinbase.rest.RESTClient.preview_stop_limit_order_gtd
    .. autofunction:: coinbase.rest.RESTClient.preview_stop_limit_order_gtd_buy
    .. autofunction:: coinbase.rest.RESTClient.preview_stop_limit_order_gtd_sell
    .. autofunction:: coinbase.rest.RESTClient.preview_trigger_bracket_order_gtc
    .. autofunction:: coinbase.rest.RESTClient.preview_trigger_bracket_order_gtc_buy
    .. autofunction:: coinbase.rest.RESTClient.preview_trigger_bracket_order_gtc_sell
    .. autofunction:: coinbase.rest.RESTClient.preview_trigger_bracket_order_gtd
    .. autofunction:: coinbase.rest.RESTClient.preview_trigger_bracket_order_gtd_buy
    .. autofunction:: coinbase.rest.RESTClient.preview_trigger_bracket_order_gtd_sell
    .. autofunction:: coinbase.rest.RESTClient.close_position
    
    Portfolios
    -------------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.get_portfolios
    .. autofunction:: coinbase.rest.RESTClient.create_portfolio
    .. autofunction:: coinbase.rest.RESTClient.get_portfolio_breakdown
    .. autofunction:: coinbase.rest.RESTClient.move_portfolio_funds
    .. autofunction:: coinbase.rest.RESTClient.edit_portfolio
    .. autofunction:: coinbase.rest.RESTClient.delete_portfolio
    
    Futures
    ----------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.get_futures_balance_summary
    .. autofunction:: coinbase.rest.RESTClient.list_futures_positions
    .. autofunction:: coinbase.rest.RESTClient.get_futures_position
    .. autofunction:: coinbase.rest.RESTClient.schedule_futures_sweep
    .. autofunction:: coinbase.rest.RESTClient.list_futures_sweeps
    .. autofunction:: coinbase.rest.RESTClient.cancel_pending_futures_sweep
    .. autofunction:: coinbase.rest.RESTClient.get_intraday_margin_setting
    .. autofunction:: coinbase.rest.RESTClient.get_current_margin_window
    .. autofunction:: coinbase.rest.RESTClient.set_intraday_margin_setting
    
    Perpetuals
    ---------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.allocate_portfolio
    .. autofunction:: coinbase.rest.RESTClient.get_perps_portfolio_summary
    .. autofunction:: coinbase.rest.RESTClient.list_perps_positions
    .. autofunction:: coinbase.rest.RESTClient.get_perps_position
    .. autofunction:: coinbase.rest.RESTClient.get_perps_portfolio_balances
    .. autofunction:: coinbase.rest.RESTClient.opt_in_or_out_multi_asset_collateral
    
    Fees
    -------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.get_transaction_summary
    
    Converts
    ----------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.create_convert_quote
    .. autofunction:: coinbase.rest.RESTClient.get_convert_trade
    .. autofunction:: coinbase.rest.RESTClient.commit_convert_trade
    
    Public
    ---------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.get_unix_time
    .. autofunction:: coinbase.rest.RESTClient.get_public_product_book
    .. autofunction:: coinbase.rest.RESTClient.get_public_products
    .. autofunction:: coinbase.rest.RESTClient.get_public_product
    .. autofunction:: coinbase.rest.RESTClient.get_public_candles
    .. autofunction:: coinbase.rest.RESTClient.get_public_market_trades
    
    Payments
    -------------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.list_payment_methods
    .. autofunction:: coinbase.rest.RESTClient.get_payment_method
    
    Data API
    -------------------------------
    
    .. autofunction:: coinbase.rest.RESTClient.get_api_key_permissions



================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/_sources/coinbase.websocket.user.rst.txt
================================================================================


    Websocket User API Client
    =====================
    
    WSUserClient Constructor
    ---------------------------
    
    .. autoclass:: coinbase.websocket.WSUserClient
    
    WebSocket Utils
    ---------------------------
    
    .. autofunction:: coinbase.websocket.WSUserClient.open
    .. autofunction:: coinbase.websocket.WSUserClient.open_async
    .. autofunction:: coinbase.websocket.WSUserClient.close
    .. autofunction:: coinbase.websocket.WSUserClient.close_async
    .. autofunction:: coinbase.websocket.WSUserClient.subscribe
    .. autofunction:: coinbase.websocket.WSUserClient.subscribe_async
    .. autofunction:: coinbase.websocket.WSUserClient.unsubscribe
    .. autofunction:: coinbase.websocket.WSUserClient.unsubscribe_async
    .. autofunction:: coinbase.websocket.WSUserClient.unsubscribe_all
    .. autofunction:: coinbase.websocket.WSUserClient.unsubscribe_all_async
    .. autofunction:: coinbase.websocket.WSUserClient.sleep_with_exception_check
    .. autofunction:: coinbase.websocket.WSUserClient.sleep_with_exception_check_async
    .. autofunction:: coinbase.websocket.WSUserClient.run_forever_with_exception_check
    .. autofunction:: coinbase.websocket.WSUserClient.run_forever_with_exception_check_async
    .. autofunction:: coinbase.websocket.WSUserClient.raise_background_exception
    
    Channels
    -----------------------------
    
    .. autofunction:: coinbase.websocket.WSUserClient.heartbeats
    .. autofunction:: coinbase.websocket.WSUserClient.heartbeats_async
    .. autofunction:: coinbase.websocket.WSUserClient.heartbeats_unsubscribe
    .. autofunction:: coinbase.websocket.WSUserClient.heartbeats_unsubscribe_async
    .. autofunction:: coinbase.websocket.WSUserClient.user
    .. autofunction:: coinbase.websocket.WSUserClient.user_async
    .. autofunction:: coinbase.websocket.WSUserClient.user_unsubscribe
    .. autofunction:: coinbase.websocket.WSUserClient.user_unsubscribe_async
    .. autofunction:: coinbase.websocket.WSUserClient.futures_balance_summary
    .. autofunction:: coinbase.websocket.WSUserClient.futures_balance_summary_async
    .. autofunction:: coinbase.websocket.WSUserClient.futures_balance_summary_unsubscribe
    .. autofunction:: coinbase.websocket.WSUserClient.futures_balance_summary_unsubscribe_async
    
    Exceptions
    ---------------------------
    
    .. autofunction:: coinbase.websocket.WSUserClientException
    .. autofunction:: coinbase.websocket.WSUserClientConnectionClosedException
    



================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/_sources/jwt_generator.rst.txt
================================================================================


    Authentication
    -----------------------------
    
    .. automodule:: coinbase.jwt_generator
       :members:
       :undoc-members:
       :show-inheritance:



================================================================================
# 📍 Source: https://coinbase.github.io/coinbase-advanced-py/_modules/index.html
================================================================================

# All modules for which code is available

  * [coinbase.jwt_generator](coinbase/jwt_generator.html)
  * [coinbase.rest](coinbase/rest.html)
  * [coinbase.websocket](coinbase/websocket.html)
    * [coinbase.websocket.websocket_base](coinbase/websocket/websocket_base.html)


