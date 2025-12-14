you are helping build an atonomous trading system powered by large language models

the trading system should feed data from the coinbase api, the schwab api, streaming data from coinbase web sockets, shchwab web sockets, publicly available apis for news, any web sockets we can subscribe to, for free information that is relevant to an atonomous trading system.

prefer delegation to the large language model for deciding how to respond to user input

the trading system should feed this data as input to a large language model that can call predictive functions that use the market data for input

the schwab apis have functions for returning large amounts of historical data for stocks, options, documented in resources/schwab-api.md

the coinbase apis have functions for returning large amount of historical data crypto currency and perpetual crypto, documented in resources/coinbase-api.md

the schwab and coinbase apis have web sockets that can be subscribed to for getting realtime data

for stocks and options, the atonomous trading system should be aware of market hours when markets are open for trading

resources to assist with building the application are in the resources folder

resources/ISLR_First_Printing.md has information about how to build predictive models; prediction models are also known as regression models

when a major change is implemented offer to commit and push everything

always use test-driven development (TDD)

try and maintain 95% or more code coverage

tests that take longer tha one second to complete are considered long-running

don't put long-running tests in test/

put end-to-end tests in e2e_tests/

if there is some other form of long-running test

if there is a long-runing test that is not an end-to-end test, ask what folder to create to put it in

after major changes check for opportunities to refactor for things like removing dead code, factoring out duplicated logic, and clarifying logic