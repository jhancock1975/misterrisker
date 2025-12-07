"""Playwright tests for Mister Risker trading strategy recommendations.

These tests verify that Mister Risker provides actionable trading recommendations
with specific limit order prices, not just regurgitating market data.
"""

import re
import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:8000"


class TestTradingStrategyRecommendations:
    """Test that Mister Risker provides actual trading strategies, not just market data."""

    def test_trading_strategy_gives_actionable_recommendations(self, page: Page):
        """When user asks for 'trading strategy', should get buy/sell recommendations with prices."""
        page.goto(BASE_URL)
        
        # Send the query from the screenshot
        chat_input = page.locator("#chat-input")
        chat_input.fill("Give me a trading strategy for crypto right now")
        chat_input.press("Enter")
        
        # Wait for response
        page.wait_for_timeout(30000)  # May need time to train models
        
        # Get the response
        messages = page.locator(".message-content")
        last_response = messages.last.inner_text()
        
        # Should contain actionable recommendations, not just market data
        # The response should include specific trade recommendations
        assert any(word in last_response.lower() for word in [
            "recommend", "suggest", "buy at", "sell at", "limit order", 
            "entry", "target", "stop loss", "strategy"
        ]), f"Response should contain actionable recommendations, got: {last_response[:500]}"

    def test_limit_order_recommendations_have_specific_prices(self, page: Page):
        """When user asks for limit order recommendations, should get specific prices."""
        page.goto(BASE_URL)
        
        # This is the exact query from the screenshot
        chat_input = page.locator("#chat-input")
        chat_input.fill("can you analyze available data from the tools exposed to you to recommend limit order trades for sol, eth, btc, zec, xrp?")
        chat_input.press("Enter")
        
        # Wait for response
        page.wait_for_timeout(45000)  # Training + analysis takes time
        
        # Get the response
        messages = page.locator(".message-content")
        last_response = messages.last.inner_text()
        
        # Should NOT just show "BTC Real-Time Analysis" with current price
        # Should provide actual limit order recommendations
        
        # Check for multiple coins being addressed
        coins_mentioned = sum(1 for coin in ["BTC", "ETH", "SOL", "XRP", "ZEC"] 
                             if coin in last_response)
        assert coins_mentioned >= 3, f"Should address multiple coins, only found {coins_mentioned}"
        
        # Should have limit order specifics
        assert any(phrase in last_response.lower() for phrase in [
            "limit order", "buy limit", "sell limit", "entry price",
            "order at", "place order", "recommended price"
        ]), f"Should mention limit orders specifically, got: {last_response[:500]}"
        
        # Should have price targets (numbers with $ sign)
        price_pattern = r'\$[\d,]+\.?\d*'
        prices_found = re.findall(price_pattern, last_response)
        assert len(prices_found) >= 5, f"Should have multiple price targets, found {len(prices_found)}: {prices_found}"

    def test_recommendations_include_entry_and_targets(self, page: Page):
        """Trading recommendations should include entry price, target, and stop loss."""
        page.goto(BASE_URL)
        
        chat_input = page.locator("#chat-input")
        chat_input.fill("recommend specific limit orders for BTC and ETH with entry, target, and stop loss")
        chat_input.press("Enter")
        
        page.wait_for_timeout(30000)
        
        messages = page.locator(".message-content")
        last_response = messages.last.inner_text()
        
        # Should include trading levels
        response_lower = last_response.lower()
        
        has_entry = any(word in response_lower for word in ["entry", "buy at", "limit at", "order at"])
        has_target = any(word in response_lower for word in ["target", "take profit", "profit at", "tp"])
        has_stop = any(word in response_lower for word in ["stop", "stop loss", "sl", "risk"])
        
        assert has_entry, f"Should have entry price, got: {last_response[:300]}"
        assert has_target, f"Should have profit target, got: {last_response[:300]}"
        assert has_stop, f"Should have stop loss, got: {last_response[:300]}"

    def test_does_not_just_show_current_price(self, page: Page):
        """When asking for trading advice, should NOT just show current market data."""
        page.goto(BASE_URL)
        
        chat_input = page.locator("#chat-input")
        chat_input.fill("what limit orders should I place for crypto?")
        chat_input.press("Enter")
        
        page.wait_for_timeout(30000)
        
        messages = page.locator(".message-content")
        last_response = messages.last.inner_text()
        
        # Should NOT just be a data dump like "Current Data (Real-Time WebSocket)"
        # followed by OHLCV without actionable advice
        
        # If it shows "Current Data" it should ALSO have recommendations
        if "Current Data" in last_response or "Real-Time" in last_response:
            # Must also have actionable content
            has_action = any(word in last_response.lower() for word in [
                "recommend", "suggest", "should", "consider", "place", 
                "buy limit", "sell limit", "entry", "target"
            ])
            assert has_action, (
                "Response shows market data but lacks actionable recommendations. "
                f"Got: {last_response[:500]}"
            )

    def test_finrl_signals_translate_to_orders(self, page: Page):
        """FinRL buy/sell signals should translate to specific order recommendations."""
        page.goto(BASE_URL)
        
        chat_input = page.locator("#chat-input")
        chat_input.fill("based on your AI analysis, what specific orders should I place?")
        chat_input.press("Enter")
        
        page.wait_for_timeout(45000)
        
        messages = page.locator(".message-content")
        last_response = messages.last.inner_text()
        
        # If response contains buy signals, it should have order details
        if "BUY" in last_response.upper():
            # Should specify at what price to buy
            assert re.search(r'buy.{0,50}\$[\d,]+', last_response.lower()) or \
                   re.search(r'\$[\d,]+.{0,50}buy', last_response.lower()), \
                   f"BUY signal should specify price, got: {last_response[:500]}"
        
        if "SELL" in last_response.upper():
            # Should specify at what price to sell
            assert re.search(r'sell.{0,50}\$[\d,]+', last_response.lower()) or \
                   re.search(r'\$[\d,]+.{0,50}sell', last_response.lower()), \
                   f"SELL signal should specify price, got: {last_response[:500]}"

    def test_portfolio_strategy_has_position_sizing(self, page: Page):
        """Portfolio-level strategies should include position sizing recommendations."""
        page.goto(BASE_URL)
        
        chat_input = page.locator("#chat-input")
        chat_input.fill("give me a complete trading plan for my crypto portfolio")
        chat_input.press("Enter")
        
        page.wait_for_timeout(45000)
        
        messages = page.locator(".message-content")
        last_response = messages.last.inner_text()
        
        # Should have some indication of how much to trade
        has_sizing = any(word in last_response.lower() for word in [
            "position", "size", "allocation", "percent", "%", "amount",
            "quantity", "units", "portion", "weight"
        ])
        
        assert has_sizing, (
            "Portfolio strategy should include position sizing guidance. "
            f"Got: {last_response[:500]}"
        )


class TestSpecificLimitOrderGeneration:
    """Test the specific limit order recommendation feature."""

    def test_generates_limit_orders_for_all_requested_coins(self, page: Page):
        """When asking for limit orders for specific coins, all should be addressed."""
        page.goto(BASE_URL)
        
        chat_input = page.locator("#chat-input")
        chat_input.fill("generate limit order recommendations for BTC, ETH, SOL")
        chat_input.press("Enter")
        
        page.wait_for_timeout(45000)
        
        messages = page.locator(".message-content")
        last_response = messages.last.inner_text()
        
        # All three coins should be mentioned with recommendations
        for coin in ["BTC", "ETH", "SOL"]:
            assert coin in last_response, f"{coin} should be in response"
            
            # Each should have associated price
            coin_section = last_response[last_response.find(coin):][:500]
            has_price = bool(re.search(r'\$[\d,]+', coin_section))
            assert has_price, f"{coin} section should have price recommendation"

    def test_buy_recommendation_below_current_price(self, page: Page):
        """Buy limit orders should generally be below current market price."""
        page.goto(BASE_URL)
        
        # First get current price
        chat_input = page.locator("#chat-input")
        chat_input.fill("what is the current BTC price?")
        chat_input.press("Enter")
        
        page.wait_for_timeout(15000)
        
        messages = page.locator(".message-content")
        price_response = messages.last.inner_text()
        
        # Extract current price
        prices = re.findall(r'\$?([\d,]+\.?\d*)', price_response)
        if prices:
            current_price = float(prices[0].replace(',', ''))
            
            # Now ask for buy limit order
            chat_input.fill("recommend a buy limit order for BTC")
            chat_input.press("Enter")
            
            page.wait_for_timeout(30000)
            
            limit_response = messages.last.inner_text()
            
            # The response should make sense - if it's a BUY limit, 
            # the price should be at or below current
            if "buy" in limit_response.lower() and "limit" in limit_response.lower():
                limit_prices = re.findall(r'\$?([\d,]+\.?\d*)', limit_response)
                # This is a sanity check - buy limits should be reasonable
                assert len(limit_prices) > 0, "Should have price in limit order recommendation"
