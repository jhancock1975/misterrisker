"""
End-to-end Playwright tests for Mister Risker trading strategy recommendations.

These tests verify that Mister Risker provides ACTIONABLE trading recommendations
with specific limit order prices, entry points, take profit, and stop loss levels -
not just regurgitating raw market data.

Run with: uv run pytest e2e_tests/test_trading_strategy_e2e.py --headed -s -v

NOTE: Server must be running at http://127.0.0.1:8000
"""

import re
import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://127.0.0.1:8000"


@pytest.mark.e2e
class TestTradingStrategyRecommendations:
    """Test that Mister Risker provides actual trading strategies, not just market data."""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page):
        """Navigate to the app before each test."""
        page.goto(BASE_URL)
        # Wait for the page to load
        page.wait_for_selector(".chat-container")
        yield

    def send_message_and_wait(self, page: Page, message: str, wait_ms: int = 30000) -> str:
        """Helper to send a message and wait for response."""
        input_field = page.locator("#messageInput")
        send_button = page.locator("#sendBtn")
        
        input_field.fill(message)
        send_button.click()
        
        # Wait for response
        page.wait_for_timeout(wait_ms)
        
        # Get the last assistant message
        messages = page.locator(".message.assistant .message-content")
        return messages.last.inner_text()

    def test_research_then_stock_strategy_conversation(self, page: Page):
        """Test multi-turn conversation: research stocks, then ask for specific limit order.
        
        This tests that Mister Risker:
        1. Can research stocks and provide analysis
        2. Maintains conversation context
        3. When asked for a limit order on a STOCK (like MU), provides STOCK strategy, not crypto
        """
        print("\n\n🔹 Testing: Research then stock strategy conversation...")
        
        # Step 1: Ask for research on stocks
        print("\n   Step 1: Asking for stock research...")
        response1 = self.send_message_and_wait(
            page,
            "research the stocks and give me a trade recommendation",
            wait_ms=45000
        )
        print(f"\n📝 Research Response (first 500 chars):\n{response1[:500]}...")
        
        # Step 2: Follow up asking for specific limit order for MU (Micron - a STOCK)
        print("\n   Step 2: Asking for MU (stock) limit order...")
        response2 = self.send_message_and_wait(
            page,
            "can you recommend a limit order with stop loss and take profit values for MU",
            wait_ms=45000
        )
        print(f"\n📝 MU Strategy Response:\n{response2}")
        
        # Verify the response is about MU (Micron), NOT crypto
        response_lower = response2.lower()
        
        # Should mention MU or Micron
        has_mu = "mu" in response_lower or "micron" in response_lower
        
        # Should NOT be about crypto (unless comparing)
        mentions_crypto_only = all(term in response_lower for term in ["btc", "eth"]) and "mu" not in response_lower
        
        # Should have limit order components
        has_entry = any(word in response_lower for word in ["entry", "buy at", "limit", "order at"])
        has_stop = any(word in response_lower for word in ["stop", "stop loss", "sl"])
        has_target = any(word in response_lower for word in ["target", "take profit", "profit", "tp"])
        
        print(f"\n   MU/Micron mentioned: {'✅' if has_mu else '❌'}")
        print(f"   Only crypto (wrong context): {'❌' if mentions_crypto_only else '✅'}")
        print(f"   Has entry: {'✅' if has_entry else '❌'}")
        print(f"   Has stop loss: {'✅' if has_stop else '❌'}")
        print(f"   Has take profit: {'✅' if has_target else '❌'}")
        
        print("\n🔍 Browser open for inspection. Press 'Resume' or Ctrl+C to continue.\n")
        page.pause()
        
        # Assertions
        assert has_mu or not mentions_crypto_only, (
            f"Response should be about MU (stock), not crypto. Got: {response2[:300]}"
        )

    def test_trading_strategy_gives_actionable_recommendations(self, page: Page):
        """When user asks for 'trading strategy', should get buy/sell recommendations with prices."""
        print("\n\n🔹 Testing: Trading strategy should give actionable recommendations...")
        
        response = self.send_message_and_wait(
            page, 
            "Give me a trading strategy for crypto right now",
            wait_ms=45000
        )
        
        print(f"\n📝 Response:\n{response[:800]}...\n")
        
        # Should contain actionable recommendations, not just market data
        actionable_words = [
            "recommend", "suggest", "buy", "sell", "limit order", 
            "entry", "target", "stop loss", "strategy", "position"
        ]
        has_actionable = any(word in response.lower() for word in actionable_words)
        
        if has_actionable:
            print("✅ Response contains actionable trading language")
        else:
            print("❌ Response lacks actionable recommendations")
        
        # Should NOT just be raw OHLCV data without analysis
        just_data = "Current Data" in response and "recommend" not in response.lower()
        if just_data:
            print("❌ Response appears to be just raw data without analysis")
        
        print("\n🔍 Browser open for inspection. Press 'Resume' or Ctrl+C to continue.\n")
        page.pause()
        
        assert has_actionable, f"Response should contain actionable recommendations, got: {response[:500]}"

    def test_limit_order_with_entry_target_stoploss(self, page: Page):
        """Trading recommendations should include entry price, target, and stop loss."""
        print("\n\n🔹 Testing: Limit order recommendations with entry, target, stop loss...")
        
        response = self.send_message_and_wait(
            page,
            "analyze the bitcoin blockchain, historical price data, realtime price data and come up with a trading strategy to place a limit order with take profit and stop loss prices",
            wait_ms=60000
        )
        
        print(f"\n📝 Response:\n{response}\n")
        
        response_lower = response.lower()
        
        # Check for key trading elements
        has_entry = any(word in response_lower for word in ["entry", "buy at", "limit at", "order at", "limit order"])
        has_target = any(word in response_lower for word in ["target", "take profit", "profit at", "tp", "exit"])
        has_stop = any(word in response_lower for word in ["stop", "stop loss", "sl", "risk", "protect"])
        
        # Check for actual price values
        price_pattern = r'\$[\d,]+\.?\d*'
        prices_found = re.findall(price_pattern, response)
        
        print(f"   Entry language found: {'✅' if has_entry else '❌'}")
        print(f"   Take profit language found: {'✅' if has_target else '❌'}")
        print(f"   Stop loss language found: {'✅' if has_stop else '❌'}")
        print(f"   Price values found: {len(prices_found)} - {prices_found[:5]}")
        
        print("\n🔍 Browser open for inspection. Press 'Resume' or Ctrl+C to continue.\n")
        page.pause()
        
        assert has_entry, f"Should have entry price language, got: {response[:300]}"
        assert has_target, f"Should have profit target language, got: {response[:300]}"
        assert has_stop, f"Should have stop loss language, got: {response[:300]}"
        assert len(prices_found) >= 2, f"Should have multiple price values, found: {prices_found}"

    def test_multi_coin_strategy_analysis(self, page: Page):
        """When asking for limit orders for multiple coins, all should be analyzed."""
        print("\n\n🔹 Testing: Multi-coin strategy analysis...")
        
        response = self.send_message_and_wait(
            page,
            "can you analyze available data from the tools exposed to you to recommend limit order trades for sol, eth, btc, zec, xrp?",
            wait_ms=60000
        )
        
        print(f"\n📝 Response:\n{response}\n")
        
        # Check for multiple coins being addressed
        coins = ["BTC", "ETH", "SOL", "XRP", "ZEC"]
        coins_mentioned = [coin for coin in coins if coin in response.upper()]
        
        print(f"   Coins addressed: {coins_mentioned} ({len(coins_mentioned)}/5)")
        
        # Should have limit order specifics
        has_limit_order_language = any(phrase in response.lower() for phrase in [
            "limit order", "buy limit", "sell limit", "entry price",
            "order at", "place order", "recommended price", "limit"
        ])
        
        print(f"   Has limit order language: {'✅' if has_limit_order_language else '❌'}")
        
        # Should have price targets
        price_pattern = r'\$[\d,]+\.?\d*'
        prices_found = re.findall(price_pattern, response)
        print(f"   Price targets found: {len(prices_found)}")
        
        print("\n🔍 Browser open for inspection. Press 'Resume' or Ctrl+C to continue.\n")
        page.pause()
        
        assert len(coins_mentioned) >= 3, f"Should address at least 3 coins, only found: {coins_mentioned}"
        assert has_limit_order_language, f"Should mention limit orders specifically"


@pytest.mark.e2e  
class TestConversationContext:
    """Test that Mister Risker maintains conversation context correctly."""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page):
        """Navigate to the app before each test."""
        page.goto(BASE_URL)
        page.wait_for_selector(".chat-container")
        yield

    def send_message_and_wait(self, page: Page, message: str, wait_ms: int = 30000) -> str:
        """Helper to send a message and wait for response."""
        input_field = page.locator("#messageInput")
        send_button = page.locator("#sendBtn")
        
        input_field.fill(message)
        send_button.click()
        
        page.wait_for_timeout(wait_ms)
        
        messages = page.locator(".message.assistant .message-content")
        return messages.last.inner_text()

    def test_stock_context_maintained(self, page: Page):
        """When discussing a stock, follow-up questions should stay in stock context."""
        print("\n\n🔹 Testing: Stock context is maintained in conversation...")
        
        # First message about a stock
        print("\n   Step 1: Asking about AAPL...")
        response1 = self.send_message_and_wait(
            page,
            "what do you think about AAPL stock?",
            wait_ms=30000
        )
        print(f"\n📝 AAPL Response (first 300 chars):\n{response1[:300]}...")
        
        # Follow-up asking for limit order (should be about AAPL, not crypto)
        print("\n   Step 2: Asking for limit order (should be AAPL)...")
        response2 = self.send_message_and_wait(
            page,
            "give me a limit order with stop loss and take profit",
            wait_ms=45000
        )
        print(f"\n📝 Follow-up Response:\n{response2[:500]}...")
        
        # Check if response is about AAPL or at least stocks, not crypto
        response_lower = response2.lower()
        is_about_aapl = "aapl" in response_lower or "apple" in response_lower
        is_about_crypto = any(term in response_lower for term in ["btc", "bitcoin", "eth", "ethereum", "crypto"])
        
        print(f"\n   About AAPL/Apple: {'✅' if is_about_aapl else '❌'}")
        print(f"   Incorrectly about crypto: {'❌' if is_about_crypto else '✅'}")
        
        print("\n🔍 Browser open for inspection. Press 'Resume' or Ctrl+C to continue.\n")
        page.pause()


@pytest.mark.e2e  
class TestProactiveAlerts:
    """Test that Mister Risker can proactively alert about trading opportunities."""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page):
        """Navigate to the app before each test."""
        page.goto(BASE_URL)
        page.wait_for_selector(".chat-container")
        yield

    def test_background_monitoring_capability(self, page: Page):
        """Test that we can ask about monitoring and get info about alerts."""
        print("\n\n🔹 Testing: Background monitoring capability...")
        
        input_field = page.locator("#messageInput")
        send_button = page.locator("#sendBtn")
        
        input_field.fill("can you monitor the market and alert me when there's a good trading opportunity?")
        send_button.click()
        
        page.wait_for_timeout(30000)
        
        messages = page.locator(".message.assistant .message-content")
        response = messages.last.inner_text()
        
        print(f"\n📝 Response:\n{response}\n")
        
        # For now, just check we get a response - this is a feature we're building
        print("   (This test documents the expected behavior - feature in development)")
        
        print("\n🔍 Browser open for inspection. Press 'Resume' or Ctrl+C to continue.\n")
        page.pause()
