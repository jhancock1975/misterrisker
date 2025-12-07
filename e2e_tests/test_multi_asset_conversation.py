"""
End-to-end test for multi-asset conversation flow.

Tests that Mister Risker can:
1. Generate crypto trading strategies with limit orders
2. Maintain conversation context
3. Switch to stocks and generate strategies using appropriate data sources

Run with: uv run pytest e2e_tests/test_multi_asset_conversation.py -v -s --headed
"""

import re
import pytest
from playwright.sync_api import Page


BASE_URL = "http://127.0.0.1:8000"


@pytest.mark.e2e
class TestMultiAssetConversation:
    """Test multi-turn conversation across crypto and stock assets."""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page):
        """Navigate to the app before each test."""
        page.goto(BASE_URL)
        page.wait_for_selector(".chat-container")
        yield

    def send_message_and_wait(self, page: Page, message: str, wait_ms: int = 45000) -> str:
        """Helper to send a message and wait for response."""
        input_field = page.locator("#messageInput")
        send_button = page.locator("#sendBtn")
        
        input_field.fill(message)
        send_button.click()
        
        page.wait_for_timeout(wait_ms)
        
        messages = page.locator(".message.assistant .message-content")
        return messages.last.inner_text()

    def test_crypto_context_then_stock_strategy(self, page: Page):
        """Test full conversation: BTC strategy -> context check -> IBM strategy.
        
        Verifies:
        1. Crypto strategies come from WebSocket data source
        2. Context is maintained between messages
        3. Stock strategies come from Schwab data source
        4. All strategies include entry, take profit, stop loss
        """
        print("\n" + "=" * 80)
        print("TEST 1: Bitcoin limit order with take profit and stop loss")
        print("=" * 80)
        
        # Step 1: Ask for Bitcoin trading strategy
        response1 = self.send_message_and_wait(
            page,
            "analyze the bitcoin block chain, historical price data, realtime price data "
            "and come up with a trading strategy to place a limit order with take profit "
            "and stop loss prices",
            wait_ms=45000
        )
        
        print(f"\nRESPONSE 1:\n{'-' * 80}\n{response1}\n{'-' * 80}")
        
        # Verify BTC response has required elements
        assert "BTC" in response1.upper(), "Response should mention BTC"
        assert any(term in response1.lower() for term in ["entry", "limit order"]), \
            "Response should have entry/limit order"
        assert any(term in response1.lower() for term in ["take profit", "target"]), \
            "Response should have take profit"
        assert "stop loss" in response1.lower(), "Response should have stop loss"
        assert "websocket" in response1.lower(), "BTC data should come from websocket"
        
        # Check for price values
        prices = re.findall(r'\$[\d,]+\.?\d*', response1)
        assert len(prices) >= 3, f"Should have at least 3 price values, found: {prices}"
        
        print("\n" + "=" * 80)
        print("TEST 2: Which crypto did I just ask for the analysis")
        print("=" * 80)
        
        # Step 2: Test context retention
        response2 = self.send_message_and_wait(
            page,
            "which crypto did I just ask for the analysis",
            wait_ms=30000
        )
        
        print(f"\nRESPONSE 2:\n{'-' * 80}\n{response2}\n{'-' * 80}")
        
        # Verify context is maintained
        assert any(term in response2.lower() for term in ["bitcoin", "btc"]), \
            f"Response should remember Bitcoin was discussed. Got: {response2}"
        
        print("\n" + "=" * 80)
        print("TEST 3: Strategy to trade IBM (stock)")
        print("=" * 80)
        
        # Step 3: Ask for IBM stock strategy
        response3 = self.send_message_and_wait(
            page,
            "give me a trading strategy with limit order, take profit and stop loss for IBM",
            wait_ms=45000
        )
        
        print(f"\nRESPONSE 3:\n{'-' * 80}\n{response3}\n{'-' * 80}")
        
        # Verify IBM response has required elements
        assert "IBM" in response3.upper(), "Response should mention IBM"
        assert any(term in response3.lower() for term in ["entry", "limit order"]), \
            "Response should have entry/limit order"
        assert any(term in response3.lower() for term in ["take profit", "target"]), \
            "Response should have take profit"
        assert "stop loss" in response3.lower(), "Response should have stop loss"
        assert "schwab" in response3.lower(), "IBM data should come from Schwab"
        
        # Check for price values
        prices = re.findall(r'\$[\d,]+\.?\d*', response3)
        assert len(prices) >= 3, f"Should have at least 3 price values, found: {prices}"
        
        print("\n✅ All assertions passed!")
        print("   - BTC strategy generated with WebSocket data")
        print("   - Context maintained (remembered Bitcoin)")
        print("   - IBM strategy generated with Schwab data")
        print("   - All strategies include entry, take profit, stop loss")
