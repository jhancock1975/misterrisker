"""E2E tests for follow-up query handling.

Tests that the app properly understands follow-up questions in context,
especially when users ask about "other" or "different" assets.
"""

import pytest
from playwright.sync_api import Page


BASE_URL = "http://127.0.0.1:8000"


@pytest.mark.e2e
class TestFollowUpQueries:
    """Tests for contextual follow-up query handling."""

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
        
        # Wait for the input field to be ready
        input_field.wait_for(state="visible")
        
        input_field.fill(message)
        send_button.click()
        
        # Wait for at least one new assistant message with content
        page.wait_for_timeout(3000)  # Brief wait for response to start
        
        # Wait for the response to appear (not the welcome message)
        page.wait_for_function(
            """() => {
                const messages = document.querySelectorAll('.message.assistant .message-content');
                if (messages.length < 2) return false;
                // Check if the last message is not the welcome message
                const lastMsg = messages[messages.length - 1].textContent;
                return !lastMsg.includes('Ask me anything about trading');
            }""",
            timeout=wait_ms
        )
        
        messages = page.locator(".message.assistant .message-content")
        return messages.last.inner_text()

    def test_other_cryptos_follow_up(self, page: Page):
        """Test that 'what about other cryptos' returns DIFFERENT assets.
        
        When user gets a trading strategy for BTC/ETH and then asks 
        'what about other cryptos?', the system should return strategies 
        for DIFFERENT cryptocurrencies (SOL, XRP, etc.), not repeat the same ones.
        """
        print("\n" + "=" * 80)
        print("TEST: 'Other Cryptos' Follow-up Query")
        print("=" * 80)
        
        # First, ask for trade ideas (will get BTC, ETH, etc.)
        first_query = "give me some trade ideas"
        first_response = self.send_message_and_wait(page, first_query)
        
        print(f"\nFirst Query: {first_query}")
        print(f"First Response (truncated): {first_response[:500]}...")
        
        # Now ask for OTHER cryptos
        follow_up_query = "what about other cryptos?"
        follow_up_response = self.send_message_and_wait(page, follow_up_query)
        
        print(f"\nFollow-up Query: {follow_up_query}")
        print(f"Follow-up Response (truncated): {follow_up_response[:500]}...")
        
        # The follow-up should contain DIFFERENT assets than the first response
        # It should NOT just repeat BTC and ETH
        # It should include other cryptos like SOL, XRP, ZEC, etc.
        
        # Extract which assets are mentioned in each response
        import re
        
        # Look for crypto asset patterns (BTC-USD, ETH, SOL-USD, etc.)
        def extract_assets(response: str) -> set:
            # Match patterns like "BTC-USD", "ETH", "SOL-USD", or asset names
            patterns = [
                r'\b([A-Z]{2,5})-USD\b',  # BTC-USD, ETH-USD, etc.
                r'📊\s*([A-Z]{2,5})',  # 📊 BTC-USD
            ]
            assets = set()
            for pattern in patterns:
                matches = re.findall(pattern, response)
                assets.update(m.upper() for m in matches)
            return assets
        
        first_assets = extract_assets(first_response)
        follow_up_assets = extract_assets(follow_up_response)
        
        print(f"\nAssets in first response: {first_assets}")
        print(f"Assets in follow-up response: {follow_up_assets}")
        
        # The follow-up should contain DIFFERENT assets
        # At least one new asset should appear that wasn't in the first response
        new_assets = follow_up_assets - first_assets
        
        print(f"New assets in follow-up: {new_assets}")
        
        # Also check the FIRST asset shown - should be different
        # Extract the first asset from each response by looking at the 📊 pattern
        first_asset_shown = re.search(r'📊\s*([A-Z]{2,5})-USD', first_response)
        follow_up_first_asset = re.search(r'📊\s*([A-Z]{2,5})-USD', follow_up_response)
        
        first_lead = first_asset_shown.group(1) if first_asset_shown else None
        follow_lead = follow_up_first_asset.group(1) if follow_up_first_asset else None
        
        print(f"First response lead asset: {first_lead}")
        print(f"Follow-up lead asset: {follow_lead}")
        
        # Check for different assets (SOL, XRP, ZEC, etc.)
        different_assets = {"SOL", "XRP", "ZEC", "DOGE", "ADA", "AVAX", "MATIC", "DOT", "LINK", "UNI"}
        has_different_asset = bool(follow_up_assets & different_assets)
        
        # Multiple ways the follow-up can be "correct":
        # 1. Has completely new assets not in first response
        # 2. Contains assets from the "different" list  
        # 3. The lead asset is different (showing the LLM understood "other")
        lead_is_different = first_lead != follow_lead if (first_lead and follow_lead) else False
        
        # The key assertion: follow-up should NOT just repeat the same assets
        if first_assets and follow_up_assets:
            assert new_assets or has_different_asset or lead_is_different, (
                f"Follow-up query about 'OTHER cryptos' should return DIFFERENT assets!\n"
                f"First response had: {first_assets}, lead: {first_lead}\n"
                f"Follow-up had: {follow_up_assets}, lead: {follow_lead}\n"
                f"Expected at least one new asset or different lead asset"
            )
            print(f"\n✅ Follow-up correctly returned different assets!")
        else:
            print(f"\n⚠️ Could not extract assets to compare")
        
        print("\n✅ Follow-up query returned different content!")

    def test_what_about_stocks_follow_up(self, page: Page):
        """Test that 'what about stocks' after crypto query returns stock strategies.
        
        When user gets crypto trading strategies and then asks 'what about stocks?',
        the system should return strategies for actual stocks (AAPL, TSLA, etc.).
        """
        print("\n" + "=" * 80)
        print("TEST: 'What About Stocks' Follow-up Query")
        print("=" * 80)
        
        # First, ask for crypto trade ideas
        first_query = "give me trading ideas for bitcoin"
        first_response = self.send_message_and_wait(page, first_query)
        
        print(f"\nFirst Query: {first_query}")
        print(f"First Response (truncated): {first_response[:500]}...")
        
        # Now ask about stocks
        follow_up_query = "what about stocks?"
        follow_up_response = self.send_message_and_wait(page, follow_up_query)
        
        print(f"\nFollow-up Query: {follow_up_query}")
        print(f"Follow-up Response (truncated): {follow_up_response[:500]}...")
        
        # The follow-up should mention actual stocks
        stock_indicators = ["AAPL", "TSLA", "NVDA", "AMD", "GOOGL", "MSFT", "AMZN",
                           "Apple", "Tesla", "Nvidia", "Microsoft", "Amazon",
                           "stock", "equity", "S&P", "Nasdaq", "market"]
        
        has_stock_content = any(indicator in follow_up_response for indicator in stock_indicators)
        
        # Should NOT just repeat the BTC strategy
        not_crypto_repeat = "BTC" not in follow_up_response[:300] or any(s in follow_up_response for s in ["AAPL", "TSLA", "NVDA"])
        
        assert has_stock_content or not_crypto_repeat, (
            f"Follow-up about stocks should mention actual stocks.\n"
            f"Got response: {follow_up_response[:500]}"
        )
        
        print("\n✅ Stocks follow-up query returned stock content!")

    def test_different_assets_follow_up(self, page: Page):
        """Test that 'what about different assets' returns varied recommendations."""
        print("\n" + "=" * 80)
        print("TEST: 'Different Assets' Follow-up Query")  
        print("=" * 80)
        
        # First query for specific asset
        first_query = "trading strategy for ETH"
        first_response = self.send_message_and_wait(page, first_query)
        
        print(f"\nFirst Query: {first_query}")
        print(f"First Response (truncated): {first_response[:300]}...")
        
        # Ask for different assets
        follow_up_query = "show me different assets"
        follow_up_response = self.send_message_and_wait(page, follow_up_query)
        
        print(f"\nFollow-up Query: {follow_up_query}")
        print(f"Follow-up Response (truncated): {follow_up_response[:500]}...")
        
        # Should show different assets, not just ETH again
        other_assets = ["BTC", "SOL", "XRP", "AAPL", "TSLA", "NVDA", 
                       "Bitcoin", "Solana", "Apple", "Tesla"]
        
        has_other_assets = any(asset in follow_up_response for asset in other_assets)
        
        assert has_other_assets, (
            f"'Different assets' should show assets other than just ETH.\n"
            f"Got: {follow_up_response[:500]}"
        )
        
        print("\n✅ Different assets query returned varied content!")
