"""E2E tests for blockchain comparison analysis.

Tests that the app properly fetches and compares blockchain data
from Bitcoin and Solana when asked for correlation analysis.
"""

import pytest
from playwright.sync_api import Page


BASE_URL = "http://127.0.0.1:8000"


@pytest.mark.e2e
class TestBlockchainComparison:
    """Tests for blockchain comparison and correlation analysis."""

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
        # The welcome message is the first .message.assistant, so we wait for count > 1 or new content
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

    def test_bitcoin_solana_blockchain_comparison(self, page: Page):
        """Test that blockchain comparison query returns actual blockchain data.
        
        When user asks to analyze Bitcoin and Solana blockchains for correlations,
        the response should include:
        - Bitcoin network stats (block height, mempool, difficulty, etc.)
        - Solana network stats (block height, epoch, supply, etc.)
        - Comparison analysis section
        """
        print("\n" + "=" * 80)
        print("TEST: Bitcoin vs Solana Blockchain Comparison")
        print("=" * 80)
        
        query = "analyze bitcoin and solana block chains for the last 24 hours and tell me whether there are correlations"
        response_text = self.send_message_and_wait(page, query)
        
        print(f"\nQuery: {query}")
        print(f"\nResponse:\n{response_text[:1000]}...")
        
        # Verify we got blockchain data, not just price data
        # Check for Bitcoin blockchain indicators
        assert any(indicator in response_text for indicator in [
            "BITCOIN Network",
            "Bitcoin Network", 
            "Block Height",
            "Mempool",
            "Difficulty"
        ]), f"Response should contain Bitcoin blockchain data. Got: {response_text[:500]}"
        
        # Check for Solana blockchain indicators
        assert any(indicator in response_text for indicator in [
            "SOLANA Network",
            "Solana Network",
            "Epoch",
            "Slot"
        ]), f"Response should contain Solana blockchain data. Got: {response_text[:500]}"
        
        # Check for comparison analysis section
        assert any(indicator in response_text for indicator in [
            "Comparison Analysis",
            "Correlation",
            "correlation"
        ]), f"Response should contain comparison analysis. Got: {response_text[:500]}"
        
        # Verify we did NOT just get price data (the old incorrect behavior)
        if "Real-Time Analysis" in response_text and "Mempool" not in response_text:
            pytest.fail("Response appears to be price data only, not blockchain data")
        
        print("\n✅ Blockchain comparison test passed!")

    def test_bitcoin_blockchain_transactions(self, page: Page):
        """Test that Bitcoin blockchain transaction query returns transaction data."""
        print("\n" + "=" * 80)
        print("TEST: Bitcoin Blockchain Transactions")
        print("=" * 80)
        
        query = "show me recent bitcoin blockchain transactions"
        response_text = self.send_message_and_wait(page, query, wait_ms=20000)
        
        print(f"\nQuery: {query}")
        print(f"\nResponse:\n{response_text[:800]}...")
        
        # Should have Bitcoin transaction indicators
        assert any(indicator in response_text for indicator in [
            "Bitcoin Transactions",
            "TxID",
            "Transfer",
            "Coinbase (Mining"
        ]), f"Response should contain Bitcoin transactions. Got: {response_text[:500]}"
        
        # Should have transaction details
        assert "BTC" in response_text, "Response should mention BTC amounts"
        
        print("\n✅ Bitcoin transactions test passed!")

    def test_solana_blockchain_transactions(self, page: Page):
        """Test that Solana blockchain transaction query returns transaction data."""
        print("\n" + "=" * 80)
        print("TEST: Solana Blockchain Transactions")
        print("=" * 80)
        
        query = "show me recent solana blockchain transactions"
        response_text = self.send_message_and_wait(page, query, wait_ms=20000)
        
        print(f"\nQuery: {query}")
        print(f"\nResponse:\n{response_text[:800]}...")
        
        # Should have Solana transaction indicators
        assert any(indicator in response_text for indicator in [
            "Solana Transactions",
            "Signature",
            "SOL"
        ]), f"Response should contain Solana transactions. Got: {response_text[:500]}"
        
        print("\n✅ Solana transactions test passed!")

    def test_price_analysis_still_works(self, page: Page):
        """Test that price analysis queries still return price data (not blockchain).
        
        Ensures we didn't break the price analysis feature when fixing blockchain routing.
        """
        print("\n" + "=" * 80)
        print("TEST: Price Analysis (verify not broken)")
        print("=" * 80)
        
        # This query has "analyze" but NO "blockchain" - should get price data
        query = "analyze BTC price trends"
        response_text = self.send_message_and_wait(page, query, wait_ms=20000)
        
        print(f"\nQuery: {query}")
        print(f"\nResponse:\n{response_text[:500]}...")
        
        # Should have price analysis indicators
        assert any(indicator in response_text for indicator in [
            "Real-Time Analysis",
            "Price:",
            "Trend:",
            "Volatility:"
        ]), f"Response should contain price analysis. Got: {response_text[:500]}"
        
        print("\n✅ Price analysis test passed!")
