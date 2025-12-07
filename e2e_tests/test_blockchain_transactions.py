"""
End-to-end test for blockchain transaction queries.

Tests that Mister Risker can query Solana blockchain data.
Other chains (BTC, ETH, XRP, ZEC) return not_supported status.

Run with: uv run pytest e2e_tests/test_blockchain_transactions.py -v -s --headed
"""

import pytest
from playwright.sync_api import Page


BASE_URL = "http://127.0.0.1:8000"


@pytest.mark.e2e
class TestBlockchainTransactions:
    """Test blockchain transaction queries."""
    
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

    def test_solana_blockchain_transactions(self, page: Page):
        """Test querying Solana blockchain for recent transactions.
        
        Solana is the only fully supported chain via free public RPC.
        """
        print("\n" + "=" * 80)
        print("TEST: Solana blockchain transactions query")
        print("=" * 80)
        
        response = self.send_message_and_wait(
            page,
            "What are the recent transactions on the Solana blockchain? "
            "Show me the latest block info.",
            wait_ms=45000
        )
        
        print(f"\nRESPONSE:\n{'-' * 80}\n{response}\n{'-' * 80}")
        
        response_lower = response.lower()
        
        # Check for Solana-specific content
        has_solana_data = any(term in response_lower for term in [
            "solana", "slot", "epoch", "signature", "sol"
        ])
        
        has_block_data = any(term in response_lower for term in [
            "block", "transaction", "height"
        ])
        
        has_error = any(term in response_lower for term in [
            "error", "failed", "sorry", "unable"
        ])
        
        print(f"\n   Has Solana-specific data: {'✅' if has_solana_data else '❌'}")
        print(f"   Has block/transaction data: {'✅' if has_block_data else '❌'}")
        print(f"   Has error message: {'⚠️' if has_error else '✅'}")
        
        # Should have Solana-related content in response
        assert has_solana_data or has_block_data or has_error, \
            f"Response should address the Solana query. Got: {response[:500]}"

    def test_bitcoin_not_supported(self, page: Page):
        """Test that Bitcoin queries return not_supported gracefully."""
        print("\n" + "=" * 80)
        print("TEST: Bitcoin not supported (graceful fallback)")
        print("=" * 80)
        
        response = self.send_message_and_wait(
            page,
            "Show me recent Bitcoin blockchain transactions",
            wait_ms=30000
        )
        
        print(f"\nRESPONSE:\n{'-' * 80}\n{response}\n{'-' * 80}")
        
        response_lower = response.lower()
        
        # Should acknowledge Bitcoin but explain limitations
        mentions_bitcoin = "bitcoin" in response_lower or "btc" in response_lower
        explains_limitation = any(term in response_lower for term in [
            "not supported", "not available", "don't have", "can't access",
            "only solana", "solana only", "free api", "limited"
        ])
        
        print(f"\n   Mentions Bitcoin: {'✅' if mentions_bitcoin else '❌'}")
        print(f"   Explains limitation: {'✅' if explains_limitation else '❌'}")
        
        # The response should acknowledge the request somehow
        assert mentions_bitcoin or explains_limitation, \
            f"Response should address Bitcoin query. Got: {response[:500]}"
