"""
End-to-end test for Solana blockchain queries.

Tests that Mister Risker can query Solana blockchain data via free public RPC.

Run with: uv run pytest e2e_tests/test_solana_blockchain.py -v -s --headed
"""

import pytest
from playwright.sync_api import Page


BASE_URL = "http://127.0.0.1:8000"


@pytest.mark.e2e
class TestSolanaBlockchain:
    """Test Solana blockchain queries."""
    
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

    def test_largest_solana_transactions_24h(self, page: Page):
        """Test querying for the 3 largest Solana transactions in the last 24 hours.
        
        This tests Mister Risker's ability to:
        1. Query Solana blockchain via public RPC
        2. Find and report large/significant transactions
        3. Present transaction data clearly
        """
        print("\n" + "=" * 80)
        print("TEST: 3 Largest Solana Transactions (24h)")
        print("=" * 80)
        
        response = self.send_message_and_wait(
            page,
            "What are the 3 largest transactions on the Solana (SOL) blockchain "
            "in the last 24 hours?",
            wait_ms=90000  # Increased timeout for blockchain queries
        )
        
        print(f"\n{'=' * 80}")
        print("FULL RESPONSE FROM AGENT:")
        print("=" * 80)
        print(response)
        print("=" * 80)
        
        response_lower = response.lower()
        
        # Check for Solana-specific content
        has_solana_mention = any(term in response_lower for term in [
            "solana", "sol"
        ])
        
        has_transaction_data = any(term in response_lower for term in [
            "transaction", "signature", "transfer", "slot", "block"
        ])
        
        has_value_info = any(term in response_lower for term in [
            "sol", "lamport", "value", "amount", "largest", "biggest"
        ])
        
        # Check if there's an explanation about limitations
        has_limitation_explanation = any(term in response_lower for term in [
            "rpc", "api", "limit", "public", "available", "cannot", "don't have"
        ])
        
        print(f"\n   Mentions Solana: {'✅' if has_solana_mention else '❌'}")
        print(f"   Has transaction data: {'✅' if has_transaction_data else '❌'}")
        print(f"   Has value/amount info: {'✅' if has_value_info else '❌'}")
        print(f"   Explains limitations: {'ℹ️' if has_limitation_explanation else '➖'}")
        
        # The response should at minimum acknowledge the Solana query
        assert has_solana_mention, \
            f"Response should mention Solana. Got: {response[:500]}"
        
        # Should have some transaction-related content or explain why not
        assert has_transaction_data or has_limitation_explanation, \
            f"Response should include transaction data or explain limitations. Got: {response[:500]}"

    def test_solana_latest_block_info(self, page: Page):
        """Test querying for Solana's latest block/slot information."""
        print("\n" + "=" * 80)
        print("TEST: Solana Latest Block Info")
        print("=" * 80)
        
        response = self.send_message_and_wait(
            page,
            "What is the current Solana block height and epoch info?",
            wait_ms=45000
        )
        
        print(f"\nRESPONSE:\n{'-' * 80}\n{response}\n{'-' * 80}")
        
        response_lower = response.lower()
        
        has_block_info = any(term in response_lower for term in [
            "slot", "block", "height", "epoch"
        ])
        
        has_numbers = any(char.isdigit() for char in response)
        
        print(f"\n   Has block/slot info: {'✅' if has_block_info else '❌'}")
        print(f"   Contains numbers: {'✅' if has_numbers else '❌'}")
        
        assert has_block_info, \
            f"Response should include block/slot information. Got: {response[:500]}"

    def test_solana_network_stats(self, page: Page):
        """Test querying for Solana network statistics."""
        print("\n" + "=" * 80)
        print("TEST: Solana Network Stats")
        print("=" * 80)
        
        response = self.send_message_and_wait(
            page,
            "What are the current Solana network statistics? "
            "Like TPS, total supply, and circulating supply.",
            wait_ms=45000
        )
        
        print(f"\nRESPONSE:\n{'-' * 80}\n{response}\n{'-' * 80}")
        
        response_lower = response.lower()
        
        has_stats = any(term in response_lower for term in [
            "supply", "tps", "transaction", "validator", "stake", "epoch"
        ])
        
        has_solana = "solana" in response_lower or "sol" in response_lower
        
        print(f"\n   Mentions Solana: {'✅' if has_solana else '❌'}")
        print(f"   Has network stats: {'✅' if has_stats else '❌'}")
        
        assert has_solana, \
            f"Response should mention Solana. Got: {response[:500]}"
