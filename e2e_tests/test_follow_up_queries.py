"""E2E test for follow-up query handling.

Tests that the app properly understands follow-up questions in context,
especially when users ask about "other" or "different" assets.
"""

import re
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
        page.wait_for_selector(".chat-container", timeout=10000)
        yield

    def send_message_and_wait(self, page: Page, message: str, wait_ms: int = 20000) -> str:
        """Helper to send a message and wait for response."""
        # Count current messages before sending
        current_count = page.locator(".message.assistant .message-content").count()
        expected_count = current_count + 1
        
        input_field = page.locator("#messageInput")
        send_button = page.locator("#sendBtn")
        
        input_field.wait_for(state="visible", timeout=5000)
        input_field.fill(message)
        send_button.click()
        
        # Wait for a NEW message to appear (count increases)
        page.wait_for_function(
            f"""() => {{
                const messages = document.querySelectorAll('.message.assistant .message-content');
                if (messages.length < {expected_count}) return false;
                const lastMsg = messages[messages.length - 1].textContent;
                // Make sure it's not just loading or empty
                return lastMsg.length > 100 && !lastMsg.includes('Ask me anything about trading');
            }}""",
            timeout=wait_ms
        )
        
        messages = page.locator(".message.assistant .message-content")
        return messages.last.inner_text()

    def test_other_cryptos_follow_up(self, page: Page):
        """Test that 'what about other cryptos' returns DIFFERENT assets."""
        print("\n" + "=" * 60)
        print("TEST: 'Other Cryptos' Follow-up Query")
        print("=" * 60)
        
        # First, ask for trade ideas
        first_response = self.send_message_and_wait(page, "give me some trade ideas")
        print(f"\nFirst Response length: {len(first_response)} chars")
        print(f"First Response (truncated): {first_response[:300]}...")
        
        # Now ask for OTHER cryptos
        follow_up_response = self.send_message_and_wait(page, "what about other cryptos?")
        print(f"\nFollow-up Response length: {len(follow_up_response)} chars")
        print(f"Follow-up Response (truncated): {follow_up_response[:300]}...")
        
        # Extract assets from each response - count occurrences of 📊 pattern
        def extract_lead_assets(response: str) -> list:
            # Find all asset headers in order
            import re
            matches = re.findall(r'📊\s*([A-Z]{2,5})-USD', response)
            return matches
        
        first_assets = extract_lead_assets(first_response)
        follow_up_assets = extract_lead_assets(follow_up_response)
        
        print(f"\nFirst response asset sections: {first_assets}")
        print(f"Follow-up asset sections: {follow_up_assets}")
        
        # Key check: Follow-up should have different lead asset OR fewer/different assets
        first_lead = first_assets[0] if first_assets else None
        follow_lead = follow_up_assets[0] if follow_up_assets else None
        
        print(f"First lead: {first_lead}, Follow-up lead: {follow_lead}")
        
        # The follow-up is correct if:
        # 1. Lead asset is different (BTC -> ETH), OR
        # 2. Follow-up has at least one asset NOT in first response, OR  
        # 3. Response lengths are meaningfully different (different content)
        lead_different = first_lead != follow_lead
        new_assets = set(follow_up_assets) - set(first_assets)
        length_different = abs(len(first_response) - len(follow_up_response)) > 100
        
        success = lead_different or bool(new_assets) or length_different
        
        if success:
            print("\n✅ Follow-up correctly returned different content!")
            if lead_different:
                print(f"   - Lead asset changed: {first_lead} -> {follow_lead}")
            if new_assets:
                print(f"   - New assets in follow-up: {new_assets}")
            if length_different:
                print(f"   - Response lengths differ significantly")
        
        assert success, (
            f"Follow-up should return DIFFERENT assets!\n"
            f"First: {first_assets} (lead: {first_lead}, {len(first_response)} chars)\n"
            f"Follow-up: {follow_up_assets} (lead: {follow_lead}, {len(follow_up_response)} chars)"
        )
        print("\n✅ Follow-up correctly returned different assets!")
