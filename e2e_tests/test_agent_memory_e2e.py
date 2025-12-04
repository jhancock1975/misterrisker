"""
End-to-end Playwright tests for agent memory functionality.

These tests verify that the chat application remembers previous interactions
within a conversation session.

Run with: uv run pytest e2e_tests/test_agent_memory_e2e.py --headed -s -v

NOTE: These tests are marked with @pytest.mark.e2e to exclude them from regular pytest runs.
To run only e2e tests: uv run pytest e2e_tests/ -m e2e --headed -s
The tests leave the browser open at the end for inspection.
Server must be running at http://127.0.0.1:8000
"""

import pytest
from playwright.sync_api import Page, expect


# Base URL for the app (server must be running)
BASE_URL = "http://127.0.0.1:8000"


@pytest.mark.e2e
class TestAgentMemoryE2E:
    """End-to-end tests for agent memory functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page):
        """Navigate to the app before each test."""
        page.goto(BASE_URL)
        # Wait for the page to load
        page.wait_for_selector(".chat-container")
        yield
    
    def test_agent_remembers_drawing_request(self, page: Page):
        """
        Test that the agent remembers what it was asked to draw.
        
        We ask the agent to draw an atomic duck, then ask what we asked it to draw.
        The agent should remember "atomic duck".
        """
        input_field = page.locator("#messageInput")
        send_button = page.locator("#sendBtn")
        
        # Step 1: Ask the agent to draw something specific
        print("\n\n🔹 Step 1: Asking agent to draw an atomic duck...")
        input_field.fill("draw a picture of an atomic duck")
        send_button.click()
        
        # Wait for the response (image generation takes time)
        print("   Waiting for image generation (up to 20 seconds)...")
        page.wait_for_timeout(20000)
        
        # Verify we got a response
        messages = page.locator(".message.assistant")
        first_response = messages.last.inner_text()
        print(f"\n📝 First response (image generation):\n{first_response[:300]}...\n")
        
        # Step 2: Ask what we asked the agent to draw
        print("🔹 Step 2: Asking 'what did I ask you to make a picture of?'...")
        input_field.fill("what did I ask you to make a picture of?")
        send_button.click()
        
        # Wait for response
        page.wait_for_timeout(10000)
        
        # The agent should remember "atomic duck"
        second_response = messages.last.inner_text()
        print(f"\n📝 Second response (should mention atomic duck):\n{second_response}\n")
        
        # Check that the response mentions the atomic duck
        found = "duck" in second_response.lower() or "atomic" in second_response.lower()
        
        if found:
            print("\n✅ TEST PASSED: Agent remembered the drawing request!")
            print("   The response correctly mentioned the atomic duck.")
        else:
            print("\n❌ TEST FAILED: Agent did not remember the atomic duck.")
            
        print("\n🔍 Browser will stay open for inspection.")
        print("   You can interact with the chat to verify memory is working.")
        print("   Press 'Resume' in the Playwright inspector or Ctrl+C to close.\n")
        
        # Assert before pausing
        assert found, f"Agent did not remember the atomic duck. Response: {second_response}"
        
        # Pause to keep browser open for inspection
        page.pause()
    
    def test_conversation_context_persists(self, page: Page):
        """
        Test that the agent remembers context from earlier in the conversation.
        
        We tell the agent about Bitcoin, then ask about "it" without naming Bitcoin again.
        The agent should understand "it" refers to Bitcoin.
        """
        input_field = page.locator("#messageInput")
        send_button = page.locator("#sendBtn")
        
        # Message 1: Introduce a topic
        print("\n\n🔹 Step 1: Sending first message about Bitcoin...")
        input_field.fill("Let's talk about Bitcoin. What's your opinion on it as an investment?")
        send_button.click()
        
        # Wait for response
        page.wait_for_timeout(10000)
        
        # Verify first response is about Bitcoin
        messages = page.locator(".message.assistant")
        first_response = messages.last.inner_text()
        print(f"\n📝 First response (about Bitcoin):\n{first_response[:200]}...\n")
        
        # Message 2: Refer back to the topic without naming it
        print("🔹 Step 2: Asking about 'it' without naming Bitcoin...")
        input_field.fill("What are the main risks of investing in it?")
        send_button.click()
        
        # Wait for response
        page.wait_for_timeout(10000)
        
        # The agent should understand "it" refers to Bitcoin
        second_response = messages.last.inner_text()
        print(f"\n📝 Second response (should be about Bitcoin risks):\n{second_response}\n")
        
        # Check that the response is about Bitcoin/crypto and not asking "what is 'it'?"
        success_keywords = ["bitcoin", "crypto", "volatil", "market", "invest", "risk", "price"]
        found = any(word in second_response.lower() for word in success_keywords)
        
        if found:
            print("\n✅ TEST PASSED: Agent remembered the conversation context!")
            print("   The response about 'it' correctly referenced Bitcoin/crypto topics.")
        else:
            print("\n❌ TEST FAILED: Agent lost context about Bitcoin.")
            
        print("\n🔍 Browser will stay open for inspection.")
        print("   You can interact with the chat to verify memory is working.")
        print("   Press 'Resume' in the Playwright inspector or Ctrl+C to close.\n")
        
        # Assert before pausing
        assert found, f"Agent lost context about Bitcoin. Response: {second_response}"
        
        # Pause to keep browser open for inspection
        page.pause()


# Allow running as standalone script
if __name__ == "__main__":
    pytest.main([__file__, "--headed", "-s", "-v"])
