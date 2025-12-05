"""
Tests for the Mister Risker Web Chat Application.

These tests cover the TradingChatBot class and its integration
with the LLM and MCP servers (Coinbase and Schwab).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_mcp_server():
    """Create a mock Coinbase MCP server."""
    server = MagicMock()
    
    async def mock_call_tool(tool_name: str, params: dict):
        responses = {
            "get_accounts": {
                "accounts": [
                    {"uuid": "acc-123", "currency": "BTC", "available_balance": {"value": "1.5", "currency": "BTC"}},
                    {"uuid": "acc-456", "currency": "USD", "available_balance": {"value": "5000.00", "currency": "USD"}}
                ]
            },
            "get_best_bid_ask": {
                "pricebooks": [
                    {"product_id": "BTC-USD", "bids": [{"price": "45000"}], "asks": [{"price": "45001"}]}
                ]
            }
        }
        return responses.get(tool_name, {})
    
    server.call_tool = AsyncMock(side_effect=mock_call_tool)
    return server


@pytest.fixture
def mock_llm_response_string():
    """Create a mock LLM response with string content."""
    response = MagicMock()
    response.content = '{"tool": "get_accounts", "params": {}}'
    return response


@pytest.fixture
def mock_llm_response_list():
    """Create a mock LLM response with list content (Responses API format)."""
    response = MagicMock()
    # Responses API can return content as a list of content blocks
    response.content = [
        {"type": "text", "text": '{"tool": "get_accounts", "params": {}}'}
    ]
    return response


@pytest.fixture
def mock_llm_response_text_only():
    """Create a mock LLM response with plain text."""
    response = MagicMock()
    response.content = "Hello! How can I help you today?"
    return response


@pytest.fixture
def mock_llm_response_text_as_list():
    """Create a mock LLM response with plain text as list (Responses API format)."""
    response = MagicMock()
    response.content = [
        {"type": "text", "text": "Hello! How can I help you today?"}
    ]
    return response


# =============================================================================
# Tool Call Extraction Tests
# =============================================================================

class TestToolCallExtraction:
    """Tests for extracting tool calls from LLM responses."""

    def test_extract_tool_call_from_pure_json(self):
        """Should extract tool call from pure JSON response."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        text = '{"tool": "get_accounts", "params": {}}'
        result = bot._extract_tool_call(text)
        
        assert result is not None
        assert result["tool"] == "get_accounts"
        assert result["params"] == {}

    def test_extract_tool_call_from_mixed_text_and_json(self):
        """Should extract tool call when LLM prefixes with text like 'Let me check...'"""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        text = 'Let me check your balance for you. {"tool": "get_accounts", "params": {}}'
        result = bot._extract_tool_call(text)
        
        assert result is not None
        assert result["tool"] == "get_accounts"

    def test_extract_tool_call_with_params(self):
        """Should extract tool call with nested params object."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        text = 'I will look that up. {"tool": "get_product", "params": {"product_id": "BTC-USD"}}'
        result = bot._extract_tool_call(text)
        
        assert result is not None
        assert result["tool"] == "get_product"
        assert result["params"]["product_id"] == "BTC-USD"

    def test_extract_tool_call_returns_none_for_plain_text(self):
        """Should return None when no tool call is present."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        text = "Hello! How can I help you today?"
        result = bot._extract_tool_call(text)
        
        assert result is None

    def test_extract_tool_call_ignores_non_tool_json(self):
        """Should return None for JSON that doesn't have 'tool' key."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        text = '{"name": "John", "age": 30}'
        result = bot._extract_tool_call(text)
        
        assert result is None

    def test_extract_tool_call_with_text_after_json(self):
        """Should extract tool call even with text after the JSON."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        text = 'Let me check. {"tool": "get_accounts", "params": {}} I will get that for you.'
        result = bot._extract_tool_call(text)
        
        assert result is not None
        assert result["tool"] == "get_accounts"


# =============================================================================
# Response Content Extraction Tests
# =============================================================================

class TestResponseContentExtraction:
    """Tests for extracting content from LLM responses."""

    def test_extract_content_from_string(self):
        """Should extract content when response.content is a string."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        # String content
        content = "Hello world"
        result = bot._extract_content(content)
        
        assert result == "Hello world"

    def test_extract_content_from_list_with_text_block(self):
        """Should extract content when response.content is a list with text blocks."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        # List content (Responses API format)
        content = [{"type": "text", "text": "Hello world"}]
        result = bot._extract_content(content)
        
        assert result == "Hello world"

    def test_extract_content_from_list_with_multiple_blocks(self):
        """Should concatenate multiple text blocks."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        content = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"}
        ]
        result = bot._extract_content(content)
        
        assert result == "Hello world"

    def test_extract_content_from_empty_list(self):
        """Should return empty string for empty list."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        content = []
        result = bot._extract_content(content)
        
        assert result == ""

    def test_extract_content_handles_none(self):
        """Should handle None content gracefully."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        result = bot._extract_content(None)
        
        assert result == ""


# =============================================================================
# Message Processing Tests
# =============================================================================

class TestMessageProcessing:
    """Tests for processing user messages."""

    @pytest.mark.asyncio
    async def test_process_message_with_list_content_response(self, mock_mcp_server):
        """Should handle LLM response with list content (Responses API)."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.coinbase_server = mock_mcp_server
        
        # Mock LLM that returns list content
        mock_llm = AsyncMock()
        # First call returns tool call as list
        mock_llm.ainvoke.side_effect = [
            MagicMock(content=[{"type": "text", "text": '{"tool": "get_accounts", "params": {}}'}]),
            MagicMock(content=[{"type": "text", "text": "You have 1.5 BTC and $5000 USD in your accounts."}])
        ]
        bot.llm = mock_llm
        
        result = await bot.process_message("What's my account balance?")
        
        assert "error" not in result.lower() or "strip" not in result.lower()
        assert mock_mcp_server.call_tool.called

    @pytest.mark.asyncio
    async def test_process_message_with_string_content_response(self, mock_mcp_server):
        """Should handle LLM response with string content (standard format)."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.coinbase_server = mock_mcp_server
        
        # Mock LLM that returns string content
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content='{"tool": "get_accounts", "params": {}}'),
            MagicMock(content="You have 1.5 BTC and $5000 USD in your accounts.")
        ]
        bot.llm = mock_llm
        
        result = await bot.process_message("What's my account balance?")
        
        assert "error" not in result.lower() or "strip" not in result.lower()

    @pytest.mark.asyncio
    async def test_process_message_plain_text_response(self, mock_mcp_server):
        """Should handle plain text responses without tool calls."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.coinbase_server = mock_mcp_server
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="Hello! How can I help you today?")
        bot.llm = mock_llm
        
        result = await bot.process_message("Hi there!")
        
        assert result == "Hello! How can I help you today?"

    @pytest.mark.asyncio
    async def test_process_message_no_llm_configured(self):
        """Should return error when LLM is not configured."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.llm = None
        bot.coinbase_server = MagicMock()
        
        result = await bot.process_message("Hello")
        
        assert "OPENAI_API_KEY" in result

    @pytest.mark.asyncio
    async def test_process_message_no_mcp_server_configured(self):
        """Should return error when trying to trade without broker configured."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.llm = MagicMock()
        bot.coinbase_server = None
        bot.coinbase_agent = None  # Ensure agent is also None
        
        # A trading-related query should fail when broker not configured
        result = await bot.process_message("What's my balance?")
        
        # Should indicate Coinbase is not configured
        assert "coinbase not configured" in result.lower() or "error" in result.lower()


# =============================================================================
# Tool Execution Tests
# =============================================================================

class TestToolExecution:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mock_mcp_server):
        """Should execute tool and return result."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.coinbase_server = mock_mcp_server
        
        result = await bot._execute_tool("get_accounts", {})
        
        assert "accounts" in result
        mock_mcp_server.call_tool.assert_called_once_with("get_accounts", {})

    @pytest.mark.asyncio
    async def test_execute_tool_handles_error(self, mock_mcp_server):
        """Should handle tool execution errors gracefully."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        mock_mcp_server.call_tool.side_effect = Exception("API Error")
        bot.coinbase_server = mock_mcp_server
        
        result = await bot._execute_tool("get_accounts", {})
        
        assert "error" in result


# =============================================================================
# Conversation History Tests
# =============================================================================

class TestConversationHistory:
    """Tests for conversation history management."""

    def test_clear_history(self):
        """Should clear conversation history."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.conversation_history = [{"role": "user", "content": "test"}]
        
        bot.clear_history()
        
        assert bot.conversation_history == []

    @pytest.mark.asyncio
    async def test_history_accumulates_messages(self, mock_mcp_server):
        """Should accumulate messages in conversation history."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.coinbase_server = mock_mcp_server
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="Response 1")
        bot.llm = mock_llm
        
        await bot.process_message("Message 1")
        
        assert len(bot.conversation_history) == 2  # User message + AI response


# =============================================================================
# End-to-End Balance Query Tests
# =============================================================================

class TestBalanceQueryEndToEnd:
    """Tests for the full balance query flow - the user asks for balance and gets a readable response."""

    @pytest.mark.asyncio
    async def test_balance_query_returns_interpreted_result_not_json(self, mock_mcp_server):
        """When user asks for balance, should return friendly text, not raw JSON."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.coinbase_server = mock_mcp_server
        
        mock_llm = AsyncMock()
        # First LLM call: decides to call get_accounts tool
        # Second LLM call: interprets the result
        mock_llm.ainvoke.side_effect = [
            MagicMock(content='{"tool": "get_accounts", "params": {}}'),
            MagicMock(content="You have 1.5 BTC and $5,000.00 USD in your accounts.")
        ]
        bot.llm = mock_llm
        
        result = await bot.process_message("What is my balance?")
        
        # Should NOT contain raw JSON
        assert '"tool"' not in result
        assert '"accounts"' not in result
        assert '"available_balance"' not in result
        
        # Should contain friendly interpretation
        assert "1.5" in result or "BTC" in result or "5,000" in result or "5000" in result

    @pytest.mark.asyncio
    async def test_balance_query_with_mixed_llm_response_extracts_tool_call(self, mock_mcp_server):
        """When LLM responds with 'Let me check...' + JSON, should still execute tool."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.coinbase_server = mock_mcp_server
        
        mock_llm = AsyncMock()
        # LLM returns text before the JSON (common pattern)
        mock_llm.ainvoke.side_effect = [
            MagicMock(content='Let me check your account balance. {"tool": "get_accounts", "params": {}}'),
            MagicMock(content="Your portfolio contains 1.5 BTC and $5,000.00 USD.")
        ]
        bot.llm = mock_llm
        
        result = await bot.process_message("What is my balance?")
        
        # Tool should have been called
        mock_mcp_server.call_tool.assert_called_once_with("get_accounts", {})
        
        # Should return the interpreted result, not the mixed text/JSON
        assert "Let me check" not in result
        assert '"tool"' not in result

    @pytest.mark.asyncio
    async def test_balance_query_calls_mcp_server(self, mock_mcp_server):
        """Should actually call the MCP server to get real balances."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.coinbase_server = mock_mcp_server
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content='{"tool": "get_accounts", "params": {}}'),
            MagicMock(content="You have accounts with balances.")
        ]
        bot.llm = mock_llm
        
        await bot.process_message("What is my balance?")
        
        # Verify MCP server was called with correct tool
        mock_mcp_server.call_tool.assert_called_once()
        call_args = mock_mcp_server.call_tool.call_args
        assert call_args[0][0] == "get_accounts"

    @pytest.mark.asyncio
    async def test_balance_query_with_responses_api_list_format(self, mock_mcp_server):
        """Should handle OpenAI Responses API format (list content) correctly."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.coinbase_server = mock_mcp_server
        
        mock_llm = AsyncMock()
        # Responses API returns content as list of blocks
        mock_llm.ainvoke.side_effect = [
            MagicMock(content=[{"type": "text", "text": '{"tool": "get_accounts", "params": {}}'}]),
            MagicMock(content=[{"type": "text", "text": "You have 1.5 BTC worth approximately $67,500."}])
        ]
        bot.llm = mock_llm
        
        result = await bot.process_message("What is my balance?")
        
        # Should not error out
        assert "error" not in result.lower() or "strip" not in result.lower()
        # Tool should have been called
        mock_mcp_server.call_tool.assert_called_once_with("get_accounts", {})
        # Should return friendly text
        assert "1.5" in result or "BTC" in result or "67,500" in result

    @pytest.mark.asyncio
    async def test_balance_query_llm_interprets_tool_result(self, mock_mcp_server):
        """The LLM should be asked to interpret the raw tool result."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.coinbase_server = mock_mcp_server
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content='{"tool": "get_accounts", "params": {}}'),
            MagicMock(content="Here are your account balances:\n- 1.5 BTC\n- $5,000.00 USD")
        ]
        bot.llm = mock_llm
        
        result = await bot.process_message("Show me my balances")
        
        # LLM should have been called twice: once for tool decision, once for interpretation
        assert mock_llm.ainvoke.call_count == 2
        
        # Second call should include the tool result
        second_call_messages = mock_llm.ainvoke.call_args_list[1][0][0]
        # Look for the tool result in the messages
        found_tool_result = False
        for msg in second_call_messages:
            if hasattr(msg, 'content') and 'accounts' in str(msg.content):
                found_tool_result = True
                break
        assert found_tool_result, "LLM should receive tool result for interpretation"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling when MCP server fails or returns errors."""

    @pytest.fixture
    def mock_mcp_server_with_api_error(self):
        """Create a mock MCP server that returns API errors."""
        server = MagicMock()
        
        async def mock_call_tool(tool_name: str, params: dict):
            # Simulate a Coinbase API error
            return {"error": "API Error: Unable to load PEM file. See https://cryptography.io/en/latest/faq/#why-can-t-i-import-my-pem-file for more details."}
        
        server.call_tool = AsyncMock(side_effect=mock_call_tool)
        return server

    @pytest.fixture
    def mock_mcp_server_raises_exception(self):
        """Create a mock MCP server that raises exceptions."""
        from src.mcp_servers.coinbase import CoinbaseAPIError
        
        server = MagicMock()
        server.call_tool = AsyncMock(side_effect=CoinbaseAPIError("Invalid API key format"))
        return server

    @pytest.mark.asyncio
    async def test_api_error_returns_user_friendly_message(self, mock_mcp_server_with_api_error):
        """When MCP server returns an error dict, LLM should interpret it for the user."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.coinbase_server = mock_mcp_server_with_api_error
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content='{"tool": "get_accounts", "params": {}}'),
            MagicMock(content="There was an issue connecting to your Coinbase account. Please check your API key configuration.")
        ]
        bot.llm = mock_llm
        
        result = await bot.process_message("What is my balance?")
        
        # Should NOT show raw error details to user
        assert "PEM file" not in result
        assert "cryptography.io" not in result
        
        # Should show user-friendly message (from LLM interpretation)
        assert "issue" in result.lower() or "error" in result.lower() or "check" in result.lower()

    @pytest.mark.asyncio
    async def test_api_exception_is_caught_and_interpreted(self, mock_mcp_server_raises_exception):
        """When MCP server raises CoinbaseAPIError, should catch and interpret."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.coinbase_server = mock_mcp_server_raises_exception
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content='{"tool": "get_accounts", "params": {}}'),
            MagicMock(content="I couldn't retrieve your account information. The API key may be invalid.")
        ]
        bot.llm = mock_llm
        
        result = await bot.process_message("What is my balance?")
        
        # Tool should have been called
        mock_mcp_server_raises_exception.call_tool.assert_called_once()
        
        # LLM should interpret the error
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_error_result_passed_to_llm_for_interpretation(self):
        """Error results should be passed to LLM for interpretation, not returned raw."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        # MCP server returns error dict
        mock_server = MagicMock()
        mock_server.call_tool = AsyncMock(return_value={"error": "Some API error message"})
        bot.coinbase_server = mock_server
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content='{"tool": "get_accounts", "params": {}}'),
            MagicMock(content="Sorry, there was a problem accessing your account.")
        ]
        bot.llm = mock_llm
        
        result = await bot.process_message("Check my balance")
        
        # LLM should have been called twice
        assert mock_llm.ainvoke.call_count == 2
        
        # Second call should contain the error for interpretation
        second_call = mock_llm.ainvoke.call_args_list[1]
        messages = second_call[0][0]
        message_contents = [str(getattr(m, 'content', '')) for m in messages]
        assert any("error" in content.lower() for content in message_contents)

    @pytest.mark.asyncio
    async def test_generic_exception_handled_gracefully(self):
        """Generic exceptions should be handled without crashing."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        # MCP server raises generic exception
        mock_server = MagicMock()
        mock_server.call_tool = AsyncMock(side_effect=Exception("Network timeout"))
        bot.coinbase_server = mock_server
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content='{"tool": "get_accounts", "params": {}}'),
            MagicMock(content="There was a connection issue. Please try again.")
        ]
        bot.llm = mock_llm
        
        result = await bot.process_message("Show balance")
        
        # Should not crash - should return some response
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_execute_tool_returns_error_dict_on_coinbase_api_error(self):
        """_execute_tool should return error dict when CoinbaseAPIError is raised."""
        from web.app import TradingChatBot
        from src.mcp_servers.coinbase import CoinbaseAPIError
        
        bot = TradingChatBot()
        
        mock_server = MagicMock()
        mock_server.call_tool = AsyncMock(side_effect=CoinbaseAPIError("Invalid credentials"))
        bot.coinbase_server = mock_server
        
        result = await bot._execute_tool("get_accounts", {})
        
        assert isinstance(result, dict)
        assert "error" in result
        assert "Invalid credentials" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_tool_returns_error_dict_on_generic_exception(self):
        """_execute_tool should return error dict when generic exception is raised."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        mock_server = MagicMock()
        mock_server.call_tool = AsyncMock(side_effect=RuntimeError("Connection failed"))
        bot.coinbase_server = mock_server
        
        result = await bot._execute_tool("get_accounts", {})
        
        assert isinstance(result, dict)
        assert "error" in result
        assert "Connection failed" in result["error"]


# =============================================================================
# Broker Switching Tests
# =============================================================================

class TestBrokerSwitching:
    """Tests for broker switching functionality."""

    @pytest.mark.asyncio
    async def test_default_broker_is_coinbase(self):
        """Default broker should be coinbase."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        assert bot.active_broker == "coinbase"

    @pytest.mark.asyncio
    async def test_switch_to_schwab(self):
        """Should switch to Schwab when requested."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.llm = MagicMock()
        
        result = await bot.process_message("switch to schwab")
        
        assert bot.active_broker == "schwab"
        assert "Schwab" in result

    @pytest.mark.asyncio
    async def test_switch_to_coinbase(self):
        """Should switch to Coinbase when requested."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.llm = MagicMock()
        bot.active_broker = "schwab"  # Start in Schwab mode
        
        result = await bot.process_message("switch to coinbase")
        
        assert bot.active_broker == "coinbase"
        assert "Coinbase" in result

    @pytest.mark.asyncio
    async def test_coinbase_mode_checks_coinbase_server(self):
        """In Coinbase mode, should check coinbase_server not schwab_server."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.llm = MagicMock()
        bot.active_broker = "coinbase"
        bot.coinbase_server = None
        bot.schwab_server = MagicMock()  # Schwab is configured but shouldn't matter
        
        result = await bot.process_message("what's my balance?")
        
        # Should show Coinbase error (primary), may contain hint about Schwab
        assert "coinbase not configured" in result.lower()
        assert result.lower().startswith("error: coinbase")

    @pytest.mark.asyncio
    async def test_schwab_mode_checks_schwab_server(self):
        """In Schwab mode, should check schwab_server not coinbase_server."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.llm = MagicMock()
        bot.active_broker = "schwab"
        bot.schwab_server = None
        bot.coinbase_server = MagicMock()  # Coinbase is configured but shouldn't matter
        
        result = await bot.process_message("what's my balance?")
        
        # Should show Schwab error (primary), may contain hint about Coinbase
        assert "schwab not configured" in result.lower()
        assert result.lower().startswith("error: schwab")

    def test_clear_history_resets_broker_to_coinbase(self):
        """Clear history should reset broker to coinbase."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.active_broker = "schwab"
        bot.conversation_history = [{"test": "data"}]
        
        bot.clear_history()
        
        assert bot.active_broker == "coinbase"
        assert bot.conversation_history == []

    @pytest.mark.asyncio
    async def test_coinbase_mode_with_configured_server_proceeds(self, mock_mcp_server):
        """In Coinbase mode with configured server, should proceed to LLM."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.active_broker = "coinbase"
        bot.coinbase_server = mock_mcp_server
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="Here are your balances...")
        bot.llm = mock_llm
        
        result = await bot.process_message("what's my balance?")
        
        # Should have called the LLM (no error about server config)
        mock_llm.ainvoke.assert_called_once()
        assert "not configured" not in result.lower()


# =============================================================================
# HTML/UI Response Tests  
# =============================================================================

class TestUIResponses:
    """Tests for UI-related response formatting."""

    @pytest.mark.asyncio
    async def test_switch_response_contains_broker_name_for_ui_update(self):
        """Switch response should contain broker name in format UI can detect."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.llm = MagicMock()
        
        # Switch to Schwab
        result = await bot.process_message("switch to schwab")
        
        # Response should contain "Switched to **Schwab**" for UI detection
        assert "switched to" in result.lower()
        assert "schwab" in result.lower()

    @pytest.mark.asyncio
    async def test_switch_to_coinbase_response_for_ui_update(self):
        """Switch to coinbase response should be detectable by UI."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        bot.llm = MagicMock()
        bot.active_broker = "schwab"
        
        result = await bot.process_message("switch to coinbase")
        
        # Response should contain text for UI detection
        assert "switched to" in result.lower()
        assert "coinbase" in result.lower()
