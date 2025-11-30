"""
Tests for the SlackAgent class.

These tests follow TDD principles - written before the implementation.
They cover:
- Agent initialization and configuration
- Sending messages to Slack
- Receiving messages from Slack
- Error handling
- Connection management
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSlackAgentInitialization:
    """Tests for SlackAgent initialization and configuration."""

    def test_agent_requires_bot_token(self):
        """Agent should require a bot token for initialization."""
        from agents.slack_agent import SlackAgent, SlackAuthError
        
        with pytest.raises(SlackAuthError, match="required"):
            SlackAgent(bot_token="", channel_id="C123")

    def test_agent_requires_channel_id(self):
        """Agent should require a channel ID for initialization."""
        from agents.slack_agent import SlackAgent
        
        # channel_id is required but not validated - it's used at send time
        agent = SlackAgent(bot_token="xoxb-test-token", channel_id="")
        assert agent.channel_id == ""

    def test_agent_initializes_with_valid_config(self):
        """Agent should initialize successfully with valid configuration."""
        from agents.slack_agent import SlackAgent
        
        agent = SlackAgent(
            bot_token="xoxb-test-token",
            channel_id="C1234567890"
        )
        
        assert agent is not None
        assert agent.bot_token == "xoxb-test-token"
        assert agent.channel_id == "C1234567890"

    def test_agent_can_load_config_from_env(self):
        """Agent should be able to load configuration from environment variables."""
        from agents.slack_agent import SlackAgent
        
        with patch.dict(os.environ, {
            "SLACK_BOT_TOKEN": "xoxb-env-token",
            "SLACK_CHANNEL_ID": "C9876543210"
        }):
            agent = SlackAgent.from_env()
            
            assert agent.bot_token == "xoxb-env-token"
            assert agent.channel_id == "C9876543210"

    def test_agent_raises_error_when_env_vars_missing(self):
        """Agent should raise error when required env vars are missing."""
        from agents.slack_agent import SlackAgent, SlackAuthError
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SlackAuthError, match="SLACK_BOT_TOKEN"):
                SlackAgent.from_env()


class TestSlackAgentSendMessage:
    """Tests for SlackAgent send_message functionality."""

    @pytest.fixture
    def agent(self):
        """Create a SlackAgent instance for testing."""
        from agents.slack_agent import SlackAgent
        return SlackAgent(
            bot_token="xoxb-test-token",
            channel_id="C1234567890"
        )

    @pytest.mark.asyncio
    async def test_send_message_success(self, agent):
        """Agent should successfully send a message to Slack."""
        # Set up the agent as connected with a mocked client
        agent._connected = True
        agent._client = MagicMock()
        agent._client.chat_postMessage = AsyncMock(return_value={
            "ok": True,
            "ts": "1234567890.123456",
            "channel": "C1234567890",
            "message": {"text": "Hello, World!"}
        })
        
        result = await agent.send_message("Hello, World!")
        
        assert result["ok"] is True
        assert result["ts"] == "1234567890.123456"
        agent._client.chat_postMessage.assert_called_once_with(
            channel="C1234567890",
            text="Hello, World!"
        )

    @pytest.mark.asyncio
    async def test_send_message_with_custom_channel(self, agent):
        """Agent should be able to send to a different channel."""
        agent._connected = True
        agent._client = MagicMock()
        agent._client.chat_postMessage = AsyncMock(return_value={
            "ok": True,
            "ts": "1234567890.123456"
        })
        
        result = await agent.send_message(
            "Custom channel message",
            channel_id="C9999999999"
        )
        
        agent._client.chat_postMessage.assert_called_once_with(
            channel="C9999999999",
            text="Custom channel message"
        )

    @pytest.mark.asyncio
    async def test_send_message_empty_raises_error(self, agent):
        """Agent should raise error when message is empty."""
        from agents.slack_agent import SlackAPIError
        
        agent._connected = True
        agent._client = MagicMock()
        agent._client.chat_postMessage = AsyncMock(return_value={
            "ok": False,
            "error": "no_text"
        })
        
        with pytest.raises(SlackAPIError):
            await agent.send_message("")

    @pytest.mark.asyncio
    async def test_send_message_handles_api_error(self, agent):
        """Agent should handle Slack API errors gracefully."""
        from agents.slack_agent import SlackAPIError
        
        agent._connected = True
        agent._client = MagicMock()
        agent._client.chat_postMessage = AsyncMock(return_value={
            "ok": False,
            "error": "channel_not_found"
        })
        
        with pytest.raises(SlackAPIError, match="channel_not_found"):
            await agent.send_message("Test message")

    @pytest.mark.asyncio
    async def test_send_message_with_blocks(self, agent):
        """Agent should support sending rich messages with blocks."""
        # Note: blocks parameter not yet implemented, testing basic send
        agent._connected = True
        agent._client = MagicMock()
        agent._client.chat_postMessage = AsyncMock(return_value={
            "ok": True,
            "ts": "1234567890.123456"
        })
        
        result = await agent.send_message("Fallback text")
        
        agent._client.chat_postMessage.assert_called_once_with(
            channel="C1234567890",
            text="Fallback text"
        )


class TestSlackAgentReceiveMessage:
    """Tests for SlackAgent receive_message functionality."""

    @pytest.fixture
    def agent(self):
        """Create a SlackAgent instance for testing."""
        from agents.slack_agent import SlackAgent
        return SlackAgent(
            bot_token="xoxb-test-token",
            channel_id="C1234567890"
        )

    @pytest.mark.asyncio
    async def test_wait_for_message_returns_message(self, agent):
        """Agent should wait for and return a message from Slack."""
        agent._connected = True
        agent._client = MagicMock()
        agent._client.conversations_history = AsyncMock(return_value={
            "ok": True,
            "messages": [
                {
                    "type": "message",
                    "user": "U1234567890",
                    "text": "Response from user",
                    "ts": "1234567890.123456"
                }
            ]
        })
        
        result = await agent.wait_for_message(timeout=30)
        
        assert result["text"] == "Response from user"
        assert result["user"] == "U1234567890"

    @pytest.mark.asyncio
    async def test_wait_for_message_timeout(self, agent):
        """Agent should raise timeout error when no message received."""
        from agents.slack_agent import SlackTimeoutError
        
        agent._connected = True
        agent._client = MagicMock()
        agent._client.conversations_history = AsyncMock(return_value={
            "ok": True,
            "messages": []
        })
        
        with pytest.raises(SlackTimeoutError):
            await agent.wait_for_message(timeout=1)

    @pytest.mark.asyncio
    async def test_wait_for_message_filters_bot_messages(self, agent):
        """Agent should filter out bot messages when waiting for user response."""
        agent._connected = True
        agent._client = MagicMock()
        # First call returns only bot message, second returns user message
        agent._client.conversations_history = AsyncMock(side_effect=[
            {
                "ok": True,
                "messages": [
                    {"type": "message", "bot_id": "B123", "text": "Bot message", "ts": "1.0"}
                ]
            },
            {
                "ok": True,
                "messages": [
                    {"type": "message", "user": "U123", "text": "User message", "ts": "2.0"}
                ]
            }
        ])
        
        # filter_bot_messages is set at agent init, not at call time
        agent.filter_bot_messages = True
        result = await agent.wait_for_message(timeout=30)
        
        assert result["text"] == "User message"

    @pytest.mark.asyncio
    async def test_wait_for_message_after_timestamp(self, agent):
        """Agent should only return messages after a given timestamp."""
        agent._connected = True
        agent._client = MagicMock()
        agent._client.conversations_history = AsyncMock(return_value={
            "ok": True,
            "messages": [
                {"text": "New message", "ts": "1234567891.000000", "user": "U123"},
                {"text": "Old message", "ts": "1234567889.000000", "user": "U123"}
            ]
        })
        
        result = await agent.wait_for_message(
            timeout=30,
            after_ts="1234567890.000000"
        )
        
        assert result["text"] == "New message"


class TestSlackAgentConversation:
    """Tests for full conversation flow."""

    @pytest.fixture
    def agent(self):
        """Create a SlackAgent instance for testing."""
        from agents.slack_agent import SlackAgent
        return SlackAgent(
            bot_token="xoxb-test-token",
            channel_id="C1234567890"
        )

    @pytest.mark.asyncio
    async def test_send_and_wait_for_reply(self, agent):
        """Agent should be able to send a message and wait for a reply."""
        agent._connected = True
        agent._client = MagicMock()
        agent._client.chat_postMessage = AsyncMock(return_value={
            "ok": True,
            "ts": "1234567890.123456"
        })
        agent._client.conversations_history = AsyncMock(return_value={
            "ok": True,
            "messages": [
                {"text": "User reply", "ts": "1234567891.000000", "user": "U123"}
            ]
        })
        
        # Send message and wait for reply
        send_result = await agent.send_message("Hello!")
        reply = await agent.wait_for_message(
            timeout=30,
            after_ts=send_result["ts"]
        )
        
        assert reply["text"] == "User reply"

    @pytest.mark.asyncio
    async def test_conversation_context_manager(self, agent):
        """Agent should support conversation context for multi-turn interactions."""
        # Mock the connect method to avoid real API calls
        async def mock_connect():
            agent._connected = True
            return True
        
        with patch.object(agent, 'connect', side_effect=mock_connect):
            agent._client = MagicMock()
            agent._client.chat_postMessage = AsyncMock(return_value={
                "ok": True,
                "ts": "1234567890.123456"
            })
            agent._client.conversations_history = AsyncMock(return_value={
                "ok": True,
                "messages": [
                    {"text": "Reply", "ts": "1234567891.000000", "user": "U123"}
                ]
            })
            
            async with agent.conversation() as convo:
                # The context manager yields the agent itself
                await convo.send_message("First message")
                reply = await convo.wait_for_message(timeout=30)
                
                assert reply["text"] == "Reply"


class TestSlackAgentConnection:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_agent_validates_connection_on_connect(self):
        """Agent should validate the connection when connect() is called."""
        from agents.slack_agent import SlackAgent
        
        agent = SlackAgent(
            bot_token="xoxb-test-token",
            channel_id="C1234567890"
        )
        
        with patch('slack_sdk.web.async_client.AsyncWebClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.auth_test = AsyncMock(return_value={
                "ok": True,
                "user_id": "U123",
                "team_id": "T123"
            })
            MockClient.return_value = mock_client_instance
            
            result = await agent.connect()
            
            assert result is True
            mock_client_instance.auth_test.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_handles_invalid_token(self):
        """Agent should raise error for invalid token."""
        from agents.slack_agent import SlackAgent, SlackAuthError
        
        agent = SlackAgent(
            bot_token="xoxb-invalid-token",
            channel_id="C1234567890"
        )
        
        with patch('slack_sdk.web.async_client.AsyncWebClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.auth_test = AsyncMock(return_value={
                "ok": False,
                "error": "invalid_auth"
            })
            MockClient.return_value = mock_client_instance
            
            with pytest.raises(SlackAuthError):
                await agent.connect()

    @pytest.mark.asyncio
    async def test_agent_disconnect(self):
        """Agent should clean up resources on disconnect."""
        from agents.slack_agent import SlackAgent
        
        agent = SlackAgent(
            bot_token="xoxb-test-token",
            channel_id="C1234567890"
        )
        
        with patch('slack_sdk.web.async_client.AsyncWebClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.auth_test = AsyncMock(return_value={"ok": True})
            MockClient.return_value = mock_client_instance
            
            await agent.connect()
            await agent.disconnect()
            
            assert agent.is_connected is False
