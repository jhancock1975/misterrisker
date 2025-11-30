"""
Slack Agent for LangGraph workflow.

This module provides a SlackAgent class for sending and receiving messages
via Slack, designed for use within LangGraph workflows.
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager


# Custom Exceptions
class SlackAPIError(Exception):
    """Raised when Slack API returns an error."""
    pass


class SlackTimeoutError(Exception):
    """Raised when waiting for a Slack message times out."""
    pass


class SlackAuthError(Exception):
    """Raised when Slack authentication fails."""
    pass


class SlackAgent:
    """
    Agent for interacting with Slack channels.
    
    This agent provides methods to send messages to Slack and wait for
    responses, designed for integration with LangGraph workflows.
    
    Attributes:
        bot_token: Slack bot OAuth token
        channel_id: Default channel ID for messaging
        timeout: Default timeout for waiting for messages
        filter_bot_messages: Whether to ignore messages from bots
        _client: Slack WebClient instance
        _connected: Connection state flag
    """
    
    def __init__(
        self,
        bot_token: str,
        channel_id: str,
        timeout: int = 30,
        filter_bot_messages: bool = True
    ):
        """
        Initialize the Slack Agent.
        
        Args:
            bot_token: Slack bot OAuth token (xoxb-...)
            channel_id: Default Slack channel ID
            timeout: Default timeout in seconds for waiting for messages
            filter_bot_messages: If True, ignore messages from other bots
            
        Raises:
            SlackAuthError: If bot_token is empty or invalid format
        """
        if not bot_token:
            raise SlackAuthError("Bot token is required")
        
        if not bot_token.startswith("xoxb-"):
            raise SlackAuthError("Invalid bot token format. Expected 'xoxb-...'")
        
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.timeout = timeout
        self.filter_bot_messages = filter_bot_messages
        self._client = None
        self._connected = False
        self._last_message_ts: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "SlackAgent":
        """
        Create a SlackAgent from environment variables.
        
        Expected environment variables:
            SLACK_BOT_TOKEN: Slack bot OAuth token
            SLACK_CHANNEL_ID: Default channel ID
            SLACK_TIMEOUT: (optional) Timeout in seconds
            
        Returns:
            SlackAgent: Configured agent instance
            
        Raises:
            SlackAuthError: If required environment variables are missing
        """
        bot_token = os.environ.get("SLACK_BOT_TOKEN")
        channel_id = os.environ.get("SLACK_CHANNEL_ID")
        timeout = int(os.environ.get("SLACK_TIMEOUT", "30"))
        
        if not bot_token:
            raise SlackAuthError("SLACK_BOT_TOKEN environment variable is required")
        if not channel_id:
            raise SlackAuthError("SLACK_CHANNEL_ID environment variable is required")
        
        return cls(
            bot_token=bot_token,
            channel_id=channel_id,
            timeout=timeout
        )
    
    @property
    def is_connected(self) -> bool:
        """Check if the agent is connected to Slack."""
        return self._connected
    
    async def connect(self) -> bool:
        """
        Establish connection to Slack.
        
        Returns:
            bool: True if connection successful
            
        Raises:
            SlackAuthError: If authentication fails
        """
        try:
            # Import here to allow mocking in tests
            from slack_sdk.web.async_client import AsyncWebClient
            
            self._client = AsyncWebClient(token=self.bot_token)
            
            # Test the connection with an auth.test call
            response = await self._client.auth_test()
            
            if not response.get("ok"):
                raise SlackAuthError(f"Auth test failed: {response.get('error', 'Unknown error')}")
            
            self._connected = True
            return True
            
        except Exception as e:
            if "invalid_auth" in str(e).lower():
                raise SlackAuthError(f"Invalid authentication: {e}")
            raise SlackAPIError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Slack and cleanup resources."""
        self._connected = False
        self._client = None
    
    async def send_message(
        self,
        message: str,
        channel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message to a Slack channel.
        
        Args:
            message: The message text to send
            channel_id: Optional channel ID (uses default if not provided)
            
        Returns:
            dict: Slack API response with 'ok', 'ts', and other fields
            
        Raises:
            SlackAPIError: If the message fails to send
            SlackAuthError: If not connected
        """
        if not self._connected or not self._client:
            raise SlackAuthError("Not connected to Slack. Call connect() first.")
        
        target_channel = channel_id or self.channel_id
        
        try:
            response = await self._client.chat_postMessage(
                channel=target_channel,
                text=message
            )
            
            if not response.get("ok"):
                raise SlackAPIError(f"Failed to send message: {response.get('error', 'Unknown error')}")
            
            self._last_message_ts = response.get("ts")
            
            return {
                "ok": True,
                "ts": response.get("ts"),
                "channel": response.get("channel"),
                "message": response.get("message", {})
            }
            
        except SlackAPIError:
            raise
        except Exception as e:
            raise SlackAPIError(f"Failed to send message: {e}")
    
    async def wait_for_message(
        self,
        timeout: Optional[int] = None,
        after_ts: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Wait for a new message in the channel.
        
        Args:
            timeout: Timeout in seconds (uses default if not provided)
            after_ts: Only consider messages after this timestamp
            
        Returns:
            dict: Message data with 'text', 'user', 'ts' fields
            
        Raises:
            SlackTimeoutError: If no message received within timeout
            SlackAPIError: If API call fails
        """
        if not self._connected or not self._client:
            raise SlackAuthError("Not connected to Slack. Call connect() first.")
        
        effective_timeout = timeout or self.timeout
        after_timestamp = after_ts or self._last_message_ts or "0"
        
        start_time = asyncio.get_event_loop().time()
        poll_interval = 1.0  # Poll every second
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= effective_timeout:
                raise SlackTimeoutError(
                    f"No message received within {effective_timeout} seconds"
                )
            
            try:
                # Fetch recent messages from the channel
                response = await self._client.conversations_history(
                    channel=self.channel_id,
                    oldest=after_timestamp,
                    limit=10
                )
                
                if not response.get("ok"):
                    raise SlackAPIError(
                        f"Failed to fetch messages: {response.get('error', 'Unknown error')}"
                    )
                
                messages = response.get("messages", [])
                
                # Filter and find new messages
                for msg in messages:
                    # Skip if this is our own message
                    if msg.get("ts") == self._last_message_ts:
                        continue
                    
                    # Skip bot messages if filtering is enabled
                    if self.filter_bot_messages and msg.get("bot_id"):
                        continue
                    
                    # Skip messages older than our reference
                    if float(msg.get("ts", 0)) <= float(after_timestamp):
                        continue
                    
                    # Found a valid message
                    return {
                        "text": msg.get("text", ""),
                        "user": msg.get("user", ""),
                        "ts": msg.get("ts", ""),
                        "channel": self.channel_id
                    }
                
            except (SlackTimeoutError, SlackAPIError):
                raise
            except Exception as e:
                raise SlackAPIError(f"Error polling for messages: {e}")
            
            # Wait before polling again
            await asyncio.sleep(poll_interval)
    
    @asynccontextmanager
    async def conversation(self):
        """
        Context manager for a conversation session.
        
        Automatically connects on entry and disconnects on exit.
        
        Usage:
            async with agent.conversation():
                await agent.send_message("Hello!")
                response = await agent.wait_for_message()
        
        Yields:
            SlackAgent: The connected agent instance
        """
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()
