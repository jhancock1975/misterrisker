"""
Tests for the LangGraph Slack workflow.

These tests follow TDD principles - written before the implementation.
They cover:
- Workflow state management
- Graph node execution
- End-to-end conversation flow
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import TypedDict, Annotated, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSlackWorkflowState:
    """Tests for Slack workflow state management."""

    def test_state_initializes_with_required_fields(self):
        """Workflow state should have required fields."""
        from workflows.slack_workflow import SlackState
        
        state = SlackState(
            user_message="Hello",
            messages=[],
            awaiting_response=False
        )
        
        assert state["user_message"] == "Hello"
        assert state["messages"] == []
        assert state["awaiting_response"] is False

    def test_state_tracks_conversation_history(self):
        """State should maintain conversation history."""
        from workflows.slack_workflow import SlackState
        
        state = SlackState(
            user_message="Hello",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            awaiting_response=False
        )
        
        assert len(state["messages"]) == 2
        assert state["messages"][0]["role"] == "user"
        assert state["messages"][1]["role"] == "assistant"

    def test_state_includes_slack_metadata(self):
        """State should include Slack-specific metadata."""
        from workflows.slack_workflow import SlackState
        
        state = SlackState(
            user_message="Test",
            messages=[],
            awaiting_response=False,
            last_message_ts="1234567890.123456",
            channel_id="C1234567890"
        )
        
        assert state["last_message_ts"] == "1234567890.123456"
        assert state["channel_id"] == "C1234567890"


class TestSlackWorkflowNodes:
    """Tests for individual workflow nodes."""

    @pytest.fixture
    def mock_slack_agent(self):
        """Create a mock SlackAgent."""
        agent = MagicMock()
        agent.send_message = AsyncMock(return_value={
            "ok": True,
            "ts": "1234567890.123456"
        })
        agent.wait_for_message = AsyncMock(return_value={
            "text": "User response",
            "user": "U123",
            "ts": "1234567891.000000"
        })
        return agent

    @pytest.mark.asyncio
    async def test_send_message_node(self, mock_slack_agent):
        """Send message node should send message and update state."""
        from workflows.slack_workflow import SlackWorkflow
        
        workflow = SlackWorkflow(slack_agent=mock_slack_agent)
        
        state = {
            "user_message": "Hello, Slack!",
            "messages": [],
            "awaiting_response": False
        }
        
        new_state = await workflow.send_message_node(state)
        
        mock_slack_agent.send_message.assert_called_once()
        assert new_state["last_message_ts"] == "1234567890.123456"
        assert len(new_state["messages"]) == 1
        assert new_state["messages"][0]["content"] == "Hello, Slack!"
        assert new_state["awaiting_response"] is True

    @pytest.mark.asyncio
    async def test_receive_message_node(self, mock_slack_agent):
        """Receive message node should wait for and capture response."""
        from workflows.slack_workflow import SlackWorkflow
        
        workflow = SlackWorkflow(slack_agent=mock_slack_agent)
        
        state = {
            "user_message": "",
            "messages": [{"role": "assistant", "content": "Hello!"}],
            "awaiting_response": True,
            "last_message_ts": "1234567890.123456"
        }
        
        new_state = await workflow.receive_message_node(state)
        
        mock_slack_agent.wait_for_message.assert_called_once()
        assert new_state["awaiting_response"] is False
        assert len(new_state["messages"]) == 2
        assert new_state["messages"][1]["content"] == "User response"
        assert new_state["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_receive_message_node_with_timeout(self, mock_slack_agent):
        """Receive message node should handle timeout gracefully."""
        from workflows.slack_workflow import SlackWorkflow
        from agents.slack_agent import SlackTimeoutError
        
        mock_slack_agent.wait_for_message = AsyncMock(
            side_effect=SlackTimeoutError("Timeout waiting for message")
        )
        
        workflow = SlackWorkflow(
            slack_agent=mock_slack_agent,
            receive_timeout=30
        )
        
        state = {
            "user_message": "",
            "messages": [],
            "awaiting_response": True,
            "last_message_ts": "1234567890.123456"
        }
        
        new_state = await workflow.receive_message_node(state)
        
        assert new_state["timeout_occurred"] is True
        assert new_state["awaiting_response"] is False


class TestSlackWorkflowGraph:
    """Tests for the complete workflow graph."""

    @pytest.fixture
    def mock_slack_agent(self):
        """Create a mock SlackAgent."""
        agent = MagicMock()
        agent.send_message = AsyncMock(return_value={
            "ok": True,
            "ts": "1234567890.123456"
        })
        agent.wait_for_message = AsyncMock(return_value={
            "text": "User response",
            "user": "U123",
            "ts": "1234567891.000000"
        })
        agent.connect = AsyncMock(return_value=True)
        agent.disconnect = AsyncMock()
        return agent

    def test_workflow_creates_valid_graph(self, mock_slack_agent):
        """Workflow should create a valid LangGraph graph."""
        from workflows.slack_workflow import SlackWorkflow
        
        workflow = SlackWorkflow(slack_agent=mock_slack_agent)
        graph = workflow.build_graph()
        
        assert graph is not None
        # Graph should have the expected nodes
        assert "send_message" in workflow.node_names
        assert "receive_message" in workflow.node_names

    @pytest.mark.asyncio
    async def test_workflow_executes_send_receive_flow(self, mock_slack_agent):
        """Workflow should execute send -> receive flow."""
        from workflows.slack_workflow import SlackWorkflow
        
        workflow = SlackWorkflow(slack_agent=mock_slack_agent)
        
        initial_state = {
            "user_message": "Hello from LangGraph!",
            "messages": [],
            "awaiting_response": False
        }
        
        final_state = await workflow.run(initial_state)
        
        # Verify send was called
        mock_slack_agent.send_message.assert_called_once()
        
        # Verify receive was called
        mock_slack_agent.wait_for_message.assert_called_once()
        
        # Verify final state
        assert len(final_state["messages"]) == 2  # sent + received
        assert final_state["awaiting_response"] is False

    @pytest.mark.asyncio
    async def test_workflow_handles_send_only_mode(self, mock_slack_agent):
        """Workflow should support send-only mode without waiting."""
        from workflows.slack_workflow import SlackWorkflow
        
        workflow = SlackWorkflow(
            slack_agent=mock_slack_agent,
            wait_for_response=False
        )
        
        initial_state = {
            "user_message": "Fire and forget message",
            "messages": [],
            "awaiting_response": False
        }
        
        final_state = await workflow.run(initial_state)
        
        mock_slack_agent.send_message.assert_called_once()
        mock_slack_agent.wait_for_message.assert_not_called()
        assert len(final_state["messages"]) == 1

    @pytest.mark.asyncio
    async def test_workflow_conditional_routing(self, mock_slack_agent):
        """Workflow should route based on state conditions."""
        from workflows.slack_workflow import SlackWorkflow
        
        workflow = SlackWorkflow(slack_agent=mock_slack_agent)
        
        # When user_message is empty, should not send
        initial_state = {
            "user_message": "",
            "messages": [],
            "awaiting_response": False
        }
        
        final_state = await workflow.run(initial_state)
        
        mock_slack_agent.send_message.assert_not_called()


class TestSlackWorkflowIntegration:
    """Integration tests for the complete workflow."""

    @pytest.fixture
    def mock_slack_agent(self):
        """Create a mock SlackAgent with full conversation simulation."""
        agent = MagicMock()
        agent.connect = AsyncMock(return_value=True)
        agent.disconnect = AsyncMock()
        
        # Simulate a multi-turn conversation
        agent.send_message = AsyncMock(side_effect=[
            {"ok": True, "ts": "1000.000"},
            {"ok": True, "ts": "2000.000"},
        ])
        agent.wait_for_message = AsyncMock(side_effect=[
            {"text": "First reply", "user": "U123", "ts": "1500.000"},
            {"text": "Second reply", "user": "U123", "ts": "2500.000"},
        ])
        
        return agent

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, mock_slack_agent):
        """Workflow should support multi-turn conversations."""
        from workflows.slack_workflow import SlackWorkflow
        
        workflow = SlackWorkflow(slack_agent=mock_slack_agent)
        
        # First turn
        state = await workflow.run({
            "user_message": "Hello!",
            "messages": [],
            "awaiting_response": False
        })
        
        assert len(state["messages"]) == 2
        
        # Second turn - continue the conversation
        state["user_message"] = "How are you?"
        state = await workflow.run(state)
        
        assert len(state["messages"]) == 4
        assert mock_slack_agent.send_message.call_count == 2
        assert mock_slack_agent.wait_for_message.call_count == 2


class TestSlackWorkflowConfiguration:
    """Tests for workflow configuration options."""

    def test_workflow_accepts_custom_timeout(self):
        """Workflow should accept custom timeout configuration."""
        from workflows.slack_workflow import SlackWorkflow
        from agents.slack_agent import SlackAgent
        
        agent = MagicMock(spec=SlackAgent)
        workflow = SlackWorkflow(
            slack_agent=agent,
            receive_timeout=60
        )
        
        assert workflow.receive_timeout == 60

    def test_workflow_accepts_custom_channel(self):
        """Workflow should accept custom channel override."""
        from workflows.slack_workflow import SlackWorkflow
        from agents.slack_agent import SlackAgent
        
        agent = MagicMock(spec=SlackAgent)
        workflow = SlackWorkflow(
            slack_agent=agent,
            channel_override="C9999999999"
        )
        
        assert workflow.channel_override == "C9999999999"

    def test_workflow_configurable_filter_bots(self):
        """Workflow should allow configuring bot message filtering."""
        from workflows.slack_workflow import SlackWorkflow
        from agents.slack_agent import SlackAgent
        
        agent = MagicMock(spec=SlackAgent)
        workflow = SlackWorkflow(
            slack_agent=agent,
            filter_bot_messages=False
        )
        
        assert workflow.filter_bot_messages is False
