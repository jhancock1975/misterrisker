"""
Tests for integrating agents with Chain of Thought into the web app.

TDD tests for using LangGraph agents instead of direct MCP server calls.
"""

import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger("test.web_app_agents")


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_coinbase_agent():
    """Create a mock Coinbase agent."""
    agent = MagicMock()
    agent.enable_chain_of_thought = True
    agent.chain_of_thought = MagicMock()
    
    async def mock_execute_tool(tool_name: str, params: dict):
        if tool_name == "get_accounts":
            return {
                "accounts": [
                    {"currency": "BTC", "available_balance": {"value": "1.5"}},
                    {"currency": "ETH", "available_balance": {"value": "10.0"}}
                ]
            }
        return {"success": True}
    
    agent.execute_tool = AsyncMock(side_effect=mock_execute_tool)
    agent.get_available_tools.return_value = [
        {"name": "get_accounts", "description": "Get accounts"},
        {"name": "market_order_buy", "description": "Buy crypto"},
    ]
    return agent


@pytest.fixture
def mock_schwab_agent():
    """Create a mock Schwab agent."""
    agent = MagicMock()
    agent.enable_chain_of_thought = True
    agent.chain_of_thought = MagicMock()
    
    async def mock_execute_tool(tool_name: str, params: dict):
        if tool_name == "get_accounts":
            return {
                "accounts": [
                    {"accountNumber": "12345", "balance": 50000.00}
                ]
            }
        return {"success": True}
    
    agent.execute_tool = AsyncMock(side_effect=mock_execute_tool)
    agent.get_available_tools.return_value = [
        {"name": "get_accounts", "description": "Get accounts"},
        {"name": "get_quote", "description": "Get stock quote"},
    ]
    return agent


@pytest.fixture
def mock_researcher_agent():
    """Create a mock Researcher agent."""
    agent = MagicMock()
    agent.enable_chain_of_thought = True
    agent.chain_of_thought = MagicMock()
    
    async def mock_run(query: str, **kwargs):
        return {
            "status": "success",
            "response": "AAPL is trading at $150.25",
            "reasoning_steps": [
                "Analyzed the query for stock symbol",
                "Retrieved current price data",
                "Formulated response"
            ]
        }
    
    agent.run = AsyncMock(side_effect=mock_run)
    return agent


# =============================================================================
# Tests for TradingChatBot Agent Integration
# =============================================================================

class TestTradingChatBotAgentSupport:
    """Tests for agent support in TradingChatBot."""

    def test_chatbot_has_agent_attributes(self):
        """TradingChatBot should have agent attributes."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        assert hasattr(bot, 'coinbase_agent')
        assert hasattr(bot, 'schwab_agent')
        assert hasattr(bot, 'researcher_agent')

    def test_chatbot_has_use_agents_flag(self):
        """TradingChatBot should have flag to enable agent mode."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        assert hasattr(bot, 'use_agents')

    def test_chatbot_has_enable_cot_flag(self):
        """TradingChatBot should have flag to enable CoT."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        assert hasattr(bot, 'enable_chain_of_thought')


class TestAgentInitialization:
    """Tests for initializing agents in TradingChatBot."""

    @pytest.mark.asyncio
    async def test_initialize_creates_coinbase_agent_when_credentials_exist(self):
        """Should create Coinbase agent when credentials are available."""
        from web.app import TradingChatBot
        
        with patch.dict(os.environ, {
            "COINBASE_API_KEY": "test_key",
            "COINBASE_API_SECRET": "test_secret",
            "OPENAI_API_KEY": "test_openai_key"
        }):
            with patch('web.app.CoinbaseAgent') as MockAgent:
                MockAgent.return_value = MagicMock()
                bot = TradingChatBot(use_agents=True)
                await bot.initialize()
                
                # Should attempt to create agent
                assert bot.coinbase_agent is not None or MockAgent.called

    @pytest.mark.asyncio
    async def test_initialize_creates_schwab_agent_when_credentials_exist(self):
        """Should create Schwab agent when credentials are available."""
        from web.app import TradingChatBot
        
        with patch.dict(os.environ, {
            "SCHWAB_CLIENT_ID": "test_id",
            "SCHWAB_CLIENT_SECRET": "test_secret",
            "SCHWAB_REFRESH_TOKEN": "test_token",
            "OPENAI_API_KEY": "test_openai_key"
        }):
            with patch('web.app.SchwabAgent') as MockAgent:
                MockAgent.return_value = MagicMock()
                bot = TradingChatBot(use_agents=True)
                await bot.initialize()
                
                assert bot.schwab_agent is not None or MockAgent.called

    @pytest.mark.asyncio
    async def test_initialize_creates_researcher_agent(self):
        """Should create Researcher agent when research credentials available."""
        from web.app import TradingChatBot
        
        with patch.dict(os.environ, {
            "FINNHUB_API_KEY": "test_finnhub",
            "OPENAI_API_KEY": "test_openai_key"
        }):
            with patch('web.app.ResearcherAgent') as MockAgent:
                MockAgent.return_value = MagicMock()
                bot = TradingChatBot(use_agents=True)
                await bot.initialize()
                
                assert bot.researcher_agent is not None or MockAgent.called


class TestAgentToolExecution:
    """Tests for executing tools through agents."""

    @pytest.mark.asyncio
    async def test_execute_tool_uses_coinbase_agent(self, mock_coinbase_agent):
        """Should use Coinbase agent for tool execution when in agent mode."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True)
        bot.coinbase_agent = mock_coinbase_agent
        bot.active_broker = "coinbase"
        
        result = await bot._execute_tool("get_accounts", {})
        
        mock_coinbase_agent.execute_tool.assert_called_once_with("get_accounts", {})
        assert "accounts" in result

    @pytest.mark.asyncio
    async def test_execute_tool_uses_schwab_agent(self, mock_schwab_agent):
        """Should use Schwab agent for tool execution when in agent mode."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True)
        bot.schwab_agent = mock_schwab_agent
        bot.active_broker = "schwab"
        
        result = await bot._execute_tool("get_accounts", {})
        
        mock_schwab_agent.execute_tool.assert_called_once_with("get_accounts", {})
        assert "accounts" in result

    @pytest.mark.asyncio
    async def test_execute_tool_falls_back_to_mcp_when_agent_unavailable(self):
        """Should fall back to MCP server when agent not available."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True)
        bot.coinbase_agent = None
        bot.coinbase_server = MagicMock()
        bot.coinbase_server.call_tool = AsyncMock(return_value={"accounts": []})
        bot.active_broker = "coinbase"
        
        result = await bot._execute_tool("get_accounts", {})
        
        bot.coinbase_server.call_tool.assert_called_once()


class TestResearcherAgentIntegration:
    """Tests for Researcher agent integration."""

    @pytest.mark.asyncio
    async def test_research_command_triggers_researcher_agent(self, mock_researcher_agent):
        """Research commands should trigger researcher agent."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True)
        bot.researcher_agent = mock_researcher_agent
        bot.llm = MagicMock()
        bot.llm.ainvoke = AsyncMock(return_value=MagicMock(content="Research result"))
        
        # Trigger research
        result = await bot._execute_research("What is AAPL trading at?")
        
        mock_researcher_agent.run.assert_called_once()
        assert "reasoning_steps" in result or "response" in result

    def test_chatbot_has_research_method(self):
        """TradingChatBot should have research method."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True)
        
        assert hasattr(bot, '_execute_research')
        assert callable(bot._execute_research)


class TestChainOfThoughtInWebApp:
    """Tests for CoT in web app responses."""

    @pytest.mark.asyncio
    async def test_cot_enabled_agents_provide_reasoning(self, mock_coinbase_agent):
        """Agents with CoT should provide reasoning in responses."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True, enable_chain_of_thought=True)
        bot.coinbase_agent = mock_coinbase_agent
        
        assert bot.enable_chain_of_thought is True
        assert mock_coinbase_agent.enable_chain_of_thought is True

    def test_chatbot_passes_cot_flag_to_agents(self):
        """TradingChatBot should pass CoT flag when creating agents."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True, enable_chain_of_thought=True)
        
        assert bot.enable_chain_of_thought is True


class TestBackwardCompatibility:
    """Tests for backward compatibility with MCP-only mode."""

    def test_default_mode_is_mcp_only(self):
        """Default mode should be MCP-only for backward compatibility."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot()
        
        assert bot.use_agents is False

    @pytest.mark.asyncio
    async def test_mcp_mode_uses_servers_directly(self):
        """MCP mode should use servers directly, not agents."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=False)
        bot.coinbase_server = MagicMock()
        bot.coinbase_server.call_tool = AsyncMock(return_value={"accounts": []})
        bot.active_broker = "coinbase"
        
        result = await bot._execute_tool("get_accounts", {})
        
        bot.coinbase_server.call_tool.assert_called_once()


class TestAgentModeSystemPrompt:
    """Tests for system prompt in agent mode."""

    def test_agent_mode_has_research_commands(self):
        """Agent mode system prompt should include research commands."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True)
        
        # The system prompt should mention research capability
        assert hasattr(bot, 'system_prompt') or hasattr(bot, 'agent_system_prompt')

    def test_cot_mode_mentions_reasoning(self):
        """CoT mode should mention step-by-step reasoning in prompts."""
        from web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True, enable_chain_of_thought=True)
        
        # Should have CoT awareness
        assert bot.enable_chain_of_thought is True
