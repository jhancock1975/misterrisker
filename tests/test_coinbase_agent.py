"""
Tests for the Coinbase LangGraph Agent.

These tests follow TDD principles - written before the implementation.
They cover the agent that uses Coinbase MCP tools.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

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
    
    # Mock tool responses
    async def mock_call_tool(tool_name: str, params: dict):
        responses = {
            "get_accounts": {
                "accounts": [
                    {"uuid": "acc-123", "currency": "BTC", "available_balance": {"value": "1.5"}}
                ]
            },
            "get_products": {
                "products": [
                    {"product_id": "BTC-USD", "price": "45000.00"}
                ]
            },
            "get_product": {
                "product_id": "BTC-USD",
                "price": "45000.00",
                "base_currency_id": "BTC"
            },
            "get_best_bid_ask": {
                "pricebooks": [
                    {"product_id": "BTC-USD", "bids": [{"price": "44999"}], "asks": [{"price": "45001"}]}
                ]
            },
            "market_order_buy": {
                "success": True,
                "order_id": "order-123"
            },
            "market_order_sell": {
                "success": True,
                "order_id": "order-124"
            },
            "get_order": {
                "order": {"order_id": "order-123", "status": "FILLED"}
            },
            "list_orders": {
                "orders": [{"order_id": "order-123", "status": "OPEN"}]
            },
            "cancel_orders": {
                "results": [{"success": True, "order_id": "order-123"}]
            },
            "get_portfolios": {
                "portfolios": [{"uuid": "port-123", "name": "Default"}]
            },
            "get_portfolio_breakdown": {
                "breakdown": {"total_balance": {"value": "10000.00"}}
            },
            "get_candles": {
                "candles": [{"open": "44800", "close": "45200", "high": "45500", "low": "44500"}]
            },
            "get_transaction_summary": {
                "fee_tier": {"maker_fee_rate": "0.004", "taker_fee_rate": "0.006"}
            }
        }
        return responses.get(tool_name, {})
    
    server.call_tool = AsyncMock(side_effect=mock_call_tool)
    server.list_tools.return_value = [
        {"name": "get_accounts", "description": "Get accounts"},
        {"name": "get_products", "description": "Get products"},
        {"name": "get_product", "description": "Get product"},
        {"name": "get_best_bid_ask", "description": "Get best bid/ask"},
        {"name": "market_order_buy", "description": "Market buy"},
        {"name": "market_order_sell", "description": "Market sell"},
        {"name": "get_order", "description": "Get order"},
        {"name": "list_orders", "description": "List orders"},
        {"name": "cancel_orders", "description": "Cancel orders"},
        {"name": "get_portfolios", "description": "Get portfolios"},
        {"name": "get_portfolio_breakdown", "description": "Get portfolio breakdown"},
        {"name": "get_candles", "description": "Get candles"},
        {"name": "get_transaction_summary", "description": "Get fees"},
    ]
    
    return server


@pytest.fixture
def coinbase_agent(mock_mcp_server):
    """Create a Coinbase agent with mocked MCP server."""
    from agents.coinbase_agent import CoinbaseAgent
    
    agent = CoinbaseAgent(mcp_server=mock_mcp_server)
    return agent


# =============================================================================
# Agent Initialization Tests
# =============================================================================

class TestCoinbaseAgentInitialization:
    """Tests for agent initialization."""

    def test_agent_initializes_with_mcp_server(self, mock_mcp_server):
        """Agent should initialize with an MCP server."""
        from agents.coinbase_agent import CoinbaseAgent
        
        agent = CoinbaseAgent(mcp_server=mock_mcp_server)
        
        assert agent is not None
        assert agent.mcp_server == mock_mcp_server

    def test_agent_discovers_available_tools(self, coinbase_agent, mock_mcp_server):
        """Agent should discover tools from MCP server."""
        tools = coinbase_agent.get_available_tools()
        
        assert len(tools) > 0
        mock_mcp_server.list_tools.assert_called()

    def test_agent_creates_langgraph_workflow(self, coinbase_agent):
        """Agent should create a LangGraph workflow."""
        workflow = coinbase_agent.get_workflow()
        
        assert workflow is not None


# =============================================================================
# Agent State Tests
# =============================================================================

class TestCoinbaseAgentState:
    """Tests for agent state management."""

    def test_state_tracks_conversation(self):
        """Agent state should track conversation history."""
        from agents.coinbase_agent import CoinbaseAgentState
        
        state = CoinbaseAgentState(
            messages=[],
            tool_calls=[],
            tool_results=[],
            current_task=""
        )
        
        assert state["messages"] == []
        assert state["tool_calls"] == []

    def test_state_tracks_portfolio_context(self):
        """Agent state should track portfolio context."""
        from agents.coinbase_agent import CoinbaseAgentState
        
        state = CoinbaseAgentState(
            messages=[],
            tool_calls=[],
            tool_results=[],
            current_task="",
            portfolio_balance=10000.0,
            positions={"BTC": 1.5, "ETH": 10.0}
        )
        
        assert state["portfolio_balance"] == 10000.0
        assert state["positions"]["BTC"] == 1.5


# =============================================================================
# Tool Execution Tests
# =============================================================================

class TestCoinbaseAgentToolExecution:
    """Tests for agent tool execution."""

    @pytest.mark.asyncio
    async def test_agent_executes_get_accounts(self, coinbase_agent, mock_mcp_server):
        """Agent should execute get_accounts tool."""
        result = await coinbase_agent.execute_tool("get_accounts", {})
        
        mock_mcp_server.call_tool.assert_called_with("get_accounts", {})
        assert "accounts" in result

    @pytest.mark.asyncio
    async def test_agent_executes_get_price(self, coinbase_agent, mock_mcp_server):
        """Agent should get current price for a product."""
        result = await coinbase_agent.execute_tool("get_product", {"product_id": "BTC-USD"})
        
        assert result["product_id"] == "BTC-USD"
        assert "price" in result

    @pytest.mark.asyncio
    async def test_agent_executes_market_buy(self, coinbase_agent, mock_mcp_server):
        """Agent should execute market buy order."""
        result = await coinbase_agent.execute_tool("market_order_buy", {
            "product_id": "BTC-USD",
            "quote_size": "100.00"
        })
        
        assert result["success"] is True
        assert "order_id" in result

    @pytest.mark.asyncio
    async def test_agent_executes_market_sell(self, coinbase_agent, mock_mcp_server):
        """Agent should execute market sell order."""
        result = await coinbase_agent.execute_tool("market_order_sell", {
            "product_id": "BTC-USD",
            "base_size": "0.01"
        })
        
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_agent_checks_order_status(self, coinbase_agent, mock_mcp_server):
        """Agent should check order status."""
        result = await coinbase_agent.execute_tool("get_order", {
            "order_id": "order-123"
        })
        
        assert result["order"]["status"] == "FILLED"

    @pytest.mark.asyncio
    async def test_agent_cancels_order(self, coinbase_agent, mock_mcp_server):
        """Agent should cancel orders."""
        result = await coinbase_agent.execute_tool("cancel_orders", {
            "order_ids": ["order-123"]
        })
        
        assert result["results"][0]["success"] is True


# =============================================================================
# Workflow Tests
# =============================================================================

class TestCoinbaseAgentWorkflow:
    """Tests for agent LangGraph workflow."""

    @pytest.mark.asyncio
    async def test_workflow_processes_query(self, coinbase_agent):
        """Workflow should process user queries."""
        from agents.coinbase_agent import CoinbaseAgentState
        
        initial_state = CoinbaseAgentState(
            messages=[{"role": "user", "content": "What is my BTC balance?"}],
            tool_calls=[],
            tool_results=[],
            current_task="check_balance"
        )
        
        result = await coinbase_agent.run(initial_state)
        
        assert result is not None
        assert "messages" in result

    @pytest.mark.asyncio
    async def test_workflow_handles_buy_request(self, coinbase_agent):
        """Workflow should handle buy requests."""
        from agents.coinbase_agent import CoinbaseAgentState
        
        initial_state = CoinbaseAgentState(
            messages=[{"role": "user", "content": "Buy $100 worth of BTC"}],
            tool_calls=[],
            tool_results=[],
            current_task="buy_crypto"
        )
        
        result = await coinbase_agent.run(initial_state)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_workflow_handles_price_check(self, coinbase_agent):
        """Workflow should handle price check requests."""
        from agents.coinbase_agent import CoinbaseAgentState
        
        initial_state = CoinbaseAgentState(
            messages=[{"role": "user", "content": "What is the current price of ETH?"}],
            tool_calls=[],
            tool_results=[],
            current_task="check_price"
        )
        
        result = await coinbase_agent.run(initial_state)
        
        assert result is not None


# =============================================================================
# Agent Capabilities Tests
# =============================================================================

class TestCoinbaseAgentCapabilities:
    """Tests for high-level agent capabilities."""

    @pytest.mark.asyncio
    async def test_agent_can_get_portfolio_summary(self, coinbase_agent, mock_mcp_server):
        """Agent should provide portfolio summary."""
        summary = await coinbase_agent.get_portfolio_summary()
        
        assert summary is not None
        mock_mcp_server.call_tool.assert_called()

    @pytest.mark.asyncio
    async def test_agent_can_get_market_data(self, coinbase_agent, mock_mcp_server):
        """Agent should fetch market data."""
        data = await coinbase_agent.get_market_data("BTC-USD")
        
        assert data is not None

    @pytest.mark.asyncio
    async def test_agent_can_place_trade(self, coinbase_agent, mock_mcp_server):
        """Agent should place trades."""
        result = await coinbase_agent.place_market_buy("BTC-USD", quote_size="100.00")
        
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_agent_can_check_open_orders(self, coinbase_agent, mock_mcp_server):
        """Agent should check open orders."""
        orders = await coinbase_agent.get_open_orders()
        
        assert orders is not None


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestCoinbaseAgentErrorHandling:
    """Tests for agent error handling."""

    @pytest.mark.asyncio
    async def test_agent_handles_tool_error(self, coinbase_agent, mock_mcp_server):
        """Agent should handle tool execution errors."""
        from agents.coinbase_agent import CoinbaseAgentError
        
        mock_mcp_server.call_tool.side_effect = Exception("API Error")
        
        with pytest.raises(CoinbaseAgentError):
            await coinbase_agent.execute_tool("get_accounts", {})

    @pytest.mark.asyncio
    async def test_agent_handles_invalid_tool(self, coinbase_agent):
        """Agent should handle invalid tool requests."""
        from agents.coinbase_agent import CoinbaseAgentError
        
        with pytest.raises(CoinbaseAgentError, match="Unknown tool"):
            await coinbase_agent.execute_tool("invalid_tool", {})

    @pytest.mark.asyncio
    async def test_agent_handles_order_failure(self, coinbase_agent, mock_mcp_server):
        """Agent should handle order failures gracefully."""
        # Override the side_effect to return failure response
        async def mock_order_failure(tool_name: str, params: dict):
            if tool_name == "market_order_buy":
                return {
                    "success": False,
                    "error_response": {"error": "INSUFFICIENT_FUNDS"}
                }
            return {}
        
        mock_mcp_server.call_tool.side_effect = mock_order_failure
        
        result = await coinbase_agent.place_market_buy("BTC-USD", quote_size="1000000.00")
        
        assert result.get("success") is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestCoinbaseAgentIntegration:
    """Integration tests for agent with MCP server."""

    @pytest.mark.asyncio
    async def test_full_trading_flow(self, coinbase_agent, mock_mcp_server):
        """Test complete trading flow: check balance -> get price -> place order -> check order."""
        # Step 1: Check account balance
        accounts = await coinbase_agent.execute_tool("get_accounts", {})
        assert "accounts" in accounts
        
        # Step 2: Get current price
        price = await coinbase_agent.execute_tool("get_product", {"product_id": "BTC-USD"})
        assert "price" in price
        
        # Step 3: Place order
        order = await coinbase_agent.execute_tool("market_order_buy", {
            "product_id": "BTC-USD",
            "quote_size": "100.00"
        })
        assert order["success"] is True
        
        # Step 4: Check order status
        status = await coinbase_agent.execute_tool("get_order", {
            "order_id": order["order_id"]
        })
        assert "order" in status

    @pytest.mark.asyncio
    async def test_portfolio_analysis_flow(self, coinbase_agent, mock_mcp_server):
        """Test portfolio analysis: get portfolios -> get breakdown -> get fees."""
        # Get portfolios
        portfolios = await coinbase_agent.execute_tool("get_portfolios", {})
        assert "portfolios" in portfolios
        
        # Get breakdown
        breakdown = await coinbase_agent.execute_tool("get_portfolio_breakdown", {
            "portfolio_uuid": "port-123"
        })
        assert "breakdown" in breakdown
        
        # Get fee info
        fees = await coinbase_agent.execute_tool("get_transaction_summary", {})
        assert "fee_tier" in fees
