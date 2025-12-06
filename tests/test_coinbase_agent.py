"""
Tests for the Coinbase LangGraph Agent.

These tests follow TDD principles - written before the implementation.
They cover the agent that uses Coinbase MCP tools.
"""

import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger("test.coinbase_agent")


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
            "limit_order_gtc_buy": {
                "success": True,
                "order_id": "order-125"
            },
            "limit_order_gtc_sell": {
                "success": True,
                "order_id": "order-126"
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
        {"name": "limit_order_gtc_buy", "description": "Limit order GTC buy"},
        {"name": "limit_order_gtc_sell", "description": "Limit order GTC sell"},
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

    def test_agent_initializes_with_mcp_server(self, mock_mcp_server, log):
        """Agent should initialize with an MCP server."""
        log.info("Testing agent initialization with MCP server")
        log.info(f"Input params: mock_mcp_server={mock_mcp_server}")
        from agents.coinbase_agent import CoinbaseAgent
        
        agent = CoinbaseAgent(mcp_server=mock_mcp_server)
        
        log.info(f"Result: agent={agent}, agent.mcp_server={agent.mcp_server}")
        assert agent is not None
        assert agent.mcp_server == mock_mcp_server
        log.info("RESULT: Agent initialized successfully with MCP server")

    def test_agent_discovers_available_tools(self, coinbase_agent, mock_mcp_server, log):
        """Agent should discover tools from MCP server."""
        log.info("Testing agent tool discovery")
        log.info(f"Input params: coinbase_agent={coinbase_agent}")
        
        tools = coinbase_agent.get_available_tools()
        
        log.info(f"Result: tools={tools}, count={len(tools)}")
        assert len(tools) > 0
        mock_mcp_server.list_tools.assert_called()
        log.info("RESULT: Agent discovered available tools successfully")

    def test_agent_creates_langgraph_workflow(self, coinbase_agent, log):
        """Agent should create a LangGraph workflow."""
        log.info("Testing LangGraph workflow creation")
        log.info(f"Input params: coinbase_agent={coinbase_agent}")
        
        workflow = coinbase_agent.get_workflow()
        
        log.info(f"Result: workflow={workflow}")
        assert workflow is not None
        log.info("RESULT: LangGraph workflow created successfully")


# =============================================================================
# Agent State Tests
# =============================================================================

class TestCoinbaseAgentState:
    """Tests for agent state management."""

    def test_state_tracks_conversation(self, log):
        """Agent state should track conversation history."""
        log.info("Testing state conversation tracking")
        log.info("Input params: messages=[], tool_calls=[], tool_results=[], current_task=''")
        from agents.coinbase_agent import CoinbaseAgentState
        
        state = CoinbaseAgentState(
            messages=[],
            tool_calls=[],
            tool_results=[],
            current_task=""
        )
        
        log.info(f"Result: state={state}")
        assert state["messages"] == []
        assert state["tool_calls"] == []
        log.info("RESULT: State tracks conversation history correctly")

    def test_state_tracks_portfolio_context(self, log):
        """Agent state should track portfolio context."""
        log.info("Testing state portfolio context tracking")
        log.info("Input params: portfolio_balance=10000.0, positions={'BTC': 1.5, 'ETH': 10.0}")
        from agents.coinbase_agent import CoinbaseAgentState
        
        state = CoinbaseAgentState(
            messages=[],
            tool_calls=[],
            tool_results=[],
            current_task="",
            portfolio_balance=10000.0,
            positions={"BTC": 1.5, "ETH": 10.0}
        )
        
        log.info(f"Result: portfolio_balance={state['portfolio_balance']}, positions={state['positions']}")
        assert state["portfolio_balance"] == 10000.0
        assert state["positions"]["BTC"] == 1.5
        log.info("RESULT: State tracks portfolio context correctly")


# =============================================================================
# Tool Execution Tests
# =============================================================================

class TestCoinbaseAgentToolExecution:
    """Tests for agent tool execution."""

    @pytest.mark.asyncio
    async def test_agent_executes_get_accounts(self, coinbase_agent, mock_mcp_server, log):
        """Agent should execute get_accounts tool."""
        log.info("Testing get_accounts tool execution")
        log.info("Input params: tool_name='get_accounts', params={}")
        
        result = await coinbase_agent.execute_tool("get_accounts", {})
        
        log.info(f"Result: {result}")
        mock_mcp_server.call_tool.assert_called_with("get_accounts", {})
        assert "accounts" in result
        log.info("RESULT: get_accounts executed successfully")

    @pytest.mark.asyncio
    async def test_agent_executes_get_price(self, coinbase_agent, mock_mcp_server, log):
        """Agent should get current price for a product."""
        log.info("Testing get_product tool execution for price")
        log.info("Input params: tool_name='get_product', params={'product_id': 'BTC-USD'}")
        
        result = await coinbase_agent.execute_tool("get_product", {"product_id": "BTC-USD"})
        
        log.info(f"Result: {result}")
        assert result["product_id"] == "BTC-USD"
        assert "price" in result
        log.info("RESULT: get_product executed successfully, price retrieved")

    @pytest.mark.asyncio
    async def test_agent_executes_market_buy(self, coinbase_agent, mock_mcp_server, log):
        """Agent should execute market buy order."""
        log.info("Testing market_order_buy tool execution")
        log.info("Input params: tool_name='market_order_buy', params={'product_id': 'BTC-USD', 'quote_size': '100.00'}")
        
        result = await coinbase_agent.execute_tool("market_order_buy", {
            "product_id": "BTC-USD",
            "quote_size": "100.00"
        })
        
        log.info(f"Result: {result}")
        assert result["success"] is True
        assert "order_id" in result
        log.info("RESULT: market_order_buy executed successfully")

    @pytest.mark.asyncio
    async def test_agent_executes_market_sell(self, coinbase_agent, mock_mcp_server, log):
        """Agent should execute market sell order."""
        log.info("Testing market_order_sell tool execution")
        log.info("Input params: tool_name='market_order_sell', params={'product_id': 'BTC-USD', 'base_size': '0.01'}")
        
        result = await coinbase_agent.execute_tool("market_order_sell", {
            "product_id": "BTC-USD",
            "base_size": "0.01"
        })
        
        log.info(f"Result: {result}")
        assert result["success"] is True
        log.info("RESULT: market_order_sell executed successfully")

    @pytest.mark.asyncio
    async def test_agent_checks_order_status(self, coinbase_agent, mock_mcp_server, log):
        """Agent should check order status."""
        log.info("Testing get_order tool execution for order status")
        log.info("Input params: tool_name='get_order', params={'order_id': 'order-123'}")
        
        result = await coinbase_agent.execute_tool("get_order", {
            "order_id": "order-123"
        })
        
        log.info(f"Result: {result}")
        assert result["order"]["status"] == "FILLED"
        log.info("RESULT: get_order executed successfully, status=FILLED")

    @pytest.mark.asyncio
    async def test_agent_cancels_order(self, coinbase_agent, mock_mcp_server, log):
        """Agent should cancel orders."""
        log.info("Testing cancel_orders tool execution")
        log.info("Input params: tool_name='cancel_orders', params={'order_ids': ['order-123']}")
        
        result = await coinbase_agent.execute_tool("cancel_orders", {
            "order_ids": ["order-123"]
        })
        
        log.info(f"Result: {result}")
        assert result["results"][0]["success"] is True
        log.info("RESULT: cancel_orders executed successfully")


# =============================================================================
# Workflow Tests
# =============================================================================

class TestCoinbaseAgentWorkflow:
    """Tests for agent LangGraph workflow."""

    @pytest.mark.asyncio
    async def test_workflow_processes_query(self, coinbase_agent, log):
        """Workflow should process user queries."""
        log.info("Testing workflow query processing")
        log.info("Input params: message='What is my BTC balance?', current_task='check_balance'")
        from agents.coinbase_agent import CoinbaseAgentState
        
        initial_state = CoinbaseAgentState(
            messages=[{"role": "user", "content": "What is my BTC balance?"}],
            tool_calls=[],
            tool_results=[],
            current_task="check_balance"
        )
        
        result = await coinbase_agent.run(initial_state)
        
        log.info(f"Result: {result}")
        assert result is not None
        assert "messages" in result
        log.info("RESULT: Workflow processed query successfully")

    @pytest.mark.asyncio
    async def test_workflow_handles_buy_request(self, coinbase_agent, log):
        """Workflow should handle buy requests."""
        log.info("Testing workflow buy request handling")
        log.info("Input params: message='Buy $100 worth of BTC', current_task='buy_crypto'")
        from agents.coinbase_agent import CoinbaseAgentState
        
        initial_state = CoinbaseAgentState(
            messages=[{"role": "user", "content": "Buy $100 worth of BTC"}],
            tool_calls=[],
            tool_results=[],
            current_task="buy_crypto"
        )
        
        result = await coinbase_agent.run(initial_state)
        
        log.info(f"Result: {result}")
        assert result is not None
        log.info("RESULT: Workflow handled buy request successfully")

    @pytest.mark.asyncio
    async def test_workflow_handles_price_check(self, coinbase_agent, log):
        """Workflow should handle price check requests via process_query."""
        log.info("Testing workflow price check handling via process_query")
        log.info("Input params: query='What is the current price of ETH?'")
        
        # Test process_query which is what the supervisor uses
        result = await coinbase_agent.process_query("What is the current price of ETH?")
        
        log.info(f"Result: {result}")
        assert result is not None
        assert "response" in result
        assert isinstance(result["response"], str)
        # Should not contain error messages
        assert "error" not in result["response"].lower() or result.get("status") == "success"
        log.info("RESULT: Workflow handled price check successfully via process_query")


# =============================================================================
# Agent Capabilities Tests
# =============================================================================

class TestCoinbaseAgentCapabilities:
    """Tests for high-level agent capabilities."""

    @pytest.mark.asyncio
    async def test_agent_can_get_portfolio_summary(self, coinbase_agent, mock_mcp_server, log):
        """Agent should provide portfolio summary."""
        log.info("Testing get_portfolio_summary capability")
        log.info("Input params: none")
        
        summary = await coinbase_agent.get_portfolio_summary()
        
        log.info(f"Result: {summary}")
        assert summary is not None
        mock_mcp_server.call_tool.assert_called()
        log.info("RESULT: Portfolio summary retrieved successfully")

    @pytest.mark.asyncio
    async def test_agent_can_get_market_data(self, coinbase_agent, mock_mcp_server, log):
        """Agent should fetch market data."""
        log.info("Testing get_market_data capability")
        log.info("Input params: product_id='BTC-USD'")
        
        data = await coinbase_agent.get_market_data("BTC-USD")
        
        log.info(f"Result: {data}")
        assert data is not None
        log.info("RESULT: Market data retrieved successfully")

    @pytest.mark.asyncio
    async def test_agent_can_place_trade(self, coinbase_agent, mock_mcp_server, log):
        """Agent should place trades."""
        log.info("Testing place_market_buy capability")
        log.info("Input params: product_id='BTC-USD', quote_size='100.00'")
        
        result = await coinbase_agent.place_market_buy("BTC-USD", quote_size="100.00")
        
        log.info(f"Result: {result}")
        assert result["success"] is True
        log.info("RESULT: Trade placed successfully")

    @pytest.mark.asyncio
    async def test_agent_can_check_open_orders(self, coinbase_agent, mock_mcp_server, log):
        """Agent should check open orders."""
        log.info("Testing get_open_orders capability")
        log.info("Input params: none")
        
        orders = await coinbase_agent.get_open_orders()
        
        log.info(f"Result: {orders}")
        assert orders is not None
        log.info("RESULT: Open orders retrieved successfully")


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestCoinbaseAgentErrorHandling:
    """Tests for agent error handling."""

    @pytest.mark.asyncio
    async def test_agent_handles_tool_error(self, coinbase_agent, mock_mcp_server, log):
        """Agent should handle tool execution errors."""
        log.info("Testing tool error handling")
        log.info("Input params: tool_name='get_accounts', simulated_error='API Error'")
        from agents.coinbase_agent import CoinbaseAgentError
        
        mock_mcp_server.call_tool.side_effect = Exception("API Error")
        
        with pytest.raises(CoinbaseAgentError):
            await coinbase_agent.execute_tool("get_accounts", {})
        log.info("RESULT: Tool error handled correctly, CoinbaseAgentError raised")

    @pytest.mark.asyncio
    async def test_agent_handles_invalid_tool(self, coinbase_agent, log):
        """Agent should handle invalid tool requests."""
        log.info("Testing invalid tool handling")
        log.info("Input params: tool_name='invalid_tool', params={}")
        from agents.coinbase_agent import CoinbaseAgentError
        
        with pytest.raises(CoinbaseAgentError, match="Unknown tool"):
            await coinbase_agent.execute_tool("invalid_tool", {})
        log.info("RESULT: Invalid tool handled correctly, CoinbaseAgentError raised with 'Unknown tool'")

    @pytest.mark.asyncio
    async def test_agent_handles_order_failure(self, coinbase_agent, mock_mcp_server, log):
        """Agent should handle order failures gracefully."""
        log.info("Testing order failure handling")
        log.info("Input params: product_id='BTC-USD', quote_size='1000000.00', simulated_error='INSUFFICIENT_FUNDS'")
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
        
        log.info(f"Result: {result}")
        assert result.get("success") is False
        log.info("RESULT: Order failure handled gracefully, success=False returned")


# =============================================================================
# Integration Tests
# =============================================================================

class TestCoinbaseAgentIntegration:
    """Integration tests for agent with MCP server."""

    @pytest.mark.asyncio
    async def test_full_trading_flow(self, coinbase_agent, mock_mcp_server, log):
        """Test complete trading flow: check balance -> get price -> place order -> check order."""
        log.info("Testing full trading flow")
        log.info("Input params: product_id='BTC-USD', quote_size='100.00'")
        
        # Step 1: Check account balance
        log.info("Step 1: Checking account balance")
        accounts = await coinbase_agent.execute_tool("get_accounts", {})
        log.info(f"Accounts result: {accounts}")
        assert "accounts" in accounts
        
        # Step 2: Get current price
        log.info("Step 2: Getting current price")
        price = await coinbase_agent.execute_tool("get_product", {"product_id": "BTC-USD"})
        log.info(f"Price result: {price}")
        assert "price" in price
        
        # Step 3: Place order
        log.info("Step 3: Placing order")
        order = await coinbase_agent.execute_tool("market_order_buy", {
            "product_id": "BTC-USD",
            "quote_size": "100.00"
        })
        log.info(f"Order result: {order}")
        assert order["success"] is True
        
        # Step 4: Check order status
        log.info("Step 4: Checking order status")
        status = await coinbase_agent.execute_tool("get_order", {
            "order_id": order["order_id"]
        })
        log.info(f"Status result: {status}")
        assert "order" in status
        log.info("RESULT: Full trading flow completed successfully")

    @pytest.mark.asyncio
    async def test_portfolio_analysis_flow(self, coinbase_agent, mock_mcp_server, log):
        """Test portfolio analysis: get portfolios -> get breakdown -> get fees."""
        log.info("Testing portfolio analysis flow")
        log.info("Input params: portfolio_uuid='port-123'")
        
        # Get portfolios
        log.info("Step 1: Getting portfolios")
        portfolios = await coinbase_agent.execute_tool("get_portfolios", {})
        log.info(f"Portfolios result: {portfolios}")
        assert "portfolios" in portfolios
        
        # Get breakdown
        log.info("Step 2: Getting portfolio breakdown")
        breakdown = await coinbase_agent.execute_tool("get_portfolio_breakdown", {
            "portfolio_uuid": "port-123"
        })
        log.info(f"Breakdown result: {breakdown}")
        assert "breakdown" in breakdown
        
        # Get fee info
        log.info("Step 3: Getting fee info")
        fees = await coinbase_agent.execute_tool("get_transaction_summary", {})
        log.info(f"Fees result: {fees}")
        assert "fee_tier" in fees
        log.info("RESULT: Portfolio analysis flow completed successfully")
