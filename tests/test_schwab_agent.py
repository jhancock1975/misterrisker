"""
Tests for the Schwab LangGraph Agent.

These tests follow TDD principles - written before the implementation.
They cover the agent that uses Schwab MCP tools.
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
    """Create a mock Schwab MCP server."""
    server = MagicMock()
    
    # Mock tool responses
    async def mock_call_tool(tool_name: str, params: dict):
        responses = {
            "get_account_numbers": [
                {"accountNumber": "12345678", "hashValue": "ABC123HASH"}
            ],
            "get_account": {
                "securitiesAccount": {
                    "accountNumber": "12345678",
                    "currentBalances": {"cashBalance": 10000.00, "buyingPower": 19000.00},
                    "positions": [{"symbol": "AAPL", "longQuantity": 100}]
                }
            },
            "get_accounts": [
                {"securitiesAccount": {"accountNumber": "12345678", "currentBalances": {"cashBalance": 10000.00}}}
            ],
            "get_quote": {
                "AAPL": {"quote": {"lastPrice": 174.52, "bidPrice": 174.50, "askPrice": 174.55}}
            },
            "get_quotes": {
                "AAPL": {"quote": {"lastPrice": 174.52}},
                "MSFT": {"quote": {"lastPrice": 378.91}}
            },
            "get_orders_for_account": [
                {"orderId": 123456789, "status": "FILLED"}
            ],
            "get_order": {
                "orderId": 123456789,
                "status": "FILLED",
                "orderType": "LIMIT"
            },
            "place_order": {"order_id": "987654321", "success": True},
            "cancel_order": {"success": True},
            "get_option_chain": {
                "symbol": "AAPL",
                "callExpDateMap": {"2024-01-19:30": {"175.0": [{"bid": 2.50, "ask": 2.55}]}}
            },
            "get_price_history": {
                "candles": [{"open": 173.0, "high": 175.5, "low": 172.5, "close": 174.52}]
            },
            "get_price_history_every_day": {
                "candles": [{"close": 174.52}]
            },
            "get_movers": {
                "screeners": [{"symbol": "NVDA", "netPercentChange": 3.25}]
            },
            "get_market_hours": {
                "equity": {"EQ": {"isOpen": True}}
            },
            "get_transactions": [
                {"transactionId": 12345, "type": "TRADE"}
            ],
            "get_user_preferences": {
                "accounts": [{"accountNumber": "12345678"}]
            },
            "equity_buy_market": {"order_id": "111", "success": True},
            "equity_buy_limit": {"order_id": "222", "success": True},
            "equity_sell_market": {"order_id": "333", "success": True},
            "equity_sell_limit": {"order_id": "444", "success": True},
        }
        return responses.get(tool_name, {})
    
    server.call_tool = AsyncMock(side_effect=mock_call_tool)
    server.list_tools.return_value = [
        {"name": "get_account_numbers", "description": "Get account numbers"},
        {"name": "get_account", "description": "Get account details"},
        {"name": "get_accounts", "description": "Get all accounts"},
        {"name": "get_quote", "description": "Get quote"},
        {"name": "get_quotes", "description": "Get multiple quotes"},
        {"name": "get_orders_for_account", "description": "Get orders"},
        {"name": "get_order", "description": "Get order"},
        {"name": "place_order", "description": "Place order"},
        {"name": "cancel_order", "description": "Cancel order"},
        {"name": "get_option_chain", "description": "Get option chain"},
        {"name": "get_price_history", "description": "Get price history"},
        {"name": "get_price_history_every_day", "description": "Get daily prices"},
        {"name": "get_movers", "description": "Get market movers"},
        {"name": "get_market_hours", "description": "Get market hours"},
        {"name": "get_transactions", "description": "Get transactions"},
        {"name": "get_user_preferences", "description": "Get user preferences"},
        {"name": "equity_buy_market", "description": "Market buy"},
        {"name": "equity_buy_limit", "description": "Limit buy"},
        {"name": "equity_sell_market", "description": "Market sell"},
        {"name": "equity_sell_limit", "description": "Limit sell"},
    ]
    
    return server


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing Schwab agent routing."""
    from unittest.mock import MagicMock, AsyncMock
    import json
    
    llm = MagicMock()
    
    # Default routing response for account_balance
    def create_routing_response(intent="account_balance", details=None):
        response = MagicMock()
        response.content = json.dumps({
            "intent": intent,
            "details": details or {}
        })
        return response
    
    # Make ainvoke return a mock async response
    async def mock_ainvoke(messages):
        # Analyze the query to determine appropriate response
        query = ""
        for msg in messages:
            if hasattr(msg, 'content') and 'buy' in str(msg.content).lower():
                return create_routing_response("place_order", {"action": "buy", "symbol": "AAPL", "quantity": 10})
            if hasattr(msg, 'content') and 'price' in str(msg.content).lower():
                return create_routing_response("get_quote", {"symbol": "AAPL"})
            if hasattr(msg, 'content') and 'balance' in str(msg.content).lower():
                return create_routing_response("account_balance", {})
        return create_routing_response("account_balance", {})
    
    llm.ainvoke = mock_ainvoke
    
    # Regular invoke for CoT and other uses
    mock_response = MagicMock()
    mock_response.content = "Analysis complete"
    llm.invoke = MagicMock(return_value=mock_response)
    
    return llm


@pytest.fixture
def schwab_agent(mock_mcp_server, mock_llm):
    """Create a Schwab agent with mocked MCP server and LLM."""
    from agents.schwab_agent import SchwabAgent
    
    agent = SchwabAgent(mcp_server=mock_mcp_server, llm=mock_llm)
    return agent


# =============================================================================
# Agent Initialization Tests
# =============================================================================

class TestSchwabAgentInitialization:
    """Tests for agent initialization."""

    def test_agent_initializes_with_mcp_server(self, mock_mcp_server):
        """Agent should initialize with an MCP server."""
        from agents.schwab_agent import SchwabAgent
        
        agent = SchwabAgent(mcp_server=mock_mcp_server)
        
        assert agent is not None
        assert agent.mcp_server == mock_mcp_server

    def test_agent_discovers_available_tools(self, schwab_agent, mock_mcp_server):
        """Agent should discover tools from MCP server."""
        tools = schwab_agent.get_available_tools()
        
        assert len(tools) > 0
        mock_mcp_server.list_tools.assert_called()

    def test_agent_creates_langgraph_workflow(self, schwab_agent):
        """Agent should create a LangGraph workflow."""
        workflow = schwab_agent.get_workflow()
        
        assert workflow is not None

    def test_agent_initializes_with_account_hash(self, mock_mcp_server):
        """Agent should accept default account hash."""
        from agents.schwab_agent import SchwabAgent
        
        agent = SchwabAgent(
            mcp_server=mock_mcp_server,
            default_account_hash="ABC123HASH"
        )
        
        assert agent.default_account_hash == "ABC123HASH"


# =============================================================================
# Agent State Tests
# =============================================================================

class TestSchwabAgentState:
    """Tests for agent state management."""

    def test_state_tracks_conversation(self):
        """Agent state should track conversation history."""
        from agents.schwab_agent import SchwabAgentState
        
        state = SchwabAgentState(
            messages=[],
            tool_calls=[],
            tool_results=[],
            current_task=""
        )
        
        assert state["messages"] == []
        assert state["tool_calls"] == []

    def test_state_tracks_account_context(self):
        """Agent state should track account context."""
        from agents.schwab_agent import SchwabAgentState
        
        state = SchwabAgentState(
            messages=[],
            tool_calls=[],
            tool_results=[],
            current_task="",
            account_hash="ABC123HASH",
            cash_balance=10000.0,
            buying_power=19000.0,
            positions={"AAPL": 100, "MSFT": 50}
        )
        
        assert state["account_hash"] == "ABC123HASH"
        assert state["cash_balance"] == 10000.0
        assert state["buying_power"] == 19000.0
        assert state["positions"]["AAPL"] == 100

    def test_state_tracks_market_data(self):
        """Agent state should track market data."""
        from agents.schwab_agent import SchwabAgentState
        
        state = SchwabAgentState(
            messages=[],
            tool_calls=[],
            tool_results=[],
            current_task="",
            quotes={"AAPL": 174.52},
            market_hours={"equity": {"isOpen": True}}
        )
        
        assert state["quotes"]["AAPL"] == 174.52
        assert state["market_hours"]["equity"]["isOpen"] is True


# =============================================================================
# Tool Execution Tests
# =============================================================================

class TestSchwabAgentToolExecution:
    """Tests for agent tool execution."""

    @pytest.mark.asyncio
    async def test_agent_executes_get_account_numbers(self, schwab_agent, mock_mcp_server):
        """Agent should execute get_account_numbers tool."""
        result = await schwab_agent.execute_tool("get_account_numbers", {})
        
        mock_mcp_server.call_tool.assert_called_with("get_account_numbers", {})
        assert isinstance(result, list)
        assert "accountNumber" in result[0]

    @pytest.mark.asyncio
    async def test_agent_executes_get_account(self, schwab_agent, mock_mcp_server):
        """Agent should get account details."""
        result = await schwab_agent.execute_tool("get_account", {
            "account_hash": "ABC123HASH"
        })
        
        assert "securitiesAccount" in result

    @pytest.mark.asyncio
    async def test_agent_executes_get_quote(self, schwab_agent, mock_mcp_server):
        """Agent should get stock quote."""
        result = await schwab_agent.execute_tool("get_quote", {
            "symbol": "AAPL"
        })
        
        assert "AAPL" in result
        assert "quote" in result["AAPL"]

    @pytest.mark.asyncio
    async def test_agent_executes_place_order(self, schwab_agent, mock_mcp_server):
        """Agent should place orders."""
        result = await schwab_agent.execute_tool("place_order", {
            "account_hash": "ABC123HASH",
            "order_spec": {"orderType": "LIMIT", "price": 150.00}
        })
        
        assert result.get("success") is True

    @pytest.mark.asyncio
    async def test_agent_executes_cancel_order(self, schwab_agent, mock_mcp_server):
        """Agent should cancel orders."""
        result = await schwab_agent.execute_tool("cancel_order", {
            "account_hash": "ABC123HASH",
            "order_id": 123456789
        })
        
        assert result.get("success") is True

    @pytest.mark.asyncio
    async def test_agent_executes_get_option_chain(self, schwab_agent, mock_mcp_server):
        """Agent should get option chains."""
        result = await schwab_agent.execute_tool("get_option_chain", {
            "symbol": "AAPL"
        })
        
        assert "symbol" in result
        assert "callExpDateMap" in result


# =============================================================================
# Workflow Tests
# =============================================================================

class TestSchwabAgentWorkflow:
    """Tests for agent LangGraph workflow."""

    @pytest.mark.asyncio
    async def test_workflow_processes_query(self, schwab_agent):
        """Workflow should process user queries."""
        from agents.schwab_agent import SchwabAgentState
        
        initial_state = SchwabAgentState(
            messages=[{"role": "user", "content": "What is my account balance?"}],
            tool_calls=[],
            tool_results=[],
            current_task="check_balance"
        )
        
        result = await schwab_agent.run(initial_state)
        
        assert result is not None
        assert "messages" in result

    @pytest.mark.asyncio
    async def test_workflow_handles_buy_request(self, schwab_agent):
        """Workflow should handle buy requests."""
        from agents.schwab_agent import SchwabAgentState
        
        initial_state = SchwabAgentState(
            messages=[{"role": "user", "content": "Buy 10 shares of AAPL at $150"}],
            tool_calls=[],
            tool_results=[],
            current_task="buy_stock"
        )
        
        result = await schwab_agent.run(initial_state)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_workflow_handles_price_check(self, schwab_agent):
        """Workflow should handle price check requests."""
        from agents.schwab_agent import SchwabAgentState
        
        initial_state = SchwabAgentState(
            messages=[{"role": "user", "content": "What is the current price of MSFT?"}],
            tool_calls=[],
            tool_results=[],
            current_task="check_price"
        )
        
        result = await schwab_agent.run(initial_state)
        
        assert result is not None


# =============================================================================
# Agent Capabilities Tests
# =============================================================================

class TestSchwabAgentCapabilities:
    """Tests for high-level agent capabilities."""

    @pytest.mark.asyncio
    async def test_agent_can_get_account_summary(self, schwab_agent, mock_mcp_server):
        """Agent should provide account summary."""
        summary = await schwab_agent.get_account_summary("ABC123HASH")
        
        assert summary is not None
        mock_mcp_server.call_tool.assert_called()

    @pytest.mark.asyncio
    async def test_agent_can_get_portfolio(self, schwab_agent, mock_mcp_server):
        """Agent should get portfolio positions."""
        portfolio = await schwab_agent.get_portfolio("ABC123HASH")
        
        assert portfolio is not None

    @pytest.mark.asyncio
    async def test_agent_can_get_stock_price(self, schwab_agent, mock_mcp_server):
        """Agent should get stock prices."""
        price = await schwab_agent.get_stock_price("AAPL")
        
        assert price is not None

    @pytest.mark.asyncio
    async def test_agent_can_place_market_buy(self, schwab_agent, mock_mcp_server):
        """Agent should place market buy orders."""
        result = await schwab_agent.place_market_buy(
            account_hash="ABC123HASH",
            symbol="AAPL",
            quantity=10
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_can_place_limit_buy(self, schwab_agent, mock_mcp_server):
        """Agent should place limit buy orders."""
        result = await schwab_agent.place_limit_buy(
            account_hash="ABC123HASH",
            symbol="AAPL",
            quantity=10,
            price=150.00
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_can_place_market_sell(self, schwab_agent, mock_mcp_server):
        """Agent should place market sell orders."""
        result = await schwab_agent.place_market_sell(
            account_hash="ABC123HASH",
            symbol="AAPL",
            quantity=10
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_can_place_limit_sell(self, schwab_agent, mock_mcp_server):
        """Agent should place limit sell orders."""
        result = await schwab_agent.place_limit_sell(
            account_hash="ABC123HASH",
            symbol="AAPL",
            quantity=10,
            price=180.00
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_can_check_open_orders(self, schwab_agent, mock_mcp_server):
        """Agent should check open orders."""
        orders = await schwab_agent.get_open_orders("ABC123HASH")
        
        assert orders is not None

    @pytest.mark.asyncio
    async def test_agent_can_get_market_movers(self, schwab_agent, mock_mcp_server):
        """Agent should get market movers."""
        movers = await schwab_agent.get_market_movers("$SPX")
        
        assert movers is not None

    @pytest.mark.asyncio
    async def test_agent_can_check_market_hours(self, schwab_agent, mock_mcp_server):
        """Agent should check market hours."""
        hours = await schwab_agent.get_market_hours()
        
        assert hours is not None


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestSchwabAgentErrorHandling:
    """Tests for agent error handling."""

    @pytest.mark.asyncio
    async def test_agent_handles_tool_error(self, schwab_agent, mock_mcp_server):
        """Agent should handle tool execution errors."""
        from agents.schwab_agent import SchwabAgentError
        
        mock_mcp_server.call_tool.side_effect = Exception("API Error")
        
        with pytest.raises(SchwabAgentError):
            await schwab_agent.execute_tool("get_account_numbers", {})

    @pytest.mark.asyncio
    async def test_agent_handles_invalid_tool(self, schwab_agent):
        """Agent should handle invalid tool requests."""
        from agents.schwab_agent import SchwabAgentError
        
        with pytest.raises(SchwabAgentError, match="Unknown tool"):
            await schwab_agent.execute_tool("invalid_tool", {})

    @pytest.mark.asyncio
    async def test_agent_handles_order_failure(self, schwab_agent, mock_mcp_server):
        """Agent should handle order failures gracefully."""
        async def mock_order_failure(tool_name: str, params: dict):
            if tool_name == "equity_buy_market":
                return {"success": False, "error": "INSUFFICIENT_FUNDS"}
            return {}
        
        mock_mcp_server.call_tool.side_effect = mock_order_failure
        
        result = await schwab_agent.place_market_buy(
            account_hash="ABC123HASH",
            symbol="AAPL",
            quantity=1000000
        )
        
        assert result.get("success") is False

    @pytest.mark.asyncio
    async def test_agent_handles_market_closed(self, schwab_agent, mock_mcp_server):
        """Agent should handle market closed scenarios."""
        async def mock_market_closed(tool_name: str, params: dict):
            if tool_name == "get_market_hours":
                return {"equity": {"EQ": {"isOpen": False}}}
            return {}
        
        mock_mcp_server.call_tool.side_effect = mock_market_closed
        
        hours = await schwab_agent.get_market_hours()
        
        assert hours["equity"]["EQ"]["isOpen"] is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestSchwabAgentIntegration:
    """Integration tests for agent with MCP server."""

    @pytest.mark.asyncio
    async def test_full_trading_flow(self, schwab_agent, mock_mcp_server):
        """Test complete trading flow: get accounts -> check balance -> get price -> place order."""
        # Step 1: Get account numbers
        accounts = await schwab_agent.execute_tool("get_account_numbers", {})
        assert isinstance(accounts, list)
        account_hash = accounts[0]["hashValue"]
        
        # Step 2: Get account details
        account = await schwab_agent.execute_tool("get_account", {
            "account_hash": account_hash
        })
        assert "securitiesAccount" in account
        
        # Step 3: Get stock price
        quote = await schwab_agent.execute_tool("get_quote", {"symbol": "AAPL"})
        assert "AAPL" in quote
        
        # Step 4: Place order
        order = await schwab_agent.execute_tool("place_order", {
            "account_hash": account_hash,
            "order_spec": {"orderType": "LIMIT", "price": 150.00}
        })
        assert order.get("success") is True

    @pytest.mark.asyncio
    async def test_portfolio_analysis_flow(self, schwab_agent, mock_mcp_server):
        """Test portfolio analysis: get positions -> get quotes -> get history."""
        # Get account with positions
        account = await schwab_agent.execute_tool("get_account", {
            "account_hash": "ABC123HASH",
            "fields": ["positions"]
        })
        assert "securitiesAccount" in account
        
        # Get quotes for multiple symbols
        quotes = await schwab_agent.execute_tool("get_quotes", {
            "symbols": ["AAPL", "MSFT"]
        })
        assert "AAPL" in quotes
        
        # Get price history
        history = await schwab_agent.execute_tool("get_price_history_every_day", {
            "symbol": "AAPL"
        })
        assert "candles" in history

    @pytest.mark.asyncio
    async def test_options_flow(self, schwab_agent, mock_mcp_server):
        """Test options flow: get chain -> analyze."""
        # Get option chain
        chain = await schwab_agent.execute_tool("get_option_chain", {
            "symbol": "AAPL"
        })
        assert "callExpDateMap" in chain

    @pytest.mark.asyncio
    async def test_market_research_flow(self, schwab_agent, mock_mcp_server):
        """Test market research: movers -> quotes -> history."""
        # Get market movers
        movers = await schwab_agent.execute_tool("get_movers", {"index": "$SPX"})
        assert "screeners" in movers
        
        # Check market hours
        hours = await schwab_agent.execute_tool("get_market_hours", {
            "markets": ["equity"]
        })
        assert "equity" in hours


# =============================================================================
# Account Selection Tests
# =============================================================================

class TestAccountSelection:
    """Tests for account selection functionality."""

    @pytest.mark.asyncio
    async def test_agent_uses_default_account(self, mock_mcp_server):
        """Agent should use default account when set."""
        from agents.schwab_agent import SchwabAgent
        
        agent = SchwabAgent(
            mcp_server=mock_mcp_server,
            default_account_hash="ABC123HASH"
        )
        
        result = await agent.get_account_summary()
        
        # Should use default account hash
        mock_mcp_server.call_tool.assert_called()

    @pytest.mark.asyncio
    async def test_agent_auto_selects_first_account(self, schwab_agent, mock_mcp_server):
        """Agent should auto-select first account if no default."""
        result = await schwab_agent.get_account_summary()
        
        # Should have called get_account_numbers first
        calls = mock_mcp_server.call_tool.call_args_list
        tool_names = [call[0][0] for call in calls]
        assert "get_account_numbers" in tool_names


# =============================================================================
# Process Query Tests
# =============================================================================

class TestProcessQuery:
    """Tests for query processing and order handling."""

    @pytest.mark.asyncio
    async def test_process_query_routes_balance_request(self, schwab_agent, mock_mcp_server):
        """Process query should route balance requests to account summary."""
        result = await schwab_agent.process_query("what's my account balance?")
        
        assert result["status"] == "success"
        assert "response" in result

    @pytest.mark.asyncio
    async def test_process_query_routes_price_request(self, schwab_agent, mock_mcp_server):
        """Process query should route price requests to get_quote."""
        # The fixture mock_llm already handles "price" queries by returning get_quote intent
        result = await schwab_agent.process_query("what's the price of AAPL?")
        
        assert result["status"] == "success"
        # The response should contain price info (even if it's "no data" since mock server returns empty)
        assert "response" in result

    @pytest.mark.asyncio
    async def test_handle_order_request_parses_limit_buy(self, schwab_agent, mock_mcp_server):
        """Order handler should parse and execute limit buy orders."""
        # Mock the LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"action": "buy", "symbol": "NVDA", "quantity": 1, "order_type": "limit", "price": 160, "take_profit": 180, "stop_loss": 120}'
        
        schwab_agent.llm = AsyncMock()
        schwab_agent.llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        
        result = await schwab_agent._handle_order_request(
            "buy 1 share of NVDA at $160 with take profit at 180 and stop loss at 120"
        )
        
        # Should have placed a limit buy order
        assert "NVDA" in result
        assert "limit buy" in result.lower() or "placed" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_order_request_handles_list_content(self, schwab_agent, mock_mcp_server):
        """Order handler should handle LLM responses that return list content."""
        # Mock the LLM response with list content (the bug we fixed)
        mock_llm_response = MagicMock()
        mock_llm_response.content = ['{"action": "buy", "symbol": "AAPL", "quantity": 10, "order_type": "market", "price": null}']
        
        schwab_agent.llm = AsyncMock()
        schwab_agent.llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        
        result = await schwab_agent._handle_order_request("buy 10 shares of AAPL at market")
        
        # Should not raise an error about 'list' object has no attribute 'strip'
        assert "AAPL" in result or "error" in result.lower() or "couldn't" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_order_request_handles_dict_in_list(self, schwab_agent, mock_mcp_server):
        """Order handler should handle LLM responses with dict in list."""
        mock_llm_response = MagicMock()
        mock_llm_response.content = [{"text": '{"action": "sell", "symbol": "TSLA", "quantity": 5, "order_type": "limit", "price": 250}'}]
        
        schwab_agent.llm = AsyncMock()
        schwab_agent.llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        
        result = await schwab_agent._handle_order_request("sell 5 shares of TSLA at $250")
        
        # Should handle dict-in-list content
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_process_query_routes_buy_to_order_handler(self, schwab_agent, mock_mcp_server):
        """Process query should route buy requests to order handler."""
        # Create a mock that handles both routing and order parsing calls
        call_count = [0]
        
        async def multi_call_mock(messages):
            call_count[0] += 1
            mock_response = MagicMock()
            if call_count[0] == 1:
                # First call: routing - return place_order intent
                mock_response.content = '{"intent": "place_order", "details": {"action": "buy", "symbol": "GOOGL", "quantity": 2}}'
            else:
                # Second call: order parsing
                mock_response.content = '{"action": "buy", "symbol": "GOOGL", "quantity": 2, "order_type": "market", "price": null}'
            return mock_response
        
        schwab_agent.llm.ainvoke = multi_call_mock
        
        result = await schwab_agent.process_query("buy 2 shares of GOOGL")
        
        assert result["status"] == "success"
        # Should have called equity_buy_market
        calls = mock_mcp_server.call_tool.call_args_list
        tool_names = [call[0][0] for call in calls]
        assert "equity_buy_market" in tool_names
    @pytest.mark.asyncio
    async def test_process_query_routes_transaction_history(self, schwab_agent, mock_mcp_server):
        """Process query should route transaction history requests."""
        # Mock LLM to return transaction_history intent
        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"intent": "transaction_history", "details": {}}'
        
        schwab_agent.llm = AsyncMock()
        schwab_agent.llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        
        result = await schwab_agent.process_query("show me my recent transaction history")
        
        assert result["status"] == "success"
        assert "transaction" in result["response"].lower() or "no recent" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_process_query_routes_top_stocks(self, schwab_agent, mock_mcp_server):
        """Process query should route top stocks requests to market movers."""
        # Mock LLM to return top_stocks intent
        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"intent": "top_stocks", "details": {"count": 100}}'
        
        schwab_agent.llm = AsyncMock()
        schwab_agent.llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        
        result = await schwab_agent.process_query("show me top 100 stocks")
        
        assert result["status"] == "success"
        # Should call get_movers
        calls = mock_mcp_server.call_tool.call_args_list
        tool_names = [call[0][0] for call in calls]
        assert "get_movers" in tool_names


class TestFormatters:
    """Tests for formatter methods to prevent slicing errors."""

    def test_format_transactions_with_list(self, schwab_agent):
        """Format transactions should handle list input."""
        transactions = [
            {"type": "TRADE", "transactionDate": "2024-01-15", "netAmount": -1500.00},
            {"type": "DIVIDEND", "transactionDate": "2024-01-10", "netAmount": 25.50}
        ]
        
        result = schwab_agent._format_transactions(transactions)
        
        assert isinstance(result, str)
        assert "TRADE" in result or "transaction" in result.lower()

    def test_format_transactions_with_dict_wrapper(self, schwab_agent):
        """Format transactions should handle dict wrapper."""
        data = {
            "transactions": [
                {"type": "TRADE", "transactionDate": "2024-01-15", "netAmount": -500.00}
            ]
        }
        
        result = schwab_agent._format_transactions(data)
        
        assert isinstance(result, str)
        assert "TRADE" in result or "transaction" in result.lower()

    def test_format_transactions_with_empty_list(self, schwab_agent):
        """Format transactions should handle empty list."""
        result = schwab_agent._format_transactions([])
        
        assert isinstance(result, str)
        assert "no" in result.lower()

    def test_format_transactions_with_none(self, schwab_agent):
        """Format transactions should handle None input."""
        result = schwab_agent._format_transactions(None)
        
        assert isinstance(result, str)
        assert "no" in result.lower()

    def test_format_positions_with_list(self, schwab_agent):
        """Format positions should handle list input."""
        positions = [
            {"instrument": {"symbol": "AAPL"}, "longQuantity": 100, "marketValue": 17500.00},
            {"instrument": {"symbol": "MSFT"}, "longQuantity": 50, "marketValue": 18900.00}
        ]
        
        result = schwab_agent._format_positions(positions)
        
        assert isinstance(result, str)
        assert "AAPL" in result
        assert "MSFT" in result

    def test_format_positions_with_empty_list(self, schwab_agent):
        """Format positions should handle empty list."""
        result = schwab_agent._format_positions([])
        
        assert isinstance(result, str)
        assert "no positions" in result.lower()

    def test_format_positions_with_none(self, schwab_agent):
        """Format positions should handle None without slicing error."""
        # This tests the bug that caused slice(None, 10, None) error
        result = schwab_agent._format_positions(None)
        
        assert isinstance(result, str)
        assert "no positions" in result.lower()

    def test_format_market_movers_with_valid_data(self, schwab_agent):
        """Format market movers should handle valid screener data."""
        data = {
            "screeners": [
                {
                    "direction": "up",
                    "instruments": [
                        {"symbol": "NVDA", "netChange": 5.50, "netPercentChange": 3.5, "lastPrice": 150.00}
                    ]
                }
            ]
        }
        
        result = schwab_agent._format_market_movers(data)
        
        assert isinstance(result, str)
        assert "NVDA" in result

    def test_format_market_movers_with_empty_screeners(self, schwab_agent):
        """Format market movers should handle empty screeners."""
        result = schwab_agent._format_market_movers({"screeners": []})
        
        assert isinstance(result, str)

    def test_format_orders_with_list(self, schwab_agent):
        """Format orders should handle list input."""
        orders = [
            {"orderId": 12345, "status": "FILLED", "orderLegCollection": [{"instrument": {"symbol": "AAPL"}}]}
        ]
        
        result = schwab_agent._format_orders(orders)
        
        assert isinstance(result, str)

    def test_format_orders_with_none(self, schwab_agent):
        """Format orders should handle None without error."""
        result = schwab_agent._format_orders(None)
        
        assert isinstance(result, str)
        assert "no" in result.lower()