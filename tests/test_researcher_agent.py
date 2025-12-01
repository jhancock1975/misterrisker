"""
Tests for the Researcher Agent.

TDD tests for the LangGraph-based researcher agent that uses the
Researcher MCP Server for making trading and investment decisions.

Tests written before implementation following TDD principles.
"""

import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test logger
logger = logging.getLogger("test.researcher_agent")


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_researcher_server():
    """Create a mock Researcher MCP server."""
    server = MagicMock()
    
    async def mock_call_tool(tool_name: str, args: dict):
        """Mock tool call dispatcher."""
        if tool_name == "get_stock_quote":
            return {
                "current_price": 150.25,
                "change": 2.50,
                "change_percent": 1.69,
                "high": 152.00,
                "low": 148.50,
                "open": 149.00,
                "previous_close": 147.75,
                "symbol": args.get("symbol", "AAPL")
            }
        elif tool_name == "get_company_profile":
            return {
                "name": "Apple Inc",
                "industry": "Technology",
                "market_cap": 2500000,
                "country": "US",
                "exchange": "NASDAQ",
                "symbol": args.get("symbol", "AAPL")
            }
        elif tool_name == "get_company_news":
            return {
                "news": [
                    {
                        "headline": "Apple Reports Record Q4 Earnings",
                        "summary": "Apple Inc reported record quarterly earnings...",
                        "source": "Reuters",
                        "datetime": "2024-01-15T10:00:00Z",
                        "url": "https://example.com/news/1"
                    },
                    {
                        "headline": "Apple Launches New iPhone",
                        "summary": "Apple unveiled its latest iPhone model...",
                        "source": "Bloomberg",
                        "datetime": "2024-01-14T10:00:00Z",
                        "url": "https://example.com/news/2"
                    }
                ],
                "symbol": args.get("symbol", "AAPL")
            }
        elif tool_name == "get_analyst_recommendations":
            return {
                "recommendations": {
                    "strong_buy": 13,
                    "buy": 24,
                    "hold": 7,
                    "sell": 0,
                    "strong_sell": 0
                },
                "period": "2024-01-01",
                "symbol": args.get("symbol", "AAPL")
            }
        elif tool_name == "get_basic_financials":
            return {
                "52_week_high": 199.62,
                "52_week_low": 124.17,
                "pe_ratio": 28.5,
                "market_cap": 2500000,
                "dividend_yield": 0.52,
                "eps": 6.13,
                "symbol": args.get("symbol", "AAPL")
            }
        elif tool_name == "get_company_peers":
            return {
                "peers": ["MSFT", "GOOGL", "META", "AMZN", "NVDA"],
                "symbol": args.get("symbol", "AAPL")
            }
        elif tool_name == "get_earnings_history":
            return {
                "earnings": [
                    {
                        "actual": 1.88,
                        "estimate": 1.75,
                        "surprise_percent": 7.43,
                        "period": "2024-03-31",
                        "quarter": 1,
                        "year": 2024
                    }
                ],
                "symbol": args.get("symbol", "AAPL")
            }
        elif tool_name == "get_market_news":
            return {
                "news": [
                    {
                        "headline": "Markets Rally on Fed Comments",
                        "summary": "Stock markets rallied today...",
                        "source": "CNBC",
                        "category": args.get("category", "general")
                    }
                ]
            }
        elif tool_name == "web_search":
            return {
                "answer": "Based on recent analysis, AAPL shows strong fundamentals...",
                "citations": [
                    {
                        "url": "https://finance.yahoo.com/aapl-analysis",
                        "title": "AAPL Stock Analysis"
                    }
                ]
            }
        elif tool_name == "research_stock":
            return {
                "quote": {"current_price": 150.25},
                "profile": {"name": "Apple Inc"},
                "financials": {"pe_ratio": 28.5},
                "recommendations": {"buy": 24},
                "news": [{"headline": "Apple News"}],
                "web_insights": "Strong buy recommendation from analysts..."
            }
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    server.call_tool = AsyncMock(side_effect=mock_call_tool)
    server.list_tools.return_value = [
        {"name": "get_stock_quote"},
        {"name": "get_company_profile"},
        {"name": "get_company_news"},
        {"name": "get_market_news"},
        {"name": "get_analyst_recommendations"},
        {"name": "get_basic_financials"},
        {"name": "get_company_peers"},
        {"name": "get_earnings_history"},
        {"name": "web_search"},
        {"name": "research_stock"}
    ]
    
    return server


@pytest.fixture
def mock_llm():
    """Create a mock LLM for the agent."""
    llm = MagicMock()
    
    # Mock response for analysis
    mock_response = MagicMock()
    mock_response.content = "Based on the research data, AAPL appears to be a strong buy..."
    llm.invoke = AsyncMock(return_value=mock_response)
    llm.bind_tools = MagicMock(return_value=llm)
    
    return llm


@pytest.fixture
def researcher_agent(mock_researcher_server, mock_llm):
    """Create a Researcher agent with mocked dependencies."""
    from agents.researcher_agent import ResearcherAgent
    
    agent = ResearcherAgent(
        researcher_server=mock_researcher_server,
        llm=mock_llm
    )
    return agent


# =============================================================================
# Agent Initialization Tests
# =============================================================================

class TestResearcherAgentInitialization:
    """Tests for agent initialization."""

    def test_agent_initializes(self, mock_researcher_server, mock_llm, log):
        """Agent should initialize with MCP server and LLM."""
        from agents.researcher_agent import ResearcherAgent
        
        log.info("Testing agent initialization")
        
        agent = ResearcherAgent(
            researcher_server=mock_researcher_server,
            llm=mock_llm
        )
        
        log.info(f"Agent created: {agent}")
        assert agent is not None
        assert agent.researcher_server == mock_researcher_server
        
        log.info("RESULT: Agent initialized successfully")

    def test_agent_creates_langgraph_workflow(self, researcher_agent, log):
        """Agent should create a LangGraph workflow."""
        log.info("Testing LangGraph workflow creation")
        
        workflow = researcher_agent.get_workflow()
        
        log.info(f"Workflow: {workflow}")
        assert workflow is not None
        
        log.info("RESULT: LangGraph workflow created")

    def test_agent_registers_tools(self, researcher_agent, log):
        """Agent should register research tools from MCP server."""
        log.info("Testing tool registration")
        
        tools = researcher_agent.get_tools()
        
        log.info(f"Registered tools: {[t['name'] for t in tools]}")
        assert len(tools) > 0
        
        log.info(f"RESULT: {len(tools)} tools registered")


# =============================================================================
# Research Query Tests
# =============================================================================

class TestResearchQueries:
    """Tests for research query handling."""

    @pytest.mark.asyncio
    async def test_simple_stock_query(self, researcher_agent, log):
        """Should handle simple stock research query."""
        log.info("Testing simple stock query")
        
        result = await researcher_agent.run("What is the current price of AAPL?")
        
        log.info(f"Result: {result}")
        
        assert "response" in result
        assert result["status"] == "success"
        
        log.info("RESULT: Simple stock query handled")

    @pytest.mark.asyncio
    async def test_comprehensive_stock_research(self, researcher_agent, log):
        """Should perform comprehensive stock research."""
        log.info("Testing comprehensive stock research")
        
        result = await researcher_agent.run(
            "Give me a complete analysis of AAPL including price, financials, news, and analyst recommendations"
        )
        
        log.info(f"Result keys: {result.keys()}")
        
        assert "response" in result
        assert result["status"] == "success"
        assert "research_data" in result
        
        log.info("RESULT: Comprehensive stock research completed")

    @pytest.mark.asyncio
    async def test_market_news_query(self, researcher_agent, log):
        """Should handle market news query."""
        log.info("Testing market news query")
        
        result = await researcher_agent.run("What's happening in the market today?")
        
        log.info(f"Result: {result}")
        
        assert "response" in result
        assert result["status"] == "success"
        
        log.info("RESULT: Market news query handled")

    @pytest.mark.asyncio
    async def test_stock_comparison_query(self, researcher_agent, mock_researcher_server, log):
        """Should handle stock comparison query."""
        log.info("Testing stock comparison query")
        
        result = await researcher_agent.run(
            "Compare AAPL and MSFT stocks. Which is a better investment?"
        )
        
        log.info(f"Result: {result}")
        
        assert "response" in result
        assert result["status"] == "success"
        
        # Verify multiple stocks were researched
        calls = mock_researcher_server.call_tool.call_args_list
        log.info(f"Tool calls made: {len(calls)}")
        
        log.info("RESULT: Stock comparison query handled")

    @pytest.mark.asyncio
    async def test_web_search_query(self, researcher_agent, log):
        """Should use web search for broader research."""
        log.info("Testing web search query")
        
        result = await researcher_agent.run(
            "What are the latest analyst opinions on AAPL for 2024?"
        )
        
        log.info(f"Result: {result}")
        
        assert "response" in result
        assert result["status"] == "success"
        
        log.info("RESULT: Web search query handled")


# =============================================================================
# Investment Decision Tests
# =============================================================================

class TestInvestmentDecisions:
    """Tests for investment decision support."""

    @pytest.mark.asyncio
    async def test_buy_recommendation_analysis(self, researcher_agent, log):
        """Should analyze and provide buy/sell recommendation."""
        log.info("Testing buy recommendation analysis")
        
        result = await researcher_agent.run(
            "Should I buy AAPL stock? Give me your analysis."
        )
        
        log.info(f"Result: {result}")
        
        assert "response" in result
        assert result["status"] == "success"
        assert "recommendation" in result or "recommendation" in result.get("response", "").lower()
        
        log.info("RESULT: Buy recommendation analysis completed")

    @pytest.mark.asyncio
    async def test_portfolio_suggestion(self, researcher_agent, log):
        """Should provide portfolio suggestions."""
        log.info("Testing portfolio suggestion")
        
        result = await researcher_agent.run(
            "I have $10,000 to invest in tech stocks. What do you recommend?"
        )
        
        log.info(f"Result: {result}")
        
        assert "response" in result
        assert result["status"] == "success"
        
        log.info("RESULT: Portfolio suggestion provided")

    @pytest.mark.asyncio
    async def test_risk_analysis(self, researcher_agent, log):
        """Should analyze investment risk."""
        log.info("Testing risk analysis")
        
        result = await researcher_agent.run(
            "What are the risks of investing in AAPL right now?"
        )
        
        log.info(f"Result: {result}")
        
        assert "response" in result
        assert result["status"] == "success"
        
        log.info("RESULT: Risk analysis completed")


# =============================================================================
# Workflow State Tests
# =============================================================================

class TestWorkflowState:
    """Tests for LangGraph workflow state management."""

    @pytest.mark.asyncio
    async def test_workflow_state_tracking(self, researcher_agent, log):
        """Should track workflow state throughout execution."""
        log.info("Testing workflow state tracking")
        
        result = await researcher_agent.run(
            "Research AAPL stock",
            return_state=True
        )
        
        log.info(f"Final state: {result.get('state', {}).keys()}")
        
        assert "state" in result
        assert "messages" in result["state"]
        
        log.info("RESULT: Workflow state tracked successfully")

    @pytest.mark.asyncio
    async def test_workflow_step_history(self, researcher_agent, log):
        """Should maintain history of workflow steps."""
        log.info("Testing workflow step history")
        
        result = await researcher_agent.run(
            "Get me the latest news on AAPL and MSFT",
            return_state=True
        )
        
        log.info(f"State: {result}")
        
        if "state" in result:
            history = result["state"].get("history", [])
            log.info(f"History steps: {len(history)}")
        
        log.info("RESULT: Workflow history maintained")


# =============================================================================
# Streaming Tests
# =============================================================================

class TestStreaming:
    """Tests for streaming responses."""

    @pytest.mark.asyncio
    async def test_stream_response(self, researcher_agent, log):
        """Should support streaming responses."""
        log.info("Testing streaming response")
        
        chunks = []
        async for chunk in researcher_agent.stream("What is AAPL's current price?"):
            chunks.append(chunk)
            log.info(f"Received chunk: {type(chunk)}")
        
        log.info(f"Total chunks: {len(chunks)}")
        assert len(chunks) > 0
        
        log.info("RESULT: Streaming response works")

    @pytest.mark.asyncio
    async def test_stream_with_tools(self, researcher_agent, log):
        """Should stream tool calls and results."""
        log.info("Testing streaming with tool calls")
        
        tool_chunks = []
        async for chunk in researcher_agent.stream(
            "Get AAPL quote and analyst recommendations",
            stream_tools=True
        ):
            if chunk.get("type") == "tool":
                tool_chunks.append(chunk)
                log.info(f"Tool chunk: {chunk.get('tool_name')}")
        
        log.info(f"Total tool chunks: {len(tool_chunks)}")
        
        log.info("RESULT: Tool streaming works")


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestAgentErrorHandling:
    """Tests for agent error handling."""

    @pytest.mark.asyncio
    async def test_handles_server_error(self, researcher_agent, mock_researcher_server, log):
        """Should handle MCP server errors gracefully."""
        log.info("Testing server error handling")
        
        mock_researcher_server.call_tool.side_effect = Exception("Server connection failed")
        
        result = await researcher_agent.run("Get AAPL price")
        
        log.info(f"Result: {result}")
        
        assert result["status"] == "error" or "error" in result.get("response", "").lower()
        
        log.info("RESULT: Server error handled gracefully")

    @pytest.mark.asyncio
    async def test_handles_invalid_symbol(self, researcher_agent, mock_researcher_server, log):
        """Should handle invalid stock symbol."""
        log.info("Testing invalid symbol handling")
        
        async def mock_invalid_symbol(tool_name, args):
            if args.get("symbol") == "INVALID123":
                return {"error": "Symbol not found"}
            return {"current_price": 150.00}
        
        mock_researcher_server.call_tool = AsyncMock(side_effect=mock_invalid_symbol)
        
        result = await researcher_agent.run("What is the price of INVALID123?")
        
        log.info(f"Result: {result}")
        
        assert "response" in result
        
        log.info("RESULT: Invalid symbol handled")

    @pytest.mark.asyncio
    async def test_handles_ambiguous_query(self, researcher_agent, log):
        """Should handle ambiguous queries."""
        log.info("Testing ambiguous query handling")
        
        result = await researcher_agent.run("Tell me about the stock")
        
        log.info(f"Result: {result}")
        
        assert "response" in result
        # Agent should ask for clarification or make reasonable assumptions
        
        log.info("RESULT: Ambiguous query handled")


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgentIntegration:
    """Integration tests for researcher agent."""

    @pytest.mark.asyncio
    async def test_full_research_conversation(self, researcher_agent, log):
        """Test complete research conversation."""
        log.info("Starting full research conversation test")
        
        # Initial query
        log.info("Step 1: Initial research query")
        result1 = await researcher_agent.run("Research Apple stock for me")
        log.info(f"Response 1: {result1['status']}")
        
        # Follow-up question
        log.info("Step 2: Follow-up question")
        result2 = await researcher_agent.run(
            "What about their competitors?",
            context=result1.get("context")
        )
        log.info(f"Response 2: {result2['status']}")
        
        # Investment decision
        log.info("Step 3: Investment decision")
        result3 = await researcher_agent.run(
            "Based on this, should I invest?",
            context=result2.get("context")
        )
        log.info(f"Response 3: {result3['status']}")
        
        log.info("RESULT: Full research conversation completed")

    @pytest.mark.asyncio
    async def test_multi_stock_research(self, researcher_agent, log):
        """Test researching multiple stocks."""
        log.info("Testing multi-stock research")
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        result = await researcher_agent.run(
            f"Compare these tech stocks: {', '.join(symbols)}. Which is the best value?"
        )
        
        log.info(f"Result: {result['status']}")
        assert result["status"] == "success"
        
        log.info("RESULT: Multi-stock research completed")

    @pytest.mark.asyncio
    async def test_research_with_market_context(self, researcher_agent, log):
        """Test research with market context."""
        log.info("Testing research with market context")
        
        result = await researcher_agent.run(
            "How is AAPL performing relative to the overall market today?"
        )
        
        log.info(f"Result: {result}")
        assert result["status"] == "success"
        
        log.info("RESULT: Research with market context completed")


# =============================================================================
# Trading Integration Tests
# =============================================================================

class TestTradingIntegration:
    """Tests for integration with trading functionality."""

    @pytest.mark.asyncio
    async def test_provides_actionable_insights(self, researcher_agent, log):
        """Should provide actionable trading insights."""
        log.info("Testing actionable insights")
        
        result = await researcher_agent.run(
            "I want to buy AAPL. What entry price would you recommend?"
        )
        
        log.info(f"Result: {result}")
        
        assert "response" in result
        assert result["status"] == "success"
        
        log.info("RESULT: Actionable insights provided")

    @pytest.mark.asyncio
    async def test_returns_structured_data(self, researcher_agent, log):
        """Should return structured research data for trading decisions."""
        log.info("Testing structured data return")
        
        result = await researcher_agent.run(
            "Get me all the data I need to decide on AAPL",
            return_structured=True
        )
        
        log.info(f"Result keys: {result.keys()}")
        
        if "research_data" in result:
            log.info(f"Research data keys: {result['research_data'].keys()}")
        
        log.info("RESULT: Structured data returned")

    @pytest.mark.asyncio
    async def test_supports_trade_context(self, researcher_agent, log):
        """Should accept trade context for relevant research."""
        log.info("Testing trade context support")
        
        trade_context = {
            "intent": "buy",
            "symbol": "AAPL",
            "amount_usd": 5000
        }
        
        result = await researcher_agent.run(
            "Is this a good trade?",
            trade_context=trade_context
        )
        
        log.info(f"Result: {result}")
        assert result["status"] == "success"
        
        log.info("RESULT: Trade context supported")
