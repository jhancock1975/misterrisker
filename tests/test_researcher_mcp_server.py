"""
Tests for the Researcher MCP Server.

TDD tests for research capabilities including:
- Finnhub financial data API
- OpenAI web search API
- Alpha Vantage (free tier)
- News aggregation

Tests written before implementation following TDD principles.
"""

import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test logger
logger = logging.getLogger("test.researcher_mcp_server")


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_openai_responses():
    """Create a mock OpenAI responses API."""
    responses = MagicMock()
    
    mock_response = MagicMock()
    mock_response.output = [
        {
            "type": "web_search_call",
            "id": "ws_123",
            "status": "completed"
        },
        {
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Based on recent news, Apple Inc (AAPL) reported strong Q4 earnings...",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "start_index": 0,
                            "end_index": 50,
                            "url": "https://reuters.com/apple-earnings",
                            "title": "Apple Q4 Earnings Report"
                        }
                    ]
                }
            ]
        }
    ]
    mock_response.output_text = "Based on recent news, Apple Inc (AAPL) reported strong Q4 earnings..."
    
    responses.create = MagicMock(return_value=mock_response)
    return responses


@pytest.fixture
def mock_finnhub_client():
    """Create a mock Finnhub API client."""
    client = MagicMock()
    
    # Mock quote data
    client.quote.return_value = {
        "c": 150.25,  # Current price
        "d": 2.50,    # Change
        "dp": 1.69,   # Percent change
        "h": 152.00,  # High
        "l": 148.50,  # Low
        "o": 149.00,  # Open
        "pc": 147.75  # Previous close
    }
    
    # Mock company profile
    client.company_profile2.return_value = {
        "country": "US",
        "currency": "USD",
        "exchange": "NASDAQ",
        "finnhubIndustry": "Technology",
        "ipo": "1980-12-12",
        "logo": "https://example.com/logo.png",
        "marketCapitalization": 2500000,
        "name": "Apple Inc",
        "phone": "14089961010",
        "shareOutstanding": 16000,
        "ticker": "AAPL",
        "weburl": "https://apple.com"
    }
    
    # Mock company news
    client.company_news.return_value = [
        {
            "category": "company",
            "datetime": 1699900000,
            "headline": "Apple Reports Record Q4 Earnings",
            "id": 12345,
            "image": "https://example.com/image.jpg",
            "related": "AAPL",
            "source": "Reuters",
            "summary": "Apple Inc reported record quarterly earnings...",
            "url": "https://example.com/news/12345"
        },
        {
            "category": "company",
            "datetime": 1699800000,
            "headline": "Apple Launches New iPhone",
            "id": 12346,
            "image": "https://example.com/image2.jpg",
            "related": "AAPL",
            "source": "Bloomberg",
            "summary": "Apple unveiled its latest iPhone model...",
            "url": "https://example.com/news/12346"
        }
    ]
    
    # Mock recommendation trends
    client.recommendation_trends.return_value = [
        {
            "buy": 24,
            "hold": 7,
            "period": "2024-01-01",
            "sell": 0,
            "strongBuy": 13,
            "strongSell": 0,
            "symbol": "AAPL"
        }
    ]
    
    # Mock basic financials
    client.company_basic_financials.return_value = {
        "metric": {
            "52WeekHigh": 199.62,
            "52WeekLow": 124.17,
            "peNormalizedAnnual": 28.5,
            "marketCapitalization": 2500000,
            "dividendYieldIndicatedAnnual": 0.52,
            "epsNormalizedAnnual": 6.13
        },
        "metricType": "all",
        "symbol": "AAPL"
    }
    
    # Mock peers
    client.company_peers.return_value = ["MSFT", "GOOGL", "META", "AMZN", "NVDA"]
    
    # Mock earnings
    client.company_earnings.return_value = [
        {
            "actual": 1.88,
            "estimate": 1.75,
            "period": "2024-03-31",
            "quarter": 1,
            "surprise": 0.13,
            "surprisePercent": 7.43,
            "symbol": "AAPL",
            "year": 2024
        }
    ]
    
    # Mock market news
    client.general_news.return_value = [
        {
            "category": "general",
            "datetime": 1699900000,
            "headline": "Markets Rally on Fed Comments",
            "id": 99999,
            "image": "https://example.com/market.jpg",
            "related": "",
            "source": "CNBC",
            "summary": "Stock markets rallied today after Federal Reserve...",
            "url": "https://example.com/news/99999"
        }
    ]
    
    return client


@pytest.fixture
def mock_openai_client(mock_openai_responses):
    """Create a mock OpenAI client for web search."""
    client = MagicMock()
    client.responses = mock_openai_responses
    return client


@pytest.fixture
def researcher_server(mock_finnhub_client, mock_openai_client):
    """Create a Researcher MCP server with mocked clients."""
    from mcp_servers.researcher import ResearcherMCPServer
    
    server = ResearcherMCPServer(
        finnhub_client=mock_finnhub_client,
        openai_client=mock_openai_client
    )
    return server


# =============================================================================
# Server Initialization Tests
# =============================================================================

class TestResearcherServerInitialization:
    """Tests for server initialization."""

    def test_server_initializes_with_api_keys(self, log):
        """Server should initialize with API keys from environment."""
        from mcp_servers.researcher import ResearcherMCPServer
        
        log.info("Testing server initialization with environment API keys")
        
        with patch.dict(os.environ, {
            "FINNHUB_API_KEY": "test_finnhub_key",
            "OPENAI_API_KEY": "test_openai_key"
        }):
            server = ResearcherMCPServer()
            
            log.info(f"Server created: {server}")
            assert server is not None
            log.info("RESULT: Server initialized successfully")

    def test_server_registers_tools(self, researcher_server, log):
        """Server should register all research tools."""
        log.info("Testing tool registration")
        
        tools = researcher_server.list_tools()
        tool_names = [t["name"] for t in tools]
        
        log.info(f"Registered tools: {tool_names}")
        
        expected_tools = [
            "get_stock_quote",
            "get_company_profile",
            "get_company_news",
            "get_market_news",
            "get_analyst_recommendations",
            "get_basic_financials",
            "get_company_peers",
            "get_earnings_history",
            "web_search",
            "research_stock"
        ]
        
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"
        
        log.info(f"RESULT: All {len(expected_tools)} expected tools registered")


# =============================================================================
# Finnhub API Tests
# =============================================================================

class TestFinnhubStockQuote:
    """Tests for stock quote functionality."""

    @pytest.mark.asyncio
    async def test_get_stock_quote(self, researcher_server, mock_finnhub_client, log):
        """Should get real-time stock quote."""
        log.info("Testing get_stock_quote for AAPL")
        
        result = await researcher_server.call_tool("get_stock_quote", {"symbol": "AAPL"})
        
        log.info(f"Result: {result}")
        
        assert "current_price" in result
        assert "change" in result
        assert "change_percent" in result
        assert result["current_price"] == 150.25
        mock_finnhub_client.quote.assert_called_with("AAPL")
        
        log.info("RESULT: Stock quote retrieved successfully")

    @pytest.mark.asyncio
    async def test_get_stock_quote_invalid_symbol(self, researcher_server, mock_finnhub_client, log):
        """Should handle invalid symbol gracefully."""
        log.info("Testing get_stock_quote with invalid symbol")
        
        mock_finnhub_client.quote.return_value = {"c": 0, "d": None, "dp": None}
        
        result = await researcher_server.call_tool("get_stock_quote", {"symbol": "INVALID123"})
        
        log.info(f"Result: {result}")
        
        assert "error" in result or result.get("current_price") == 0
        log.info("RESULT: Invalid symbol handled gracefully")


class TestFinnhubCompanyProfile:
    """Tests for company profile functionality."""

    @pytest.mark.asyncio
    async def test_get_company_profile(self, researcher_server, mock_finnhub_client, log):
        """Should get company profile information."""
        log.info("Testing get_company_profile for AAPL")
        
        result = await researcher_server.call_tool("get_company_profile", {"symbol": "AAPL"})
        
        log.info(f"Result: {result}")
        
        assert "name" in result
        assert "industry" in result
        assert "market_cap" in result
        assert result["name"] == "Apple Inc"
        
        log.info("RESULT: Company profile retrieved successfully")


class TestFinnhubCompanyNews:
    """Tests for company news functionality."""

    @pytest.mark.asyncio
    async def test_get_company_news(self, researcher_server, mock_finnhub_client, log):
        """Should get recent company news."""
        log.info("Testing get_company_news for AAPL")
        
        result = await researcher_server.call_tool("get_company_news", {
            "symbol": "AAPL",
            "days": 7
        })
        
        log.info(f"Result: {result}")
        
        assert "news" in result
        assert len(result["news"]) > 0
        assert "headline" in result["news"][0]
        assert "source" in result["news"][0]
        
        log.info(f"RESULT: Retrieved {len(result['news'])} news articles")

    @pytest.mark.asyncio
    async def test_get_company_news_with_limit(self, researcher_server, mock_finnhub_client, log):
        """Should respect limit parameter."""
        log.info("Testing get_company_news with limit=1")
        
        result = await researcher_server.call_tool("get_company_news", {
            "symbol": "AAPL",
            "limit": 1
        })
        
        log.info(f"Result: {result}")
        
        assert len(result["news"]) <= 1
        log.info("RESULT: News limit respected")


class TestFinnhubAnalystRecommendations:
    """Tests for analyst recommendations functionality."""

    @pytest.mark.asyncio
    async def test_get_analyst_recommendations(self, researcher_server, mock_finnhub_client, log):
        """Should get analyst recommendations."""
        log.info("Testing get_analyst_recommendations for AAPL")
        
        result = await researcher_server.call_tool("get_analyst_recommendations", {"symbol": "AAPL"})
        
        log.info(f"Result: {result}")
        
        assert "recommendations" in result
        assert "buy" in result["recommendations"]
        assert "hold" in result["recommendations"]
        assert "sell" in result["recommendations"]
        
        log.info("RESULT: Analyst recommendations retrieved successfully")


class TestFinnhubBasicFinancials:
    """Tests for basic financials functionality."""

    @pytest.mark.asyncio
    async def test_get_basic_financials(self, researcher_server, mock_finnhub_client, log):
        """Should get basic financial metrics."""
        log.info("Testing get_basic_financials for AAPL")
        
        result = await researcher_server.call_tool("get_basic_financials", {"symbol": "AAPL"})
        
        log.info(f"Result: {result}")
        
        assert "52_week_high" in result
        assert "52_week_low" in result
        assert "pe_ratio" in result
        assert "market_cap" in result
        
        log.info("RESULT: Basic financials retrieved successfully")


class TestFinnhubCompanyPeers:
    """Tests for company peers functionality."""

    @pytest.mark.asyncio
    async def test_get_company_peers(self, researcher_server, mock_finnhub_client, log):
        """Should get company peers/competitors."""
        log.info("Testing get_company_peers for AAPL")
        
        result = await researcher_server.call_tool("get_company_peers", {"symbol": "AAPL"})
        
        log.info(f"Result: {result}")
        
        assert "peers" in result
        assert len(result["peers"]) > 0
        assert "MSFT" in result["peers"]
        
        log.info(f"RESULT: Found {len(result['peers'])} peers")


class TestFinnhubEarnings:
    """Tests for earnings history functionality."""

    @pytest.mark.asyncio
    async def test_get_earnings_history(self, researcher_server, mock_finnhub_client, log):
        """Should get historical earnings data."""
        log.info("Testing get_earnings_history for AAPL")
        
        result = await researcher_server.call_tool("get_earnings_history", {"symbol": "AAPL"})
        
        log.info(f"Result: {result}")
        
        assert "earnings" in result
        assert len(result["earnings"]) > 0
        assert "actual" in result["earnings"][0]
        assert "estimate" in result["earnings"][0]
        assert "surprise_percent" in result["earnings"][0]
        
        log.info("RESULT: Earnings history retrieved successfully")


class TestFinnhubMarketNews:
    """Tests for market news functionality."""

    @pytest.mark.asyncio
    async def test_get_market_news(self, researcher_server, mock_finnhub_client, log):
        """Should get general market news."""
        log.info("Testing get_market_news")
        
        result = await researcher_server.call_tool("get_market_news", {"category": "general"})
        
        log.info(f"Result: {result}")
        
        assert "news" in result
        assert len(result["news"]) > 0
        
        log.info(f"RESULT: Retrieved {len(result['news'])} market news articles")

    @pytest.mark.asyncio
    async def test_get_market_news_categories(self, researcher_server, mock_finnhub_client, log):
        """Should support different news categories."""
        log.info("Testing get_market_news with different categories")
        
        categories = ["general", "forex", "crypto", "merger"]
        
        for category in categories:
            mock_finnhub_client.general_news.return_value = [{"category": category, "headline": f"{category} news"}]
            result = await researcher_server.call_tool("get_market_news", {"category": category})
            log.info(f"Category {category}: {len(result['news'])} articles")
        
        log.info("RESULT: All news categories supported")


# =============================================================================
# OpenAI Web Search Tests
# =============================================================================

class TestOpenAIWebSearch:
    """Tests for OpenAI web search functionality."""

    @pytest.mark.asyncio
    async def test_web_search(self, researcher_server, mock_openai_client, log):
        """Should perform web search and return results with citations."""
        log.info("Testing web_search")
        
        result = await researcher_server.call_tool("web_search", {
            "query": "Apple Inc latest earnings report 2024"
        })
        
        log.info(f"Result: {result}")
        
        assert "answer" in result
        assert "citations" in result
        assert len(result["answer"]) > 0
        
        log.info("RESULT: Web search completed with citations")

    @pytest.mark.asyncio
    async def test_web_search_with_domain_filter(self, researcher_server, mock_openai_client, log):
        """Should support domain filtering for search."""
        log.info("Testing web_search with domain filter")
        
        result = await researcher_server.call_tool("web_search", {
            "query": "Apple stock analysis",
            "domains": ["reuters.com", "bloomberg.com"]
        })
        
        log.info(f"Result: {result}")
        
        assert "answer" in result
        log.info("RESULT: Web search with domain filter completed")

    @pytest.mark.asyncio
    async def test_web_search_financial_focus(self, researcher_server, mock_openai_client, log):
        """Should handle financial/investment queries."""
        log.info("Testing web_search with financial query")
        
        result = await researcher_server.call_tool("web_search", {
            "query": "Is AAPL stock a good buy in 2024?",
            "domains": ["seekingalpha.com", "fool.com", "finance.yahoo.com"]
        })
        
        log.info(f"Result: {result}")
        
        assert "answer" in result
        log.info("RESULT: Financial web search completed")


# =============================================================================
# Combined Research Tests
# =============================================================================

class TestResearchStock:
    """Tests for comprehensive stock research functionality."""

    @pytest.mark.asyncio
    async def test_research_stock_basic(self, researcher_server, log):
        """Should perform comprehensive stock research."""
        log.info("Testing research_stock for AAPL")
        
        result = await researcher_server.call_tool("research_stock", {"symbol": "AAPL"})
        
        log.info(f"Result keys: {result.keys()}")
        
        assert "quote" in result
        assert "profile" in result
        assert "financials" in result
        assert "recommendations" in result
        assert "news" in result
        
        log.info("RESULT: Comprehensive stock research completed")

    @pytest.mark.asyncio
    async def test_research_stock_with_web_search(self, researcher_server, log):
        """Should include web search in research."""
        log.info("Testing research_stock with web search enabled")
        
        result = await researcher_server.call_tool("research_stock", {
            "symbol": "AAPL",
            "include_web_search": True
        })
        
        log.info(f"Result keys: {result.keys()}")
        
        assert "web_insights" in result
        log.info("RESULT: Stock research with web search completed")


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestResearcherErrorHandling:
    """Tests for error handling in researcher server."""

    @pytest.mark.asyncio
    async def test_handles_finnhub_api_error(self, researcher_server, mock_finnhub_client, log):
        """Should handle Finnhub API errors gracefully."""
        from mcp_servers.researcher import ResearcherAPIError
        
        log.info("Testing Finnhub API error handling")
        
        mock_finnhub_client.quote.side_effect = Exception("API rate limit exceeded")
        
        with pytest.raises(ResearcherAPIError):
            await researcher_server.call_tool("get_stock_quote", {"symbol": "AAPL"})
        
        log.info("RESULT: Finnhub API error handled correctly")

    @pytest.mark.asyncio
    async def test_handles_openai_api_error(self, researcher_server, mock_openai_client, log):
        """Should handle OpenAI API errors gracefully."""
        from mcp_servers.researcher import ResearcherAPIError
        
        log.info("Testing OpenAI API error handling")
        
        mock_openai_client.responses.create.side_effect = Exception("API error")
        
        with pytest.raises(ResearcherAPIError):
            await researcher_server.call_tool("web_search", {"query": "test"})
        
        log.info("RESULT: OpenAI API error handled correctly")

    @pytest.mark.asyncio
    async def test_handles_missing_api_key(self, log):
        """Should raise error when API key is missing."""
        from mcp_servers.researcher import ResearcherMCPServer, ResearcherConfigError
        
        log.info("Testing missing API key handling")
        
        with patch.dict(os.environ, {}, clear=True):
            # Remove API keys from environment
            os.environ.pop("FINNHUB_API_KEY", None)
            
            with pytest.raises(ResearcherConfigError):
                ResearcherMCPServer()
        
        log.info("RESULT: Missing API key error raised correctly")

    @pytest.mark.asyncio
    async def test_handles_invalid_tool_name(self, researcher_server, log):
        """Should handle invalid tool name gracefully."""
        from mcp_servers.researcher import ResearcherAPIError
        
        log.info("Testing invalid tool name handling")
        
        with pytest.raises(ResearcherAPIError, match="Unknown tool"):
            await researcher_server.call_tool("invalid_tool", {})
        
        log.info("RESULT: Invalid tool name handled correctly")


# =============================================================================
# Integration Tests
# =============================================================================

class TestResearcherIntegration:
    """Integration tests for researcher server."""

    @pytest.mark.asyncio
    async def test_full_research_workflow(self, researcher_server, log):
        """Test complete research workflow for investment decision."""
        log.info("Starting full research workflow integration test")
        
        # Step 1: Get stock quote
        log.info("Step 1: Getting stock quote")
        quote = await researcher_server.call_tool("get_stock_quote", {"symbol": "AAPL"})
        log.info(f"Quote: ${quote['current_price']}")
        
        # Step 2: Get company profile
        log.info("Step 2: Getting company profile")
        profile = await researcher_server.call_tool("get_company_profile", {"symbol": "AAPL"})
        log.info(f"Company: {profile['name']}")
        
        # Step 3: Get financials
        log.info("Step 3: Getting basic financials")
        financials = await researcher_server.call_tool("get_basic_financials", {"symbol": "AAPL"})
        log.info(f"P/E Ratio: {financials['pe_ratio']}")
        
        # Step 4: Get analyst recommendations
        log.info("Step 4: Getting analyst recommendations")
        recommendations = await researcher_server.call_tool("get_analyst_recommendations", {"symbol": "AAPL"})
        log.info(f"Recommendations: {recommendations['recommendations']}")
        
        # Step 5: Get recent news
        log.info("Step 5: Getting recent news")
        news = await researcher_server.call_tool("get_company_news", {"symbol": "AAPL", "limit": 5})
        log.info(f"News articles: {len(news['news'])}")
        
        # Step 6: Web search for additional insights
        log.info("Step 6: Performing web search")
        web_results = await researcher_server.call_tool("web_search", {
            "query": "AAPL stock outlook 2024"
        })
        log.info(f"Web search answer length: {len(web_results['answer'])}")
        
        log.info("RESULT: Full research workflow completed successfully")

    @pytest.mark.asyncio
    async def test_compare_stocks_workflow(self, researcher_server, mock_finnhub_client, log):
        """Test comparing multiple stocks."""
        log.info("Starting stock comparison workflow")
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        comparison = {}
        
        for symbol in symbols:
            log.info(f"Researching {symbol}")
            
            # Update mock for each symbol
            mock_finnhub_client.quote.return_value["c"] = 150 + (symbols.index(symbol) * 10)
            
            quote = await researcher_server.call_tool("get_stock_quote", {"symbol": symbol})
            peers = await researcher_server.call_tool("get_company_peers", {"symbol": symbol})
            
            comparison[symbol] = {
                "price": quote["current_price"],
                "peer_count": len(peers["peers"])
            }
        
        log.info(f"Comparison results: {comparison}")
        
        assert len(comparison) == 3
        log.info("RESULT: Stock comparison workflow completed")
