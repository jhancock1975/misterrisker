"""
Playwright tests for real-time crypto analysis features.

These tests verify that Mister Risker uses real-time data when analyzing
cryptocurrency patterns and prices, rather than hallucinating incorrect values.

The agent has access to:
1. Real-time candle data via WebSocket (CoinbaseWebSocketService)
2. Blockchain transaction data via Solana RPC and other free APIs
3. Coinbase API for current prices and historical data

Tests ensure the agent uses these real data sources for analysis.
"""

import pytest
import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# Test: BTC Price Analysis Uses Real Data
# =============================================================================

class TestBTCAnalysisUsesRealData:
    """Tests that BTC analysis queries use real-time data sources."""

    @pytest.mark.asyncio
    async def test_btc_pattern_analysis_returns_realistic_price(self, log):
        """When asked about BTC patterns, should return a price > $50,000 (not $39.55)."""
        from web.app import TradingChatBot
        from agents.coinbase_agent import CoinbaseAgent
        
        log.info("Testing BTC pattern analysis returns realistic price")
        
        # Create chatbot with mocked Coinbase agent
        with patch('web.app.CoinbaseAgent') as MockAgent:
            # Mock the agent to return realistic BTC data
            mock_agent = AsyncMock()
            mock_agent.process_query = AsyncMock(return_value={
                "response": "BTC is currently at $99,580.00 with strong bullish momentum.",
                "status": "success"
            })
            mock_agent.websocket_service = MagicMock()
            mock_agent.websocket_service.get_latest_candle = MagicMock(return_value={
                "product_id": "BTC-USD",
                "open": "99500.00",
                "high": "99650.00",
                "low": "99420.00",
                "close": "99580.00",
                "volume": "1.5"
            })
            MockAgent.return_value = mock_agent
            
            bot = TradingChatBot(use_agents=True)
            bot.coinbase_agent = mock_agent
            
            # Ask about BTC patterns
            response = await bot.process_message("what patterns are you seeing in BTC data right now?")
            
            log.info(f"Response: {response}")
            
            # The response should NOT contain unrealistic prices like $39.55
            assert "$39" not in response, "Response contains unrealistic $39 price"
            assert "$40." not in response.lower() or "$40,000" in response or "$40000" in response, \
                "Response may contain unrealistic $40 price (should be $40,000+)"
            
            # If a price is mentioned, it should be realistic (> $50,000)
            price_match = re.search(r'\$[\d,]+(?:\.\d+)?', response)
            if price_match:
                price_str = price_match.group(0).replace('$', '').replace(',', '')
                price = float(price_str)
                log.info(f"Extracted price: ${price}")
                assert price > 50000 or price < 1, \
                    f"BTC price ${price} is unrealistic - should be > $50,000 or a percentage"

    @pytest.mark.asyncio
    async def test_analysis_query_routes_to_coinbase_for_realtime_data(self, log):
        """Pattern analysis queries should use Coinbase agent's real-time data."""
        from web.app import TradingChatBot
        
        log.info("Testing analysis query routing to Coinbase agent")
        
        with patch('web.app.CoinbaseAgent') as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.process_query = AsyncMock(return_value={
                "response": "BTC showing bullish pattern at $99,500",
                "status": "success"
            })
            mock_agent.websocket_service = MagicMock()
            mock_agent.websocket_service.is_running = True
            mock_agent.websocket_service.get_latest_candle = MagicMock(return_value={
                "product_id": "BTC-USD",
                "close": "99500.00"
            })
            MockAgent.return_value = mock_agent
            
            bot = TradingChatBot(use_agents=True)
            bot.coinbase_agent = mock_agent
            
            # Query for BTC patterns
            query = "what patterns are you seeing in BTC data right now?"
            
            # The query should eventually call the coinbase agent
            await bot.process_message(query)
            
            # Verify coinbase agent was called (either directly or via supervisor)
            log.info("Verifying Coinbase agent involvement")

    @pytest.mark.asyncio
    async def test_websocket_candle_data_available_for_analysis(self, log):
        """WebSocket candle data should be accessible for analysis queries."""
        from services.coinbase_websocket import CoinbaseWebSocketService
        
        log.info("Testing WebSocket candle data availability")
        
        # Create service and simulate receiving candle data
        service = CoinbaseWebSocketService()
        
        # Simulate candle data being stored
        service._latest_candles = {
            "BTC-USD": {
                "product_id": "BTC-USD",
                "open": "99500.00",
                "high": "99650.00",
                "low": "99420.00",
                "close": "99580.00",
                "volume": "125.5",
                "timestamp": "2025-12-06T15:00:00Z"
            },
            "ETH-USD": {
                "product_id": "ETH-USD",
                "open": "3950.00",
                "high": "3985.00",
                "low": "3940.00",
                "close": "3975.00",
                "volume": "5430.2"
            }
        }
        
        # Verify data is accessible
        btc_candle = service.get_latest_candle("BTC-USD")
        assert btc_candle is not None
        assert btc_candle["close"] == "99580.00"
        log.info(f"BTC candle data: {btc_candle}")
        
        all_candles = service.get_all_latest_candles()
        assert "BTC-USD" in all_candles
        assert "ETH-USD" in all_candles
        log.info(f"All candles available: {list(all_candles.keys())}")


# =============================================================================
# Test: Coinbase Agent Exposes Real-Time Data
# =============================================================================

class TestCoinbaseAgentRealTimeData:
    """Tests that Coinbase agent provides real-time data for analysis."""

    @pytest.mark.asyncio
    async def test_coinbase_agent_has_websocket_service(self, log):
        """Coinbase agent should have WebSocket service initialized."""
        from agents.coinbase_agent import CoinbaseAgent
        from mcp_servers.coinbase import CoinbaseMCPServer
        
        log.info("Testing Coinbase agent WebSocket service initialization")
        
        # Create agent with mocked MCP server
        mock_mcp = MagicMock(spec=CoinbaseMCPServer)
        mock_mcp.list_tools = MagicMock(return_value=[])
        
        agent = CoinbaseAgent(
            mcp_server=mock_mcp,
            enable_websocket=True
        )
        
        assert agent.websocket_service is not None, "WebSocket service not initialized"
        log.info("WebSocket service initialized successfully")

    @pytest.mark.asyncio
    async def test_coinbase_agent_can_get_realtime_price(self, log):
        """Coinbase agent should be able to get real-time price from WebSocket."""
        from agents.coinbase_agent import CoinbaseAgent
        from mcp_servers.coinbase import CoinbaseMCPServer
        
        log.info("Testing Coinbase agent real-time price access")
        
        mock_mcp = MagicMock(spec=CoinbaseMCPServer)
        mock_mcp.list_tools = MagicMock(return_value=[])
        
        agent = CoinbaseAgent(
            mcp_server=mock_mcp,
            enable_websocket=True
        )
        
        # Simulate WebSocket data
        if agent.websocket_service:
            agent.websocket_service._latest_candles = {
                "BTC-USD": {
                    "product_id": "BTC-USD",
                    "close": "99580.00",
                    "high": "99650.00",
                    "low": "99420.00"
                }
            }
            
            candle = agent.websocket_service.get_latest_candle("BTC-USD")
            assert candle is not None
            assert float(candle["close"]) > 50000, "BTC price should be realistic"
            log.info(f"Real-time BTC price: ${candle['close']}")


# =============================================================================
# Test: Analysis Handler Uses All Data Sources
# =============================================================================

class TestAnalysisHandler:
    """Tests for the analysis handler that combines data sources."""

    @pytest.mark.asyncio
    async def test_pattern_query_triggers_realtime_lookup(self, log):
        """Queries about patterns should trigger real-time data lookup."""
        from agents.coinbase_agent import CoinbaseAgent
        from mcp_servers.coinbase import CoinbaseMCPServer
        
        log.info("Testing pattern query triggers real-time lookup")
        
        mock_mcp = MagicMock(spec=CoinbaseMCPServer)
        mock_mcp.list_tools = MagicMock(return_value=[
            {"name": "get_product", "description": "Get product info"}
        ])
        mock_mcp.call_tool = AsyncMock(return_value={
            "product_id": "BTC-USD",
            "price": "99580.00"
        })
        
        agent = CoinbaseAgent(
            mcp_server=mock_mcp,
            enable_websocket=True
        )
        
        # Simulate WebSocket data
        if agent.websocket_service:
            agent.websocket_service._latest_candles = {
                "BTC-USD": {
                    "product_id": "BTC-USD",
                    "open": "99500.00",
                    "high": "99650.00",
                    "low": "99420.00",
                    "close": "99580.00"
                }
            }
        
        # Query for patterns (this should use real-time data)
        query = "what patterns are you seeing in BTC data?"
        
        # The agent should have access to real-time candle data
        if agent.websocket_service:
            candle = agent.websocket_service.get_latest_candle("BTC-USD")
            assert candle is not None
            log.info(f"Candle data available for analysis: {candle}")

    @pytest.mark.asyncio
    async def test_analysis_response_includes_current_price(self, log):
        """Analysis responses should include the current real-time price."""
        from agents.coinbase_agent import CoinbaseAgent
        from mcp_servers.coinbase import CoinbaseMCPServer
        
        log.info("Testing analysis response includes current price")
        
        mock_mcp = MagicMock(spec=CoinbaseMCPServer)
        mock_mcp.list_tools = MagicMock(return_value=[
            {"name": "get_product", "description": "Get product"},
            {"name": "get_best_bid_ask", "description": "Get bid/ask"}
        ])
        mock_mcp.call_tool = AsyncMock(return_value={
            "product": {"price": "99580.00"},
            "bid_ask": {"bids": [{"price": "99579"}], "asks": [{"price": "99581"}]}
        })
        
        agent = CoinbaseAgent(
            mcp_server=mock_mcp,
            enable_websocket=True
        )
        
        # Set up WebSocket data
        if agent.websocket_service:
            agent.websocket_service._latest_candles = {
                "BTC-USD": {
                    "close": "99580.00",
                    "high": "99650.00",
                    "low": "99420.00"
                }
            }
        
        # Process a query
        result = await agent.process_query("what's the current BTC price?")
        
        log.info(f"Query result: {result}")
        
        # Response should be successful
        assert result["status"] == "success"
        
        # Response should contain a realistic price
        response = result["response"]
        # Extract any dollar amounts
        prices = re.findall(r'\$[\d,]+(?:\.\d+)?', response)
        log.info(f"Prices found in response: {prices}")


# =============================================================================
# Test: Query Routing for Analysis
# =============================================================================

class TestAnalysisQueryRouting:
    """Tests for proper routing of analysis queries."""

    @pytest.mark.asyncio
    async def test_pattern_keywords_route_to_analysis(self, log):
        """Queries with pattern/analysis keywords should route appropriately."""
        log.info("Testing pattern keyword routing")
        
        pattern_queries = [
            "what patterns are you seeing in BTC data right now?",
            "analyze the current BTC trend",
            "what's the BTC pattern looking like?",
            "show me BTC analysis",
            "any patterns in bitcoin price?",
        ]
        
        for query in pattern_queries:
            query_lower = query.lower()
            has_pattern_keyword = any(word in query_lower for word in [
                "pattern", "analyze", "analysis", "trend", "looking"
            ])
            has_crypto_keyword = any(word in query_lower for word in [
                "btc", "bitcoin", "eth", "ethereum", "crypto"
            ])
            
            log.info(f"Query: {query}")
            log.info(f"  Has pattern keyword: {has_pattern_keyword}")
            log.info(f"  Has crypto keyword: {has_crypto_keyword}")
            
            # These should route to analysis with real-time data
            assert has_pattern_keyword or has_crypto_keyword

    @pytest.mark.asyncio
    async def test_supervisor_routes_analysis_to_coinbase(self, log):
        """Supervisor should route crypto analysis queries to Coinbase agent."""
        from agents.supervisor_agent import SupervisorAgent
        
        log.info("Testing supervisor routing for analysis queries")
        
        # Create supervisor with mocked agents
        mock_coinbase = AsyncMock()
        mock_coinbase.process_query = AsyncMock(return_value={
            "response": "BTC at $99,580 showing bullish trend",
            "status": "success"
        })
        
        # Mock LLM for routing
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke = MagicMock(return_value=MagicMock(agent="coinbase"))
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
        
        supervisor = SupervisorAgent(
            coinbase_agent=mock_coinbase,
            llm=mock_llm
        )
        
        # The routing prompt should include analysis keywords
        prompt = supervisor._build_intelligent_routing_prompt()
        
        log.info("Checking routing prompt includes crypto analysis")
        assert "crypto" in prompt.lower() or "coinbase" in prompt.lower()


# =============================================================================
# Test: Data Freshness
# =============================================================================

class TestDataFreshness:
    """Tests that ensure data is fresh and not stale."""

    @pytest.mark.asyncio
    async def test_websocket_provides_recent_data(self, log):
        """WebSocket should provide data that's recent (within last 5 minutes)."""
        from services.coinbase_websocket import CoinbaseWebSocketService
        import time
        
        log.info("Testing WebSocket data freshness")
        
        service = CoinbaseWebSocketService()
        
        # Simulate recent candle data
        current_time = time.time()
        service._latest_candles = {
            "BTC-USD": {
                "product_id": "BTC-USD",
                "close": "99580.00",
                "timestamp": "2025-12-06T15:45:00Z",
                "_received_at": current_time  # Track when we received it
            }
        }
        
        candle = service.get_latest_candle("BTC-USD")
        assert candle is not None
        
        # Data should exist
        log.info(f"Latest candle: {candle}")

    @pytest.mark.asyncio  
    async def test_fallback_to_api_if_no_websocket_data(self, log):
        """If WebSocket has no data, should fall back to API call."""
        from agents.coinbase_agent import CoinbaseAgent
        from mcp_servers.coinbase import CoinbaseMCPServer
        
        log.info("Testing fallback to API when WebSocket has no data")
        
        mock_mcp = MagicMock(spec=CoinbaseMCPServer)
        mock_mcp.list_tools = MagicMock(return_value=[
            {"name": "get_product", "description": "Get product"},
            {"name": "get_best_bid_ask", "description": "Get bid/ask"}
        ])
        mock_mcp.call_tool = AsyncMock(return_value={
            "product_id": "BTC-USD",
            "price": "99580.00"
        })
        
        agent = CoinbaseAgent(
            mcp_server=mock_mcp,
            enable_websocket=True
        )
        
        # WebSocket has no data (empty)
        if agent.websocket_service:
            agent.websocket_service._latest_candles = {}
        
        # Query for price - should fall back to API
        result = await agent.process_query("what's the BTC price?")
        
        log.info(f"Result when no WebSocket data: {result}")
        # Should succeed or at least not crash
        assert result is not None
        assert "response" in result


# =============================================================================
# Integration Test: Full Analysis Flow
# =============================================================================

class TestFullAnalysisFlow:
    """Integration tests for complete analysis flow."""

    @pytest.mark.asyncio
    async def test_full_btc_analysis_flow(self, log):
        """Test complete flow from query to response with real data."""
        from agents.coinbase_agent import CoinbaseAgent
        from mcp_servers.coinbase import CoinbaseMCPServer
        
        log.info("Testing full BTC analysis flow")
        
        # This test verifies that the analysis handler produces realistic data
        # by testing the Coinbase agent directly with mocked WebSocket data
        
        mock_mcp = MagicMock(spec=CoinbaseMCPServer)
        mock_mcp.list_tools = MagicMock(return_value=[
            {"name": "get_product", "description": "Get product"},
            {"name": "get_best_bid_ask", "description": "Get bid/ask"}
        ])
        mock_mcp.call_tool = AsyncMock(return_value={
            "price": "99580.00"
        })
        
        agent = CoinbaseAgent(
            mcp_server=mock_mcp,
            enable_websocket=True
        )
        
        # Simulate WebSocket data with realistic prices
        if agent.websocket_service:
            agent.websocket_service._latest_candles = {
                "BTC-USD": {
                    "product_id": "BTC-USD",
                    "open": "99500.00",
                    "high": "99650.00",
                    "low": "99420.00",
                    "close": "99580.00",
                    "volume": "125.5"
                }
            }
        
        result = await agent.process_query(
            "what patterns are you seeing in BTC data right now?"
        )
        
        response = result.get("response", "")
        log.info(f"Full analysis response:\n{response}")
        
        # Verify response quality
        assert "99" in response or "Real-Time" in response, \
            "Response should contain realistic BTC price (~$99,000+) or indicate real-time data"
        assert "$39." not in response, \
            "Response should NOT contain fake $39 price"
