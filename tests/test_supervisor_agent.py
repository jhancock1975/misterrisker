"""Tests for the Supervisor Agent that routes to sub-agents."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.supervisor_agent import SupervisorAgent, RoutingDecision


class TestSupervisorAgentRouting:
    """Tests for supervisor agent routing decisions."""

    @pytest.mark.asyncio
    async def test_supervisor_routes_schwab_query_to_schwab_agent(self):
        """Supervisor should route Schwab-related queries to SchwabAgent."""
        # Create mock LLM that returns routing decision
        mock_llm = MagicMock()
        mock_router = MagicMock()
        mock_router.ainvoke = AsyncMock(return_value=RoutingDecision(
            agent="schwab",
            reasoning="User is asking about Schwab balance",
            query_for_agent="what's my schwab balance?"
        ))
        mock_llm.with_structured_output = MagicMock(return_value=mock_router)
        
        # Create mock Schwab agent with process_query method
        mock_schwab_agent = MagicMock()
        mock_schwab_agent.process_query = AsyncMock(return_value={
            "response": "Your Schwab balance is $10,000",
            "status": "success"
        })
        
        supervisor = SupervisorAgent(
            llm=mock_llm,
            schwab_agent=mock_schwab_agent
        )
        
        result = await supervisor.execute(
            user_message="what's my schwab balance?",
            conversation_history=[],
            config={"configurable": {"thread_id": "test-123"}}
        )
        
        # Should have called the Schwab agent
        assert result["agent_used"] == "schwab"
        assert "schwab balance" in result["response"].lower() or "error" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_supervisor_routes_coinbase_query_to_coinbase_agent(self):
        """Supervisor should route Coinbase-related queries to CoinbaseAgent."""
        mock_llm = MagicMock()
        mock_router = MagicMock()
        mock_router.ainvoke = AsyncMock(return_value=RoutingDecision(
            agent="coinbase",
            reasoning="User is asking about Bitcoin price",
            query_for_agent="what's the price of bitcoin?"
        ))
        mock_llm.with_structured_output = MagicMock(return_value=mock_router)
        
        mock_coinbase_agent = MagicMock()
        mock_coinbase_agent.process_query = AsyncMock(return_value={
            "response": "Bitcoin is currently at $100,000",
            "status": "success"
        })
        
        supervisor = SupervisorAgent(
            llm=mock_llm,
            coinbase_agent=mock_coinbase_agent
        )
        
        result = await supervisor.execute(
            user_message="what's the price of bitcoin?",
            conversation_history=[],
            config={"configurable": {"thread_id": "test-123"}}
        )
        
        assert result["agent_used"] == "coinbase"

    @pytest.mark.asyncio
    async def test_supervisor_routes_web_search_to_researcher_agent(self):
        """Supervisor should route web search/internet queries to ResearcherAgent."""
        mock_llm = MagicMock()
        mock_router = MagicMock()
        mock_router.ainvoke = AsyncMock(return_value=RoutingDecision(
            agent="researcher",
            reasoning="User wants to search the internet for weather information",
            query_for_agent="search the internet and tell me the weather report for Taipei, Taiwan today"
        ))
        mock_llm.with_structured_output = MagicMock(return_value=mock_router)
        
        mock_researcher_agent = MagicMock()
        mock_researcher_agent.run = AsyncMock(return_value={
            "response": "The weather in Taipei today is 22°C with partly cloudy skies.",
            "status": "success"
        })
        
        supervisor = SupervisorAgent(
            llm=mock_llm,
            researcher_agent=mock_researcher_agent
        )
        
        result = await supervisor.execute(
            user_message="search the internet, and tell me the weather report for Taipei, Taiwan today",
            conversation_history=[],
            config={"configurable": {"thread_id": "test-123"}}
        )
        
        # Should have routed to researcher, not direct
        assert result["agent_used"] == "researcher", \
            f"Expected 'researcher' but got '{result['agent_used']}'. Web search queries should go to researcher agent."
        assert "weather" in result["response"].lower() or "taipei" in result["response"].lower() or "error" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_supervisor_routes_general_question_to_researcher_agent(self):
        """Supervisor should route general knowledge questions to ResearcherAgent."""
        mock_llm = MagicMock()
        mock_router = MagicMock()
        mock_router.ainvoke = AsyncMock(return_value=RoutingDecision(
            agent="researcher",
            reasoning="User is asking a general knowledge question that requires web search",
            query_for_agent="What are the latest news headlines today?"
        ))
        mock_llm.with_structured_output = MagicMock(return_value=mock_router)
        
        mock_researcher_agent = MagicMock()
        mock_researcher_agent.run = AsyncMock(return_value={
            "response": "Here are today's top headlines: ...",
            "status": "success"
        })
        
        supervisor = SupervisorAgent(
            llm=mock_llm,
            researcher_agent=mock_researcher_agent
        )
        
        result = await supervisor.execute(
            user_message="What are the latest news headlines today?",
            conversation_history=[],
            config={"configurable": {"thread_id": "test-123"}}
        )
        
        assert result["agent_used"] == "researcher", \
            f"Expected 'researcher' but got '{result['agent_used']}'. General questions should go to researcher agent."

    @pytest.mark.asyncio
    async def test_supervisor_routes_blockchain_data_query_to_researcher_agent(self):
        """Supervisor should route blockchain data queries to ResearcherAgent, not Coinbase.
        
        Blockchain data queries like 'get the last 10 transactions on the bitcoin blockchain'
        require external information/web search, so should go to researcher, not coinbase.
        Coinbase agent is for trading operations on the user's account, not blockchain exploration.
        """
        mock_llm = MagicMock()
        mock_router = MagicMock()
        mock_router.ainvoke = AsyncMock(return_value=RoutingDecision(
            agent="researcher",
            reasoning="User is asking for blockchain transaction data which requires external lookup",
            query_for_agent="get the list of the last ten transactions on the bitcoin blockchain"
        ))
        mock_llm.with_structured_output = MagicMock(return_value=mock_router)
        
        mock_researcher_agent = MagicMock()
        mock_researcher_agent.run = AsyncMock(return_value={
            "response": "Here are the latest Bitcoin transactions from blockchain explorers...",
            "status": "success"
        })
        
        supervisor = SupervisorAgent(
            llm=mock_llm,
            researcher_agent=mock_researcher_agent
        )
        
        result = await supervisor.execute(
            user_message="get the list of the last ten transactions on the bitcoin blockchain",
            conversation_history=[],
            config={"configurable": {"thread_id": "test-123"}}
        )
        
        assert result["agent_used"] == "researcher", \
            f"Expected 'researcher' but got '{result['agent_used']}'. Blockchain data queries should go to researcher agent, not coinbase."

    @pytest.mark.asyncio
    async def test_supervisor_routes_stock_chart_to_schwab_not_coinbase(self):
        """Supervisor should route stock chart requests (NVDA, AAPL) to Schwab, not Coinbase.
        
        When user asks to 'draw a chart of NVDA prices', the system should:
        1. Recognize NVDA as a stock (not crypto)
        2. Route to Schwab agent (has stock market data)
        3. NOT route to Coinbase (only has crypto data)
        """
        mock_llm = MagicMock()
        mock_router = MagicMock()
        mock_router.ainvoke = AsyncMock(return_value=RoutingDecision(
            agent="schwab",
            reasoning="NVDA is a stock ticker. Stock charts should go to Schwab which has stock price data.",
            query_for_agent="draw a chart of NVDA prices from december 12th"
        ))
        mock_llm.with_structured_output = MagicMock(return_value=mock_router)
        
        mock_schwab_agent = MagicMock()
        mock_schwab_agent.process_query = AsyncMock(return_value={
            "response": "📊 **Price Chart: NVDA**\n<div class='generated-chart'>...</div>",
            "status": "success"
        })
        
        # Also provide a mock coinbase agent to ensure it's NOT used
        mock_coinbase_agent = MagicMock()
        mock_coinbase_agent.process_query = AsyncMock(return_value={
            "response": "Crypto chart",
            "status": "success"
        })
        
        supervisor = SupervisorAgent(
            llm=mock_llm,
            schwab_agent=mock_schwab_agent,
            coinbase_agent=mock_coinbase_agent
        )
        
        result = await supervisor.execute(
            user_message="can you draw a chart of nvda prices from december 12th?",
            conversation_history=[],
            config={"configurable": {"thread_id": "test-123"}}
        )
        
        # Should route to Schwab, NOT Coinbase
        assert result["agent_used"] == "schwab", \
            f"Expected 'schwab' but got '{result['agent_used']}'. Stock chart requests (NVDA) should go to Schwab, not Coinbase."
        
        # Schwab agent should have been called
        mock_schwab_agent.process_query.assert_called()
        # Coinbase agent should NOT have been called
        mock_coinbase_agent.process_query.assert_not_called()


class TestSchwabAgentProcessQuery:
    """Tests for SchwabAgent.process_query method."""

    @pytest.mark.asyncio
    async def test_schwab_agent_has_process_query_method(self):
        """SchwabAgent should have a process_query method for supervisor integration."""
        from src.agents.schwab_agent import SchwabAgent
        
        # Create agent with mocked server
        agent = SchwabAgent(
            api_key="test_key",
            app_secret="test_secret",
            callback_url="https://localhost:8443/callback"
        )
        
        assert hasattr(agent, 'process_query'), \
            "SchwabAgent must have process_query method for supervisor integration"
        
        # Check that it's callable and async
        import inspect
        assert callable(agent.process_query)
        assert inspect.iscoroutinefunction(agent.process_query), \
            "process_query must be an async method"

    @pytest.mark.asyncio
    async def test_schwab_agent_process_query_accepts_query_and_config(self):
        """SchwabAgent.process_query should accept query string and optional config."""
        from src.agents.schwab_agent import SchwabAgent
        import inspect
        
        agent = SchwabAgent(
            api_key="test_key",
            app_secret="test_secret",
            callback_url="https://localhost:8443/callback"
        )
        
        sig = inspect.signature(agent.process_query)
        params = list(sig.parameters.keys())
        
        assert 'query' in params, "process_query must accept 'query' parameter"
        # config should be optional
        assert 'config' in params, "process_query should accept 'config' parameter"


class TestCoinbaseAgentProcessQuery:
    """Tests for CoinbaseAgent.process_query method."""

    @pytest.mark.asyncio
    async def test_coinbase_agent_has_process_query_method(self):
        """CoinbaseAgent should have a process_query method for supervisor integration."""
        from src.agents.coinbase_agent import CoinbaseAgent
        
        agent = CoinbaseAgent(
            api_key="test_key",
            api_secret="test_secret"
        )
        
        assert hasattr(agent, 'process_query'), \
            "CoinbaseAgent must have process_query method for supervisor integration"
        
        import inspect
        assert callable(agent.process_query)
        assert inspect.iscoroutinefunction(agent.process_query), \
            "process_query must be an async method"

    @pytest.mark.asyncio
    async def test_coinbase_agent_process_query_accepts_query_and_config(self):
        """CoinbaseAgent.process_query should accept query string and optional config."""
        from src.agents.coinbase_agent import CoinbaseAgent
        import inspect
        
        agent = CoinbaseAgent(
            api_key="test_key",
            api_secret="test_secret"
        )
        
        sig = inspect.signature(agent.process_query)
        params = list(sig.parameters.keys())
        
        assert 'query' in params, "process_query must accept 'query' parameter"
        assert 'config' in params, "process_query should accept 'config' parameter"
