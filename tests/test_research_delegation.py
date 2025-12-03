"""Integration tests for research delegation from TradingChatBot to ResearcherAgent.

These tests verify that Mister Risker properly delegates research queries
to the Researcher Agent, which in turn uses web_search and finnhub tools
to gather information.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import logging

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test")


class TestContentExtraction:
    """Test that OpenAI Responses API content is properly extracted."""
    
    def test_extract_string_content(self):
        """String content should be returned as-is."""
        from src.agents.researcher_agent import _extract_content
        
        assert _extract_content("Hello world") == "Hello world"
    
    def test_extract_list_content_with_text_blocks(self):
        """List content with text blocks should be concatenated."""
        from src.agents.researcher_agent import _extract_content
        
        content = [
            {"type": "text", "text": "Here is the news: "},
            {"type": "text", "text": "Markets are up today."}
        ]
        assert _extract_content(content) == "Here is the news: Markets are up today."
    
    def test_extract_mixed_list_content(self):
        """List with mixed content types should extract text only."""
        from src.agents.researcher_agent import _extract_content
        
        content = [
            {"type": "text", "text": "Summary: "},
            {"type": "image", "url": "http://example.com/img.png"},
            {"type": "text", "text": "End of report."}
        ]
        assert _extract_content(content) == "Summary: End of report."
    
    def test_extract_none_content(self):
        """None content should return empty string."""
        from src.agents.researcher_agent import _extract_content
        
        assert _extract_content(None) == ""
    
    def test_extract_other_content(self):
        """Other content types should be converted to string."""
        from src.agents.researcher_agent import _extract_content
        
        assert _extract_content(123) == "123"


class TestResearchQueryDetection:
    """Test that research queries are correctly detected."""
    
    def test_latest_news_is_research_query(self):
        """'tell me the latest news' should be detected as a research query."""
        from src.web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True, enable_chain_of_thought=True)
        
        assert bot._is_research_query("tell me the latest news") is True
        assert bot._is_research_query("what's the latest news") is True
        assert bot._is_research_query("show me recent news") is True
    
    def test_news_about_stock_is_research_query(self):
        """News queries about stocks should be research queries."""
        from src.web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True)
        
        assert bot._is_research_query("what's the news on AAPL") is True
        assert bot._is_research_query("TSLA news today") is True
        assert bot._is_research_query("tell me about Tesla news") is True
    
    def test_analysis_is_research_query(self):
        """Analysis queries should be research queries."""
        from src.web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True)
        
        assert bot._is_research_query("analyze AAPL") is True
        assert bot._is_research_query("what do you think about Tesla") is True
        assert bot._is_research_query("should I buy NVDA") is True
    
    def test_balance_is_not_research_query(self):
        """Balance queries should NOT be research queries."""
        from src.web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True)
        
        assert bot._is_research_query("what is my balance") is False
        assert bot._is_research_query("show my portfolio") is False
        assert bot._is_research_query("buy 100 AAPL") is False


class TestResearcherAgentAnalyzeQuery:
    """Test that ResearcherAgent properly analyzes queries and calls tools."""
    
    def test_latest_news_triggers_market_news_tool(self):
        """'tell me the latest news' should trigger get_market_news tool."""
        from src.agents.researcher_agent import ResearcherAgent, ResearcherAgentState
        
        agent = ResearcherAgent(
            finnhub_api_key="test_key",
            openai_api_key="test_key"
        )
        
        state: ResearcherAgentState = {
            "query": "tell me the latest news",
            "messages": [],
            "tool_calls": [],
            "tool_results": [],
            "research_data": {},
            "response": "",
            "context": {},
            "history": [],
            "status": "pending"
        }
        
        result = agent._analyze_query(state)
        
        # Should have tool calls
        assert len(result["tool_calls"]) > 0, "Should have at least one tool call"
        
        # Should include get_market_news
        tool_names = [tc["tool"] for tc in result["tool_calls"]]
        assert "get_market_news" in tool_names, f"Expected get_market_news in {tool_names}"
    
    def test_latest_news_triggers_web_search_tool(self):
        """'tell me the latest news' should also trigger web_search tool."""
        from src.agents.researcher_agent import ResearcherAgent, ResearcherAgentState
        
        agent = ResearcherAgent(
            finnhub_api_key="test_key",
            openai_api_key="test_key"
        )
        
        state: ResearcherAgentState = {
            "query": "tell me the latest news",
            "messages": [],
            "tool_calls": [],
            "tool_results": [],
            "research_data": {},
            "response": "",
            "context": {},
            "history": [],
            "status": "pending"
        }
        
        result = agent._analyze_query(state)
        
        tool_names = [tc["tool"] for tc in result["tool_calls"]]
        assert "web_search" in tool_names, f"Expected web_search in {tool_names}"
    
    def test_stock_news_triggers_company_news_tool(self):
        """'AAPL news' should trigger get_company_news tool."""
        from src.agents.researcher_agent import ResearcherAgent, ResearcherAgentState
        
        agent = ResearcherAgent(
            finnhub_api_key="test_key",
            openai_api_key="test_key"
        )
        
        state: ResearcherAgentState = {
            "query": "what's the latest news on AAPL",
            "messages": [],
            "tool_calls": [],
            "tool_results": [],
            "research_data": {},
            "response": "",
            "context": {},
            "history": [],
            "status": "pending"
        }
        
        result = agent._analyze_query(state)
        
        tool_names = [tc["tool"] for tc in result["tool_calls"]]
        assert "get_company_news" in tool_names, f"Expected get_company_news in {tool_names}"
    
    def test_what_happened_triggers_news_tools(self):
        """'what happened in the news' should trigger news tools."""
        from src.agents.researcher_agent import ResearcherAgent, ResearcherAgentState
        
        agent = ResearcherAgent(
            finnhub_api_key="test_key",
            openai_api_key="test_key"
        )
        
        state: ResearcherAgentState = {
            "query": "what happened in the news last night",
            "messages": [],
            "tool_calls": [],
            "tool_results": [],
            "research_data": {},
            "response": "",
            "context": {},
            "history": [],
            "status": "pending"
        }
        
        result = agent._analyze_query(state)
        
        assert len(result["tool_calls"]) > 0, "Should have tool calls"
        tool_names = [tc["tool"] for tc in result["tool_calls"]]
        # Should have either market news or web search
        assert "get_market_news" in tool_names or "web_search" in tool_names, \
            f"Expected news tools in {tool_names}"


class TestResearchDelegationIntegration:
    """Integration tests for full research delegation flow."""
    
    @pytest.mark.asyncio
    async def test_chatbot_delegates_news_query_to_researcher(self):
        """TradingChatBot should delegate 'latest news' to ResearcherAgent."""
        from src.web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True, enable_chain_of_thought=True)
        
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=MagicMock(content="Here's the news summary."))
        bot.llm = mock_llm
        
        # Mock ResearcherAgent with controlled response
        mock_researcher = AsyncMock()
        mock_researcher.run = AsyncMock(return_value={
            "status": "success",
            "response": "Breaking news: Markets are up today. Tech stocks leading gains.",
            "research_data": {
                "market_news": {"news": [{"headline": "Markets rally"}]},
                "web_search": {"answer": "Tech stocks are leading the market."}
            }
        })
        bot.researcher_agent = mock_researcher
        
        # Process a news query
        response = await bot.process_message("tell me the latest news")
        
        # Verify researcher agent was called
        mock_researcher.run.assert_called_once()
        call_args = mock_researcher.run.call_args
        assert "latest news" in call_args.kwargs.get("query", call_args.args[0] if call_args.args else "")
        
        # Verify response contains news content
        assert "Markets" in response or "news" in response.lower()
    
    @pytest.mark.asyncio
    async def test_chatbot_returns_research_response_not_fallback(self):
        """When research succeeds, should return research response, not fallback."""
        from src.web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True, enable_chain_of_thought=True)
        bot.llm = MagicMock()
        
        # Mock successful research
        mock_researcher = AsyncMock()
        expected_response = "Here are today's market headlines: 1. Fed announces rate decision..."
        mock_researcher.run = AsyncMock(return_value={
            "status": "success",
            "response": expected_response,
            "research_data": {}
        })
        bot.researcher_agent = mock_researcher
        
        response = await bot.process_message("what's the latest news")
        
        # Should NOT contain the fallback message
        assert "couldn't find any specific stocks" not in response.lower()
        # Should contain the research response
        assert expected_response in response or "Research Analysis" in response
    
    @pytest.mark.asyncio
    async def test_researcher_agent_uses_tools_for_news(self):
        """ResearcherAgent should actually call news tools."""
        from src.agents.researcher_agent import ResearcherAgent
        
        # Create agent with mocked MCP server
        agent = ResearcherAgent(
            finnhub_api_key="test_key",
            openai_api_key="test_key"
        )
        
        # Mock the MCP server's call_tool method
        mock_news_response = {
            "news": [
                {"headline": "Markets rally on Fed news", "summary": "Stocks up 2%"},
                {"headline": "Tech earnings beat expectations", "summary": "AAPL, MSFT up"}
            ]
        }
        mock_web_response = {
            "answer": "Today's top stories include market rally and tech earnings."
        }
        
        async def mock_call_tool(tool_name, params):
            if tool_name == "get_market_news":
                return mock_news_response
            elif tool_name == "web_search":
                return mock_web_response
            return {}
        
        agent.researcher_server.call_tool = AsyncMock(side_effect=mock_call_tool)
        
        # Mock LLM for response generation
        agent.llm = MagicMock()
        agent.llm.invoke = MagicMock(return_value=MagicMock(
            content="Based on the latest news, markets are rallying on Fed news and tech earnings beat expectations."
        ))
        
        # Run research
        result = await agent.run("tell me the latest news", return_structured=True)
        
        # Verify tools were called
        call_args_list = agent.researcher_server.call_tool.call_args_list
        tool_names_called = [call.args[0] for call in call_args_list]
        
        assert "get_market_news" in tool_names_called, f"Expected get_market_news to be called, got {tool_names_called}"
        
        # Verify response
        assert result["status"] == "success"
        assert len(result["response"]) > 0


class TestFullIntegrationWithMocks:
    """Full integration test with mocked external services."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_news_query(self):
        """Test complete flow: User asks for news -> delegation -> tools -> response."""
        from src.web.app import TradingChatBot
        from src.agents.researcher_agent import ResearcherAgent
        
        # Create real chatbot with real researcher agent
        bot = TradingChatBot(use_agents=True, enable_chain_of_thought=True)
        bot.llm = MagicMock()
        bot.llm.invoke = MagicMock(return_value=MagicMock(
            content="Here's a summary of today's news..."
        ))
        
        # Create researcher agent with mocked server
        researcher = ResearcherAgent(
            finnhub_api_key="test_key",
            openai_api_key="test_key",
            llm=bot.llm,
            enable_chain_of_thought=True
        )
        
        # Mock the tool calls
        async def mock_call_tool(tool_name, params):
            if tool_name == "get_market_news":
                return {
                    "news": [
                        {"headline": "S&P 500 hits new high", "summary": "Markets rally"},
                        {"headline": "Fed holds rates steady", "summary": "No change in policy"}
                    ]
                }
            elif tool_name == "web_search":
                return {
                    "answer": "Major financial news today includes market highs and Fed decisions."
                }
            return {}
        
        researcher.researcher_server.call_tool = AsyncMock(side_effect=mock_call_tool)
        bot.researcher_agent = researcher
        
        # Make the query
        response = await bot.process_message("tell me the latest news")
        
        # Assertions
        assert "couldn't find" not in response.lower(), \
            f"Should not return fallback message. Got: {response}"
        assert len(response) > 20, "Response should have content"
        
        # Verify the researcher was used (check if call_tool was called)
        assert researcher.researcher_server.call_tool.called, \
            "Researcher should have called tools"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_research_error_falls_back_gracefully(self):
        """If research fails, should fall back to LLM without crashing."""
        from src.web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True)
        
        # Mock LLM for fallback
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="I can help with that."))
        bot.llm = mock_llm
        
        # Mock failing researcher
        mock_researcher = AsyncMock()
        mock_researcher.run = AsyncMock(return_value={"error": "API rate limit exceeded"})
        bot.researcher_agent = mock_researcher
        
        # Also need to mock coinbase server to avoid broker config error
        bot.coinbase_server = MagicMock()
        
        # Should not raise, should fall back
        response = await bot.process_message("tell me the latest news")
        
        # Should have some response (either from fallback LLM or error message)
        assert response is not None
        assert len(response) > 0
    
    def test_empty_query_not_research(self):
        """Empty query should not be detected as research."""
        from src.web.app import TradingChatBot
        
        bot = TradingChatBot(use_agents=True)
        
        assert bot._is_research_query("") is False
        assert bot._is_research_query("   ") is False
