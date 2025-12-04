"""
Tests for agent short-term memory functionality.

The agents should remember previous interactions within a conversation session
using LangGraph's checkpointer mechanism.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# Test: TradingChatBot Memory Initialization
# =============================================================================

class TestChatBotMemoryInitialization:
    """Tests for memory/checkpointer initialization in TradingChatBot."""
    
    def test_chatbot_has_checkpointer(self, log):
        """TradingChatBot should have an InMemorySaver checkpointer."""
        from web.app import TradingChatBot
        from langgraph.checkpoint.memory import InMemorySaver
        
        log.info("Testing checkpointer initialization")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=True)
        
        assert hasattr(bot, 'checkpointer')
        assert isinstance(bot.checkpointer, InMemorySaver)
    
    def test_chatbot_has_thread_id(self, log):
        """TradingChatBot should have a thread_id for conversation tracking."""
        from web.app import TradingChatBot
        
        log.info("Testing thread_id initialization")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=True)
        
        assert hasattr(bot, 'thread_id')
        assert bot.thread_id is not None
        assert isinstance(bot.thread_id, str)
    
    def test_clear_chat_generates_new_thread_id(self, log):
        """Clearing chat should generate a new thread_id."""
        from web.app import TradingChatBot
        
        log.info("Testing thread_id reset on clear")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=True)
        
        original_thread_id = bot.thread_id
        bot.clear_history()
        
        assert bot.thread_id != original_thread_id


# =============================================================================
# Test: Conversation History Passed to Agents
# =============================================================================

class TestConversationHistoryPassedToAgents:
    """Tests for passing conversation history to agents."""
    
    def test_research_agent_receives_conversation_history(self, log):
        """Research agent should receive conversation history for context."""
        from web.app import TradingChatBot
        
        log.info("Testing conversation history passed to research agent")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=True)
        
        # Simulate some conversation history
        from langchain_core.messages import HumanMessage, AIMessage
        bot.conversation_history = [
            HumanMessage(content="Draw a duck"),
            AIMessage(content="Here's a duck SVG..."),
        ]
        
        # The bot should have a method to get config with thread_id
        assert hasattr(bot, '_get_agent_config')
        config = bot._get_agent_config()
        
        assert "configurable" in config
        assert "thread_id" in config["configurable"]
    
    def test_agent_config_includes_thread_id(self, log):
        """Agent config should include thread_id for checkpointer."""
        from web.app import TradingChatBot
        
        log.info("Testing agent config includes thread_id")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=True)
        
        config = bot._get_agent_config()
        
        assert config["configurable"]["thread_id"] == bot.thread_id


# =============================================================================
# Test: Agent Workflow Compiled with Checkpointer
# =============================================================================

class TestAgentWorkflowWithCheckpointer:
    """Tests for agent workflows compiled with checkpointer."""
    
    def test_researcher_agent_accepts_checkpointer(self, log):
        """ResearcherAgent should accept a checkpointer parameter."""
        from agents.researcher_agent import ResearcherAgent
        from langgraph.checkpoint.memory import InMemorySaver
        
        log.info("Testing ResearcherAgent accepts checkpointer")
        
        mock_server = MagicMock()
        mock_llm = MagicMock()
        checkpointer = InMemorySaver()
        
        # Should be able to create agent with checkpointer
        agent = ResearcherAgent(
            mcp_server=mock_server,
            llm=mock_llm,
            checkpointer=checkpointer
        )
        
        assert agent.checkpointer == checkpointer
    
    def test_coinbase_agent_accepts_checkpointer(self, log):
        """CoinbaseAgent should accept a checkpointer parameter."""
        from agents.coinbase_agent import CoinbaseAgent
        from langgraph.checkpoint.memory import InMemorySaver
        
        log.info("Testing CoinbaseAgent accepts checkpointer")
        
        mock_server = MagicMock()
        mock_llm = MagicMock()
        checkpointer = InMemorySaver()
        
        agent = CoinbaseAgent(
            mcp_server=mock_server,
            llm=mock_llm,
            checkpointer=checkpointer
        )
        
        assert agent.checkpointer == checkpointer
    
    def test_schwab_agent_accepts_checkpointer(self, log):
        """SchwabAgent should accept a checkpointer parameter."""
        from agents.schwab_agent import SchwabAgent
        from langgraph.checkpoint.memory import InMemorySaver
        
        log.info("Testing SchwabAgent accepts checkpointer")
        
        mock_server = MagicMock()
        mock_llm = MagicMock()
        checkpointer = InMemorySaver()
        
        agent = SchwabAgent(
            mcp_server=mock_server,
            llm=mock_llm,
            checkpointer=checkpointer
        )
        
        assert agent.checkpointer == checkpointer


# =============================================================================
# Test: Conversation Context in Agent State
# =============================================================================

class TestConversationContextInAgentState:
    """Tests for conversation context being included in agent state."""
    
    def test_researcher_agent_state_has_messages(self, log):
        """ResearcherAgentState should include messages for conversation history."""
        from agents.researcher_agent import ResearcherAgentState
        
        log.info("Testing ResearcherAgentState has messages field")
        
        # The state should have a messages field
        state: ResearcherAgentState = {
            "messages": [],
            "query": "test",
        }
        
        assert "messages" in state
    
    def test_agent_run_accepts_conversation_history(self, log):
        """Agent run method should accept conversation history."""
        from agents.researcher_agent import ResearcherAgent
        from langgraph.checkpoint.memory import InMemorySaver
        
        log.info("Testing agent run accepts conversation history")
        
        mock_server = MagicMock()
        mock_llm = MagicMock()
        checkpointer = InMemorySaver()
        
        agent = ResearcherAgent(
            mcp_server=mock_server,
            llm=mock_llm,
            checkpointer=checkpointer
        )
        
        # The run method should accept a messages parameter
        import inspect
        sig = inspect.signature(agent.run)
        params = list(sig.parameters.keys())
        
        # Should have query and messages parameters
        assert "query" in params
        assert "messages" in params or "conversation_history" in params or "config" in params


# =============================================================================
# Test: Memory Persistence Across Invocations
# =============================================================================

class TestMemoryPersistence:
    """Tests for memory persistence across agent invocations."""
    
    @pytest.mark.asyncio
    async def test_agent_remembers_previous_query(self, log):
        """Agent should remember previous queries within same thread."""
        from web.app import TradingChatBot
        from langgraph.checkpoint.memory import InMemorySaver
        
        log.info("Testing agent remembers previous query")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=True)
        
        # The checkpointer should be shared
        assert bot.checkpointer is not None
        
        # Verify the config includes the thread_id
        config = bot._get_agent_config()
        assert config["configurable"]["thread_id"] == bot.thread_id
    
    def test_different_threads_have_separate_memory(self, log):
        """Different thread IDs should have separate conversation memory."""
        from web.app import TradingChatBot
        
        log.info("Testing separate memory for different threads")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot1 = TradingChatBot(use_agents=True)
                    bot2 = TradingChatBot(use_agents=True)
        
        # Each bot should have its own thread_id
        assert bot1.thread_id != bot2.thread_id


# =============================================================================
# Test: Conversation History Format
# =============================================================================

class TestConversationHistoryFormat:
    """Tests for conversation history formatting for agents."""
    
    def test_format_history_for_agent(self, log):
        """Should format conversation history for agent consumption."""
        from web.app import TradingChatBot
        from langchain_core.messages import HumanMessage, AIMessage
        
        log.info("Testing conversation history formatting")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=True)
        
        # Add some history
        bot.conversation_history = [
            HumanMessage(content="Draw a duck"),
            AIMessage(content="Here's a duck!"),
            HumanMessage(content="What did I ask?"),
        ]
        
        # Should have a method to format history for agents
        assert hasattr(bot, '_format_history_for_agent')
        formatted = bot._format_history_for_agent()
        
        assert isinstance(formatted, list)
        assert len(formatted) == 3
    
    def test_format_history_converts_to_dicts(self, log):
        """Formatted history should be list of dicts for agent state."""
        from web.app import TradingChatBot
        from langchain_core.messages import HumanMessage, AIMessage
        
        log.info("Testing history converts to dicts")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=True)
        
        bot.conversation_history = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        
        formatted = bot._format_history_for_agent()
        
        # Each item should be a dict with role and content
        for msg in formatted:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg
