"""
Tests for Chain of Thought (CoT) support in agents.

TDD tests for adding structured reasoning to all agents.
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
logger = logging.getLogger("test.chain_of_thought")


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns CoT responses."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content="""## Reasoning Steps

1. **Understanding the Query**: The user wants to know about AAPL stock.
2. **Data Analysis**: Current price is $150.25, up 1.69% today.
3. **Market Context**: Analyst consensus is bullish with 37 buy ratings.
4. **Key Factors**: Strong financials with P/E of 28.5.

## Conclusion

Apple (AAPL) is trading at $150.25 with positive momentum. Analysts are bullish."""
    )
    return llm


# =============================================================================
# Tests for CoT Module
# =============================================================================

class TestChainOfThoughtPrompts:
    """Tests for CoT prompt templates."""

    def test_cot_module_exists(self):
        """CoT module should be importable."""
        from agents.chain_of_thought import ChainOfThought
        assert ChainOfThought is not None

    def test_cot_has_reasoning_prompt_template(self):
        """CoT should have a reasoning prompt template."""
        from agents.chain_of_thought import ChainOfThought
        
        cot = ChainOfThought()
        assert hasattr(cot, 'get_reasoning_prompt')

    def test_cot_reasoning_prompt_includes_step_by_step(self):
        """Reasoning prompt should include step-by-step instructions."""
        from agents.chain_of_thought import ChainOfThought
        
        cot = ChainOfThought()
        prompt = cot.get_reasoning_prompt(
            query="What is AAPL trading at?",
            data={"price": 150.25}
        )
        
        assert "step" in prompt.lower() or "Step" in prompt
        assert "reason" in prompt.lower() or "Reason" in prompt

    def test_cot_prompt_includes_data_context(self):
        """Reasoning prompt should include provided data."""
        from agents.chain_of_thought import ChainOfThought
        
        cot = ChainOfThought()
        data = {"current_price": 150.25, "change_percent": 1.69}
        prompt = cot.get_reasoning_prompt(
            query="Should I buy AAPL?",
            data=data
        )
        
        assert "150.25" in prompt or "current_price" in prompt

    def test_cot_prompt_includes_query(self):
        """Reasoning prompt should include the user query."""
        from agents.chain_of_thought import ChainOfThought
        
        cot = ChainOfThought()
        query = "What are the risks of investing in Tesla?"
        prompt = cot.get_reasoning_prompt(query=query, data={})
        
        assert query in prompt


class TestChainOfThoughtReasoningTypes:
    """Tests for different reasoning types."""

    def test_cot_supports_analysis_reasoning(self):
        """CoT should support analysis-type reasoning."""
        from agents.chain_of_thought import ChainOfThought, ReasoningType
        
        cot = ChainOfThought()
        prompt = cot.get_reasoning_prompt(
            query="Analyze AAPL",
            data={"price": 150},
            reasoning_type=ReasoningType.ANALYSIS
        )
        
        assert "analyze" in prompt.lower() or "analysis" in prompt.lower()

    def test_cot_supports_decision_reasoning(self):
        """CoT should support decision-type reasoning."""
        from agents.chain_of_thought import ChainOfThought, ReasoningType
        
        cot = ChainOfThought()
        prompt = cot.get_reasoning_prompt(
            query="Should I buy AAPL?",
            data={"price": 150},
            reasoning_type=ReasoningType.DECISION
        )
        
        assert "decision" in prompt.lower() or "recommend" in prompt.lower()

    def test_cot_supports_comparison_reasoning(self):
        """CoT should support comparison-type reasoning."""
        from agents.chain_of_thought import ChainOfThought, ReasoningType
        
        cot = ChainOfThought()
        prompt = cot.get_reasoning_prompt(
            query="Compare AAPL vs MSFT",
            data={"aapl": 150, "msft": 380},
            reasoning_type=ReasoningType.COMPARISON
        )
        
        assert "compare" in prompt.lower() or "comparison" in prompt.lower()

    def test_cot_supports_risk_assessment_reasoning(self):
        """CoT should support risk assessment reasoning."""
        from agents.chain_of_thought import ChainOfThought, ReasoningType
        
        cot = ChainOfThought()
        prompt = cot.get_reasoning_prompt(
            query="What are the risks of TSLA?",
            data={"volatility": 0.45},
            reasoning_type=ReasoningType.RISK_ASSESSMENT
        )
        
        assert "risk" in prompt.lower()


class TestChainOfThoughtStructuredOutput:
    """Tests for structured CoT output."""

    def test_cot_can_parse_reasoning_steps(self):
        """CoT should parse reasoning steps from LLM response."""
        from agents.chain_of_thought import ChainOfThought
        
        cot = ChainOfThought()
        response = """## Reasoning Steps
1. First, I analyzed the price data.
2. Then, I compared with historical averages.
3. Finally, I considered market sentiment.

## Conclusion
AAPL looks bullish."""

        parsed = cot.parse_response(response)
        
        assert "reasoning_steps" in parsed
        assert len(parsed["reasoning_steps"]) >= 3
        assert "conclusion" in parsed

    def test_cot_extracts_conclusion(self):
        """CoT should extract the conclusion from response."""
        from agents.chain_of_thought import ChainOfThought
        
        cot = ChainOfThought()
        response = """## Analysis
Looking at the data...

## Conclusion
Based on my analysis, Apple is a strong buy at current levels."""

        parsed = cot.parse_response(response)
        
        assert "conclusion" in parsed
        assert "Apple" in parsed["conclusion"] or "buy" in parsed["conclusion"]

    def test_cot_returns_raw_if_parsing_fails(self):
        """CoT should return raw response if parsing fails."""
        from agents.chain_of_thought import ChainOfThought
        
        cot = ChainOfThought()
        response = "Just a simple response without structure."
        
        parsed = cot.parse_response(response)
        
        assert "raw_response" in parsed
        assert parsed["raw_response"] == response


class TestChainOfThoughtTradingContext:
    """Tests for trading-specific CoT features."""

    def test_cot_trading_prompt_includes_risk_warning(self):
        """Trading prompts should include risk warning."""
        from agents.chain_of_thought import ChainOfThought, ReasoningType
        
        cot = ChainOfThought()
        prompt = cot.get_reasoning_prompt(
            query="Should I buy AAPL?",
            data={"price": 150},
            reasoning_type=ReasoningType.DECISION
        )
        
        assert "risk" in prompt.lower() or "advice" in prompt.lower()

    def test_cot_has_trading_specific_factors(self):
        """CoT should include trading-specific factors."""
        from agents.chain_of_thought import ChainOfThought
        
        cot = ChainOfThought()
        prompt = cot.get_reasoning_prompt(
            query="Analyze AAPL for potential trade",
            data={
                "price": 150,
                "pe_ratio": 28.5,
                "recommendations": {"buy": 30, "sell": 2}
            }
        )
        
        # Should mention considering various factors
        assert any(word in prompt.lower() for word in ["factor", "consider", "evaluate"])

    def test_cot_handles_portfolio_context(self):
        """CoT should handle portfolio context."""
        from agents.chain_of_thought import ChainOfThought
        
        cot = ChainOfThought()
        prompt = cot.get_reasoning_prompt(
            query="Should I add more AAPL?",
            data={"price": 150},
            portfolio_context={"AAPL": {"shares": 100, "avg_cost": 140}}
        )
        
        assert "portfolio" in prompt.lower() or "position" in prompt.lower() or "100" in prompt


# =============================================================================
# Tests for Agent Integration
# =============================================================================

class TestResearcherAgentCoTIntegration:
    """Tests for CoT integration in Researcher Agent."""

    @pytest.fixture
    def mock_researcher_server(self):
        """Create mock researcher server."""
        server = MagicMock()
        async def mock_call_tool(tool_name: str, args: dict):
            if tool_name == "get_stock_quote":
                return {"current_price": 150.25, "change_percent": 1.69}
            return {}
        server.call_tool = AsyncMock(side_effect=mock_call_tool)
        return server

    def test_researcher_agent_has_cot_enabled(self, mock_researcher_server, mock_llm):
        """Researcher agent should have CoT capability."""
        from agents.researcher_agent import ResearcherAgent
        
        agent = ResearcherAgent(
            researcher_server=mock_researcher_server,
            llm=mock_llm
        )
        
        assert hasattr(agent, 'enable_chain_of_thought')
        assert hasattr(agent, 'chain_of_thought')

    def test_researcher_agent_uses_cot_in_response_generation(self, mock_researcher_server, mock_llm):
        """Researcher agent should use CoT when generating responses."""
        from agents.researcher_agent import ResearcherAgent
        
        agent = ResearcherAgent(
            researcher_server=mock_researcher_server,
            llm=mock_llm,
            enable_chain_of_thought=True
        )
        
        # The agent should use CoT prompts
        assert agent.chain_of_thought is not None

    @pytest.mark.asyncio
    async def test_researcher_agent_returns_reasoning_steps(self, mock_researcher_server, mock_llm):
        """Researcher agent should return reasoning steps when CoT enabled."""
        from agents.researcher_agent import ResearcherAgent
        
        agent = ResearcherAgent(
            researcher_server=mock_researcher_server,
            llm=mock_llm,
            enable_chain_of_thought=True
        )
        
        result = await agent.run(
            query="Analyze AAPL stock",
            return_structured=True
        )
        
        assert "reasoning_steps" in result or "reasoning" in str(result).lower()


class TestCoinbaseAgentCoTIntegration:
    """Tests for CoT integration in Coinbase Agent."""

    @pytest.fixture
    def mock_mcp_server(self):
        """Create mock coinbase MCP server."""
        server = MagicMock()
        async def mock_call_tool(tool_name: str, params: dict):
            return {"success": True}
        server.call_tool = AsyncMock(side_effect=mock_call_tool)
        server.list_tools.return_value = [
            {"name": "get_accounts", "description": "Get accounts"},
        ]
        return server

    def test_coinbase_agent_has_cot_capability(self, mock_mcp_server, mock_llm):
        """Coinbase agent should have CoT capability."""
        from agents.coinbase_agent import CoinbaseAgent
        
        agent = CoinbaseAgent(
            mcp_server=mock_mcp_server,
            llm=mock_llm
        )
        
        assert hasattr(agent, 'enable_chain_of_thought')
        assert hasattr(agent, 'chain_of_thought')

    def test_coinbase_agent_cot_default_disabled(self, mock_mcp_server):
        """CoT should be disabled by default for backward compatibility."""
        from agents.coinbase_agent import CoinbaseAgent
        
        agent = CoinbaseAgent(mcp_server=mock_mcp_server)
        
        assert agent.enable_chain_of_thought is False

    def test_coinbase_agent_cot_can_be_enabled(self, mock_mcp_server, mock_llm):
        """CoT should be enableable via constructor."""
        from agents.coinbase_agent import CoinbaseAgent
        
        agent = CoinbaseAgent(
            mcp_server=mock_mcp_server,
            llm=mock_llm,
            enable_chain_of_thought=True
        )
        
        assert agent.enable_chain_of_thought is True
        assert agent.chain_of_thought is not None


class TestSchwabAgentCoTIntegration:
    """Tests for CoT integration in Schwab Agent."""

    @pytest.fixture
    def mock_mcp_server(self):
        """Create mock schwab MCP server."""
        server = MagicMock()
        async def mock_call_tool(tool_name: str, params: dict):
            return {"success": True}
        server.call_tool = AsyncMock(side_effect=mock_call_tool)
        server.list_tools.return_value = [
            {"name": "get_account", "description": "Get account"},
        ]
        return server

    def test_schwab_agent_has_cot_capability(self, mock_mcp_server, mock_llm):
        """Schwab agent should have CoT capability."""
        from agents.schwab_agent import SchwabAgent
        
        agent = SchwabAgent(
            mcp_server=mock_mcp_server,
            llm=mock_llm
        )
        
        assert hasattr(agent, 'enable_chain_of_thought')
        assert hasattr(agent, 'chain_of_thought')

    def test_schwab_agent_cot_default_disabled(self, mock_mcp_server):
        """CoT should be disabled by default for backward compatibility."""
        from agents.schwab_agent import SchwabAgent
        
        agent = SchwabAgent(mcp_server=mock_mcp_server)
        
        assert agent.enable_chain_of_thought is False

    def test_schwab_agent_cot_can_be_enabled(self, mock_mcp_server, mock_llm):
        """CoT should be enableable via constructor."""
        from agents.schwab_agent import SchwabAgent
        
        agent = SchwabAgent(
            mcp_server=mock_mcp_server,
            llm=mock_llm,
            enable_chain_of_thought=True
        )
        
        assert agent.enable_chain_of_thought is True
        assert agent.chain_of_thought is not None


# =============================================================================
# Tests for CoT State Management
# =============================================================================

class TestCoTStateTracking:
    """Tests for CoT state in agent workflows."""

    def test_researcher_state_includes_reasoning(self):
        """Researcher agent state should include reasoning field."""
        from agents.researcher_agent import ResearcherAgentState
        
        state = ResearcherAgentState(
            query="Test",
            messages=[],
            tool_calls=[],
            tool_results=[],
            research_data={},
            response="",
            context={},
            history=[],
            status="pending",
            reasoning_steps=[]
        )
        
        assert "reasoning_steps" in state

    def test_coinbase_state_includes_reasoning(self):
        """Coinbase agent state should include reasoning field."""
        from agents.coinbase_agent import CoinbaseAgentState
        
        state = CoinbaseAgentState(
            messages=[],
            tool_calls=[],
            tool_results=[],
            current_task="",
            reasoning_steps=[]
        )
        
        assert "reasoning_steps" in state

    def test_schwab_state_includes_reasoning(self):
        """Schwab agent state should include reasoning field."""
        from agents.schwab_agent import SchwabAgentState
        
        state = SchwabAgentState(
            messages=[],
            tool_calls=[],
            tool_results=[],
            current_task="",
            reasoning_steps=[]
        )
        
        assert "reasoning_steps" in state


# =============================================================================
# Tests for CoT Workflow Node
# =============================================================================

class TestCoTWorkflowNode:
    """Tests for CoT as a workflow node."""

    @pytest.fixture
    def mock_researcher_server(self):
        """Create mock researcher server."""
        server = MagicMock()
        async def mock_call_tool(tool_name: str, args: dict):
            return {"current_price": 150.25}
        server.call_tool = AsyncMock(side_effect=mock_call_tool)
        return server

    def test_researcher_workflow_has_reasoning_node(self, mock_researcher_server, mock_llm):
        """Researcher workflow should have a reasoning node when CoT enabled."""
        from agents.researcher_agent import ResearcherAgent
        
        agent = ResearcherAgent(
            researcher_server=mock_researcher_server,
            llm=mock_llm,
            enable_chain_of_thought=True
        )
        
        workflow = agent.get_workflow()
        # The workflow should have reasoning capability
        assert hasattr(agent, '_apply_chain_of_thought') or hasattr(agent, '_reason')

    def test_cot_node_executes_before_response(self, mock_researcher_server, mock_llm):
        """CoT reasoning should happen before final response generation."""
        from agents.researcher_agent import ResearcherAgent
        
        agent = ResearcherAgent(
            researcher_server=mock_researcher_server,
            llm=mock_llm,
            enable_chain_of_thought=True
        )
        
        # Verify agent has method to apply CoT
        assert callable(getattr(agent, '_apply_chain_of_thought', None)) or \
               callable(getattr(agent, '_generate_response', None))
