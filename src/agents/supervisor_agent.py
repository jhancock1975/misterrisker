"""Mister Risker Supervisor Agent - Orchestrates sub-agents for trading tasks.

This module implements the supervisor pattern where Mister Risker acts as the main
orchestrator agent, delegating tasks to specialized sub-agents:
- CoinbaseAgent: For cryptocurrency trading (buy/sell crypto, check balances)
- SchwabAgent: For stock/options trading (quotes, orders, account info)
- ResearcherAgent: For investment research and analysis

The supervisor decides which agent(s) to invoke based on the user's query,
without requiring manual broker switching.
"""

import logging
from typing import Any, Literal, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# Configure logging
logger = logging.getLogger("mister_risker.supervisor")


class SupervisorAgentError(Exception):
    """Exception raised by the Supervisor Agent."""
    pass


class RoutingDecision(BaseModel):
    """Schema for the supervisor's routing decision."""
    agent: Literal["coinbase", "schwab", "researcher", "direct"] = Field(
        description="Which agent should handle this request. 'direct' means respond without delegation."
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was chosen."
    )
    query_for_agent: str = Field(
        description="The query or instruction to pass to the sub-agent, or direct response if agent='direct'."
    )


class SupervisorState(MessagesState):
    """State for the supervisor agent workflow."""
    routing_decision: Optional[dict] = None
    agent_response: Optional[str] = None
    sub_agent_logs: list[str] = []


class SupervisorAgent:
    """Mister Risker - The main supervisor agent that orchestrates sub-agents.
    
    This agent receives all user queries and decides which specialized sub-agent
    should handle them. It implements the "tool calling" multi-agent pattern
    where sub-agents are treated as tools.
    
    Attributes:
        llm: The language model for decision making
        coinbase_agent: Sub-agent for crypto trading
        schwab_agent: Sub-agent for stock/options trading
        researcher_agent: Sub-agent for investment research
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        coinbase_agent: Any = None,
        schwab_agent: Any = None,
        researcher_agent: Any = None,
        checkpointer: Optional[InMemorySaver] = None,
        enable_chain_of_thought: bool = True
    ):
        """Initialize the Supervisor Agent.
        
        Args:
            llm: Language model for routing decisions
            coinbase_agent: CoinbaseAgent instance (optional)
            schwab_agent: SchwabAgent instance (optional)
            researcher_agent: ResearcherAgent instance (optional)
            checkpointer: State persistence (optional)
            enable_chain_of_thought: Whether to log reasoning steps
        """
        self.llm = llm
        self.coinbase_agent = coinbase_agent
        self.schwab_agent = schwab_agent
        self.researcher_agent = researcher_agent
        self.checkpointer = checkpointer or InMemorySaver()
        self.enable_chain_of_thought = enable_chain_of_thought
        
        # Build routing decision maker with structured output
        self.router = llm.with_structured_output(RoutingDecision)
        
        # Track available agents for routing
        self.available_agents = self._get_available_agents()
        
        logger.info(f"SupervisorAgent initialized with sub-agents: {list(self.available_agents.keys())}")
    
    def _get_available_agents(self) -> dict[str, Any]:
        """Get dictionary of available sub-agents."""
        agents = {}
        if self.coinbase_agent:
            agents["coinbase"] = self.coinbase_agent
        if self.schwab_agent:
            agents["schwab"] = self.schwab_agent
        if self.researcher_agent:
            agents["researcher"] = self.researcher_agent
        return agents
    
    def _build_routing_prompt(self) -> str:
        """Build the system prompt for routing decisions."""
        agent_descriptions = []
        
        if self.coinbase_agent:
            agent_descriptions.append("""
**Coinbase Agent** (agent="coinbase"):
- Check cryptocurrency balances and portfolio on YOUR Coinbase account
- Get current crypto prices (Bitcoin, Ethereum, etc.)
- Place buy/sell orders for cryptocurrencies
- View open orders and transaction history on YOUR account
- Blockchain data queries: recent transactions, blocks, on-chain data for BTC, ETH, XRP, SOL, ZEC
- **REAL-TIME crypto analysis**: patterns, trends, technical analysis for BTC, ETH, XRP, SOL, ZEC
- Use for crypto trading, account balances, price checks, blockchain data, AND crypto pattern/trend analysis""")
        
        if self.schwab_agent:
            agent_descriptions.append("""
**Schwab Agent** (agent="schwab"):
- Check stock account balances and positions
- Get stock quotes and option chains
- Place equity and options orders
- View market movers and market hours
- Use for ANY stock/options/equity-related queries""")
        
        if self.researcher_agent:
            agent_descriptions.append("""
**Researcher Agent** (agent="researcher"):
- Web search and internet queries (weather, news, general information)
- Investment research and analysis
- Stock/company analysis with news and sentiment
- Compare investments
- Risk assessment and recommendations
- Analyst ratings and financial metrics
- ANY question requiring external information or web search
- Use for questions asking for opinions, analysis, research, or general knowledge""")
        
        agents_section = "\n".join(agent_descriptions)
        
        return f"""You are Mister Risker, a multi-broker trading assistant supervisor.
Your job is to analyze user requests and route them to the appropriate specialized sub-agent.

## Available Sub-Agents:
{agents_section}

## Routing Guidelines:
1. **Coinbase**: Route here for YOUR crypto balances, crypto prices, crypto buy/sell orders, blockchain data queries, AND any analysis/patterns/trends for cryptocurrencies (BTC, ETH, XRP, SOL, ZEC, Bitcoin, Ethereum, etc.)
2. **Schwab**: Route here for stock quotes, stock orders, options, equity positions, market hours, etc.
3. **Researcher**: Route here for STOCK analysis, opinions, news about STOCKS, comparisons, "what do you think" about STOCKS, recommendations, web search, weather, general questions, or ANY query that needs external information (but NOT for crypto analysis - use Coinbase for that)
4. **Direct**: ONLY use for simple greetings like "hi" or "hello", or questions about YOUR capabilities

## Important Rules:
- If user mentions "Schwab account", "stock balance", or stock symbols like AAPL/TSLA → use Schwab
- If user wants to check THEIR crypto balance, crypto prices, or place crypto orders → use Coinbase
- If user asks about blockchain transactions, blocks, on-chain data, ledger, or network info → use Coinbase
- **CRITICAL: If user asks about BTC, Bitcoin, ETH, Ethereum, crypto patterns, trends, analysis, "what are you seeing", or any cryptocurrency analysis → ALWAYS use Coinbase (it has real-time WebSocket data with ACCURATE prices)**
- The Researcher agent does NOT have accurate crypto price data - it will hallucinate wrong prices like $39 for Bitcoin
- If user asks "what do you think" about STOCKS, or wants stock opinions/research → use Researcher
- If user asks about weather, news, or general information → use Researcher
- For stock-related research or external knowledge → use Researcher
- NEVER ask the user to "switch brokers" - YOU decide which agent to use
- AVOID using "direct" unless it's a simple greeting
- Pass the full context needed to the sub-agent in query_for_agent

Analyze the user's message and decide which agent should handle it."""
    
    def _is_crypto_analysis_query(self, message: str) -> bool:
        """Check if the query is about crypto analysis/patterns (should use Coinbase with real-time data).
        
        Returns True if the message contains crypto-related analysis keywords.
        This bypasses the LLM routing to ensure real-time WebSocket data is used.
        """
        message_lower = message.lower()
        
        # Crypto symbols and names
        crypto_terms = ['btc', 'bitcoin', 'eth', 'ethereum', 'xrp', 'ripple', 
                        'sol', 'solana', 'zec', 'zcash', 'crypto', 'cryptocurrency']
        
        # Analysis keywords
        analysis_terms = ['pattern', 'analyze', 'analysis', 'trend', 'seeing', 
                         'prediction', 'trajectory', 'movement', 'price action',
                         'technical', 'looking at', 'what do you see', 'how is',
                         'watch', 'monitoring', 'real-time', 'realtime', 'live']
        
        has_crypto = any(term in message_lower for term in crypto_terms)
        has_analysis = any(term in message_lower for term in analysis_terms)
        
        return has_crypto and has_analysis

    async def route_query(
        self, 
        user_message: str, 
        conversation_history: list[dict] = None
    ) -> RoutingDecision:
        """Route a user query to the appropriate sub-agent.
        
        Args:
            user_message: The user's query
            conversation_history: Previous conversation context
        
        Returns:
            RoutingDecision with agent choice and reasoning
        """
        logger.info(f"[SUPERVISOR] Routing query: '{user_message[:80]}...'")
        
        # FAST PATH: Force crypto analysis queries to Coinbase (has real-time WebSocket data)
        if self._is_crypto_analysis_query(user_message) and self.coinbase_agent:
            logger.info("[SUPERVISOR] Crypto analysis detected - forcing route to Coinbase (real-time data)")
            return RoutingDecision(
                agent="coinbase",
                reasoning="Crypto analysis query detected - using Coinbase agent with real-time WebSocket data for accurate prices",
                query_for_agent=user_message
            )
        
        # Build routing messages
        messages = [
            SystemMessage(content=self._build_routing_prompt())
        ]
        
        # Add relevant conversation context
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        # Add current query
        messages.append(HumanMessage(content=user_message))
        
        try:
            decision = await self.router.ainvoke(messages)
            logger.info(f"[SUPERVISOR] Routing decision: agent={decision.agent}, reason={decision.reasoning}")
            return decision
        except Exception as e:
            logger.error(f"[SUPERVISOR] Routing error: {e}")
            # Default to direct response on error
            return RoutingDecision(
                agent="direct",
                reasoning=f"Routing error: {e}",
                query_for_agent="I apologize, I had trouble processing your request. Could you please rephrase it?"
            )
    
    async def execute(
        self,
        user_message: str,
        conversation_history: list[dict] = None,
        config: dict = None
    ) -> dict:
        """Execute the supervisor workflow.
        
        Args:
            user_message: The user's query
            conversation_history: Previous messages for context
            config: LangGraph config with thread_id
        
        Returns:
            Dict with response, agent_used, and logs
        """
        logs = []
        logs.append(f"[SUPERVISOR] Received query: {user_message}")
        
        # Step 1: Route the query
        decision = await self.route_query(user_message, conversation_history)
        logs.append(f"[SUPERVISOR] Routing to: {decision.agent} (reason: {decision.reasoning})")
        
        # Step 2: Execute on chosen agent
        if decision.agent == "direct":
            # Supervisor responds directly using LLM with conversation context
            logs.append(f"[SUPERVISOR] Generating direct response with conversation context")
            response = await self._generate_direct_response(user_message, conversation_history)
            logs.append(f"[SUPERVISOR] Direct response: {len(response)} chars")
            
        elif decision.agent == "coinbase":
            if not self.coinbase_agent:
                logs.append(f"[SUPERVISOR] ERROR: Coinbase agent not available")
                response = "I'd like to help with your Coinbase request, but the Coinbase integration is not configured. Please check your API credentials."
            else:
                logs.append(f"[SUPERVISOR] Delegating to Coinbase Agent")
                try:
                    result = await self._call_coinbase_agent(decision.query_for_agent, config)
                    response = result.get("response", str(result))
                    logs.append(f"[COINBASE AGENT] Completed: {len(response)} chars")
                except Exception as e:
                    logs.append(f"[COINBASE AGENT] Error: {e}")
                    response = f"I encountered an error with the Coinbase agent: {e}"
        
        elif decision.agent == "schwab":
            if not self.schwab_agent:
                logs.append(f"[SUPERVISOR] ERROR: Schwab agent not available")
                response = "I'd like to help with your Schwab request, but the Schwab integration is not configured. Please check your API credentials."
            else:
                logs.append(f"[SUPERVISOR] Delegating to Schwab Agent")
                try:
                    result = await self._call_schwab_agent(decision.query_for_agent, config)
                    response = result.get("response", str(result))
                    logs.append(f"[SCHWAB AGENT] Completed: {len(response)} chars")
                except Exception as e:
                    logs.append(f"[SCHWAB AGENT] Error: {e}")
                    response = f"I encountered an error with the Schwab agent: {e}"
        
        elif decision.agent == "researcher":
            if not self.researcher_agent:
                logs.append(f"[SUPERVISOR] ERROR: Researcher agent not available")
                response = "I'd like to help with your research request, but the Researcher agent is not configured. Please check your FINNHUB_API_KEY."
            else:
                logs.append(f"[SUPERVISOR] Delegating to Researcher Agent")
                try:
                    result = await self._call_researcher_agent(decision.query_for_agent, conversation_history, config)
                    response = result.get("response", str(result))
                    logs.append(f"[RESEARCHER AGENT] Completed: {len(response)} chars")
                except Exception as e:
                    logs.append(f"[RESEARCHER AGENT] Error: {e}")
                    response = f"I encountered an error with the Researcher agent: {e}"
        
        else:
            logs.append(f"[SUPERVISOR] Unknown agent: {decision.agent}")
            response = f"I'm not sure how to handle this request. Could you please rephrase?"
        
        # Log all steps
        for log in logs:
            logger.info(log)
        
        return {
            "response": response,
            "agent_used": decision.agent,
            "reasoning": decision.reasoning,
            "logs": logs
        }
    
    async def _call_coinbase_agent(self, query: str, config: dict = None) -> dict:
        """Call the Coinbase sub-agent.
        
        Args:
            query: Query to pass to the agent
            config: LangGraph config
        
        Returns:
            Agent response dict
        """
        logger.info(f"[COINBASE AGENT] Processing: {query[:60]}...")
        
        try:
            result = await self.coinbase_agent.process_query(query, config=config)
            return result if isinstance(result, dict) else {"response": str(result)}
        except Exception as e:
            logger.error(f"[COINBASE AGENT] Error: {e}")
            raise
    
    async def _call_schwab_agent(self, query: str, config: dict = None) -> dict:
        """Call the Schwab sub-agent.
        
        Args:
            query: Query to pass to the agent
            config: LangGraph config
        
        Returns:
            Agent response dict
        """
        logger.info(f"[SCHWAB AGENT] Processing: {query[:60]}...")
        
        try:
            result = await self.schwab_agent.process_query(query, config=config)
            return result if isinstance(result, dict) else {"response": str(result)}
        except Exception as e:
            logger.error(f"[SCHWAB AGENT] Error: {e}")
            raise
    
    async def _call_researcher_agent(
        self, 
        query: str, 
        conversation_history: list[dict] = None,
        config: dict = None
    ) -> dict:
        """Call the Researcher sub-agent.
        
        Args:
            query: Query to pass to the agent
            conversation_history: Conversation context
            config: LangGraph config
        
        Returns:
            Agent response dict
        """
        logger.info(f"[RESEARCHER AGENT] Processing: {query[:60]}...")
        
        try:
            result = await self.researcher_agent.run(
                query=query,
                messages=conversation_history or [],
                config=config,
                return_structured=True
            )
            return result if isinstance(result, dict) else {"response": str(result)}
        except Exception as e:
            logger.error(f"[RESEARCHER AGENT] Error: {e}")
            raise

    async def _generate_direct_response(
        self,
        user_message: str,
        conversation_history: list[dict] = None
    ) -> str:
        """Generate an intelligent direct response using LLM with conversation context.
        
        This is used when no sub-agent is needed, but we still want to provide
        a thoughtful, context-aware response rather than just echoing the question.
        
        Args:
            user_message: The user's current message
            conversation_history: Previous messages for context
        
        Returns:
            An intelligent response string
        """
        logger.info(f"[SUPERVISOR] Generating direct response for: {user_message[:60]}...")
        
        # Build messages list with conversation history
        messages = [
            SystemMessage(content="""You are Mister Risker, a helpful AI trading assistant.
You have access to Coinbase for crypto trading, Schwab for stocks/options, and research tools.

When responding to the user:
- Be helpful, friendly, and conversational
- If they ask about something from earlier in the conversation, refer to the chat history
- If they ask a question you can answer from context, answer it directly
- If they need trading/research capabilities, let them know what you can do
- Never just echo their question back - always provide a thoughtful response
- Keep responses concise but complete""")
        ]
        
        # Add conversation history for context
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        
        # Add the current user message
        messages.append(HumanMessage(content=user_message))
        
        # Generate response using LLM
        try:
            response = await self.llm.ainvoke(messages)
            # Extract content safely - response.content might be a string or a list
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                # Handle list of content blocks (e.g., from Claude)
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                return "".join(text_parts) if text_parts else str(content)
            return content if isinstance(content, str) else str(content)
        except Exception as e:
            logger.error(f"[SUPERVISOR] Error generating direct response: {e}")
            return f"I apologize, I had trouble processing that. Could you please rephrase your question?"
