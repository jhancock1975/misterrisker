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

# Import TradingStrategyService for generating actionable recommendations
from src.services.trading_strategy import TradingStrategyService

# Configure logging
logger = logging.getLogger("mister_risker.supervisor")


class SupervisorAgentError(Exception):
    """Exception raised by the Supervisor Agent."""
    pass


class RoutingDecision(BaseModel):
    """Schema for the supervisor's routing decision."""
    agent: Literal["coinbase", "schwab", "researcher", "finrl", "strategy", "direct"] = Field(
        description="Which agent should handle this request. 'direct' means respond without delegation."
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was chosen."
    )
    query_for_agent: str = Field(
        description="The query or instruction to pass to the sub-agent, or direct response if agent='direct'."
    )
    symbols: list[str] = Field(
        default=[],
        description="Trading symbols mentioned or implied. For crypto use format like 'BTC-USD'. For stocks use ticker like 'AAPL'. Extract ALL symbols the user is asking about."
    )
    asset_type: Literal["crypto", "stock", "mixed", "unknown"] = Field(
        default="unknown",
        description="Type of assets: 'crypto' for cryptocurrency, 'stock' for equities, 'mixed' if both, 'unknown' if unclear."
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
        finrl_agent: Sub-agent for AI/RL trading decisions
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        coinbase_agent: Any = None,
        schwab_agent: Any = None,
        researcher_agent: Any = None,
        finrl_agent: Any = None,
        trading_strategy_service: Optional[TradingStrategyService] = None,
        checkpointer: Optional[InMemorySaver] = None,
        enable_chain_of_thought: bool = True
    ):
        """Initialize the Supervisor Agent.
        
        Args:
            llm: Language model for routing decisions
            coinbase_agent: CoinbaseAgent instance (optional)
            schwab_agent: SchwabAgent instance (optional)
            researcher_agent: ResearcherAgent instance (optional)
            finrl_agent: FinRLAgent instance for AI trading (optional)
            trading_strategy_service: Service for generating trading strategies (optional)
            checkpointer: State persistence (optional)
            enable_chain_of_thought: Whether to log reasoning steps
        """
        self.llm = llm
        self.coinbase_agent = coinbase_agent
        self.schwab_agent = schwab_agent
        self.researcher_agent = researcher_agent
        self.finrl_agent = finrl_agent
        self.trading_strategy_service = trading_strategy_service
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
        if self.finrl_agent:
            agents["finrl"] = self.finrl_agent
        return agents
    
    def _extract_symbols_from_query(self, message: str) -> tuple[list[str], str]:
        """Extract trading symbols from the query and determine asset type.
        
        Returns:
            Tuple of (symbols list, asset_type: "crypto" | "stock" | "mixed")
        """
        message_upper = message.upper()
        message_lower = message.lower()
        symbols = []
        asset_type = None
        
        # Check for crypto symbols first
        crypto_map = {
            'btc': 'BTC-USD', 'bitcoin': 'BTC-USD',
            'eth': 'ETH-USD', 'ethereum': 'ETH-USD',
            'sol': 'SOL-USD', 'solana': 'SOL-USD',
            'xrp': 'XRP-USD', 'ripple': 'XRP-USD',
            'zec': 'ZEC-USD', 'zcash': 'ZEC-USD',
        }
        
        for keyword, symbol in crypto_map.items():
            if keyword in message_lower and symbol not in symbols:
                symbols.append(symbol)
                asset_type = "crypto"
        
        # Check for stock symbols (from TradingStrategyService.COMMON_STOCKS)
        common_stocks = {
            'AAPL', 'TSLA', 'NVDA', 'AMD', 'GOOG', 'GOOGL', 'MSFT', 'AMZN', 'META',
            'MU', 'INTC', 'NFLX', 'BABA', 'DIS', 'V', 'MA', 'JPM', 'BAC', 'WFC',
            'XOM', 'CVX', 'PFE', 'JNJ', 'UNH', 'HD', 'WMT', 'TGT', 'COST', 'KO',
            'PEP', 'MCD', 'SBUX', 'NKE', 'ABNB', 'UBER', 'LYFT', 'SQ', 'PYPL',
            'CRM', 'ORCL', 'ADBE', 'NOW', 'SHOP', 'SNOW', 'PLTR', 'NET', 'DDOG',
            'ZM', 'DOCU', 'TWLO', 'OKTA', 'CRWD', 'ZS', 'PANW', 'FTNT', 'SPLK',
            'COIN', 'HOOD', 'RBLX', 'U', 'SNAP', 'PINS', 'TWTR', 'SPOT', 'ROKU',
            'GME', 'AMC', 'BB', 'NOK', 'SOFI', 'LCID', 'RIVN', 'F', 'GM', 'TM',
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'ARKK', 'XLF', 'XLK', 'XLE'
        }
        
        # Look for stock symbols in the message (as whole words)
        import re
        words = re.findall(r'\b[A-Z]{1,5}\b', message_upper)
        for word in words:
            if word in common_stocks and word not in symbols:
                symbols.append(word)
                if asset_type == "crypto":
                    asset_type = "mixed"
                else:
                    asset_type = "stock"
        
        # If no symbols found, check context words
        if not symbols:
            if any(term in message_lower for term in ['crypto', 'bitcoin', 'cryptocurrency', 'coin']):
                symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ZEC-USD']
                asset_type = "crypto"
            elif any(term in message_lower for term in ['stock', 'equity', 'equities', 'share']):
                # No specific stocks mentioned, can't default to a list
                asset_type = "stock"
            else:
                # Default to crypto for backward compatibility
                symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ZEC-USD']
                asset_type = "crypto"
        
        return symbols, asset_type or "crypto"
    
    async def _generate_trading_strategy_response(
        self, 
        message: str,
        routing_decision: RoutingDecision = None
    ) -> str:
        """Generate a trading strategy response using TradingStrategyService.
        
        This method synthesizes data from multiple sources to create actionable
        trading recommendations with entry, take profit, and stop loss levels.
        Supports ANY tradeable asset - crypto, stocks, ETFs, etc.
        
        Args:
            message: The user's query
            routing_decision: Optional pre-computed routing decision with extracted symbols
        """
        if not self.trading_strategy_service:
            return "Trading strategy service is not available. Please try again later."
        
        try:
            # Use LLM-extracted symbols if available, otherwise fall back to pattern matching
            if routing_decision and routing_decision.symbols:
                symbols = routing_decision.symbols
                asset_type = routing_decision.asset_type
            else:
                symbols, asset_type = self._extract_symbols_from_query(message)
            
            logger.info(f"[SUPERVISOR] Generating strategy for {symbols} (type: {asset_type})")
            
            if not symbols:
                return ("I couldn't identify specific trading symbols. "
                        "Please specify what you want to trade (e.g., 'BTC', 'AAPL', 'MU').")
            
            # Generate recommendations for all symbols using unified method
            recommendations = await self.trading_strategy_service.generate_recommendations(
                symbols=symbols,
                asset_type=asset_type,
                risk_tolerance="conservative"
            )
            
            if not recommendations:
                return "Unable to generate recommendations. No market data available for the requested symbols."
            
            if len(recommendations) == 1:
                return self.trading_strategy_service.format_recommendation_text(recommendations[0])
            else:
                # Multiple recommendations - format as portfolio
                strategy = self.trading_strategy_service.create_portfolio_from_recommendations(
                    recommendations=recommendations
                )
                return self.trading_strategy_service.format_portfolio_strategy_text(strategy)
                
        except Exception as e:
            logger.error(f"[SUPERVISOR] Trading strategy error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating trading strategy: {e}"

    def _build_intelligent_routing_prompt(self) -> str:
        """Build a comprehensive prompt for intelligent LLM-based routing.
        
        This prompt gives the LLM full context about all available agents and services,
        allowing it to make nuanced routing decisions rather than relying on keyword matching.
        """
        
        # Build dynamic capability descriptions based on what's available
        capabilities = []
        
        if self.coinbase_agent:
            capabilities.append("""
## Coinbase Agent (agent="coinbase")
**Purpose**: Cryptocurrency trading, real-time market data, AND blockchain queries
**Capabilities**:
- Access to YOUR Coinbase account: balances, portfolio, transaction history
- Real-time WebSocket price feeds for BTC, ETH, SOL, XRP, ZEC (ACCURATE live prices)
- Place buy/sell orders for cryptocurrencies
- **BLOCKCHAIN DATA**: On-chain transactions, blocks, largest transactions, blockchain analytics
- Technical analysis with live candle data (OHLCV)
- Pattern detection: bullish/bearish trends, volatility analysis
**Use when**: 
- User asks about crypto prices, wants to trade crypto, check crypto balances
- User asks about **blockchain transactions** (e.g., "largest transactions on Solana blockchain")
- User asks about **on-chain data** for any cryptocurrency
- User mentions "blockchain" + a crypto name (Bitcoin, Ethereum, Solana, etc.)
**CRITICAL**: If user asks about "Solana blockchain" or "Bitcoin blockchain" transactions, 
route to Coinbase - this is blockchain data, NOT stock ticker "SOL"!
**Recognized crypto symbols**: BTC, ETH, SOL, XRP, ZEC, Bitcoin, Ethereum, Solana, Ripple, Zcash
**DO NOT use for**: Stock trading, general web searches, or generating trading strategies with specific limit orders""")
        
        if self.schwab_agent:
            capabilities.append("""
## Schwab Agent (agent="schwab")
**Purpose**: Stock and options trading via Schwab brokerage
**Capabilities**:
- Access to YOUR Schwab brokerage account
- Stock quotes and real-time equity prices  
- Option chains and options pricing
- Place equity and options orders
- View positions, market movers, market hours
- Account balances and transaction history
**Use when**: User asks about stocks, equities, options, or anything stock-related
**Common stock symbols this handles**: AAPL, TSLA, NVDA, AMD, GOOG, GOOGL, MSFT, AMZN, META, 
MU (Micron), INTC (Intel), NFLX, and ALL OTHER stock ticker symbols
**KEY DISTINCTION**: If user mentions a stock symbol (like MU, AAPL, NVDA, etc.) and wants 
trading advice or limit orders, route to Schwab for stock-specific handling
**DO NOT use for**: Cryptocurrency trading (use Coinbase instead)""")
        
        if self.researcher_agent:
            capabilities.append("""
## Researcher Agent (agent="researcher")
**Purpose**: Investment research, web search, and external information
**Capabilities**:
- Web search for news, analyst reports, company information
- Fundamental analysis: earnings, financials, analyst ratings
- Sentiment analysis from news and social media
- Compare investments and companies
- Risk assessment and investment recommendations
- General knowledge queries (weather, news, etc.)
- Access to external data sources via internet search
**Use when**: User wants research, opinions, news, comparisons, or any external information
**DO NOT use for**: 
- Real-time crypto prices (will hallucinate)
- Placing trades
- Account operations
- **BLOCKCHAIN QUERIES**: If user asks about "Solana blockchain", "Bitcoin blockchain", 
  or any blockchain transactions, route to Coinbase agent NOT here!
  "SOL blockchain" means Solana cryptocurrency, NOT the stock ticker SOL.""")
        
        if self.finrl_agent:
            capabilities.append("""
## FinRL Agent (agent="finrl")
**Purpose**: AI-powered trading decisions using Deep Reinforcement Learning
**Capabilities**:
- Trained RL models (PPO, A2C, SAC, TD3) for crypto trading signals
- Buy/sell/hold recommendations based on machine learning
- Train new models on historical data
- AI confidence scores for trading decisions
- Portfolio optimization using reinforcement learning
**Use when**: User specifically asks for AI/ML trading decisions, mentions reinforcement learning,
asks "should I buy/sell" from an AI perspective, or wants machine learning-based signals
**DO NOT use for**: Specific limit order prices, detailed trading strategies with entry/exit levels""")
        
        if self.trading_strategy_service:
            capabilities.append("""
## Trading Strategy Service (agent="strategy")
**Purpose**: Generate actionable trading recommendations with specific prices for ANY asset
**Capabilities**:
- **Limit order recommendations** with specific entry prices
- **Take profit targets** based on technical analysis and risk/reward ratios
- **Stop loss levels** to protect against downside
- Position sizing recommendations (% of portfolio)
- Risk/reward ratio calculations
- Multi-asset portfolio strategies
- Applies "Intelligent Investor" principles: margin of safety, risk management
- For CRYPTO: Uses real-time WebSocket data + FinRL AI signals
- For STOCKS: Uses Schwab quote data + fundamental analysis
**Supported assets**: ANY tradeable asset - stocks, crypto, ETFs, etc.
**Use when**: User asks for:
  - Trading strategies with specific prices (any asset type)
  - Limit orders, entry points, exit points
  - Take profit and stop loss recommendations
  - "What orders should I place?"
  - "Give me a trading plan for [symbol]"
  - Actionable trade recommendations (not just analysis)
**This is the RIGHT choice when user wants SPECIFIC PRICES to trade at**""")
        
        capabilities.append("""
## Direct Response (agent="direct")
**Purpose**: Simple interactions that don't need delegation
**Use when**: Simple greetings ("hi", "hello"), questions about Mister Risker's capabilities,
or when no other agent is appropriate
**DO NOT use for**: Any actual trading, research, or analysis tasks""")
        
        capabilities_text = "\n".join(capabilities)
        
        return f"""You are Mister Risker, an intelligent trading assistant supervisor.
Your job is to analyze user requests and route them to the BEST agent or service.

# Available Agents and Services
{capabilities_text}

# Symbol Extraction
When routing, you MUST also extract:
1. **symbols**: List of trading symbols mentioned (e.g., ["BTC-USD", "AAPL", "ETH-USD"])
   - For crypto: use format "XXX-USD" (e.g., "BTC-USD", "ETH-USD")
   - For stocks: use ticker symbol (e.g., "AAPL", "MU", "TSLA")
2. **asset_type**: "crypto", "stock", "mixed", or "unknown"

# Routing Decision Guidelines

## Key Routing Rules:
1. **"Give me a limit order/strategy for X"** → Strategy (generates specific prices)
2. **"What is the current price of X?"** → Coinbase (crypto) or Schwab (stocks)
3. **"Research X" or "What do analysts think?"** → Researcher
4. **"Buy/sell X" (execute trade)** → Coinbase (crypto) or Schwab (stocks)
5. **"Check my balance"** → Coinbase (crypto) or Schwab (stocks)
6. **AI/ML trading signals** → FinRL
7. **BLOCKCHAIN queries** (transactions, blocks, on-chain data) → Coinbase
   - "Solana blockchain transactions" → Coinbase (NOT researcher!)
   - "Bitcoin blockchain" anything → Coinbase
   - "largest transactions on [crypto] blockchain" → Coinbase

## CRITICAL Disambiguation:
- "Solana blockchain" or "SOL blockchain" = cryptocurrency Solana → route to Coinbase
- "SOL stock" = Emeren Group Ltd stock ticker → route to Schwab
- When "blockchain" is mentioned with a crypto name, ALWAYS route to Coinbase

## Asset Type Detection:
- **CRYPTO**: BTC, ETH, SOL, XRP, ZEC, Bitcoin, Ethereum, Solana, etc.
- **STOCKS**: Any ticker trading on NYSE/NASDAQ (AAPL, TSLA, MU, NVDA, etc.)
- Use conversation context if asset type is ambiguous

## Response Requirements:
- **agent**: The agent code to route to
- **reasoning**: Brief explanation (1-2 sentences)
- **query_for_agent**: The full query to pass
- **symbols**: List of trading symbols extracted from the query
- **asset_type**: "crypto", "stock", "mixed", or "unknown"

Be intelligent about symbol detection - if user says "MU" that's Micron stock, 
if they say "Bitcoin" that's BTC-USD crypto. Extract ALL symbols mentioned."""
    
    async def route_query(
        self, 
        user_message: str, 
        conversation_history: list[dict] = None
    ) -> RoutingDecision:
        """Route a user query using intelligent LLM-based decision making.
        
        This method uses the LLM to analyze the user's query and determine
        the best agent/service to handle it, rather than using keyword matching.
        
        Args:
            user_message: The user's query
            conversation_history: Previous conversation context
        
        Returns:
            RoutingDecision with agent choice and reasoning
        """
        logger.info(f"[SUPERVISOR] Intelligent routing for: '{user_message[:80]}...'")
        
        # Build routing messages with comprehensive context
        messages = [
            SystemMessage(content=self._build_intelligent_routing_prompt())
        ]
        
        # Add relevant conversation context for better routing decisions
        if conversation_history:
            context_summary = []
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]  # Truncate for context
                context_summary.append(f"{role}: {content}")
            
            if context_summary:
                context_text = "\n".join(context_summary)
                messages.append(SystemMessage(
                    content=f"Recent conversation context:\n{context_text}\n\nNow routing the following new message:"
                ))
        
        # Add current query
        messages.append(HumanMessage(content=user_message))
        
        try:
            # Use LLM with structured output for routing decision
            decision = await self.router.ainvoke(messages)
            
            # Validate the agent choice is available
            valid_agents = ["direct"]
            if self.coinbase_agent:
                valid_agents.append("coinbase")
            if self.schwab_agent:
                valid_agents.append("schwab")
            if self.researcher_agent:
                valid_agents.append("researcher")
            if self.finrl_agent:
                valid_agents.append("finrl")
            if self.trading_strategy_service:
                valid_agents.append("strategy")
            
            if decision.agent not in valid_agents:
                logger.warning(f"[SUPERVISOR] LLM chose unavailable agent '{decision.agent}', falling back to direct")
                decision = RoutingDecision(
                    agent="direct",
                    reasoning=f"Requested agent '{decision.agent}' not available, responding directly",
                    query_for_agent=user_message
                )
            
            logger.info(f"[SUPERVISOR] LLM routing decision: agent={decision.agent}, reason={decision.reasoning[:80]}...")
            return decision
            
        except Exception as e:
            logger.error(f"[SUPERVISOR] LLM routing error: {e}")
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
        
        elif decision.agent == "finrl":
            if not self.finrl_agent:
                logs.append(f"[SUPERVISOR] ERROR: FinRL agent not available")
                response = "I'd like to help with AI trading decisions, but the FinRL agent is not configured."
            else:
                logs.append(f"[SUPERVISOR] Delegating to FinRL Agent")
                try:
                    response = await self.finrl_agent.process(decision.query_for_agent)
                    logs.append(f"[FINRL AGENT] Completed: {len(response)} chars")
                except Exception as e:
                    logs.append(f"[FINRL AGENT] Error: {e}")
                    response = f"I encountered an error with the FinRL agent: {e}"
        
        elif decision.agent == "strategy":
            if not self.trading_strategy_service:
                logs.append(f"[SUPERVISOR] ERROR: Trading Strategy service not available")
                response = "I'd like to help with trading strategy, but the strategy service is not configured."
            else:
                logs.append(f"[SUPERVISOR] Generating trading strategy with TradingStrategyService")
                logs.append(f"[SUPERVISOR] Symbols: {decision.symbols}, Asset type: {decision.asset_type}")
                try:
                    response = await self._generate_trading_strategy_response(
                        decision.query_for_agent,
                        routing_decision=decision
                    )
                    logs.append(f"[STRATEGY SERVICE] Completed: {len(response)} chars")
                except Exception as e:
                    logs.append(f"[STRATEGY SERVICE] Error: {e}")
                    response = f"I encountered an error generating trading strategy: {e}"
        
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
For blockchain data, we use the Solana RPC API directly.

When responding to the user:
- Be helpful, friendly, and conversational
- If they ask about something from earlier in the conversation, refer to the chat history
- If they ask a question you can answer from context, answer it directly
- If they need trading/research capabilities, let them know what you can do
- Never just echo their question back - always provide a thoughtful response
- Keep responses concise but complete
- Do NOT suggest external blockchain explorers (blockchair.com, etherscan.io, etc.) - our data comes directly from chain APIs""")
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
