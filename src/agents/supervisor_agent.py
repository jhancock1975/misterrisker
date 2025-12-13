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
                # No specific stocks mentioned, suggest popular ones
                symbols = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'GOOG', 'MSFT', 'AMZN', 'META']
                asset_type = "stock"
            # Don't default to anything - let the caller ask for clarification
        
        return symbols, asset_type or "unknown"
    
    async def _ask_for_clarification(self, message: str) -> str:
        """Return a message asking the user to clarify what they want."""
        return message
    
    async def _generate_trading_strategy_response(
        self, 
        message: str,
        routing_decision: RoutingDecision = None,
        conversation_history: list[dict] = None,
        config: dict = None
    ) -> str:
        """Generate an intelligent trading strategy using AGENTIC workflow.
        
        This method orchestrates multiple sub-agents to gather real data,
        then uses LLM reasoning to synthesize actionable trading recommendations.
        
        WORKFLOW:
        1. Extract symbols from user query
        2. Gather real-time price data from Coinbase/Schwab
        3. Gather news and research from Researcher agent
        4. Get AI signals from FinRL if available  
        5. Retrieve relevant wisdom from RAG knowledge base
        6. Use LLM to reason about ALL data and generate recommendations
        
        Args:
            message: The user's query
            routing_decision: Optional pre-computed routing decision with extracted symbols
            conversation_history: Previous messages for context
            config: LangGraph config
        """
        logger.info(f"[SUPERVISOR] Starting AGENTIC trading strategy workflow")
        
        # Step 1: Extract symbols
        if routing_decision and routing_decision.symbols:
            symbols = routing_decision.symbols
            asset_type = routing_decision.asset_type
        else:
            symbols, asset_type = self._extract_symbols_from_query(message)
        
        if not symbols:
            # Don't assume crypto - ask what the user wants
            return await self._ask_for_clarification(
                "I'd be happy to provide trading recommendations! What assets are you interested in?\n\n"
                "**Crypto**: BTC, ETH, SOL, XRP, ZEC, etc.\n"
                "**Stocks**: AAPL, TSLA, NVDA, AMD, GOOG, MSFT, AMZN, META, etc.\n\n"
                "Just tell me which symbols you'd like me to analyze!"
            )
        
        logger.info(f"[SUPERVISOR] Analyzing symbols: {symbols} (type: {asset_type})")
        
        # Step 2: Gather data from multiple agents IN PARALLEL
        import asyncio
        gathered_data = {
            "prices": {},
            "research": "",
            "finrl_signals": {},
            "rag_wisdom": ""
        }
        
        # Helper for no-op async tasks
        async def noop_str():
            return ""
        
        async def noop_dict():
            return {}
        
        # Create tasks for parallel execution
        tasks = []
        
        # Task: Get price data
        tasks.append(self._gather_price_data(symbols, asset_type))
        
        # Task: Get research/news (if user asked for it or if researcher is available)
        if self.researcher_agent:
            research_query = f"Latest news and analysis for {', '.join(symbols[:3])}"
            tasks.append(self._gather_research(research_query, conversation_history, config))
        else:
            tasks.append(noop_str())
        
        # Task: Get FinRL signals (if available)
        if self.finrl_agent:
            tasks.append(self._gather_finrl_signals(symbols))
        else:
            tasks.append(noop_dict())
        
        # Task: Get RAG wisdom
        if self.trading_strategy_service and self.trading_strategy_service.rag_service:
            tasks.append(self._gather_rag_wisdom(symbols))
        else:
            tasks.append(noop_str())
        
        # Execute all tasks in parallel
        logger.info(f"[SUPERVISOR] Gathering data from {len(tasks)} sources in parallel...")
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            gathered_data["prices"] = results[0] if not isinstance(results[0], Exception) else {}
            gathered_data["research"] = results[1] if not isinstance(results[1], Exception) else ""
            gathered_data["finrl_signals"] = results[2] if not isinstance(results[2], Exception) else {}
            gathered_data["rag_wisdom"] = results[3] if not isinstance(results[3], Exception) else ""
        except Exception as e:
            logger.error(f"[SUPERVISOR] Error gathering data: {e}")
        
        logger.info(f"[SUPERVISOR] Data gathered: prices={len(gathered_data['prices'])} symbols, "
                   f"research={len(gathered_data['research'])} chars, "
                   f"finrl={len(gathered_data['finrl_signals'])} signals, "
                   f"rag={len(gathered_data['rag_wisdom'])} chars")
        
        # Step 3: Use LLM to reason and generate recommendations
        response = await self._llm_generate_recommendations(
            user_message=message,
            symbols=symbols,
            asset_type=asset_type,
            gathered_data=gathered_data,
            conversation_history=conversation_history
        )
        
        return response
    
    async def _gather_price_data(self, symbols: list[str], asset_type: str) -> dict:
        """Gather real-time price data from appropriate agents."""
        prices = {}
        
        for symbol in symbols:
            try:
                is_crypto = asset_type == "crypto" or symbol.endswith("-USD")
                
                if is_crypto and self.coinbase_agent:
                    # Get crypto price from Coinbase
                    result = await self.coinbase_agent.process_query(
                        f"Get the current price for {symbol}"
                    )
                    if isinstance(result, dict) and "response" in result:
                        prices[symbol] = {"source": "coinbase", "data": result["response"]}
                elif not is_crypto and self.schwab_agent:
                    # Get stock price from Schwab
                    result = await self.schwab_agent.process_query(
                        f"Get quote for {symbol}"
                    )
                    if isinstance(result, dict) and "response" in result:
                        prices[symbol] = {"source": "schwab", "data": result["response"]}
            except Exception as e:
                logger.warning(f"[SUPERVISOR] Error getting price for {symbol}: {e}")
                prices[symbol] = {"source": "error", "data": str(e)}
        
        return prices
    
    async def _gather_research(self, query: str, conversation_history: list[dict], config: dict) -> str:
        """Gather research and news from the Researcher agent."""
        if not self.researcher_agent:
            return ""
        
        try:
            logger.info(f"[SUPERVISOR] Calling Researcher agent: {query[:50]}...")
            result = await self.researcher_agent.run(
                query=query,
                messages=conversation_history or [],
                config=config,
                return_structured=True
            )
            return result.get("response", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            logger.warning(f"[SUPERVISOR] Research error: {e}")
            return f"Research unavailable: {e}"
    
    async def _gather_finrl_signals(self, symbols: list[str]) -> dict:
        """Gather AI trading signals from FinRL agent for ALL asset types."""
        if not self.finrl_agent:
            return {}
        
        signals = {}
        for symbol in symbols:
            try:
                # Get FinRL signals for any tradeable asset
                result = await self.finrl_agent.process(f"Get trading signal for {symbol}")
                signals[symbol] = result
            except Exception as e:
                logger.warning(f"[SUPERVISOR] FinRL error for {symbol}: {e}")
        
        return signals
    
    async def _gather_rag_wisdom(self, symbols: list[str]) -> str:
        """Retrieve relevant trading wisdom from RAG knowledge base."""
        if not self.trading_strategy_service or not self.trading_strategy_service.rag_service:
            return ""
        
        try:
            query = f"Trading strategy and risk management for {', '.join(symbols[:3])}"
            wisdom = self.trading_strategy_service.get_trading_wisdom(query, n_results=3)
            return wisdom
        except Exception as e:
            logger.warning(f"[SUPERVISOR] RAG error: {e}")
            return ""
    
    async def _llm_generate_recommendations(
        self,
        user_message: str,
        symbols: list[str],
        asset_type: str,
        gathered_data: dict,
        conversation_history: list[dict] = None
    ) -> str:
        """Use LLM to reason about gathered data and generate recommendations.
        
        This is the BRAIN of the agentic workflow - it takes all gathered data
        and uses the LLM to synthesize intelligent trading recommendations.
        """
        logger.info(f"[SUPERVISOR] LLM reasoning about {len(symbols)} symbols with gathered data...")
        
        # Build the context for LLM
        context_parts = []
        
        # Add price data
        if gathered_data["prices"]:
            context_parts.append("## Current Market Data")
            for symbol, data in gathered_data["prices"].items():
                context_parts.append(f"**{symbol}** (source: {data['source']}):\n{data['data']}\n")
        
        # Add research
        if gathered_data["research"]:
            context_parts.append(f"## News & Research\n{gathered_data['research']}\n")
        
        # Add FinRL signals
        if gathered_data["finrl_signals"]:
            context_parts.append("## AI/ML Trading Signals (FinRL)")
            for symbol, signal in gathered_data["finrl_signals"].items():
                context_parts.append(f"**{symbol}**: {signal}\n")
        
        # Add RAG wisdom
        if gathered_data["rag_wisdom"]:
            context_parts.append(f"## Trading Wisdom from Knowledge Base\n{gathered_data['rag_wisdom']}\n")
        
        context = "\n".join(context_parts)
        
        # Build the prompt
        system_prompt = """You are Mister Risker, an expert trading analyst and advisor.
Your job is to analyze the provided market data, news, AI signals, and trading wisdom
to generate SPECIFIC, ACTIONABLE trading recommendations.

For EACH symbol the user asked about, provide this EXACT format:

## 📊 [SYMBOL]
1. **Action**: BUY, SELL, or HOLD
2. **Confidence Level**: X% - REQUIRED! Calculate based on:
   - If FinRL signal available: Use FinRL confidence directly
   - If no FinRL: Estimate based on trend alignment, volatility, and data quality
   - Show as percentage (e.g., 75%)
3. **Entry Price**: $X.XX (to provide a margin of safety)
4. **Take Profit**: $X.XX
5. **Stop Loss**: $X.XX
6. **Risk/Reward Ratio**: X:X
7. **Position Size**: X% of portfolio
8. **Reasoning**: 2-3 sentences explaining WHY based on the data

CONFIDENCE CALCULATION RULES:
- 80%+: Strong signal alignment (FinRL + trend + fundamentals agree)
- 60-79%: Moderate confidence (most signals agree, some uncertainty)
- 40-59%: Low confidence (mixed signals, proceed with caution)
- <40%: Very low confidence (conflicting signals, small position only)

Apply these principles from The Intelligent Investor:
- Margin of Safety: Entry should be below current price for buys
- Risk Management: Always include stop loss
- Diversification: Consider position sizing based on confidence

Format your response with clear headers and emojis.
Be specific with numbers - don't be vague.
CONFIDENCE LEVEL IS MANDATORY for every recommendation!
"""

        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history for context
        if conversation_history:
            for msg in conversation_history[-5:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        
        # Add the current request with all gathered context
        user_prompt = f"""User request: {user_message}

Symbols to analyze: {', '.join(symbols)}
Asset type: {asset_type}

=== GATHERED DATA ===
{context}
=== END GATHERED DATA ===

Please analyze this data and provide specific trading recommendations for each symbol.
Include entry prices, take profit levels, stop losses, and your reasoning based on the data above."""

        messages.append(HumanMessage(content=user_prompt))
        
        # Generate response
        try:
            response = await self.llm.ainvoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            
            # Handle list of content blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                return "".join(text_parts) if text_parts else str(content)
            
            return content if isinstance(content, str) else str(content)
            
        except Exception as e:
            logger.error(f"[SUPERVISOR] LLM recommendation error: {e}")
            return f"I apologize, I had trouble generating recommendations: {e}"

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
**Purpose**: Cryptocurrency trading, real-time market data, blockchain queries, AND account history
**Capabilities**:
- Access to YOUR Coinbase account: balances, portfolio, positions
- **TRANSACTION HISTORY**: Account activity, deposits, withdrawals, fees, trade summary
- **FILLS/TRADES**: Executed orders, fill history, completed trades
- **ORDERS**: View, place, and cancel orders (market and limit orders)
- Real-time WebSocket price feeds for BTC, ETH, SOL, XRP, ZEC (ACCURATE live prices)
- **BLOCKCHAIN DATA**: On-chain transactions, blocks, largest transactions, blockchain analytics
- Technical analysis with live candle data (OHLCV)
- Pattern detection: bullish/bearish trends, volatility analysis
- **DATA CHARTS**: Generate actual price charts, correlation plots, performance graphs with REAL data
**Use when**: 
- User asks about crypto prices, wants to trade crypto, check crypto balances
- User asks about **transaction history**, **account activity**, or **trades**
- User asks about **fills** or **executed trades**
- User asks about **order status** or **my orders**
- User asks about **blockchain transactions** (e.g., "largest transactions on Solana blockchain")
- User asks about **on-chain data** for any cryptocurrency
- User mentions "blockchain" + a crypto name (Bitcoin, Ethereum, Solana, etc.)
- **User asks to DRAW, PLOT, CHART, or GRAPH crypto data** (correlation, prices, etc.)
**CRITICAL**: If user asks about "Solana blockchain" or "Bitcoin blockchain" transactions, 
route to Coinbase - this is blockchain data, NOT stock ticker "SOL"!
**CRITICAL**: If user asks to draw/plot/chart crypto prices or correlations, route to Coinbase!
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
- Account balances and **transaction history**
- Order status, open orders, executed orders
**Use when**: User asks about stocks, equities, options, or anything stock-related
- User asks about **transaction history** for stocks
- User asks about **order status** for stock orders
**Common stock symbols this handles**: AAPL, TSLA, NVDA, AMD, GOOG, GOOGL, MSFT, AMZN, META, 
MU (Micron), INTC (Intel), NFLX, and ALL OTHER stock ticker symbols
**KEY DISTINCTION**: If user mentions a stock symbol (like MU, AAPL, NVDA, etc.) and wants 
trading advice or limit orders, route to Schwab for stock-specific handling
**DO NOT use for**: Cryptocurrency trading (use Coinbase instead)""")
        
        if self.researcher_agent:
            capabilities.append("""
## Researcher Agent (agent="researcher")
**Purpose**: Investment research, web search, and external information gathering
**Capabilities**:
- **Web search for ANYTHING**: news, current events, world news, weather, sports
- Investment research: analyst reports, company information
- Fundamental analysis: earnings, financials, analyst ratings
- Sentiment analysis from news and social media
- Compare investments and companies
- Risk assessment and investment recommendations
- **CRITICAL**: Use for ANY question that requires external/up-to-date information
- **CRITICAL**: Use when asked about news, current events, "what's happening with X"
**Use when**: 
- User wants research, opinions, news, comparisons
- User asks about world events, current news, "what about [topic]?"
- User asks something that requires internet search
- User asks about topics outside of trading (weather, sports, politics, etc.)
- You don't have direct knowledge of the answer
**This agent has INTERNET ACCESS - use it when external info is needed!**
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
  - **"Trade ideas"** - even generic requests for trade ideas
  - **"What about other cryptos/stocks?"** - follow-up requests for different assets
**This is the RIGHT choice when user wants SPECIFIC PRICES to trade at**
**IMPORTANT**: Any request for "trade ideas" or "trading recommendations" should go to strategy!""")
        
        capabilities.append("""
## Direct Response (agent="direct")
**Purpose**: Simple interactions that don't need delegation
**Use when**: Simple greetings ("hi", "hello"), questions about Mister Risker's capabilities,
or basic clarifying questions
**DO NOT use for**: 
- Any actual trading, research, or analysis tasks
- News queries (use researcher!)
- Current events questions (use researcher!)
- Any question that requires external information (use researcher!)
- Weather, sports, world events (use researcher!)""")
        
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
   - **For generic queries like "trade ideas" or "trading strategy" with no specific symbols**:
     Use ALL available cryptos: ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ZEC-USD"]
2. **asset_type**: "crypto", "stock", "mixed", or "unknown"

# Routing Decision Guidelines

## Key Routing Rules:
1. **"Trade ideas" or "trading strategy"** → Strategy (ALWAYS! generates specific limit order prices)
2. **"What about other cryptos/stocks?"** → Strategy (generates specific prices for DIFFERENT assets)
3. **"Give me a limit order/strategy for X"** → Strategy (generates specific prices)
4. **"What is the current price of X?"** → Coinbase (crypto) or Schwab (stocks)
5. **"Research X" or "What do analysts think?"** → Researcher
6. **"Buy/sell X" (execute trade)** → Coinbase (crypto) or Schwab (stocks)
7. **"Check my balance"** → Coinbase (crypto) or Schwab (stocks)
8. **AI/ML trading signals** → FinRL
9. **BLOCKCHAIN queries** (transactions, blocks, on-chain data) → Coinbase
   - "Solana blockchain transactions" → Coinbase (NOT researcher!)
   - "Bitcoin blockchain" anything → Coinbase
   - "largest transactions on [crypto] blockchain" → Coinbase
10. **NEWS and CURRENT EVENTS** → Researcher (has internet access!)
11. **CHART/PLOT/GRAPH requests for crypto data** → Coinbase

**CRITICAL**: "trade ideas", "trading recommendations", "what should I trade" → ALWAYS route to Strategy!

## CRITICAL: Follow-Up "Other/Different" Queries
When user asks about "other", "different", "more", or "else" assets:
1. Route to **Strategy** agent (NOT coinbase!)
2. Look at conversation history to find what symbols were already discussed
3. **EXCLUDE those symbols from the symbols list**
4. Provide DIFFERENT assets from the available list: BTC-USD, ETH-USD, SOL-USD, XRP-USD, ZEC-USD

**IMPORTANT**: For crypto, we ONLY have real-time data for: BTC, ETH, SOL, XRP, ZEC
When user asks for "other cryptos", only suggest from this list (excluding what was previously discussed).

Example:
- Previous: Strategies for BTC, ETH
- User: "what about other cryptos?"
- DO NOT include BTC, ETH
- Instead, suggest from remaining: SOL-USD, XRP-USD, ZEC-USD

- Previous: BTC, ETH, SOL, XRP, ZEC (all 5 cryptos)
- User: "what about other cryptos?"
- All cryptos already shown! Suggest stocks instead: AAPL, TSLA, NVDA, AMD, etc.

For stocks:
- Previous: AAPL, TSLA, NVDA
- User: "what about other stocks?"
- DO NOT include AAPL, TSLA, NVDA
- Instead, suggest: AMD, GOOG, MSFT, AMZN, META, etc.

**ALWAYS check conversation history when user says "other" or "different"!**

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
            previously_mentioned_symbols = []
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:500]  # Larger window to capture more symbols
                context_summary.append(f"{role}: {content}")
                
                # Also extract any symbols mentioned in previous messages
                # Look for crypto patterns (BTC-USD, ETH, etc.) and stock tickers
                import re
                symbols_in_msg = re.findall(r'\b([A-Z]{2,5})-USD\b', content)
                symbols_in_msg.extend(re.findall(r'📊\s*([A-Z]{2,5})', content))
                previously_mentioned_symbols.extend(symbols_in_msg)
            
            if context_summary:
                context_text = "\n".join(context_summary)
                symbols_note = ""
                if previously_mentioned_symbols:
                    unique_symbols = list(set(previously_mentioned_symbols))
                    symbols_note = f"\n\n**PREVIOUSLY DISCUSSED SYMBOLS**: {unique_symbols}\nIf user asks about 'other' or 'different' assets, EXCLUDE these from your response!"
                
                messages.append(SystemMessage(
                    content=f"Recent conversation context:\n{context_text}{symbols_note}\n\nNow routing the following new message:"
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
                logs.append(f"[SUPERVISOR] Starting AGENTIC trading strategy workflow")
                logs.append(f"[SUPERVISOR] Symbols: {decision.symbols}, Asset type: {decision.asset_type}")
                try:
                    response = await self._generate_trading_strategy_response(
                        decision.query_for_agent,
                        routing_decision=decision,
                        conversation_history=conversation_history,
                        config=config
                    )
                    logs.append(f"[SUPERVISOR] AGENTIC workflow completed: {len(response)} chars")
                except Exception as e:
                    logs.append(f"[SUPERVISOR] AGENTIC workflow error: {e}")
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
