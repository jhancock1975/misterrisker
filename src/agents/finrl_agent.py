"""FinRL Agent - AI Trading Decisions using Deep Reinforcement Learning.

This agent uses FinRL and stable-baselines3 to provide AI-powered trading
recommendations based on real-time market data.

This is the CORE personality of Mister Risker - a sophisticated AI trading
advisor that combines deep reinforcement learning with LLM reasoning.
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.services.finrl_service import FinRLService, TradingSignal

logger = logging.getLogger("mister_risker.finrl_agent")

# Mister Risker's core personality - used in all responses
MISTER_RISKER_PERSONALITY = """You are **Mister Risker**, an elite AI trading advisor with deep expertise in:
- Deep Reinforcement Learning (PPO, A2C, SAC, TD3 algorithms)
- Quantitative finance and algorithmic trading
- Technical analysis and market microstructure
- Risk management and portfolio theory
- Cryptocurrency and traditional equity markets

**Your personality**:
- Confident but measured - you're an expert but always acknowledge uncertainty
- Data-driven - you back up opinions with quantitative analysis
- Risk-aware - you always discuss risk/reward tradeoffs
- Educational - you explain your reasoning and teach trading concepts
- Witty and engaging - you make complex topics accessible
- Bold when appropriate - you give clear recommendations when confidence is high

**Your communication style**:
- Use emojis appropriately: 🤖 for AI insights, 📊 for data, ⚠️ for warnings
- Format responses with clear structure (headers, bullet points)
- Include confidence levels when giving recommendations
- Always mention that this is AI analysis, not financial advice
- Use mathematical notation when discussing formulas (LaTeX: $formula$)

**Core principles**:
- Reinforcement learning finds patterns humans miss
- Risk management is more important than returns
- Markets are probabilistic, not deterministic
- Continuous learning improves predictions over time"""


@dataclass
class FinRLAgentState:
    """State for FinRL agent workflow."""
    query: str
    symbol: Optional[str] = None
    signal: Optional[TradingSignal] = None
    response: str = ""
    error: Optional[str] = None


class FinRLAgent:
    """Agent for AI-powered trading decisions using deep reinforcement learning.
    
    This agent wraps the FinRL service and provides natural language interface
    for getting trading recommendations.
    """
    
    # Crypto symbol mappings
    CRYPTO_MAP = {
        "btc": "BTC-USD", "bitcoin": "BTC-USD",
        "eth": "ETH-USD", "ethereum": "ETH-USD",
        "sol": "SOL-USD", "solana": "SOL-USD",
        "xrp": "XRP-USD", "ripple": "XRP-USD",
        "zec": "ZEC-USD", "zcash": "ZEC-USD",
    }
    
    # Common stock symbols
    STOCK_SYMBOLS = {
        "aapl", "apple", "tsla", "tesla", "nvda", "nvidia", 
        "amd", "goog", "googl", "google", "msft", "microsoft",
        "amzn", "amazon", "meta", "facebook", "nflx", "netflix",
        "spy", "qqq", "dia", "iwm"
    }
    
    # Combined symbol map
    SYMBOL_MAP = {
        **CRYPTO_MAP,
        # Stocks use their own ticker
    }
    
    def __init__(
        self,
        llm: ChatOpenAI,
        finrl_service: FinRLService,
        websocket_service: Any = None
    ):
        """Initialize the FinRL Agent.
        
        Args:
            llm: Language model for response generation
            finrl_service: FinRL service for trading signals
            websocket_service: WebSocket service for real-time data
        """
        self.llm = llm
        self.finrl_service = finrl_service
        self.websocket_service = websocket_service
        
        # Ensure FinRL service has WebSocket access
        if websocket_service and not finrl_service.websocket_service:
            finrl_service.websocket_service = websocket_service
        
        self.workflow = self._build_workflow()
        
        logger.info("FinRL Agent initialized")
    
    def _extract_symbol(self, query: str) -> Optional[str]:
        """Extract trading symbol from query (crypto or stock)."""
        query_lower = query.lower()
        query_upper = query.upper()
        
        # Check crypto mappings first
        for keyword, symbol in self.CRYPTO_MAP.items():
            if keyword in query_lower:
                return symbol
        
        # Check for stock tickers (uppercase in query)
        import re
        # Find uppercase words that look like tickers (2-5 chars)
        potential_tickers = re.findall(r'\b([A-Z]{2,5})\b', query_upper)
        for ticker in potential_tickers:
            if ticker.lower() in self.STOCK_SYMBOLS:
                return ticker
        
        # Also check lowercase mentions
        for stock in self.STOCK_SYMBOLS:
            if stock in query_lower:
                # Return uppercase ticker
                ticker_map = {
                    "apple": "AAPL", "tesla": "TSLA", "nvidia": "NVDA",
                    "google": "GOOG", "microsoft": "MSFT", "amazon": "AMZN",
                    "facebook": "META", "netflix": "NFLX"
                }
                return ticker_map.get(stock, stock.upper())
        
        return None
    
    def _build_workflow(self) -> CompiledStateGraph:
        """Build the agent workflow."""
        
        def analyze_query(state: dict) -> dict:
            """Analyze query to extract symbol and intent."""
            query = state["query"]
            symbol = self._extract_symbol(query)
            
            return {**state, "symbol": symbol}
        
        def get_trading_signal(state: dict) -> dict:
            """Get trading signal from FinRL service."""
            symbol = state.get("symbol")
            
            if symbol:
                signal = self.finrl_service.get_trading_signal(symbol)
                return {**state, "signal": signal}
            else:
                # Get portfolio recommendation
                portfolio = self.finrl_service.get_portfolio_recommendation()
                return {**state, "portfolio": portfolio}
        
        def format_response(state: dict) -> dict:
            """Format the response for the user."""
            signal = state.get("signal")
            portfolio = state.get("portfolio")
            
            if signal:
                response = self._format_signal(signal)
            elif portfolio:
                response = self._format_portfolio(portfolio)
            else:
                response = "Unable to generate trading recommendation. Please try again."
            
            return {**state, "response": response}
        
        # Build graph
        workflow = StateGraph(dict)
        
        workflow.add_node("analyze", analyze_query)
        workflow.add_node("get_signal", get_trading_signal)
        workflow.add_node("format", format_response)
        
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "get_signal")
        workflow.add_edge("get_signal", "format")
        workflow.add_edge("format", END)
        
        return workflow.compile()
    
    def _format_signal(self, signal: TradingSignal) -> str:
        """Format a trading signal for display."""
        # Action emoji
        action_emoji = {
            "buy": "🟢 BUY",
            "sell": "🔴 SELL",
            "hold": "🟡 HOLD"
        }
        
        confidence_bar = "█" * int(signal.confidence * 10) + "░" * (10 - int(signal.confidence * 10))
        
        return f"""🤖 **FinRL AI Trading Signal**

**{signal.symbol}** - {action_emoji.get(signal.action, signal.action.upper())}

💰 **Current Price**: ${signal.price:,.2f}
📊 **Confidence**: {confidence_bar} {signal.confidence:.0%}
🧠 **Model**: {signal.model_name} (Deep RL)

**Analysis**: {signal.reasoning}

⚠️ *This is an AI-generated signal based on reinforcement learning. Not financial advice.*"""
    
    def _format_portfolio(self, portfolio: dict) -> str:
        """Format portfolio recommendation."""
        recommendation = portfolio.get("recommendation", "NEUTRAL")
        
        rec_emoji = {
            "BULLISH": "📈",
            "BEARISH": "📉",
            "NEUTRAL": "➡️"
        }
        
        signals_text = ""
        for s in portfolio.get("signals", [])[:5]:  # Top 5
            action_emoji = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(s["action"], "⚪")
            signals_text += f"\n  {action_emoji} **{s['symbol']}**: {s['action'].upper()} (${s['price']:,.2f})"
        
        return f"""🤖 **FinRL AI Portfolio Analysis**

{rec_emoji.get(recommendation, "📊")} **Overall Outlook**: {recommendation}

📊 **Signal Summary**:
  - 🟢 Buy Signals: {portfolio.get('buy_count', 0)}
  - 🔴 Sell Signals: {portfolio.get('sell_count', 0)}
  - 🟡 Hold Signals: {portfolio.get('hold_count', 0)}

**Individual Signals**:{signals_text}

💡 **Summary**: {portfolio.get('summary', 'N/A')}

⚠️ *AI-generated analysis using Deep Reinforcement Learning. Not financial advice.*"""
    
    async def process(self, query: str) -> str:
        """Process a query and return trading recommendation or discussion.
        
        As Mister Risker, this agent can handle:
        - Trading signal requests
        - Model training
        - Portfolio analysis
        - General trading discussions and education
        - Market analysis and explanations
        
        Args:
            query: User's query about trading
        
        Returns:
            Formatted response with trading signal or discussion
        """
        logger.info(f"[FINRL AGENT] Processing: {query[:50]}...")
        
        try:
            # Use LLM to determine the intent and extract symbol
            intent, symbol = await self._llm_analyze_query(query)
            
            if intent == "train":
                symbol = symbol or "BTC-USD"
                result = self.finrl_service.train_model(symbol, total_timesteps=10000)
                return f"""🎓 **Model Training Complete**

✅ Trained {result['algorithm']} model for **{result['symbol']}**
⏱️ Training time: {result['training_time']:.1f} seconds
📁 Model saved to: `{result['model_path']}`
🔄 Timesteps: {result['total_timesteps']:,}

The model is now ready to generate trading signals!"""
            
            elif intent == "portfolio":
                portfolio = self.finrl_service.get_portfolio_recommendation()
                return self._format_portfolio(portfolio)
            
            elif intent == "signal" and symbol:
                signal = self.finrl_service.get_trading_signal(symbol)
                return self._format_signal(signal)
            
            elif intent == "discuss":
                # General trading discussion - use LLM with Mister Risker personality
                return await self._generate_discussion_response(query, symbol)
            
            elif intent == "explain":
                # Educational explanation - use LLM to explain concepts
                return await self._generate_educational_response(query)
            
            elif intent == "analyze":
                # Market analysis with RL insights
                return await self._generate_analysis_response(query, symbol)
            
            else:
                # Default: check if it's a general question or needs portfolio
                if any(word in query.lower() for word in ['what', 'how', 'why', 'explain', 'tell me', 'can you']):
                    return await self._generate_discussion_response(query, symbol)
                portfolio = self.finrl_service.get_portfolio_recommendation()
                return self._format_portfolio(portfolio)
            
        except Exception as e:
            logger.error(f"[FINRL AGENT] Error: {e}")
            return f"❌ Error generating response: {str(e)}"
    
    async def _generate_discussion_response(self, query: str, symbol: Optional[str] = None) -> str:
        """Generate a conversational response as Mister Risker.
        
        Uses the LLM with Mister Risker personality to discuss trading topics.
        """
        import asyncio
        
        # Get any relevant RL signals to include in context
        context = ""
        if symbol:
            try:
                signal = self.finrl_service.get_trading_signal(symbol)
                context = f"""
Current RL Model Signal for {symbol}:
- Action: {signal.action.upper()}
- Confidence: {signal.confidence:.0%}
- Price: ${signal.price:,.2f}
- Reasoning: {signal.reasoning}
"""
            except:
                pass
        
        system_prompt = f"""{MISTER_RISKER_PERSONALITY}

{context}

Respond to the user's query in character as Mister Risker. Be helpful, insightful, 
and engaging. If the question relates to specific assets, include relevant RL insights.
If asked about complex topics, use proper mathematical notation with LaTeX ($formula$).
"""
        
        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            
            content = response.content
            if isinstance(content, list):
                content = content[0].get("text", "") if content else ""
            
            return content
            
        except Exception as e:
            logger.error(f"Discussion generation failed: {e}")
            return f"🤖 I had trouble formulating a response. Let me try a simpler approach: {str(e)}"
    
    async def _generate_educational_response(self, query: str) -> str:
        """Generate an educational response explaining trading concepts."""
        import asyncio
        
        system_prompt = f"""{MISTER_RISKER_PERSONALITY}

The user wants to learn about a trading concept. As Mister Risker, explain it clearly:
1. Start with a simple explanation
2. Add technical details with proper mathematical notation (LaTeX: $formula$)
3. Give practical examples
4. Relate it to reinforcement learning if applicable
5. Include relevant emojis for engagement

Be thorough but accessible. Use formulas when appropriate."""
        
        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            
            content = response.content
            if isinstance(content, list):
                content = content[0].get("text", "") if content else ""
            
            return content
            
        except Exception as e:
            logger.error(f"Educational response failed: {e}")
            return f"🎓 I couldn't generate an educational response: {str(e)}"
    
    async def _generate_analysis_response(self, query: str, symbol: Optional[str] = None) -> str:
        """Generate a market analysis response with RL insights."""
        import asyncio
        
        # Gather RL signals for context
        signals_context = ""
        if symbol:
            try:
                signal = self.finrl_service.get_trading_signal(symbol)
                signals_context = f"""
**RL Analysis for {symbol}:**
- Signal: {signal.action.upper()} 
- Confidence: {signal.confidence:.0%}
- Current Price: ${signal.price:,.2f}
- Model: {signal.model_name}
- Reasoning: {signal.reasoning}
"""
            except:
                pass
        else:
            # Get portfolio overview
            try:
                portfolio = self.finrl_service.get_portfolio_recommendation()
                signals_context = f"""
**RL Portfolio Analysis:**
- Overall Outlook: {portfolio.get('recommendation', 'NEUTRAL')}
- Buy Signals: {portfolio.get('buy_count', 0)}
- Sell Signals: {portfolio.get('sell_count', 0)}
- Hold Signals: {portfolio.get('hold_count', 0)}
"""
            except:
                pass
        
        system_prompt = f"""{MISTER_RISKER_PERSONALITY}

You have access to real-time RL model analysis:
{signals_context}

Provide market analysis that incorporates this RL data. Be specific about:
1. What the RL model is seeing
2. Technical factors affecting the signals
3. Risk considerations
4. Actionable insights

Use proper mathematical notation for any formulas (LaTeX: $formula$)."""
        
        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            
            content = response.content
            if isinstance(content, list):
                content = content[0].get("text", "") if content else ""
            
            return content
            
        except Exception as e:
            logger.error(f"Analysis response failed: {e}")
            return f"📊 I couldn't generate analysis: {str(e)}"
    
    async def _llm_analyze_query(self, query: str) -> tuple[str, Optional[str]]:
        """Use LLM to analyze query intent and extract symbol.
        
        Args:
            query: User's query
            
        Returns:
            Tuple of (intent, symbol) where intent determines response type
        """
        import json
        import asyncio
        
        system_prompt = """Analyze the user's query and determine the best intent:

**INTENTS**:
- "signal": User wants a specific trading signal for an asset (buy/sell/hold)
- "portfolio": User wants overview of multiple assets or general portfolio advice
- "train": User wants to train/retrain a model
- "discuss": User wants to discuss trading, ask opinions, or have a conversation
- "explain": User wants to learn about a concept, formula, or trading topic
- "analyze": User wants market analysis or technical discussion

**SYMBOL EXTRACTION**:
If the query mentions a specific asset, extract the symbol:
- Bitcoin/BTC -> BTC-USD
- Ethereum/ETH -> ETH-USD  
- Solana/SOL -> SOL-USD
- Apple -> AAPL
- Tesla -> TSLA
- Nvidia -> NVDA
- Microsoft -> MSFT
- Amazon -> AMZN
- Google -> GOOG

**EXAMPLES**:
- "Should I buy BTC?" -> {"intent": "signal", "symbol": "BTC-USD"}
- "What do you think about the market?" -> {"intent": "discuss", "symbol": null}
- "Explain the Sharpe ratio" -> {"intent": "explain", "symbol": null}
- "Analyze TSLA" -> {"intent": "analyze", "symbol": "TSLA"}
- "How does reinforcement learning work?" -> {"intent": "explain", "symbol": null}
- "Train a model for ETH" -> {"intent": "train", "symbol": "ETH-USD"}
- "Print a formula" -> {"intent": "explain", "symbol": null}
- "What can you do?" -> {"intent": "discuss", "symbol": null}

Return ONLY valid JSON: {"intent": "signal|portfolio|train|discuss|explain|analyze", "symbol": "SYMBOL-OR-NULL"}"""

        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            
            content = response.content
            if isinstance(content, list):
                content = content[0].get("text", "") if content else ""
            
            # Extract JSON from response (handle markdown code blocks)
            if "```" in content:
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            
            result = json.loads(content)
            intent = result.get("intent", "discuss")
            symbol = result.get("symbol")
            if symbol in ["null", "NULL", "None", "", None]:
                symbol = None
                
            return intent, symbol
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, defaulting to discuss")
            # Default to discussion for general queries
            return "discuss", None
