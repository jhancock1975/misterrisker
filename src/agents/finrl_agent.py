"""FinRL Agent - AI Trading Decisions using Deep Reinforcement Learning.

This agent uses FinRL and stable-baselines3 to provide AI-powered trading
recommendations based on real-time market data.
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
        """Process a query and return trading recommendation.
        
        Args:
            query: User's query about trading
        
        Returns:
            Formatted response with trading signal
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
            
            else:
                # Default: portfolio overview
                portfolio = self.finrl_service.get_portfolio_recommendation()
                return self._format_portfolio(portfolio)
            
        except Exception as e:
            logger.error(f"[FINRL AGENT] Error: {e}")
            return f"❌ Error generating trading signal: {str(e)}"
    
    async def _llm_analyze_query(self, query: str) -> tuple[str, Optional[str]]:
        """Use LLM to analyze query intent and extract symbol.
        
        Args:
            query: User's query
            
        Returns:
            Tuple of (intent, symbol) where intent is 'train', 'portfolio', or 'signal'
        """
        import json
        import asyncio
        
        system_prompt = """Analyze the user's trading query and determine:
1. Intent: "train" (train a model), "portfolio" (overview of all assets), or "signal" (specific asset trading signal)
2. Symbol: If asking about specific asset, extract the symbol (e.g., BTC-USD, ETH-USD, AAPL, NVDA)

Symbol mappings:
- Bitcoin/BTC -> BTC-USD
- Ethereum/ETH -> ETH-USD  
- Solana/SOL -> SOL-USD
- Apple -> AAPL
- Tesla -> TSLA
- Nvidia -> NVDA
- Microsoft -> MSFT
- Amazon -> AMZN
- Google -> GOOG

Return JSON only: {"intent": "train|portfolio|signal", "symbol": "SYMBOL-OR-NULL"}"""

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
            
            result = json.loads(content)
            intent = result.get("intent", "portfolio")
            symbol = result.get("symbol")
            if symbol in ["null", "NULL", "None", ""]:
                symbol = None
                
            return intent, symbol
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, defaulting to portfolio")
            # No fallback to keyword matching - default to portfolio
            return "portfolio", None
