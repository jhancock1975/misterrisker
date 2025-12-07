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
    
    SYMBOL_MAP = {
        "btc": "BTC-USD", "bitcoin": "BTC-USD",
        "eth": "ETH-USD", "ethereum": "ETH-USD",
        "sol": "SOL-USD", "solana": "SOL-USD",
        "xrp": "XRP-USD", "ripple": "XRP-USD",
        "zec": "ZEC-USD", "zcash": "ZEC-USD",
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
        """Extract cryptocurrency symbol from query."""
        query_lower = query.lower()
        
        for keyword, symbol in self.SYMBOL_MAP.items():
            if keyword in query_lower:
                return symbol
        
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
            # Check for specific query types
            query_lower = query.lower()
            
            # Train model request
            if "train" in query_lower and ("model" in query_lower or "finrl" in query_lower):
                symbol = self._extract_symbol(query) or "BTC-USD"
                result = self.finrl_service.train_model(symbol, total_timesteps=10000)
                return f"""🎓 **Model Training Complete**

✅ Trained {result['algorithm']} model for **{result['symbol']}**
⏱️ Training time: {result['training_time']:.1f} seconds
📁 Model saved to: `{result['model_path']}`
🔄 Timesteps: {result['total_timesteps']:,}

The model is now ready to generate trading signals!"""
            
            # Portfolio overview request
            if any(word in query_lower for word in ["portfolio", "all", "overview", "market"]):
                portfolio = self.finrl_service.get_portfolio_recommendation()
                return self._format_portfolio(portfolio)
            
            # Single symbol request
            symbol = self._extract_symbol(query)
            if symbol:
                signal = self.finrl_service.get_trading_signal(symbol)
                return self._format_signal(signal)
            
            # Default: portfolio overview
            portfolio = self.finrl_service.get_portfolio_recommendation()
            return self._format_portfolio(portfolio)
            
        except Exception as e:
            logger.error(f"[FINRL AGENT] Error: {e}")
            return f"❌ Error generating trading signal: {str(e)}"
