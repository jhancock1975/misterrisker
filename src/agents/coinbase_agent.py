"""Coinbase Agent - LangGraph agent for trading on Coinbase.

This module provides a LangGraph-based agent that uses the Coinbase MCP
Server tools to execute trading operations.
"""

import asyncio
import logging
import os
from typing import Any, Literal, TypedDict
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.mcp_servers.coinbase import CoinbaseMCPServer, CoinbaseAPIError
from src.agents.chain_of_thought import ChainOfThought, ReasoningType
from src.services.blockchain_data import BlockchainDataService
from src.services.coinbase_websocket import CoinbaseWebSocketService

# Configure logger for this module
logger = logging.getLogger(__name__)


class CoinbaseToolDecision(BaseModel):
    """Structured output for LLM tool selection in Coinbase agent."""
    capability: str = Field(
        description="The capability to use. Must be one of: portfolio, price, historical, blockchain, analysis, buy, sell, order, orders, fills, transaction_history, cancel"
    )
    crypto: str | None = Field(
        default=None,
        description="The cryptocurrency symbol if relevant (e.g., BTC, ETH, SOL)"
    )
    time_period: str | None = Field(
        default=None,
        description="Time period if relevant (e.g., 'week', 'month', 'year')"
    )
    amount: str | None = Field(
        default=None,
        description="Dollar amount if relevant for trading"
    )
    reasoning: str = Field(
        description="Brief explanation of why this capability was chosen"
    )


class CoinbaseAgentError(Exception):
    """Exception raised by the Coinbase Agent.
    
    Attributes:
        message: Error message
        tool_name: Name of the tool that caused the error (if applicable)
    """
    
    def __init__(self, message: str, tool_name: str | None = None):
        """Initialize CoinbaseAgentError.
        
        Args:
            message: Error message
            tool_name: Name of the tool that caused the error
        """
        self.message = message
        self.tool_name = tool_name
        super().__init__(self.message)


class CoinbaseAgentState(TypedDict, total=False):
    """State for the Coinbase agent workflow.
    
    Attributes:
        messages: Conversation history
        tool_calls: List of tool calls made
        tool_results: Results from tool calls
        current_task: Current task being executed
        portfolio_balance: Current portfolio balance
        positions: Dictionary of currency positions
        reasoning_steps: Chain of thought reasoning steps
    """
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    current_task: str
    portfolio_balance: float
    positions: dict[str, float]
    reasoning_steps: list[str]


class CoinbaseAgent:
    """LangGraph agent for Coinbase trading operations.
    
    This agent uses the CoinbaseMCPServer to execute trading operations
    through a structured workflow.
    
    Attributes:
        mcp_server: The Coinbase MCP Server instance
        llm: Language model for reasoning (optional)
        enable_chain_of_thought: Whether CoT reasoning is enabled
        chain_of_thought: ChainOfThought instance for structured reasoning
        _workflow: The compiled LangGraph workflow
        checkpointer: Optional checkpointer for state persistence
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        mcp_server: CoinbaseMCPServer | None = None,
        llm: Any | None = None,
        enable_chain_of_thought: bool = False,
        checkpointer: Any | None = None,
        enable_websocket: bool = True
    ):
        """Initialize the Coinbase Agent.
        
        Args:
            api_key: Coinbase API key (optional if mcp_server provided)
            api_secret: Coinbase API secret (optional if mcp_server provided)
            mcp_server: Pre-configured MCP server instance (optional)
            llm: Language model for reasoning (optional)
            enable_chain_of_thought: Whether to enable CoT reasoning
            checkpointer: Optional checkpointer for state persistence
            enable_websocket: Whether to enable WebSocket candle monitoring (default: True)
        """
        # Store credentials for WebSocket
        self._api_key = api_key or os.getenv("COINBASE_API_KEY", "")
        self._api_secret = api_secret or os.getenv("COINBASE_API_SECRET", "")
        
        if mcp_server is not None:
            self.mcp_server = mcp_server
        else:
            self.mcp_server = CoinbaseMCPServer(
                api_key=self._api_key,
                api_secret=self._api_secret
            )
        
        self.llm = llm
        self.enable_chain_of_thought = enable_chain_of_thought
        self.chain_of_thought = ChainOfThought() if enable_chain_of_thought else None
        self.checkpointer = checkpointer
        self.blockchain_data_service = BlockchainDataService()
        
        # Set up LLM-based tool router if LLM is provided
        self.tool_router = None
        if self.llm:
            try:
                self.tool_router = self.llm.with_structured_output(CoinbaseToolDecision)
            except Exception as e:
                logger.warning(f"Could not set up LLM tool router: {e}")
        
        # Initialize WebSocket service for candle monitoring
        self.enable_websocket = enable_websocket
        self.websocket_service: CoinbaseWebSocketService | None = None
        if enable_websocket:
            self.websocket_service = CoinbaseWebSocketService(
                api_key=self._api_key,
                api_secret=self._api_secret,
            )
            # Auto-start WebSocket in background
            self._start_websocket_monitoring()
        
        self._workflow = self._build_workflow()
        self._available_tools: list[str] | None = None
    
    def _start_websocket_monitoring(self) -> None:
        """Start WebSocket monitoring in the background.
        
        This schedules the WebSocket connection to start when an event loop is available.
        """
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, schedule the coroutine
            asyncio.create_task(self._async_start_websocket())
        except RuntimeError:
            # No event loop running, we'll start it later when one is available
            logger.info("📊 WebSocket monitoring will start when event loop is available")
    
    async def _async_start_websocket(self) -> None:
        """Async helper to start the WebSocket service."""
        if self.websocket_service and not self.websocket_service.is_running:
            await self.websocket_service.start()
    
    async def ensure_websocket_started(self) -> None:
        """Ensure WebSocket monitoring is started.
        
        Call this method when you have an event loop available.
        """
        if self.websocket_service and not self.websocket_service.is_running:
            await self.websocket_service.start()
    
    def get_available_tools(self) -> list[dict[str, Any]]:
        """Get list of available tools from the MCP server.
        
        Returns:
            List of tool definitions
        """
        return self.mcp_server.list_tools()
    
    def get_workflow(self) -> CompiledStateGraph:
        """Get the compiled LangGraph workflow.
        
        Returns:
            Compiled LangGraph workflow
        """
        return self._workflow
    
    def _build_workflow(self) -> CompiledStateGraph:
        """Build the LangGraph workflow for trading operations.
        
        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(CoinbaseAgentState)
        
        # Add nodes
        workflow.add_node("analyze_task", self._analyze_task)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("process_results", self._process_results)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("analyze_task")
        
        # Add edges
        workflow.add_conditional_edges(
            "analyze_task",
            self._route_after_analyze,
            {
                "execute_tools": "execute_tools",
                "end": END,
                "error": "handle_error",
            }
        )
        
        workflow.add_conditional_edges(
            "execute_tools",
            self._route_after_execute,
            {
                "process_results": "process_results",
                "error": "handle_error",
            }
        )
        
        workflow.add_edge("process_results", END)
        workflow.add_edge("handle_error", END)
        
        # Compile with checkpointer if provided
        if self.checkpointer is not None:
            return workflow.compile(checkpointer=self.checkpointer)
        return workflow.compile()
    
    def _analyze_task(self, state: CoinbaseAgentState) -> CoinbaseAgentState:
        """Analyze the current task and determine actions.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state
        """
        messages = state.get("messages", [])
        current_task = state.get("current_task", "")
        
        # Initialize tool calls list if needed
        if "tool_calls" not in state:
            state["tool_calls"] = []
        
        if "tool_results" not in state:
            state["tool_results"] = []
        
        # Determine what tools to call based on task
        if current_task == "check_balance":
            state["tool_calls"] = [{"tool": "get_accounts", "params": {}}]
        elif current_task == "check_price":
            state["tool_calls"] = [{"tool": "get_product", "params": {"product_id": "BTC-USD"}}]
        elif current_task == "buy_crypto":
            state["tool_calls"] = [
                {"tool": "market_order_buy", "params": {"product_id": "BTC-USD", "quote_size": "100.00"}}
            ]
        
        return state
    
    def _route_after_analyze(self, state: CoinbaseAgentState) -> str:
        """Route after task analysis.
        
        Args:
            state: Current workflow state
        
        Returns:
            Name of next node
        """
        if state.get("tool_calls"):
            return "execute_tools"
        return "end"
    
    def _execute_tools(self, state: CoinbaseAgentState) -> CoinbaseAgentState:
        """Execute the queued tool calls.
        
        Note: This is synchronous in the workflow, actual async calls
        are made through execute_tool method.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state
        """
        # In the workflow, we mark that tools should be executed
        # The actual execution happens asynchronously
        return state
    
    def _route_after_execute(self, state: CoinbaseAgentState) -> str:
        """Route after tool execution.
        
        Args:
            state: Current workflow state
        
        Returns:
            Name of next node
        """
        if state.get("tool_results"):
            return "process_results"
        return "error"
    
    def _process_results(self, state: CoinbaseAgentState) -> CoinbaseAgentState:
        """Process tool results and update state.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state
        """
        results = state.get("tool_results", [])
        messages = state.get("messages", [])
        
        # Apply chain of thought if enabled
        if self.enable_chain_of_thought and self.chain_of_thought:
            state = self._apply_chain_of_thought(state)
        
        # Add results as assistant message
        if results:
            messages.append({
                "role": "assistant",
                "content": f"Tool results: {results}"
            })
        
        state["messages"] = messages
        return state
    
    def _apply_chain_of_thought(self, state: CoinbaseAgentState) -> CoinbaseAgentState:
        """Apply chain of thought reasoning to process results.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with reasoning steps
        """
        if not self.chain_of_thought or not self.llm:
            return state
        
        task = state.get("current_task", "")
        results = state.get("tool_results", [])
        
        # Detect reasoning type based on task
        reasoning_type = self.chain_of_thought.detect_reasoning_type(task)
        
        # Prepare data from tool results
        data = {}
        for result in results:
            if isinstance(result, dict):
                data.update(result)
        
        # Add portfolio context if available
        portfolio_context = None
        if state.get("positions"):
            portfolio_context = state["positions"]
        
        # Generate CoT prompt
        prompt = self.chain_of_thought.get_reasoning_prompt(
            query=task,
            data=data,
            reasoning_type=reasoning_type,
            portfolio_context=portfolio_context
        )
        
        try:
            # Get LLM response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Parse the response
            parsed = self.chain_of_thought.parse_response(response_text)
            
            # Update state with reasoning
            state["reasoning_steps"] = parsed.get("reasoning_steps", [])
        except Exception:
            # If CoT fails, continue without it
            state["reasoning_steps"] = []
        
        return state
    
    def _handle_error(self, state: CoinbaseAgentState) -> CoinbaseAgentState:
        """Handle errors in the workflow.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state
        """
        messages = state.get("messages", [])
        messages.append({
            "role": "assistant",
            "content": "An error occurred while processing your request."
        })
        state["messages"] = messages
        return state
    
    # ===================
    # Public API Methods
    # ===================
    
    async def execute_tool(
        self,
        tool_name: str,
        params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a specific MCP tool.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
        
        Returns:
            Tool execution result
        
        Raises:
            CoinbaseAgentError: If tool execution fails
        """
        # Validate tool exists
        available_tools = [t["name"] for t in self.get_available_tools()]
        if tool_name not in available_tools:
            raise CoinbaseAgentError(f"Unknown tool: {tool_name}", tool_name=tool_name)
        
        try:
            result = await self.mcp_server.call_tool(tool_name, params)
            return result
        except Exception as e:
            raise CoinbaseAgentError(f"Tool execution failed: {str(e)}", tool_name=tool_name)
    
    async def run(self, initial_state: CoinbaseAgentState) -> CoinbaseAgentState:
        """Run the agent workflow.
        
        Args:
            initial_state: Initial state for the workflow
        
        Returns:
            Final state after workflow execution
        """
        # Execute any tool calls that are queued
        if initial_state.get("tool_calls"):
            tool_results = []
            for tool_call in initial_state["tool_calls"]:
                try:
                    result = await self.execute_tool(
                        tool_call["tool"],
                        tool_call["params"]
                    )
                    tool_results.append(result)
                except CoinbaseAgentError:
                    pass  # Handle errors gracefully
            initial_state["tool_results"] = tool_results
        
        # Run the workflow
        result = self._workflow.invoke(initial_state)
        return result
    
    async def process_query(
        self,
        query: str,
        config: dict[str, Any] | None = None,
        conversation_history: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Process a natural language query using LLM-based tool selection.
        
        This method uses an LLM to intelligently decide which capability to use
        based on the user's query, rather than relying on keyword matching.
        
        Args:
            query: Natural language query from the user
            config: LangGraph config with thread_id for checkpointing (optional)
            conversation_history: Previous messages for context (optional)
        
        Returns:
            Dict with 'response' and 'status' keys
        """
        try:
            # LLM-based routing is required
            if not self.tool_router:
                raise CoinbaseAgentError("LLM is required for query processing. No LLM configured.")
            
            decision = await self._llm_route_query(query, conversation_history)
            response = await self._execute_capability(decision, query)
            
            return {
                "response": response,
                "status": "success"
            }
        except CoinbaseAgentError as e:
            return {
                "response": f"Error accessing Coinbase: {str(e)}",
                "status": "error"
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"An error occurred: {str(e)}",
                "status": "error"
            }
    
    async def _llm_route_query(
        self, 
        query: str, 
        conversation_history: list[dict[str, Any]] | None = None
    ) -> CoinbaseToolDecision:
        """Use LLM to intelligently route the query to the right capability.
        
        Args:
            query: User's natural language query
            conversation_history: Previous messages for context
            
        Returns:
            CoinbaseToolDecision with the selected capability
        """
        routing_prompt = self._build_capability_routing_prompt()
        
        messages = [
            {"role": "system", "content": routing_prompt}
        ]
        
        # Add conversation history for context if available
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                messages.append(msg)
        
        messages.append({"role": "user", "content": query})
        
        try:
            decision = await asyncio.to_thread(
                self.tool_router.invoke, messages
            )
            logger.info(f"[COINBASE] LLM routing: capability={decision.capability}, crypto={decision.crypto}, reason={decision.reasoning[:80]}...")
            return decision
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            raise CoinbaseAgentError(f"Failed to route query via LLM: {e}")
    
    def _build_capability_routing_prompt(self) -> str:
        """Build a comprehensive prompt for LLM-based capability routing."""
        return """You are the Coinbase Agent's intelligent router. Your job is to analyze user queries 
and decide which capability to use to best answer them.

## Available Capabilities

### portfolio
Get user's Coinbase account balances, holdings, and portfolio summary.
Use when: User asks about their balance, account, holdings, portfolio value, how much they have.

### price  
Get current real-time price for a cryptocurrency.
Use when: User asks for current price, quote, how much something costs right now.

### historical
Get historical price data with statistics (mean, min, max, std dev, median).
Use when: User asks about price history, past prices, historical data, statistics over time, 
trends over weeks/months/years, wants to analyze past performance, mentions "last month", 
"past week", "historical", or wants statistical analysis.

### blockchain
Get on-chain blockchain data: transactions, blocks, network stats, mempool info.
Use when: User asks about blockchain transactions, block data, on-chain activity, 
network statistics, mempool, or compares blockchain activity between chains.

### chart
Generate a data visualization chart or plot with actual data.
Use when: User asks to "draw", "plot", "chart", "graph", or "visualize" price data,
correlation, performance, or any quantitative crypto data. This generates an actual 
SVG chart with real data, not a generic image.

### analysis
Real-time technical analysis with current price patterns, trends, volatility.
Use when: User asks to "analyze" current market conditions, patterns, trends, 
what's happening right now, technical indicators.

### buy
Execute a cryptocurrency purchase.
Use when: User wants to buy crypto with a specific dollar amount.

### sell
Execute a cryptocurrency sale.  
Use when: User wants to sell crypto for dollars.

### order
Place a limit order (buy or sell at a specific price).
Use when: User wants to place an order at a specific limit price.

### orders
List and check status of existing orders.
Use when: User asks about their orders, order status, open orders, pending orders, 
what orders they have, or wants to see their order history.

### fills
Get fill history - executed trades/order fills.
Use when: User asks about filled orders, executed trades, trade history,
what trades have been completed.

### transaction_history
Get account transaction history - deposits, withdrawals, fees, trades.
Use when: User asks about transaction history, account activity, deposits,
withdrawals, fee summary, or general account transactions.

### cancel
Cancel an existing order.
Use when: User wants to cancel an order.

## Instructions
1. Analyze the user's query carefully
2. Consider what data or action would best satisfy their request
3. Select the most appropriate capability
4. Extract relevant parameters (crypto symbol, time period, amounts)

## Examples
- "What's my balance?" → portfolio
- "BTC price" → price
- "Historical bitcoin prices from last month with statistics" → historical (crypto=BTC, time_period=month)
- "Show me Solana blockchain transactions" → blockchain (crypto=SOL)
- "Draw a chart of BTC prices" → chart (crypto=BTC)
- "Plot the price correlation" → chart
- "Visualize ETH performance" → chart (crypto=ETH)
- "Analyze BTC trends" with no time period mentioned → analysis (current real-time)
- "How has ETH performed over the past year?" → historical (crypto=ETH, time_period=year)
- "Buy $100 of ETH" → buy (crypto=ETH, amount=100)
- "Compare Bitcoin and Solana blockchains" → blockchain (crypto=BTC,SOL)
- "What are the largest Bitcoin transactions today?" → blockchain (crypto=BTC)
- "Mean and standard deviation of BTC prices this month" → historical (crypto=BTC, time_period=month)
- "What are my orders?" → orders
- "Show me my order status" → orders
- "Status of my orders in coinbase" → orders
- "Do I have any open orders?" → orders
- "Show me my transaction history" → transaction_history
- "What's my account activity?" → transaction_history
- "Show my fills" → fills
- "What trades have been executed?" → fills
- "Show me my recent trades" → fills

Be precise in your selection. If the user mentions historical/past data with statistics, use 'historical'.
If they want current real-time analysis, use 'analysis'. These are different capabilities.
If they want a visual chart/plot/graph, use 'chart'.
If they ask about existing orders or order status, use 'orders'.
If they ask about transaction history or account activity, use 'transaction_history'.
If they ask about filled trades or executed orders, use 'fills'."""
    
    async def _execute_capability(self, decision: CoinbaseToolDecision, query: str) -> str:
        """Execute the selected capability based on LLM decision.
        
        Args:
            decision: The LLM's routing decision
            query: Original user query
            
        Returns:
            Formatted response string
        """
        capability = decision.capability
        crypto = decision.crypto
        
        if capability == "portfolio":
            result = await self.get_portfolio_summary()
            return f"Here's your Coinbase portfolio:\n{self._format_portfolio(result)}"
        
        elif capability == "price":
            crypto = crypto or "BTC"
            product_id = self._normalize_product_id(crypto)
            result = await self.get_market_data(product_id)
            return self._format_price(crypto, result)
        
        elif capability == "historical":
            return await self._handle_historical_query(query)
        
        elif capability == "blockchain":
            return await self._handle_blockchain_query(query)
        
        elif capability == "chart":
            return await self._handle_chart_request(query)
        
        elif capability == "analysis":
            return await self._handle_analysis_query(query)
        
        elif capability == "buy":
            return await self._handle_buy_request(query)
        
        elif capability == "sell":
            return await self._handle_sell_request(query)
        
        elif capability == "order":
            return await self._handle_order_request(query)
        
        elif capability == "orders":
            return await self._handle_orders_status_query(query)
        
        elif capability == "cancel":
            return await self._handle_cancel_request(query)
        
        elif capability == "transaction_history":
            return await self._handle_transaction_history_query(query)
        
        elif capability == "fills":
            return await self._handle_fills_query(query)
        
        else:
            # No fallback - require LLM to select a valid capability
            raise ValueError(f"Unknown capability: {capability}. Valid capabilities are: portfolio, price, historical, blockchain, analysis, buy, sell, order, orders, fills, transaction_history, cancel")
    
    def _extract_crypto(self, query: str) -> str | None:
        """Extract cryptocurrency name/symbol from a query string."""
        query_lower = query.lower()
        
        crypto_map = {
            "bitcoin": "BTC", "btc": "BTC",
            "ethereum": "ETH", "eth": "ETH",
            "solana": "SOL", "sol": "SOL",
            "dogecoin": "DOGE", "doge": "DOGE",
            "litecoin": "LTC", "ltc": "LTC",
            "ripple": "XRP", "xrp": "XRP",
            "cardano": "ADA", "ada": "ADA",
            "zcash": "ZEC", "zec": "ZEC",
            "avalanche": "AVAX", "avax": "AVAX",
            "polkadot": "DOT", "dot": "DOT",
            "chainlink": "LINK", "link": "LINK",
            "polygon": "MATIC", "matic": "MATIC",
            "uniswap": "UNI", "uni": "UNI",
            "shiba": "SHIB", "shib": "SHIB",
        }
        
        for name, symbol in crypto_map.items():
            if name in query_lower:
                return symbol
        return None

    def _extract_amount(self, query: str) -> tuple[str | None, str | None]:
        """Extract dollar amount and limit price from a query string.
        
        Returns:
            Tuple of (amount, limit_price) or (None, None) if not found
        """
        import re
        
        # Look for dollar amounts like $200, $83,000, etc.
        amounts = re.findall(r'\$[\d,]+(?:\.\d+)?', query)
        
        # Clean up the amounts (remove $ and commas)
        cleaned = [a.replace('$', '').replace(',', '') for a in amounts]
        
        if len(cleaned) >= 2:
            # First amount is usually the order size, second is limit price
            return cleaned[0], cleaned[1]
        elif len(cleaned) == 1:
            return cleaned[0], None
        return None, None
    
    def _normalize_product_id(self, crypto: str | None) -> str:
        """Normalize crypto symbol to Coinbase product_id format.
        
        Ensures we don't get double suffixes like ZEC-USD-USD.
        
        Args:
            crypto: Crypto symbol (e.g., 'BTC', 'ZEC-USD', 'eth')
            
        Returns:
            Normalized product_id (e.g., 'BTC-USD')
        """
        if not crypto:
            return "BTC-USD"
        
        crypto = crypto.upper().strip()
        
        # Already has a trading pair suffix
        if any(crypto.endswith(suffix) for suffix in ['-USD', '-USDT', '-BTC', '-EUR']):
            return crypto
        
        return f"{crypto}-USD"

    def _extract_order_id(self, result: dict[str, Any]) -> str:
        """Extract order ID from a Coinbase API response.
        
        The order ID can be in different places depending on the response structure:
        - Directly at result['order_id']
        - Nested in result['success_response']['order_id']
        
        Args:
            result: API response dictionary
            
        Returns:
            Order ID string or 'unknown' if not found
        """
        # Try direct order_id first
        if result.get("order_id"):
            return result["order_id"]
        
        # Try nested in success_response
        success_response = result.get("success_response")
        if success_response:
            if isinstance(success_response, dict):
                if success_response.get("order_id"):
                    return success_response["order_id"]
            elif hasattr(success_response, "order_id") and success_response.order_id:
                return success_response.order_id
        
        return "unknown"

    async def _handle_order_request(self, query: str) -> str:
        """Handle order placement requests."""
        query_lower = query.lower()
        crypto = self._extract_crypto(query) or "BTC"
        amount, limit_price = self._extract_amount(query)
        
        is_buy = "buy" in query_lower or "purchase" in query_lower
        is_limit = "limit" in query_lower
        
        if not amount:
            return f"Please specify an amount for your {crypto} order (e.g., 'buy $200 of BTC at $83,000')."
        
        product_id = self._normalize_product_id(crypto)
        
        try:
            if is_limit and limit_price:
                # Place limit order
                # Calculate base_size with 2 decimal precision (Coinbase requirement)
                base_size = round(float(amount) / float(limit_price), 8)  # Use 8 decimals for crypto
                base_size_str = f"{base_size:.8f}".rstrip('0').rstrip('.')
                # Ensure at least some precision
                if '.' not in base_size_str:
                    base_size_str = f"{base_size:.2f}"
                
                if is_buy:
                    result = await self.execute_tool("limit_order_gtc_buy", {
                        "product_id": product_id,
                        "base_size": base_size_str,
                        "limit_price": limit_price
                    })
                else:
                    result = await self.execute_tool("limit_order_gtc_sell", {
                        "product_id": product_id,
                        "base_size": base_size_str,
                        "limit_price": limit_price
                    })
                order_type = "buy" if is_buy else "sell"
                if result.get("success") or result.get("order_id"):
                    order_id = self._extract_order_id(result)
                    return f"✅ Limit {order_type} order placed for ${amount} of {crypto} at ${limit_price}\nOrder ID: {order_id}"
                else:
                    error_msg = result.get("error_response", {}).get("message", str(result))
                    return f"❌ Failed to place limit order: {error_msg}"
            else:
                # Place market order
                if is_buy:
                    result = await self.place_market_buy(product_id, amount)
                else:
                    result = await self.place_market_sell(product_id, amount)
                
                if result.get("success") or result.get("order_id"):
                    order_id = self._extract_order_id(result)
                    action = "bought" if is_buy else "sold"
                    return f"✅ Market order placed: {action} ${amount} of {crypto}\nOrder ID: {order_id}"
                else:
                    return f"❌ Failed to place market order: {result}"
        except Exception as e:
            return f"❌ Error placing order: {str(e)}"

    async def _handle_buy_request(self, query: str) -> str:
        """Handle buy requests with amounts."""
        crypto = self._extract_crypto(query) or "BTC"
        amount, limit_price = self._extract_amount(query)
        
        if not amount:
            return f"Please specify an amount to buy (e.g., 'buy $100 of {crypto}')."
        
        if limit_price:
            # This is a limit order
            return await self._handle_order_request(query)
        
        # Market buy
        try:
            product_id = self._normalize_product_id(crypto)
            result = await self.place_market_buy(product_id, amount)
            if result.get("success") or result.get("order_id"):
                order_id = self._extract_order_id(result)
                return f"✅ Market buy order placed for ${amount} of {crypto}\nOrder ID: {order_id}"
            else:
                return f"❌ Failed to place buy order: {result}"
        except Exception as e:
            return f"❌ Error placing buy order: {str(e)}"

    async def _handle_sell_request(self, query: str) -> str:
        """Handle sell requests with amounts."""
        crypto = self._extract_crypto(query) or "BTC"
        amount, _ = self._extract_amount(query)
        
        if not amount:
            return f"Please specify an amount to sell (e.g., 'sell $100 of {crypto}')."
        
        try:
            product_id = self._normalize_product_id(crypto)
            result = await self.place_market_sell(product_id, amount)
            if result.get("success") or result.get("order_id"):
                order_id = self._extract_order_id(result)
                return f"✅ Market sell order placed for ${amount} of {crypto}\nOrder ID: {order_id}"
            else:
                return f"❌ Failed to place sell order: {result}"
        except Exception as e:
            return f"❌ Error placing sell order: {str(e)}"

    async def _handle_orders_status_query(self, query: str) -> str:
        """Handle order status queries - list existing orders.
        
        Args:
            query: User query about their orders
            
        Returns:
            Formatted list of orders with status
        """
        try:
            # Get all orders (no filter = all statuses)
            result = await self.execute_tool("list_orders", {})
            
            if not result:
                return "📋 You don't have any orders in your Coinbase account."
            
            orders = result.get("orders", result) if isinstance(result, dict) else result
            
            if not orders or (isinstance(orders, list) and len(orders) == 0):
                return "📋 You don't have any orders in your Coinbase account."
            
            if not isinstance(orders, list):
                orders = [orders]
            
            # Format the orders
            lines = ["📋 **Your Coinbase Orders:**\n"]
            
            for order in orders[:20]:  # Limit to 20 orders
                order_id = order.get("order_id", order.get("id", "unknown"))
                product_id = order.get("product_id", "unknown")
                side = order.get("side", "unknown").upper()
                status = order.get("status", "unknown").upper()
                
                # Get order type and price info
                order_config = order.get("order_configuration", {})
                order_type = "MARKET"
                limit_price = None
                
                if "limit_limit_gtc" in order_config:
                    order_type = "LIMIT GTC"
                    limit_price = order_config["limit_limit_gtc"].get("limit_price")
                elif "limit_limit_gtd" in order_config:
                    order_type = "LIMIT GTD"
                    limit_price = order_config["limit_limit_gtd"].get("limit_price")
                elif "market_market_ioc" in order_config:
                    order_type = "MARKET"
                
                # Get size/amount
                size = order.get("filled_size", order.get("base_size", "0"))
                quote_size = order.get("filled_value", order.get("quote_size", ""))
                
                # Status emoji
                status_emoji = {
                    "PENDING": "⏳",
                    "OPEN": "📂",
                    "FILLED": "✅",
                    "CANCELLED": "❌",
                    "CANCELED": "❌",
                    "EXPIRED": "⌛",
                    "FAILED": "💥"
                }.get(status, "❓")
                
                line = f"{status_emoji} **{side}** {product_id} | {order_type}"
                if limit_price:
                    line += f" @ ${float(limit_price):,.2f}"
                if size and float(size) > 0:
                    line += f" | Size: {size}"
                if quote_size and float(quote_size) > 0:
                    line += f" | Value: ${float(quote_size):,.2f}"
                line += f" | Status: {status}"
                line += f"\n   ID: `{order_id[:12]}...`"
                
                lines.append(line)
            
            if len(orders) > 20:
                lines.append(f"\n_...and {len(orders) - 20} more orders_")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return f"❌ Error fetching orders: {str(e)}"

    async def _handle_cancel_request(self, query: str) -> str:
        """Handle order cancellation requests."""
        import re
        
        # Extract order ID (UUID format)
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        matches = re.findall(uuid_pattern, query.lower())
        
        if not matches:
            return "Please provide the order ID to cancel (e.g., 'cancel order 3fd79d98-efff-4327-94be-dd02f986a85b')."
        
        order_ids = matches
        
        try:
            result = await self.execute_tool("cancel_orders", {"order_ids": order_ids})
            
            # Check the result
            results = result.get("results", [])
            if results:
                successes = [r for r in results if r.get("success")]
                failures = [r for r in results if not r.get("success")]
                
                response_parts = []
                if successes:
                    for s in successes:
                        response_parts.append(f"✅ Order {s.get('order_id', 'unknown')} cancelled")
                if failures:
                    for f in failures:
                        reason = f.get("failure_reason", {}).get("message", "Unknown error")
                        response_parts.append(f"❌ Failed to cancel {f.get('order_id', 'unknown')}: {reason}")
                
                return "\n".join(response_parts) if response_parts else "Order cancellation processed."
            else:
                # Simple success/failure check
                if result.get("success"):
                    return f"✅ Order(s) cancelled: {', '.join(order_ids)}"
                else:
                    return f"❌ Failed to cancel order(s): {result}"
        except Exception as e:
            return f"❌ Error cancelling order: {str(e)}"

    async def _handle_transaction_history_query(self, query: str) -> str:
        """Handle transaction history/summary queries.
        
        Args:
            query: User query about transaction history or summary
            
        Returns:
            Formatted transaction summary
        """
        try:
            result = await self.execute_tool("get_transaction_summary", {})
            
            if not result:
                return "📊 No transaction summary available."
            
            lines = ["📊 **Your Coinbase Transaction Summary:**\n"]
            
            # Parse transaction summary response
            total_volume = result.get("total_volume", 0)
            total_fees = result.get("total_fees", 0)
            fee_tier = result.get("fee_tier", {})
            margin_rate = result.get("margin_rate", {})
            goods_and_services_tax = result.get("goods_and_services_tax", {})
            advanced_trade_only_volume = result.get("advanced_trade_only_volume", 0)
            advanced_trade_only_fees = result.get("advanced_trade_only_fees", 0)
            
            if total_volume:
                lines.append(f"💰 **Total Volume:** ${float(total_volume):,.2f}")
            if total_fees:
                lines.append(f"💸 **Total Fees:** ${float(total_fees):,.2f}")
            if advanced_trade_only_volume:
                lines.append(f"📈 **Advanced Trade Volume:** ${float(advanced_trade_only_volume):,.2f}")
            if advanced_trade_only_fees:
                lines.append(f"📉 **Advanced Trade Fees:** ${float(advanced_trade_only_fees):,.2f}")
            
            if fee_tier:
                pricing_tier = fee_tier.get("pricing_tier", "unknown")
                maker_fee_rate = fee_tier.get("maker_fee_rate", "N/A")
                taker_fee_rate = fee_tier.get("taker_fee_rate", "N/A")
                lines.append(f"\n**Fee Tier:** {pricing_tier}")
                lines.append(f"  • Maker Fee: {maker_fee_rate}")
                lines.append(f"  • Taker Fee: {taker_fee_rate}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error fetching transaction summary: {e}")
            return f"❌ Error fetching transaction summary: {str(e)}"

    async def _handle_fills_query(self, query: str) -> str:
        """Handle fills (executed trades) queries.
        
        Args:
            query: User query about fills/executed trades
            
        Returns:
            Formatted list of fills
        """
        try:
            # Try to extract a specific product/crypto if mentioned
            crypto = self._extract_crypto(query)
            product_id = self._normalize_product_id(crypto) if crypto else None
            
            params = {}
            if product_id:
                params["product_id"] = product_id
            
            result = await self.execute_tool("get_fills", params)
            
            if not result:
                return "📋 No fills/executed trades found."
            
            fills = result.get("fills", result) if isinstance(result, dict) else result
            
            if not fills or (isinstance(fills, list) and len(fills) == 0):
                return "📋 No fills/executed trades found."
            
            if not isinstance(fills, list):
                fills = [fills]
            
            lines = ["📋 **Your Recent Fills (Executed Trades):**\n"]
            
            for fill in fills[:25]:  # Limit to 25 fills
                entry_id = fill.get("entry_id", fill.get("id", "unknown"))
                trade_id = fill.get("trade_id", "")
                order_id = fill.get("order_id", "")
                trade_time = fill.get("trade_time", "")
                trade_type = fill.get("trade_type", "unknown")
                price = fill.get("price", "0")
                size = fill.get("size", "0")
                commission = fill.get("commission", "0")
                product_id_fill = fill.get("product_id", "unknown")
                side = fill.get("side", "UNKNOWN").upper()
                
                # Side emoji
                side_emoji = "🟢" if side == "BUY" else "🔴" if side == "SELL" else "❓"
                
                line = f"{side_emoji} **{side}** {product_id_fill}"
                if size:
                    line += f" | Size: {size}"
                if price:
                    line += f" @ ${float(price):,.2f}"
                if commission:
                    line += f" | Fee: ${float(commission):,.4f}"
                if trade_time:
                    # Format the timestamp
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))
                        line += f"\n   Time: {dt.strftime('%Y-%m-%d %H:%M')}"
                    except Exception:
                        line += f"\n   Time: {trade_time[:16]}"
                
                lines.append(line)
            
            if len(fills) > 25:
                lines.append(f"\n_...and {len(fills) - 25} more fills_")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error fetching fills: {e}")
            return f"❌ Error fetching fills: {str(e)}"

    async def _handle_historical_query(self, query: str) -> str:
        """Handle historical price data queries with LLM-based analysis.
        
        Fetches historical candle data and uses LLM to provide intelligent
        analysis, not just raw statistics.
        
        Args:
            query: User query about historical prices
            
        Returns:
            LLM-generated analysis of historical data
        """
        from datetime import datetime, timedelta
        from langchain_core.messages import SystemMessage, HumanMessage
        import statistics
        
        query_lower = query.lower()
        
        # Determine which crypto
        crypto = self._extract_crypto(query) or "BTC"
        product_id = self._normalize_product_id(crypto)
        
        # Determine time period and granularity to stay under 350 candles limit
        if "year" in query_lower:
            days = 365
            period_name = "1 Year"
            granularity = "ONE_DAY"  # 365 candles
        elif "month" in query_lower:
            days = 30
            period_name = "1 Month"
            granularity = "SIX_HOUR"  # 120 candles (30*4)
        elif "week" in query_lower:
            days = 7
            period_name = "1 Week"
            granularity = "ONE_HOUR"  # 168 candles (7*24)
        else:
            days = 30
            period_name = "1 Month"
            granularity = "SIX_HOUR"  # 120 candles (30*4)
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Format timestamps for Coinbase API (Unix timestamp as string)
        start_ts = str(int(start_time.timestamp()))
        end_ts = str(int(end_time.timestamp()))
        
        try:
            # Fetch candles from Coinbase MCP Server
            candles_result = await self.mcp_server.call_tool(
                "get_candles",
                {
                    "product_id": product_id,
                    "start": start_ts,
                    "end": end_ts,
                    "granularity": granularity
                }
            )
            
            # Extract candles data
            if hasattr(candles_result, 'candles'):
                candles = candles_result.candles
            elif isinstance(candles_result, dict) and 'candles' in candles_result:
                candles = candles_result['candles']
            else:
                candles = candles_result if isinstance(candles_result, list) else []
            
            if not candles:
                return f"❌ No historical data available for {crypto} over the past {period_name}."
            
            # Extract closing prices
            close_prices = []
            high_prices = []
            low_prices = []
            volumes = []
            timestamps = []
            
            for candle in candles:
                if hasattr(candle, 'close'):
                    close_prices.append(float(candle.close))
                    high_prices.append(float(candle.high))
                    low_prices.append(float(candle.low))
                    volumes.append(float(candle.volume))
                    timestamps.append(int(candle.start))
                elif isinstance(candle, dict):
                    close_prices.append(float(candle.get('close', 0)))
                    high_prices.append(float(candle.get('high', 0)))
                    low_prices.append(float(candle.get('low', 0)))
                    volumes.append(float(candle.get('volume', 0)))
                    timestamps.append(int(candle.get('start', 0)))
                elif isinstance(candle, (list, tuple)) and len(candle) >= 5:
                    # [timestamp, low, high, open, close, volume]
                    close_prices.append(float(candle[4]))
                    high_prices.append(float(candle[2]))
                    low_prices.append(float(candle[1]))
                    volumes.append(float(candle[5]) if len(candle) > 5 else 0)
                    timestamps.append(int(candle[0]))
            
            if not close_prices:
                return f"❌ Unable to parse historical data for {crypto}."
            
            # Calculate statistics
            num_samples = len(close_prices)
            price_mean = statistics.mean(close_prices)
            price_min = min(close_prices)
            price_max = max(close_prices)
            price_median = statistics.median(close_prices)
            price_std = statistics.stdev(close_prices) if len(close_prices) > 1 else 0
            total_volume = sum(volumes)
            
            # Current vs start price
            current_price = close_prices[0] if timestamps[0] > timestamps[-1] else close_prices[-1]
            start_price = close_prices[-1] if timestamps[0] > timestamps[-1] else close_prices[0]
            price_change = current_price - start_price
            price_change_pct = (price_change / start_price) * 100 if start_price else 0
            
            # Overall high/low
            overall_high = max(high_prices)
            overall_low = min(low_prices)
            volatility_pct = (price_std / price_mean * 100) if price_mean else 0
            
            # Build statistics summary for LLM
            stats_summary = f"""
{crypto} Historical Data ({period_name}):
- Period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}
- Data Points: {num_samples} candles
- Current Price: ${current_price:,.2f}
- Start Price: ${start_price:,.2f}
- Change: ${price_change:+,.2f} ({price_change_pct:+.2f}%)
- Mean: ${price_mean:,.2f}
- Median: ${price_median:,.2f}
- Min: ${price_min:,.2f}
- Max: ${price_max:,.2f}
- Std Dev: ${price_std:,.2f}
- Volatility: {volatility_pct:.2f}%
- Total Volume: {total_volume:,.2f}
"""
            
            # Sample price series for LLM to see trends
            if len(close_prices) > 20:
                step = len(close_prices) // 20
                sampled_prices = close_prices[::step]
            else:
                sampled_prices = close_prices
            
            price_series = "Price samples (chronological): " + " → ".join([f"${p:,.2f}" for p in sampled_prices[-20:]])
            
            # Use LLM to analyze if available
            if self.llm:
                system_prompt = f"""You are a cryptocurrency analyst. Analyze the provided {crypto} historical data and answer the user's question.

Provide intelligent analysis, not just a repeat of the statistics. Consider:
- What do these numbers tell us about the asset's behavior?
- Are there notable patterns or trends?
- What does the volatility suggest?
- How does current price compare to the historical average?
- Any insights for traders based on this data?

Be concise but insightful. Reference specific numbers to support your points."""

                user_prompt = f"""User question: {query}

{stats_summary}

{price_series}

Provide analysis addressing the user's question."""

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = await self.llm.ainvoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                
                return f"📈 **{crypto} Historical Analysis** ({period_name})\n\n{content}"
            else:
                # Fallback to basic stats if no LLM
                return f"📈 **{crypto}** ({period_name}): ${current_price:,.2f} (change: {price_change_pct:+.2f}%)"
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return f"❌ Error fetching historical data for {crypto}: {str(e)}"

    async def _handle_chart_request(self, query: str) -> str:
        """Handle chart/visualization requests by generating SVG charts with real data.
        
        Args:
            query: User query requesting a chart/plot/visualization
            
        Returns:
            SVG chart wrapped in markdown or error message
        """
        from datetime import datetime, timedelta
        
        query_lower = query.lower()
        
        # Determine chart type based on query
        is_correlation = "correlation" in query_lower
        is_comparison = any(word in query_lower for word in ["compare", "comparison", "vs", "versus"])
        
        # Extract cryptos
        cryptos = []
        crypto_keywords = {
            "bitcoin": "BTC", "btc": "BTC",
            "ethereum": "ETH", "eth": "ETH",
            "solana": "SOL", "sol": "SOL",
            "ripple": "XRP", "xrp": "XRP",
        }
        for keyword, symbol in crypto_keywords.items():
            if keyword in query_lower and symbol not in cryptos:
                cryptos.append(symbol)
        
        if not cryptos:
            cryptos = ["BTC", "ETH"]  # Default to BTC vs ETH for correlation
        
        try:
            # Fetch historical data for the cryptos
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)  # 1 week of data
            start_ts = str(int(start_time.timestamp()))
            end_ts = str(int(end_time.timestamp()))
            
            all_data = {}
            for crypto in cryptos[:2]:  # Max 2 for comparison
                product_id = self._normalize_product_id(crypto)
                candles_result = await self.mcp_server.call_tool(
                    "get_candles",
                    {
                        "product_id": product_id,
                        "start": start_ts,
                        "end": end_ts,
                        "granularity": "SIX_HOUR"
                    }
                )
                
                # Extract close prices
                prices = []
                if isinstance(candles_result, dict) and 'candles' in candles_result:
                    for candle in candles_result['candles']:
                        if isinstance(candle, dict):
                            prices.append(float(candle.get('close', 0)))
                        elif hasattr(candle, 'close'):
                            prices.append(float(candle.close))
                
                if prices:
                    all_data[crypto] = prices[:28]  # Last 28 data points (7 days * 4)
            
            if not all_data:
                return "❌ Unable to fetch price data for chart generation."
            
            # Generate SVG chart
            if is_correlation and len(all_data) >= 2:
                svg = self._generate_correlation_chart(all_data)
                title = f"Price Correlation: {' vs '.join(all_data.keys())}"
            else:
                svg = self._generate_price_chart(all_data)
                title = f"Price Chart: {', '.join(all_data.keys())}"
            
            # Return SVG wrapped for frontend
            return f"""📊 **{title}**

<div class="generated-chart">
{svg}
</div>

*Data: Last 7 days, 6-hour intervals from Coinbase*"""
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return f"❌ Error generating chart: {str(e)}"
    
    def _generate_price_chart(self, data: dict[str, list[float]]) -> str:
        """Generate an SVG line chart for price data.
        
        Args:
            data: Dict of crypto symbol to list of prices
            
        Returns:
            SVG string
        """
        width, height = 1200, 600
        margin = 100
        chart_width = width - 2 * margin
        chart_height = height - 2 * margin
        
        # Find global min/max for scaling
        all_prices = [p for prices in data.values() for p in prices]
        min_price = min(all_prices) * 0.99
        max_price = max(all_prices) * 1.01
        price_range = max_price - min_price
        
        colors = ["#2563eb", "#dc2626", "#16a34a", "#ca8a04"]
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>',
            # Grid lines
            f'<g stroke="#333" stroke-width="1">'
        ]
        
        for i in range(5):
            y = margin + (i * chart_height / 4)
            svg_parts.append(f'<line x1="{margin}" y1="{y}" x2="{width-margin}" y2="{y}"/>')
        svg_parts.append('</g>')
        
        # Price lines
        for idx, (crypto, prices) in enumerate(data.items()):
            color = colors[idx % len(colors)]
            points = []
            num_points = len(prices)
            
            for i, price in enumerate(reversed(prices)):  # Reverse for chronological
                x = margin + (i * chart_width / (num_points - 1)) if num_points > 1 else margin
                y = margin + chart_height - ((price - min_price) / price_range * chart_height)
                points.append(f"{x:.1f},{y:.1f}")
            
            svg_parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="4"/>')
            
            # Legend
            legend_y = margin + 40 + idx * 40
            svg_parts.append(f'<rect x="{margin + 20}" y="{legend_y - 18}" width="28" height="28" fill="{color}"/>')
            svg_parts.append(f'<text x="{margin + 60}" y="{legend_y}" fill="white" font-size="20">{crypto}</text>')
        
        # Axis labels
        svg_parts.append(f'<text x="{width/2}" y="{height - 20}" fill="white" font-size="20" text-anchor="middle">Time (7 days)</text>')
        svg_parts.append(f'<text x="30" y="{height/2}" fill="white" font-size="20" transform="rotate(-90, 30, {height/2})">Price (USD)</text>')
        
        # Price labels
        for i in range(5):
            y = margin + (i * chart_height / 4)
            price_val = max_price - (i * price_range / 4)
            svg_parts.append(f'<text x="{margin - 10}" y="{y + 6}" fill="white" font-size="16" text-anchor="end">${price_val:,.0f}</text>')
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)
    
    def _generate_correlation_chart(self, data: dict[str, list[float]]) -> str:
        """Generate an SVG scatter plot showing price correlation.
        
        Args:
            data: Dict of crypto symbol to list of prices (needs exactly 2)
            
        Returns:
            SVG string
        """
        if len(data) < 2:
            return self._generate_price_chart(data)
        
        symbols = list(data.keys())[:2]
        prices1 = data[symbols[0]]
        prices2 = data[symbols[1]]
        
        # Normalize prices to percentage changes for correlation
        def pct_changes(prices):
            return [(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
        
        changes1 = pct_changes(prices1)
        changes2 = pct_changes(prices2)
        
        # Calculate correlation
        n = min(len(changes1), len(changes2))
        if n < 2:
            return self._generate_price_chart(data)
        
        mean1 = sum(changes1[:n]) / n
        mean2 = sum(changes2[:n]) / n
        
        numerator = sum((changes1[i] - mean1) * (changes2[i] - mean2) for i in range(n))
        denom1 = sum((changes1[i] - mean1) ** 2 for i in range(n)) ** 0.5
        denom2 = sum((changes2[i] - mean2) ** 2 for i in range(n)) ** 0.5
        
        correlation = numerator / (denom1 * denom2) if denom1 * denom2 > 0 else 0
        
        width, height = 1200, 700
        margin = 120
        chart_size = min(width, height) - 2 * margin
        
        # Scale changes for plotting
        all_changes = changes1[:n] + changes2[:n]
        max_change = max(abs(c) for c in all_changes) * 1.1
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>',
            # Grid
            f'<line x1="{margin}" y1="{margin + chart_size/2}" x2="{margin + chart_size}" y2="{margin + chart_size/2}" stroke="#444" stroke-width="2"/>',
            f'<line x1="{margin + chart_size/2}" y1="{margin}" x2="{margin + chart_size/2}" y2="{margin + chart_size}" stroke="#444" stroke-width="2"/>',
        ]
        
        # Plot points
        for i in range(n):
            x = margin + chart_size/2 + (changes1[i] / max_change * chart_size/2)
            y = margin + chart_size/2 - (changes2[i] / max_change * chart_size/2)
            svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#60a5fa" opacity="0.8"/>')
        
        # Correlation line (trend line)
        if abs(correlation) > 0.1:
            slope = correlation * (max_change / max_change)  # Simplified
            x1, x2 = margin, margin + chart_size
            y1 = margin + chart_size/2 + (slope * chart_size/2)
            y2 = margin + chart_size/2 - (slope * chart_size/2)
            svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#f97316" stroke-width="4" stroke-dasharray="10,10"/>')
        
        # Labels
        svg_parts.append(f'<text x="{width/2}" y="{height - 30}" fill="white" font-size="20" text-anchor="middle">{symbols[0]} Price Change (%)</text>')
        svg_parts.append(f'<text x="40" y="{height/2}" fill="white" font-size="20" transform="rotate(-90, 40, {height/2})">{symbols[1]} Price Change (%)</text>')
        
        # Correlation value
        corr_color = "#22c55e" if correlation > 0.5 else "#f97316" if correlation > 0 else "#ef4444"
        svg_parts.append(f'<text x="{width - margin}" y="{margin + 40}" fill="{corr_color}" font-size="24" text-anchor="end">r = {correlation:.3f}</text>')
        
        corr_label = "Strong Positive" if correlation > 0.7 else "Moderate Positive" if correlation > 0.3 else "Weak" if correlation > -0.3 else "Negative"
        svg_parts.append(f'<text x="{width - margin}" y="{margin + 70}" fill="white" font-size="18" text-anchor="end">{corr_label} Correlation</text>')
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)

    async def _handle_analysis_query(self, query: str) -> str:
        """Handle analysis queries using LLM reasoning over actual data.
        
        This method:
        1. Parses what time period the user wants (default 2 days for "cycles")
        2. Fetches historical candle data for that period
        3. Uses the LLM to actually analyze patterns, cycles, trends
        
        Args:
            query: User's analysis question
            
        Returns:
            LLM-generated analysis based on real data
        """
        from datetime import datetime, timedelta
        from langchain_core.messages import SystemMessage, HumanMessage
        
        query_lower = query.lower()
        
        # Determine crypto
        crypto = self._extract_crypto(query) or "BTC"
        product_id = self._normalize_product_id(crypto)
        
        # Parse time period from query
        if "hour" in query_lower:
            hours = 1
            for word in query_lower.split():
                if word.isdigit():
                    hours = int(word)
                    break
            days = hours / 24
            period_name = f"{hours} hour(s)"
            granularity = "FIVE_MINUTE"
        elif "day" in query_lower:
            days = 1
            for word in query_lower.split():
                if word.isdigit():
                    days = int(word)
                    break
            period_name = f"{days} day(s)"
            granularity = "FIFTEEN_MINUTE" if days <= 2 else "ONE_HOUR"
        elif "week" in query_lower:
            days = 7
            period_name = "1 week"
            granularity = "ONE_HOUR"
        elif "month" in query_lower:
            days = 30
            period_name = "1 month"
            granularity = "SIX_HOUR"
        else:
            # Default to 2 days for analysis queries
            days = 2
            period_name = "2 days"
            granularity = "FIFTEEN_MINUTE"
        
        # Fetch historical data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        start_ts = str(int(start_time.timestamp()))
        end_ts = str(int(end_time.timestamp()))
        
        try:
            candles_result = await self.mcp_server.call_tool(
                "get_candles",
                {
                    "product_id": product_id,
                    "start": start_ts,
                    "end": end_ts,
                    "granularity": granularity
                }
            )
            
            # Parse candles into a data structure
            candles = []
            raw_candles = candles_result.get('candles', []) if isinstance(candles_result, dict) else []
            
            for candle in raw_candles:
                if isinstance(candle, dict):
                    candles.append({
                        "time": datetime.fromtimestamp(int(candle.get('start', 0))).strftime('%Y-%m-%d %H:%M'),
                        "open": float(candle.get('open', 0)),
                        "high": float(candle.get('high', 0)),
                        "low": float(candle.get('low', 0)),
                        "close": float(candle.get('close', 0)),
                        "volume": float(candle.get('volume', 0))
                    })
                elif hasattr(candle, 'close'):
                    candles.append({
                        "time": datetime.fromtimestamp(int(candle.start)).strftime('%Y-%m-%d %H:%M'),
                        "open": float(candle.open),
                        "high": float(candle.high),
                        "low": float(candle.low),
                        "close": float(candle.close),
                        "volume": float(candle.volume)
                    })
            
            if not candles:
                return f"❌ Unable to fetch historical data for {crypto}."
            
            # Sort by time
            candles.sort(key=lambda x: x['time'])
            
            # Calculate some basic stats to include
            closes = [c['close'] for c in candles]
            highs = [c['high'] for c in candles]
            lows = [c['low'] for c in candles]
            
            price_range = max(highs) - min(lows)
            current_price = closes[-1] if closes else 0
            start_price = closes[0] if closes else 0
            change_pct = ((current_price - start_price) / start_price * 100) if start_price else 0
            
            # Format candle data for LLM (sample if too many)
            if len(candles) > 50:
                # Sample every Nth candle to keep it manageable
                step = len(candles) // 50
                sampled = candles[::step]
            else:
                sampled = candles
            
            candle_text = "Time | Open | High | Low | Close | Volume\n"
            candle_text += "-" * 60 + "\n"
            for c in sampled:
                candle_text += f"{c['time']} | ${c['open']:.2f} | ${c['high']:.2f} | ${c['low']:.2f} | ${c['close']:.2f} | {c['volume']:.2f}\n"
            
            # Build prompt for LLM analysis
            system_prompt = f"""You are a cryptocurrency market analyst. Analyze the provided {crypto} price data and answer the user's specific question.

Focus on EXACTLY what the user asked about. If they ask about:
- Cycles: Look for repeating patterns, oscillations between highs and lows
- Trends: Identify direction (upward, downward, sideways)
- Patterns: Look for chart patterns, support/resistance levels
- Volatility: Analyze price swings and range

Be specific with your analysis:
- Reference actual prices and times from the data
- Quantify patterns (e.g., "price oscillated between $X and $Y every ~N hours")
- Give actionable insights based on the patterns you find

Data period: {period_name}
Number of data points: {len(candles)}
Price range: ${min(lows):.2f} - ${max(highs):.2f}
Change over period: {change_pct:+.2f}%"""

            user_prompt = f"""User question: {query}

Here is the actual {crypto} price data for the last {period_name}:

{candle_text}

Analyze this data and answer the user's question. Be specific about patterns, cycles, or trends you observe."""

            # Use LLM to analyze
            if self.llm:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = await self.llm.ainvoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                
                return f"📊 **{crypto} Analysis** ({period_name})\n\n{content}"
            else:
                return f"❌ LLM not available for analysis."
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return f"❌ Error analyzing {crypto}: {str(e)}"

    async def _handle_blockchain_query(self, query: str) -> str:
        """Handle blockchain data queries (transactions, blocks, stats).
        
        Args:
            query: User query about blockchain data
            
        Returns:
            Formatted response with blockchain data
        """
        import re
        query_lower = query.lower()
        
        # Detect which chains are mentioned (support multiple)
        chains = []
        chain_keywords = {
            "bitcoin": "bitcoin",
            "btc": "bitcoin",
            "ethereum": "ethereum",
            "eth": "ethereum",
            "ripple": "ripple",
            "xrp": "ripple",
            "solana": "solana",
            "sol": "solana",
            "zcash": "zcash",
            "zec": "zcash",
        }
        
        for keyword, chain_name in chain_keywords.items():
            # Use word boundary to avoid false matches (e.g., "sol" in "solana")
            if re.search(rf'\b{keyword}\b', query_lower) and chain_name not in chains:
                chains.append(chain_name)
        
        if not chains:
            # Default to bitcoin if no chain specified
            chains = ["bitcoin"]
        
        # Detect if this is a comparison/correlation query
        is_comparison = any(word in query_lower for word in [
            "correlation", "correlate", "compare", "comparison", "vs", "versus",
            " and ", "both", "together", "predictor", "predict"
        ]) and len(chains) >= 2
        
        # Detect query type
        is_transaction_query = any(word in query_lower for word in ["transaction", "transactions", "tx", "txs"])
        is_block_query = any(word in query_lower for word in ["block ", "blocks", "latest block"]) and not is_comparison
        is_stats_query = any(word in query_lower for word in ["stats", "statistics", "info", "network"])
        is_analysis_query = any(word in query_lower for word in ["analyze", "analysis", "24 hours", "last day"])
        
        # Extract limit if specified
        limit_match = re.search(r'(\d+)\s*(?:recent|last|latest)?', query_lower)
        limit = int(limit_match.group(1)) if limit_match else 10
        limit = min(limit, 25)  # Cap at 25 for readability
        
        try:
            # Handle multi-chain comparison/analysis
            if is_comparison or (is_analysis_query and len(chains) >= 2):
                return await self._handle_blockchain_comparison(chains, query_lower)
            
            # Single chain query
            chain = chains[0]
            
            if is_transaction_query:
                result = await self.blockchain_data_service.get_recent_transactions(chain, limit=limit)
                if result["status"] == "success":
                    return self._format_blockchain_transactions(result)
                else:
                    return f"❌ {result.get('message', 'Failed to fetch transactions')}"
            elif is_block_query:
                result = await self.blockchain_data_service.get_latest_block(chain)
                if result["status"] == "success":
                    return self._format_blockchain_block(result)
                else:
                    return f"❌ {result.get('message', 'Failed to fetch block info')}"
            else:
                # Default to stats
                result = await self.blockchain_data_service.get_blockchain_stats(chain)
                if result["status"] == "success":
                    return self._format_blockchain_stats(result)
                else:
                    return f"❌ {result.get('message', 'Failed to fetch stats')}"
        except Exception as e:
            return f"❌ Error fetching blockchain data: {str(e)}"
    
    async def _handle_blockchain_comparison(self, chains: list[str], query_lower: str) -> str:
        """Handle comparison/analysis queries across multiple blockchains.
        
        Args:
            chains: List of blockchain names to compare
            query_lower: Lowercase query for context
            
        Returns:
            Formatted comparison analysis
        """
        lines = ["# 🔗 Blockchain Comparison Analysis\n"]
        
        # Fetch stats and transactions for each chain
        chain_data = {}
        for chain in chains:
            stats = await self.blockchain_data_service.get_blockchain_stats(chain)
            txs = await self.blockchain_data_service.get_recent_transactions(chain, limit=10)
            block = await self.blockchain_data_service.get_latest_block(chain)
            chain_data[chain] = {
                "stats": stats,
                "transactions": txs,
                "block": block
            }
        
        # Display stats for each chain
        for chain in chains:
            data = chain_data[chain]
            stats = data["stats"]
            block = data["block"]
            
            if stats.get("status") == "success":
                lines.append(f"## {chain.upper()} Network\n")
                
                if chain == "bitcoin":
                    s = stats.get("stats", {})
                    lines.append(f"• **Block Height:** {s.get('block_height', 'N/A'):,}")
                    lines.append(f"• **Difficulty:** {s.get('difficulty', 0):,.0f}")
                    if s.get('hashrate_eh_s'):
                        lines.append(f"• **Hashrate:** {s['hashrate_eh_s']:.2f} EH/s")
                    lines.append(f"• **Mempool Size:** {s.get('mempool_size', 0):,} pending txs")
                    lines.append(f"• **Mempool Fees:** {s.get('mempool_total_fee_btc', 0):.4f} BTC")
                    fees = s.get('recommended_fee_sat_vb', {})
                    lines.append(f"• **Avg Fee Rate:** {fees.get('half_hour', 0)} sat/vB")
                    
                elif chain == "solana":
                    s = stats.get("stats", {})
                    lines.append(f"• **Block Height:** {s.get('block_height', 0):,}")
                    lines.append(f"• **Current Epoch:** {s.get('epoch', 'N/A')}")
                    lines.append(f"• **Slot Height:** {s.get('slot_height', 0):,}")
                    if s.get('total_supply_sol'):
                        lines.append(f"• **Total Supply:** {s['total_supply_sol']:,.0f} SOL")
                    if s.get('circulating_supply_sol'):
                        lines.append(f"• **Circulating:** {s['circulating_supply_sol']:,.0f} SOL")
                
                lines.append("")
            else:
                lines.append(f"## {chain.upper()}\n")
                lines.append(f"⚠️ {stats.get('message', 'Data unavailable')}\n")
        
        # Add comparison analysis
        lines.append("## 📊 Comparison Analysis\n")
        
        btc_data = chain_data.get("bitcoin", {})
        sol_data = chain_data.get("solana", {})
        
        if btc_data.get("stats", {}).get("status") == "success" and sol_data.get("stats", {}).get("status") == "success":
            btc_stats = btc_data["stats"].get("stats", {})
            sol_stats = sol_data["stats"].get("stats", {})
            
            # Block production comparison
            lines.append("**Block Production:**")
            lines.append(f"• Bitcoin: ~1 block every 10 minutes (PoW)")
            lines.append(f"• Solana: ~2 blocks per second (PoS)")
            lines.append("")
            
            # Mempool/pending activity
            btc_mempool = btc_stats.get('mempool_size', 0)
            lines.append("**Network Activity:**")
            lines.append(f"• Bitcoin mempool: {btc_mempool:,} pending transactions")
            lines.append(f"• Solana processes transactions in real-time (no mempool backlog)")
            lines.append("")
            
            # Fee analysis
            btc_fees = btc_stats.get('recommended_fee_sat_vb', {})
            lines.append("**Transaction Fees:**")
            lines.append(f"• Bitcoin: {btc_fees.get('half_hour', 0)} sat/vB (~${self._estimate_btc_fee_usd(btc_fees.get('half_hour', 0))} for typical tx)")
            lines.append(f"• Solana: ~0.000005 SOL (~$0.001 per transaction)")
            lines.append("")
            
            # Correlation note
            lines.append("**Correlation Analysis:**")
            lines.append("• Bitcoin and Solana operate on fundamentally different consensus mechanisms")
            lines.append("• Bitcoin (PoW) block times are variable; Solana (PoS) has consistent ~400ms slots")
            lines.append("• Network congestion on one chain does not directly affect the other")
            lines.append("• Price correlation exists but blockchain activity correlation is weak")
            lines.append("• Bitcoin mempool congestion is NOT a predictor of Solana network activity")
            lines.append("")
            lines.append("💡 *For price correlation analysis, ask about price trends instead of blockchain data.*")
        else:
            lines.append("Insufficient data to perform full comparison analysis.")
        
        return "\n".join(lines)
    
    def _estimate_btc_fee_usd(self, sat_per_vb: int) -> str:
        """Estimate USD cost for a typical Bitcoin transaction.
        
        Args:
            sat_per_vb: Fee rate in satoshis per virtual byte
            
        Returns:
            Estimated USD cost as string
        """
        # Typical transaction is ~140 vB, BTC ~$100k
        typical_vb = 140
        btc_price = 100000  # approximate
        fee_btc = (sat_per_vb * typical_vb) / 1e8
        fee_usd = fee_btc * btc_price
        return f"{fee_usd:.2f}"

    def _format_blockchain_transactions(self, result: dict[str, Any]) -> str:
        """Format blockchain transaction data for display with rich details."""
        chain = result.get("chain", "unknown").capitalize()
        transactions = result.get("transactions", [])
        count = result.get("count", len(transactions))
        slot = result.get("slot")
        block_height = result.get("block_height")
        block_time = result.get("block_time")
        
        if not transactions:
            return f"No recent transactions found for {chain}."
        
        # Format block time if available
        time_str = ""
        if block_time:
            from datetime import datetime
            dt = datetime.fromtimestamp(block_time)
            time_str = f" at {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        # Build header based on chain
        if chain.lower() == "solana":
            lines = [f"📋 **Recent {chain} Transactions** ({count} shown from slot {slot:,}{time_str})\n"]
        elif chain.lower() == "bitcoin":
            block_ref = f"block {block_height:,}" if block_height else "mempool"
            lines = [f"₿ **Recent {chain} Transactions** ({count} shown from {block_ref}{time_str})\n"]
        else:
            lines = [f"📋 **Recent {chain} Transactions** ({count} shown{time_str})\n"]
        
        for i, tx in enumerate(transactions[:10], 1):  # Limit display to 10
            if chain.lower() == "solana":
                sig = tx.get("signature", "")
                sig_short = sig[:16] + "..." if len(sig) > 16 else sig
                tx_type = tx.get("tx_type", "Transaction")
                status = "✅" if tx.get("status") == "success" else "❌"
                fee_sol = tx.get("fee_sol", 0)
                
                lines.append(f"### {i}. {tx_type} {status}")
                lines.append(f"   **Signature:** `{sig_short}`")
                
                # Show transfer details if available
                transfer_amount = tx.get("transfer_amount_sol")
                total_value = tx.get("total_value_sol", 0)
                sender = tx.get("sender")
                receiver = tx.get("receiver")
                
                if transfer_amount and transfer_amount > 0.0001:
                    lines.append(f"   **Amount:** {transfer_amount:.6f} SOL")
                elif total_value > 0.0001:
                    lines.append(f"   **Value Moved:** {total_value:.6f} SOL")
                
                if sender:
                    sender_short = sender[:8] + "..." + sender[-4:] if len(sender) > 12 else sender
                    lines.append(f"   **From:** `{sender_short}`")
                
                if receiver:
                    receiver_short = receiver[:8] + "..." + receiver[-4:] if len(receiver) > 12 else receiver
                    lines.append(f"   **To:** `{receiver_short}`")
                
                # Show token transfers if any
                token_transfers = tx.get("token_transfers", [])
                if token_transfers:
                    for tt in token_transfers[:2]:
                        direction = "📥" if tt.get("direction") == "received" else "📤"
                        mint = tt.get("mint", "unknown")
                        mint_short = mint[:8] + "..." if len(mint) > 8 else mint
                        change = abs(tt.get("change", 0))
                        lines.append(f"   {direction} **Token:** {change:,.2f} ({mint_short})")
                
                # Show balance changes summary
                balance_changes = tx.get("balance_changes", [])
                if balance_changes and not transfer_amount:
                    significant_changes = [c for c in balance_changes if abs(c.get("change_sol", 0)) > 0.0001]
                    if significant_changes:
                        lines.append(f"   **Balance Changes:** {len(significant_changes)} accounts affected")
                
                lines.append(f"   **Fee:** {fee_sol:.9f} SOL | **Accounts:** {tx.get('num_accounts', 0)} | **Instructions:** {tx.get('num_instructions', 0)}")
                
                if tx.get("has_error"):
                    lines.append(f"   ⚠️ **Error:** {tx.get('error', 'Unknown error')}")
                
                lines.append("")  # Blank line between transactions
            
            elif chain.lower() == "bitcoin":
                txid = tx.get("txid", "")
                txid_short = txid[:16] + "..." if len(txid) > 16 else txid
                tx_type = tx.get("tx_type", "📝 Transaction")
                status_str = tx.get("status", "unknown")
                confirmed = status_str == "confirmed"
                status = "✅" if confirmed else "⏳"
                
                lines.append(f"### {i}. {tx_type} {status}")
                lines.append(f"   **TxID:** `{txid_short}`")
                
                # Show value (use transfer_amount_btc or fall back to output_total_btc)
                value_btc = tx.get("transfer_amount_btc") or tx.get("output_total_btc", 0)
                if value_btc and value_btc > 0:
                    lines.append(f"   **Value:** {value_btc:.8f} BTC")
                
                # Show sender and receiver
                sender = tx.get("sender")
                receiver = tx.get("receiver")
                
                if sender:
                    sender_short = sender[:12] + "..." + sender[-4:] if len(sender) > 16 else sender
                    lines.append(f"   **From:** `{sender_short}`")
                
                if receiver:
                    receiver_short = receiver[:12] + "..." + receiver[-4:] if len(receiver) > 16 else receiver
                    lines.append(f"   **To:** `{receiver_short}`")
                
                # Show fee
                fee_btc = tx.get("fee_btc", 0)
                fee_rate = tx.get("fee_rate_sat_vb")
                fee_str = f"**Fee:** {fee_btc:.8f} BTC"
                if fee_rate:
                    fee_str += f" ({fee_rate:.1f} sat/vB)"
                lines.append(f"   {fee_str}")
                
                # Show input/output counts and size
                input_count = tx.get("num_inputs", 0)
                output_count = tx.get("num_outputs", 0)
                size = tx.get("size_bytes")
                weight = tx.get("weight")
                vsize = weight // 4 if weight else size
                
                size_str = f"{vsize:,} vB" if vsize else f"{size:,} B" if size else ""
                lines.append(f"   **Inputs:** {input_count} | **Outputs:** {output_count} | **Size:** {size_str}")
                
                # Show confirmations if confirmed
                confirmations = tx.get("confirmations", 0)
                if confirmed and confirmations:
                    lines.append(f"   **Confirmations:** {confirmations:,}")
                
                lines.append("")  # Blank line between transactions
            
            else:
                tx_hash = tx.get("hash", "")
                if len(tx_hash) > 20:
                    tx_hash = tx_hash[:20] + "..."
                block = tx.get("block", tx.get("ledger_index", ""))
                time = tx.get("time", "")
                
                # Add value info if available
                value_str = ""
                if "input_total_btc" in tx:
                    value_str = f" ({tx['input_total_btc']:.6f} BTC)"
                elif "value_eth" in tx:
                    value_str = f" ({tx['value_eth']:.6f} ETH)"
                elif "input_total_zec" in tx:
                    value_str = f" ({tx['input_total_zec']:.6f} ZEC)"
                
                lines.append(f"{i}. Block {block}: `{tx_hash}`{value_str}")
        
        # Add explorer link note based on chain
        if chain.lower() == "solana":
            lines.append("\n💡 *View full details on Solana Explorer: https://explorer.solana.com*")
        elif chain.lower() == "bitcoin":
            lines.append("\n💡 *View full details on Mempool.space: https://mempool.space*")
        
        return "\n".join(lines)

    def _format_blockchain_block(self, result: dict[str, Any]) -> str:
        """Format blockchain block info for display."""
        chain = result.get("chain", "unknown").capitalize()
        block = result.get("block", {})
        
        if chain.lower() == "solana":
            lines = [f"🔲 **Latest {chain} Slot**\n"]
            lines.append(f"• Slot: {block.get('slot', 'N/A'):,}")
            lines.append(f"• Epoch: {block.get('epoch', 'N/A')}")
            lines.append(f"• Slot Index: {block.get('slot_index', 'N/A'):,} / {block.get('slots_in_epoch', 'N/A'):,}")
        elif chain.lower() == "bitcoin":
            lines = [f"₿ **Latest {chain} Block**\n"]
            lines.append(f"• Height: {block.get('height', 'N/A'):,}")
            if block.get('hash'):
                hash_display = block['hash'][:20] + "..." if len(block.get('hash', '')) > 20 else block.get('hash')
                lines.append(f"• Hash: `{hash_display}`")
            if block.get('timestamp'):
                from datetime import datetime
                dt = datetime.fromtimestamp(block['timestamp'])
                lines.append(f"• Time: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            if block.get('tx_count'):
                lines.append(f"• Transactions: {block['tx_count']:,}")
            if block.get('size_bytes'):
                size_mb = block['size_bytes'] / 1_000_000
                lines.append(f"• Size: {size_mb:.2f} MB")
            if block.get('difficulty'):
                lines.append(f"• Difficulty: {block['difficulty']:,.0f}")
            if block.get('nonce'):
                lines.append(f"• Nonce: {block['nonce']:,}")
        else:
            lines = [f"🔲 **Latest {chain} Block**\n"]
            lines.append(f"• Height: {block.get('height', 'N/A'):,}")
            if block.get('hash'):
                hash_display = block['hash'][:20] + "..." if len(block.get('hash', '')) > 20 else block.get('hash')
                lines.append(f"• Hash: `{hash_display}`")
            if block.get('time'):
                lines.append(f"• Time: {block['time']}")
            if block.get('total_transactions'):
                lines.append(f"• Total Transactions: {block['total_transactions']:,}")
        
        return "\n".join(lines)

    def _format_blockchain_stats(self, result: dict[str, Any]) -> str:
        """Format blockchain statistics for display."""
        chain = result.get("chain", "unknown").capitalize()
        stats = result.get("stats", {})
        
        lines = [f"📊 **{chain} Network Statistics**\n"]
        
        if chain.lower() == "solana":
            lines.append(f"• Current Epoch: {stats.get('epoch', 'N/A')}")
            lines.append(f"• Slot Height: {stats.get('slot_height', 0):,}")
            lines.append(f"• Block Height: {stats.get('block_height', 0):,}")
            if stats.get('total_supply_sol'):
                lines.append(f"• Total Supply: {stats['total_supply_sol']:,.2f} SOL")
            if stats.get('circulating_supply_sol'):
                lines.append(f"• Circulating Supply: {stats['circulating_supply_sol']:,.2f} SOL")
        elif chain.lower() == "bitcoin":
            lines.append(f"• Block Height: {stats.get('block_height', 0):,}")
            if stats.get('difficulty'):
                lines.append(f"• Difficulty: {stats['difficulty']:,.0f}")
            if stats.get('hashrate_eh_s'):
                lines.append(f"• Hashrate: {stats['hashrate_eh_s']:.2f} EH/s")
            lines.append("")
            lines.append("**Mempool:**")
            lines.append(f"• Pending Transactions: {stats.get('mempool_size', 0):,}")
            lines.append(f"• Mempool Size: {stats.get('mempool_vsize_mb', 0):.2f} MB")
            lines.append(f"• Total Fees: {stats.get('mempool_total_fee_btc', 0):.4f} BTC")
            lines.append("")
            lines.append("**Recommended Fees (sat/vB):**")
            fees = stats.get('recommended_fee_sat_vb', {})
            lines.append(f"• Fastest (~10 min): {fees.get('fastest', 0)} sat/vB")
            lines.append(f"• Half Hour: {fees.get('half_hour', 0)} sat/vB")
            lines.append(f"• Hour: {fees.get('hour', 0)} sat/vB")
            lines.append(f"• Economy: {fees.get('economy', 0)} sat/vB")
        else:
            if stats.get('blocks'):
                lines.append(f"• Blocks: {stats['blocks']:,}")
            if stats.get('transactions'):
                lines.append(f"• Total Transactions: {stats['transactions']:,}")
            if stats.get('difficulty'):
                lines.append(f"• Difficulty: {stats['difficulty']:,.0f}")
            if stats.get('hashrate'):
                lines.append(f"• Hashrate (24h): {stats['hashrate']:,.0f}")
            if stats.get('mempool_transactions'):
                lines.append(f"• Mempool: {stats['mempool_transactions']:,} txs")
            if stats.get('market_price_usd'):
                lines.append(f"• Price: ${stats['market_price_usd']:,.2f} USD")
            if stats.get('market_cap_usd'):
                lines.append(f"• Market Cap: ${stats['market_cap_usd']:,.0f} USD")
        
        return "\n".join(lines)
    
    def _format_portfolio(self, data: dict[str, Any]) -> str:
        """Format portfolio data for display."""
        if not data or not data.get("accounts"):
            return "No portfolio data available."
        
        lines = []
        
        # Filter to accounts with non-zero balances
        for account in data.get("accounts", []):
            name = account.get("name", "Account")
            available = account.get("available_balance", {})
            amount = float(available.get("value", 0) or 0)
            currency = available.get("currency", "USD")
            
            # Only show accounts with balance > 0
            if amount > 0:
                # Format nicely based on size
                if amount >= 1:
                    formatted = f"{amount:,.2f}"
                else:
                    formatted = f"{amount:.6f}".rstrip('0').rstrip('.')
                lines.append(f"• {name}: {formatted} {currency}")
        
        if not lines:
            return "No accounts with balances found."
        
        return "\n".join(lines)
    
    def _format_price(self, crypto: str, data: dict[str, Any]) -> str:
        """Format price data for display."""
        if not data:
            return f"Could not get price for {crypto}."
        
        # Handle nested structure from get_market_data
        if "product" in data:
            product = data.get("product", {})
            price = product.get("price", "Unknown")
        else:
            price = data.get("amount", data.get("price", "Unknown"))
        
        # Try to get bid/ask for more info
        bid_ask = data.get("bid_ask", {})
        if bid_ask and "pricebooks" in bid_ask:
            pricebooks = bid_ask.get("pricebooks", [])
            if pricebooks:
                bids = pricebooks[0].get("bids", [])
                asks = pricebooks[0].get("asks", [])
                bid = bids[0].get("price") if bids else None
                ask = asks[0].get("price") if asks else None
                if bid and ask:
                    return f"💰 {crypto}: ${price} USD (Bid: ${bid} / Ask: ${ask})"
        
        return f"💰 {crypto} is currently ${price} USD"
    
    async def get_portfolio_summary(self) -> dict[str, Any]:
        """Get a summary of the portfolio.
        
        Returns:
            Portfolio summary including balances and positions
        """
        # Fetch all accounts using pagination to get complete portfolio
        all_accounts = []
        cursor = None
        
        for _ in range(5):  # Max 5 pages to prevent infinite loops
            params = {"limit": 50}
            if cursor:
                params["cursor"] = cursor
            
            result = await self.execute_tool("get_accounts", params)
            accounts = result.get("accounts", [])
            all_accounts.extend(accounts)
            
            if not result.get("has_next"):
                break
            cursor = result.get("cursor")
        
        summary = {
            "accounts": all_accounts,
            "total_positions": len(all_accounts)
        }
        
        return summary
    
    async def get_market_data(self, product_id: str) -> dict[str, Any]:
        """Get market data for a product.
        
        Args:
            product_id: Product ID (e.g., "BTC-USD")
        
        Returns:
            Market data including price and bid/ask
        """
        product = await self.execute_tool("get_product", {"product_id": product_id})
        bid_ask = await self.execute_tool("get_best_bid_ask", {"product_ids": [product_id]})
        
        return {
            "product": product,
            "bid_ask": bid_ask
        }
    
    async def place_market_buy(
        self,
        product_id: str,
        quote_size: str
    ) -> dict[str, Any]:
        """Place a market buy order.
        
        Args:
            product_id: Product ID (e.g., "BTC-USD")
            quote_size: Amount in quote currency
        
        Returns:
            Order result
        """
        return await self.execute_tool("market_order_buy", {
            "product_id": product_id,
            "quote_size": quote_size
        })
    
    async def place_market_sell(
        self,
        product_id: str,
        base_size: str
    ) -> dict[str, Any]:
        """Place a market sell order.
        
        Args:
            product_id: Product ID (e.g., "BTC-USD")
            base_size: Amount in base currency
        
        Returns:
            Order result
        """
        return await self.execute_tool("market_order_sell", {
            "product_id": product_id,
            "base_size": base_size
        })
    
    async def get_open_orders(
        self,
        product_id: str | None = None
    ) -> dict[str, Any]:
        """Get open orders.
        
        Args:
            product_id: Optional filter by product ID
        
        Returns:
            List of open orders
        """
        params = {"order_status": ["OPEN", "PENDING"]}
        if product_id:
            params["product_id"] = product_id
        
        return await self.execute_tool("list_orders", params)
