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

from src.mcp_servers.coinbase import CoinbaseMCPServer, CoinbaseAPIError
from src.agents.chain_of_thought import ChainOfThought, ReasoningType
from src.services.blockchain_data import BlockchainDataService
from src.services.coinbase_websocket import CoinbaseWebSocketService

# Configure logger for this module
logger = logging.getLogger(__name__)


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
        """Process a natural language query through the agent workflow.
        
        This is the high-level interface used by the Supervisor Agent for delegation.
        It accepts a query string and optional config, builds the appropriate state,
        and returns a structured response.
        
        Args:
            query: Natural language query from the user
            config: LangGraph config with thread_id for checkpointing (optional)
            conversation_history: Previous messages for context (optional)
        
        Returns:
            Dict with 'response' and 'status' keys
        """
        try:
            # Determine what tool to call based on the query
            query_lower = query.lower()
            
            # Route to appropriate tool based on query content
            # Check for cancel requests FIRST
            if "cancel" in query_lower:
                response = await self._handle_cancel_request(query)
            # Check for analysis/pattern queries - use real-time data
            elif any(word in query_lower for word in ["pattern", "patterns", "analyze", "analysis", "trend", "trends", "seeing"]):
                response = await self._handle_analysis_query(query)
            # Check for blockchain data queries (transactions, blocks, on-chain data)
            elif any(word in query_lower for word in ["blockchain", "transaction", "transactions", "block ", "blocks", "on-chain", "onchain", "ledger"]):
                response = await self._handle_blockchain_query(query)
            # Check for order/trade requests (before price checks)
            elif any(word in query_lower for word in ["limit order", "place.*order", "order"]):
                response = await self._handle_order_request(query)
            elif any(word in query_lower for word in ["buy", "purchase"]) and any(word in query_lower for word in ["$", "usd", "dollar"]):
                response = await self._handle_buy_request(query)
            elif any(word in query_lower for word in ["sell"]) and any(word in query_lower for word in ["$", "usd", "dollar"]):
                response = await self._handle_sell_request(query)
            elif any(word in query_lower for word in ["balance", "account", "cash", "value", "worth", "portfolio"]):
                result = await self.get_portfolio_summary()
                response = f"Here's your Coinbase portfolio:\n{self._format_portfolio(result)}"
            elif any(word in query_lower for word in ["price", "quote", "how much", "cost"]):
                # Extract crypto from query
                crypto = self._extract_crypto(query)
                if crypto:
                    result = await self.get_market_data(f"{crypto}-USD")
                    response = f"{self._format_price(crypto, result)}"
                else:
                    # Default to Bitcoin
                    result = await self.get_market_data("BTC-USD")
                    response = f"{self._format_price('BTC', result)}"
            elif any(word in query_lower for word in ["buy", "purchase"]):
                crypto = self._extract_crypto(query) or "BTC"
                response = f"To buy {crypto}, please specify an amount (e.g., 'buy $100 of BTC')."
            elif any(word in query_lower for word in ["sell"]):
                crypto = self._extract_crypto(query) or "BTC"
                response = f"To sell {crypto}, please specify an amount."
            else:
                # Default to portfolio summary
                result = await self.get_portfolio_summary()
                response = f"Here's your Coinbase account:\n{self._format_portfolio(result)}"
            
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
            return {
                "response": f"An error occurred: {str(e)}",
                "status": "error"
            }
    
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
        
        product_id = f"{crypto}-USD"
        
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
            result = await self.place_market_buy(f"{crypto}-USD", amount)
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
            result = await self.place_market_sell(f"{crypto}-USD", amount)
            if result.get("success") or result.get("order_id"):
                order_id = self._extract_order_id(result)
                return f"✅ Market sell order placed for ${amount} of {crypto}\nOrder ID: {order_id}"
            else:
                return f"❌ Failed to place sell order: {result}"
        except Exception as e:
            return f"❌ Error placing sell order: {str(e)}"

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

    async def _handle_analysis_query(self, query: str) -> str:
        """Handle analysis/pattern queries using real-time WebSocket data.
        
        Uses the WebSocket candle data for current prices and combines with
        API data for a comprehensive analysis.
        
        Args:
            query: User query about patterns or analysis
            
        Returns:
            Formatted analysis response with real data
        """
        query_lower = query.lower()
        
        # Determine which crypto to analyze
        crypto = self._extract_crypto(query) or "BTC"
        product_id = f"{crypto}-USD"
        
        # Get real-time data from WebSocket if available
        realtime_data = None
        if self.websocket_service:
            realtime_data = self.websocket_service.get_latest_candle(product_id)
        
        # Also get data from Coinbase API for comparison
        try:
            api_data = await self.get_market_data(product_id)
        except Exception as e:
            logger.warning(f"Failed to get API data: {e}")
            api_data = None
        
        # Build the analysis response
        lines = [f"📊 **{crypto} Real-Time Analysis**\n"]
        
        # Use WebSocket data if available (most current)
        if realtime_data:
            close = realtime_data.get("close", "N/A")
            high = realtime_data.get("high", "N/A")
            low = realtime_data.get("low", "N/A")
            open_price = realtime_data.get("open", "N/A")
            volume = realtime_data.get("volume", "N/A")
            
            lines.append("**Current Data** (Real-Time WebSocket):")
            lines.append(f"• Price: ${close}")
            lines.append(f"• High: ${high}")
            lines.append(f"• Low: ${low}")
            lines.append(f"• Open: ${open_price}")
            lines.append(f"• Volume: {volume}")
            
            # Calculate simple pattern indicators
            try:
                close_f = float(close)
                open_f = float(open_price)
                high_f = float(high)
                low_f = float(low)
                
                # Price change
                change = close_f - open_f
                change_pct = (change / open_f * 100) if open_f > 0 else 0
                
                lines.append(f"\n**Pattern Analysis**:")
                
                # Trend direction
                if change > 0:
                    lines.append(f"• Trend: 📈 Bullish (+${change:.2f}, {change_pct:+.2f}%)")
                elif change < 0:
                    lines.append(f"• Trend: 📉 Bearish (${change:.2f}, {change_pct:.2f}%)")
                else:
                    lines.append(f"• Trend: ➡️ Sideways (0.00%)")
                
                # Volatility (high-low range as % of price)
                range_size = high_f - low_f
                volatility = (range_size / close_f * 100) if close_f > 0 else 0
                if volatility > 3:
                    lines.append(f"• Volatility: 🔴 High ({volatility:.2f}% range)")
                elif volatility > 1:
                    lines.append(f"• Volatility: 🟡 Moderate ({volatility:.2f}% range)")
                else:
                    lines.append(f"• Volatility: 🟢 Low ({volatility:.2f}% range)")
                
                # Candle pattern
                body = abs(close_f - open_f)
                upper_wick = high_f - max(close_f, open_f)
                lower_wick = min(close_f, open_f) - low_f
                
                if body < range_size * 0.1:
                    lines.append("• Candle Pattern: Doji (indecision)")
                elif upper_wick > body * 2 and close_f < open_f:
                    lines.append("• Candle Pattern: Shooting Star (bearish)")
                elif lower_wick > body * 2 and close_f > open_f:
                    lines.append("• Candle Pattern: Hammer (bullish)")
                elif close_f > open_f:
                    lines.append("• Candle Pattern: Bullish candle")
                else:
                    lines.append("• Candle Pattern: Bearish candle")
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Error calculating patterns: {e}")
        
        elif api_data:
            # Fallback to API data
            product = api_data.get("product", {})
            price = product.get("price", "N/A")
            lines.append("**Current Data** (API):")
            lines.append(f"• Price: ${price}")
            
            bid_ask = api_data.get("bid_ask", {})
            if bid_ask and "pricebooks" in bid_ask:
                pricebooks = bid_ask.get("pricebooks", [])
                if pricebooks:
                    bids = pricebooks[0].get("bids", [])
                    asks = pricebooks[0].get("asks", [])
                    if bids and asks:
                        bid = bids[0].get("price")
                        ask = asks[0].get("price")
                        spread = float(ask) - float(bid) if bid and ask else 0
                        lines.append(f"• Bid: ${bid}")
                        lines.append(f"• Ask: ${ask}")
                        lines.append(f"• Spread: ${spread:.2f}")
        else:
            lines.append("⚠️ Unable to fetch real-time data. Please try again.")
        
        lines.append("\n*Data source: Coinbase Advanced Trade*")
        
        return "\n".join(lines)

    async def _handle_blockchain_query(self, query: str) -> str:
        """Handle blockchain data queries (transactions, blocks, stats).
        
        Args:
            query: User query about blockchain data
            
        Returns:
            Formatted response with blockchain data
        """
        import re
        query_lower = query.lower()
        
        # Detect which chain
        chain = None
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
            if keyword in query_lower:
                chain = chain_name
                break
        
        if not chain:
            # Default to bitcoin if no chain specified
            chain = "bitcoin"
        
        # Detect query type
        is_transaction_query = any(word in query_lower for word in ["transaction", "transactions", "tx", "txs"])
        is_block_query = any(word in query_lower for word in ["block ", "blocks", "latest block"])
        is_stats_query = any(word in query_lower for word in ["stats", "statistics", "info", "network"])
        
        # Extract limit if specified
        limit_match = re.search(r'(\d+)\s*(?:recent|last|latest)?', query_lower)
        limit = int(limit_match.group(1)) if limit_match else 10
        limit = min(limit, 25)  # Cap at 25 for readability
        
        try:
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

    def _format_blockchain_transactions(self, result: dict[str, Any]) -> str:
        """Format blockchain transaction data for display."""
        chain = result.get("chain", "unknown").capitalize()
        transactions = result.get("transactions", [])
        count = result.get("count", len(transactions))
        
        if not transactions:
            return f"No recent transactions found for {chain}."
        
        lines = [f"📋 **Recent {chain} Transactions** ({count} shown)\n"]
        
        for i, tx in enumerate(transactions[:10], 1):  # Limit display to 10
            if chain.lower() == "solana":
                sig = tx.get("signature", "")[:20] + "..."
                slot = tx.get("slot", "")
                lines.append(f"{i}. Slot {slot}: `{sig}`")
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
        
        lines.append(f"\n🔗 View more at blockchair.com/{chain.lower()}")
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
