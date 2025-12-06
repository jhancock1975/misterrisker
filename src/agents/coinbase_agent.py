"""Coinbase Agent - LangGraph agent for trading on Coinbase.

This module provides a LangGraph-based agent that uses the Coinbase MCP
Server tools to execute trading operations.
"""

import asyncio
import os
from typing import Any, Literal, TypedDict
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.mcp_servers.coinbase import CoinbaseMCPServer, CoinbaseAPIError
from src.agents.chain_of_thought import ChainOfThought, ReasoningType


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
        checkpointer: Any | None = None
    ):
        """Initialize the Coinbase Agent.
        
        Args:
            api_key: Coinbase API key (optional if mcp_server provided)
            api_secret: Coinbase API secret (optional if mcp_server provided)
            mcp_server: Pre-configured MCP server instance (optional)
            llm: Language model for reasoning (optional)
            enable_chain_of_thought: Whether to enable CoT reasoning
            checkpointer: Optional checkpointer for state persistence
        """
        if mcp_server is not None:
            self.mcp_server = mcp_server
        else:
            self.mcp_server = CoinbaseMCPServer(
                api_key=api_key or os.getenv("COINBASE_API_KEY", ""),
                api_secret=api_secret or os.getenv("COINBASE_API_SECRET", "")
            )
        
        self.llm = llm
        self.enable_chain_of_thought = enable_chain_of_thought
        self.chain_of_thought = ChainOfThought() if enable_chain_of_thought else None
        self.checkpointer = checkpointer
        
        self._workflow = self._build_workflow()
        self._available_tools: list[str] | None = None
    
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
            # Check for order/trade requests FIRST (before price checks)
            if any(word in query_lower for word in ["limit order", "place.*order", "order"]):
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
                order_type = "buy" if is_buy else "sell"
                result = await self.execute_tool("limit_order_gtc", {
                    "product_id": product_id,
                    "side": order_type.upper(),
                    "base_size": str(float(amount) / float(limit_price)),  # Convert USD to crypto amount
                    "limit_price": limit_price
                })
                if result.get("success") or result.get("order_id"):
                    order_id = result.get("order_id", "unknown")
                    return f"✅ Limit {order_type} order placed for ${amount} of {crypto} at ${limit_price}\\nOrder ID: {order_id}"
                else:
                    return f"❌ Failed to place limit order: {result}"
            else:
                # Place market order
                if is_buy:
                    result = await self.place_market_buy(product_id, amount)
                else:
                    result = await self.place_market_sell(product_id, amount)
                
                if result.get("success") or result.get("order_id"):
                    order_id = result.get("order_id", "unknown")
                    action = "bought" if is_buy else "sold"
                    return f"✅ Market order placed: {action} ${amount} of {crypto}\\nOrder ID: {order_id}"
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
                order_id = result.get("order_id", "unknown")
                return f"✅ Market buy order placed for ${amount} of {crypto}\\nOrder ID: {order_id}"
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
                order_id = result.get("order_id", "unknown")
                return f"✅ Market sell order placed for ${amount} of {crypto}\\nOrder ID: {order_id}"
            else:
                return f"❌ Failed to place sell order: {result}"
        except Exception as e:
            return f"❌ Error placing sell order: {str(e)}"
    
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
