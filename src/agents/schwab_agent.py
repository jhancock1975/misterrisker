"""Schwab Agent - LangGraph agent for trading on Schwab.

This module provides a LangGraph-based agent that uses the Schwab MCP
Server tools to execute trading operations.
"""

import asyncio
import os
from typing import Any, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.mcp_servers.schwab import SchwabMCPServer, SchwabAPIError
from src.agents.chain_of_thought import ChainOfThought, ReasoningType


class SchwabAgentError(Exception):
    """Exception raised by the Schwab Agent.
    
    Attributes:
        message: Error message
        tool_name: Name of the tool that caused the error (if applicable)
    """
    
    def __init__(self, message: str, tool_name: str | None = None):
        """Initialize SchwabAgentError.
        
        Args:
            message: Error message
            tool_name: Name of the tool that caused the error
        """
        self.message = message
        self.tool_name = tool_name
        super().__init__(self.message)


class SchwabAgentState(TypedDict, total=False):
    """State for the Schwab agent workflow.
    
    Attributes:
        messages: Conversation history
        tool_calls: List of tool calls made
        tool_results: Results from tool calls
        current_task: Current task being executed
        account_hash: Current account hash
        cash_balance: Current cash balance
        buying_power: Current buying power
        positions: Dictionary of positions
        quotes: Dictionary of quotes
        market_hours: Market hours info
        reasoning_steps: Chain of thought reasoning steps
    """
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    current_task: str
    account_hash: str
    cash_balance: float
    buying_power: float
    positions: dict[str, float]
    quotes: dict[str, float]
    market_hours: dict[str, Any]
    reasoning_steps: list[str]


class SchwabAgent:
    """LangGraph agent for Schwab trading operations.
    
    This agent uses the SchwabMCPServer to execute trading operations
    through a structured workflow.
    
    Attributes:
        mcp_server: The Schwab MCP Server instance
        default_account_hash: Default account hash to use
        llm: Language model for reasoning (optional)
        enable_chain_of_thought: Whether CoT reasoning is enabled
        chain_of_thought: ChainOfThought instance for structured reasoning
        _workflow: The compiled LangGraph workflow
        checkpointer: Optional checkpointer for state persistence
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        app_secret: str | None = None,
        callback_url: str | None = None,
        token_path: str | None = None,
        mcp_server: SchwabMCPServer | None = None,
        default_account_hash: str | None = None,
        llm: Any | None = None,
        enable_chain_of_thought: bool = False,
        checkpointer: Any | None = None
    ):
        """Initialize the Schwab Agent.
        
        Args:
            api_key: Schwab API key (optional if mcp_server provided)
            app_secret: Schwab app secret (optional if mcp_server provided)
            callback_url: OAuth callback URL (optional if mcp_server provided)
            token_path: Token file path (optional if mcp_server provided)
            mcp_server: Pre-configured MCP server instance (optional)
            default_account_hash: Default account hash to use
            llm: Language model for reasoning (optional)
            enable_chain_of_thought: Whether to enable CoT reasoning
            checkpointer: Optional checkpointer for state persistence
        """
        if mcp_server is not None:
            self.mcp_server = mcp_server
        else:
            self.mcp_server = SchwabMCPServer(
                api_key=api_key or os.getenv("SCHWAB_API_KEY", ""),
                app_secret=app_secret or os.getenv("SCHWAB_APP_SECRET", ""),
                callback_url=callback_url or os.getenv("SCHWAB_CALLBACK_URL", ""),
                token_path=token_path or os.getenv("SCHWAB_TOKEN_PATH", "")
            )
        
        self.default_account_hash = default_account_hash
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
        workflow = StateGraph(SchwabAgentState)
        
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
    
    def _analyze_task(self, state: SchwabAgentState) -> SchwabAgentState:
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
            account_hash = state.get("account_hash", self.default_account_hash)
            if account_hash:
                state["tool_calls"] = [
                    {"tool": "get_account", "params": {"account_hash": account_hash, "fields": ["positions"]}}
                ]
            else:
                state["tool_calls"] = [{"tool": "get_account_numbers", "params": {}}]
        elif current_task == "check_price":
            state["tool_calls"] = [{"tool": "get_quote", "params": {"symbol": "AAPL"}}]
        elif current_task == "buy_stock":
            account_hash = state.get("account_hash", self.default_account_hash)
            state["tool_calls"] = [
                {"tool": "equity_buy_limit", "params": {
                    "account_hash": account_hash,
                    "symbol": "AAPL",
                    "quantity": 10,
                    "price": "150.00"
                }}
            ]
        
        return state
    
    def _route_after_analyze(self, state: SchwabAgentState) -> str:
        """Route after task analysis.
        
        Args:
            state: Current workflow state
        
        Returns:
            Name of next node
        """
        if state.get("tool_calls"):
            return "execute_tools"
        return "end"
    
    def _execute_tools(self, state: SchwabAgentState) -> SchwabAgentState:
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
    
    def _route_after_execute(self, state: SchwabAgentState) -> str:
        """Route after tool execution.
        
        Args:
            state: Current workflow state
        
        Returns:
            Name of next node
        """
        if state.get("tool_results"):
            return "process_results"
        return "error"
    
    def _process_results(self, state: SchwabAgentState) -> SchwabAgentState:
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
    
    def _apply_chain_of_thought(self, state: SchwabAgentState) -> SchwabAgentState:
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
    
    def _handle_error(self, state: SchwabAgentState) -> SchwabAgentState:
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
    ) -> dict[str, Any] | list:
        """Execute a specific MCP tool.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
        
        Returns:
            Tool execution result
        
        Raises:
            SchwabAgentError: If tool execution fails
        """
        # Validate tool exists
        available_tools = [t["name"] for t in self.get_available_tools()]
        if tool_name not in available_tools:
            raise SchwabAgentError(f"Unknown tool: {tool_name}", tool_name=tool_name)
        
        try:
            result = await self.mcp_server.call_tool(tool_name, params)
            return result
        except Exception as e:
            raise SchwabAgentError(f"Tool execution failed: {str(e)}", tool_name=tool_name)
    
    async def run(self, initial_state: SchwabAgentState) -> SchwabAgentState:
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
                except SchwabAgentError:
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
            # Use LLM to understand user intent and route appropriately
            from langchain_core.messages import SystemMessage, HumanMessage
            import json
            
            routing_prompt = """You are a trading assistant router. Analyze the user's request and determine what action to take.

Return a JSON object with:
- intent: one of "place_order", "cancel_order", "check_orders", "get_quote", "account_balance", "portfolio", "market_movers", "market_hours", "transaction_history", "top_stocks"
- details: any relevant extracted details (symbol, quantity, price, order_type, order_id, etc.)

Intent definitions:
- "place_order": User wants to BUY or SELL securities (stocks, options)
- "cancel_order": User wants to CANCEL an existing order
- "check_orders": User wants to VIEW their existing orders
- "get_quote": User wants current price/quote for a symbol
- "account_balance": User wants to see account balance, cash, buying power
- "portfolio": User wants to see positions/holdings
- "market_movers": User wants to see what's moving in the market, top gainers/losers
- "market_hours": User wants to know if market is open
- "transaction_history": User wants to see recent transactions, trades, activity history
- "top_stocks": User wants to see top/popular stocks, stock rankings, or screen for stocks

Examples:
- "buy 10 shares of AAPL" → {"intent": "place_order", "details": {"action": "buy", "symbol": "AAPL", "quantity": 10}}
- "put in a limit order to buy NVDA at 160" → {"intent": "place_order", "details": {"action": "buy", "symbol": "NVDA", "price": 160, "order_type": "limit"}}
- "cancel the order" → {"intent": "cancel_order", "details": {}}
- "show me my orders" → {"intent": "check_orders", "details": {}}
- "what's the price of TSLA" → {"intent": "get_quote", "details": {"symbol": "TSLA"}}
- "show me my recent transactions" → {"intent": "transaction_history", "details": {}}
- "top 100 stocks" → {"intent": "top_stocks", "details": {"count": 100}}
- "what are the top movers today" → {"intent": "market_movers", "details": {}}

Return ONLY the JSON object."""

            messages = [
                SystemMessage(content=routing_prompt),
                HumanMessage(content=query)
            ]
            
            llm_response = await self.llm.ainvoke(messages)
            content = llm_response.content
            
            # Handle list content from some LLMs
            if isinstance(content, list):
                content = content[0] if content else "{}"
                if isinstance(content, dict):
                    content = content.get("text", str(content))
            content = str(content).strip()
            
            # Clean markdown if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            routing = json.loads(content)
            intent = routing.get("intent", "account_balance")
            details = routing.get("details", {})
            
            # Execute based on LLM-determined intent
            if intent == "place_order":
                response = await self._handle_order_request(query)
            elif intent == "cancel_order":
                response = await self._handle_cancel_order(query, details)
            elif intent == "check_orders":
                result = await self.get_open_orders()
                response = f"Here are your recent orders:\n{self._format_orders(result)}"
            elif intent == "get_quote":
                symbol = details.get("symbol", "").upper()
                if symbol:
                    result = await self.get_stock_price(symbol)
                    response = f"Quote for {symbol}:\n{self._format_quote(result)}"
                else:
                    # Try to extract from original query
                    symbols = self._extract_symbols(query)
                    if symbols:
                        result = await self.get_stock_price(symbols[0])
                        response = f"Quote for {symbols[0]}:\n{self._format_quote(result)}"
                    else:
                        response = "Please specify a stock symbol to get a quote."
            elif intent == "portfolio":
                result = await self.get_portfolio()
                response = f"Here are your current positions:\n{self._format_positions(result)}"
            elif intent == "market_movers":
                result = await self.get_market_movers()
                response = f"Here's what's moving in the market today:\n{self._format_market_movers(result)}"
            elif intent == "market_hours":
                result = await self.get_market_hours()
                is_open = result.get("equity", {}).get("EQ", {}).get("isOpen", False)
                response = f"The market is currently {'**open**' if is_open else '**closed**'}."
            elif intent == "transaction_history":
                result = await self.get_transactions()
                response = f"Here's your recent transaction history:\n{self._format_transactions(result)}"
            elif intent == "top_stocks":
                # Get top movers as proxy for "top stocks"
                result = await self.get_market_movers()
                response = f"Here are today's top stock movers:\n{self._format_market_movers(result)}"
            else:  # account_balance or default
                result = await self.get_account_summary()
                response = f"Here's your Schwab account summary:\n{self._format_account_summary(result)}"
            
            return {
                "response": response,
                "status": "success"
            }
        except json.JSONDecodeError as e:
            # LLM failed to return valid JSON - raise error instead of falling back to keywords
            logger.error(f"LLM returned invalid JSON: {e}")
            raise SchwabAgentError(f"Failed to route query - LLM returned invalid response: {e}")
        except SchwabAgentError as e:
            return {
                "response": f"Error accessing Schwab: {str(e)}",
                "status": "error"
            }
        except Exception as e:
            return {
                "response": f"An error occurred: {str(e)}",
                "status": "error"
            }
    
    def _extract_symbols(self, query: str) -> list[str]:
        """Extract stock symbols from a query string."""
        import re
        # Match uppercase words that look like stock symbols (1-5 letters)
        matches = re.findall(r'\b[A-Z]{1,5}\b', query)
        return matches if matches else []
    
    def _format_account_summary(self, data: dict[str, Any]) -> str:
        """Format account summary data for display."""
        if not data:
            return "No account data available."
        
        lines = []
        
        # Navigate to the correct nested structure
        securities_account = data.get("securitiesAccount", data)
        current_balances = securities_account.get("currentBalances", {})
        
        if current_balances:
            cash_available = current_balances.get('cashAvailableForTrading', 0) or 0
            cash_balance = current_balances.get('cashBalance', 0) or 0
            liquidation_value = current_balances.get('liquidationValue', 0) or 0
            equity = current_balances.get('equity', 0) or 0
            long_market_value = current_balances.get('longMarketValue', 0) or 0
            buying_power = current_balances.get('buyingPower', 0) or 0
            
            lines.append("### 💰 Account Balances")
            lines.append(f"• **Cash Available for Trading:** ${cash_available:,.2f}")
            lines.append(f"• **Cash Balance:** ${cash_balance:,.2f}")
            lines.append(f"• **Account Value:** ${liquidation_value:,.2f}")
            lines.append(f"• **Equity:** ${equity:,.2f}")
            lines.append(f"• **Long Market Value:** ${long_market_value:,.2f}")
            lines.append(f"• **Buying Power:** ${buying_power:,.2f}")
        else:
            return "Could not find balance information in account data."
        
        return "\n".join(lines)
    
    def _format_positions(self, data: list[dict[str, Any]]) -> str:
        """Format positions data for display."""
        if not data:
            return "No positions found."
        
        lines = []
        for pos in data[:10]:  # Limit to first 10
            symbol = pos.get("instrument", {}).get("symbol", "Unknown")
            qty = pos.get("longQuantity", 0) or pos.get("shortQuantity", 0)
            value = pos.get("marketValue", 0)
            lines.append(f"• {symbol}: {qty} shares (${value:,.2f})")
        return "\n".join(lines)
    
    def _format_quote(self, data: dict[str, Any]) -> str:
        """Format quote data for display."""
        if not data:
            return "No quote data available."
        
        # API returns {"SYMBOL": {"quote": {...}}} structure
        # Unwrap the symbol key if present
        if data and len(data) == 1:
            symbol_key = next(iter(data.keys()))
            if isinstance(data[symbol_key], dict):
                data = data[symbol_key]
        
        quote = data.get("quote", data)
        price = quote.get("lastPrice", quote.get("mark", 0))
        change = quote.get("netChange", 0)
        pct = quote.get("netPercentChange", quote.get("netPercentChangeInDouble", 0))
        return f"• Price: ${price:,.2f}\n• Change: ${change:+,.2f} ({pct:+.2f}%)"
    
    def _format_orders(self, data: list[dict[str, Any]]) -> str:
        """Format orders data for display."""
        if not data:
            return "No recent orders found."
        
        lines = []
        for order in data[:5]:  # Limit to first 5
            status = order.get("status", "Unknown")
            symbol = order.get("orderLegCollection", [{}])[0].get("instrument", {}).get("symbol", "Unknown")
            lines.append(f"• {symbol}: {status}")
        return "\n".join(lines)
    
    def _format_market_movers(self, data: dict[str, Any]) -> str:
        """Format market movers data for display.
        
        Args:
            data: Market movers data from Schwab API
            
        Returns:
            Formatted string with gainers and losers
        """
        lines = []
        
        # Handle different response formats
        screeners = data.get("screeners", [])
        if screeners:
            for screener in screeners:
                direction = screener.get("direction", "")
                instruments = screener.get("instruments", [])
                
                if direction.lower() == "up":
                    lines.append("📈 **Top Gainers:**")
                elif direction.lower() == "down":
                    lines.append("📉 **Top Losers:**")
                else:
                    lines.append(f"**{direction}:**")
                
                for inst in instruments[:5]:  # Top 5
                    symbol = inst.get("symbol", "?")
                    desc = inst.get("description", "")[:30]
                    change = inst.get("netChange", 0)
                    pct = inst.get("netPercentChange", 0)
                    last = inst.get("lastPrice", inst.get("last", 0))
                    
                    emoji = "🟢" if change >= 0 else "🔴"
                    lines.append(f"  {emoji} **{symbol}** ${last:,.2f} ({pct:+.2f}%)")
                
                lines.append("")
        else:
            # Fallback for simple format
            lines.append("📊 Market movers data not available in expected format.")
        
        return "\n".join(lines) if lines else "No market mover data available."

    def _format_transactions(self, data: list[dict[str, Any]] | dict[str, Any]) -> str:
        """Format transaction history for display.
        
        Args:
            data: Transaction data from Schwab API
            
        Returns:
            Formatted string with transaction history
        """
        # Handle dict wrapper
        if isinstance(data, dict):
            transactions = data.get("transactions", data.get("securitiesAccount", {}).get("transactions", []))
            if not transactions:
                transactions = [data] if data else []
        else:
            transactions = data if data else []
        
        if not transactions:
            return "No recent transactions found."
        
        lines = ["📜 **Recent Transactions:**\n"]
        
        for txn in transactions[:20]:  # Limit to 20
            txn_type = txn.get("type", txn.get("transactionType", "Unknown"))
            description = txn.get("description", "")
            date = txn.get("transactionDate", txn.get("settlementDate", ""))[:10]  # Just the date
            
            # Get net amount
            net_amount = txn.get("netAmount", 0)
            
            # Get trade details if available
            trade_info = ""
            if "transferItems" in txn:
                for item in txn.get("transferItems", []):
                    inst = item.get("instrument", {})
                    symbol = inst.get("symbol", "")
                    amount = item.get("amount", 0)
                    if symbol:
                        trade_info = f" | {symbol}"
                        if amount:
                            trade_info += f" ({amount} shares)"
            
            # Format line
            emoji = "💰" if net_amount > 0 else "💸" if net_amount < 0 else "📋"
            line = f"{emoji} **{date}** - {txn_type}"
            if description:
                line += f": {description[:40]}"
            if trade_info:
                line += trade_info
            if net_amount != 0:
                line += f" | ${net_amount:,.2f}"
            
            lines.append(line)
        
        if len(transactions) > 20:
            lines.append(f"\n_...and {len(transactions) - 20} more transactions_")
        
        return "\n".join(lines)

    async def _handle_order_request(self, query: str) -> str:
        """Handle order placement requests using LLM to parse user intent.
        
        Args:
            query: User's order request (e.g., "buy 10 shares of AAPL at $150")
            
        Returns:
            Formatted response about the order
        """
        import json
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Use LLM to parse the order request
        system_prompt = """You are a trading order parser. Extract order details from user requests.
        
Return a JSON object with:
- action: "buy" or "sell"
- symbol: stock ticker (uppercase)
- quantity: number of shares (integer)
- order_type: "market" or "limit"
- price: limit price if specified (number, or null for market orders)
- take_profit: take profit price if specified (number, or null)
- stop_loss: stop loss price if specified (number, or null)

Example input: "buy 10 shares of AAPL at $150 with take profit at 180 and stop loss at 120"
Example output: {"action": "buy", "symbol": "AAPL", "quantity": 10, "order_type": "limit", "price": 150, "take_profit": 180, "stop_loss": 120}

Return ONLY the JSON object, no other text."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = response.content
            
            # Handle case where content is a list (some LLM responses)
            if isinstance(content, list):
                content = content[0] if content else ""
                if isinstance(content, dict):
                    content = content.get("text", str(content))
            
            content = str(content).strip()
            
            # Parse the LLM response
            # Clean up markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            order_details = json.loads(content)
            
            action = order_details.get("action", "").lower()
            symbol = order_details.get("symbol", "").upper()
            quantity = int(order_details.get("quantity", 0))
            order_type = order_details.get("order_type", "market").lower()
            price = order_details.get("price")
            take_profit = order_details.get("take_profit")
            stop_loss = order_details.get("stop_loss")
            
            if not symbol or quantity <= 0:
                return "I couldn't understand the order details. Please specify the stock symbol and quantity."
            
            # Get account hash
            account_hash = await self._get_account_hash()
            
            # Place the main order
            if order_type == "limit" and price:
                if action == "buy":
                    result = await self.place_limit_buy(account_hash, symbol, quantity, float(price))
                else:
                    result = await self.place_limit_sell(account_hash, symbol, quantity, float(price))
                order_desc = f"limit {action} order for {quantity} shares of {symbol} at ${price:,.2f}"
            else:
                if action == "buy":
                    result = await self.place_market_buy(account_hash, symbol, quantity)
                else:
                    result = await self.place_market_sell(account_hash, symbol, quantity)
                order_desc = f"market {action} order for {quantity} shares of {symbol}"
            
            # Build response
            response_lines = [f"✅ Placed {order_desc}"]
            
            # Note about bracket orders (take profit / stop loss)
            if take_profit or stop_loss:
                response_lines.append("")
                response_lines.append("⚠️ **Note:** Schwab API requires separate orders for take-profit and stop-loss.")
                if take_profit:
                    response_lines.append(f"  • To set take-profit at ${take_profit:,.2f}, place a separate limit sell order")
                if stop_loss:
                    response_lines.append(f"  • To set stop-loss at ${stop_loss:,.2f}, place a separate stop order")
            
            return "\n".join(response_lines)
            
        except json.JSONDecodeError as e:
            return f"I couldn't parse the order details. Please try again with a clearer format like 'buy 10 shares of AAPL at $150'."
        except Exception as e:
            return f"Error placing order: {str(e)}"

    async def _handle_cancel_order(self, query: str, details: dict) -> str:
        """Handle order cancellation requests.
        
        Args:
            query: User's cancel request
            details: Extracted details from routing (may contain order_id)
            
        Returns:
            Formatted response about the cancellation
        """
        try:
            # Get account hash
            account_hash = await self._get_account_hash()
            
            # Check if order_id was provided
            order_id = details.get("order_id")
            
            if order_id:
                # Cancel specific order
                result = await self.execute_tool("cancel_order", {
                    "account_hash": account_hash,
                    "order_id": int(order_id)
                })
                return f"✅ Order {order_id} has been cancelled."
            else:
                # No order ID - need to find the most recent order
                orders = await self.get_open_orders()
                
                if not orders:
                    return "You don't have any open orders to cancel."
                
                # Get the most recent order
                if isinstance(orders, list) and len(orders) > 0:
                    recent_order = orders[0]
                    order_id = recent_order.get("orderId")
                    symbol = recent_order.get("orderLegCollection", [{}])[0].get("instrument", {}).get("symbol", "Unknown")
                    
                    if order_id:
                        result = await self.execute_tool("cancel_order", {
                            "account_hash": account_hash,
                            "order_id": order_id
                        })
                        return f"✅ Cancelled your most recent order (Order #{order_id} for {symbol})."
                
                return "I couldn't find an order to cancel. Please specify the order ID."
                
        except Exception as e:
            return f"Error cancelling order: {str(e)}"

    async def _get_account_hash(self, account_hash: str | None = None) -> str:
        """Get account hash, using default or fetching first available.
        
        Args:
            account_hash: Explicit account hash (optional)
        
        Returns:
            Account hash string
        """
        if account_hash:
            return account_hash
        if self.default_account_hash:
            return self.default_account_hash
        
        # Fetch account numbers and use first one
        accounts = await self.execute_tool("get_account_numbers", {})
        if accounts and len(accounts) > 0:
            return accounts[0]["hashValue"]
        
        raise SchwabAgentError("No accounts found")
    
    async def get_account_summary(self, account_hash: str | None = None) -> dict[str, Any]:
        """Get a summary of the account.
        
        Args:
            account_hash: Account hash (uses default if not provided)
        
        Returns:
            Account summary including balances and positions
        """
        hash_value = await self._get_account_hash(account_hash)
        
        account = await self.execute_tool("get_account", {
            "account_hash": hash_value,
            "fields": ["positions"]
        })
        
        return account
    
    async def get_portfolio(self, account_hash: str | None = None) -> dict[str, Any]:
        """Get portfolio positions.
        
        Args:
            account_hash: Account hash (uses default if not provided)
        
        Returns:
            Portfolio with positions
        """
        hash_value = await self._get_account_hash(account_hash)
        
        account = await self.execute_tool("get_account", {
            "account_hash": hash_value,
            "fields": ["positions"]
        })
        
        return {
            "positions": account.get("securitiesAccount", {}).get("positions", []),
            "balances": account.get("securitiesAccount", {}).get("currentBalances", {})
        }
    
    async def get_stock_price(self, symbol: str) -> dict[str, Any]:
        """Get stock price.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Quote data including price
        """
        quote = await self.execute_tool("get_quote", {"symbol": symbol})
        return quote
    
    async def place_market_buy(
        self,
        account_hash: str | None = None,
        symbol: str = "",
        quantity: int = 0
    ) -> dict[str, Any]:
        """Place a market buy order.
        
        Args:
            account_hash: Account hash (uses default if not provided)
            symbol: Stock symbol
            quantity: Number of shares
        
        Returns:
            Order result
        """
        hash_value = await self._get_account_hash(account_hash)
        
        return await self.execute_tool("equity_buy_market", {
            "account_hash": hash_value,
            "symbol": symbol,
            "quantity": quantity
        })
    
    async def place_limit_buy(
        self,
        account_hash: str | None = None,
        symbol: str = "",
        quantity: int = 0,
        price: float = 0.0
    ) -> dict[str, Any]:
        """Place a limit buy order.
        
        Args:
            account_hash: Account hash (uses default if not provided)
            symbol: Stock symbol
            quantity: Number of shares
            price: Limit price
        
        Returns:
            Order result
        """
        hash_value = await self._get_account_hash(account_hash)
        
        return await self.execute_tool("equity_buy_limit", {
            "account_hash": hash_value,
            "symbol": symbol,
            "quantity": quantity,
            "price": str(price)
        })
    
    async def place_market_sell(
        self,
        account_hash: str | None = None,
        symbol: str = "",
        quantity: int = 0
    ) -> dict[str, Any]:
        """Place a market sell order.
        
        Args:
            account_hash: Account hash (uses default if not provided)
            symbol: Stock symbol
            quantity: Number of shares
        
        Returns:
            Order result
        """
        hash_value = await self._get_account_hash(account_hash)
        
        return await self.execute_tool("equity_sell_market", {
            "account_hash": hash_value,
            "symbol": symbol,
            "quantity": quantity
        })
    
    async def place_limit_sell(
        self,
        account_hash: str | None = None,
        symbol: str = "",
        quantity: int = 0,
        price: float = 0.0
    ) -> dict[str, Any]:
        """Place a limit sell order.
        
        Args:
            account_hash: Account hash (uses default if not provided)
            symbol: Stock symbol
            quantity: Number of shares
            price: Limit price
        
        Returns:
            Order result
        """
        hash_value = await self._get_account_hash(account_hash)
        
        return await self.execute_tool("equity_sell_limit", {
            "account_hash": hash_value,
            "symbol": symbol,
            "quantity": quantity,
            "price": str(price)
        })
    
    async def get_open_orders(self, account_hash: str | None = None) -> list:
        """Get open orders.
        
        Args:
            account_hash: Account hash (uses default if not provided)
        
        Returns:
            List of open orders
        """
        import datetime
        hash_value = await self._get_account_hash(account_hash)
        
        # Schwab API requires both from and to entered datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        from_date = (now - datetime.timedelta(days=60)).strftime("%Y-%m-%dT00:00:00.000Z")
        to_date = now.strftime("%Y-%m-%dT23:59:59.000Z")
        
        # Don't filter by status - get all orders and filter client-side for cancelable ones
        orders = await self.execute_tool("get_orders_for_account", {
            "account_hash": hash_value,
            "from_entered_datetime": from_date,
            "to_entered_datetime": to_date
        })
        
        # Filter to only show cancelable orders (open orders)
        if isinstance(orders, list):
            return [o for o in orders if o.get("cancelable", False)]
        return orders
    
    async def get_market_movers(self, index: str = "$SPX") -> dict[str, Any]:
        """Get market movers.
        
        Args:
            index: Index symbol (default: $SPX)
        
        Returns:
            Market movers data
        """
        return await self.execute_tool("get_movers", {"index": index})
    
    async def get_transactions(self, days: int = 30) -> list[dict[str, Any]]:
        """Get recent transaction history.
        
        Args:
            days: Number of days of history (default: 30)
        
        Returns:
            List of transactions
        """
        from datetime import datetime, timedelta, timezone
        
        account = await self._get_account_hash()
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        return await self.execute_tool("get_transactions", {
            "account_hash": account,
            "start_date": start_date.strftime("%Y-%m-%dT00:00:00.000Z"),
            "end_date": end_date.strftime("%Y-%m-%dT23:59:59.000Z")
        })
    
    async def get_market_hours(self, markets: list[str] | None = None) -> dict[str, Any]:
        """Get market hours.
        
        Args:
            markets: List of markets (default: ["equity"])
        
        Returns:
            Market hours data
        """
        if markets is None:
            markets = ["equity"]
        
        return await self.execute_tool("get_market_hours", {"markets": markets})
