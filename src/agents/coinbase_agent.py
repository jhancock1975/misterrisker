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
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        mcp_server: CoinbaseMCPServer | None = None,
        llm: Any | None = None,
        enable_chain_of_thought: bool = False
    ):
        """Initialize the Coinbase Agent.
        
        Args:
            api_key: Coinbase API key (optional if mcp_server provided)
            api_secret: Coinbase API secret (optional if mcp_server provided)
            mcp_server: Pre-configured MCP server instance (optional)
            llm: Language model for reasoning (optional)
            enable_chain_of_thought: Whether to enable CoT reasoning
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
    
    async def get_portfolio_summary(self) -> dict[str, Any]:
        """Get a summary of the portfolio.
        
        Returns:
            Portfolio summary including balances and positions
        """
        accounts = await self.execute_tool("get_accounts", {})
        
        summary = {
            "accounts": accounts.get("accounts", []),
            "total_positions": len(accounts.get("accounts", []))
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
