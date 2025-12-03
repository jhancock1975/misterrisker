"""Researcher Agent - LangGraph agent for investment research.

This module provides a LangGraph-based agent that uses the Researcher MCP
Server tools to gather information for trading and investment decisions.
"""

import asyncio
import os
from typing import Any, AsyncIterator, TypedDict
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.mcp_servers.researcher import ResearcherMCPServer, ResearcherAPIError
from src.agents.chain_of_thought import ChainOfThought, ReasoningType


def _extract_content(content) -> str:
    """Extract text content from LLM response.
    
    The OpenAI Responses API returns content as a list of content blocks,
    not a simple string. This handles both formats.
    
    Args:
        content: Response content (string or list)
    
    Returns:
        Extracted text as a string
    """
    if content is None:
        return ""
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        # Responses API returns list of content blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    
    return str(content)


class ResearcherAgentError(Exception):
    """Exception raised by the Researcher Agent.
    
    Attributes:
        message: Error message
        tool_name: Name of the tool that caused the error (if applicable)
    """
    
    def __init__(self, message: str, tool_name: str | None = None):
        """Initialize ResearcherAgentError.
        
        Args:
            message: Error message
            tool_name: Name of the tool that caused the error
        """
        self.message = message
        self.tool_name = tool_name
        super().__init__(self.message)


class ResearcherAgentState(TypedDict, total=False):
    """State for the Researcher agent workflow.
    
    Attributes:
        messages: Conversation history
        query: Current user query
        tool_calls: List of tool calls to make
        tool_results: Results from tool calls
        research_data: Aggregated research data
        response: Final response to user
        context: Conversation context for follow-ups
        history: History of workflow steps
        status: Current status (success/error)
        reasoning_steps: Chain of thought reasoning steps
    """
    messages: list[dict[str, Any]]
    query: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    research_data: dict[str, Any]
    response: str
    context: dict[str, Any]
    history: list[dict[str, Any]]
    status: str
    reasoning_steps: list[str]


class ResearcherAgent:
    """LangGraph agent for investment research operations.
    
    This agent uses the ResearcherMCPServer to gather financial data
    and analysis for making trading and investment decisions.
    
    Attributes:
        researcher_server: The Researcher MCP Server instance
        llm: Language model for reasoning
        enable_chain_of_thought: Whether CoT reasoning is enabled
        chain_of_thought: ChainOfThought instance for structured reasoning
        _workflow: The compiled LangGraph workflow
    """
    
    def __init__(
        self,
        researcher_server: ResearcherMCPServer | None = None,
        llm: Any | None = None,
        finnhub_api_key: str | None = None,
        openai_api_key: str | None = None,
        enable_chain_of_thought: bool = False
    ):
        """Initialize the Researcher Agent.
        
        Args:
            researcher_server: Pre-configured MCP server instance (optional)
            llm: Language model instance for reasoning (optional)
            finnhub_api_key: Finnhub API key (optional if server provided)
            openai_api_key: OpenAI API key (optional if server provided)
            enable_chain_of_thought: Whether to enable CoT reasoning
        """
        if researcher_server is not None:
            self.researcher_server = researcher_server
        else:
            self.researcher_server = ResearcherMCPServer(
                finnhub_api_key=finnhub_api_key,
                openai_api_key=openai_api_key
            )
        
        self.llm = llm
        self.enable_chain_of_thought = enable_chain_of_thought
        self.chain_of_thought = ChainOfThought() if enable_chain_of_thought else None
        self._workflow = self._build_workflow()
    
    def get_tools(self) -> list[dict[str, Any]]:
        """Get list of available tools from the MCP server.
        
        Returns:
            List of tool definitions
        """
        return self.researcher_server.list_tools()
    
    def get_workflow(self) -> CompiledStateGraph:
        """Get the compiled LangGraph workflow.
        
        Returns:
            Compiled LangGraph workflow
        """
        return self._workflow
    
    def _build_workflow(self) -> CompiledStateGraph:
        """Build the LangGraph workflow for research operations.
        
        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(ResearcherAgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("execute_research", self._execute_research)
        workflow.add_node("synthesize_results", self._synthesize_results)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add edges
        workflow.add_conditional_edges(
            "analyze_query",
            self._route_after_analyze,
            {
                "execute_research": "execute_research",
                "end": END,
                "error": "handle_error",
            }
        )
        
        workflow.add_conditional_edges(
            "execute_research",
            self._route_after_execute,
            {
                "synthesize_results": "synthesize_results",
                "error": "handle_error",
            }
        )
        
        workflow.add_edge("synthesize_results", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: ResearcherAgentState) -> ResearcherAgentState:
        """Analyze the user query and determine what research to perform.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with tool calls to make
        """
        query = state.get("query", "")
        messages = state.get("messages", [])
        context = state.get("context", {})
        
        # Initialize state
        state["tool_calls"] = []
        state["tool_results"] = []
        state["history"] = state.get("history", [])
        state["history"].append({"step": "analyze_query", "query": query})
        
        # Simple query analysis to determine tools to call
        query_lower = query.lower()
        
        # Extract stock symbols from query
        symbols = self._extract_symbols(query)
        
        if symbols:
            for symbol in symbols:
                # Determine what data to fetch based on query
                if any(word in query_lower for word in ["price", "quote", "trading"]):
                    state["tool_calls"].append({
                        "tool": "get_stock_quote",
                        "params": {"symbol": symbol}
                    })
                
                if any(word in query_lower for word in ["analysis", "research", "complete", "comprehensive", "all"]):
                    state["tool_calls"].append({
                        "tool": "research_stock",
                        "params": {"symbol": symbol, "include_web_search": True}
                    })
                elif any(word in query_lower for word in ["financials", "metrics", "pe", "ratio"]):
                    state["tool_calls"].append({
                        "tool": "get_basic_financials",
                        "params": {"symbol": symbol}
                    })
                    
                if any(word in query_lower for word in ["news", "latest", "recent"]):
                    state["tool_calls"].append({
                        "tool": "get_company_news",
                        "params": {"symbol": symbol, "limit": 5}
                    })
                
                if any(word in query_lower for word in ["analyst", "recommendation", "buy", "sell", "hold", "rating"]):
                    state["tool_calls"].append({
                        "tool": "get_analyst_recommendations",
                        "params": {"symbol": symbol}
                    })
                
                if any(word in query_lower for word in ["competitor", "peer", "compare"]):
                    state["tool_calls"].append({
                        "tool": "get_company_peers",
                        "params": {"symbol": symbol}
                    })
                
                if any(word in query_lower for word in ["earnings", "eps", "quarter"]):
                    state["tool_calls"].append({
                        "tool": "get_earnings_history",
                        "params": {"symbol": symbol}
                    })
                
                if any(word in query_lower for word in ["profile", "company", "about", "what is"]):
                    state["tool_calls"].append({
                        "tool": "get_company_profile",
                        "params": {"symbol": symbol}
                    })
                
                # If no specific request, do comprehensive research
                if not state["tool_calls"]:
                    state["tool_calls"].append({
                        "tool": "research_stock",
                        "params": {"symbol": symbol, "include_web_search": False}
                    })
        
        # Check for market/general news request (no specific symbol)
        news_keywords = ["news", "latest", "recent", "happening", "today", "market", "headlines"]
        if any(word in query_lower for word in news_keywords):
            # Add market news if not already getting company-specific news
            if not any(tc.get("tool") == "get_company_news" for tc in state["tool_calls"]):
                state["tool_calls"].append({
                    "tool": "get_market_news",
                    "params": {"category": "general", "limit": 10}
                })
            # Also do a web search for more context
            if not any(tc.get("tool") == "web_search" for tc in state["tool_calls"]):
                state["tool_calls"].append({
                    "tool": "web_search",
                    "params": {"query": query}
                })
        
        # Check for web search request
        if any(word in query_lower for word in ["search", "find", "look up", "outlook", "opinion"]):
            if not any(tc.get("tool") == "web_search" for tc in state["tool_calls"]):
                state["tool_calls"].append({
                    "tool": "web_search",
                    "params": {"query": query}
                })
        
        return state
    
    def _extract_symbols(self, query: str) -> list[str]:
        """Extract stock symbols from a query.
        
        Args:
            query: User query string
        
        Returns:
            List of stock symbols found
        """
        import re
        
        # Common stock symbols pattern (1-5 uppercase letters)
        # This is a simple heuristic - in production, validate against known symbols
        symbols = []
        
        # Look for explicit symbol mentions
        pattern = r'\b([A-Z]{1,5})\b'
        matches = re.findall(pattern, query)
        
        # Filter common words that might match
        exclude_words = {
            "I", "A", "THE", "AND", "OR", "IS", "IT", "TO", "IN", "OF",
            "FOR", "ON", "AT", "BY", "FROM", "UP", "OUT", "IF", "SO",
            "AN", "AS", "BE", "DO", "GO", "HE", "ME", "MY", "NO", "OK",
            "WE", "US", "VS", "BUY", "SELL", "GET", "ALL", "USD", "P", "E"
        }
        
        for match in matches:
            if match not in exclude_words:
                symbols.append(match)
        
        # Also look for common company name to symbol mappings
        company_symbols = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "meta": "META",
            "facebook": "META",
            "tesla": "TSLA",
            "nvidia": "NVDA",
            "netflix": "NFLX"
        }
        
        query_lower = query.lower()
        for company, symbol in company_symbols.items():
            if company in query_lower and symbol not in symbols:
                symbols.append(symbol)
        
        return symbols
    
    def _route_after_analyze(self, state: ResearcherAgentState) -> str:
        """Route after query analysis.
        
        Args:
            state: Current workflow state
        
        Returns:
            Name of next node
        """
        if state.get("tool_calls"):
            return "execute_research"
        return "end"
    
    async def _execute_research_async(self, state: ResearcherAgentState) -> ResearcherAgentState:
        """Execute the research tool calls asynchronously.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with tool results
        """
        tool_calls = state.get("tool_calls", [])
        results = []
        
        for tool_call in tool_calls:
            try:
                result = await self.researcher_server.call_tool(
                    tool_call["tool"],
                    tool_call["params"]
                )
                results.append({
                    "tool": tool_call["tool"],
                    "params": tool_call["params"],
                    "result": result,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "tool": tool_call["tool"],
                    "params": tool_call["params"],
                    "result": {"error": str(e)},
                    "success": False
                })
        
        state["tool_results"] = results
        state["history"].append({
            "step": "execute_research",
            "tools_called": len(tool_calls),
            "successful": sum(1 for r in results if r["success"])
        })
        
        return state
    
    def _execute_research(self, state: ResearcherAgentState) -> ResearcherAgentState:
        """Execute research - wrapper for async execution.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state
        """
        # Mark that async execution is needed
        state["_needs_async"] = True
        return state
    
    def _route_after_execute(self, state: ResearcherAgentState) -> str:
        """Route after tool execution.
        
        Args:
            state: Current workflow state
        
        Returns:
            Name of next node
        """
        if state.get("tool_results"):
            return "synthesize_results"
        return "error"
    
    def _synthesize_results(self, state: ResearcherAgentState) -> ResearcherAgentState:
        """Synthesize research results into structured data.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with research data
        """
        results = state.get("tool_results", [])
        research_data = {}
        
        for result in results:
            tool = result["tool"]
            data = result["result"]
            
            if result["success"]:
                if tool == "research_stock":
                    # Comprehensive research result
                    research_data.update(data)
                elif tool == "get_stock_quote":
                    research_data["quote"] = data
                elif tool == "get_company_profile":
                    research_data["profile"] = data
                elif tool == "get_basic_financials":
                    research_data["financials"] = data
                elif tool == "get_analyst_recommendations":
                    research_data["recommendations"] = data
                elif tool == "get_company_news":
                    research_data["news"] = data
                elif tool == "get_company_peers":
                    research_data["peers"] = data
                elif tool == "get_earnings_history":
                    research_data["earnings"] = data
                elif tool == "get_market_news":
                    research_data["market_news"] = data
                elif tool == "web_search":
                    research_data["web_search"] = data
        
        state["research_data"] = research_data
        state["history"].append({
            "step": "synthesize_results",
            "data_keys": list(research_data.keys())
        })
        
        return state
    
    def _generate_response(self, state: ResearcherAgentState) -> ResearcherAgentState:
        """Generate a response based on research data.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with response
        """
        research_data = state.get("research_data", {})
        query = state.get("query", "")
        
        # Apply chain of thought if enabled
        if self.enable_chain_of_thought and self.chain_of_thought:
            state = self._apply_chain_of_thought(state)
        
        # If we have an LLM, use it to generate a response
        if self.llm:
            try:
                prompt = self._build_response_prompt(query, research_data)
                response = self.llm.invoke(prompt)
                raw_content = response.content if hasattr(response, "content") else str(response)
                state["response"] = _extract_content(raw_content)
            except Exception as e:
                state["response"] = self._generate_fallback_response(research_data)
        else:
            state["response"] = self._generate_fallback_response(research_data)
        
        state["status"] = "success"
        state["context"] = {
            "last_query": query,
            "symbols": list(set(
                r.get("params", {}).get("symbol", "")
                for r in state.get("tool_results", [])
                if r.get("params", {}).get("symbol")
            )),
            "research_data": research_data
        }
        
        state["history"].append({
            "step": "generate_response",
            "response_length": len(state["response"])
        })
        
        return state
    
    def _build_response_prompt(self, query: str, research_data: dict) -> str:
        """Build a prompt for the LLM to generate a response.
        
        Args:
            query: User query
            research_data: Gathered research data
        
        Returns:
            Prompt string
        """
        # Use CoT prompt if enabled
        if self.enable_chain_of_thought and self.chain_of_thought:
            reasoning_type = self.chain_of_thought.detect_reasoning_type(query)
            return self.chain_of_thought.get_reasoning_prompt(
                query=query,
                data=research_data,
                reasoning_type=reasoning_type
            )
        
        return f"""Based on the following research data, answer the user's question.

User Question: {query}

Research Data:
{research_data}

Please provide a clear, helpful response that addresses the user's question.
Include relevant data points and be specific with numbers when available.
If the user is asking for investment advice, provide analysis but remind them
that this is not financial advice and they should consult a professional."""
    
    def _apply_chain_of_thought(self, state: ResearcherAgentState) -> ResearcherAgentState:
        """Apply chain of thought reasoning to generate structured response.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with reasoning steps
        """
        if not self.chain_of_thought or not self.llm:
            return state
        
        query = state.get("query", "")
        research_data = state.get("research_data", {})
        
        # Detect reasoning type
        reasoning_type = self.chain_of_thought.detect_reasoning_type(query)
        
        # Generate CoT prompt
        prompt = self.chain_of_thought.get_reasoning_prompt(
            query=query,
            data=research_data,
            reasoning_type=reasoning_type
        )
        
        try:
            # Get LLM response
            response = self.llm.invoke(prompt)
            raw_content = response.content if hasattr(response, "content") else str(response)
            response_text = _extract_content(raw_content)
            
            # Parse the response
            parsed = self.chain_of_thought.parse_response(response_text)
            
            # Update state with reasoning
            state["reasoning_steps"] = parsed.get("reasoning_steps", [])
            
            # Include reasoning in history
            state["history"].append({
                "step": "chain_of_thought",
                "reasoning_type": reasoning_type.value,
                "steps_count": len(state["reasoning_steps"])
            })
        except Exception:
            # If CoT fails, continue without it
            state["reasoning_steps"] = []
        
        return state
    
    def _generate_fallback_response(self, research_data: dict) -> str:
        """Generate a fallback response without LLM.
        
        Args:
            research_data: Gathered research data
        
        Returns:
            Response string
        """
        parts = []
        
        if "quote" in research_data:
            q = research_data["quote"]
            symbol = q.get("symbol", "Stock")
            price = q.get("current_price", "N/A")
            change = q.get("change_percent", 0)
            direction = "up" if change and change > 0 else "down"
            parts.append(f"{symbol} is currently trading at ${price} ({direction} {abs(change or 0):.2f}%).")
        
        if "profile" in research_data:
            p = research_data["profile"]
            name = p.get("name", "")
            industry = p.get("industry", "")
            if name:
                parts.append(f"{name} operates in the {industry} industry.")
        
        if "financials" in research_data:
            f = research_data["financials"]
            pe = f.get("pe_ratio")
            if pe:
                parts.append(f"P/E ratio: {pe:.2f}.")
        
        if "recommendations" in research_data:
            r = research_data["recommendations"].get("recommendations", {})
            buy = r.get("buy", 0) + r.get("strong_buy", 0)
            hold = r.get("hold", 0)
            sell = r.get("sell", 0) + r.get("strong_sell", 0)
            total = buy + hold + sell
            if total > 0:
                parts.append(f"Analyst ratings: {buy} Buy, {hold} Hold, {sell} Sell.")
        
        if "news" in research_data:
            news = research_data["news"].get("news", [])
            if news:
                parts.append(f"Recent news: {news[0].get('headline', 'N/A')}")
        
        if "web_search" in research_data:
            answer = research_data["web_search"].get("answer", "")
            if answer:
                parts.append(f"\nWeb Insights: {answer[:500]}...")
        
        return " ".join(parts) if parts else "Research data gathered successfully."
    
    def _handle_error(self, state: ResearcherAgentState) -> ResearcherAgentState:
        """Handle errors in the workflow.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state
        """
        messages = state.get("messages", [])
        messages.append({
            "role": "assistant",
            "content": "I encountered an error while researching. Please try again."
        })
        state["messages"] = messages
        state["status"] = "error"
        state["response"] = "An error occurred while processing your request."
        
        return state
    
    # ===================
    # Public API Methods
    # ===================
    
    async def run(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        return_state: bool = False,
        return_structured: bool = False,
        trade_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run the research agent with a query.
        
        Args:
            query: User query
            context: Optional conversation context from previous run
            return_state: Whether to include full state in response
            return_structured: Whether to return structured research data
            trade_context: Optional trade context for relevant research
        
        Returns:
            Response dictionary with status, response, and optional data
        """
        # Build initial state
        initial_state: ResearcherAgentState = {
            "query": query,
            "messages": [],
            "tool_calls": [],
            "tool_results": [],
            "research_data": {},
            "response": "",
            "context": context or {},
            "history": [],
            "status": "pending"
        }
        
        # Add trade context to query if provided
        if trade_context:
            symbol = trade_context.get("symbol", "")
            intent = trade_context.get("intent", "")
            if symbol:
                initial_state["query"] = f"{query} (Context: {intent} {symbol})"
        
        # Run the workflow steps manually to handle async
        state = self._analyze_query(initial_state)
        
        # Execute research asynchronously
        if state.get("tool_calls"):
            state = await self._execute_research_async(state)
            state = self._synthesize_results(state)
            state = self._generate_response(state)
        else:
            state["status"] = "success"
            state["response"] = "I couldn't find any specific stocks or research topics in your query. Please specify a stock symbol or topic."
        
        # Build response
        result = {
            "status": state.get("status", "success"),
            "response": state.get("response", "")
        }
        
        # Include reasoning steps if available
        if state.get("reasoning_steps"):
            result["reasoning_steps"] = state["reasoning_steps"]
        
        if return_state:
            result["state"] = state
        
        if return_structured:
            result["research_data"] = state.get("research_data", {})
        
        if context is not None or trade_context is not None:
            result["context"] = state.get("context", {})
        
        return result
    
    async def stream(
        self,
        query: str,
        stream_tools: bool = False
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream research results.
        
        Args:
            query: User query
            stream_tools: Whether to stream tool call events
        
        Yields:
            Stream chunks with type and data
        """
        # Analyze query
        initial_state: ResearcherAgentState = {
            "query": query,
            "messages": [],
            "tool_calls": [],
            "tool_results": [],
            "research_data": {},
            "response": "",
            "context": {},
            "history": [],
            "status": "pending"
        }
        
        state = self._analyze_query(initial_state)
        
        yield {"type": "status", "data": "Analyzing query..."}
        
        # Stream tool calls
        for tool_call in state.get("tool_calls", []):
            if stream_tools:
                yield {
                    "type": "tool",
                    "tool_name": tool_call["tool"],
                    "params": tool_call["params"],
                    "status": "started"
                }
            
            try:
                result = await self.researcher_server.call_tool(
                    tool_call["tool"],
                    tool_call["params"]
                )
                state["tool_results"].append({
                    "tool": tool_call["tool"],
                    "params": tool_call["params"],
                    "result": result,
                    "success": True
                })
                
                if stream_tools:
                    yield {
                        "type": "tool",
                        "tool_name": tool_call["tool"],
                        "status": "completed",
                        "result": result
                    }
            except Exception as e:
                state["tool_results"].append({
                    "tool": tool_call["tool"],
                    "params": tool_call["params"],
                    "result": {"error": str(e)},
                    "success": False
                })
                
                if stream_tools:
                    yield {
                        "type": "tool",
                        "tool_name": tool_call["tool"],
                        "status": "error",
                        "error": str(e)
                    }
        
        # Synthesize and generate response
        state = self._synthesize_results(state)
        state = self._generate_response(state)
        
        yield {"type": "status", "data": "Generating response..."}
        yield {"type": "response", "data": state.get("response", "")}
        yield {"type": "complete", "status": state.get("status", "success")}
