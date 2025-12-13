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
        checkpointer: Optional checkpointer for state persistence
    """
    
    def __init__(
        self,
        researcher_server: ResearcherMCPServer | None = None,
        llm: Any | None = None,
        finnhub_api_key: str | None = None,
        openai_api_key: str | None = None,
        enable_chain_of_thought: bool = False,
        checkpointer: Any | None = None,
        mcp_server: Any | None = None  # Alias for researcher_server
    ):
        """Initialize the Researcher Agent.
        
        Args:
            researcher_server: Pre-configured MCP server instance (optional)
            llm: Language model instance for reasoning (optional)
            finnhub_api_key: Finnhub API key (optional if server provided)
            openai_api_key: OpenAI API key (optional if server provided)
            enable_chain_of_thought: Whether to enable CoT reasoning
            checkpointer: Optional checkpointer for state persistence
            mcp_server: Alias for researcher_server (for consistency with other agents)
        """
        # Support both researcher_server and mcp_server parameter names
        server = researcher_server or mcp_server
        if server is not None:
            self.researcher_server = server
        else:
            self.researcher_server = ResearcherMCPServer(
                finnhub_api_key=finnhub_api_key,
                openai_api_key=openai_api_key
            )
        
        self.llm = llm
        self.enable_chain_of_thought = enable_chain_of_thought
        self.chain_of_thought = ChainOfThought() if enable_chain_of_thought else None
        self.checkpointer = checkpointer
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
        
        # Compile with checkpointer if provided
        if self.checkpointer is not None:
            return workflow.compile(checkpointer=self.checkpointer)
        return workflow.compile()
    
    def _analyze_query(self, state: ResearcherAgentState) -> ResearcherAgentState:
        """Analyze the user query using LLM to determine what research to perform.
        
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
        
        # Extract stock symbols from query
        symbols = self._extract_symbols(query)
        
        # Use LLM to determine what tools to call
        if self.llm:
            tool_calls = self._llm_analyze_query(query, symbols)
            state["tool_calls"] = tool_calls
        else:
            raise ResearcherAgentError("LLM is required for query analysis. No LLM configured.")
        
        return state
    
    def _llm_analyze_query(self, query: str, symbols: list[str]) -> list[dict]:
        """Use LLM to determine which research tools to call.
        
        Args:
            query: User's query
            symbols: Extracted stock symbols
            
        Returns:
            List of tool calls to make
        """
        import json
        
        system_prompt = """You are a research query analyzer. Analyze the user's query and determine which research tools to call.

Available tools:
- get_stock_quote: Get current stock quote with price, change, volume. Use for price/quote requests.
- research_stock: Comprehensive stock research with financials, news, analysis. Use for deep research.
- get_basic_financials: Financial metrics like P/E ratio, market cap. Use for financial analysis.
- get_company_news: Recent news about a company. Use for news requests.
- get_analyst_recommendations: Analyst buy/sell/hold ratings. Use for recommendation requests.
- get_company_peers: Get competitor companies. Use for peer/comparison requests.
- get_earnings_history: Historical earnings data. Use for earnings analysis.
- get_company_profile: Company description and info. Use for company overview.
- get_market_news: General market news (no symbol needed). Use for market-wide news.
- web_search: Search the web for information. Use for general lookups or opinions.

Given the user query and any stock symbols found, return a JSON array of tool calls.
Each tool call should have "tool" (tool name) and "params" (parameters object).

For symbol-specific tools, include the symbol in params.
For comprehensive queries about a stock, use research_stock.
For multiple aspects, include multiple tool calls.

Example output:
[
  {"tool": "get_stock_quote", "params": {"symbol": "AAPL"}},
  {"tool": "get_company_news", "params": {"symbol": "AAPL", "limit": 5}}
]

Return ONLY valid JSON array, no other text."""
        
        user_content = f"Query: {query}\nSymbols found: {symbols if symbols else 'None'}"
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ])
            
            content = _extract_content(response.content)
            
            # Parse JSON response
            tool_calls = json.loads(content)
            
            if isinstance(tool_calls, list):
                return tool_calls
            else:
                # If single tool call, wrap in list
                return [tool_calls]
                
        except json.JSONDecodeError as e:
            raise ResearcherAgentError(f"LLM returned invalid JSON for tool selection: {e}")
        except Exception as e:
            raise ResearcherAgentError(f"Failed to analyze query with LLM: {e}")
    
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
        messages = state.get("messages", [])
        
        # Apply chain of thought if enabled
        if self.enable_chain_of_thought and self.chain_of_thought:
            state = self._apply_chain_of_thought(state)
        
        # If we have an LLM, use it to generate a response
        if self.llm:
            try:
                prompt = self._build_response_prompt(query, research_data, messages)
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

    async def _generate_conversational_response(self, state: ResearcherAgentState) -> ResearcherAgentState:
        """Generate a helpful conversational response when no research tools are needed.
        
        This is used when the query doesn't require stock lookups or web searches,
        but we still want to provide a thoughtful, context-aware response.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with response
        """
        query = state.get("query", "")
        messages = state.get("messages", [])
        
        if not self.llm:
            state["response"] = "I can help you with investment research, stock analysis, and market news. What would you like to know?"
            return state
        
        # Build conversation context
        history_context = ""
        if messages:
            history_lines = []
            for msg in messages[-10:]:  # Last 10 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            if history_lines:
                history_context = "Previous conversation:\n" + "\n".join(history_lines)
        
        prompt = f"""You are Mister Risker, an AI trading and investment assistant.
You have access to tools for:
- Stock quotes and analysis (Finnhub)
- Company financials and news
- Web search for current information
- Coinbase for crypto trading
- Schwab for stocks/options trading
- Solana blockchain data via Solana RPC API

The user has asked a question that may not require looking up specific data.
Use the conversation history to provide a helpful, intelligent response.

IMPORTANT: Do NOT suggest external blockchain explorers like blockchair.com, etherscan.io, or similar services. 
Our data comes directly from the Solana RPC API - you can only reference solana.com or the data we provide.

{history_context}

Current question: {query}

Instructions:
- If the user is asking about something from earlier in the conversation, answer based on that context
- If they're asking a general question you can answer, answer it directly
- If they're asking something that would benefit from research, explain what you can look up for them
- Be conversational, helpful, and never just echo their question back
- Keep your response concise but complete"""

        try:
            from langchain_core.messages import HumanMessage
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            state["response"] = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            state["response"] = f"I can help you with investment research, stock analysis, and market news. Please let me know what you'd like to explore!"
        
        state["history"].append({
            "step": "conversational_response",
            "response_length": len(state["response"])
        })
        
        return state
    
    def _build_response_prompt(self, query: str, research_data: dict, messages: list = None) -> str:
        """Build a prompt for the LLM to generate a response.
        
        Args:
            query: User query
            research_data: Gathered research data
            messages: Optional conversation history for context
        
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
        
        # Build conversation history context
        history_context = ""
        if messages:
            history_lines = []
            for msg in messages[-10:]:  # Last 10 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]  # Truncate long messages
                history_lines.append(f"{role}: {content}")
            if history_lines:
                history_context = f"\n\nPrevious conversation context:\n" + "\n".join(history_lines)
        
        return f"""Based on the following research data, answer the user's question.
{history_context}
User Question: {query}

Research Data:
{research_data}

Please provide a clear, helpful response that addresses the user's question.
Include relevant data points and be specific with numbers when available.
If the user is asking for investment advice, provide analysis but remind them
that this is not financial advice and they should consult a professional.
If the user is asking about something from earlier in the conversation, refer to
the conversation context above.

IMPORTANT: Do NOT suggest external blockchain explorers like blockchair.com, etherscan.io, etc.
Our blockchain data comes directly from chain-specific APIs (like Solana RPC)."""
    
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
        trade_context: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
        conversation_history: list[dict[str, Any]] | None = None  # Alias for messages
    ) -> dict[str, Any]:
        """Run the research agent with a query.
        
        Args:
            query: User query
            context: Optional conversation context from previous run
            return_state: Whether to include full state in response
            return_structured: Whether to return structured research data
            trade_context: Optional trade context for relevant research
            messages: Conversation history for context
            config: LangGraph config with thread_id for checkpointer
            conversation_history: Alias for messages parameter
        
        Returns:
            Response dictionary with status, response, and optional data
        """
        # Support both messages and conversation_history parameter names
        history = messages or conversation_history or []
        
        # Build initial state
        initial_state: ResearcherAgentState = {
            "query": query,
            "messages": history,  # Include conversation history
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
            # No specific tools needed - use LLM to generate a helpful response
            # based on conversation history and the user's query
            state = await self._generate_conversational_response(state)
            state["status"] = "success"
        
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
