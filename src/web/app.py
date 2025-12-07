"""FastAPI web application for Mister Risker trading chat interface.

This module provides a browser-based chat interface that uses an LLM
to interpret natural language requests and execute trades via Coinbase and Schwab.
"""

import os
import json
import asyncio
import logging
import base64
import time
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("mister_risker")
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import uuid

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.mcp_servers.coinbase import CoinbaseMCPServer, CoinbaseAPIError
from src.mcp_servers.schwab import SchwabMCPServer, SchwabAPIError
from src.agents.coinbase_agent import CoinbaseAgent, CoinbaseAgentError
from src.agents.schwab_agent import SchwabAgent, SchwabAgentError
from src.agents.researcher_agent import ResearcherAgent, ResearcherAgentError
from src.agents.supervisor_agent import SupervisorAgent, SupervisorAgentError
from src.agents.finrl_agent import FinRLAgent
from src.services.finrl_service import FinRLService
from src.services.trading_strategy import TradingStrategyService

# Load environment variables
load_dotenv()


class TradingChatBot:
    """Chat bot that uses LLM to interact with Coinbase and Schwab.
    
    This bot interprets natural language requests and uses either
    LangGraph agents (with optional Chain of Thought reasoning) or
    MCP Servers directly to execute trading operations.
    
    Attributes:
        use_agents: Whether to use LangGraph agents instead of direct MCP calls
        enable_chain_of_thought: Whether to enable CoT reasoning in agents
    """
    
    def __init__(
        self,
        use_agents: bool = False,
        enable_chain_of_thought: bool = False
    ):
        """Initialize the chat bot.
        
        Args:
            use_agents: Whether to use LangGraph agents (default: False for backward compat)
            enable_chain_of_thought: Whether to enable CoT reasoning
        """
        # MCP Servers (used when use_agents=False or as fallback)
        self.coinbase_server: CoinbaseMCPServer | None = None
        self.schwab_server: SchwabMCPServer | None = None
        
        # LangGraph Agents (used when use_agents=True)
        self.coinbase_agent: CoinbaseAgent | None = None
        self.schwab_agent: SchwabAgent | None = None
        self.researcher_agent: ResearcherAgent | None = None
        self.finrl_agent: FinRLAgent | None = None
        self.finrl_service: FinRLService | None = None
        self.trading_strategy_service: TradingStrategyService | None = None
        self.supervisor_agent: SupervisorAgent | None = None
        
        # Configuration
        self.use_agents = use_agents
        self.enable_chain_of_thought = enable_chain_of_thought
        
        self.llm: ChatOpenAI | None = None
        self.conversation_history: list = []
        self.active_broker: str = "coinbase"  # Default broker
        
        # Memory/checkpointing for agent state persistence
        self.checkpointer = InMemorySaver()
        self.thread_id = str(uuid.uuid4())
        
        # Store last supervisor logs for frontend debugging
        self.last_supervisor_logs: list = []
        self.last_agent_used: str = ""
        
        # System prompt for the LLM
        self.system_prompt = """You are Mister Risker, a helpful multi-broker trading assistant. You can help users trade on both Coinbase (crypto) and Schwab (stocks/options).

Current active broker: {broker}

You can help users:
**Coinbase (Crypto):**
- Check crypto account balances
- Get current prices for cryptocurrencies
- Place buy and sell orders (market and limit)
- View open orders and portfolio

**Schwab (Stocks/Options):**
- Check stock account balances and positions
- Get stock quotes and option chains
- Place equity and options orders
- View transactions and market data

When users ask about trading, first confirm the details before executing trades.
For buy orders, clarify the amount they want to spend or quantity.
For sell orders, clarify the amount they want to sell.

IMPORTANT: To switch brokers, the user can say "switch to schwab" or "switch to coinbase".

When you need to call a tool, respond with a JSON object in this format:
{{"tool": "tool_name", "params": {{"param1": "value1"}}}}

**Coinbase Tools:**
- get_accounts: Get all crypto account balances
- get_product: Get details for a product (params: product_id like "BTC-USD")
- get_best_bid_ask: Get current prices (params: product_ids as list)
- market_order_buy: Buy crypto with USD (params: product_id, quote_size)
- market_order_sell: Sell crypto (params: product_id, base_size)
- list_orders: List orders (params: order_status as list)

**Schwab Tools:**
- get_account_numbers: Get all Schwab account numbers
- get_account: Get account details (params: account_hash)
- get_accounts: Get all accounts with positions
- get_quote: Get stock quote (params: symbol)
- get_quotes: Get multiple quotes (params: symbols as list)
- get_orders_for_account: Get orders (params: account_hash)
- place_order: Place an order (params: account_hash, order)
- get_option_chain: Get option chain (params: symbol)
- get_movers: Get market movers (params: index like "$DJI")
- get_market_hours: Get market hours (params: markets like "EQUITY")

**Research Commands** (when research agent is available):
- "research AAPL" or "analyze TSLA" - Get investment research with analyst recommendations
- "compare AAPL vs MSFT" - Compare two stocks
- "what are the risks of NVDA" - Get risk assessment

If you don't need to call a tool, just respond normally with text."""

    async def initialize(self):
        """Initialize the MCP servers, agents, and LLM."""
        logger.info("Initializing TradingChatBot...")
        logger.info(f"  use_agents={self.use_agents}, enable_chain_of_thought={self.enable_chain_of_thought}")
        
        # Initialize LLM first (needed for agents)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                api_key=openai_api_key,
                use_responses_api=True,
            )
            logger.info("  ✓ LLM initialized (gpt-4o-mini)")
        else:
            logger.warning("  ✗ OPENAI_API_KEY not set. LLM features will be limited.")
        
        # Initialize Coinbase
        coinbase_api_key = os.getenv("COINBASE_API_KEY")
        coinbase_api_secret = os.getenv("COINBASE_API_SECRET")
        
        if coinbase_api_key and coinbase_api_secret:
            try:
                self.coinbase_server = CoinbaseMCPServer(
                    api_key=coinbase_api_key,
                    api_secret=coinbase_api_secret
                )
                logger.info("  ✓ Coinbase MCP Server initialized")
                # Create agent if use_agents is enabled
                if self.use_agents:
                    self.coinbase_agent = CoinbaseAgent(
                        api_key=coinbase_api_key,
                        api_secret=coinbase_api_secret,
                        mcp_server=self.coinbase_server,
                        llm=self.llm,
                        enable_chain_of_thought=self.enable_chain_of_thought,
                        checkpointer=self.checkpointer,
                        enable_websocket=True  # Enable WebSocket candle monitoring
                    )
                    logger.info("  ✓ Coinbase Agent initialized (with WebSocket monitoring)")
            except Exception as e:
                logger.warning(f"  ✗ Could not initialize Coinbase: {e}")
        
        # Initialize Schwab - using new env var names with refresh token auth
        schwab_client_id = os.getenv("SCHWAB_CLIENT_ID", "")
        schwab_client_secret = os.getenv("SCHWAB_CLIENT_SECRET", "")
        schwab_refresh_token = os.getenv("SCHWAB_REFRESH_TOKEN", "")
        
        # Check for real credentials (not placeholders)
        has_real_schwab_creds = (
            schwab_client_id and 
            schwab_client_secret and
            schwab_refresh_token and
            "your_" not in schwab_client_id.lower() and
            "your_" not in schwab_client_secret.lower() and
            "your_" not in schwab_refresh_token.lower()
        )
        
        if has_real_schwab_creds:
            try:
                self.schwab_server = SchwabMCPServer(
                    client_id=schwab_client_id,
                    client_secret=schwab_client_secret,
                    refresh_token=schwab_refresh_token
                )
                logger.info("  ✓ Schwab MCP Server initialized")
                # Create agent if use_agents is enabled
                if self.use_agents:
                    self.schwab_agent = SchwabAgent(
                        mcp_server=self.schwab_server,
                        llm=self.llm,
                        enable_chain_of_thought=self.enable_chain_of_thought,
                        checkpointer=self.checkpointer
                    )
                    logger.info("  ✓ Schwab Agent initialized")
            except Exception as e:
                logger.warning(f"  ✗ Could not initialize Schwab: {e}")
        else:
            logger.info("  - Schwab credentials not configured (skipping)")
        
        # Initialize Researcher agent (for investment research)
        finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        if self.use_agents and openai_api_key and finnhub_api_key:
            try:
                self.researcher_agent = ResearcherAgent(
                    llm=self.llm,
                    finnhub_api_key=finnhub_api_key,
                    openai_api_key=openai_api_key,
                    enable_chain_of_thought=self.enable_chain_of_thought,
                    checkpointer=self.checkpointer
                )
                logger.info("  ✓ Researcher Agent initialized")
            except Exception as e:
                logger.warning(f"  ✗ Could not initialize Researcher Agent: {e}")
        elif not finnhub_api_key:
            logger.info("  - FINNHUB_API_KEY not set (Researcher Agent disabled)")
        
        # Initialize FinRL Agent (AI Trading with Deep Reinforcement Learning)
        if self.use_agents and self.llm:
            try:
                # Get WebSocket service from Coinbase agent if available
                websocket_service = None
                if self.coinbase_agent and hasattr(self.coinbase_agent, 'websocket_service'):
                    websocket_service = self.coinbase_agent.websocket_service
                
                self.finrl_service = FinRLService(
                    websocket_service=websocket_service,
                    default_algorithm="PPO"
                )
                self.finrl_agent = FinRLAgent(
                    llm=self.llm,
                    finrl_service=self.finrl_service,
                    websocket_service=websocket_service
                )
                logger.info("  ✓ FinRL Agent initialized (Deep RL trading)")
            except Exception as e:
                logger.warning(f"  ✗ Could not initialize FinRL Agent: {e}")
        
        # Initialize Trading Strategy Service (multi-source analysis for actionable recommendations)
        if self.use_agents:
            try:
                # Get WebSocket service from Coinbase agent if available
                websocket_service = None
                if self.coinbase_agent and hasattr(self.coinbase_agent, 'websocket_service'):
                    websocket_service = self.coinbase_agent.websocket_service
                
                # Get Schwab MCP server for stock quotes
                schwab_mcp_server = None
                if self.schwab_agent and hasattr(self.schwab_agent, 'mcp_server'):
                    schwab_mcp_server = self.schwab_agent.mcp_server
                
                self.trading_strategy_service = TradingStrategyService(
                    websocket_service=websocket_service,
                    finrl_service=self.finrl_service,
                    researcher_agent=self.researcher_agent,
                    schwab_mcp_server=schwab_mcp_server
                )
                logger.info("  ✓ Trading Strategy Service initialized (Intelligent Investor principles)")
                if schwab_mcp_server:
                    logger.info("    - Stock support enabled via Schwab MCP")
            except Exception as e:
                logger.warning(f"  ✗ Could not initialize Trading Strategy Service: {e}")
        
        # Initialize Supervisor Agent (Mister Risker as orchestrator)
        if self.use_agents and self.llm:
            try:
                self.supervisor_agent = SupervisorAgent(
                    llm=self.llm,
                    coinbase_agent=self.coinbase_agent,
                    schwab_agent=self.schwab_agent,
                    researcher_agent=self.researcher_agent,
                    finrl_agent=self.finrl_agent,
                    trading_strategy_service=self.trading_strategy_service,
                    checkpointer=self.checkpointer,
                    enable_chain_of_thought=self.enable_chain_of_thought
                )
                logger.info("  ✓ Supervisor Agent (Mister Risker) initialized")
            except Exception as e:
                logger.warning(f"  ✗ Could not initialize Supervisor Agent: {e}")
        
        logger.info("TradingChatBot initialization complete")
    
    async def process_message(self, user_message: str) -> str:
        """Process a user message and return a response.
        
        Uses the Supervisor Agent pattern where Mister Risker decides which
        sub-agent to delegate to based on the query content.
        
        Args:
            user_message: The user's message
        
        Returns:
            The bot's response
        """
        logger.info(f"Processing message: '{user_message[:100]}...' " if len(user_message) > 100 else f"Processing message: '{user_message}'")
        
        if not self.llm:
            logger.error("LLM not configured")
            return "Error: LLM not configured. Please set OPENAI_API_KEY in your .env file."
        
        # Handle explicit broker switching commands (legacy support)
        lower_message = user_message.lower().strip()
        if "switch to schwab" in lower_message:
            self.active_broker = "schwab"
            self.last_agent_used = "direct"
            self.last_supervisor_logs = ["[SUPERVISOR] Direct broker switch to Schwab"]
            logger.info(">>> BROKER SWITCH: Switched to Schwab")
            # Add to conversation history
            self.conversation_history.append(HumanMessage(content=user_message))
            response = "Switched to **Schwab**! I'll now focus on stock and options trading. How can I help you with your Schwab account?"
            self.conversation_history.append(AIMessage(content=response))
            return response
        
        if "switch to coinbase" in lower_message:
            self.active_broker = "coinbase"
            self.last_agent_used = "direct"
            self.last_supervisor_logs = ["[SUPERVISOR] Direct broker switch to Coinbase"]
            logger.info(">>> BROKER SWITCH: Switched to Coinbase")
            # Add to conversation history
            self.conversation_history.append(HumanMessage(content=user_message))
            response = "Switched to **Coinbase**! I'll now focus on cryptocurrency trading. How can I help you with your Coinbase account?"
            self.conversation_history.append(AIMessage(content=response))
            return response
        
        # Check if this is an image generation request (handle separately)
        if self._is_image_generation_request(user_message):
            logger.info(">>> IMAGE GENERATION request detected")
            try:
                image_result = await self.generate_image(user_message)
                
                if image_result.get("type") == "error":
                    logger.warning(f"Image generation failed: {image_result.get('content')}")
                    # Fall through to normal processing
                else:
                    # Wrap the content for frontend rendering
                    response_text = self._wrap_generated_content(
                        image_result["type"],
                        image_result["content"],
                        image_result.get("description", "")
                    )
                    
                    # Add to conversation history
                    self.conversation_history.append(HumanMessage(content=user_message))
                    self.conversation_history.append(AIMessage(content=response_text))
                    
                    logger.info(f"<<< Image generation completed: type={image_result['type']}")
                    return response_text
            except Exception as e:
                logger.error(f"Image generation error: {e}", exc_info=True)
                # Fall through to normal processing
        
        # === SUPERVISOR AGENT PATTERN ===
        # Mister Risker (supervisor) decides which sub-agent to use
        if self.use_agents and self.supervisor_agent:
            logger.info(">>> SUPERVISOR AGENT routing query")
            try:
                # Format conversation history for the supervisor
                history = self._format_history_for_agent()
                config = self._get_agent_config()
                
                # Execute supervisor workflow - it will route to appropriate sub-agent
                result = await self.supervisor_agent.execute(
                    user_message=user_message,
                    conversation_history=history,
                    config=config
                )
                
                response_text = result.get("response", "")
                agent_used = result.get("agent_used", "unknown")
                reasoning = result.get("reasoning", "")
                logs = result.get("logs", [])
                
                # Store logs for frontend debugging
                self.last_supervisor_logs = logs
                self.last_agent_used = agent_used
                
                # Log the delegation details
                logger.info(f"[SUPERVISOR] Delegated to: {agent_used}")
                logger.info(f"[SUPERVISOR] Reasoning: {reasoning}")
                for log_entry in logs:
                    logger.info(log_entry)
                
                # Add to conversation history
                self.conversation_history.append(HumanMessage(content=user_message))
                self.conversation_history.append(AIMessage(content=response_text))
                
                logger.info(f"<<< SUPERVISOR completed via {agent_used}")
                return response_text
                
            except SupervisorAgentError as e:
                logger.error(f"Supervisor agent error: {e}")
                # Fall through to legacy handling
            except Exception as e:
                logger.error(f"Supervisor error: {e}", exc_info=True)
                # Fall through to legacy handling
        
        # === LEGACY FALLBACK (when supervisor not available) ===
        logger.info(f"Using legacy flow (supervisor not available)")
        
        # Early check for broker availability on trading-related queries
        trading_keywords = ["balance", "buy", "sell", "order", "position", "account", "portfolio", "trade"]
        if any(kw in lower_message for kw in trading_keywords):
            if self.active_broker == "coinbase":
                if not self.coinbase_server and not self.coinbase_agent:
                    other_broker_msg = " (Schwab is available - say 'switch to schwab' to use it)" if self.schwab_server or self.schwab_agent else ""
                    return f"Error: Coinbase not configured. Please set your Coinbase API credentials in the .env file.{other_broker_msg}"
            elif self.active_broker == "schwab":
                if not self.schwab_server and not self.schwab_agent:
                    other_broker_msg = " (Coinbase is available - say 'switch to coinbase' to use it)" if self.coinbase_server or self.coinbase_agent else ""
                    return f"Error: Schwab not configured. Please set your Schwab API credentials in the .env file.{other_broker_msg}"
        
        # Check for research-related queries and delegate to researcher agent
        research_keywords = ["news", "research", "analyze", "analysis", "earnings", "financials", "recommend", "recommendation"]
        if self.researcher_agent and any(kw in lower_message for kw in research_keywords):
            logger.info(">>> LEGACY: Delegating to Researcher Agent")
            try:
                # Add to conversation history
                self.conversation_history.append(HumanMessage(content=user_message))
                
                # Execute research
                result = await self.researcher_agent.run(
                    query=user_message,
                    messages=self._format_history_for_agent(),
                    config=self._get_agent_config()
                )
                
                if result.get("status") == "success":
                    response_text = result.get("response", "Research completed.")
                    self.conversation_history.append(AIMessage(content=response_text))
                    self.last_agent_used = "researcher"
                    self.last_supervisor_logs = [
                        f"[LEGACY] Detected research query",
                        f"[LEGACY] Delegated to Researcher Agent",
                        f"[LEGACY] Research completed successfully"
                    ]
                    logger.info("<<< LEGACY: Research delegation completed")
                    return response_text
            except Exception as e:
                logger.error(f"Research delegation error: {e}", exc_info=True)
                # Fall through to normal processing
        
        # Add user message to history
        self.conversation_history.append(HumanMessage(content=user_message))
        
        # Build messages for LLM with current broker context
        system_prompt = self.system_prompt.format(broker=self.active_broker.upper())
        messages = [
            SystemMessage(content=system_prompt),
            *self.conversation_history
        ]
        
        try:
            # Get LLM response
            response = await self.llm.ainvoke(messages)
            response_text = self._extract_content(response.content)
            
            # Check if LLM wants to call a tool
            tool_call = self._extract_tool_call(response_text)
            
            if tool_call:
                # Execute the tool on the appropriate broker
                tool_result = await self._execute_tool(
                    tool_call["tool"],
                    tool_call.get("params", {})
                )
                
                # Get LLM to interpret the result
                self.conversation_history.append(AIMessage(content=f"Tool result: {json.dumps(tool_result, indent=2)}"))
                
                interpret_messages = [
                    SystemMessage(content=f"""You are Mister Risker, a helpful trading assistant. 
Interpret the following tool result and explain it to the user in a friendly, clear way.
Current broker: {self.active_broker.upper()}
- Format currency amounts with $ signs and appropriate decimal places
- Format crypto amounts with appropriate precision
- List each account/balance/position on its own line
- Highlight important information
- Be concise but informative
- Don't mention JSON or technical details"""),
                    HumanMessage(content=f"The user asked: {user_message}\n\nTool called: {tool_call['tool']}\n\nResult: {json.dumps(tool_result, indent=2)}")
                ]
                
                interpretation = await self.llm.ainvoke(interpret_messages)
                response_text = self._extract_content(interpretation.content)
            
            # Add response to history
            self.conversation_history.append(AIMessage(content=response_text))
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            self.conversation_history.append(AIMessage(content=error_msg))
            return error_msg
    
    async def process_message_with_image(self, user_message: str, image_data: str) -> str:
        """Process a user message with an attached image using vision capabilities.
        
        Args:
            user_message: The user's text message
            image_data: Base64-encoded image data URL (data:image/png;base64,...)
        
        Returns:
            The bot's response
        """
        logger.info(f"Processing message with image: '{user_message[:50]}...' " if len(user_message) > 50 else f"Processing message with image: '{user_message}'")
        
        if not self.llm:
            logger.error("LLM not configured")
            return "Error: LLM not configured. Please set OPENAI_API_KEY in your .env file."
        
        try:
            # Use OpenAI's vision model directly for image analysis
            from openai import OpenAI
            
            client = OpenAI()
            
            # Build the message with image
            messages = [
                {
                    "role": "system",
                    "content": f"""You are Mister Risker, a helpful trading assistant specializing in cryptocurrency and stock trading.
You can analyze images including charts, screenshots of trading platforms, financial documents, and more.
Provide helpful insights about any trading-related images the user shares.
Current active broker: {self.active_broker.upper()}"""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message or "What can you tell me about this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        }
                    ]
                }
            ]
            
            # Use gpt-4o for vision capabilities
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # Add to conversation history (text only for history)
            self.conversation_history.append(HumanMessage(content=f"{user_message} [with image]"))
            self.conversation_history.append(AIMessage(content=response_text))
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
    
    async def _execute_tool(self, tool_name: str, params: dict) -> dict:
        """Execute a tool on the active broker.
        
        Uses LangGraph agents if use_agents is enabled and agent is available,
        otherwise falls back to direct MCP server calls.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
        
        Returns:
            Tool execution result
        """
        try:
            if self.active_broker == "coinbase":
                # Try agent first if enabled
                if self.use_agents and self.coinbase_agent:
                    result = await self.coinbase_agent.execute_tool(tool_name, params)
                elif self.coinbase_server:
                    result = await self.coinbase_server.call_tool(tool_name, params)
                else:
                    return {"error": "Coinbase not configured"}
            else:  # schwab
                # Try agent first if enabled
                if self.use_agents and self.schwab_agent:
                    result = await self.schwab_agent.execute_tool(tool_name, params)
                elif self.schwab_server:
                    result = await self.schwab_server.call_tool(tool_name, params)
                else:
                    return {"error": "Schwab not configured"}
            return result
        except (CoinbaseAPIError, SchwabAPIError, CoinbaseAgentError, SchwabAgentError) as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    async def _execute_research(self, query: str) -> dict:
        """Execute a research query using the Researcher agent.
        
        Args:
            query: Research query from the user
        
        Returns:
            Research result with response and optional reasoning steps
        """
        logger.info(f"  Executing research for query: '{query}'")
        
        if not self.researcher_agent:
            logger.error("  Researcher agent not configured")
            return {
                "error": "Researcher agent not configured. Set FINNHUB_API_KEY in .env file."
            }
        
        try:
            logger.info("  Calling researcher_agent.run()...")
            # Pass conversation history for context
            messages = self._format_history_for_agent()
            config = self._get_agent_config()
            
            result = await self.researcher_agent.run(
                query=query,
                messages=messages,
                config=config,
                return_structured=True
            )
            logger.info(f"  Research completed. Status: {result.get('status')}, Response length: {len(result.get('response', ''))}")
            return result
        except ResearcherAgentError as e:
            logger.error(f"  ResearcherAgentError: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"  Research failed with exception: {e}", exc_info=True)
            return {"error": f"Research failed: {str(e)}"}
    
    def clear_history(self):
        """Clear the conversation history, reset broker, and generate new thread_id."""
        self.conversation_history = []
        self.active_broker = "coinbase"
        # Generate new thread_id for fresh memory context
        self.thread_id = str(uuid.uuid4())
    
    # Alias for backward compatibility
    def clear_conversation(self):
        """Alias for clear_history."""
        self.clear_history()
    
    def _get_agent_config(self) -> dict:
        """Get configuration dict for agent invocations with thread_id.
        
        Returns:
            Config dict with thread_id for checkpointer
        """
        return {
            "configurable": {
                "thread_id": self.thread_id
            }
        }
    
    def _format_history_for_agent(self) -> list[dict]:
        """Format conversation history for agent state.
        
        Converts LangChain message objects to dicts with role/content.
        
        Returns:
            List of message dicts
        """
        formatted = []
        for msg in self.conversation_history:
            if isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            else:
                # Fallback for other message types
                formatted.append({"role": "user", "content": str(msg.content)})
        return formatted
    
    def _extract_content(self, content) -> str:
        """Extract text content from LLM response.
        
        The OpenAI Responses API can return content as either a string
        or a list of content blocks. This method handles both formats.
        
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
    
    def _extract_tool_call(self, text: str) -> dict | None:
        """Extract a tool call JSON from text that may contain other content.
        
        The LLM might respond with text like:
        "Let me check that for you. {"tool": "get_accounts", "params": {}}"
        
        This method finds and extracts the JSON tool call.
        
        Args:
            text: Response text that may contain a JSON tool call
        
        Returns:
            Extracted tool call dict, or None if no valid tool call found
        """
        import re
        
        # Look for JSON object pattern in the text
        # Find content between { and } that contains "tool"
        json_pattern = r'\{[^{}]*"tool"[^{}]*\}'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if "tool" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # Also try to find nested JSON (with params object)
        # This handles: {"tool": "x", "params": {"a": "b"}}
        try:
            # Find the start of a JSON object
            start = text.find('{"tool"')
            if start == -1:
                start = text.find("{'tool")
            
            if start != -1:
                # Try to parse from this position
                bracket_count = 0
                end = start
                for i, char in enumerate(text[start:]):
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end = start + i + 1
                            break
                
                json_str = text[start:end]
                parsed = json.loads(json_str)
                if "tool" in parsed:
                    return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
    
    def _is_research_query(self, message: str) -> bool:
        """Detect if a message is a research query that should use the researcher agent.
        
        Args:
            message: User message
        
        Returns:
            True if this is a research query
        """
        lower_msg = message.lower()
        
        # Research indicators
        research_keywords = [
            "research", "analyze", "analysis", "what do you think",
            "should i buy", "should i sell", "should i invest",
            "news", "latest", "what happened", "tell me about",
            "compare", "vs", "versus", "which is better",
            "risk", "outlook", "forecast", "prediction",
            "earnings", "financials", "pe ratio", "market cap",
            "recommendation", "analyst", "rating"
        ]
        
        matched = [kw for kw in research_keywords if kw in lower_msg]
        is_research = len(matched) > 0
        logger.debug(f"  _is_research_query: matched_keywords={matched}, is_research={is_research}")
        
        return is_research
    
    def _is_image_generation_request(self, message: str) -> bool:
        """Detect if a message is requesting image/visual content generation.
        
        Args:
            message: User message
        
        Returns:
            True if this is an image generation request
        """
        lower_msg = message.lower()
        
        # Image generation indicators
        image_keywords = [
            "draw", "sketch", "create an image", "create a picture",
            "generate an svg", "generate svg", "make an svg", "make svg",
            "create an animation", "make an animation", "animate",
            "visualization", "visualize", "create a graphic", "make a graphic",
            "show me a graphic", "show me a picture", "show me an image",
            "design", "illustrate", "render"
        ]
        
        # Check for keyword matches
        for keyword in image_keywords:
            if keyword in lower_msg:
                return True
        
        return False
    
    async def generate_image(self, prompt: str) -> dict:
        """Generate an image (SVG or animation) based on the user's prompt.
        
        Args:
            prompt: User's request for image generation
        
        Returns:
            Dict with type, content, and optional description
        """
        logger.info(f"Generating image for prompt: '{prompt[:50]}...' " if len(prompt) > 50 else f"Generating image for prompt: '{prompt}'")
        
        if not self.llm:
            return {"type": "error", "content": "LLM not configured", "description": ""}
        
        try:
            # Determine if user wants animation or static SVG
            lower_prompt = prompt.lower()
            is_animation = any(kw in lower_prompt for kw in ["animation", "animate", "moving", "bouncing", "spinning"])
            
            if is_animation:
                system_prompt = """You are a creative assistant that generates interactive HTML/JavaScript animations.
When asked to create an animation, respond with ONLY the HTML code (including inline CSS and JavaScript).
The animation should be self-contained and work in an iframe.
Use canvas or CSS animations. Keep the code compact but functional.
Do NOT include any explanation text - ONLY output the HTML code.
Start directly with <!DOCTYPE html> or <html> or <div>."""
            else:
                system_prompt = """You are a creative assistant that generates SVG images.
When asked to draw or create an image, respond with ONLY the SVG code.
Create colorful, visually appealing SVGs.
Do NOT include any explanation text - ONLY output the SVG code.
Start directly with <svg and end with </svg>."""
            
            from langchain_core.messages import SystemMessage, HumanMessage as LCHumanMessage
            
            messages = [
                SystemMessage(content=system_prompt),
                LCHumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            content = self._extract_content(response.content)
            
            # Determine the type based on content
            if "<svg" in content.lower():
                content_type = "svg"
                # Clean up - extract just the SVG if there's extra text
                svg_start = content.lower().find("<svg")
                svg_end = content.lower().rfind("</svg>") + 6
                if svg_start != -1 and svg_end > svg_start:
                    content = content[svg_start:svg_end]
            elif "<script>" in content.lower() or "<canvas" in content.lower() or "requestanimationframe" in content.lower():
                content_type = "animation"
            elif "<html" in content.lower() or "<!doctype" in content.lower():
                content_type = "animation"
            else:
                # LLM didn't generate proper image content
                content_type = "text"
            
            return {
                "type": content_type,
                "content": content,
                "description": prompt
            }
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)
            return {"type": "error", "content": str(e), "description": prompt}
    
    def _wrap_generated_content(self, content_type: str, content: str, description: str) -> str:
        """Wrap generated content with markers for frontend detection.
        
        Args:
            content_type: Type of content (svg, animation, html)
            content: The raw generated content
            description: Description of what was generated
        
        Returns:
            Wrapped content string
        """
        if content_type == "svg":
            return f"""<!--GENERATED_IMAGE:svg-->
Here's the image you requested:

```svg
{content}
```

{description}"""
        elif content_type == "animation":
            return f"""<!--GENERATED_IMAGE:animation-->
Here's the animation you requested:

```html
{content}
```

{description}"""
        else:
            return content


# Global chatbot instance - agents enabled with Chain of Thought
chatbot = TradingChatBot(use_agents=True, enable_chain_of_thought=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    await chatbot.initialize()
    
    # Start WebSocket monitoring if Coinbase agent is available
    if chatbot.coinbase_agent and hasattr(chatbot.coinbase_agent, 'ensure_websocket_started'):
        await chatbot.coinbase_agent.ensure_websocket_started()
        logger.info("  ✓ Coinbase WebSocket candle monitoring started")
    
    yield
    
    # Cleanup: Stop WebSocket monitoring
    if chatbot.coinbase_agent and hasattr(chatbot.coinbase_agent, 'websocket_service'):
        ws_service = chatbot.coinbase_agent.websocket_service
        if ws_service and ws_service.is_running:
            await ws_service.stop()
            logger.info("  ✓ Coinbase WebSocket monitoring stopped")


# Create FastAPI app
app = FastAPI(
    title="Mister Risker",
    description="A multi-broker trading assistant for Coinbase and Schwab",
    lifespan=lifespan
)

# Serve static files (favicon, etc.)
import pathlib
static_dir = pathlib.Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# HTML template for the chat interface
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mister Risker</title>
    <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
    <link rel="shortcut icon" type="image/svg+xml" href="/static/favicon.svg">
    <!-- Markdown parser -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- html2canvas for exporting chat to image -->
    <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
    <!-- Syntax highlighting for code blocks -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github-dark.min.css">
    <script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/core.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/languages/python.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/languages/javascript.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/languages/json.min.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            width: 95%;
            max-width: 1400px;
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 20px 24px;
            display: flex;
            align-items: center;
            gap: 12px;
            position: relative;
            z-index: 100;
        }
        
        .header-logo {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #00d4aa 0%, #0052ff 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .header h1 {
            font-size: 1.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, #00d4aa 0%, #0052ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header-right {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .chat-container {
            height: 550px;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 20px;
            background: #f8f9fa;
            position: relative;
            z-index: 1;
        }
        
        .message {
            margin-bottom: 16px;
            display: flex;
            flex-direction: column;
        }
        
        .message.user {
            align-items: flex-end;
        }
        
        .message.assistant {
            align-items: flex-start;
        }
        
        .message-wrapper {
            max-width: 85%;
            position: relative;
            display: flex;
            align-items: flex-start;
            gap: 8px;
        }
        
        .message.user .message-wrapper {
            flex-direction: row-reverse;
        }
        
        .message-content {
            padding: 14px 18px;
            border-radius: 16px;
            line-height: 1.6;
            flex: 1;
        }
        
        /* Copy button styling - square icon outside the message */
        .copy-btn {
            flex-shrink: 0;
            width: 32px;
            height: 32px;
            background: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            transition: background 0.2s, border-color 0.2s, color 0.2s;
            margin-top: 4px;
        }
        
        .copy-btn:hover {
            background: #e8e8e8;
            border-color: #ccc;
            color: #333;
        }
        
        .copy-btn.copied {
            background: #00d4aa;
            color: white;
            border-color: #00d4aa;
        }

        .message.user .message-content {
            background: linear-gradient(135deg, #00d4aa 0%, #0052ff 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }        .message.assistant .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* Markdown content styling */
        .message-content h1, .message-content h2, .message-content h3 {
            margin: 12px 0 8px 0;
            color: #1a1a2e;
        }
        .message-content h1 { font-size: 1.4rem; }
        .message-content h2 { font-size: 1.2rem; }
        .message-content h3 { font-size: 1.1rem; }
        
        .message-content p {
            margin: 8px 0;
        }
        
        .message-content ul, .message-content ol {
            margin: 8px 0;
            padding-left: 24px;
        }
        
        .message-content li {
            margin: 4px 0;
        }
        
        .message-content strong {
            color: #0052ff;
            font-weight: 600;
        }
        
        .message.user .message-content strong {
            color: #fff;
        }
        
        .message-content em {
            font-style: italic;
            color: #666;
        }
        
        .message-content code {
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
            color: #d63384;
        }
        
        .message.user .message-content code {
            background: rgba(255,255,255,0.2);
            color: #fff;
        }
        
        .message-content pre {
            background: #1e1e1e;
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
            overflow-x: auto;
        }
        
        .message-content pre code {
            background: none;
            padding: 0;
            color: #d4d4d4;
            font-size: 0.85em;
        }
        
        .message-content blockquote {
            border-left: 4px solid #00d4aa;
            padding-left: 16px;
            margin: 12px 0;
            color: #555;
            font-style: italic;
        }
        
        .message-content table {
            border-collapse: collapse;
            width: 100%;
            margin: 12px 0;
            font-size: 0.9em;
        }
        
        .message-content th, .message-content td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        
        .message-content th {
            background: #f5f5f5;
            font-weight: 600;
        }
        
        .message-content a {
            color: #0052ff;
            text-decoration: none;
        }
        
        .message-content a:hover {
            text-decoration: underline;
        }
        
        .message-content hr {
            border: none;
            border-top: 1px solid #e0e0e0;
            margin: 16px 0;
        }
        
        /* Research badge */
        .research-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        /* Generated image/animation badge */
        .generated-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        /* Generated image containers */
        .generated-image {
            margin: 12px 0;
            border-radius: 12px;
            overflow: hidden;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
        }
        
        .generated-image.svg-container {
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            background: white;
        }
        
        .generated-image.svg-container svg {
            max-width: 100%;
            height: auto;
        }
        
        .generated-image.animation-container {
            min-height: 200px;
        }
        
        .generated-image.animation-container iframe {
            width: 100%;
            min-height: 250px;
            border: none;
            background: white;
        }
        
        .message-label {
            font-size: 0.75rem;
            color: #666;
            margin-bottom: 4px;
            padding: 0 8px;
        }
        
        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e9ecef;
            display: flex;
            gap: 12px;
        }
        
        .input-container input {
            flex: 1;
            padding: 14px 18px;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
        }
        
        .input-container input:focus {
            border-color: #00d4aa;
        }
        
        .input-container button {
            padding: 14px 28px;
            background: linear-gradient(135deg, #00d4aa 0%, #0052ff 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        
        .input-container button:hover {
            opacity: 0.9;
        }
        
        .input-container button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .typing-indicator {
            display: none;
            padding: 12px 16px;
            background: white;
            border-radius: 16px;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            max-width: 80px;
        }
        
        .typing-indicator.show {
            display: block;
        }
        
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #00d4aa;
            border-radius: 50%;
            margin-right: 4px;
            animation: bounce 1.4s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(1) { animation-delay: 0s; }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-6px); }
        }
        
        .suggestions {
            padding: 12px 20px;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .suggestion {
            padding: 8px 14px;
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 20px;
            font-size: 0.875rem;
            color: #666;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .suggestion:hover {
            background: linear-gradient(135deg, #00d4aa 0%, #0052ff 100%);
            color: white;
            border-color: transparent;
        }
        
        .clear-btn {
            padding: 6px 12px;
            background: rgba(255,255,255,0.2);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 0.75rem;
            cursor: pointer;
        }
        
        .clear-btn:hover {
            background: rgba(255,255,255,0.3);
        }
        
        .export-btn {
            padding: 6px 12px;
            background: rgba(255,255,255,0.2);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 0.75rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        .export-btn:hover {
            background: rgba(255,255,255,0.3);
        }
        
        .export-btn.copying {
            background: #00d4aa;
        }
        
        .export-btn svg {
            width: 14px;
            height: 14px;
        }
        
        /* Image preview styling */
        .image-preview-container {
            display: none;
            padding: 10px 20px;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
        }
        
        .image-preview-container.has-image {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .image-preview {
            max-width: 120px;
            max-height: 80px;
            border-radius: 8px;
            border: 2px solid #e9ecef;
            object-fit: cover;
        }
        
        .image-preview-info {
            flex: 1;
            font-size: 0.875rem;
            color: #666;
        }
        
        .image-preview-info .filename {
            font-weight: 600;
            color: #333;
        }
        
        .remove-image-btn {
            width: 28px;
            height: 28px;
            background: #ff4757;
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }
        
        .remove-image-btn:hover {
            background: #ff3344;
        }
        
        .input-container input::placeholder {
            color: #999;
        }
        
        .message-image {
            display: block;
            max-width: 300px;
            max-height: 200px;
            border-radius: 8px;
            margin-top: 10px;
            cursor: pointer;
        }
        
        .message-image:hover {
            opacity: 0.9;
        }
        
        /* Ensure text and image stack properly in user messages */
        .message.user .message-content {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }
        
        .message-text {
            margin-bottom: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-logo">M</div>
            <h1>Mister Risker</h1>
            <div class="header-right">
                <button class="export-btn" onclick="exportChatToClipboard()" title="Copy conversation as image">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                    Export
                </button>
                <button class="clear-btn" onclick="clearChat()">Clear</button>
            </div>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message assistant">
                <span class="message-label">Mister Risker</span>
                <div class="message-content" id="welcomeMessage"></div>
            </div>
        </div>
        
        <div class="suggestions">
            <span class="suggestion" onclick="sendSuggestion('What are my account balances?')">💰 Balances</span>
            <span class="suggestion" onclick="sendSuggestion('What is the current price of Bitcoin?')">₿ BTC Price</span>
            <span class="suggestion" onclick="sendSuggestion('Show my portfolio')">💼 Portfolio</span>
            <span class="suggestion" onclick="sendSuggestion('Tell me the latest market news')">📰 Latest News</span>
            <span class="suggestion" onclick="sendSuggestion('What do you think about AAPL?')">🔍 Research AAPL</span>
        </div>
        
        <div class="image-preview-container" id="imagePreviewContainer">
            <img id="imagePreview" class="image-preview" src="" alt="Preview">
            <div class="image-preview-info">
                <div class="filename" id="imageFilename">image.png</div>
                <div>Ready to send with your message</div>
            </div>
            <button class="remove-image-btn" onclick="removeImage()" title="Remove image">×</button>
        </div>
        
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Ask me anything about trading... (paste image with Ctrl+V)" onkeypress="handleKeyPress(event)">
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>
    
    <script>
        // Configure marked.js
        marked.setOptions({
            breaks: true,
            gfm: true,
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (e) {}
                }
                return code;
            }
        });
        
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');

        const imagePreviewContainer = document.getElementById('imagePreviewContainer');
        const imagePreview = document.getElementById('imagePreview');
        const imageFilename = document.getElementById('imageFilename');
        
        // Store the current image data
        let currentImageData = null;
        
        // Handle paste events for images
        document.addEventListener('paste', function(e) {
            const items = e.clipboardData?.items;
            if (!items) return;
            
            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf('image') !== -1) {
                    e.preventDefault();
                    const blob = items[i].getAsFile();
                    const reader = new FileReader();
                    
                    reader.onload = function(event) {
                        currentImageData = event.target.result;
                        imagePreview.src = currentImageData;
                        imageFilename.textContent = blob.name || 'Pasted image';
                        imagePreviewContainer.classList.add('has-image');
                        messageInput.focus();
                    };
                    
                    reader.readAsDataURL(blob);
                    break;
                }
            }
        });
        
        function removeImage() {
            currentImageData = null;
            imagePreview.src = '';
            imagePreviewContainer.classList.remove('has-image');
            messageInput.focus();
        }
        
        // Welcome message content
        const welcomeMarkdown = `👋 **Mister Risker** - Your AI Trading Assistant

🪙 **Crypto** • 📈 **Stocks & Options** • 🔍 **Research** • 🎨 **Charts**

Ask me anything about trading!`;

        // Render welcome message
        document.getElementById('welcomeMessage').innerHTML = marked.parse(welcomeMarkdown);
        
        function renderMarkdown(text) {
            // Check if it's a research response
            const isResearch = text.startsWith('🔍');
            
            // Check if it contains generated image content
            const hasSvg = text.includes('<!--GENERATED_IMAGE:svg-->') || text.includes('```svg');
            const hasAnimation = text.includes('<!--GENERATED_IMAGE:animation-->') || (text.includes('```html') && (text.includes('<script>') || text.includes('<canvas')));
            
            // Parse markdown
            let html = marked.parse(text);
            
            // Handle SVG content - render inline
            if (hasSvg) {
                // Extract SVG from code block and render it directly
                html = html.replace(/<pre><code class="language-svg">([\s\S]*?)<\/code><\/pre>/g, function(match, svgCode) {
                    // Decode HTML entities
                    const textarea = document.createElement('textarea');
                    textarea.innerHTML = svgCode;
                    const decodedSvg = textarea.value;
                    return '<div class="generated-image svg-container">' + decodedSvg + '</div>';
                });
                // Remove the comment marker
                html = html.replace(/<!--GENERATED_IMAGE:svg-->/g, '<div class="generated-badge">🎨 Generated Image</div>');
            }
            
            // Handle animation content - render in iframe
            if (hasAnimation) {
                html = html.replace(/<pre><code class="language-html">([\s\S]*?)<\/code><\/pre>/g, function(match, htmlCode) {
                    // Decode HTML entities
                    const textarea = document.createElement('textarea');
                    textarea.innerHTML = htmlCode;
                    const decodedHtml = textarea.value;
                    // Create a data URL for the iframe
                    const dataUrl = 'data:text/html;charset=utf-8,' + encodeURIComponent(decodedHtml);
                    return '<div class="generated-image animation-container"><iframe src="' + dataUrl + '" sandbox="allow-scripts" frameborder="0"></iframe></div>';
                });
                // Remove the comment marker
                html = html.replace(/<!--GENERATED_IMAGE:animation-->/g, '<div class="generated-badge">🎬 Generated Animation</div>');
            }
            
            // Add research badge if applicable
            if (isResearch) {
                html = '<div class="research-badge">🔍 Research Analysis</div>' + html.replace('🔍 **Research Analysis:**', '').replace('🔍', '');
            }
            
            return html;
        }
        
        // SVG icons for copy button
        const copyIcon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
        const checkIcon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
        
        function copyToClipboard(text, button) {
            // Try modern clipboard API first, fallback to execCommand
            function showSuccess() {
                const originalHTML = button.innerHTML;
                button.innerHTML = checkIcon;
                button.classList.add('copied');
                button.title = 'Copied!';
                setTimeout(() => {
                    button.innerHTML = originalHTML;
                    button.classList.remove('copied');
                    button.title = 'Copy to clipboard';
                }, 2000);
            }
            
            function fallbackCopy(text) {
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.left = '-9999px';
                textArea.style.top = '-9999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                try {
                    document.execCommand('copy');
                    showSuccess();
                } catch (err) {
                    console.error('Fallback copy failed:', err);
                }
                document.body.removeChild(textArea);
            }
            
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(() => {
                    showSuccess();
                }).catch(err => {
                    console.warn('Clipboard API failed, using fallback:', err);
                    fallbackCopy(text);
                });
            } else {
                fallbackCopy(text);
            }
        }
        
        function addMessage(content, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
            
            const label = document.createElement('span');
            label.className = 'message-label';
            label.textContent = isUser ? 'You' : 'Mister Risker';
            
            // Wrapper for content and copy button
            const wrapperDiv = document.createElement('div');
            wrapperDiv.className = 'message-wrapper';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            // Create copy button with square icon
            const copyBtn = document.createElement('button');
            copyBtn.className = 'copy-btn';
            copyBtn.innerHTML = copyIcon;
            copyBtn.title = 'Copy to clipboard';
            copyBtn.onclick = () => copyToClipboard(content, copyBtn);
            
            if (isUser) {
                // User messages as plain text - wrap in span for proper layout with images
                const textSpan = document.createElement('span');
                textSpan.className = 'message-text';
                textSpan.textContent = content;
                contentDiv.appendChild(textSpan);
            } else {
                // Bot messages rendered as Markdown
                contentDiv.innerHTML = renderMarkdown(content);
                // Highlight code blocks
                contentDiv.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            }
            
            // Add image if provided
            if (arguments[2]) {
                const img = document.createElement('img');
                img.src = arguments[2];
                img.className = 'message-image';
                img.onclick = () => window.open(arguments[2], '_blank');
                contentDiv.appendChild(img);
            }
            
            wrapperDiv.appendChild(contentDiv);
            wrapperDiv.appendChild(copyBtn);
            messageDiv.appendChild(label);
            messageDiv.appendChild(wrapperDiv);
            chatContainer.appendChild(messageDiv);
            
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function showTyping() {
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message assistant';
            typingDiv.id = 'typingIndicator';
            
            const indicator = document.createElement('div');
            indicator.className = 'typing-indicator show';
            indicator.innerHTML = '<span></span><span></span><span></span>';
            
            typingDiv.appendChild(indicator);
            chatContainer.appendChild(typingDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function hideTyping() {
            const typing = document.getElementById('typingIndicator');
            if (typing) typing.remove();
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message && !currentImageData) return;
            
            // Log to console for debugging
            console.log('[MISTER RISKER] User message:', message);
            console.log('[MISTER RISKER] Sending to supervisor agent for routing...');
            
            // Show user message with image if present
            addMessage(message || '[Image]', true, currentImageData);
            
            const imageToSend = currentImageData;
            messageInput.value = '';
            removeImage();
            sendBtn.disabled = true;
            showTyping();
            
            try {
                const payload = { message: message || 'What is in this image?' };
                if (imageToSend) {
                    payload.image = imageToSend;
                }
                
                console.log('[MISTER RISKER] Sending request to /chat endpoint...');
                const startTime = performance.now();
                
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                const elapsed = (performance.now() - startTime).toFixed(0);
                
                console.log('[MISTER RISKER] Response received in', elapsed, 'ms');
                console.log('[MISTER RISKER] Agent used:', data.agent_used || 'unknown');
                console.log('[MISTER RISKER] Response preview:', data.response?.substring(0, 100) + '...');
                
                // Log supervisor delegation info if available
                if (data.logs && Array.isArray(data.logs)) {
                    console.group('[MISTER RISKER] Supervisor Logs:');
                    data.logs.forEach(log => console.log(log));
                    console.groupEnd();
                }
                
                hideTyping();
                addMessage(data.response, false);
            } catch (error) {
                console.error('[MISTER RISKER] Error:', error);
                hideTyping();
                addMessage('Sorry, there was an error processing your request.', false);
            }
            
            sendBtn.disabled = false;
            messageInput.focus();
        }
        
        function sendSuggestion(text) {
            messageInput.value = text;
            sendMessage();
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        async function clearChat() {
            try {
                await fetch('/clear', { method: 'POST' });
                chatContainer.innerHTML = `
                    <div class="message assistant">
                        <span class="message-label">Mister Risker</span>
                        <div class="message-content" id="welcomeMessage"></div>
                    </div>
                `;
                document.getElementById('welcomeMessage').innerHTML = marked.parse(welcomeMarkdown);
            } catch (error) {
                console.error('Error clearing chat:', error);
            }
        }
        
        async function exportChatToClipboard() {
            const exportBtn = document.querySelector('.export-btn');
            const originalHTML = exportBtn.innerHTML;
            
            try {
                // Show loading state
                exportBtn.innerHTML = '⏳ Capturing...';
                exportBtn.classList.add('copying');
                
                // Store current scroll position and height
                const originalScrollTop = chatContainer.scrollTop;
                const originalHeight = chatContainer.style.height;
                const originalOverflow = chatContainer.style.overflow;
                const originalMaxHeight = chatContainer.style.maxHeight;
                
                // Temporarily expand the container to show all content
                chatContainer.style.height = 'auto';
                chatContainer.style.maxHeight = 'none';
                chatContainer.style.overflow = 'visible';
                
                // Wait for layout to update
                await new Promise(resolve => setTimeout(resolve, 100));
                
                // Capture the chat container
                const canvas = await html2canvas(chatContainer, {
                    backgroundColor: '#f8f9fa',
                    scale: 2, // Higher quality
                    useCORS: true,
                    allowTaint: true,
                    logging: false
                });
                
                // Restore original dimensions
                chatContainer.style.height = originalHeight;
                chatContainer.style.maxHeight = originalMaxHeight;
                chatContainer.style.overflow = originalOverflow;
                chatContainer.scrollTop = originalScrollTop;
                
                // Save the image to /tmp via API and open in new window
                const dataUrl = canvas.toDataURL('image/png');
                
                // Send to server to save as file
                const saveResponse = await fetch('/save-export', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl })
                });
                const saveResult = await saveResponse.json();
                
                if (saveResult.error) {
                    throw new Error(saveResult.error);
                }
                
                // Open the saved image in a new window
                const imageUrl = '/export/' + saveResult.filename;
                const newWindow = window.open('', '_blank');
                if (newWindow) {
                    newWindow.document.write(`
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>Mister Risker Chat Export</title>
                            <style>
                                body { 
                                    margin: 0; 
                                    padding: 20px; 
                                    background: #333; 
                                    display: flex; 
                                    justify-content: center;
                                    font-family: Arial, sans-serif;
                                }
                                img { 
                                    max-width: 100%; 
                                    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                                    border-radius: 8px;
                                }
                                .instructions {
                                    position: fixed;
                                    top: 10px;
                                    left: 50%;
                                    transform: translateX(-50%);
                                    background: rgba(0,0,0,0.8);
                                    color: white;
                                    padding: 10px 20px;
                                    border-radius: 20px;
                                    font-size: 14px;
                                }
                            </style>
                        </head>
                        <body>
                            <div class="instructions">Right-click the image and select "Save image as..." or "Copy image"<br>File saved to: ${saveResult.filepath}</div>
                            <img src="${imageUrl}" alt="Mister Risker Chat Export">
                        </body>
                        </html>
                    `);
                    newWindow.document.close();
                    exportBtn.innerHTML = '✓ Opened!';
                } else {
                    // Fallback: download the file
                    const link = document.createElement('a');
                    link.download = 'mister-risker-chat-' + new Date().toISOString().slice(0,10) + '.png';
                    link.href = dataUrl;
                    link.click();
                    exportBtn.innerHTML = '⬇️ Downloaded!';
                }
                
                setTimeout(() => {
                    exportBtn.innerHTML = originalHTML;
                    exportBtn.classList.remove('copying');
                }, 2000);
                
            } catch (error) {
                console.error('Export failed:', error);
                exportBtn.innerHTML = '❌ Failed';
                setTimeout(() => {
                    exportBtn.innerHTML = originalHTML;
                    exportBtn.classList.remove('copying');
                }, 2000);
            }
        }
        
        messageInput.focus();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    """Serve the chat interface."""
    return HTML_TEMPLATE


@app.post("/chat")
async def chat(request: Request):
    """Process a chat message, optionally with an image."""
    data = await request.json()
    message = data.get("message", "")
    image_data = data.get("image", None)
    
    if not message and not image_data:
        return {"response": "Please enter a message."}
    
    # If there's an image, include it in the prompt for vision models
    if image_data:
        response = await chatbot.process_message_with_image(message, image_data)
    else:
        response = await chatbot.process_message(message)
    
    # Ensure response is always a string
    if not isinstance(response, str):
        if isinstance(response, list):
            # Handle list of content blocks
            text_parts = []
            for block in response:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            response = "".join(text_parts) if text_parts else str(response)
        else:
            response = str(response) if response else "Sorry, there was an error processing your request."
    
    # Include supervisor logs for frontend debugging
    return {
        "response": response,
        "agent_used": chatbot.last_agent_used,
        "logs": chatbot.last_supervisor_logs
    }


@app.post("/clear")
async def clear_chat():
    """Clear the chat history."""
    chatbot.clear_history()
    return {"status": "ok"}


@app.post("/save-export")
async def save_export(request: Request):
    """Save the exported chat image to /tmp and return the filename."""
    data = await request.json()
    image_data = data.get("image", "")
    
    if not image_data:
        return {"error": "No image data provided"}
    
    # Remove the data:image/png;base64, prefix if present
    if "," in image_data:
        image_data = image_data.split(",")[1]
    
    # Decode the base64 image
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        return {"error": f"Invalid base64 image: {e}"}
    
    # Generate a unique filename
    timestamp = int(time.time() * 1000)
    filename = f"mister-risker-export-{timestamp}.png"
    filepath = f"/tmp/{filename}"
    
    # Save the image
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    
    return {"filename": filename, "filepath": filepath}


@app.get("/export/{filename}")
async def get_export(filename: str):
    """Serve an exported image from /tmp."""
    filepath = f"/tmp/{filename}"
    if not os.path.exists(filepath):
        return {"error": "File not found"}
    return FileResponse(filepath, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
