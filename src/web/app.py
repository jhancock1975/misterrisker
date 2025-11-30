"""FastAPI web application for Mister Risker trading chat interface.

This module provides a browser-based chat interface that uses an LLM
to interpret natural language requests and execute trades via Coinbase and Schwab.
"""

import os
import json
import asyncio
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.mcp_servers.coinbase import CoinbaseMCPServer, CoinbaseAPIError
from src.mcp_servers.schwab import SchwabMCPServer, SchwabAPIError

# Load environment variables
load_dotenv()


class TradingChatBot:
    """Chat bot that uses LLM to interact with Coinbase and Schwab.
    
    This bot interprets natural language requests and uses the
    MCP Servers to execute trading operations.
    """
    
    def __init__(self):
        """Initialize the chat bot."""
        self.coinbase_server: CoinbaseMCPServer | None = None
        self.schwab_server: SchwabMCPServer | None = None
        self.llm: ChatOpenAI | None = None
        self.conversation_history: list = []
        self.active_broker: str = "coinbase"  # Default broker
        
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

If you don't need to call a tool, just respond normally with text."""

    async def initialize(self):
        """Initialize the MCP servers and LLM."""
        # Initialize Coinbase
        coinbase_api_key = os.getenv("COINBASE_API_KEY")
        coinbase_api_secret = os.getenv("COINBASE_API_SECRET")
        
        if coinbase_api_key and coinbase_api_secret:
            try:
                self.coinbase_server = CoinbaseMCPServer(
                    api_key=coinbase_api_key,
                    api_secret=coinbase_api_secret
                )
            except Exception as e:
                print(f"Warning: Could not initialize Coinbase MCP Server: {e}")
        
        # Initialize Schwab - skip if using placeholder credentials
        schwab_api_key = os.getenv("SCHWAB_API_KEY", "")
        schwab_app_secret = os.getenv("SCHWAB_APP_SECRET", "")
        schwab_callback_url = os.getenv("SCHWAB_CALLBACK_URL")
        schwab_token_path = os.getenv("SCHWAB_TOKEN_PATH")
        
        # Check for real credentials (not placeholders)
        has_real_schwab_creds = (
            schwab_api_key and 
            schwab_app_secret and
            "your_" not in schwab_api_key.lower() and
            "your_" not in schwab_app_secret.lower()
        )
        
        if has_real_schwab_creds:
            try:
                self.schwab_server = SchwabMCPServer(
                    api_key=schwab_api_key,
                    app_secret=schwab_app_secret,
                    callback_url=schwab_callback_url or "https://127.0.0.1:8182/",
                    token_path=schwab_token_path or "/tmp/schwab_token.json"
                )
            except Exception as e:
                print(f"Warning: Could not initialize Schwab MCP Server: {e}")
        
        # Initialize LLM
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                api_key=openai_api_key,
                use_responses_api=True,
            )
        else:
            print("Warning: OPENAI_API_KEY not set. LLM features will be limited.")
    
    async def process_message(self, user_message: str) -> str:
        """Process a user message and return a response.
        
        Args:
            user_message: The user's message
        
        Returns:
            The bot's response
        """
        if not self.llm:
            return "Error: LLM not configured. Please set OPENAI_API_KEY in your .env file."
        
        # Check for broker switch commands
        lower_msg = user_message.lower()
        if "switch to schwab" in lower_msg or "use schwab" in lower_msg:
            self.active_broker = "schwab"
            return "🔄 Switched to **Schwab** (stocks and options trading). How can I help you with your Schwab account?"
        elif "switch to coinbase" in lower_msg or "use coinbase" in lower_msg:
            self.active_broker = "coinbase"
            return "🔄 Switched to **Coinbase** (crypto trading). How can I help you with your crypto portfolio?"
        
        # Check if active broker is configured
        if self.active_broker == "coinbase" and not self.coinbase_server:
            return "Error: Coinbase not configured. Please set COINBASE_API_KEY and COINBASE_API_SECRET in your .env file, or say 'switch to schwab'."
        elif self.active_broker == "schwab" and not self.schwab_server:
            return "Error: Schwab not configured. Please set SCHWAB_API_KEY, SCHWAB_APP_SECRET, SCHWAB_CALLBACK_URL, and SCHWAB_TOKEN_PATH in your .env file, or say 'switch to coinbase'."
        
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
    
    async def _execute_tool(self, tool_name: str, params: dict) -> dict:
        """Execute a tool on the active broker.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
        
        Returns:
            Tool execution result
        """
        try:
            if self.active_broker == "coinbase":
                result = await self.coinbase_server.call_tool(tool_name, params)
            else:  # schwab
                result = await self.schwab_server.call_tool(tool_name, params)
            return result
        except (CoinbaseAPIError, SchwabAPIError) as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def clear_history(self):
        """Clear the conversation history and reset broker to default."""
        self.conversation_history = []
        self.active_broker = "coinbase"
    
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


# Global chatbot instance
chatbot = TradingChatBot()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    await chatbot.initialize()
    yield


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
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mister Risker</title>
    <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
    <link rel="shortcut icon" type="image/svg+xml" href="/static/favicon.svg">
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
            width: 100%;
            max-width: 800px;
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
        
        .broker-indicator {
            margin-left: auto;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .broker-coinbase {
            background: linear-gradient(135deg, #0052ff 0%, #0039b3 100%);
        }
        
        .broker-schwab {
            background: linear-gradient(135deg, #00a0dc 0%, #006b99 100%);
        }
        
        .chat-container {
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
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
        
        .message-content {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 16px;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #00d4aa 0%, #0052ff 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .message.assistant .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
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
        
        .suggestion.broker-switch {
            border-color: #00d4aa;
            color: #00d4aa;
        }
        
        .clear-btn {
            position: absolute;
            top: 50%;
            right: 20px;
            transform: translateY(-50%);
            padding: 8px 16px;
            background: rgba(255,255,255,0.2);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 0.875rem;
            cursor: pointer;
        }
        
        .clear-btn:hover {
            background: rgba(255,255,255,0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-logo">M</div>
            <h1>Mister Risker</h1>
            <span class="broker-indicator broker-coinbase" id="brokerIndicator">Coinbase</span>
            <button class="clear-btn" onclick="clearChat()">Clear</button>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message assistant">
                <span class="message-label">Mister Risker</span>
                <div class="message-content">
                    👋 Hello! I'm Mister Risker, your multi-broker trading assistant.

I can help you trade on both **Coinbase** (crypto) and **Schwab** (stocks/options).

**Coinbase (Currently Active):**
• Check crypto balances and prices
• Buy/sell Bitcoin, Ethereum, and more
• View your crypto portfolio

**Schwab:**
• Check stock account balances
• Get stock quotes and option chains
• Place equity and options orders

Say "switch to schwab" or "switch to coinbase" to change brokers.

What would you like to do today?
                </div>
            </div>
        </div>
        
        <div class="suggestions">
            <span class="suggestion" onclick="sendSuggestion('What are my account balances?')">💰 Balances</span>
            <span class="suggestion" onclick="sendSuggestion('What is the current price of Bitcoin?')">₿ BTC Price</span>
            <span class="suggestion" onclick="sendSuggestion('Show my portfolio')">💼 Portfolio</span>
            <span class="suggestion broker-switch" onclick="sendSuggestion('Switch to Schwab')">🔄 Switch to Schwab</span>
            <span class="suggestion" onclick="sendSuggestion('What are the market movers today?')">📈 Market Movers</span>
        </div>
        
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Ask me anything about trading..." onkeypress="handleKeyPress(event)">
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>
    
    <script>
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const brokerIndicator = document.getElementById('brokerIndicator');
        const brokerSwitchBtn = document.querySelector('.broker-switch');
        
        function updateBrokerUI(text) {
            const lowerText = text.toLowerCase();
            if (lowerText.includes('switched to schwab') || lowerText.includes('🔄 switched to **schwab**')) {
                brokerIndicator.textContent = 'Schwab';
                brokerIndicator.className = 'broker-indicator broker-schwab';
                brokerSwitchBtn.textContent = '🔄 Switch to Coinbase';
                brokerSwitchBtn.onclick = function() { sendSuggestion('Switch to Coinbase'); };
            } else if (lowerText.includes('switched to coinbase') || lowerText.includes('🔄 switched to **coinbase**')) {
                brokerIndicator.textContent = 'Coinbase';
                brokerIndicator.className = 'broker-indicator broker-coinbase';
                brokerSwitchBtn.textContent = '🔄 Switch to Schwab';
                brokerSwitchBtn.onclick = function() { sendSuggestion('Switch to Schwab'); };
            }
        }
        
        function addMessage(content, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
            
            const label = document.createElement('span');
            label.className = 'message-label';
            label.textContent = isUser ? 'You' : 'Mister Risker';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(label);
            messageDiv.appendChild(contentDiv);
            chatContainer.appendChild(messageDiv);
            
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            if (!isUser) {
                updateBrokerUI(content);
            }
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
            if (!message) return;
            
            addMessage(message, true);
            messageInput.value = '';
            sendBtn.disabled = true;
            showTyping();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await response.json();
                hideTyping();
                addMessage(data.response, false);
            } catch (error) {
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
                brokerIndicator.textContent = 'Coinbase';
                brokerIndicator.className = 'broker-indicator broker-coinbase';
                // Reset broker switch button
                brokerSwitchBtn.textContent = '🔄 Switch to Schwab';
                brokerSwitchBtn.onclick = function() { sendSuggestion('Switch to Schwab'); };
                chatContainer.innerHTML = `
                    <div class="message assistant">
                        <span class="message-label">Mister Risker</span>
                        <div class="message-content">
                    👋 Hello! I'm Mister Risker, your multi-broker trading assistant.

I can help you trade on both **Coinbase** (crypto) and **Schwab** (stocks/options).

**Coinbase (Currently Active):**
• Check crypto balances and prices
• Buy/sell Bitcoin, Ethereum, and more
• View your crypto portfolio

**Schwab:**
• Check stock account balances
• Get stock quotes and option chains
• Place equity and options orders

Say "switch to schwab" or "switch to coinbase" to change brokers.

What would you like to do today?
                        </div>
                    </div>
                `;
            } catch (error) {
                console.error('Error clearing chat:', error);
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
    """Process a chat message."""
    data = await request.json()
    message = data.get("message", "")
    
    if not message:
        return {"response": "Please enter a message."}
    
    response = await chatbot.process_message(message)
    return {"response": response}


@app.post("/clear")
async def clear_chat():
    """Clear the chat history."""
    chatbot.clear_history()
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
