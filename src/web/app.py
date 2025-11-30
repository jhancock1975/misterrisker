"""FastAPI web application for Coinbase trading chat interface.

This module provides a browser-based chat interface that uses an LLM
to interpret natural language requests and execute Coinbase trades.
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

# Load environment variables
load_dotenv()


class CoinbaseChatBot:
    """Chat bot that uses LLM to interact with Coinbase.
    
    This bot interprets natural language requests and uses the
    Coinbase MCP Server to execute trading operations.
    """
    
    def __init__(self):
        """Initialize the chat bot."""
        self.mcp_server: CoinbaseMCPServer | None = None
        self.llm: ChatOpenAI | None = None
        self.conversation_history: list = []
        
        # System prompt for the LLM
        self.system_prompt = """You are a helpful Coinbase trading assistant. You can help users:
- Check their account balances
- Get current prices for cryptocurrencies
- Place buy and sell orders (market and limit)
- View their open orders
- Get portfolio summaries
- View market data and candles

When users ask about trading, first confirm the details before executing trades.
For buy orders, clarify the amount in USD they want to spend.
For sell orders, clarify the amount of crypto they want to sell.

Always be helpful and explain what you're doing. If there's an error, explain it clearly.

Available trading pairs include: BTC-USD, ETH-USD, SOL-USD, and many others.

IMPORTANT: When you need to call a tool, respond with a JSON object in this format:
{"tool": "tool_name", "params": {"param1": "value1"}}

Available tools:
- get_accounts: Get all account balances (no params needed)
- get_product: Get details for a product (params: product_id like "BTC-USD")
- get_best_bid_ask: Get current prices (params: product_ids as list like ["BTC-USD"])
- market_order_buy: Buy crypto with USD (params: product_id, quote_size as string like "100.00")
- market_order_sell: Sell crypto (params: product_id, base_size as string like "0.01")
- list_orders: List orders (params: order_status as list like ["OPEN"])
- get_portfolios: Get portfolios (no params needed)
- get_candles: Get price history (params: product_id, start, end, granularity like "ONE_HOUR")

If you don't need to call a tool, just respond normally with text."""

    async def initialize(self):
        """Initialize the MCP server and LLM."""
        api_key = os.getenv("COINBASE_API_KEY")
        api_secret = os.getenv("COINBASE_API_SECRET")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key and api_secret:
            try:
                self.mcp_server = CoinbaseMCPServer(
                    api_key=api_key,
                    api_secret=api_secret
                )
            except Exception as e:
                print(f"Warning: Could not initialize Coinbase MCP Server: {e}")
        
        if openai_api_key:
            # Use OpenAI Responses API for enhanced tool calling and conversation state
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                api_key=openai_api_key,
                use_responses_api=True,  # Enable OpenAI Responses API
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
        
        if not self.mcp_server:
            return "Error: Coinbase not configured. Please set COINBASE_API_KEY and COINBASE_API_SECRET in your .env file."
        
        # Add user message to history
        self.conversation_history.append(HumanMessage(content=user_message))
        
        # Build messages for LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            *self.conversation_history
        ]
        
        try:
            # Get LLM response
            response = await self.llm.ainvoke(messages)
            response_text = self._extract_content(response.content)
            
            # Check if LLM wants to call a tool - look for JSON anywhere in the response
            tool_call = self._extract_tool_call(response_text)
            
            if tool_call:
                # Execute the tool
                tool_result = await self._execute_tool(
                    tool_call["tool"],
                    tool_call.get("params", {})
                )
                
                # Get LLM to interpret the result
                self.conversation_history.append(AIMessage(content=f"Tool result: {json.dumps(tool_result, indent=2)}"))
                
                interpret_messages = [
                    SystemMessage(content="""You are a helpful Coinbase trading assistant. 
Interpret the following tool result and explain it to the user in a friendly, clear way.
- Format currency amounts with $ signs and 2 decimal places for USD
- Format crypto amounts with appropriate precision (e.g., 0.00123456 BTC)
- List each account/balance on its own line
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
        """Execute a Coinbase MCP tool.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
        
        Returns:
            Tool execution result
        """
        try:
            result = await self.mcp_server.call_tool(tool_name, params)
            return result
        except CoinbaseAPIError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
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
chatbot = CoinbaseChatBot()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    await chatbot.initialize()
    yield


# Create FastAPI app
app = FastAPI(
    title="Coinbase Trading Assistant",
    description="A chat interface for trading on Coinbase",
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
    <title>Coinbase Trading Assistant</title>
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
            background: linear-gradient(135deg, #0052ff 0%, #0039b3 100%);
            color: white;
            padding: 20px 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .header svg {
            width: 32px;
            height: 32px;
        }
        
        .header h1 {
            font-size: 1.5rem;
            font-weight: 600;
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
            background: #0052ff;
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
            border-color: #0052ff;
        }
        
        .input-container button {
            padding: 14px 28px;
            background: #0052ff;
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .input-container button:hover {
            background: #0039b3;
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
            background: #0052ff;
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
            background: #0052ff;
            color: white;
            border-color: #0052ff;
        }
        
        .clear-btn {
            position: absolute;
            top: 20px;
            right: 20px;
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
        
        .header {
            position: relative;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <svg viewBox="0 0 1024 1024" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="512" cy="512" r="512" fill="white"/>
                <path d="M512 256C371.2 256 256 371.2 256 512s115.2 256 256 256 256-115.2 256-256S652.8 256 512 256zm0 384c-70.4 0-128-57.6-128-128s57.6-128 128-128 128 57.6 128 128-57.6 128-128 128z" fill="#0052FF"/>
            </svg>
            <h1>Coinbase Trading Assistant</h1>
            <button class="clear-btn" onclick="clearChat()">Clear Chat</button>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message assistant">
                <span class="message-label">Assistant</span>
                <div class="message-content">
                    👋 Hello! I'm your Coinbase trading assistant. I can help you:
                    
• Check your account balances
• Get current crypto prices
• Place buy and sell orders
• View your open orders
• Get market data

What would you like to do today?
                </div>
            </div>
        </div>
        
        <div class="suggestions">
            <span class="suggestion" onclick="sendSuggestion('What are my account balances?')">💰 My Balances</span>
            <span class="suggestion" onclick="sendSuggestion('What is the current price of Bitcoin?')">📈 BTC Price</span>
            <span class="suggestion" onclick="sendSuggestion('What is the current price of Ethereum?')">📊 ETH Price</span>
            <span class="suggestion" onclick="sendSuggestion('Show my open orders')">📋 Open Orders</span>
            <span class="suggestion" onclick="sendSuggestion('Show my portfolio')">💼 Portfolio</span>
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
        
        function addMessage(content, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
            
            const label = document.createElement('span');
            label.className = 'message-label';
            label.textContent = isUser ? 'You' : 'Assistant';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(label);
            messageDiv.appendChild(contentDiv);
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
                chatContainer.innerHTML = `
                    <div class="message assistant">
                        <span class="message-label">Assistant</span>
                        <div class="message-content">
                            👋 Hello! I'm your Coinbase trading assistant. I can help you:
                            
• Check your account balances
• Get current crypto prices
• Place buy and sell orders
• View your open orders
• Get market data

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
