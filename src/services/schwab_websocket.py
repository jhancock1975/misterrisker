"""Schwab WebSocket Service for real-time Level One equity quotes.

This service connects to the Schwab Streaming API WebSocket
and monitors real-time quote data for specified symbols.

Based on the schwabase implementation pattern for Schwab streaming.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable

import requests
import websockets

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

SCHWAB_TRADER_API_URL = os.getenv("SCHWAB_TRADER_API_URL", "https://api.schwabapi.com/trader/v1")
SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"


class SchwabWebSocketService:
    """WebSocket service for monitoring Schwab Level One equity quotes.
    
    Connects to the Schwab Streaming API via WebSocket and subscribes
    to real-time quote data for specified symbols.
    """
    
    # Default symbols to monitor
    DEFAULT_SYMBOLS = [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "NVDA",   # NVIDIA
        "GOOGL",  # Alphabet
        "AMZN",   # Amazon
    ]
    
    # Level One Equity fields to subscribe to
    # 0=Symbol, 1=Bid, 2=Ask, 3=Last, 4=Bid Size, 5=Ask Size, 
    # 6=Ask ID, 7=Bid ID, 8=Total Volume, 9=Last Size
    DEFAULT_FIELDS = "0,1,2,3,4,5,8,9"
    
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        symbols: list[str] | None = None,
        on_quote: Callable[[dict[str, Any]], None] | None = None,
    ):
        """Initialize the WebSocket service.
        
        Args:
            client_id: Schwab Client ID
            client_secret: Schwab Client Secret
            refresh_token: Schwab OAuth Refresh Token
            symbols: List of stock symbols to monitor
            on_quote: Optional callback for quote updates
        """
        self.client_id = client_id or os.getenv("SCHWAB_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SCHWAB_CLIENT_SECRET")
        self.refresh_token = refresh_token or os.getenv("SCHWAB_REFRESH_TOKEN")
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.on_quote = on_quote
        
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._summary_task: asyncio.Task | None = None
        self._reconnect_delay = 5  # seconds
        self._summary_interval = 300  # 5 minutes in seconds
        
        # Token state
        self._access_token: str | None = None
        self._token_expires_at: float = 0
        
        # Streamer connection info
        self._streamer_url: str | None = None
        self._customer_id: str | None = None
        self._correl_id: str | None = None
        
        # Store latest quote data for each symbol
        self._latest_quotes: dict[str, dict[str, Any]] = {}
        
        # Track quote update counts for summary
        self._quote_counts: dict[str, int] = {}
        self._last_summary_time: float = time.time()
    
    def _refresh_access_token(self) -> str:
        """Refresh the Schwab access token.
        
        Returns:
            The new access token
            
        Raises:
            Exception: If token refresh fails
        """
        if not self.client_id or not self.client_secret or not self.refresh_token:
            raise ValueError(
                "SCHWAB_CLIENT_ID, SCHWAB_CLIENT_SECRET, and SCHWAB_REFRESH_TOKEN must be configured"
            )
        
        response = requests.post(
            SCHWAB_TOKEN_URL,
            auth=(self.client_id, self.client_secret),
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            },
            timeout=30,
        )
        
        if not response.ok:
            raise Exception(f"Failed to refresh Schwab token: {response.text}")
        
        data = response.json()
        self._access_token = data["access_token"]
        self._token_expires_at = time.time() + data.get("expires_in", 1800) - 60
        logger.info(f"Schwab WebSocket token refreshed, valid for {data.get('expires_in', 1800)}s")
        return data["access_token"]
    
    def _get_access_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        if self._token_expires_at < time.time() or not self._access_token:
            return self._refresh_access_token()
        return self._access_token
    
    def _get_streamer_info(self) -> dict[str, Any]:
        """Fetch streamer connection info from user preferences.
        
        Returns:
            Dict with streamerSocketUrl, schwabClientCustomerId, schwabClientCorrelId
            
        Raises:
            Exception: If streamer info cannot be fetched
        """
        token = self._get_access_token()
        
        response = requests.get(
            f"{SCHWAB_TRADER_API_URL}/userPreference",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            timeout=30,
        )
        
        if not response.ok:
            raise Exception(f"Failed to get user preferences: {response.text}")
        
        prefs = response.json()
        logger.debug(f"userPreference response: {prefs}")
        
        # Normalize various response shapes to a dict
        candidate = prefs
        if isinstance(candidate, list):
            candidate = next((item for item in candidate if isinstance(item, dict)), {})
        if isinstance(candidate, dict) and "preferences" in candidate:
            if isinstance(candidate["preferences"], list):
                candidate = next(
                    (item for item in candidate["preferences"] if isinstance(item, dict)), 
                    candidate
                )
        
        # Try pulling streamerInfo
        info = candidate.get("streamerInfo", {}) if isinstance(candidate, dict) else {}
        if isinstance(info, list):
            info = next((item for item in info if isinstance(item, dict)), {})
        if not isinstance(info, dict):
            info = {}
        
        self._streamer_url = info.get("streamerSocketUrl")
        self._customer_id = info.get("schwabClientCustomerId")
        self._correl_id = info.get("schwabClientCorrelId")
        
        logger.info(
            f"Streamer info: url={self._streamer_url}, "
            f"customerId={self._customer_id[:8] + '...' if self._customer_id else None}"
        )
        
        return info
    
    async def _connect(self) -> bool:
        """Connect to the Schwab WebSocket and authenticate.
        
        Returns:
            True if connection and login succeeded
        """
        try:
            # Get streamer info if we don't have it
            if not self._streamer_url or not self._customer_id or not self._correl_id:
                self._get_streamer_info()
            
            if not self._streamer_url:
                logger.error("No streamer URL available")
                return False
            
            logger.info(f"🔌 Connecting to Schwab WebSocket: {self._streamer_url}")
            self._ws = await websockets.connect(self._streamer_url)
            logger.info("✅ Connected to Schwab WebSocket")
            
            # Send LOGIN request
            token = self._get_access_token()
            login_payload = {
                "requests": [
                    {
                        "service": "ADMIN",
                        "command": "LOGIN",
                        "requestid": "0",
                        "SchwabClientCustomerId": self._customer_id,
                        "SchwabClientCorrelId": self._correl_id,
                        "parameters": {
                            "Authorization": token,
                            "SchwabClientChannel": "API",
                            "SchwabClientFunctionId": "MISTERRISKER",
                        },
                    }
                ]
            }
            
            await self._ws.send(json.dumps(login_payload))
            logger.debug("LOGIN request sent")
            
            # Wait for login response
            raw_login = await self._ws.recv()
            login_msg = json.loads(raw_login)
            logger.debug(f"LOGIN response: {login_msg}")
            
            # Verify login success (code 0)
            login_code = None
            if isinstance(login_msg, dict) and "response" in login_msg:
                resp_list = login_msg["response"]
                if isinstance(resp_list, list) and resp_list:
                    content = resp_list[0].get("content", {})
                    if isinstance(content, dict):
                        login_code = content.get("code")
            
            if login_code not in (0, "0"):
                logger.error(f"❌ Schwab streamer login failed: {login_msg}")
                return False
            
            logger.info("✅ Schwab WebSocket login successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Schwab WebSocket: {e}")
            return False
    
    async def _subscribe(self) -> None:
        """Subscribe to Level One equity quotes."""
        if not self._ws:
            return
        
        subs_payload = {
            "requests": [
                {
                    "service": "LEVELONE_EQUITIES",
                    "command": "SUBS",
                    "requestid": "1",
                    "SchwabClientCustomerId": self._customer_id,
                    "SchwabClientCorrelId": self._correl_id,
                    "parameters": {
                        "keys": ",".join(self.symbols),
                        "fields": self.DEFAULT_FIELDS,
                    },
                }
            ]
        }
        
        await self._ws.send(json.dumps(subs_payload))
        logger.info(f"📊 Subscribed to Level One quotes for: {', '.join(self.symbols)}")
    
    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            # Handle response messages (subscription confirmations)
            if "response" in data:
                for resp in data.get("response", []):
                    service = resp.get("service", "")
                    content = resp.get("content", {})
                    code = content.get("code") if isinstance(content, dict) else None
                    if code == 0:
                        logger.info(f"📡 Subscription confirmed: {service}")
                    else:
                        logger.warning(f"⚠️ Subscription response: {resp}")
            
            # Handle data messages (actual quote updates)
            if "data" in data:
                for item in data.get("data", []):
                    service = item.get("service", "")
                    if service == "LEVELONE_EQUITIES":
                        await self._handle_quote_data(item)
            
            # Handle notify messages (heartbeats, etc.)
            if "notify" in data:
                logger.debug(f"Heartbeat: {data.get('notify')}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
    
    async def _handle_quote_data(self, data: dict[str, Any]) -> None:
        """Handle Level One equity quote data.
        
        Field mapping:
        - 0: Symbol
        - 1: Bid Price
        - 2: Ask Price  
        - 3: Last Price
        - 4: Bid Size
        - 5: Ask Size
        - 8: Total Volume
        - 9: Last Size
        """
        timestamp = data.get("timestamp")
        content = data.get("content", [])
        
        for quote in content:
            symbol = quote.get("key", quote.get("0", "UNKNOWN"))
            
            quote_data = {
                "symbol": symbol,
                "bid": quote.get("1"),
                "ask": quote.get("2"),
                "last": quote.get("3"),
                "bid_size": quote.get("4"),
                "ask_size": quote.get("5"),
                "volume": quote.get("8"),
                "last_size": quote.get("9"),
                "timestamp": timestamp,
            }
            
            # Store latest quote
            self._latest_quotes[symbol] = quote_data
            
            # Increment count
            self._quote_counts[symbol] = self._quote_counts.get(symbol, 0) + 1
            
            # Call callback if provided
            if self.on_quote:
                try:
                    self.on_quote(quote_data)
                except Exception as e:
                    logger.error(f"Error in quote callback: {e}")
    
    async def _log_summary(self) -> None:
        """Log a summary of quote data every 5 minutes."""
        while self._running:
            await asyncio.sleep(self._summary_interval)
            
            if not self._latest_quotes:
                logger.info("📊 No Schwab quote data received yet")
                continue
            
            # Build summary
            now = datetime.now().strftime("%H:%M:%S")
            total_updates = sum(self._quote_counts.values())
            
            lines = [f"📊 SCHWAB QUOTE SUMMARY ({now}) - {total_updates} updates in last 5 min:"]
            for symbol in sorted(self._latest_quotes.keys()):
                quote = self._latest_quotes[symbol]
                count = self._quote_counts.get(symbol, 0)
                last = quote.get("last", "N/A")
                bid = quote.get("bid", "N/A")
                ask = quote.get("ask", "N/A")
                lines.append(f"  {symbol}: ${last} (Bid:${bid} Ask:${ask}) [{count} updates]")
            
            logger.info("\n".join(lines))
            
            # Reset counts
            self._quote_counts = {}
            self._last_summary_time = time.time()
    
    async def _run(self) -> None:
        """Main run loop - connect, subscribe, and handle messages."""
        logger.info("🚀 Schwab WebSocket monitoring started (summary every 5 min)")
        
        while self._running:
            try:
                # Connect and login
                if not await self._connect():
                    logger.warning(f"⚠️ Connection failed, retrying in {self._reconnect_delay}s...")
                    await asyncio.sleep(self._reconnect_delay)
                    continue
                
                # Subscribe to quotes
                await self._subscribe()
                
                # Handle messages
                while self._running and self._ws:
                    try:
                        message = await asyncio.wait_for(
                            self._ws.recv(),
                            timeout=30.0  # Heartbeat timeout
                        )
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        # Send a ping to keep connection alive
                        logger.debug("Sending keepalive ping")
                        continue
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"⚠️ WebSocket connection closed: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
            
            if self._running:
                logger.info(f"🔄 Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
        
        logger.info("🛑 Schwab WebSocket monitoring stopped")
    
    async def start(self) -> None:
        """Start the WebSocket monitoring in the background."""
        if self._running:
            logger.warning("WebSocket service already running")
            return
        
        # Check credentials
        if not self.client_id or not self.client_secret or not self.refresh_token:
            logger.warning("⚠️ Schwab credentials not configured, skipping WebSocket start")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run())
        self._summary_task = asyncio.create_task(self._log_summary())
    
    async def stop(self) -> None:
        """Stop the WebSocket monitoring."""
        self._running = False
        
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        if self._summary_task:
            self._summary_task.cancel()
            try:
                await self._summary_task
            except asyncio.CancelledError:
                pass
            self._summary_task = None
    
    def get_latest_quote(self, symbol: str) -> dict[str, Any] | None:
        """Get the latest quote for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest quote data or None if not available
        """
        return self._latest_quotes.get(symbol.upper())
    
    def get_all_quotes(self) -> dict[str, dict[str, Any]]:
        """Get all latest quotes.
        
        Returns:
            Dict mapping symbols to their latest quote data
        """
        return self._latest_quotes.copy()
    
    def add_symbols(self, symbols: list[str]) -> None:
        """Add symbols to monitor (will take effect on next connection)."""
        for symbol in symbols:
            if symbol.upper() not in [s.upper() for s in self.symbols]:
                self.symbols.append(symbol.upper())
    
    def is_running(self) -> bool:
        """Check if the WebSocket service is running."""
        return self._running
