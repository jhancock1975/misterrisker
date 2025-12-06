"""Coinbase WebSocket Service for real-time candle monitoring.

This service connects to the Coinbase Advanced Trade WebSocket API
and monitors candle data for specified products.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable
import secrets

import websockets
import jwt

# Configure logging
logger = logging.getLogger(__name__)


class CoinbaseWebSocketService:
    """WebSocket service for monitoring Coinbase candle data.
    
    Automatically connects to the Coinbase WebSocket API and subscribes
    to the candles channel for specified products.
    """
    
    WS_URL = "wss://advanced-trade-ws.coinbase.com"
    
    # Products to monitor (trading pairs)
    DEFAULT_PRODUCTS = [
        "BTC-USD",   # Bitcoin
        "ETH-USD",   # Ethereum
        "XRP-USD",   # Ripple
        "SOL-USD",   # Solana
        "ZEC-USD",   # Zcash
    ]
    
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        products: list[str] | None = None,
        on_candle: Callable[[dict[str, Any]], None] | None = None,
    ):
        """Initialize the WebSocket service.
        
        Args:
            api_key: Coinbase CDP API key (format: organizations/{org_id}/apiKeys/{key_id})
            api_secret: Coinbase CDP API secret (EC private key in PEM format)
            products: List of product IDs to monitor (default: BTC, ETH, XRP, SOL, ZEC)
            on_candle: Optional callback for candle updates
        """
        self.api_key = api_key or os.getenv("COINBASE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET", "")
        self.products = products or self.DEFAULT_PRODUCTS
        self.on_candle = on_candle
        
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._summary_task: asyncio.Task | None = None
        self._reconnect_delay = 5  # seconds
        self._summary_interval = 300  # 5 minutes in seconds
        
        # Store latest candle data for each product
        self._latest_candles: dict[str, dict[str, Any]] = {}
        
        # Track candle update counts for summary
        self._candle_counts: dict[str, int] = {}
        self._last_summary_time: float = time.time()
    
    def _generate_jwt(self) -> str:
        """Generate a JWT for WebSocket authentication.
        
        Returns:
            Signed JWT token string
        """
        now = int(time.time())
        
        payload = {
            "iss": "cdp",
            "nbf": now,
            "exp": now + 120,  # 2 minutes expiry
            "sub": self.api_key,
        }
        
        headers = {
            "kid": self.api_key,
            "nonce": secrets.token_hex(16),
            "alg": "ES256",
        }
        
        token = jwt.encode(
            payload,
            self.api_secret,
            algorithm="ES256",
            headers=headers,
        )
        
        return token
    
    async def _subscribe(self) -> None:
        """Send subscription message for candles channel."""
        if not self._ws:
            return
        
        message = {
            "type": "subscribe",
            "product_ids": self.products,
            "channel": "candles",
        }
        
        # Add JWT if we have credentials
        if self.api_key and self.api_secret:
            try:
                message["jwt"] = self._generate_jwt()
            except Exception as e:
                logger.warning(f"Failed to generate JWT, subscribing without auth: {e}")
        
        await self._ws.send(json.dumps(message))
        logger.info(f"📊 Subscribed to candles channel for: {', '.join(self.products)}")
    
    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message.
        
        Args:
            message: Raw JSON message string
        """
        try:
            data = json.loads(message)
            channel = data.get("channel", "")
            
            if channel == "candles":
                await self._handle_candle_message(data)
            elif channel == "subscriptions":
                logger.info(f"📡 Subscription confirmed: {data}")
            elif data.get("type") == "error":
                logger.error(f"❌ WebSocket error: {data.get('message', data)}")
            else:
                # Log other message types at debug level
                logger.debug(f"Received message: {channel or data.get('type', 'unknown')}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
    
    async def _handle_candle_message(self, data: dict[str, Any]) -> None:
        """Handle candle update message.
        
        Args:
            data: Parsed candle message
        """
        events = data.get("events", [])
        timestamp = data.get("timestamp", "")
        
        for event in events:
            event_type = event.get("type", "")
            candles = event.get("candles", [])
            
            for candle in candles:
                product_id = candle.get("product_id", "unknown")
                
                # Extract candle data
                candle_data = {
                    "product_id": product_id,
                    "start": candle.get("start"),
                    "open": candle.get("open"),
                    "high": candle.get("high"),
                    "low": candle.get("low"),
                    "close": candle.get("close"),
                    "volume": candle.get("volume"),
                    "timestamp": timestamp,
                    "event_type": event_type,
                }
                
                # Store latest candle
                self._latest_candles[product_id] = candle_data
                
                # Increment candle count for this product
                self._candle_counts[product_id] = self._candle_counts.get(product_id, 0) + 1
                
                # Call callback if provided
                if self.on_candle:
                    try:
                        self.on_candle(candle_data)
                    except Exception as e:
                        logger.error(f"Error in candle callback: {e}")
    
    async def _log_summary(self) -> None:
        """Log a summary of candle data every 5 minutes."""
        while self._running:
            await asyncio.sleep(self._summary_interval)
            
            if not self._latest_candles:
                logger.info("📊 No candle data received yet")
                continue
            
            # Build summary
            now = datetime.now().strftime("%H:%M:%S")
            total_updates = sum(self._candle_counts.values())
            
            summary_lines = [f"📊 CANDLE SUMMARY ({now}) - {total_updates} updates in last 5 min:"]
            
            for product_id in sorted(self._latest_candles.keys()):
                candle = self._latest_candles[product_id]
                count = self._candle_counts.get(product_id, 0)
                price = candle.get('close', 'N/A')
                high = candle.get('high', 'N/A')
                low = candle.get('low', 'N/A')
                summary_lines.append(
                    f"  {product_id}: ${price} (H:${high} L:${low}) [{count} updates]"
                )
            
            logger.info("\n".join(summary_lines))
            
            # Reset counts for next interval
            self._candle_counts = {}
            self._last_summary_time = time.time()
    
    async def _connect_and_listen(self) -> None:
        """Main connection loop with reconnection logic."""
        while self._running:
            try:
                logger.info(f"🔌 Connecting to Coinbase WebSocket: {self.WS_URL}")
                
                async with websockets.connect(self.WS_URL) as ws:
                    self._ws = ws
                    logger.info("✅ Connected to Coinbase WebSocket")
                    
                    # Subscribe to candles channel
                    await self._subscribe()
                    
                    # Listen for messages
                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)
                
            except websockets.ConnectionClosed as e:
                logger.warning(f"⚠️ WebSocket connection closed: {e}")
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
            finally:
                self._ws = None
            
            if self._running:
                logger.info(f"🔄 Reconnecting in {self._reconnect_delay} seconds...")
                await asyncio.sleep(self._reconnect_delay)
    
    async def start(self) -> None:
        """Start the WebSocket connection and monitoring.
        
        This runs in the background and automatically reconnects on disconnection.
        """
        if self._running:
            logger.warning("WebSocket service already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._connect_and_listen())
        self._summary_task = asyncio.create_task(self._log_summary())
        logger.info("🚀 Coinbase WebSocket monitoring started (summary every 5 min)")
    
    async def stop(self) -> None:
        """Stop the WebSocket connection and monitoring."""
        self._running = False
        
        if self._ws:
            await self._ws.close()
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        if self._summary_task:
            self._summary_task.cancel()
            try:
                await self._summary_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Coinbase WebSocket monitoring stopped")
    
    def get_latest_candle(self, product_id: str) -> dict[str, Any] | None:
        """Get the latest candle data for a product.
        
        Args:
            product_id: Product ID (e.g., "BTC-USD")
            
        Returns:
            Latest candle data or None if not available
        """
        return self._latest_candles.get(product_id)
    
    def get_all_latest_candles(self) -> dict[str, dict[str, Any]]:
        """Get latest candle data for all monitored products.
        
        Returns:
            Dictionary of product_id -> candle data
        """
        return self._latest_candles.copy()
    
    @property
    def is_running(self) -> bool:
        """Check if the WebSocket service is running."""
        return self._running
    
    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket is currently connected."""
        return self._ws is not None and self._ws.open
