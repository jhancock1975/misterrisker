"""Coinbase MCP Server - Exposes Coinbase API as MCP tools.

This module provides a Model Context Protocol (MCP) server that wraps
the Coinbase Advanced Trade API, exposing all functionality as tools
that can be used by AI agents.
"""

import asyncio
import os
from typing import Any
from collections.abc import Callable

from coinbase.rest import RESTClient
from coinbase.websocket import WSClient

from .exceptions import CoinbaseAPIError, CoinbaseAuthError


class CoinbaseToolError(Exception):
    """Exception raised for MCP tool errors.
    
    Attributes:
        message: Error message describing what went wrong
        tool_name: Name of the tool that caused the error
    """
    
    def __init__(self, message: str, tool_name: str | None = None):
        """Initialize CoinbaseToolError.
        
        Args:
            message: Error message
            tool_name: Name of the tool that caused the error
        """
        self.message = message
        self.tool_name = tool_name
        super().__init__(self.message)


class CoinbaseMCPServer:
    """MCP Server that exposes Coinbase API tools.
    
    This server wraps the Coinbase Advanced Trade API and exposes
    all functionality as MCP-compatible tools that can be discovered
    and invoked by AI agents.
    
    Attributes:
        api_key: Coinbase API key
        api_secret: Coinbase API secret
        _client: Coinbase REST API client
        _ws_client: Coinbase WebSocket client (lazy initialized)
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str
    ):
        """Initialize the Coinbase MCP Server.
        
        Args:
            api_key: Coinbase API key
            api_secret: Coinbase API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Initialize REST client
        self._client = RESTClient(
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        
        # WebSocket client is lazy initialized
        self._ws_client: WSClient | None = None
        
        # Define tool schemas
        self._tool_schemas = self._define_tool_schemas()
    
    @classmethod
    def from_env(cls) -> "CoinbaseMCPServer":
        """Create a CoinbaseMCPServer from environment variables.
        
        Returns:
            CoinbaseMCPServer instance
        
        Raises:
            CoinbaseAuthError: If credentials are not found in environment
        """
        api_key = os.getenv("COINBASE_API_KEY")
        api_secret = os.getenv("COINBASE_API_SECRET")
        
        if not api_key:
            raise CoinbaseAuthError(
                message="API key not found in environment variables",
                status_code=401,
                error_code="MISSING_API_KEY"
            )
        
        if not api_secret:
            raise CoinbaseAuthError(
                message="COINBASE_API_SECRET environment variable not set",
                status_code=401,
                error_code="MISSING_API_SECRET"
            )
        
        return cls(api_key=api_key, api_secret=api_secret)
    
    @property
    def ws_client(self) -> WSClient:
        """Lazy initialize and return WebSocket client."""
        if self._ws_client is None:
            self._ws_client = WSClient(
                api_key=self.api_key,
                api_secret=self.api_secret
            )
        return self._ws_client
    
    def _define_tool_schemas(self) -> list[dict[str, Any]]:
        """Define all MCP tool schemas.
        
        Returns:
            List of tool schema dictionaries.
        """
        return [
            # Account tools
            {
                "name": "get_accounts",
                "description": "Get all accounts for the authenticated user. Returns account balances for all currencies.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Maximum number of accounts to return"},
                        "cursor": {"type": "string", "description": "Pagination cursor for next page"}
                    }
                }
            },
            {
                "name": "get_account",
                "description": "Get a specific account by UUID.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_uuid": {"type": "string", "description": "The account UUID"}
                    },
                    "required": ["account_uuid"]
                }
            },
            # Product tools
            {
                "name": "get_products",
                "description": "Get all available trading products/pairs.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_type": {"type": "string", "description": "Filter by product type (e.g., 'SPOT')"},
                        "product_ids": {"type": "array", "items": {"type": "string"}, "description": "Filter by product IDs"}
                    }
                }
            },
            {
                "name": "get_product",
                "description": "Get details for a specific product.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "The product ID (e.g., 'BTC-USD')"}
                    },
                    "required": ["product_id"]
                }
            },
            {
                "name": "get_product_book",
                "description": "Get the order book for a product.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "The product ID (e.g., 'BTC-USD')"},
                        "limit": {"type": "integer", "description": "Number of price levels to return"}
                    },
                    "required": ["product_id"]
                }
            },
            {
                "name": "get_best_bid_ask",
                "description": "Get the best bid/ask prices for products.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_ids": {"type": "array", "items": {"type": "string"}, "description": "List of product IDs"}
                    }
                }
            },
            # Market data tools
            {
                "name": "get_candles",
                "description": "Get historical OHLCV candles for a product.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "The product ID"},
                        "start": {"type": "string", "description": "Start time (Unix timestamp or ISO)"},
                        "end": {"type": "string", "description": "End time (Unix timestamp or ISO)"},
                        "granularity": {"type": "string", "description": "Candle granularity (e.g., 'ONE_HOUR')"}
                    },
                    "required": ["product_id", "start", "end", "granularity"]
                }
            },
            {
                "name": "get_market_trades",
                "description": "Get recent market trades for a product.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "The product ID"},
                        "limit": {"type": "integer", "description": "Maximum number of trades to return"}
                    },
                    "required": ["product_id"]
                }
            },
            # Order tools
            {
                "name": "market_order_buy",
                "description": "Place a market buy order.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "The product ID (e.g., 'BTC-USD')"},
                        "quote_size": {"type": "string", "description": "Amount in quote currency (e.g., USD)"}
                    },
                    "required": ["product_id", "quote_size"]
                }
            },
            {
                "name": "market_order_sell",
                "description": "Place a market sell order.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "The product ID (e.g., 'BTC-USD')"},
                        "base_size": {"type": "string", "description": "Amount in base currency (e.g., BTC)"}
                    },
                    "required": ["product_id", "base_size"]
                }
            },
            {
                "name": "limit_order_gtc_buy",
                "description": "Place a Good-Till-Cancelled limit buy order.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "The product ID"},
                        "base_size": {"type": "string", "description": "Amount in base currency"},
                        "limit_price": {"type": "string", "description": "Limit price"}
                    },
                    "required": ["product_id", "base_size", "limit_price"]
                }
            },
            {
                "name": "limit_order_gtc_sell",
                "description": "Place a Good-Till-Cancelled limit sell order.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "The product ID"},
                        "base_size": {"type": "string", "description": "Amount in base currency"},
                        "limit_price": {"type": "string", "description": "Limit price"}
                    },
                    "required": ["product_id", "base_size", "limit_price"]
                }
            },
            {
                "name": "cancel_orders",
                "description": "Cancel one or more orders.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "order_ids": {"type": "array", "items": {"type": "string"}, "description": "List of order IDs to cancel"}
                    },
                    "required": ["order_ids"]
                }
            },
            {
                "name": "get_order",
                "description": "Get details for a specific order.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The order ID"}
                    },
                    "required": ["order_id"]
                }
            },
            {
                "name": "list_orders",
                "description": "List orders with optional filters.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "Filter by product ID"},
                        "order_status": {"type": "array", "items": {"type": "string"}, "description": "Filter by status"}
                    }
                }
            },
            {
                "name": "get_fills",
                "description": "Get fill history for orders.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "Filter by product ID"},
                        "order_id": {"type": "string", "description": "Filter by order ID"}
                    }
                }
            },
            # Portfolio tools
            {
                "name": "get_portfolios",
                "description": "Get all portfolios.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "portfolio_type": {"type": "string", "description": "Filter by portfolio type"}
                    }
                }
            },
            {
                "name": "get_portfolio_breakdown",
                "description": "Get detailed breakdown of a portfolio.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "portfolio_uuid": {"type": "string", "description": "The portfolio UUID"}
                    },
                    "required": ["portfolio_uuid"]
                }
            },
            # Public tools
            {
                "name": "get_unix_time",
                "description": "Get the current server time.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_public_products",
                "description": "Get public product list without authentication.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_public_product",
                "description": "Get public product details without authentication.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "The product ID"}
                    },
                    "required": ["product_id"]
                }
            },
            # Fee tools
            {
                "name": "get_transaction_summary",
                "description": "Get transaction summary with fee information.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date for summary"},
                        "end_date": {"type": "string", "description": "End date for summary"}
                    }
                }
            },
        ]
    
    def list_tools(self) -> list[dict[str, Any]]:
        """List all available MCP tools with their schemas.
        
        Returns:
            List of tool definitions with name, description, and inputSchema.
        """
        return self._tool_schemas
    
    async def call_tool(
        self,
        tool_name: str,
        params: dict[str, Any]
    ) -> dict[str, Any]:
        """Call an MCP tool by name.
        
        Args:
            tool_name: Name of the tool to call
            params: Parameters to pass to the tool
        
        Returns:
            Result from the tool call
        
        Raises:
            CoinbaseToolError: If tool_name is not found or required params missing
            CoinbaseAPIError: If the API call fails
        """
        # Check if tool exists
        tool_names = [t["name"] for t in self._tool_schemas]
        if tool_name not in tool_names:
            raise CoinbaseToolError(f"Unknown tool: {tool_name}", tool_name=tool_name)
        
        # Get tool schema for validation
        tool_schema = next(t for t in self._tool_schemas if t["name"] == tool_name)
        
        # Validate required parameters
        required_params = tool_schema["inputSchema"].get("required", [])
        for param in required_params:
            if param not in params:
                raise CoinbaseToolError(
                    f"Missing required parameter '{param}' for tool '{tool_name}'",
                    tool_name=tool_name
                )
        
        # Route to appropriate handler
        try:
            handler = getattr(self, f"_tool_{tool_name}")
            return await handler(params)
        except CoinbaseAPIError:
            raise
        except CoinbaseToolError:
            raise
        except Exception as e:
            raise CoinbaseAPIError(f"API Error: {str(e)}")
    
    # ===================
    # Tool Handlers
    # ===================
    
    async def _tool_get_accounts(self, params: dict) -> dict:
        """Handle get_accounts tool call."""
        kwargs = {}
        if "limit" in params:
            kwargs["limit"] = params["limit"]
        if "cursor" in params:
            kwargs["cursor"] = params["cursor"]
        return self._process_response(self._client.get_accounts(**kwargs))
    
    async def _tool_get_account(self, params: dict) -> dict:
        """Handle get_account tool call."""
        return self._process_response(
            self._client.get_account(params["account_uuid"])
        )
    
    async def _tool_get_products(self, params: dict) -> dict:
        """Handle get_products tool call."""
        kwargs = {}
        if "product_type" in params:
            kwargs["product_type"] = params["product_type"]
        if "product_ids" in params:
            kwargs["product_ids"] = params["product_ids"]
        return self._process_response(self._client.get_products(**kwargs))
    
    async def _tool_get_product(self, params: dict) -> dict:
        """Handle get_product tool call."""
        return self._process_response(
            self._client.get_product(params["product_id"])
        )
    
    async def _tool_get_product_book(self, params: dict) -> dict:
        """Handle get_product_book tool call."""
        kwargs = {"product_id": params["product_id"]}
        if "limit" in params:
            kwargs["limit"] = params["limit"]
        return self._process_response(self._client.get_product_book(**kwargs))
    
    async def _tool_get_best_bid_ask(self, params: dict) -> dict:
        """Handle get_best_bid_ask tool call."""
        kwargs = {}
        if "product_ids" in params:
            kwargs["product_ids"] = params["product_ids"]
        return self._process_response(self._client.get_best_bid_ask(**kwargs))
    
    async def _tool_get_candles(self, params: dict) -> dict:
        """Handle get_candles tool call."""
        return self._process_response(
            self._client.get_candles(
                product_id=params["product_id"],
                start=params["start"],
                end=params["end"],
                granularity=params["granularity"]
            )
        )
    
    async def _tool_get_market_trades(self, params: dict) -> dict:
        """Handle get_market_trades tool call."""
        kwargs = {"product_id": params["product_id"]}
        if "limit" in params:
            kwargs["limit"] = params["limit"]
        return self._process_response(self._client.get_market_trades(**kwargs))
    
    async def _tool_market_order_buy(self, params: dict) -> dict:
        """Handle market_order_buy tool call."""
        return self._process_response(
            self._client.market_order_buy(
                product_id=params["product_id"],
                quote_size=params["quote_size"]
            )
        )
    
    async def _tool_market_order_sell(self, params: dict) -> dict:
        """Handle market_order_sell tool call."""
        return self._process_response(
            self._client.market_order_sell(
                product_id=params["product_id"],
                base_size=params["base_size"]
            )
        )
    
    async def _tool_limit_order_gtc_buy(self, params: dict) -> dict:
        """Handle limit_order_gtc_buy tool call."""
        return self._process_response(
            self._client.limit_order_gtc_buy(
                product_id=params["product_id"],
                base_size=params["base_size"],
                limit_price=params["limit_price"]
            )
        )
    
    async def _tool_limit_order_gtc_sell(self, params: dict) -> dict:
        """Handle limit_order_gtc_sell tool call."""
        return self._process_response(
            self._client.limit_order_gtc_sell(
                product_id=params["product_id"],
                base_size=params["base_size"],
                limit_price=params["limit_price"]
            )
        )
    
    async def _tool_cancel_orders(self, params: dict) -> dict:
        """Handle cancel_orders tool call."""
        return self._process_response(
            self._client.cancel_orders(order_ids=params["order_ids"])
        )
    
    async def _tool_get_order(self, params: dict) -> dict:
        """Handle get_order tool call."""
        return self._process_response(
            self._client.get_order(params["order_id"])
        )
    
    async def _tool_list_orders(self, params: dict) -> dict:
        """Handle list_orders tool call."""
        kwargs = {}
        if "product_id" in params:
            kwargs["product_id"] = params["product_id"]
        if "order_status" in params:
            kwargs["order_status"] = params["order_status"]
        return self._process_response(self._client.list_orders(**kwargs))
    
    async def _tool_get_fills(self, params: dict) -> dict:
        """Handle get_fills tool call."""
        kwargs = {}
        if "product_id" in params:
            kwargs["product_id"] = params["product_id"]
        if "order_id" in params:
            kwargs["order_id"] = params["order_id"]
        return self._process_response(self._client.get_fills(**kwargs))
    
    async def _tool_get_portfolios(self, params: dict) -> dict:
        """Handle get_portfolios tool call."""
        kwargs = {}
        if "portfolio_type" in params:
            kwargs["portfolio_type"] = params["portfolio_type"]
        return self._process_response(self._client.get_portfolios(**kwargs))
    
    async def _tool_get_portfolio_breakdown(self, params: dict) -> dict:
        """Handle get_portfolio_breakdown tool call."""
        return self._process_response(
            self._client.get_portfolio_breakdown(params["portfolio_uuid"])
        )
    
    async def _tool_get_unix_time(self, params: dict) -> dict:
        """Handle get_unix_time tool call."""
        return self._process_response(self._client.get_unix_time())
    
    async def _tool_get_public_products(self, params: dict) -> dict:
        """Handle get_public_products tool call."""
        return self._process_response(self._client.get_public_products())
    
    async def _tool_get_public_product(self, params: dict) -> dict:
        """Handle get_public_product tool call."""
        return self._process_response(
            self._client.get_public_product(params["product_id"])
        )
    
    async def _tool_get_transaction_summary(self, params: dict) -> dict:
        """Handle get_transaction_summary tool call."""
        kwargs = {}
        if "start_date" in params:
            kwargs["start_date"] = params["start_date"]
        if "end_date" in params:
            kwargs["end_date"] = params["end_date"]
        return self._process_response(self._client.get_transaction_summary(**kwargs))
    
    # ===================
    # Utility Methods
    # ===================
    
    def _process_response(self, response: Any) -> dict[str, Any]:
        """Process API response into a consistent format.
        
        Args:
            response: Raw API response
        
        Returns:
            Processed response as dictionary
        """
        if response is None:
            return {}
        
        # Handle different response types
        if isinstance(response, dict):
            return response
        
        # If response has a to_dict method, use it
        if hasattr(response, "to_dict"):
            return response.to_dict()
        
        # If response has __dict__, convert it
        if hasattr(response, "__dict__"):
            return vars(response)
        
        # Try to convert to dict
        try:
            return dict(response)
        except (TypeError, ValueError):
            return {"data": response}
    
    def close_websocket(self) -> None:
        """Close the WebSocket connection."""
        if self._ws_client is not None:
            try:
                self._ws_client.close()
            except Exception:
                pass
            self._ws_client = None
