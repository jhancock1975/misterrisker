"""Schwab MCP Server - Exposes Schwab API as MCP tools.

This module provides a Model Context Protocol (MCP) server that wraps
the Schwab API using direct HTTP requests with refresh token authentication,
exposing all functionality as tools that can be used by AI agents.

Based on the schwabase implementation pattern for reliable token management.
"""

import asyncio
import os
import time
import datetime
import logging
from typing import Any

import requests

from .exceptions import SchwabAPIError, SchwabAuthError


# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger("schwab-mcp")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Configuration Constants
# =============================================================================

SCHWAB_TRADER_API_URL = "https://api.schwabapi.com/trader/v1"
SCHWAB_MARKETDATA_API_URL = "https://api.schwabapi.com/marketdata/v1"
SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
TOKEN_REFRESH_INTERVAL = 15 * 60  # Refresh every 15 minutes


class SchwabToolError(Exception):
    """Exception raised for MCP tool errors.
    
    Attributes:
        message: Error message describing what went wrong
        tool_name: Name of the tool that caused the error
    """
    
    def __init__(self, message: str, tool_name: str | None = None):
        """Initialize SchwabToolError.
        
        Args:
            message: Error message
            tool_name: Name of the tool that caused the error
        """
        self.message = message
        self.tool_name = tool_name
        super().__init__(self.message)


class SchwabMCPServer:
    """MCP Server that exposes Schwab API tools.
    
    This server wraps the Schwab API using direct HTTP requests with
    refresh token authentication, exposing all functionality as MCP-compatible
    tools that can be discovered and invoked by AI agents.
    
    Attributes:
        client_id: Schwab Client ID (API Key)
        client_secret: Schwab Client Secret (App Secret)
        refresh_token: Schwab OAuth Refresh Token
        _token: Current token state with access_token and expires_at
    """
    
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        api_key: str | None = None,  # Alias for client_id (backward compat)
        app_secret: str | None = None,  # Alias for client_secret (backward compat)
        callback_url: str | None = None,  # Not needed with refresh token
        token_path: str | None = None,  # Not needed with refresh token
    ):
        """Initialize the Schwab MCP Server.
        
        Args:
            client_id: Schwab Client ID (or use api_key alias)
            client_secret: Schwab Client Secret (or use app_secret alias)
            refresh_token: Schwab OAuth Refresh Token
            api_key: Alias for client_id (backward compatibility)
            app_secret: Alias for client_secret (backward compatibility)
            callback_url: Ignored (not needed with refresh token auth)
            token_path: Ignored (not needed with refresh token auth)
        """
        # Support both new and old parameter names
        self.client_id = client_id or api_key
        self.client_secret = client_secret or app_secret
        self.refresh_token = refresh_token
        
        # For backward compatibility
        self.api_key = self.client_id
        self.app_secret = self.client_secret
        self.callback_url = callback_url
        self.token_path = token_path
        
        # Token state
        self._token: dict[str, Any] = {
            "access_token": None,
            "expires_at": 0,  # Force refresh on first request
        }
        
        # Background refresh task handle
        self._refresh_task: asyncio.Task | None = None
        
        # Define tool schemas
        self._tool_schemas = self._define_tool_schemas()
    
    @classmethod
    def from_env(cls) -> "SchwabMCPServer":
        """Create a SchwabMCPServer from environment variables.
        
        Supports both new env var names (SCHWAB_CLIENT_ID) and old names (SCHWAB_API_KEY).
        
        Returns:
            SchwabMCPServer instance
        
        Raises:
            SchwabAuthError: If credentials are not found in environment
        """
        # Support both naming conventions
        client_id = os.getenv("SCHWAB_CLIENT_ID") or os.getenv("SCHWAB_API_KEY")
        client_secret = os.getenv("SCHWAB_CLIENT_SECRET") or os.getenv("SCHWAB_APP_SECRET")
        refresh_token = os.getenv("SCHWAB_REFRESH_TOKEN")
        callback_url = os.getenv("SCHWAB_CALLBACK_URL")
        token_path = os.getenv("SCHWAB_TOKEN_PATH")
        
        if not client_id:
            raise SchwabAuthError(
                message="SCHWAB_CLIENT_ID (or SCHWAB_API_KEY) environment variable not set",
                status_code=401,
                error_code="MISSING_CLIENT_ID"
            )
        
        if not client_secret:
            raise SchwabAuthError(
                message="SCHWAB_CLIENT_SECRET (or SCHWAB_APP_SECRET) environment variable not set",
                status_code=401,
                error_code="MISSING_CLIENT_SECRET"
            )
        
        if not refresh_token:
            raise SchwabAuthError(
                message="SCHWAB_REFRESH_TOKEN environment variable not set",
                status_code=401,
                error_code="MISSING_REFRESH_TOKEN"
            )
        
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
            callback_url=callback_url,
            token_path=token_path,
        )
    
    # ===================
    # Token Management
    # ===================
    
    def refresh_access_token(self) -> str:
        """Refresh the Schwab access token using the refresh token.
        
        Returns:
            The new access token
            
        Raises:
            SchwabAPIError: If token refresh fails
        """
        if not self.client_id or not self.client_secret or not self.refresh_token:
            raise SchwabAPIError(
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
            raise SchwabAPIError(f"Failed to refresh Schwab token: {response.text}")
        
        data = response.json()
        self._token["access_token"] = data["access_token"]
        self._token["expires_at"] = time.time() + data.get("expires_in", 1800) - 60
        logger.info(f"Schwab token refreshed, valid for {data.get('expires_in', 1800)}s")
        return data["access_token"]
    
    def get_access_token(self) -> str:
        """Get a valid Schwab access token, refreshing if expired.
        
        Returns:
            A valid access token
            
        Raises:
            SchwabAPIError: If no token is available and refresh fails
        """
        if self._token["expires_at"] < time.time():
            return self.refresh_access_token()
        if not self._token["access_token"]:
            raise SchwabAPIError("No access token available and refresh is not configured")
        return self._token["access_token"]
    
    def _schwab_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        api: str = "trader",
    ) -> Any:
        """Make an authenticated Schwab API request.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API path (e.g., /accounts)
            params: Query parameters
            body: Request body for POST/PUT
            api: API type - 'trader' or 'marketdata'
            
        Returns:
            Parsed JSON response
            
        Raises:
            SchwabAPIError: If the API call fails
        """
        token = self.get_access_token()
        base_url = SCHWAB_MARKETDATA_API_URL if api == "marketdata" else SCHWAB_TRADER_API_URL
        url = f"{base_url}{path}"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        
        response = requests.request(
            method.upper(),
            url,
            headers=headers,
            params=params,
            json=body,
            timeout=30,
        )
        
        if not response.ok:
            raise SchwabAPIError(f"Schwab API error: {response.text}")
        
        if not response.text:
            return {}
        
        try:
            return response.json()
        except ValueError:
            return {"raw": response.text}
    
    async def start_background_refresh(self) -> None:
        """Start the background token refresh task.
        
        This should be called when the server starts to ensure
        the token stays fresh.
        """
        if self.client_id and self.client_secret and self.refresh_token:
            try:
                self.refresh_access_token()
                logger.info("Initial Schwab token refresh completed")
            except Exception as exc:
                logger.warning(f"Initial Schwab token refresh failed: {exc}")
            
            async def periodic_refresh():
                while True:
                    await asyncio.sleep(TOKEN_REFRESH_INTERVAL)
                    try:
                        self.refresh_access_token()
                        logger.info("Schwab token auto-refreshed")
                    except Exception as exc:
                        logger.warning(f"Periodic Schwab refresh failed: {exc}")
            
            self._refresh_task = asyncio.create_task(periodic_refresh())
            logger.info(f"Background Schwab refresh scheduled every {TOKEN_REFRESH_INTERVAL // 60} minutes")
    
    async def stop_background_refresh(self) -> None:
        """Stop the background token refresh task."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
    
    def _define_tool_schemas(self) -> list[dict[str, Any]]:
        """Define all MCP tool schemas.
        
        Returns:
            List of tool schema dictionaries.
        """
        return [
            # =================== Account Tools ===================
            {
                "name": "get_account_numbers",
                "description": "Get account numbers and their corresponding hash values. The hash values are required for most account-specific operations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_account",
                "description": "Get details for a specific account including balances and optionally positions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value from get_account_numbers"},
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional fields to include (e.g., ['positions'])"
                        }
                    },
                    "required": ["account_hash"]
                }
            },
            {
                "name": "get_accounts",
                "description": "Get details for all linked accounts.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional fields to include (e.g., ['positions'])"
                        }
                    }
                }
            },
            # =================== Order Tools ===================
            {
                "name": "get_orders_for_account",
                "description": "Get orders for a specific account with optional filters.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "max_results": {"type": "integer", "description": "Maximum number of orders to return"},
                        "from_entered_datetime": {"type": "string", "description": "Start datetime (ISO format)"},
                        "to_entered_datetime": {"type": "string", "description": "End datetime (ISO format)"},
                        "status": {"type": "string", "description": "Filter by order status (FILLED, WORKING, etc.)"}
                    },
                    "required": ["account_hash"]
                }
            },
            {
                "name": "get_orders_for_all_linked_accounts",
                "description": "Get orders for all linked accounts.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_results": {"type": "integer", "description": "Maximum number of orders to return"},
                        "from_entered_datetime": {"type": "string", "description": "Start datetime (ISO format)"},
                        "to_entered_datetime": {"type": "string", "description": "End datetime (ISO format)"},
                        "status": {"type": "string", "description": "Filter by order status"}
                    }
                }
            },
            {
                "name": "get_order",
                "description": "Get details for a specific order.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "order_id": {"type": "integer", "description": "Order ID"}
                    },
                    "required": ["account_hash", "order_id"]
                }
            },
            {
                "name": "place_order",
                "description": "Place an order using a complete order specification.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "order_spec": {"type": "object", "description": "Complete order specification"}
                    },
                    "required": ["account_hash", "order_spec"]
                }
            },
            {
                "name": "replace_order",
                "description": "Replace an existing order with a new one.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "order_id": {"type": "integer", "description": "Order ID to replace"},
                        "order_spec": {"type": "object", "description": "New order specification"}
                    },
                    "required": ["account_hash", "order_id", "order_spec"]
                }
            },
            {
                "name": "cancel_order",
                "description": "Cancel an existing order.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "order_id": {"type": "integer", "description": "Order ID to cancel"}
                    },
                    "required": ["account_hash", "order_id"]
                }
            },
            {
                "name": "preview_order",
                "description": "Preview an order before placing it.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "order_spec": {"type": "object", "description": "Order specification to preview"}
                    },
                    "required": ["account_hash", "order_spec"]
                }
            },
            # =================== Quote Tools ===================
            {
                "name": "get_quote",
                "description": "Get quote for a single symbol.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fields to include: quote, fundamental, extended, reference, regular"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_quotes",
                "description": "Get quotes for multiple symbols.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stock symbols"
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fields to include"
                        },
                        "indicative": {"type": "boolean", "description": "Include indicative quotes"}
                    },
                    "required": ["symbols"]
                }
            },
            # =================== Option Tools ===================
            {
                "name": "get_option_chain",
                "description": "Get option chain for a symbol.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Underlying symbol"},
                        "contract_type": {"type": "string", "description": "CALL, PUT, or ALL"},
                        "strike_count": {"type": "integer", "description": "Number of strikes above/below ATM"},
                        "include_underlying_quote": {"type": "boolean", "description": "Include underlying quote"},
                        "strategy": {"type": "string", "description": "Option strategy type"},
                        "strike": {"type": "number", "description": "Specific strike price"},
                        "strike_range": {"type": "string", "description": "ITM, NTM, OTM, etc."},
                        "from_date": {"type": "string", "description": "Start date for expiration filter"},
                        "to_date": {"type": "string", "description": "End date for expiration filter"},
                        "exp_month": {"type": "string", "description": "Expiration month filter"},
                        "option_type": {"type": "string", "description": "STANDARD, NON_STANDARD, or ALL"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_option_expiration_chain",
                "description": "Get option expiration dates for a symbol.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Underlying symbol"}
                    },
                    "required": ["symbol"]
                }
            },
            # =================== Price History Tools ===================
            {
                "name": "get_price_history",
                "description": "Get historical price data for a symbol.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "period_type": {"type": "string", "description": "day, month, year, ytd"},
                        "period": {"type": "integer", "description": "Number of periods"},
                        "frequency_type": {"type": "string", "description": "minute, daily, weekly, monthly"},
                        "frequency": {"type": "integer", "description": "Frequency value"},
                        "start_datetime": {"type": "string", "description": "Start datetime (ISO format)"},
                        "end_datetime": {"type": "string", "description": "End datetime (ISO format)"},
                        "need_extended_hours_data": {"type": "boolean", "description": "Include extended hours"},
                        "need_previous_close": {"type": "boolean", "description": "Include previous close"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_price_history_every_minute",
                "description": "Get per-minute price history (up to ~48 days).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "start_datetime": {"type": "string", "description": "Start datetime"},
                        "end_datetime": {"type": "string", "description": "End datetime"},
                        "need_extended_hours_data": {"type": "boolean"},
                        "need_previous_close": {"type": "boolean"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_price_history_every_five_minutes",
                "description": "Get per-5-minute price history (~9 months).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "start_datetime": {"type": "string"},
                        "end_datetime": {"type": "string"},
                        "need_extended_hours_data": {"type": "boolean"},
                        "need_previous_close": {"type": "boolean"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_price_history_every_ten_minutes",
                "description": "Get per-10-minute price history (~9 months).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "start_datetime": {"type": "string"},
                        "end_datetime": {"type": "string"},
                        "need_extended_hours_data": {"type": "boolean"},
                        "need_previous_close": {"type": "boolean"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_price_history_every_fifteen_minutes",
                "description": "Get per-15-minute price history (~9 months).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "start_datetime": {"type": "string"},
                        "end_datetime": {"type": "string"},
                        "need_extended_hours_data": {"type": "boolean"},
                        "need_previous_close": {"type": "boolean"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_price_history_every_thirty_minutes",
                "description": "Get per-30-minute price history (~9 months).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "start_datetime": {"type": "string"},
                        "end_datetime": {"type": "string"},
                        "need_extended_hours_data": {"type": "boolean"},
                        "need_previous_close": {"type": "boolean"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_price_history_every_day",
                "description": "Get daily price history (20+ years for some symbols).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "start_datetime": {"type": "string"},
                        "end_datetime": {"type": "string"},
                        "need_extended_hours_data": {"type": "boolean"},
                        "need_previous_close": {"type": "boolean"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_price_history_every_week",
                "description": "Get weekly price history (20+ years).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "start_datetime": {"type": "string"},
                        "end_datetime": {"type": "string"},
                        "need_extended_hours_data": {"type": "boolean"},
                        "need_previous_close": {"type": "boolean"}
                    },
                    "required": ["symbol"]
                }
            },
            # =================== Market Data Tools ===================
            {
                "name": "get_movers",
                "description": "Get market movers for an index.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "string", "description": "Index symbol ($DJI, $COMPX, $SPX, etc.)"},
                        "sort_order": {"type": "string", "description": "VOLUME, TRADES, PERCENT_CHANGE_UP, PERCENT_CHANGE_DOWN"},
                        "frequency": {"type": "integer", "description": "0, 1, 5, 10, 30, or 60"}
                    },
                    "required": ["index"]
                }
            },
            {
                "name": "get_market_hours",
                "description": "Get market hours for specified markets.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "markets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Markets: equity, option, future, forex, bond"
                        },
                        "date": {"type": "string", "description": "Date to check (YYYY-MM-DD)"}
                    },
                    "required": ["markets"]
                }
            },
            # =================== Instrument Tools ===================
            {
                "name": "get_instruments",
                "description": "Search for instruments.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol or search term"},
                        "projection": {
                            "type": "string",
                            "description": "symbol-search, symbol-regex, desc-search, desc-regex, search, fundamental"
                        }
                    },
                    "required": ["symbol", "projection"]
                }
            },
            {
                "name": "get_instrument_by_cusip",
                "description": "Get instrument by CUSIP identifier.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "cusip": {"type": "string", "description": "CUSIP identifier"}
                    },
                    "required": ["cusip"]
                }
            },
            # =================== Transaction Tools ===================
            {
                "name": "get_transactions",
                "description": "Get transaction history for an account.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "start_date": {"type": "string", "description": "Start date (ISO format)"},
                        "end_date": {"type": "string", "description": "End date (ISO format)"},
                        "transaction_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Transaction types: TRADE, DIVIDEND_OR_INTEREST, etc."
                        },
                        "symbol": {"type": "string", "description": "Filter by symbol"}
                    },
                    "required": ["account_hash"]
                }
            },
            {
                "name": "get_transaction",
                "description": "Get a specific transaction.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "transaction_id": {"type": "integer", "description": "Transaction ID"}
                    },
                    "required": ["account_hash", "transaction_id"]
                }
            },
            # =================== User Tools ===================
            {
                "name": "get_user_preferences",
                "description": "Get user preferences and account information.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            # =================== Equity Order Helpers ===================
            {
                "name": "equity_buy_market",
                "description": "Place a market buy order for equity.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "quantity": {"type": "integer", "description": "Number of shares"}
                    },
                    "required": ["account_hash", "symbol", "quantity"]
                }
            },
            {
                "name": "equity_buy_limit",
                "description": "Place a limit buy order for equity.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "quantity": {"type": "integer", "description": "Number of shares"},
                        "price": {"type": "string", "description": "Limit price"}
                    },
                    "required": ["account_hash", "symbol", "quantity", "price"]
                }
            },
            {
                "name": "equity_sell_market",
                "description": "Place a market sell order for equity.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "quantity": {"type": "integer", "description": "Number of shares"}
                    },
                    "required": ["account_hash", "symbol", "quantity"]
                }
            },
            {
                "name": "equity_sell_limit",
                "description": "Place a limit sell order for equity.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "quantity": {"type": "integer", "description": "Number of shares"},
                        "price": {"type": "string", "description": "Limit price"}
                    },
                    "required": ["account_hash", "symbol", "quantity", "price"]
                }
            },
            # =================== Option Order Helpers ===================
            {
                "name": "option_buy_to_open_market",
                "description": "Place a market buy-to-open order for options.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "symbol": {"type": "string", "description": "Option symbol"},
                        "quantity": {"type": "integer", "description": "Number of contracts"}
                    },
                    "required": ["account_hash", "symbol", "quantity"]
                }
            },
            {
                "name": "option_buy_to_open_limit",
                "description": "Place a limit buy-to-open order for options.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "symbol": {"type": "string", "description": "Option symbol"},
                        "quantity": {"type": "integer", "description": "Number of contracts"},
                        "price": {"type": "string", "description": "Limit price"}
                    },
                    "required": ["account_hash", "symbol", "quantity", "price"]
                }
            },
            {
                "name": "option_sell_to_close_market",
                "description": "Place a market sell-to-close order for options.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "symbol": {"type": "string", "description": "Option symbol"},
                        "quantity": {"type": "integer", "description": "Number of contracts"}
                    },
                    "required": ["account_hash", "symbol", "quantity"]
                }
            },
            {
                "name": "option_sell_to_close_limit",
                "description": "Place a limit sell-to-close order for options.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account_hash": {"type": "string", "description": "Account hash value"},
                        "symbol": {"type": "string", "description": "Option symbol"},
                        "quantity": {"type": "integer", "description": "Number of contracts"},
                        "price": {"type": "string", "description": "Limit price"}
                    },
                    "required": ["account_hash", "symbol", "quantity", "price"]
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
    ) -> dict[str, Any] | list:
        """Call an MCP tool by name.
        
        Args:
            tool_name: Name of the tool to call
            params: Parameters to pass to the tool
        
        Returns:
            Result from the tool call
        
        Raises:
            SchwabToolError: If tool_name is not found or required params missing
            SchwabAPIError: If the API call fails
        """
        # Check if tool exists
        tool_names = [t["name"] for t in self._tool_schemas]
        if tool_name not in tool_names:
            raise SchwabToolError(f"Unknown tool: {tool_name}", tool_name=tool_name)
        
        # Get tool schema for validation
        tool_schema = next(t for t in self._tool_schemas if t["name"] == tool_name)
        
        # Validate required parameters
        required_params = tool_schema["inputSchema"].get("required", [])
        for param in required_params:
            if param not in params:
                raise SchwabToolError(
                    f"Missing required parameter '{param}' for tool '{tool_name}'",
                    tool_name=tool_name
                )
        
        # Route to appropriate handler
        try:
            handler = getattr(self, f"_tool_{tool_name}")
            return await handler(params)
        except SchwabAPIError:
            raise
        except SchwabToolError:
            raise
        except Exception as e:
            raise SchwabAPIError(f"API Error: {str(e)}")
    
    # ===================
    # Account Tool Handlers
    # ===================
    
    async def _tool_get_account_numbers(self, params: dict) -> list:
        """Handle get_account_numbers tool call."""
        return self._schwab_request("GET", "/accounts/accountNumbers")
    
    async def _tool_get_account(self, params: dict) -> dict:
        """Handle get_account tool call."""
        account_hash = params["account_hash"]
        query_params = {}
        if "fields" in params and params["fields"]:
            query_params["fields"] = ",".join(params["fields"])
        return self._schwab_request("GET", f"/accounts/{account_hash}", params=query_params)
    
    async def _tool_get_accounts(self, params: dict) -> list:
        """Handle get_accounts tool call."""
        query_params = {}
        if "fields" in params and params["fields"]:
            query_params["fields"] = ",".join(params["fields"])
        return self._schwab_request("GET", "/accounts", params=query_params)
    
    # ===================
    # Order Tool Handlers
    # ===================
    
    async def _tool_get_orders_for_account(self, params: dict) -> list:
        """Handle get_orders_for_account tool call."""
        account_hash = params["account_hash"]
        query_params = {}
        if "max_results" in params:
            query_params["maxResults"] = params["max_results"]
        if "from_entered_datetime" in params:
            query_params["fromEnteredTime"] = params["from_entered_datetime"]
        if "to_entered_datetime" in params:
            query_params["toEnteredTime"] = params["to_entered_datetime"]
        if "status" in params:
            query_params["status"] = params["status"]
        return self._schwab_request("GET", f"/accounts/{account_hash}/orders", params=query_params)
    
    async def _tool_get_orders_for_all_linked_accounts(self, params: dict) -> list:
        """Handle get_orders_for_all_linked_accounts tool call."""
        query_params = {}
        if "max_results" in params:
            query_params["maxResults"] = params["max_results"]
        if "from_entered_datetime" in params:
            query_params["fromEnteredTime"] = params["from_entered_datetime"]
        if "to_entered_datetime" in params:
            query_params["toEnteredTime"] = params["to_entered_datetime"]
        if "status" in params:
            query_params["status"] = params["status"]
        return self._schwab_request("GET", "/orders", params=query_params)
    
    async def _tool_get_order(self, params: dict) -> dict:
        """Handle get_order tool call."""
        account_hash = params["account_hash"]
        order_id = params["order_id"]
        return self._schwab_request("GET", f"/accounts/{account_hash}/orders/{order_id}")
    
    async def _tool_place_order(self, params: dict) -> dict:
        """Handle place_order tool call."""
        account_hash = params["account_hash"]
        order_spec = params["order_spec"]
        result = self._schwab_request("POST", f"/accounts/{account_hash}/orders", body=order_spec)
        return {"success": True, "result": result}
    
    async def _tool_replace_order(self, params: dict) -> dict:
        """Handle replace_order tool call."""
        account_hash = params["account_hash"]
        order_id = params["order_id"]
        order_spec = params["order_spec"]
        result = self._schwab_request("PUT", f"/accounts/{account_hash}/orders/{order_id}", body=order_spec)
        return {"success": True, "result": result}
    
    async def _tool_cancel_order(self, params: dict) -> dict:
        """Handle cancel_order tool call."""
        account_hash = params["account_hash"]
        order_id = params["order_id"]
        self._schwab_request("DELETE", f"/accounts/{account_hash}/orders/{order_id}")
        return {"success": True}
    
    async def _tool_preview_order(self, params: dict) -> dict:
        """Handle preview_order tool call."""
        account_hash = params["account_hash"]
        order_spec = params["order_spec"]
        return self._schwab_request("POST", f"/accounts/{account_hash}/previewOrder", body=order_spec)
    
    # ===================
    # Quote Tool Handlers
    # ===================
    
    async def _tool_get_quote(self, params: dict) -> dict:
        """Handle get_quote tool call."""
        symbol = params["symbol"]
        query_params = {}
        if "fields" in params and params["fields"]:
            query_params["fields"] = ",".join(params["fields"])
        return self._schwab_request("GET", f"/{symbol}/quotes", params=query_params, api="marketdata")
    
    async def _tool_get_quotes(self, params: dict) -> dict:
        """Handle get_quotes tool call."""
        query_params = {"symbols": ",".join(params["symbols"])}
        if "fields" in params and params["fields"]:
            query_params["fields"] = ",".join(params["fields"])
        if "indicative" in params:
            query_params["indicative"] = str(params["indicative"]).lower()
        return self._schwab_request("GET", "/quotes", params=query_params, api="marketdata")
    
    # ===================
    # Option Tool Handlers
    # ===================
    
    async def _tool_get_option_chain(self, params: dict) -> dict:
        """Handle get_option_chain tool call."""
        query_params = {"symbol": params["symbol"]}
        
        if "contract_type" in params:
            query_params["contractType"] = params["contract_type"].upper()
        if "strike_count" in params:
            query_params["strikeCount"] = params["strike_count"]
        if "include_underlying_quote" in params:
            query_params["includeUnderlyingQuote"] = str(params["include_underlying_quote"]).lower()
        if "strategy" in params:
            query_params["strategy"] = params["strategy"]
        if "strike" in params:
            query_params["strike"] = params["strike"]
        if "strike_range" in params:
            query_params["range"] = params["strike_range"]
        if "from_date" in params:
            query_params["fromDate"] = params["from_date"]
        if "to_date" in params:
            query_params["toDate"] = params["to_date"]
        if "exp_month" in params:
            query_params["expMonth"] = params["exp_month"]
        if "option_type" in params:
            query_params["optionType"] = params["option_type"]
        
        return self._schwab_request("GET", "/chains", params=query_params, api="marketdata")
    
    async def _tool_get_option_expiration_chain(self, params: dict) -> dict:
        """Handle get_option_expiration_chain tool call."""
        query_params = {"symbol": params["symbol"]}
        return self._schwab_request("GET", "/expirationchain", params=query_params, api="marketdata")
    
    # ===================
    # Price History Tool Handlers
    # ===================
    
    async def _tool_get_price_history(self, params: dict) -> dict:
        """Handle get_price_history tool call."""
        symbol = params["symbol"]
        query_params = {}
        
        if "period_type" in params:
            query_params["periodType"] = params["period_type"]
        if "period" in params:
            query_params["period"] = params["period"]
        if "frequency_type" in params:
            query_params["frequencyType"] = params["frequency_type"]
        if "frequency" in params:
            query_params["frequency"] = params["frequency"]
        if "start_datetime" in params:
            query_params["startDate"] = params["start_datetime"]
        if "end_datetime" in params:
            query_params["endDate"] = params["end_datetime"]
        if "need_extended_hours_data" in params:
            query_params["needExtendedHoursData"] = str(params["need_extended_hours_data"]).lower()
        if "need_previous_close" in params:
            query_params["needPreviousClose"] = str(params["need_previous_close"]).lower()
        
        return self._schwab_request("GET", f"/pricehistory", params={"symbol": symbol, **query_params}, api="marketdata")
    
    def _build_price_history_params(self, params: dict) -> dict:
        """Build common price history query parameters."""
        query_params = {"symbol": params["symbol"]}
        if "start_datetime" in params:
            query_params["startDate"] = params["start_datetime"]
        if "end_datetime" in params:
            query_params["endDate"] = params["end_datetime"]
        if "need_extended_hours_data" in params:
            query_params["needExtendedHoursData"] = str(params["need_extended_hours_data"]).lower()
        if "need_previous_close" in params:
            query_params["needPreviousClose"] = str(params["need_previous_close"]).lower()
        return query_params
    
    async def _tool_get_price_history_every_minute(self, params: dict) -> dict:
        """Handle get_price_history_every_minute tool call."""
        query_params = self._build_price_history_params(params)
        query_params["periodType"] = "day"
        query_params["frequencyType"] = "minute"
        query_params["frequency"] = 1
        return self._schwab_request("GET", "/pricehistory", params=query_params, api="marketdata")
    
    async def _tool_get_price_history_every_five_minutes(self, params: dict) -> dict:
        """Handle get_price_history_every_five_minutes tool call."""
        query_params = self._build_price_history_params(params)
        query_params["periodType"] = "day"
        query_params["frequencyType"] = "minute"
        query_params["frequency"] = 5
        return self._schwab_request("GET", "/pricehistory", params=query_params, api="marketdata")
    
    async def _tool_get_price_history_every_ten_minutes(self, params: dict) -> dict:
        """Handle get_price_history_every_ten_minutes tool call."""
        query_params = self._build_price_history_params(params)
        query_params["periodType"] = "day"
        query_params["frequencyType"] = "minute"
        query_params["frequency"] = 10
        return self._schwab_request("GET", "/pricehistory", params=query_params, api="marketdata")
    
    async def _tool_get_price_history_every_fifteen_minutes(self, params: dict) -> dict:
        """Handle get_price_history_every_fifteen_minutes tool call."""
        query_params = self._build_price_history_params(params)
        query_params["periodType"] = "day"
        query_params["frequencyType"] = "minute"
        query_params["frequency"] = 15
        return self._schwab_request("GET", "/pricehistory", params=query_params, api="marketdata")
    
    async def _tool_get_price_history_every_thirty_minutes(self, params: dict) -> dict:
        """Handle get_price_history_every_thirty_minutes tool call."""
        query_params = self._build_price_history_params(params)
        query_params["periodType"] = "day"
        query_params["frequencyType"] = "minute"
        query_params["frequency"] = 30
        return self._schwab_request("GET", "/pricehistory", params=query_params, api="marketdata")
    
    async def _tool_get_price_history_every_day(self, params: dict) -> dict:
        """Handle get_price_history_every_day tool call."""
        query_params = self._build_price_history_params(params)
        query_params["periodType"] = "year"
        query_params["frequencyType"] = "daily"
        query_params["frequency"] = 1
        return self._schwab_request("GET", "/pricehistory", params=query_params, api="marketdata")
    
    async def _tool_get_price_history_every_week(self, params: dict) -> dict:
        """Handle get_price_history_every_week tool call."""
        query_params = self._build_price_history_params(params)
        query_params["periodType"] = "year"
        query_params["frequencyType"] = "weekly"
        query_params["frequency"] = 1
        return self._schwab_request("GET", "/pricehistory", params=query_params, api="marketdata")
    
    # ===================
    # Market Data Tool Handlers
    # ===================
    
    async def _tool_get_movers(self, params: dict) -> dict:
        """Handle get_movers tool call."""
        index = params["index"]
        query_params = {}
        if "sort_order" in params:
            query_params["sort"] = params["sort_order"]
        if "frequency" in params:
            query_params["frequency"] = params["frequency"]
        return self._schwab_request("GET", f"/movers/{index}", params=query_params, api="marketdata")
    
    async def _tool_get_market_hours(self, params: dict) -> dict:
        """Handle get_market_hours tool call."""
        query_params = {"markets": ",".join(params["markets"])}
        if "date" in params:
            query_params["date"] = params["date"]
        return self._schwab_request("GET", "/markets", params=query_params, api="marketdata")
    
    # ===================
    # Instrument Tool Handlers
    # ===================
    
    async def _tool_get_instruments(self, params: dict) -> dict:
        """Handle get_instruments tool call."""
        query_params = {
            "symbol": params["symbol"],
            "projection": params["projection"]
        }
        return self._schwab_request("GET", "/instruments", params=query_params, api="marketdata")
    
    async def _tool_get_instrument_by_cusip(self, params: dict) -> dict:
        """Handle get_instrument_by_cusip tool call."""
        cusip = params["cusip"]
        return self._schwab_request("GET", f"/instruments/{cusip}", api="marketdata")
    
    # ===================
    # Transaction Tool Handlers
    # ===================
    
    async def _tool_get_transactions(self, params: dict) -> list:
        """Handle get_transactions tool call."""
        account_hash = params["account_hash"]
        query_params = {}
        if "start_date" in params:
            query_params["startDate"] = params["start_date"]
        if "end_date" in params:
            query_params["endDate"] = params["end_date"]
        if "symbol" in params:
            query_params["symbol"] = params["symbol"]
        if "transaction_types" in params:
            query_params["types"] = ",".join(params["transaction_types"])
        return self._schwab_request("GET", f"/accounts/{account_hash}/transactions", params=query_params)
    
    async def _tool_get_transaction(self, params: dict) -> dict:
        """Handle get_transaction tool call."""
        account_hash = params["account_hash"]
        transaction_id = params["transaction_id"]
        return self._schwab_request("GET", f"/accounts/{account_hash}/transactions/{transaction_id}")
    
    # ===================
    # User Tool Handlers
    # ===================
    
    async def _tool_get_user_preferences(self, params: dict) -> dict:
        """Handle get_user_preferences tool call."""
        return self._schwab_request("GET", "/userPreference")
    
    # ===================
    # Equity Order Helper Handlers
    # ===================
    
    def _build_equity_order(
        self,
        symbol: str,
        quantity: int,
        instruction: str,
        order_type: str,
        price: float | None = None
    ) -> dict:
        """Build an equity order specification.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            instruction: BUY or SELL
            order_type: MARKET or LIMIT
            price: Limit price (required for LIMIT orders)
            
        Returns:
            Order specification dict
        """
        order = {
            "orderType": order_type,
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "EQUITY"
                    }
                }
            ]
        }
        if price is not None:
            order["price"] = str(price)
        return order
    
    async def _tool_equity_buy_market(self, params: dict) -> dict:
        """Handle equity_buy_market tool call."""
        order = self._build_equity_order(
            params["symbol"], params["quantity"], "BUY", "MARKET"
        )
        result = self._schwab_request(
            "POST", f"/accounts/{params['account_hash']}/orders", body=order
        )
        return {"success": True, "result": result}
    
    async def _tool_equity_buy_limit(self, params: dict) -> dict:
        """Handle equity_buy_limit tool call."""
        order = self._build_equity_order(
            params["symbol"], params["quantity"], "BUY", "LIMIT", float(params["price"])
        )
        result = self._schwab_request(
            "POST", f"/accounts/{params['account_hash']}/orders", body=order
        )
        return {"success": True, "result": result}
    
    async def _tool_equity_sell_market(self, params: dict) -> dict:
        """Handle equity_sell_market tool call."""
        order = self._build_equity_order(
            params["symbol"], params["quantity"], "SELL", "MARKET"
        )
        result = self._schwab_request(
            "POST", f"/accounts/{params['account_hash']}/orders", body=order
        )
        return {"success": True, "result": result}
    
    async def _tool_equity_sell_limit(self, params: dict) -> dict:
        """Handle equity_sell_limit tool call."""
        order = self._build_equity_order(
            params["symbol"], params["quantity"], "SELL", "LIMIT", float(params["price"])
        )
        result = self._schwab_request(
            "POST", f"/accounts/{params['account_hash']}/orders", body=order
        )
        return {"success": True, "result": result}
    
    # ===================
    # Option Order Helper Handlers
    # ===================
    
    def _build_option_order(
        self,
        symbol: str,
        quantity: int,
        instruction: str,
        order_type: str,
        price: float | None = None
    ) -> dict:
        """Build an option order specification.
        
        Args:
            symbol: Option symbol
            quantity: Number of contracts
            instruction: BUY_TO_OPEN, SELL_TO_CLOSE, etc.
            order_type: MARKET or LIMIT
            price: Limit price (required for LIMIT orders)
            
        Returns:
            Order specification dict
        """
        order = {
            "orderType": order_type,
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "OPTION"
                    }
                }
            ]
        }
        if price is not None:
            order["price"] = str(price)
        return order
    
    async def _tool_option_buy_to_open_market(self, params: dict) -> dict:
        """Handle option_buy_to_open_market tool call."""
        order = self._build_option_order(
            params["symbol"], params["quantity"], "BUY_TO_OPEN", "MARKET"
        )
        result = self._schwab_request(
            "POST", f"/accounts/{params['account_hash']}/orders", body=order
        )
        return {"success": True, "result": result}
    
    async def _tool_option_buy_to_open_limit(self, params: dict) -> dict:
        """Handle option_buy_to_open_limit tool call."""
        order = self._build_option_order(
            params["symbol"], params["quantity"], "BUY_TO_OPEN", "LIMIT", float(params["price"])
        )
        result = self._schwab_request(
            "POST", f"/accounts/{params['account_hash']}/orders", body=order
        )
        return {"success": True, "result": result}
    
    async def _tool_option_sell_to_close_market(self, params: dict) -> dict:
        """Handle option_sell_to_close_market tool call."""
        order = self._build_option_order(
            params["symbol"], params["quantity"], "SELL_TO_CLOSE", "MARKET"
        )
        result = self._schwab_request(
            "POST", f"/accounts/{params['account_hash']}/orders", body=order
        )
        return {"success": True, "result": result}
    
    async def _tool_option_sell_to_close_limit(self, params: dict) -> dict:
        """Handle option_sell_to_close_limit tool call."""
        order = self._build_option_order(
            params["symbol"], params["quantity"], "SELL_TO_CLOSE", "LIMIT", float(params["price"])
        )
        result = self._schwab_request(
            "POST", f"/accounts/{params['account_hash']}/orders", body=order
        )
        return {"success": True, "result": result}
    
    # ===================
    # Utility Methods
    # ===================
    
    def build_option_symbol(
        self,
        underlying: str,
        expiration_date: str,
        contract_type: str,
        strike_price: float
    ) -> str:
        """Build an option symbol from components.
        
        Args:
            underlying: Underlying stock symbol
            expiration_date: Expiration date (YYYY-MM-DD or YYMMDD)
            contract_type: CALL or PUT
            strike_price: Strike price
        
        Returns:
            Formatted option symbol (OCC format)
        """
        # Pad underlying to 6 chars
        underlying_padded = underlying.upper().ljust(6)
        
        # Convert date format if needed
        if "-" in expiration_date:
            date_obj = datetime.date.fromisoformat(expiration_date)
            exp_str = date_obj.strftime("%y%m%d")
        else:
            exp_str = expiration_date
        
        # Contract type character
        contract_char = "C" if contract_type.upper() == "CALL" else "P"
        
        # Strike price: multiply by 1000 and pad to 8 digits
        strike_int = int(strike_price * 1000)
        strike_str = str(strike_int).zfill(8)
        
        return f"{underlying_padded}{exp_str}{contract_char}{strike_str}"
    
    def parse_option_symbol(self, symbol: str) -> dict[str, Any]:
        """Parse an option symbol into components.
        
        Args:
            symbol: Option symbol (OCC format, e.g., 'AAPL  240119C00175000')
        
        Returns:
            Dictionary with underlying, expiration_date, contract_type, strike_price
        """
        # OCC format: SYMBOL(6) + DATE(6) + TYPE(1) + STRIKE(8)
        # Strip whitespace and normalize
        symbol = symbol.strip()
        
        # Extract components
        underlying = symbol[:6].strip()
        exp_date = symbol[6:12]
        contract_type = "CALL" if symbol[12] == "C" else "PUT"
        strike_raw = int(symbol[13:21])
        strike_price = strike_raw / 1000.0
        
        # Format expiration date
        exp_formatted = f"20{exp_date[:2]}-{exp_date[2:4]}-{exp_date[4:6]}"
        
        return {
            "underlying": underlying,
            "expiration_date": exp_formatted,
            "contract_type": contract_type,
            "strike_price": strike_price
        }
