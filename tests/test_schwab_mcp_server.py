"""
Tests for the Schwab MCP Server.

These tests follow TDD principles - written before the implementation.
They cover all Schwab API tools exposed via MCP.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Dict, Any
import datetime
import json
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_token_response():
    """Create a mock token refresh response."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "access_token": "mock_access_token_123",
        "expires_in": 1800,
        "token_type": "Bearer",
        "scope": "api"
    }
    return mock_response


@pytest.fixture
def mock_api_responses():
    """Create mock API responses for various endpoints."""
    responses = {
        # Account responses
        "accounts/accountNumbers": [
            {"accountNumber": "12345678", "hashValue": "ABC123HASH"}
        ],
        "accounts/ABC123HASH": {
            "securitiesAccount": {
                "accountNumber": "12345678",
                "type": "MARGIN",
                "currentBalances": {
                    "cashBalance": 10000.00,
                    "availableFunds": 9500.00,
                    "buyingPower": 19000.00
                },
                "positions": [
                    {
                        "symbol": "AAPL",
                        "longQuantity": 100,
                        "averagePrice": 150.00,
                        "marketValue": 17500.00
                    }
                ]
            }
        },
        "accounts": [
            {
                "securitiesAccount": {
                    "accountNumber": "12345678",
                    "type": "MARGIN",
                    "currentBalances": {"cashBalance": 10000.00}
                }
            }
        ],
        # Order responses
        "orders": [
            {"orderId": 123456789, "status": "WORKING"}
        ],
        "accounts/ABC123HASH/orders": [
            {
                "orderId": 123456789,
                "status": "FILLED",
                "orderType": "LIMIT",
                "price": 150.00,
                "quantity": 10,
                "filledQuantity": 10
            }
        ],
        "accounts/ABC123HASH/orders/123456789": {
            "orderId": 123456789,
            "status": "FILLED",
            "orderType": "LIMIT"
        },
        # Quote responses
        "AAPL/quotes": {
            "AAPL": {
                "symbol": "AAPL",
                "lastPrice": 175.50,
                "bidPrice": 175.45,
                "askPrice": 175.55,
                "volume": 50000000
            }
        },
        "quotes": {
            "AAPL": {"symbol": "AAPL", "lastPrice": 175.50},
            "GOOGL": {"symbol": "GOOGL", "lastPrice": 140.25}
        },
        # Option responses
        "chains": {
            "symbol": "AAPL",
            "status": "SUCCESS",
            "callExpDateMap": {},
            "putExpDateMap": {}
        },
        "expirationchain": {
            "expirationList": [
                {"expirationDate": "2024-01-19"},
                {"expirationDate": "2024-01-26"}
            ]
        },
        # Price history
        "pricehistory": {
            "symbol": "AAPL",
            "candles": [
                {"open": 170.0, "high": 175.0, "low": 169.0, "close": 174.0, "volume": 1000000}
            ]
        },
        # Movers
        "movers/$DJI": {
            "screeners": [
                {"symbol": "AAPL", "totalVolume": 50000000}
            ]
        },
        # Market hours
        "markets": {
            "equity": {
                "marketType": "EQUITY",
                "isOpen": True
            }
        },
        # Instruments
        "instruments": {
            "instruments": [
                {"symbol": "AAPL", "description": "Apple Inc."}
            ]
        },
        "instruments/037833100": {
            "symbol": "AAPL",
            "description": "Apple Inc.",
            "cusip": "037833100"
        },
        # Transactions
        "accounts/ABC123HASH/transactions": [
            {"transactionId": 999, "type": "TRADE"}
        ],
        "accounts/ABC123HASH/transactions/999": {
            "transactionId": 999,
            "type": "TRADE",
            "amount": 1750.00
        },
        # User preferences
        "userPreference": {
            "accounts": [],
            "streamerInfo": {}
        },
    }
    return responses


def create_mock_request(mock_api_responses, mock_token_response):
    """Create a mock requests.request function that returns appropriate responses."""
    def mock_request(method, url, **kwargs):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = "{}"
        
        # Parse the URL to find the matching response
        path = url.split("/v1/")[-1] if "/v1/" in url else url
        
        # Handle various URL patterns
        for key, value in mock_api_responses.items():
            if key in path:
                mock_response.json.return_value = value
                mock_response.text = json.dumps(value)
                return mock_response
        
        # Default empty response
        mock_response.json.return_value = {}
        return mock_response
    
    return mock_request


@pytest.fixture
def schwab_server(mock_api_responses, mock_token_response):
    """Create a SchwabMCPServer with mocked HTTP requests."""
    with patch('mcp_servers.schwab.server.requests') as mock_requests:
        # Mock token refresh
        mock_requests.post.return_value = mock_token_response
        
        # Mock API requests
        mock_requests.request = MagicMock(
            side_effect=create_mock_request(mock_api_responses, mock_token_response)
        )
        
        from mcp_servers.schwab.server import SchwabMCPServer
        server = SchwabMCPServer(
            client_id="test_client_id",
            client_secret="test_client_secret",
            refresh_token="test_refresh_token",
        )
        
        # Pre-set a valid token to avoid refresh during tests
        server._token = {
            "access_token": "mock_access_token",
            "expires_at": time.time() + 3600,  # Valid for 1 hour
        }
        
        return server


# =============================================================================
# Server Initialization Tests
# =============================================================================

class TestSchwabMCPServerInitialization:
    """Tests for server initialization."""

    def test_server_initializes_with_credentials(self):
        """Server should initialize with API credentials."""
        from mcp_servers.schwab.server import SchwabMCPServer
        
        server = SchwabMCPServer(
            client_id="test_client_id",
            client_secret="test_client_secret",
            refresh_token="test_refresh_token",
        )
        
        assert server.client_id == "test_client_id"
        assert server.client_secret == "test_client_secret"
        assert server.refresh_token == "test_refresh_token"

    def test_server_initializes_with_legacy_credentials(self):
        """Server should initialize with legacy api_key/app_secret names."""
        from mcp_servers.schwab.server import SchwabMCPServer
        
        server = SchwabMCPServer(
            api_key="test_api_key",
            app_secret="test_app_secret",
            refresh_token="test_refresh_token",
        )
        
        assert server.api_key == "test_api_key"
        assert server.app_secret == "test_app_secret"
        # Aliases should also work
        assert server.client_id == "test_api_key"
        assert server.client_secret == "test_app_secret"

    def test_server_initializes_from_env(self):
        """Server should initialize from environment variables."""
        with patch.dict(os.environ, {
            'SCHWAB_CLIENT_ID': 'env_client_id',
            'SCHWAB_CLIENT_SECRET': 'env_client_secret',
            'SCHWAB_REFRESH_TOKEN': 'env_refresh_token',
        }):
            from mcp_servers.schwab.server import SchwabMCPServer
            
            server = SchwabMCPServer.from_env()
            
            assert server.client_id == "env_client_id"
            assert server.client_secret == "env_client_secret"
            assert server.refresh_token == "env_refresh_token"

    def test_server_initializes_from_env_with_legacy_names(self):
        """Server should initialize from legacy environment variable names."""
        with patch.dict(os.environ, {
            'SCHWAB_API_KEY': 'env_api_key',
            'SCHWAB_APP_SECRET': 'env_app_secret',
            'SCHWAB_REFRESH_TOKEN': 'env_refresh_token',
        }, clear=False):
            # Clear the new-style names
            os.environ.pop('SCHWAB_CLIENT_ID', None)
            os.environ.pop('SCHWAB_CLIENT_SECRET', None)
            
            from mcp_servers.schwab.server import SchwabMCPServer
            
            server = SchwabMCPServer.from_env()
            
            assert server.client_id == "env_api_key"

    def test_server_raises_on_missing_credentials(self):
        """Server should raise error when credentials missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing env vars
            for key in ['SCHWAB_CLIENT_ID', 'SCHWAB_CLIENT_SECRET', 'SCHWAB_REFRESH_TOKEN',
                       'SCHWAB_API_KEY', 'SCHWAB_APP_SECRET']:
                os.environ.pop(key, None)
            
            from mcp_servers.schwab.server import SchwabMCPServer
            from mcp_servers.schwab.exceptions import SchwabAuthError
            
            with pytest.raises(SchwabAuthError):
                SchwabMCPServer.from_env()


# =============================================================================
# Token Management Tests
# =============================================================================

class TestTokenManagement:
    """Tests for token refresh functionality."""

    def test_refresh_access_token(self, mock_token_response):
        """refresh_access_token should get new token from API."""
        with patch('mcp_servers.schwab.server.requests') as mock_requests:
            mock_requests.post.return_value = mock_token_response
            
            from mcp_servers.schwab.server import SchwabMCPServer
            server = SchwabMCPServer(
                client_id="test_client_id",
                client_secret="test_client_secret",
                refresh_token="test_refresh_token",
            )
            
            token = server.refresh_access_token()
            
            assert token == "mock_access_token_123"
            assert server._token["access_token"] == "mock_access_token_123"
            mock_requests.post.assert_called_once()

    def test_get_access_token_returns_cached_when_valid(self, mock_token_response):
        """get_access_token should return cached token if not expired."""
        with patch('mcp_servers.schwab.server.requests') as mock_requests:
            from mcp_servers.schwab.server import SchwabMCPServer
            server = SchwabMCPServer(
                client_id="test_client_id",
                client_secret="test_client_secret",
                refresh_token="test_refresh_token",
            )
            server._token = {
                "access_token": "cached_token",
                "expires_at": time.time() + 3600,
            }
            
            token = server.get_access_token()
            
            assert token == "cached_token"
            mock_requests.post.assert_not_called()

    def test_get_access_token_refreshes_when_expired(self, mock_token_response):
        """get_access_token should refresh token if expired."""
        with patch('mcp_servers.schwab.server.requests') as mock_requests:
            mock_requests.post.return_value = mock_token_response
            
            from mcp_servers.schwab.server import SchwabMCPServer
            server = SchwabMCPServer(
                client_id="test_client_id",
                client_secret="test_client_secret",
                refresh_token="test_refresh_token",
            )
            server._token = {
                "access_token": "expired_token",
                "expires_at": time.time() - 100,  # Expired
            }
            
            token = server.get_access_token()
            
            assert token == "mock_access_token_123"
            mock_requests.post.assert_called_once()


# =============================================================================
# Tool Listing Tests
# =============================================================================

class TestSchwabMCPServerToolListing:
    """Tests for tool listing functionality."""

    def test_list_tools_returns_all_tools(self, schwab_server):
        """list_tools should return all available tools."""
        tools = schwab_server.list_tools()
        
        assert len(tools) > 0
        assert all("name" in tool for tool in tools)
        assert all("description" in tool for tool in tools)
        assert all("inputSchema" in tool for tool in tools)

    def test_list_tools_includes_account_tools(self, schwab_server):
        """list_tools should include account-related tools."""
        tools = schwab_server.list_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "get_account_numbers" in tool_names
        assert "get_account" in tool_names
        assert "get_accounts" in tool_names

    def test_list_tools_includes_order_tools(self, schwab_server):
        """list_tools should include order-related tools."""
        tools = schwab_server.list_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "get_orders_for_account" in tool_names
        assert "place_order" in tool_names
        assert "cancel_order" in tool_names
        assert "replace_order" in tool_names
        assert "preview_order" in tool_names

    def test_list_tools_includes_quote_tools(self, schwab_server):
        """list_tools should include quote-related tools."""
        tools = schwab_server.list_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "get_quote" in tool_names
        assert "get_quotes" in tool_names

    def test_list_tools_includes_option_tools(self, schwab_server):
        """list_tools should include option-related tools."""
        tools = schwab_server.list_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "get_option_chain" in tool_names
        assert "get_option_expiration_chain" in tool_names

    def test_list_tools_includes_price_history_tools(self, schwab_server):
        """list_tools should include price history tools."""
        tools = schwab_server.list_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "get_price_history" in tool_names
        assert "get_price_history_every_day" in tool_names
        assert "get_price_history_every_minute" in tool_names

    def test_list_tools_includes_market_tools(self, schwab_server):
        """list_tools should include market data tools."""
        tools = schwab_server.list_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "get_movers" in tool_names
        assert "get_market_hours" in tool_names

    def test_list_tools_includes_instrument_tools(self, schwab_server):
        """list_tools should include instrument tools."""
        tools = schwab_server.list_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "get_instruments" in tool_names
        assert "get_instrument_by_cusip" in tool_names

    def test_list_tools_includes_transaction_tools(self, schwab_server):
        """list_tools should include transaction tools."""
        tools = schwab_server.list_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "get_transactions" in tool_names
        assert "get_transaction" in tool_names

    def test_list_tools_includes_user_tools(self, schwab_server):
        """list_tools should include user preference tools."""
        tools = schwab_server.list_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "get_user_preferences" in tool_names


# =============================================================================
# Account Tool Tests
# =============================================================================

class TestAccountTools:
    """Tests for account-related tools."""

    @pytest.mark.asyncio
    async def test_get_account_numbers(self, schwab_server, mock_api_responses):
        """get_account_numbers should return account numbers and hashes."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["accounts/accountNumbers"]
            
            result = await schwab_server.call_tool("get_account_numbers", {})
            
            mock_request.assert_called_once_with("GET", "/accounts/accountNumbers")
            assert isinstance(result, list)
            assert len(result) > 0
            assert "accountNumber" in result[0]
            assert "hashValue" in result[0]

    @pytest.mark.asyncio
    async def test_get_account(self, schwab_server, mock_api_responses):
        """get_account should return account details."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["accounts/ABC123HASH"]
            
            result = await schwab_server.call_tool("get_account", {
                "account_hash": "ABC123HASH"
            })
            
            assert "securitiesAccount" in result
            assert "currentBalances" in result["securitiesAccount"]

    @pytest.mark.asyncio
    async def test_get_account_with_positions(self, schwab_server, mock_api_responses):
        """get_account should return positions when requested."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["accounts/ABC123HASH"]
            
            result = await schwab_server.call_tool("get_account", {
                "account_hash": "ABC123HASH",
                "fields": ["positions"]
            })
            
            mock_request.assert_called_once()
            assert "securitiesAccount" in result

    @pytest.mark.asyncio
    async def test_get_accounts(self, schwab_server, mock_api_responses):
        """get_accounts should return all accounts."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["accounts"]
            
            result = await schwab_server.call_tool("get_accounts", {})
            
            assert isinstance(result, list)


# =============================================================================
# Order Tool Tests
# =============================================================================

class TestOrderTools:
    """Tests for order-related tools."""

    @pytest.mark.asyncio
    async def test_get_orders_for_account(self, schwab_server, mock_api_responses):
        """get_orders_for_account should return orders."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["accounts/ABC123HASH/orders"]
            
            result = await schwab_server.call_tool("get_orders_for_account", {
                "account_hash": "ABC123HASH"
            })
            
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_orders_for_all_linked_accounts(self, schwab_server, mock_api_responses):
        """get_orders_for_all_linked_accounts should return orders."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["orders"]
            
            result = await schwab_server.call_tool("get_orders_for_all_linked_accounts", {})
            
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_order(self, schwab_server, mock_api_responses):
        """get_order should return specific order."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["accounts/ABC123HASH/orders/123456789"]
            
            result = await schwab_server.call_tool("get_order", {
                "account_hash": "ABC123HASH",
                "order_id": 123456789
            })
            
            assert result["orderId"] == 123456789

    @pytest.mark.asyncio
    async def test_place_order(self, schwab_server):
        """place_order should submit order and return success."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = {}
            
            result = await schwab_server.call_tool("place_order", {
                "account_hash": "ABC123HASH",
                "order_spec": {"orderType": "MARKET"}
            })
            
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_cancel_order(self, schwab_server):
        """cancel_order should cancel and return success."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = {}
            
            result = await schwab_server.call_tool("cancel_order", {
                "account_hash": "ABC123HASH",
                "order_id": 123456789
            })
            
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_replace_order(self, schwab_server):
        """replace_order should replace order and return success."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = {}
            
            result = await schwab_server.call_tool("replace_order", {
                "account_hash": "ABC123HASH",
                "order_id": 123456789,
                "order_spec": {"orderType": "LIMIT", "price": "150.00"}
            })
            
            assert result["success"] is True


# =============================================================================
# Quote Tool Tests
# =============================================================================

class TestQuoteTools:
    """Tests for quote-related tools."""

    @pytest.mark.asyncio
    async def test_get_quote(self, schwab_server, mock_api_responses):
        """get_quote should return quote for symbol."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["AAPL/quotes"]
            
            result = await schwab_server.call_tool("get_quote", {
                "symbol": "AAPL"
            })
            
            assert "AAPL" in result

    @pytest.mark.asyncio
    async def test_get_quotes(self, schwab_server, mock_api_responses):
        """get_quotes should return quotes for multiple symbols."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["quotes"]
            
            result = await schwab_server.call_tool("get_quotes", {
                "symbols": ["AAPL", "GOOGL"]
            })
            
            assert "AAPL" in result
            assert "GOOGL" in result


# =============================================================================
# Option Tool Tests
# =============================================================================

class TestOptionTools:
    """Tests for option-related tools."""

    @pytest.mark.asyncio
    async def test_get_option_chain(self, schwab_server, mock_api_responses):
        """get_option_chain should return option chain."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["chains"]
            
            result = await schwab_server.call_tool("get_option_chain", {
                "symbol": "AAPL"
            })
            
            assert result["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_option_expiration_chain(self, schwab_server, mock_api_responses):
        """get_option_expiration_chain should return expirations."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["expirationchain"]
            
            result = await schwab_server.call_tool("get_option_expiration_chain", {
                "symbol": "AAPL"
            })
            
            assert "expirationList" in result


# =============================================================================
# Price History Tool Tests
# =============================================================================

class TestPriceHistoryTools:
    """Tests for price history tools."""

    @pytest.mark.asyncio
    async def test_get_price_history(self, schwab_server, mock_api_responses):
        """get_price_history should return price history."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["pricehistory"]
            
            result = await schwab_server.call_tool("get_price_history", {
                "symbol": "AAPL"
            })
            
            assert "candles" in result

    @pytest.mark.asyncio
    async def test_get_price_history_every_day(self, schwab_server, mock_api_responses):
        """get_price_history_every_day should return daily prices."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["pricehistory"]
            
            result = await schwab_server.call_tool("get_price_history_every_day", {
                "symbol": "AAPL"
            })
            
            assert "candles" in result

    @pytest.mark.asyncio
    async def test_get_price_history_every_minute(self, schwab_server, mock_api_responses):
        """get_price_history_every_minute should return minute prices."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["pricehistory"]
            
            result = await schwab_server.call_tool("get_price_history_every_minute", {
                "symbol": "AAPL"
            })
            
            assert "candles" in result


# =============================================================================
# Market Data Tool Tests
# =============================================================================

class TestMarketDataTools:
    """Tests for market data tools."""

    @pytest.mark.asyncio
    async def test_get_movers(self, schwab_server, mock_api_responses):
        """get_movers should return market movers."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["movers/$DJI"]
            
            result = await schwab_server.call_tool("get_movers", {
                "index": "$DJI"
            })
            
            assert "screeners" in result

    @pytest.mark.asyncio
    async def test_get_market_hours(self, schwab_server, mock_api_responses):
        """get_market_hours should return market hours."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["markets"]
            
            result = await schwab_server.call_tool("get_market_hours", {
                "markets": ["equity"]
            })
            
            assert "equity" in result


# =============================================================================
# Instrument Tool Tests
# =============================================================================

class TestInstrumentTools:
    """Tests for instrument tools."""

    @pytest.mark.asyncio
    async def test_get_instruments(self, schwab_server, mock_api_responses):
        """get_instruments should search for instruments."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["instruments"]
            
            result = await schwab_server.call_tool("get_instruments", {
                "symbol": "AAPL",
                "projection": "symbol-search"
            })
            
            assert "instruments" in result

    @pytest.mark.asyncio
    async def test_get_instrument_by_cusip(self, schwab_server, mock_api_responses):
        """get_instrument_by_cusip should return instrument by CUSIP."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["instruments/037833100"]
            
            result = await schwab_server.call_tool("get_instrument_by_cusip", {
                "cusip": "037833100"
            })
            
            assert result["cusip"] == "037833100"


# =============================================================================
# Transaction Tool Tests
# =============================================================================

class TestTransactionTools:
    """Tests for transaction tools."""

    @pytest.mark.asyncio
    async def test_get_transactions(self, schwab_server, mock_api_responses):
        """get_transactions should return transactions."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["accounts/ABC123HASH/transactions"]
            
            result = await schwab_server.call_tool("get_transactions", {
                "account_hash": "ABC123HASH"
            })
            
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_transaction(self, schwab_server, mock_api_responses):
        """get_transaction should return specific transaction."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["accounts/ABC123HASH/transactions/999"]
            
            result = await schwab_server.call_tool("get_transaction", {
                "account_hash": "ABC123HASH",
                "transaction_id": 999
            })
            
            assert result["transactionId"] == 999


# =============================================================================
# User Preference Tool Tests
# =============================================================================

class TestUserPreferenceTools:
    """Tests for user preference tools."""

    @pytest.mark.asyncio
    async def test_get_user_preferences(self, schwab_server, mock_api_responses):
        """get_user_preferences should return user preferences."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = mock_api_responses["userPreference"]
            
            result = await schwab_server.call_tool("get_user_preferences", {})
            
            assert "accounts" in result or "streamerInfo" in result


# =============================================================================
# Equity Order Helper Tests
# =============================================================================

class TestEquityOrderHelpers:
    """Tests for equity order helper tools."""

    @pytest.mark.asyncio
    async def test_equity_buy_market(self, schwab_server):
        """equity_buy_market should place market buy order."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = {}
            
            result = await schwab_server.call_tool("equity_buy_market", {
                "account_hash": "ABC123HASH",
                "symbol": "AAPL",
                "quantity": 10
            })
            
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_equity_buy_limit(self, schwab_server):
        """equity_buy_limit should place limit buy order."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = {}
            
            result = await schwab_server.call_tool("equity_buy_limit", {
                "account_hash": "ABC123HASH",
                "symbol": "AAPL",
                "quantity": 10,
                "price": "150.00"
            })
            
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_equity_sell_market(self, schwab_server):
        """equity_sell_market should place market sell order."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = {}
            
            result = await schwab_server.call_tool("equity_sell_market", {
                "account_hash": "ABC123HASH",
                "symbol": "AAPL",
                "quantity": 10
            })
            
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_equity_sell_limit(self, schwab_server):
        """equity_sell_limit should place limit sell order."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = {}
            
            result = await schwab_server.call_tool("equity_sell_limit", {
                "account_hash": "ABC123HASH",
                "symbol": "AAPL",
                "quantity": 10,
                "price": "175.00"
            })
            
            assert result["success"] is True


# =============================================================================
# Option Order Helper Tests
# =============================================================================

class TestOptionOrderHelpers:
    """Tests for option order helper tools."""

    @pytest.mark.asyncio
    async def test_option_buy_to_open_market(self, schwab_server):
        """option_buy_to_open_market should place market buy-to-open."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = {}
            
            result = await schwab_server.call_tool("option_buy_to_open_market", {
                "account_hash": "ABC123HASH",
                "symbol": "AAPL  240119C00175000",
                "quantity": 1
            })
            
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_option_buy_to_open_limit(self, schwab_server):
        """option_buy_to_open_limit should place limit buy-to-open."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = {}
            
            result = await schwab_server.call_tool("option_buy_to_open_limit", {
                "account_hash": "ABC123HASH",
                "symbol": "AAPL  240119C00175000",
                "quantity": 1,
                "price": "5.50"
            })
            
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_option_sell_to_close_market(self, schwab_server):
        """option_sell_to_close_market should place market sell-to-close."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = {}
            
            result = await schwab_server.call_tool("option_sell_to_close_market", {
                "account_hash": "ABC123HASH",
                "symbol": "AAPL  240119C00175000",
                "quantity": 1
            })
            
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_option_sell_to_close_limit(self, schwab_server):
        """option_sell_to_close_limit should place limit sell-to-close."""
        with patch.object(schwab_server, '_schwab_request') as mock_request:
            mock_request.return_value = {}
            
            result = await schwab_server.call_tool("option_sell_to_close_limit", {
                "account_hash": "ABC123HASH",
                "symbol": "AAPL  240119C00175000",
                "quantity": 1,
                "price": "6.00"
            })
            
            assert result["success"] is True


# =============================================================================
# Utility Method Tests
# =============================================================================

class TestUtilityMethods:
    """Tests for utility methods."""

    def test_build_option_symbol(self, schwab_server):
        """build_option_symbol should create valid OCC symbol."""
        symbol = schwab_server.build_option_symbol(
            underlying="AAPL",
            expiration_date="2024-01-19",
            contract_type="CALL",
            strike_price=175.00
        )
        
        assert "AAPL" in symbol
        assert "240119" in symbol
        assert "C" in symbol

    def test_parse_option_symbol(self, schwab_server):
        """parse_option_symbol should parse OCC symbol."""
        result = schwab_server.parse_option_symbol("AAPL  240119C00175000")
        
        assert result["underlying"] == "AAPL"
        assert result["contract_type"] == "CALL"
        assert result["strike_price"] == 175.0


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_error(self, schwab_server):
        """Calling unknown tool should raise SchwabToolError."""
        from mcp_servers.schwab.server import SchwabToolError
        
        with pytest.raises(SchwabToolError) as exc_info:
            await schwab_server.call_tool("unknown_tool", {})
        
        assert "Unknown tool" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_required_param_raises_error(self, schwab_server):
        """Missing required parameter should raise SchwabToolError."""
        from mcp_servers.schwab.server import SchwabToolError
        
        with pytest.raises(SchwabToolError) as exc_info:
            await schwab_server.call_tool("get_account", {})  # Missing account_hash
        
        assert "Missing required parameter" in str(exc_info.value)

    def test_token_refresh_failure_raises_error(self, mock_token_response):
        """Token refresh failure should raise SchwabAPIError."""
        with patch('mcp_servers.schwab.server.requests') as mock_requests:
            mock_response = MagicMock()
            mock_response.ok = False
            mock_response.text = "Invalid credentials"
            mock_requests.post.return_value = mock_response
            
            from mcp_servers.schwab.server import SchwabMCPServer, SchwabAPIError
            server = SchwabMCPServer(
                client_id="test_client_id",
                client_secret="test_client_secret",
                refresh_token="test_refresh_token",
            )
            
            with pytest.raises(SchwabAPIError) as exc_info:
                server.refresh_access_token()
            
            assert "Failed to refresh" in str(exc_info.value)
