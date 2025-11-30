"""
Tests for the Coinbase MCP Server.

These tests follow TDD principles - written before the implementation.
They cover all Coinbase API tools exposed via MCP.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_coinbase_client():
    """Create a mock Coinbase REST client."""
    client = MagicMock()
    
    # Account responses
    client.get_accounts.return_value = {
        "accounts": [
            {
                "uuid": "acc-123",
                "name": "BTC Wallet",
                "currency": "BTC",
                "available_balance": {"value": "1.5", "currency": "BTC"},
                "default": True,
                "active": True,
                "type": "ACCOUNT_TYPE_CRYPTO"
            }
        ],
        "has_next": False,
        "cursor": ""
    }
    
    client.get_account.return_value = {
        "account": {
            "uuid": "acc-123",
            "name": "BTC Wallet",
            "currency": "BTC",
            "available_balance": {"value": "1.5", "currency": "BTC"}
        }
    }
    
    # Product responses
    client.get_products.return_value = {
        "products": [
            {
                "product_id": "BTC-USD",
                "price": "45000.00",
                "base_currency_id": "BTC",
                "quote_currency_id": "USD",
                "trading_disabled": False
            }
        ],
        "num_products": 1
    }
    
    client.get_product.return_value = {
        "product_id": "BTC-USD",
        "price": "45000.00",
        "base_currency_id": "BTC",
        "quote_currency_id": "USD"
    }
    
    client.get_product_book.return_value = {
        "pricebook": {
            "product_id": "BTC-USD",
            "bids": [{"price": "44999.00", "size": "0.5"}],
            "asks": [{"price": "45001.00", "size": "0.3"}],
            "time": "2025-01-01T00:00:00Z"
        }
    }
    
    client.get_best_bid_ask.return_value = {
        "pricebooks": [
            {
                "product_id": "BTC-USD",
                "bids": [{"price": "44999.00", "size": "0.5"}],
                "asks": [{"price": "45001.00", "size": "0.3"}]
            }
        ]
    }
    
    # Market data responses
    client.get_candles.return_value = {
        "candles": [
            {
                "start": "1704067200",
                "low": "44500.00",
                "high": "45500.00",
                "open": "44800.00",
                "close": "45200.00",
                "volume": "1000.5"
            }
        ]
    }
    
    client.get_market_trades.return_value = {
        "trades": [
            {
                "trade_id": "trade-123",
                "product_id": "BTC-USD",
                "price": "45000.00",
                "size": "0.1",
                "time": "2025-01-01T00:00:00Z",
                "side": "BUY"
            }
        ],
        "best_bid": "44999.00",
        "best_ask": "45001.00"
    }
    
    # Order responses
    client.create_order.return_value = {
        "success": True,
        "order_id": "order-123",
        "success_response": {
            "order_id": "order-123",
            "product_id": "BTC-USD",
            "side": "BUY"
        }
    }
    
    client.market_order_buy.return_value = {
        "success": True,
        "order_id": "order-124",
        "success_response": {"order_id": "order-124"}
    }
    
    client.market_order_sell.return_value = {
        "success": True,
        "order_id": "order-125",
        "success_response": {"order_id": "order-125"}
    }
    
    client.limit_order_gtc_buy.return_value = {
        "success": True,
        "order_id": "order-126",
        "success_response": {"order_id": "order-126"}
    }
    
    client.limit_order_gtc_sell.return_value = {
        "success": True,
        "order_id": "order-127",
        "success_response": {"order_id": "order-127"}
    }
    
    client.cancel_orders.return_value = {
        "results": [
            {"success": True, "order_id": "order-123"}
        ]
    }
    
    client.get_order.return_value = {
        "order": {
            "order_id": "order-123",
            "product_id": "BTC-USD",
            "side": "BUY",
            "status": "FILLED"
        }
    }
    
    client.list_orders.return_value = {
        "orders": [
            {"order_id": "order-123", "status": "FILLED"},
            {"order_id": "order-124", "status": "PENDING"}
        ],
        "has_next": False
    }
    
    client.get_fills.return_value = {
        "fills": [
            {
                "entry_id": "fill-123",
                "trade_id": "trade-123",
                "order_id": "order-123",
                "price": "45000.00",
                "size": "0.1"
            }
        ]
    }
    
    # Portfolio responses
    client.get_portfolios.return_value = {
        "portfolios": [
            {"uuid": "port-123", "name": "Default", "type": "DEFAULT"}
        ]
    }
    
    client.get_portfolio_breakdown.return_value = {
        "breakdown": {
            "portfolio_uuid": "port-123",
            "total_balance": {"value": "10000.00", "currency": "USD"}
        }
    }
    
    # Public endpoints
    client.get_unix_time.return_value = {
        "iso": "2025-01-01T00:00:00Z",
        "epochSeconds": "1704067200",
        "epochMillis": "1704067200000"
    }
    
    client.get_public_products.return_value = {
        "products": [
            {"product_id": "BTC-USD", "price": "45000.00"}
        ]
    }
    
    client.get_public_product.return_value = {
        "product_id": "BTC-USD",
        "price": "45000.00"
    }
    
    # Fee endpoint
    client.get_transaction_summary.return_value = {
        "total_volume": 10000.0,
        "total_fees": 50.0,
        "fee_tier": {
            "pricing_tier": "Advanced",
            "maker_fee_rate": "0.004",
            "taker_fee_rate": "0.006"
        }
    }
    
    return client


@pytest.fixture
def coinbase_server(mock_coinbase_client):
    """Create a Coinbase MCP server with mocked client."""
    from mcp_servers.coinbase.server import CoinbaseMCPServer
    
    with patch('mcp_servers.coinbase.server.RESTClient', return_value=mock_coinbase_client):
        server = CoinbaseMCPServer(
            api_key="test-api-key",
            api_secret="test-api-secret"
        )
        server._client = mock_coinbase_client
        return server


# =============================================================================
# Server Initialization Tests
# =============================================================================

class TestCoinbaseMCPServerInitialization:
    """Tests for MCP server initialization."""

    def test_server_initializes_with_credentials(self):
        """Server should initialize with API credentials."""
        from mcp_servers.coinbase.server import CoinbaseMCPServer
        
        # Use a secret with PEM headers so it won't be wrapped
        pem_secret = "-----BEGIN EC PRIVATE KEY-----\ntest-secret\n-----END EC PRIVATE KEY-----"
        
        with patch('mcp_servers.coinbase.server.RESTClient') as MockClient:
            server = CoinbaseMCPServer(
                api_key="test-key",
                api_secret=pem_secret
            )
            
            assert server is not None
            MockClient.assert_called_once_with(
                api_key="test-key",
                api_secret=pem_secret
            )

    def test_server_converts_escaped_newlines_in_secret(self):
        """Server should convert escaped \\n to actual newlines in PEM secret."""
        from mcp_servers.coinbase.server import CoinbaseMCPServer
        
        # Simulate a PEM key stored in .env with escaped newlines (already has headers)
        escaped_secret = "-----BEGIN EC PRIVATE KEY-----\\nMHQC...\\n-----END EC PRIVATE KEY-----"
        expected_secret = "-----BEGIN EC PRIVATE KEY-----\nMHQC...\n-----END EC PRIVATE KEY-----"
        
        with patch('mcp_servers.coinbase.server.RESTClient') as MockClient:
            server = CoinbaseMCPServer(
                api_key="test-key",
                api_secret=escaped_secret
            )
            
            # Verify the secret was converted before passing to RESTClient
            MockClient.assert_called_once_with(
                api_key="test-key",
                api_secret=expected_secret
            )
            # Also check the stored secret
            assert server.api_secret == expected_secret

    def test_server_wraps_raw_base64_secret_with_pem_headers(self):
        """Server should add PEM headers if secret is raw base64 without headers."""
        from mcp_servers.coinbase.server import CoinbaseMCPServer
        
        # Simulate a secret without PEM headers (just the base64 content)
        raw_secret = "MHcCAQEEIOBmgaQitNbAopUt4oLHgiLdNkuOJzYyWyufxRgn4KX5oAoGCCqGSM49\\nAwEHoUQDQgAEt+HpfUtLba9Q4/fIaX1oirOQVsiZRtiKuzVmfaIAksbE57wmnEAg"
        expected_secret = (
            "-----BEGIN EC PRIVATE KEY-----\n"
            "MHcCAQEEIOBmgaQitNbAopUt4oLHgiLdNkuOJzYyWyufxRgn4KX5oAoGCCqGSM49\n"
            "AwEHoUQDQgAEt+HpfUtLba9Q4/fIaX1oirOQVsiZRtiKuzVmfaIAksbE57wmnEAg\n"
            "-----END EC PRIVATE KEY-----"
        )
        
        with patch('mcp_servers.coinbase.server.RESTClient') as MockClient:
            server = CoinbaseMCPServer(
                api_key="test-key",
                api_secret=raw_secret
            )
            
            # Verify PEM headers were added
            MockClient.assert_called_once()
            call_args = MockClient.call_args
            actual_secret = call_args.kwargs.get('api_secret') or call_args[1].get('api_secret')
            
            assert actual_secret.startswith("-----BEGIN EC PRIVATE KEY-----")
            assert actual_secret.endswith("-----END EC PRIVATE KEY-----")
            assert "MHcCAQEEIOBmgaQitNbAopUt4oLHgiLdNkuOJzYyWyufxRgn4KX5" in actual_secret

    def test_server_handles_secret_with_real_newlines(self):
        """Server should handle secrets that already have real newlines."""
        from mcp_servers.coinbase.server import CoinbaseMCPServer
        
        # Secret already has real newlines (e.g., from a file)
        real_newline_secret = "-----BEGIN EC PRIVATE KEY-----\nMHQC...\n-----END EC PRIVATE KEY-----"
        
        with patch('mcp_servers.coinbase.server.RESTClient') as MockClient:
            server = CoinbaseMCPServer(
                api_key="test-key",
                api_secret=real_newline_secret
            )
            
            # Should pass through unchanged
            MockClient.assert_called_once_with(
                api_key="test-key",
                api_secret=real_newline_secret
            )

    def test_server_initializes_from_env(self):
        """Server should initialize from environment variables."""
        from mcp_servers.coinbase.server import CoinbaseMCPServer
        
        # Use a secret with PEM headers so it won't be wrapped
        pem_secret = "-----BEGIN EC PRIVATE KEY-----\nenv-secret\n-----END EC PRIVATE KEY-----"
        
        with patch.dict(os.environ, {
            "COINBASE_API_KEY": "env-key",
            "COINBASE_API_SECRET": pem_secret
        }):
            with patch('mcp_servers.coinbase.server.RESTClient') as MockClient:
                server = CoinbaseMCPServer.from_env()
                
                MockClient.assert_called_once_with(
                    api_key="env-key",
                    api_secret=pem_secret
                )

    def test_server_raises_error_without_credentials(self):
        """Server should raise error if credentials missing."""
        from mcp_servers.coinbase.server import CoinbaseMCPServer, CoinbaseAuthError
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(CoinbaseAuthError, match="API key"):
                CoinbaseMCPServer.from_env()

    def test_server_lists_available_tools(self):
        """Server should list all available MCP tools."""
        from mcp_servers.coinbase.server import CoinbaseMCPServer
        
        with patch('mcp_servers.coinbase.server.RESTClient'):
            server = CoinbaseMCPServer(
                api_key="test-key",
                api_secret="test-secret"
            )
            
            tools = server.list_tools()
            
            assert len(tools) > 0
            tool_names = [t["name"] for t in tools]
            assert "get_accounts" in tool_names
            assert "get_products" in tool_names
            assert "market_order_buy" in tool_names


# =============================================================================
# Account Tools Tests
# =============================================================================

class TestAccountTools:
    """Tests for account-related MCP tools."""

    @pytest.mark.asyncio
    async def test_get_accounts(self, coinbase_server, mock_coinbase_client):
        """get_accounts should return list of accounts."""
        result = await coinbase_server.call_tool("get_accounts", {})
        
        mock_coinbase_client.get_accounts.assert_called_once()
        assert "accounts" in result
        assert len(result["accounts"]) == 1
        assert result["accounts"][0]["uuid"] == "acc-123"

    @pytest.mark.asyncio
    async def test_get_accounts_with_pagination(self, coinbase_server, mock_coinbase_client):
        """get_accounts should support pagination parameters."""
        result = await coinbase_server.call_tool("get_accounts", {
            "limit": 50,
            "cursor": "abc123"
        })
        
        mock_coinbase_client.get_accounts.assert_called_once_with(
            limit=50,
            cursor="abc123"
        )

    @pytest.mark.asyncio
    async def test_get_account(self, coinbase_server, mock_coinbase_client):
        """get_account should return specific account details."""
        result = await coinbase_server.call_tool("get_account", {
            "account_uuid": "acc-123"
        })
        
        mock_coinbase_client.get_account.assert_called_once_with("acc-123")
        assert result["account"]["uuid"] == "acc-123"


# =============================================================================
# Product Tools Tests
# =============================================================================

class TestProductTools:
    """Tests for product-related MCP tools."""

    @pytest.mark.asyncio
    async def test_get_products(self, coinbase_server, mock_coinbase_client):
        """get_products should return list of trading pairs."""
        result = await coinbase_server.call_tool("get_products", {})
        
        mock_coinbase_client.get_products.assert_called_once()
        assert "products" in result
        assert result["products"][0]["product_id"] == "BTC-USD"

    @pytest.mark.asyncio
    async def test_get_products_with_filters(self, coinbase_server, mock_coinbase_client):
        """get_products should support filter parameters."""
        result = await coinbase_server.call_tool("get_products", {
            "product_type": "SPOT",
            "product_ids": ["BTC-USD", "ETH-USD"]
        })
        
        mock_coinbase_client.get_products.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_product(self, coinbase_server, mock_coinbase_client):
        """get_product should return specific product details."""
        result = await coinbase_server.call_tool("get_product", {
            "product_id": "BTC-USD"
        })
        
        mock_coinbase_client.get_product.assert_called_once_with("BTC-USD")
        assert result["product_id"] == "BTC-USD"

    @pytest.mark.asyncio
    async def test_get_product_book(self, coinbase_server, mock_coinbase_client):
        """get_product_book should return order book data."""
        result = await coinbase_server.call_tool("get_product_book", {
            "product_id": "BTC-USD",
            "limit": 10
        })
        
        mock_coinbase_client.get_product_book.assert_called_once()
        assert "pricebook" in result
        assert "bids" in result["pricebook"]
        assert "asks" in result["pricebook"]

    @pytest.mark.asyncio
    async def test_get_best_bid_ask(self, coinbase_server, mock_coinbase_client):
        """get_best_bid_ask should return best prices."""
        result = await coinbase_server.call_tool("get_best_bid_ask", {
            "product_ids": ["BTC-USD"]
        })
        
        mock_coinbase_client.get_best_bid_ask.assert_called_once()
        assert "pricebooks" in result


# =============================================================================
# Market Data Tools Tests
# =============================================================================

class TestMarketDataTools:
    """Tests for market data MCP tools."""

    @pytest.mark.asyncio
    async def test_get_candles(self, coinbase_server, mock_coinbase_client):
        """get_candles should return OHLCV data."""
        result = await coinbase_server.call_tool("get_candles", {
            "product_id": "BTC-USD",
            "start": "1704067200",
            "end": "1704153600",
            "granularity": "ONE_HOUR"
        })
        
        mock_coinbase_client.get_candles.assert_called_once()
        assert "candles" in result
        assert "open" in result["candles"][0]
        assert "close" in result["candles"][0]

    @pytest.mark.asyncio
    async def test_get_market_trades(self, coinbase_server, mock_coinbase_client):
        """get_market_trades should return recent trades."""
        result = await coinbase_server.call_tool("get_market_trades", {
            "product_id": "BTC-USD",
            "limit": 100
        })
        
        mock_coinbase_client.get_market_trades.assert_called_once()
        assert "trades" in result
        assert result["trades"][0]["product_id"] == "BTC-USD"


# =============================================================================
# Order Tools Tests
# =============================================================================

class TestOrderTools:
    """Tests for order-related MCP tools."""

    @pytest.mark.asyncio
    async def test_market_order_buy_generates_client_order_id(self, coinbase_server, mock_coinbase_client):
        """market_order_buy should auto-generate client_order_id if not provided."""
        result = await coinbase_server.call_tool("market_order_buy", {
            "product_id": "BTC-USD",
            "quote_size": "100.00"
        })
        
        mock_coinbase_client.market_order_buy.assert_called_once()
        call_kwargs = mock_coinbase_client.market_order_buy.call_args.kwargs
        assert "client_order_id" in call_kwargs
        assert call_kwargs["client_order_id"] is not None
        assert len(call_kwargs["client_order_id"]) == 36  # UUID format
        assert result["success"] is True
        assert "order_id" in result

    @pytest.mark.asyncio
    async def test_market_order_buy_uses_provided_client_order_id(self, coinbase_server, mock_coinbase_client):
        """market_order_buy should use provided client_order_id."""
        custom_id = "my-custom-order-123"
        result = await coinbase_server.call_tool("market_order_buy", {
            "product_id": "BTC-USD",
            "quote_size": "100.00",
            "client_order_id": custom_id
        })
        
        mock_coinbase_client.market_order_buy.assert_called_once()
        call_kwargs = mock_coinbase_client.market_order_buy.call_args.kwargs
        assert call_kwargs["client_order_id"] == custom_id
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_market_order_sell_generates_client_order_id(self, coinbase_server, mock_coinbase_client):
        """market_order_sell should auto-generate client_order_id if not provided."""
        result = await coinbase_server.call_tool("market_order_sell", {
            "product_id": "BTC-USD",
            "base_size": "0.01"
        })
        
        mock_coinbase_client.market_order_sell.assert_called_once()
        call_kwargs = mock_coinbase_client.market_order_sell.call_args.kwargs
        assert "client_order_id" in call_kwargs
        assert call_kwargs["client_order_id"] is not None
        assert len(call_kwargs["client_order_id"]) == 36  # UUID format
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_market_order_sell_uses_provided_client_order_id(self, coinbase_server, mock_coinbase_client):
        """market_order_sell should use provided client_order_id."""
        custom_id = "my-sell-order-456"
        result = await coinbase_server.call_tool("market_order_sell", {
            "product_id": "BTC-USD",
            "base_size": "0.01",
            "client_order_id": custom_id
        })
        
        mock_coinbase_client.market_order_sell.assert_called_once()
        call_kwargs = mock_coinbase_client.market_order_sell.call_args.kwargs
        assert call_kwargs["client_order_id"] == custom_id

    @pytest.mark.asyncio
    async def test_limit_order_gtc_buy_generates_client_order_id(self, coinbase_server, mock_coinbase_client):
        """limit_order_gtc_buy should auto-generate client_order_id if not provided."""
        result = await coinbase_server.call_tool("limit_order_gtc_buy", {
            "product_id": "BTC-USD",
            "base_size": "0.01",
            "limit_price": "40000.00"
        })
        
        mock_coinbase_client.limit_order_gtc_buy.assert_called_once()
        call_kwargs = mock_coinbase_client.limit_order_gtc_buy.call_args.kwargs
        assert "client_order_id" in call_kwargs
        assert call_kwargs["client_order_id"] is not None
        assert len(call_kwargs["client_order_id"]) == 36  # UUID format
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_limit_order_gtc_buy_uses_provided_client_order_id(self, coinbase_server, mock_coinbase_client):
        """limit_order_gtc_buy should use provided client_order_id."""
        custom_id = "limit-buy-789"
        result = await coinbase_server.call_tool("limit_order_gtc_buy", {
            "product_id": "BTC-USD",
            "base_size": "0.01",
            "limit_price": "40000.00",
            "client_order_id": custom_id
        })
        
        mock_coinbase_client.limit_order_gtc_buy.assert_called_once()
        call_kwargs = mock_coinbase_client.limit_order_gtc_buy.call_args.kwargs
        assert call_kwargs["client_order_id"] == custom_id

    @pytest.mark.asyncio
    async def test_limit_order_gtc_sell_generates_client_order_id(self, coinbase_server, mock_coinbase_client):
        """limit_order_gtc_sell should auto-generate client_order_id if not provided."""
        result = await coinbase_server.call_tool("limit_order_gtc_sell", {
            "product_id": "BTC-USD",
            "base_size": "0.01",
            "limit_price": "50000.00"
        })
        
        mock_coinbase_client.limit_order_gtc_sell.assert_called_once()
        call_kwargs = mock_coinbase_client.limit_order_gtc_sell.call_args.kwargs
        assert "client_order_id" in call_kwargs
        assert call_kwargs["client_order_id"] is not None
        assert len(call_kwargs["client_order_id"]) == 36  # UUID format
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_limit_order_gtc_sell_uses_provided_client_order_id(self, coinbase_server, mock_coinbase_client):
        """limit_order_gtc_sell should use provided client_order_id."""
        custom_id = "limit-sell-xyz"
        result = await coinbase_server.call_tool("limit_order_gtc_sell", {
            "product_id": "BTC-USD",
            "base_size": "0.01",
            "limit_price": "50000.00",
            "client_order_id": custom_id
        })
        
        mock_coinbase_client.limit_order_gtc_sell.assert_called_once()
        call_kwargs = mock_coinbase_client.limit_order_gtc_sell.call_args.kwargs
        assert call_kwargs["client_order_id"] == custom_id

    @pytest.mark.asyncio
    async def test_cancel_orders(self, coinbase_server, mock_coinbase_client):
        """cancel_orders should cancel specified orders."""
        result = await coinbase_server.call_tool("cancel_orders", {
            "order_ids": ["order-123", "order-124"]
        })
        
        mock_coinbase_client.cancel_orders.assert_called_once_with(
            order_ids=["order-123", "order-124"]
        )
        assert "results" in result

    @pytest.mark.asyncio
    async def test_get_order(self, coinbase_server, mock_coinbase_client):
        """get_order should return order details."""
        result = await coinbase_server.call_tool("get_order", {
            "order_id": "order-123"
        })
        
        mock_coinbase_client.get_order.assert_called_once_with("order-123")
        assert result["order"]["order_id"] == "order-123"

    @pytest.mark.asyncio
    async def test_list_orders(self, coinbase_server, mock_coinbase_client):
        """list_orders should return orders with filters."""
        result = await coinbase_server.call_tool("list_orders", {
            "product_id": "BTC-USD",
            "order_status": ["OPEN", "PENDING"]
        })
        
        mock_coinbase_client.list_orders.assert_called_once()
        assert "orders" in result

    @pytest.mark.asyncio
    async def test_get_fills(self, coinbase_server, mock_coinbase_client):
        """get_fills should return fill history."""
        result = await coinbase_server.call_tool("get_fills", {
            "product_id": "BTC-USD"
        })
        
        mock_coinbase_client.get_fills.assert_called_once()
        assert "fills" in result


# =============================================================================
# Portfolio Tools Tests
# =============================================================================

class TestPortfolioTools:
    """Tests for portfolio-related MCP tools."""

    @pytest.mark.asyncio
    async def test_get_portfolios(self, coinbase_server, mock_coinbase_client):
        """get_portfolios should return list of portfolios."""
        result = await coinbase_server.call_tool("get_portfolios", {})
        
        mock_coinbase_client.get_portfolios.assert_called_once()
        assert "portfolios" in result

    @pytest.mark.asyncio
    async def test_get_portfolio_breakdown(self, coinbase_server, mock_coinbase_client):
        """get_portfolio_breakdown should return portfolio details."""
        result = await coinbase_server.call_tool("get_portfolio_breakdown", {
            "portfolio_uuid": "port-123"
        })
        
        mock_coinbase_client.get_portfolio_breakdown.assert_called_once_with("port-123")
        assert "breakdown" in result


# =============================================================================
# Public Tools Tests
# =============================================================================

class TestPublicTools:
    """Tests for public (no-auth) MCP tools."""

    @pytest.mark.asyncio
    async def test_get_unix_time(self, coinbase_server, mock_coinbase_client):
        """get_unix_time should return server time."""
        result = await coinbase_server.call_tool("get_unix_time", {})
        
        mock_coinbase_client.get_unix_time.assert_called_once()
        assert "epochSeconds" in result

    @pytest.mark.asyncio
    async def test_get_public_products(self, coinbase_server, mock_coinbase_client):
        """get_public_products should return products without auth."""
        result = await coinbase_server.call_tool("get_public_products", {})
        
        mock_coinbase_client.get_public_products.assert_called_once()
        assert "products" in result

    @pytest.mark.asyncio
    async def test_get_public_product(self, coinbase_server, mock_coinbase_client):
        """get_public_product should return product details without auth."""
        result = await coinbase_server.call_tool("get_public_product", {
            "product_id": "BTC-USD"
        })
        
        mock_coinbase_client.get_public_product.assert_called_once_with("BTC-USD")


# =============================================================================
# Fee Tools Tests
# =============================================================================

class TestFeeTools:
    """Tests for fee-related MCP tools."""

    @pytest.mark.asyncio
    async def test_get_transaction_summary(self, coinbase_server, mock_coinbase_client):
        """get_transaction_summary should return fee information."""
        result = await coinbase_server.call_tool("get_transaction_summary", {})
        
        mock_coinbase_client.get_transaction_summary.assert_called_once()
        assert "fee_tier" in result


# =============================================================================
# Client Order ID Tests
# =============================================================================

class TestClientOrderIdRequired:
    """Tests verifying client_order_id is properly handled for all order types.
    
    The Coinbase API requires a client_order_id for all orders. These tests
    verify our MCP server properly generates or passes through client_order_id.
    """

    @pytest.fixture
    def strict_mock_coinbase_client(self):
        """Create a mock client that validates client_order_id is provided."""
        client = MagicMock()
        
        def validate_client_order_id(**kwargs):
            """Validate that client_order_id is provided and valid."""
            if "client_order_id" not in kwargs:
                raise TypeError("market_order_buy() missing required argument: 'client_order_id'")
            if not kwargs["client_order_id"]:
                raise ValueError("client_order_id cannot be empty")
            return {
                "success": True,
                "order_id": "test-order-123",
                "client_order_id": kwargs["client_order_id"]
            }
        
        client.market_order_buy.side_effect = validate_client_order_id
        client.market_order_sell.side_effect = validate_client_order_id
        client.limit_order_gtc_buy.side_effect = validate_client_order_id
        client.limit_order_gtc_sell.side_effect = validate_client_order_id
        
        return client

    @pytest.fixture
    def strict_coinbase_server(self, strict_mock_coinbase_client):
        """Create a server with the strict mock client."""
        from mcp_servers.coinbase.server import CoinbaseMCPServer
        
        with patch('mcp_servers.coinbase.server.RESTClient', return_value=strict_mock_coinbase_client):
            pem_secret = "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
            server = CoinbaseMCPServer(api_key="test-key", api_secret=pem_secret)
            yield server

    @pytest.mark.asyncio
    async def test_market_order_buy_must_have_client_order_id(self, strict_coinbase_server, strict_mock_coinbase_client):
        """Verify market_order_buy passes client_order_id to API (required by Coinbase)."""
        result = await strict_coinbase_server.call_tool("market_order_buy", {
            "product_id": "BTC-USD",
            "quote_size": "100.00"
        })
        
        # If we got here without error, client_order_id was properly passed
        assert result["success"] is True
        assert "client_order_id" in result
        assert len(result["client_order_id"]) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_market_order_sell_must_have_client_order_id(self, strict_coinbase_server, strict_mock_coinbase_client):
        """Verify market_order_sell passes client_order_id to API (required by Coinbase)."""
        result = await strict_coinbase_server.call_tool("market_order_sell", {
            "product_id": "BTC-USD",
            "base_size": "0.01"
        })
        
        assert result["success"] is True
        assert "client_order_id" in result

    @pytest.mark.asyncio
    async def test_limit_order_gtc_buy_must_have_client_order_id(self, strict_coinbase_server, strict_mock_coinbase_client):
        """Verify limit_order_gtc_buy passes client_order_id to API (required by Coinbase)."""
        result = await strict_coinbase_server.call_tool("limit_order_gtc_buy", {
            "product_id": "BTC-USD",
            "base_size": "0.01",
            "limit_price": "40000.00"
        })
        
        assert result["success"] is True
        assert "client_order_id" in result

    @pytest.mark.asyncio
    async def test_limit_order_gtc_sell_must_have_client_order_id(self, strict_coinbase_server, strict_mock_coinbase_client):
        """Verify limit_order_gtc_sell passes client_order_id to API (required by Coinbase)."""
        result = await strict_coinbase_server.call_tool("limit_order_gtc_sell", {
            "product_id": "BTC-USD",
            "base_size": "0.01",
            "limit_price": "50000.00"
        })
        
        assert result["success"] is True
        assert "client_order_id" in result

    @pytest.mark.asyncio
    async def test_each_order_gets_unique_client_order_id(self, strict_coinbase_server, strict_mock_coinbase_client):
        """Each order should get a unique client_order_id."""
        result1 = await strict_coinbase_server.call_tool("market_order_buy", {
            "product_id": "BTC-USD",
            "quote_size": "100.00"
        })
        result2 = await strict_coinbase_server.call_tool("market_order_buy", {
            "product_id": "BTC-USD",
            "quote_size": "100.00"
        })
        
        assert result1["client_order_id"] != result2["client_order_id"]


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in MCP tools."""

    @pytest.mark.asyncio
    async def test_invalid_tool_name(self, coinbase_server):
        """Server should raise error for invalid tool name."""
        from mcp_servers.coinbase.server import CoinbaseToolError
        
        with pytest.raises(CoinbaseToolError, match="Unknown tool"):
            await coinbase_server.call_tool("invalid_tool", {})

    @pytest.mark.asyncio
    async def test_missing_required_parameter(self, coinbase_server):
        """Server should raise error for missing required parameters."""
        from mcp_servers.coinbase.server import CoinbaseToolError
        
        with pytest.raises(CoinbaseToolError, match="required"):
            await coinbase_server.call_tool("get_account", {})  # Missing account_uuid

    @pytest.mark.asyncio
    async def test_api_error_handling(self, coinbase_server, mock_coinbase_client):
        """Server should handle API errors gracefully."""
        from mcp_servers.coinbase.server import CoinbaseAPIError
        
        mock_coinbase_client.get_accounts.side_effect = Exception("API Error")
        
        with pytest.raises(CoinbaseAPIError, match="API Error"):
            await coinbase_server.call_tool("get_accounts", {})


# =============================================================================
# Tool Schema Tests
# =============================================================================

class TestToolSchemas:
    """Tests for MCP tool JSON schemas."""

    def test_tool_has_name_and_description(self, coinbase_server):
        """Each tool should have name and description."""
        tools = coinbase_server.list_tools()
        
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert len(tool["description"]) > 0

    def test_tool_has_input_schema(self, coinbase_server):
        """Each tool should have an input schema."""
        tools = coinbase_server.list_tools()
        
        for tool in tools:
            assert "inputSchema" in tool
            assert "type" in tool["inputSchema"]
            assert tool["inputSchema"]["type"] == "object"

    def test_order_tools_require_product_id(self, coinbase_server):
        """Order tools should require product_id parameter."""
        tools = coinbase_server.list_tools()
        order_tools = [t for t in tools if "order" in t["name"].lower() and "list" not in t["name"].lower() and "get" not in t["name"].lower() and "cancel" not in t["name"].lower()]
        
        for tool in order_tools:
            schema = tool["inputSchema"]
            if "required" in schema:
                assert "product_id" in schema.get("required", []), f"{tool['name']} should require product_id"
