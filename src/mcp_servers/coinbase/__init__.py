"""Coinbase MCP Server module.

This module provides a Model Context Protocol (MCP) server that exposes
the Coinbase Advanced Trade API as tools for AI agents.
"""

from .server import CoinbaseMCPServer, CoinbaseToolError
from .exceptions import CoinbaseAPIError, CoinbaseAuthError

__all__ = [
    "CoinbaseMCPServer",
    "CoinbaseToolError",
    "CoinbaseAPIError",
    "CoinbaseAuthError",
]
