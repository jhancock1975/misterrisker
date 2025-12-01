"""
Researcher MCP Server module.

This module provides a Model Context Protocol (MCP) server that exposes
research APIs (Finnhub, OpenAI Web Search) as tools for AI agents
to make trading and investment decisions.
"""

from .server import ResearcherMCPServer
from .exceptions import ResearcherAPIError, ResearcherConfigError

__all__ = [
    "ResearcherMCPServer",
    "ResearcherAPIError",
    "ResearcherConfigError",
]
