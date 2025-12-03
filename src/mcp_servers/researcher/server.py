"""Researcher MCP Server - Exposes research APIs as MCP tools.

This module provides a Model Context Protocol (MCP) server that wraps
various financial research APIs (Finnhub, OpenAI Web Search) and exposes
functionality as tools that can be used by AI agents for making
trading and investment decisions.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any

from .exceptions import ResearcherAPIError, ResearcherConfigError


class ResearcherMCPServer:
    """MCP Server that exposes research API tools.
    
    This server wraps Finnhub and OpenAI Web Search APIs and exposes
    all functionality as MCP-compatible tools that can be discovered
    and invoked by AI agents for investment research.
    
    Attributes:
        finnhub_client: Finnhub API client
        openai_client: OpenAI API client for web search
    """
    
    def __init__(
        self,
        finnhub_api_key: str | None = None,
        openai_api_key: str | None = None,
        finnhub_client: Any | None = None,
        openai_client: Any | None = None
    ):
        """Initialize the Researcher MCP Server.
        
        Args:
            finnhub_api_key: Finnhub API key (optional if client provided)
            openai_api_key: OpenAI API key (optional if client provided)
            finnhub_client: Pre-configured Finnhub client (for testing)
            openai_client: Pre-configured OpenAI client (for testing)
        
        Raises:
            ResearcherConfigError: If required API keys are not provided
        """
        # Use injected clients for testing, or create real ones
        if finnhub_client is not None:
            self.finnhub_client = finnhub_client
        else:
            api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
            if not api_key:
                raise ResearcherConfigError(
                    "Finnhub API key not provided",
                    missing_key="FINNHUB_API_KEY"
                )
            import finnhub
            self.finnhub_client = finnhub.Client(api_key=api_key)
        
        if openai_client is not None:
            self.openai_client = openai_client
        else:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ResearcherConfigError(
                    "OpenAI API key not provided",
                    missing_key="OPENAI_API_KEY"
                )
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=api_key)
        
        # Define tool schemas
        self._tool_schemas = self._define_tool_schemas()
    
    @classmethod
    def from_env(cls) -> "ResearcherMCPServer":
        """Create a ResearcherMCPServer from environment variables.
        
        Returns:
            ResearcherMCPServer instance
        
        Raises:
            ResearcherConfigError: If required credentials not found
        """
        return cls()
    
    def _define_tool_schemas(self) -> list[dict[str, Any]]:
        """Define all MCP tool schemas.
        
        Returns:
            List of tool schema dictionaries.
        """
        return [
            # Stock Quote
            {
                "name": "get_stock_quote",
                "description": "Get real-time stock quote including current price, change, and trading range.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., 'AAPL', 'MSFT')"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            # Company Profile
            {
                "name": "get_company_profile",
                "description": "Get company profile including name, industry, market cap, and other details.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., 'AAPL', 'MSFT')"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            # Company News
            {
                "name": "get_company_news",
                "description": "Get recent news articles about a specific company.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., 'AAPL', 'MSFT')"
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (default: 7)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of articles to return"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            # Market News
            {
                "name": "get_market_news",
                "description": "Get general market news by category.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "News category: general, forex, crypto, merger",
                            "enum": ["general", "forex", "crypto", "merger"]
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of articles to return"
                        }
                    }
                }
            },
            # Analyst Recommendations
            {
                "name": "get_analyst_recommendations",
                "description": "Get analyst recommendation trends (buy/sell/hold ratings).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., 'AAPL', 'MSFT')"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            # Basic Financials
            {
                "name": "get_basic_financials",
                "description": "Get basic financial metrics including P/E ratio, 52-week high/low, market cap, etc.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., 'AAPL', 'MSFT')"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            # Company Peers
            {
                "name": "get_company_peers",
                "description": "Get list of company peers/competitors.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., 'AAPL', 'MSFT')"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            # Earnings History
            {
                "name": "get_earnings_history",
                "description": "Get historical earnings data including actual vs estimated and surprise percentage.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., 'AAPL', 'MSFT')"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            # Web Search
            {
                "name": "web_search",
                "description": "Search the web for financial news and analysis using OpenAI web search.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for financial information"
                        },
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of domains to search (e.g., ['reuters.com', 'bloomberg.com'])"
                        }
                    },
                    "required": ["query"]
                }
            },
            # Comprehensive Stock Research
            {
                "name": "research_stock",
                "description": "Perform comprehensive research on a stock including quote, profile, financials, recommendations, and news.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., 'AAPL', 'MSFT')"
                        },
                        "include_web_search": {
                            "type": "boolean",
                            "description": "Whether to include web search for additional insights (default: false)"
                        }
                    },
                    "required": ["symbol"]
                }
            }
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
            ResearcherAPIError: If tool_name is not found or API call fails
        """
        # Check if tool exists
        tool_names = [t["name"] for t in self._tool_schemas]
        if tool_name not in tool_names:
            raise ResearcherAPIError(f"Unknown tool: {tool_name}", tool_name=tool_name)
        
        # Get tool schema for validation
        tool_schema = next(t for t in self._tool_schemas if t["name"] == tool_name)
        
        # Validate required parameters
        required_params = tool_schema["inputSchema"].get("required", [])
        for param in required_params:
            if param not in params:
                raise ResearcherAPIError(
                    f"Missing required parameter '{param}' for tool '{tool_name}'",
                    tool_name=tool_name
                )
        
        # Route to appropriate handler
        try:
            handler = getattr(self, f"_tool_{tool_name}")
            return await handler(params)
        except ResearcherAPIError:
            raise
        except Exception as e:
            raise ResearcherAPIError(
                f"API Error: {str(e)}",
                tool_name=tool_name,
                api_source=self._get_api_source(tool_name)
            )
    
    def _get_api_source(self, tool_name: str) -> str:
        """Get the API source for a tool.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            API source name
        """
        if tool_name == "web_search":
            return "openai"
        return "finnhub"
    
    # ===================
    # Finnhub Tool Handlers
    # ===================
    
    async def _tool_get_stock_quote(self, params: dict) -> dict:
        """Handle get_stock_quote tool call."""
        symbol = params["symbol"].upper()
        
        quote = self.finnhub_client.quote(symbol)
        
        # Check for invalid/empty response
        if quote.get("c") == 0 and quote.get("d") is None:
            return {"error": f"No quote data found for symbol: {symbol}"}
        
        return {
            "symbol": symbol,
            "current_price": quote.get("c"),
            "change": quote.get("d"),
            "change_percent": quote.get("dp"),
            "high": quote.get("h"),
            "low": quote.get("l"),
            "open": quote.get("o"),
            "previous_close": quote.get("pc")
        }
    
    async def _tool_get_company_profile(self, params: dict) -> dict:
        """Handle get_company_profile tool call."""
        symbol = params["symbol"].upper()
        
        profile = self.finnhub_client.company_profile2(symbol=symbol)
        
        if not profile:
            return {"error": f"No profile data found for symbol: {symbol}"}
        
        return {
            "symbol": symbol,
            "name": profile.get("name"),
            "industry": profile.get("finnhubIndustry"),
            "market_cap": profile.get("marketCapitalization"),
            "country": profile.get("country"),
            "currency": profile.get("currency"),
            "exchange": profile.get("exchange"),
            "ipo_date": profile.get("ipo"),
            "logo": profile.get("logo"),
            "website": profile.get("weburl"),
            "shares_outstanding": profile.get("shareOutstanding")
        }
    
    async def _tool_get_company_news(self, params: dict) -> dict:
        """Handle get_company_news tool call."""
        symbol = params["symbol"].upper()
        days = params.get("days", 7)
        limit = params.get("limit")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        news = self.finnhub_client.company_news(
            symbol,
            _from=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d")
        )
        
        # Apply limit if specified
        if limit and len(news) > limit:
            news = news[:limit]
        
        # Format news articles
        articles = []
        for article in news:
            articles.append({
                "headline": article.get("headline"),
                "summary": article.get("summary"),
                "source": article.get("source"),
                "url": article.get("url"),
                "datetime": datetime.fromtimestamp(
                    article.get("datetime", 0)
                ).isoformat() if article.get("datetime") else None,
                "image": article.get("image")
            })
        
        return {
            "symbol": symbol,
            "news": articles
        }
    
    async def _tool_get_market_news(self, params: dict) -> dict:
        """Handle get_market_news tool call."""
        category = params.get("category", "general")
        limit = params.get("limit")
        
        news = self.finnhub_client.general_news(category)
        
        # Apply limit if specified
        if limit and len(news) > limit:
            news = news[:limit]
        
        # Format news articles
        articles = []
        for article in news:
            articles.append({
                "headline": article.get("headline"),
                "summary": article.get("summary"),
                "source": article.get("source"),
                "url": article.get("url"),
                "datetime": datetime.fromtimestamp(
                    article.get("datetime", 0)
                ).isoformat() if article.get("datetime") else None,
                "category": article.get("category")
            })
        
        return {
            "category": category,
            "news": articles
        }
    
    async def _tool_get_analyst_recommendations(self, params: dict) -> dict:
        """Handle get_analyst_recommendations tool call."""
        symbol = params["symbol"].upper()
        
        recommendations = self.finnhub_client.recommendation_trends(symbol)
        
        if not recommendations:
            return {"error": f"No recommendations found for symbol: {symbol}"}
        
        # Get most recent recommendation
        latest = recommendations[0] if recommendations else {}
        
        return {
            "symbol": symbol,
            "period": latest.get("period"),
            "recommendations": {
                "strong_buy": latest.get("strongBuy", 0),
                "buy": latest.get("buy", 0),
                "hold": latest.get("hold", 0),
                "sell": latest.get("sell", 0),
                "strong_sell": latest.get("strongSell", 0)
            },
            "history": [
                {
                    "period": r.get("period"),
                    "strong_buy": r.get("strongBuy", 0),
                    "buy": r.get("buy", 0),
                    "hold": r.get("hold", 0),
                    "sell": r.get("sell", 0),
                    "strong_sell": r.get("strongSell", 0)
                }
                for r in recommendations[:4]  # Last 4 periods
            ]
        }
    
    async def _tool_get_basic_financials(self, params: dict) -> dict:
        """Handle get_basic_financials tool call."""
        symbol = params["symbol"].upper()
        
        financials = self.finnhub_client.company_basic_financials(symbol, "all")
        
        if not financials or not financials.get("metric"):
            return {"error": f"No financial data found for symbol: {symbol}"}
        
        metrics = financials.get("metric", {})
        
        return {
            "symbol": symbol,
            "52_week_high": metrics.get("52WeekHigh"),
            "52_week_low": metrics.get("52WeekLow"),
            "pe_ratio": metrics.get("peNormalizedAnnual"),
            "market_cap": metrics.get("marketCapitalization"),
            "dividend_yield": metrics.get("dividendYieldIndicatedAnnual"),
            "eps": metrics.get("epsNormalizedAnnual"),
            "pb_ratio": metrics.get("pbAnnual"),
            "ps_ratio": metrics.get("psAnnual"),
            "revenue_per_share": metrics.get("revenuePerShareAnnual"),
            "beta": metrics.get("beta")
        }
    
    async def _tool_get_company_peers(self, params: dict) -> dict:
        """Handle get_company_peers tool call."""
        symbol = params["symbol"].upper()
        
        peers = self.finnhub_client.company_peers(symbol)
        
        # Filter out the symbol itself from peers
        peers = [p for p in peers if p != symbol]
        
        return {
            "symbol": symbol,
            "peers": peers
        }
    
    async def _tool_get_earnings_history(self, params: dict) -> dict:
        """Handle get_earnings_history tool call."""
        symbol = params["symbol"].upper()
        
        earnings = self.finnhub_client.company_earnings(symbol, limit=4)
        
        if not earnings:
            return {"error": f"No earnings data found for symbol: {symbol}"}
        
        # Format earnings data
        earnings_list = []
        for e in earnings:
            earnings_list.append({
                "period": e.get("period"),
                "quarter": e.get("quarter"),
                "year": e.get("year"),
                "actual": e.get("actual"),
                "estimate": e.get("estimate"),
                "surprise": e.get("surprise"),
                "surprise_percent": e.get("surprisePercent")
            })
        
        return {
            "symbol": symbol,
            "earnings": earnings_list
        }
    
    # ===================
    # OpenAI Tool Handlers
    # ===================
    
    async def _tool_web_search(self, params: dict) -> dict:
        """Handle web_search tool call using OpenAI web search."""
        query = params["query"]
        domains = params.get("domains", [])
        
        # Build the web search request
        tools = [{"type": "web_search_preview"}]
        
        # Add domain filtering if specified
        if domains:
            tools[0]["web_search_preview"] = {
                "search_context_size": "medium",
                "user_location": {"type": "approximate", "approximate": {"country": "US"}}
            }
        
        try:
            # Create the search request using the Responses API
            response = await asyncio.to_thread(
                self.openai_client.responses.create,
                model="gpt-4o",
                tools=tools,
                input=f"Search the web for the latest information about: {query}"
            )
            
            # Extract answer and citations from the response
            answer = ""
            citations = []
            
            # Handle different response formats
            if hasattr(response, "output_text") and response.output_text:
                answer = response.output_text
            elif hasattr(response, "output"):
                for output in response.output:
                    if hasattr(output, "type"):
                        if output.type == "message":
                            content = getattr(output, "content", [])
                            for item in content:
                                item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
                                if item_type == "output_text":
                                    answer = getattr(item, "text", "") or (item.get("text", "") if isinstance(item, dict) else "")
                                    # Extract citations from annotations
                                    annotations = getattr(item, "annotations", []) or (item.get("annotations", []) if isinstance(item, dict) else [])
                                    for annotation in annotations:
                                        ann_type = getattr(annotation, "type", None) or (annotation.get("type") if isinstance(annotation, dict) else None)
                                        if ann_type == "url_citation":
                                            citations.append({
                                                "url": getattr(annotation, "url", "") or annotation.get("url", ""),
                                                "title": getattr(annotation, "title", "") or annotation.get("title", "")
                                            })
                                elif item_type == "text":
                                    answer = getattr(item, "text", "") or (item.get("text", "") if isinstance(item, dict) else "")
                    elif isinstance(output, dict):
                        if output.get("type") == "message":
                            content = output.get("content", [])
                            for item in content:
                                if item.get("type") == "output_text":
                                    answer = item.get("text", "")
                                elif item.get("type") == "text":
                                    answer = item.get("text", "")
            
            return {
                "query": query,
                "answer": answer if answer else f"Web search completed for: {query}",
                "citations": citations
            }
        except Exception as e:
            # Log the error and return a graceful fallback
            import logging
            logging.getLogger("mister_risker").warning(f"Web search failed: {e}")
            return {
                "query": query,
                "answer": f"Unable to perform web search at this time. Query was: {query}",
                "citations": [],
                "error": str(e)
            }
    
    # ===================
    # Composite Tool Handlers
    # ===================
    
    async def _tool_research_stock(self, params: dict) -> dict:
        """Handle research_stock tool call - comprehensive stock research."""
        symbol = params["symbol"].upper()
        include_web_search = params.get("include_web_search", False)
        
        # Gather all research data in parallel
        tasks = [
            self._tool_get_stock_quote({"symbol": symbol}),
            self._tool_get_company_profile({"symbol": symbol}),
            self._tool_get_basic_financials({"symbol": symbol}),
            self._tool_get_analyst_recommendations({"symbol": symbol}),
            self._tool_get_company_news({"symbol": symbol, "limit": 5}),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build research result
        research = {
            "symbol": symbol,
            "quote": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
            "profile": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
            "financials": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
            "recommendations": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
            "news": results[4] if not isinstance(results[4], Exception) else {"error": str(results[4])},
        }
        
        # Add web search if requested
        if include_web_search:
            try:
                web_result = await self._tool_web_search({
                    "query": f"{symbol} stock analysis outlook {datetime.now().year}",
                    "domains": ["seekingalpha.com", "fool.com", "finance.yahoo.com"]
                })
                research["web_insights"] = web_result.get("answer", "")
            except Exception as e:
                research["web_insights"] = f"Web search failed: {str(e)}"
        
        return research
