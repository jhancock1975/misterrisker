"""Trading Strategy Service - Intelligent Multi-Source Analysis for Trading Decisions.

This service synthesizes data from multiple sources to generate actionable trading
recommendations with specific limit order prices, entry points, take profit targets,
and stop loss levels.

Data Sources:
- Real-time WebSocket price feeds from Coinbase
- FinRL deep reinforcement learning signals
- Technical analysis (RSI, MACD, support/resistance)
- Historical price patterns
- Intelligent Investor principles (margin of safety, value investing)

Output:
- Specific limit order recommendations
- Entry prices (typically below current for buys, above for sells)
- Take profit targets (based on resistance levels and risk/reward ratio)
- Stop loss levels (based on support levels and volatility)
- Position sizing recommendations
- Risk assessment and confidence levels
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional
import numpy as np

logger = logging.getLogger("mister_risker.strategy")


@dataclass
class TradingLevel:
    """A specific price level for trading."""
    price: float
    type: Literal["entry", "take_profit", "stop_loss", "support", "resistance"]
    confidence: float  # 0.0 to 1.0
    reasoning: str


@dataclass
class LimitOrderRecommendation:
    """A complete limit order recommendation with entry, target, and stop."""
    symbol: str
    action: Literal["buy", "sell", "hold"]
    
    # Price levels
    current_price: float
    entry_price: float
    take_profit_price: float
    stop_loss_price: float
    
    # Risk metrics
    risk_reward_ratio: float
    position_size_percent: float  # Recommended % of portfolio
    confidence: float
    
    # Analysis
    signal_source: str  # Which analysis drove this recommendation
    reasoning: str
    technical_factors: list[str]
    risk_factors: list[str]
    
    # Intelligent Investor principles applied
    margin_of_safety_percent: float
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "current_price": round(self.current_price, 2),
            "entry_price": round(self.entry_price, 2),
            "take_profit_price": round(self.take_profit_price, 2),
            "stop_loss_price": round(self.stop_loss_price, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "position_size_percent": round(self.position_size_percent, 1),
            "confidence": round(self.confidence, 2),
            "signal_source": self.signal_source,
            "reasoning": self.reasoning,
            "technical_factors": self.technical_factors,
            "risk_factors": self.risk_factors,
            "margin_of_safety_percent": round(self.margin_of_safety_percent, 1),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PortfolioStrategy:
    """A complete portfolio strategy with multiple recommendations."""
    recommendations: list[LimitOrderRecommendation]
    overall_outlook: Literal["bullish", "bearish", "neutral"]
    market_conditions: str
    total_risk_percent: float
    diversification_score: float
    summary: str
    timestamp: datetime = field(default_factory=datetime.now)


class TradingStrategyService:
    """Service for generating intelligent trading strategies for ANY tradeable asset.
    
    This service combines multiple data sources and analysis methods to generate
    actionable trading recommendations based on sound investment principles.
    
    Supports:
    - Cryptocurrency (via Coinbase WebSocket real-time data)
    - Stocks/ETFs (via Schwab MCP Server quotes)
    - Any other tradeable asset with available price data
    
    Key Principles (from The Intelligent Investor):
    1. Margin of Safety - Buy below intrinsic value, set entries below current price
    2. Risk Management - Always use stop losses, never risk more than you can afford
    3. Diversification - Don't put all eggs in one basket
    4. Long-term thinking - Focus on value, not speculation
    5. Emotional discipline - Systematic approach, not emotional trading
    """
    
    # Default risk parameters (conservative, per Intelligent Investor)
    DEFAULT_MARGIN_OF_SAFETY = 0.03  # 3% below current for buy entries
    DEFAULT_STOP_LOSS_PERCENT = 0.05  # 5% max loss
    DEFAULT_TAKE_PROFIT_MULTIPLIER = 2.0  # 2:1 risk/reward minimum
    MAX_POSITION_SIZE = 0.10  # Max 10% of portfolio per position
    MAX_TOTAL_RISK = 0.25  # Max 25% of portfolio at risk
    
    def __init__(
        self,
        websocket_service: Optional[Any] = None,
        finrl_service: Optional[Any] = None,
        researcher_agent: Optional[Any] = None,
        schwab_mcp_server: Optional[Any] = None,
    ):
        """Initialize the Trading Strategy Service.
        
        Args:
            websocket_service: CoinbaseWebSocketService for real-time crypto data
            finrl_service: FinRLService for AI trading signals
            researcher_agent: ResearcherAgent for fundamental analysis
            schwab_mcp_server: SchwabMCPServer for stock quotes
        """
        self.websocket_service = websocket_service
        self.finrl_service = finrl_service
        self.researcher_agent = researcher_agent
        self.schwab_mcp_server = schwab_mcp_server
        
        # Price history for technical analysis (symbol -> list of prices)
        self._price_history: dict[str, list[float]] = {}
        self._candle_history: dict[str, list[dict]] = {}
        
        logger.info("TradingStrategyService initialized")
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Determine if symbol looks like a crypto symbol.
        
        Crypto symbols typically have -USD suffix (BTC-USD, ETH-USD) or
        are well-known crypto tickers.
        """
        symbol_upper = symbol.upper()
        
        # Crypto pairs have -USD, -USDT, -BTC suffix
        if any(suffix in symbol_upper for suffix in ['-USD', '-USDT', '-BTC', '-ETH']):
            return True
        
        # Common crypto base symbols (without suffix)
        common_crypto = {'BTC', 'ETH', 'SOL', 'XRP', 'ZEC', 'DOGE', 'ADA', 'DOT', 
                        'LINK', 'LTC', 'BCH', 'AVAX', 'SHIB', 'MATIC', 'UNI'}
        if symbol_upper in common_crypto:
            return True
        
        return False
    
    def _normalize_symbol(self, symbol: str, asset_type: str = "unknown") -> str:
        """Normalize symbol format based on asset type.
        
        Crypto: Ensure -USD suffix (BTC -> BTC-USD)
        Stocks: Keep as-is (AAPL stays AAPL)
        """
        symbol_upper = symbol.upper().strip()
        
        if asset_type == "crypto" or self._is_crypto_symbol(symbol_upper):
            # Add -USD suffix if not present
            if not any(suffix in symbol_upper for suffix in ['-USD', '-USDT', '-BTC']):
                return f"{symbol_upper}-USD"
            return symbol_upper
        
        # Stock: just uppercase
        return symbol_upper
    
    async def _get_market_data(self, symbol: str, asset_type: str = "unknown") -> Optional[dict]:
        """Get market data for any symbol, auto-detecting source.
        
        Returns unified market data dict with keys:
        - current_price: float
        - high: float (day or period high)
        - low: float (day or period low)
        - open: float (optional)
        - volume: float (optional)
        - source: str ("websocket", "schwab", "none")
        """
        normalized = self._normalize_symbol(symbol, asset_type)
        
        # Try crypto WebSocket first if it looks like crypto
        if asset_type == "crypto" or self._is_crypto_symbol(normalized):
            candle = self._get_current_candle(normalized)
            if candle:
                return {
                    "current_price": float(candle.get('close', 0)),
                    "high": float(candle.get('high', 0)),
                    "low": float(candle.get('low', 0)),
                    "open": float(candle.get('open', 0)),
                    "volume": float(candle.get('volume', 0)),
                    "source": "websocket",
                    "symbol": normalized
                }
        
        # Try Schwab for stocks
        if asset_type in ("stock", "unknown") and self.schwab_mcp_server:
            quote = await self._get_stock_quote(symbol.replace('-USD', ''))
            if quote:
                # Parse Schwab quote format
                return self._parse_schwab_quote(quote, symbol)
        
        # No data available
        return None
    
    def _parse_schwab_quote(self, quote: dict, symbol: str) -> Optional[dict]:
        """Parse Schwab quote response into unified format."""
        try:
            # Schwab returns nested quote data
            if isinstance(quote, dict):
                # Handle different response formats
                quote_data = quote.get(symbol, quote)
                if 'quote' in quote_data:
                    quote_data = quote_data['quote']
                
                current = float(quote_data.get('lastPrice', quote_data.get('mark', 0)))
                high = float(quote_data.get('highPrice', quote_data.get('52WeekHigh', current * 1.1)))
                low = float(quote_data.get('lowPrice', quote_data.get('52WeekLow', current * 0.9)))
                
                return {
                    "current_price": current,
                    "high": high,
                    "low": low,
                    "open": float(quote_data.get('openPrice', current)),
                    "volume": float(quote_data.get('totalVolume', 0)),
                    "source": "schwab",
                    "symbol": symbol,
                    "52_week_high": float(quote_data.get('52WeekHigh', high)),
                    "52_week_low": float(quote_data.get('52WeekLow', low)),
                }
        except Exception as e:
            logger.error(f"Error parsing Schwab quote: {e}")
        return None

    async def _get_stock_quote(self, symbol: str) -> Optional[dict]:
        """Get stock quote data from Schwab MCP Server.
        
        Args:
            symbol: Stock symbol (e.g., "MU", "AAPL")
            
        Returns:
            Dictionary with quote data or None
        """
        if not self.schwab_mcp_server:
            logger.warning(f"Schwab MCP server not available for stock quote: {symbol}")
            return None
        
        try:
            # Call Schwab get_quote tool
            quote = await self.schwab_mcp_server.call_tool("get_quote", {"symbol": symbol})
            logger.info(f"Got stock quote for {symbol}: {quote}")
            return quote
        except Exception as e:
            logger.error(f"Error getting stock quote for {symbol}: {e}")
            return None
    
    def _get_current_candle(self, symbol: str) -> Optional[dict]:
        """Get current candle data from WebSocket service (crypto only)."""
        if self.websocket_service:
            return self.websocket_service.get_latest_candle(symbol)
        return None
    
    async def generate_recommendations(
        self,
        symbols: list[str],
        asset_type: str = "unknown",
        risk_tolerance: Literal["conservative", "moderate", "aggressive"] = "conservative"
    ) -> list[LimitOrderRecommendation]:
        """Generate trading recommendations for any list of symbols.
        
        This is the main unified entry point that handles both crypto and stocks.
        
        Args:
            symbols: List of trading symbols (any format)
            asset_type: "crypto", "stock", "mixed", or "unknown"
            risk_tolerance: Risk level for recommendations
            
        Returns:
            List of LimitOrderRecommendation objects
        """
        recommendations = []
        
        for symbol in symbols[:10]:  # Limit to 10 symbols
            try:
                # Get market data from appropriate source
                market_data = await self._get_market_data(symbol, asset_type)
                
                if not market_data or market_data.get("current_price", 0) <= 0:
                    logger.warning(f"No market data for {symbol}")
                    continue
                
                # Generate recommendation using unified logic
                rec = self._generate_recommendation_from_data(
                    symbol=market_data.get("symbol", symbol),
                    market_data=market_data,
                    risk_tolerance=risk_tolerance
                )
                recommendations.append(rec)
                
            except Exception as e:
                logger.error(f"Error generating recommendation for {symbol}: {e}")
                continue
        
        return recommendations
    
    def _generate_recommendation_from_data(
        self,
        symbol: str,
        market_data: dict,
        risk_tolerance: Literal["conservative", "moderate", "aggressive"] = "conservative"
    ) -> LimitOrderRecommendation:
        """Generate a recommendation from unified market data."""
        current_price = market_data["current_price"]
        high = market_data.get("high", current_price * 1.02)
        low = market_data.get("low", current_price * 0.98)
        source = market_data.get("source", "unknown")
        
        # Calculate support/resistance from day range
        support = low * 0.995
        resistance = high * 1.005
        
        # Calculate volatility
        volatility = (high - low) / current_price if current_price > 0 else 0.02
        
        # Determine trend from price position
        mid = (high + low) / 2
        if current_price > mid * 1.01:
            trend = "bullish"
        elif current_price < mid * 0.99:
            trend = "bearish"
        else:
            trend = "neutral"
        
        # Get FinRL signal if available (crypto only)
        finrl_signal = None
        if source == "websocket" and self.finrl_service:
            finrl_signal = self._get_finrl_signal(symbol)
        
        # Determine action
        action = self._determine_action(trend, finrl_signal, volatility)
        
        # Set risk parameters
        if risk_tolerance == "conservative":
            mos = 0.03
            stop_loss_pct = 0.05
            rr_target = 2.5
            max_position = 5.0
        elif risk_tolerance == "moderate":
            mos = 0.02
            stop_loss_pct = 0.07
            rr_target = 2.0
            max_position = 10.0
        else:
            mos = 0.01
            stop_loss_pct = 0.10
            rr_target = 1.5
            max_position = 15.0
        
        # Calculate price levels
        if action == "buy":
            entry_price = current_price * (1 - mos)
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            risk = entry_price - stop_loss_price
            take_profit_price = entry_price + (risk * rr_target)
        elif action == "sell":
            entry_price = current_price * (1 + mos)
            take_profit_price = current_price * (1 - mos * 2)
            stop_loss_price = current_price * (1 + stop_loss_pct)
            risk = stop_loss_price - entry_price
        else:  # hold
            extra_mos = mos * 1.5
            entry_price = current_price * (1 - extra_mos)
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            risk = entry_price - stop_loss_price
            take_profit_price = entry_price + (risk * rr_target)
        
        # Calculate R:R
        if action in ("buy", "hold") and risk > 0:
            risk_reward = (take_profit_price - entry_price) / risk
        else:
            risk_reward = rr_target
        
        # Build factors
        technical_factors = [
            f"Trend: {trend.upper()}",
            f"Volatility: {volatility:.2%}",
            f"Support: ${support:,.2f}",
            f"Resistance: ${resistance:,.2f}",
            f"Data source: {source}"
        ]
        
        if finrl_signal:
            technical_factors.append(
                f"FinRL AI: {finrl_signal['action'].upper()} ({finrl_signal['confidence']:.0%})"
            )
        
        risk_factors = []
        if volatility > 0.05:
            risk_factors.append("High volatility")
        if action == "buy" and trend == "bearish":
            risk_factors.append("Buying against trend")
        
        # Confidence
        confidence = self._calculate_confidence(trend, finrl_signal, volatility)
        
        reasoning = self._generate_reasoning(
            action, trend, finrl_signal, current_price, entry_price,
            take_profit_price, stop_loss_price, mos
        )
        
        return LimitOrderRecommendation(
            symbol=symbol,
            action=action,
            current_price=current_price,
            entry_price=entry_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            risk_reward_ratio=risk_reward,
            position_size_percent=max_position,
            confidence=confidence,
            signal_source=source,
            reasoning=reasoning,
            technical_factors=technical_factors,
            risk_factors=risk_factors if risk_factors else ["Standard market risk"],
            margin_of_safety_percent=mos * 100
        )
    
    def create_portfolio_from_recommendations(
        self,
        recommendations: list[LimitOrderRecommendation]
    ) -> PortfolioStrategy:
        """Create a portfolio strategy from a list of recommendations."""
        buy_count = sum(1 for r in recommendations if r.action == "buy")
        sell_count = sum(1 for r in recommendations if r.action == "sell")
        
        if buy_count > sell_count + 1:
            outlook = "bullish"
            conditions = "Market showing bullish signals."
        elif sell_count > buy_count + 1:
            outlook = "bearish"
            conditions = "Market showing bearish signals."
        else:
            outlook = "neutral"
            conditions = "Mixed signals across assets."
        
        total_risk = sum(
            r.position_size_percent * (1 - r.stop_loss_price / r.entry_price)
            for r in recommendations if r.action == "buy" and r.entry_price > 0
        )
        
        summary = self._generate_portfolio_summary(recommendations, outlook)
        
        return PortfolioStrategy(
            recommendations=recommendations,
            overall_outlook=outlook,
            market_conditions=conditions,
            total_risk_percent=min(total_risk, self.MAX_TOTAL_RISK * 100),
            diversification_score=len(set(r.action for r in recommendations)) / 3.0,
            summary=summary
        )
    
    def _calculate_support_resistance(self, symbol: str, current_price: float) -> tuple[float, float]:
        """Calculate support and resistance levels based on recent price action.
        
        Uses the high/low from WebSocket data and applies buffer zones.
        """
        candle = self._get_current_candle(symbol)
        
        if candle:
            high = float(candle.get('high', current_price * 1.02))
            low = float(candle.get('low', current_price * 0.98))
            
            # Support is below recent low, resistance is above recent high
            support = low * 0.995  # 0.5% below low
            resistance = high * 1.005  # 0.5% above high
            
            return support, resistance
        
        # Fallback: use percentage-based levels
        return current_price * 0.97, current_price * 1.03
    
    def _calculate_volatility(self, symbol: str, current_price: float) -> float:
        """Calculate volatility as percentage range."""
        candle = self._get_current_candle(symbol)
        
        if candle:
            high = float(candle.get('high', current_price))
            low = float(candle.get('low', current_price))
            return (high - low) / current_price
        
        return 0.02  # Default 2% volatility
    
    def _get_finrl_signal(self, symbol: str) -> Optional[dict]:
        """Get FinRL AI trading signal for the symbol."""
        if self.finrl_service:
            try:
                signal = self.finrl_service.get_trading_signal(symbol)
                return {
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "reasoning": signal.reasoning
                }
            except Exception as e:
                logger.warning(f"Could not get FinRL signal for {symbol}: {e}")
        return None
    
    def _determine_trend(self, candle: dict) -> Literal["bullish", "bearish", "neutral"]:
        """Determine trend from candle data."""
        if not candle:
            return "neutral"
        
        open_price = float(candle.get('open', 0))
        close_price = float(candle.get('close', 0))
        
        if close_price > open_price * 1.001:
            return "bullish"
        elif close_price < open_price * 0.999:
            return "bearish"
        return "neutral"
    
    def generate_limit_order_recommendation(
        self,
        symbol: str,
        margin_of_safety: float = None,
        risk_tolerance: Literal["conservative", "moderate", "aggressive"] = "conservative"
    ) -> LimitOrderRecommendation:
        """Generate a complete limit order recommendation for a symbol.
        
        This is the main method that synthesizes all data sources to create
        an actionable trading recommendation.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            margin_of_safety: Override default margin of safety
            risk_tolerance: How aggressive the recommendation should be
        
        Returns:
            LimitOrderRecommendation with entry, take profit, and stop loss
        """
        # Get current market data
        candle = self._get_current_candle(symbol)
        
        if not candle:
            # No data available - return a conservative hold
            return LimitOrderRecommendation(
                symbol=symbol,
                action="hold",
                current_price=0,
                entry_price=0,
                take_profit_price=0,
                stop_loss_price=0,
                risk_reward_ratio=0,
                position_size_percent=0,
                confidence=0,
                signal_source="none",
                reasoning="No market data available. Cannot generate recommendation.",
                technical_factors=[],
                risk_factors=["No data available"],
                margin_of_safety_percent=0
            )
        
        current_price = float(candle.get('close', 0))
        high = float(candle.get('high', current_price))
        low = float(candle.get('low', current_price))
        
        # Calculate technical levels
        support, resistance = self._calculate_support_resistance(symbol, current_price)
        volatility = self._calculate_volatility(symbol, current_price)
        trend = self._determine_trend(candle)
        
        # Get FinRL AI signal
        finrl_signal = self._get_finrl_signal(symbol)
        
        # Determine action based on multiple signals
        action = self._determine_action(trend, finrl_signal, volatility)
        
        # Set risk parameters based on tolerance
        if risk_tolerance == "conservative":
            mos = margin_of_safety or 0.03  # 3% margin
            stop_loss_pct = 0.05  # 5% stop
            rr_target = 2.5  # 2.5:1 reward/risk
            max_position = 0.05  # 5% max position
        elif risk_tolerance == "moderate":
            mos = margin_of_safety or 0.02  # 2% margin
            stop_loss_pct = 0.07  # 7% stop
            rr_target = 2.0  # 2:1 reward/risk
            max_position = 0.10  # 10% max position
        else:  # aggressive
            mos = margin_of_safety or 0.01  # 1% margin
            stop_loss_pct = 0.10  # 10% stop
            rr_target = 1.5  # 1.5:1 reward/risk
            max_position = 0.15  # 15% max position
        
        # Calculate specific price levels
        if action == "buy":
            # Entry below current (margin of safety principle)
            entry_price = current_price * (1 - mos)
            # Stop loss below entry
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            # Take profit based on risk/reward ratio
            risk = entry_price - stop_loss_price
            take_profit_price = entry_price + (risk * rr_target)
            
        elif action == "sell":
            # For sell, we're looking to exit existing positions
            entry_price = current_price * (1 + mos)  # Sell above current
            take_profit_price = current_price * (1 - mos * 2)  # Target lower
            stop_loss_price = current_price * (1 + stop_loss_pct)  # Stop above
            
        else:  # hold - still provide actionable levels for cautious entry
            # For HOLD, we provide more conservative entry points
            # Entry further below current (extra margin of safety)
            extra_mos = mos * 1.5  # 50% more margin for cautious entry
            entry_price = current_price * (1 - extra_mos)
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            risk = entry_price - stop_loss_price
            take_profit_price = entry_price + (risk * rr_target)
        
        # Calculate actual risk/reward ratio
        if action in ("buy", "hold") and stop_loss_price > 0:
            risk = entry_price - stop_loss_price
            reward = take_profit_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0
        elif action == "sell":
            risk_reward = rr_target  # Simplified for sells
        else:
            risk_reward = 0
        
        # Determine confidence based on signal alignment
        confidence = self._calculate_confidence(trend, finrl_signal, volatility)
        
        # Build technical factors list
        technical_factors = [
            f"Trend: {trend.upper()}",
            f"Volatility: {volatility:.2%}",
            f"Support: ${support:,.2f}",
            f"Resistance: ${resistance:,.2f}",
        ]
        
        if finrl_signal:
            technical_factors.append(
                f"FinRL AI Signal: {finrl_signal['action'].upper()} ({finrl_signal['confidence']:.0%})"
            )
        
        # Build risk factors list
        risk_factors = []
        if volatility > 0.05:
            risk_factors.append("High volatility - increased stop loss distance")
        if confidence < 0.5:
            risk_factors.append("Low confidence - consider smaller position")
        if action == "buy" and trend == "bearish":
            risk_factors.append("Buying against trend - higher risk")
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            action, trend, finrl_signal, current_price, entry_price, 
            take_profit_price, stop_loss_price, mos
        )
        
        return LimitOrderRecommendation(
            symbol=symbol,
            action=action,
            current_price=current_price,
            entry_price=entry_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            risk_reward_ratio=risk_reward,
            position_size_percent=max_position * 100,
            confidence=confidence,
            signal_source="multi-source" if finrl_signal else "technical",
            reasoning=reasoning,
            technical_factors=technical_factors,
            risk_factors=risk_factors if risk_factors else ["Standard market risk"],
            margin_of_safety_percent=mos * 100
        )
    
    async def generate_stock_recommendation(
        self,
        symbol: str,
        margin_of_safety: float = None,
        risk_tolerance: Literal["conservative", "moderate", "aggressive"] = "conservative"
    ) -> LimitOrderRecommendation:
        """Generate a limit order recommendation for a STOCK symbol.
        
        Uses Schwab MCP Server to get real-time stock quote data.
        
        Args:
            symbol: Stock symbol (e.g., "MU", "AAPL", "NVDA")
            margin_of_safety: Override default margin of safety
            risk_tolerance: How aggressive the recommendation should be
            
        Returns:
            LimitOrderRecommendation with entry, take profit, and stop loss
        """
        logger.info(f"Generating stock recommendation for {symbol}")
        
        # Get stock quote from Schwab
        quote = await self._get_stock_quote(symbol)
        
        if not quote:
            return LimitOrderRecommendation(
                symbol=symbol,
                action="hold",
                current_price=0,
                entry_price=0,
                take_profit_price=0,
                stop_loss_price=0,
                risk_reward_ratio=0,
                position_size_percent=0,
                confidence=0,
                signal_source="none",
                reasoning=f"Could not get quote data for {symbol} from Schwab. Please check if this is a valid stock symbol and try again.",
                technical_factors=[],
                risk_factors=["No data available"],
                margin_of_safety_percent=0
            )
        
        # Extract price data from Schwab quote
        # Schwab quotes have nested structure with 'quote' containing price data
        quote_data = quote.get(symbol, {}).get('quote', quote.get('quote', quote))
        
        current_price = float(quote_data.get('lastPrice', quote_data.get('mark', 0)))
        high_52w = float(quote_data.get('52WeekHigh', current_price * 1.2))
        low_52w = float(quote_data.get('52WeekLow', current_price * 0.8))
        high_today = float(quote_data.get('highPrice', current_price * 1.02))
        low_today = float(quote_data.get('lowPrice', current_price * 0.98))
        open_price = float(quote_data.get('openPrice', current_price))
        
        if current_price <= 0:
            return LimitOrderRecommendation(
                symbol=symbol,
                action="hold",
                current_price=0,
                entry_price=0,
                take_profit_price=0,
                stop_loss_price=0,
                risk_reward_ratio=0,
                position_size_percent=0,
                confidence=0,
                signal_source="none",
                reasoning=f"Invalid price data for {symbol}. Market may be closed.",
                technical_factors=[],
                risk_factors=["Invalid price data"],
                margin_of_safety_percent=0
            )
        
        # Calculate technical levels
        support = min(low_today, low_52w * 1.05)  # Near 52-week low or today's low
        resistance = max(high_today, high_52w * 0.95)  # Near 52-week high or today's high
        
        # Calculate volatility from daily range
        volatility = (high_today - low_today) / current_price if current_price > 0 else 0.02
        
        # Determine trend based on current vs open
        if current_price > open_price * 1.001:
            trend = "bullish"
        elif current_price < open_price * 0.999:
            trend = "bearish"
        else:
            trend = "neutral"
        
        # For stocks, we don't have FinRL signals (yet), so use fundamental bias
        # If price is near 52-week low, might be undervalued (value investing)
        price_position = (current_price - low_52w) / (high_52w - low_52w) if high_52w > low_52w else 0.5
        
        # Determine action based on value investing principles
        if price_position < 0.3:  # Near 52-week lows - potential value
            action = "buy"
            finrl_signal = {"action": "buy", "confidence": 0.6, "reasoning": "Near 52-week low - potential value"}
        elif price_position > 0.85:  # Near 52-week highs - potentially overvalued
            action = "hold"  # Don't recommend sell without more analysis
            finrl_signal = {"action": "hold", "confidence": 0.5, "reasoning": "Near 52-week high - use caution"}
        else:
            # Use trend for mid-range stocks
            action = "buy" if trend == "bullish" else ("hold" if trend == "neutral" else "hold")
            finrl_signal = {"action": action, "confidence": 0.5, "reasoning": f"Based on intraday {trend} trend"}
        
        # Set risk parameters based on tolerance (stocks typically less volatile than crypto)
        if risk_tolerance == "conservative":
            mos = margin_of_safety or 0.02  # 2% margin for stocks
            stop_loss_pct = 0.05  # 5% stop
            rr_target = 2.5  # 2.5:1 reward/risk
            max_position = 0.05  # 5% max position
        elif risk_tolerance == "moderate":
            mos = margin_of_safety or 0.015  # 1.5% margin
            stop_loss_pct = 0.07  # 7% stop
            rr_target = 2.0  # 2:1 reward/risk
            max_position = 0.10  # 10% max position
        else:  # aggressive
            mos = margin_of_safety or 0.01  # 1% margin
            stop_loss_pct = 0.10  # 10% stop
            rr_target = 1.5  # 1.5:1 reward/risk
            max_position = 0.15  # 15% max position
        
        # Calculate specific price levels
        if action == "buy":
            entry_price = current_price * (1 - mos)
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            risk = entry_price - stop_loss_price
            take_profit_price = entry_price + (risk * rr_target)
        else:  # hold
            extra_mos = mos * 1.5
            entry_price = current_price * (1 - extra_mos)
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            risk = entry_price - stop_loss_price
            take_profit_price = entry_price + (risk * rr_target)
        
        # Calculate risk/reward
        if entry_price > 0 and stop_loss_price > 0:
            risk = entry_price - stop_loss_price
            reward = take_profit_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = 0
        
        # Confidence based on signals
        confidence = 0.5  # Base confidence
        if price_position < 0.3:
            confidence += 0.15  # Boost for value opportunity
        if trend == "bullish":
            confidence += 0.1
        if volatility < 0.03:
            confidence += 0.05  # Lower volatility = more predictable
        confidence = min(confidence, 0.85)
        
        # Build technical factors
        technical_factors = [
            f"Trend: {trend.upper()}",
            f"Intraday Volatility: {volatility:.2%}",
            f"52-Week Range: ${low_52w:,.2f} - ${high_52w:,.2f}",
            f"Price Position in 52W Range: {price_position:.0%}",
            f"Today's Range: ${low_today:,.2f} - ${high_today:,.2f}",
        ]
        
        # Risk factors
        risk_factors = []
        if volatility > 0.04:
            risk_factors.append("Higher than average daily volatility")
        if price_position > 0.8:
            risk_factors.append("Trading near 52-week highs - limited upside potential")
        if confidence < 0.5:
            risk_factors.append("Low confidence - consider smaller position")
        if not risk_factors:
            risk_factors.append("Standard market risk for equities")
        
        # Generate reasoning
        if action == "buy":
            reasoning = (
                f"BUY recommendation for {symbol} based on {trend} trend. "
                f"Stock is trading at {price_position:.0%} of its 52-week range. "
                f"Entry at ${entry_price:,.2f} provides {mos:.1%} margin of safety below current ${current_price:,.2f}. "
                f"Target ${take_profit_price:,.2f}, stop loss at ${stop_loss_price:,.2f}. "
                "Following Intelligent Investor: 'Buy with a margin of safety.'"
            )
        else:
            reasoning = (
                f"HOLD/CAUTIOUS for {symbol}. {finrl_signal['reasoning']}. "
                f"Current price ${current_price:,.2f}. "
                f"If you choose to enter, use cautious limit at ${entry_price:,.2f}. "
                f"Stop at ${stop_loss_price:,.2f}, target ${take_profit_price:,.2f}. "
                "Per Intelligent Investor: 'When uncertain, wait for better opportunities.'"
            )
        
        return LimitOrderRecommendation(
            symbol=symbol,
            action=action,
            current_price=current_price,
            entry_price=entry_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            risk_reward_ratio=risk_reward,
            position_size_percent=max_position * 100,
            confidence=confidence,
            signal_source="schwab-quote",
            reasoning=reasoning,
            technical_factors=technical_factors,
            risk_factors=risk_factors,
            margin_of_safety_percent=mos * 100
        )

    def _determine_action(
        self, 
        trend: str, 
        finrl_signal: Optional[dict],
        volatility: float
    ) -> Literal["buy", "sell", "hold"]:
        """Determine trading action based on multiple signals."""
        signals = []
        
        # Trend signal
        if trend == "bullish":
            signals.append(("buy", 0.3))
        elif trend == "bearish":
            signals.append(("sell", 0.3))
        else:
            signals.append(("hold", 0.3))
        
        # FinRL signal (higher weight)
        if finrl_signal:
            action = finrl_signal["action"]
            confidence = finrl_signal["confidence"]
            signals.append((action, confidence * 0.5))
        
        # High volatility suggests caution
        if volatility > 0.05:
            signals.append(("hold", 0.2))
        
        # Tally votes
        buy_score = sum(w for a, w in signals if a == "buy")
        sell_score = sum(w for a, w in signals if a == "sell")
        hold_score = sum(w for a, w in signals if a == "hold")
        
        if buy_score > sell_score and buy_score > hold_score:
            return "buy"
        elif sell_score > buy_score and sell_score > hold_score:
            return "sell"
        return "hold"
    
    def _calculate_confidence(
        self,
        trend: str,
        finrl_signal: Optional[dict],
        volatility: float
    ) -> float:
        """Calculate overall confidence in the recommendation."""
        confidence = 0.5  # Base confidence
        
        # Boost if trend and FinRL align
        if finrl_signal:
            finrl_action = finrl_signal["action"]
            if (trend == "bullish" and finrl_action == "buy") or \
               (trend == "bearish" and finrl_action == "sell"):
                confidence += 0.2
            confidence += finrl_signal["confidence"] * 0.2
        
        # Reduce confidence for high volatility
        if volatility > 0.05:
            confidence -= 0.1
        elif volatility < 0.02:
            confidence += 0.1
        
        return min(max(confidence, 0.1), 0.95)
    
    def _generate_reasoning(
        self,
        action: str,
        trend: str,
        finrl_signal: Optional[dict],
        current_price: float,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        margin_of_safety: float
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        
        if action == "hold":
            reasoning = (
                f"HOLD/CAUTIOUS BUY - Market signals are mixed ({trend} trend). "
                f"Current price ${current_price:,.2f}. "
                f"If you choose to enter, use cautious entry at ${entry_price:,.2f} "
                f"({margin_of_safety * 1.5:.1%} below current) for extra margin of safety. "
                f"Take profit at ${take_profit:,.2f}, stop loss at ${stop_loss:,.2f}. "
                "Per Intelligent Investor: 'When in doubt, be more conservative.'"
            )
            return reasoning
        
        if action == "buy":
            reasoning = (
                f"BUY recommendation based on {trend} trend"
            )
            if finrl_signal and finrl_signal["action"] == "buy":
                reasoning += f" confirmed by FinRL AI ({finrl_signal['confidence']:.0%} confidence)"
            
            reasoning += (
                f". Entry at ${entry_price:,.2f} provides {margin_of_safety:.1%} margin of safety "
                f"below current price ${current_price:,.2f}. "
                f"Target ${take_profit:,.2f} for profit. "
                f"Stop loss at ${stop_loss:,.2f} to limit downside. "
                "Following Intelligent Investor principle: 'Buy below intrinsic value.'"
            )
            return reasoning
        
        if action == "sell":
            reasoning = (
                f"SELL recommendation based on {trend} trend"
            )
            if finrl_signal and finrl_signal["action"] == "sell":
                reasoning += f" confirmed by FinRL AI ({finrl_signal['confidence']:.0%} confidence)"
            
            reasoning += (
                f". Consider reducing position at ${entry_price:,.2f}. "
                f"Stop loss at ${stop_loss:,.2f} if price rises against you. "
                "Following Intelligent Investor principle: 'Preserve capital.'"
            )
            return reasoning
        
        return "No specific recommendation at this time."
    
    def generate_portfolio_strategy(
        self,
        symbols: list[str] = None,
        risk_tolerance: Literal["conservative", "moderate", "aggressive"] = "conservative"
    ) -> PortfolioStrategy:
        """Generate a complete portfolio strategy across multiple symbols.
        
        Args:
            symbols: List of symbols to analyze (default: BTC, ETH, SOL, XRP, ZEC)
            risk_tolerance: Overall risk tolerance
        
        Returns:
            PortfolioStrategy with recommendations for all symbols
        """
        if symbols is None:
            symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ZEC-USD"]
        
        recommendations = []
        buy_count = 0
        sell_count = 0
        
        for symbol in symbols:
            rec = self.generate_limit_order_recommendation(symbol, risk_tolerance=risk_tolerance)
            recommendations.append(rec)
            
            if rec.action == "buy":
                buy_count += 1
            elif rec.action == "sell":
                sell_count += 1
        
        # Determine overall outlook
        if buy_count > sell_count + 1:
            outlook = "bullish"
            conditions = "Market showing bullish signals across multiple assets."
        elif sell_count > buy_count + 1:
            outlook = "bearish"
            conditions = "Market showing bearish signals. Consider defensive positioning."
        else:
            outlook = "neutral"
            conditions = "Mixed signals across assets. Selective opportunities exist."
        
        # Calculate total risk
        total_risk = sum(
            r.position_size_percent * (1 - r.stop_loss_price / r.entry_price)
            for r in recommendations if r.action == "buy" and r.entry_price > 0
        )
        
        # Simple diversification score (0-1 based on how spread out recommendations are)
        actions = [r.action for r in recommendations]
        unique_actions = len(set(actions))
        diversification = unique_actions / 3.0
        
        # Generate summary
        summary = self._generate_portfolio_summary(recommendations, outlook)
        
        return PortfolioStrategy(
            recommendations=recommendations,
            overall_outlook=outlook,
            market_conditions=conditions,
            total_risk_percent=min(total_risk, self.MAX_TOTAL_RISK * 100),
            diversification_score=diversification,
            summary=summary
        )
    
    def _generate_portfolio_summary(
        self, 
        recommendations: list[LimitOrderRecommendation],
        outlook: str
    ) -> str:
        """Generate a summary of the portfolio strategy."""
        buy_recs = [r for r in recommendations if r.action == "buy"]
        sell_recs = [r for r in recommendations if r.action == "sell"]
        hold_recs = [r for r in recommendations if r.action == "hold"]
        
        summary = f"Portfolio outlook: {outlook.upper()}. "
        
        if buy_recs:
            buy_symbols = ", ".join(r.symbol.replace("-USD", "") for r in buy_recs)
            summary += f"BUY opportunities: {buy_symbols}. "
        
        if sell_recs:
            sell_symbols = ", ".join(r.symbol.replace("-USD", "") for r in sell_recs)
            summary += f"Consider SELLING: {sell_symbols}. "
        
        if hold_recs:
            summary += f"{len(hold_recs)} assets recommend HOLD. "
        
        summary += (
            "All recommendations include margin of safety per Intelligent Investor principles. "
            "Use stop losses and never risk more than you can afford to lose."
        )
        
        return summary
    
    def format_recommendation_text(self, rec: LimitOrderRecommendation) -> str:
        """Format a recommendation as user-friendly text."""
        if rec.action == "hold":
            # For HOLD, still show cautious entry levels
            return f"""📊 **{rec.symbol}** - 🟡 HOLD (Cautious)

💰 **Current Price**: ${rec.current_price:,.2f}

📍 **Cautious Entry (Limit Order)**: ${rec.entry_price:,.2f}
🎯 **Take Profit Target**: ${rec.take_profit_price:,.2f}
🛑 **Stop Loss**: ${rec.stop_loss_price:,.2f}

📈 **Risk/Reward Ratio**: {rec.risk_reward_ratio:.1f}:1
💼 **Position Size**: {rec.position_size_percent:.0f}% of portfolio (conservative)
🎚️ **Confidence**: {rec.confidence:.0%}
🛡️ **Margin of Safety**: {rec.margin_of_safety_percent:.1f}% (extra cautious)

**Technical Factors**:
{chr(10).join('• ' + f for f in rec.technical_factors)}

**Risk Factors**:
{chr(10).join('• ' + f for f in rec.risk_factors)}

**Analysis**: {rec.reasoning}

💡 *Market signals are mixed. If you choose to enter, use the cautious entry point above for extra margin of safety.*

⚠️ *This is AI-generated analysis. Not financial advice. Always do your own research.*"""
        
        action_emoji = "🟢 BUY" if rec.action == "buy" else "🔴 SELL"
        
        return f"""📊 **{rec.symbol}** - {action_emoji}

💰 **Current Price**: ${rec.current_price:,.2f}

📍 **Entry (Limit Order)**: ${rec.entry_price:,.2f}
🎯 **Take Profit**: ${rec.take_profit_price:,.2f}
🛑 **Stop Loss**: ${rec.stop_loss_price:,.2f}

📈 **Risk/Reward Ratio**: {rec.risk_reward_ratio:.1f}:1
💼 **Position Size**: {rec.position_size_percent:.0f}% of portfolio
🎚️ **Confidence**: {rec.confidence:.0%}
🛡️ **Margin of Safety**: {rec.margin_of_safety_percent:.1f}%

**Technical Factors**:
{chr(10).join('• ' + f for f in rec.technical_factors)}

**Risk Factors**:
{chr(10).join('• ' + f for f in rec.risk_factors)}

**Analysis**: {rec.reasoning}

⚠️ *This is AI-generated analysis. Not financial advice. Always do your own research.*"""
    
    def format_portfolio_strategy_text(self, strategy: PortfolioStrategy) -> str:
        """Format a complete portfolio strategy as user-friendly text."""
        outlook_emoji = {
            "bullish": "📈",
            "bearish": "📉", 
            "neutral": "➡️"
        }
        
        header = f"""🤖 **Mister Risker Trading Strategy**

{outlook_emoji.get(strategy.overall_outlook, "📊")} **Overall Outlook**: {strategy.overall_outlook.upper()}
📋 **Market Conditions**: {strategy.market_conditions}
⚖️ **Diversification Score**: {strategy.diversification_score:.0%}

---

"""
        
        recommendations_text = ""
        for rec in strategy.recommendations:
            recommendations_text += self.format_recommendation_text(rec) + "\n\n---\n\n"
        
        footer = f"""📝 **Summary**: {strategy.summary}

⚠️ *Following Intelligent Investor principles: margin of safety, diversification, and risk management. This is AI-generated analysis and not financial advice.*"""
        
        return header + recommendations_text + footer
