"""Autonomous Trading Service for Mister Risker.

This service implements the core autonomous trading loop that:
1. Monitors market data continuously via WebSockets
2. Runs predictive models (FinRL + regression)
3. Makes trading decisions based on signals
4. Executes trades automatically with safety limits
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Optional
import asyncio
import logging
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


# US Eastern timezone for stock market hours
ET = ZoneInfo("America/New_York")


@dataclass
class TradingSignal:
    """A trading signal from the prediction models."""
    symbol: str
    action: str  # "buy", "sell", "hold"
    confidence: float  # 0.0 to 1.0
    price: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    asset_type: str = "crypto"  # "crypto" or "stock"


@dataclass
class TradingConfig:
    """Configuration for the autonomous trader."""
    
    # Paper trading mode (no real money)
    paper_trading: bool = True
    
    # Safety limits
    max_position_size_pct: float = 0.05  # Max 5% per position
    max_daily_loss_pct: float = 0.02  # Max 2% daily loss
    min_confidence_threshold: float = 0.7  # Need 70% confidence to trade
    
    # Trading parameters
    trading_interval_seconds: float = 60.0  # Check for signals every minute
    
    # Assets to trade
    crypto_symbols: list[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])
    stock_symbols: list[str] = field(default_factory=lambda: [])
    
    # Risk management
    max_open_positions: int = 5
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit


class MarketHoursService:
    """Service to track market hours for different asset classes."""
    
    # US Stock market hours (Eastern Time)
    STOCK_MARKET_OPEN = time(9, 30)  # 9:30 AM ET
    STOCK_MARKET_CLOSE = time(16, 0)  # 4:00 PM ET
    
    # Pre-market and after-hours
    PRE_MARKET_OPEN = time(4, 0)  # 4:00 AM ET
    AFTER_HOURS_CLOSE = time(20, 0)  # 8:00 PM ET
    
    def is_stock_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Check if the stock market is open.
        
        Regular trading hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        """
        if dt is None:
            dt = datetime.now(ET)
        
        # Convert to Eastern Time if not already
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        else:
            dt = dt.astimezone(ET)
        
        # Check if weekday (Monday=0, Sunday=6)
        if dt.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check time
        current_time = dt.time()
        return self.STOCK_MARKET_OPEN <= current_time < self.STOCK_MARKET_CLOSE
    
    def is_extended_hours_open(self, dt: Optional[datetime] = None) -> bool:
        """Check if extended trading hours are available.
        
        Pre-market: 4:00 AM - 9:30 AM ET
        After-hours: 4:00 PM - 8:00 PM ET
        """
        if dt is None:
            dt = datetime.now(ET)
        
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        else:
            dt = dt.astimezone(ET)
        
        # Check if weekday
        if dt.weekday() >= 5:
            return False
        
        current_time = dt.time()
        
        # Pre-market or after-hours
        return (self.PRE_MARKET_OPEN <= current_time < self.STOCK_MARKET_OPEN or
                self.STOCK_MARKET_CLOSE <= current_time < self.AFTER_HOURS_CLOSE)
    
    def is_crypto_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Crypto markets are open 24/7."""
        return True
    
    def can_trade(self, asset_type: str, dt: Optional[datetime] = None) -> bool:
        """Check if we can trade a given asset type right now."""
        if asset_type == "crypto":
            return self.is_crypto_market_open(dt)
        elif asset_type == "stock":
            return self.is_stock_market_open(dt)
        else:
            logger.warning(f"Unknown asset type: {asset_type}")
            return False
    
    def time_until_market_open(self, dt: Optional[datetime] = None) -> Optional[float]:
        """Get seconds until stock market opens. Returns None if market is open."""
        if self.is_stock_market_open(dt):
            return None
        
        if dt is None:
            dt = datetime.now(ET)
        
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        else:
            dt = dt.astimezone(ET)
        
        # Calculate next market open
        next_open = dt.replace(
            hour=self.STOCK_MARKET_OPEN.hour,
            minute=self.STOCK_MARKET_OPEN.minute,
            second=0,
            microsecond=0
        )
        
        # If we're past today's open, move to next trading day
        if dt.time() >= self.STOCK_MARKET_OPEN:
            next_open = next_open.replace(day=next_open.day + 1)
        
        # Skip weekends
        while next_open.weekday() >= 5:
            next_open = next_open.replace(day=next_open.day + 1)
        
        return (next_open - dt).total_seconds()


class AutonomousTrader:
    """The core autonomous trading service.
    
    This runs a continuous loop that:
    1. Monitors market data via WebSockets
    2. Generates trading signals using FinRL and regression models
    3. Filters signals based on confidence and risk limits
    4. Executes trades via Coinbase and Schwab agents
    """
    
    def __init__(
        self,
        config: TradingConfig,
        coinbase_agent,
        schwab_agent,
        finrl_agent,
        websocket_service=None
    ):
        self.config = config
        self.coinbase_agent = coinbase_agent
        self.schwab_agent = schwab_agent
        self.finrl_agent = finrl_agent
        self.websocket_service = websocket_service
        
        self.market_hours = MarketHoursService()
        self.is_running = False
        
        # Tracking
        self.daily_pnl = 0.0
        self.open_positions: dict[str, dict] = {}
        self.trade_history: list[dict] = []
        
    async def start(self):
        """Start the autonomous trading loop."""
        logger.info("Starting autonomous trading loop...")
        self.is_running = True
        
        while self.is_running:
            try:
                await self._trading_iteration()
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
            
            await asyncio.sleep(self.config.trading_interval_seconds)
    
    async def stop(self):
        """Stop the autonomous trading loop."""
        logger.info("Stopping autonomous trading loop...")
        self.is_running = False
    
    async def _trading_iteration(self):
        """Run one iteration of the trading loop."""
        # Check daily loss limit
        if self._is_daily_loss_exceeded():
            logger.warning("Daily loss limit exceeded. Pausing trading.")
            return
        
        # Generate signals for all configured assets
        signals = await self.generate_signals()
        
        # Filter to actionable signals
        actionable = self.filter_actionable_signals(signals)
        
        # Execute trades
        for signal in actionable:
            if self.market_hours.can_trade(signal.asset_type):
                await self._execute_trade(signal)
            else:
                logger.info(f"Market closed for {signal.asset_type}, skipping {signal.symbol}")
    
    async def generate_signals(self) -> list[TradingSignal]:
        """Generate trading signals for all configured assets."""
        signals = []
        
        # Generate crypto signals
        for symbol in self.config.crypto_symbols:
            try:
                signal = await self._get_signal_for_symbol(symbol, "crypto")
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        # Generate stock signals (only during market hours)
        if self.market_hours.is_stock_market_open():
            for symbol in self.config.stock_symbols:
                try:
                    signal = await self._get_signal_for_symbol(symbol, "stock")
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    async def _get_signal_for_symbol(self, symbol: str, asset_type: str) -> Optional[TradingSignal]:
        """Get a trading signal for a specific symbol."""
        # Use FinRL service to generate signal
        if hasattr(self.finrl_agent, 'finrl_service') and self.finrl_agent.finrl_service:
            raw_signal = self.finrl_agent.finrl_service.get_trading_signal(symbol)
            if raw_signal:
                return TradingSignal(
                    symbol=raw_signal.symbol,
                    action=raw_signal.action,
                    confidence=raw_signal.confidence,
                    price=raw_signal.price,
                    reasoning=raw_signal.reasoning,
                    asset_type=asset_type
                )
        return None
    
    def filter_actionable_signals(self, signals: list[TradingSignal]) -> list[TradingSignal]:
        """Filter signals to only those we should act on."""
        actionable = []
        
        for signal in signals:
            # Check confidence threshold
            if signal.confidence < self.config.min_confidence_threshold:
                logger.debug(f"Skipping {signal.symbol}: confidence {signal.confidence} below threshold")
                continue
            
            # Check if we already have max positions
            if (len(self.open_positions) >= self.config.max_open_positions and
                signal.action == "buy"):
                logger.debug(f"Skipping {signal.symbol}: max positions reached")
                continue
            
            # Skip hold signals
            if signal.action == "hold":
                continue
            
            actionable.append(signal)
        
        return actionable
    
    async def _execute_trade(self, signal: TradingSignal):
        """Execute a trade based on a signal."""
        if self.config.paper_trading:
            logger.info(f"[PAPER] Would {signal.action} {signal.symbol} at {signal.price}")
            self._record_paper_trade(signal)
            return
        
        try:
            if signal.asset_type == "crypto":
                await self._execute_crypto_trade(signal)
            else:
                await self._execute_stock_trade(signal)
        except Exception as e:
            logger.error(f"Error executing trade for {signal.symbol}: {e}")
    
    async def _execute_crypto_trade(self, signal: TradingSignal):
        """Execute a crypto trade via Coinbase."""
        # Calculate position size
        balances = await self.coinbase_agent.get_balances()
        usd_balance = balances.get("USD", 0)
        position_size = usd_balance * self.config.max_position_size_pct
        
        if signal.action == "buy":
            order_msg = f"Buy ${position_size:.2f} of {signal.symbol}"
        else:
            order_msg = f"Sell {signal.symbol}"
        
        result = await self.coinbase_agent.process(order_msg)
        logger.info(f"Trade executed: {result}")
        
        self._record_trade(signal, position_size)
    
    async def _execute_stock_trade(self, signal: TradingSignal):
        """Execute a stock trade via Schwab."""
        account = await self.schwab_agent.get_account_info()
        balance = account.get("balance", 0)
        position_size = balance * self.config.max_position_size_pct
        
        # Calculate shares
        shares = int(position_size / signal.price)
        
        if signal.action == "buy":
            order_msg = f"Buy {shares} shares of {signal.symbol}"
        else:
            order_msg = f"Sell {signal.symbol}"
        
        result = await self.schwab_agent.process(order_msg)
        logger.info(f"Trade executed: {result}")
        
        self._record_trade(signal, position_size)
    
    def _record_trade(self, signal: TradingSignal, size: float):
        """Record a trade in history."""
        trade = {
            "timestamp": datetime.now(),
            "symbol": signal.symbol,
            "action": signal.action,
            "price": signal.price,
            "size": size,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning
        }
        self.trade_history.append(trade)
        
        # Update positions
        if signal.action == "buy":
            self.open_positions[signal.symbol] = trade
        elif signal.symbol in self.open_positions:
            del self.open_positions[signal.symbol]
    
    def _record_paper_trade(self, signal: TradingSignal):
        """Record a paper trade for simulation."""
        self._record_trade(signal, 1000.0)  # Simulated position size
    
    def _is_daily_loss_exceeded(self) -> bool:
        """Check if we've exceeded the daily loss limit."""
        # For now, use a simple check
        # In production, this would track actual P&L
        return self.daily_pnl < -self.config.max_daily_loss_pct
    
    def get_status(self) -> dict:
        """Get the current status of the trader."""
        return {
            "is_running": self.is_running,
            "paper_trading": self.config.paper_trading,
            "open_positions": len(self.open_positions),
            "trades_today": len([t for t in self.trade_history 
                               if t["timestamp"].date() == datetime.now().date()]),
            "daily_pnl": self.daily_pnl,
            "stock_market_open": self.market_hours.is_stock_market_open(),
            "crypto_market_open": self.market_hours.is_crypto_market_open()
        }
