#!/usr/bin/env python3
"""
Mister Risker - Autonomous Trading System

This script runs the autonomous trading loop that:
1. Monitors market data continuously via WebSockets
2. Runs predictive models (FinRL + regression)
3. Makes trading decisions based on signals
4. Executes trades automatically with safety limits

Usage:
    # Paper trading mode (default, safe for testing):
    python scripts/run_autonomous_trader.py
    
    # Paper trading with custom config:
    python scripts/run_autonomous_trader.py --interval 30 --confidence 0.8
    
    # Live trading (CAUTION: uses real money):
    python scripts/run_autonomous_trader.py --live

Environment Variables Required:
    COINBASE_API_KEY - Coinbase CDP API key
    COINBASE_API_SECRET - Coinbase CDP API secret
    OPENAI_API_KEY - OpenAI API key for LLM
    
Optional:
    SCHWAB_API_KEY - Schwab API key (for stock trading)
    SCHWAB_API_SECRET - Schwab API secret
"""

import asyncio
import argparse
import logging
import os
import signal
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.autonomous_trader import AutonomousTrader, TradingConfig
from services.coinbase_websocket import CoinbaseWebSocketService
from agents.finrl_agent import FinRLAgent
from agents.coinbase_agent import CoinbaseAgent
from agents.schwab_agent import SchwabAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("misterrisker.autonomous")


class AutonomousTradingApp:
    """Main application for autonomous trading."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.trader: AutonomousTrader | None = None
        self.websocket_service: CoinbaseWebSocketService | None = None
        self._shutdown_event = asyncio.Event()
        
    async def start(self):
        """Start the autonomous trading system."""
        logger.info("=" * 60)
        logger.info("MISTER RISKER - AUTONOMOUS TRADING SYSTEM")
        logger.info("=" * 60)
        
        if self.config.paper_trading:
            logger.info("🧪 Running in PAPER TRADING mode (no real money)")
        else:
            logger.warning("💰 Running in LIVE TRADING mode (REAL MONEY)")
        
        logger.info(f"Trading interval: {self.config.trading_interval_seconds}s")
        logger.info(f"Crypto symbols: {self.config.crypto_symbols}")
        logger.info(f"Stock symbols: {self.config.stock_symbols}")
        logger.info(f"Min confidence: {self.config.min_confidence_threshold}")
        logger.info(f"Max position size: {self.config.max_position_size_pct * 100}%")
        logger.info("=" * 60)
        
        # Initialize agents
        try:
            coinbase_agent = CoinbaseAgent()
            logger.info("✅ Coinbase agent initialized")
        except Exception as e:
            logger.warning(f"⚠️ Coinbase agent not available: {e}")
            coinbase_agent = None
            
        try:
            schwab_agent = SchwabAgent()
            logger.info("✅ Schwab agent initialized")
        except Exception as e:
            logger.warning(f"⚠️ Schwab agent not available: {e}")
            schwab_agent = None
            
        try:
            finrl_agent = FinRLAgent()
            logger.info("✅ FinRL agent initialized")
        except Exception as e:
            logger.error(f"❌ FinRL agent failed: {e}")
            return
        
        # Initialize WebSocket service for real-time data
        try:
            self.websocket_service = CoinbaseWebSocketService(
                products=self.config.crypto_symbols,
                on_candle=self._on_candle_update
            )
            logger.info("✅ WebSocket service initialized")
        except Exception as e:
            logger.warning(f"⚠️ WebSocket service not available: {e}")
            self.websocket_service = None
        
        # Create the autonomous trader
        self.trader = AutonomousTrader(
            config=self.config,
            coinbase_agent=coinbase_agent,
            schwab_agent=schwab_agent,
            finrl_agent=finrl_agent,
            websocket_service=self.websocket_service
        )
        
        # Start WebSocket in background
        websocket_task = None
        if self.websocket_service:
            websocket_task = asyncio.create_task(
                self.websocket_service.start()
            )
            logger.info("📡 WebSocket connection started")
        
        # Start the trading loop
        trading_task = asyncio.create_task(self.trader.start())
        logger.info("🚀 Autonomous trading loop started")
        
        # Wait for shutdown signal
        await self._shutdown_event.wait()
        
        # Clean shutdown
        logger.info("Shutting down...")
        await self.trader.stop()
        
        if self.websocket_service:
            await self.websocket_service.stop()
        
        if websocket_task:
            websocket_task.cancel()
            try:
                await websocket_task
            except asyncio.CancelledError:
                pass
        
        trading_task.cancel()
        try:
            await trading_task
        except asyncio.CancelledError:
            pass
        
        logger.info("Shutdown complete")
        
    def _on_candle_update(self, candle_data: dict):
        """Handle real-time candle updates from WebSocket."""
        # This can be used to trigger immediate analysis
        # For now, just log at debug level
        logger.debug(f"Candle update: {candle_data.get('product_id')}")
        
    def shutdown(self):
        """Signal the application to shut down."""
        self._shutdown_event.set()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mister Risker Autonomous Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (CAUTION: uses real money)"
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Trading check interval in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for trades (default: 0.7)"
    )
    
    parser.add_argument(
        "--max-position",
        type=float,
        default=0.05,
        help="Maximum position size as fraction of portfolio (default: 0.05)"
    )
    
    parser.add_argument(
        "--crypto",
        nargs="+",
        default=["BTC-USD", "ETH-USD"],
        help="Crypto symbols to trade (default: BTC-USD ETH-USD)"
    )
    
    parser.add_argument(
        "--stocks",
        nargs="*",
        default=[],
        help="Stock symbols to trade (default: none)"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Build config from args
    config = TradingConfig(
        paper_trading=not args.live,
        trading_interval_seconds=args.interval,
        min_confidence_threshold=args.confidence,
        max_position_size_pct=args.max_position,
        crypto_symbols=args.crypto,
        stock_symbols=args.stocks
    )
    
    # Create the app
    app = AutonomousTradingApp(config)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        app.shutdown()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the app
    await app.start()


if __name__ == "__main__":
    asyncio.run(main())
