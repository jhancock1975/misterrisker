"""Tests for the Autonomous Trading Service.

This tests the core autonomous trading loop that:
1. Monitors market data continuously
2. Runs predictive models
3. Makes trading decisions
4. Executes trades automatically
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, time, timezone
import asyncio


class TestMarketHours:
    """Test market hours awareness for stocks/options."""
    
    def test_stock_market_is_open_during_trading_hours(self):
        """Stock market should be open 9:30 AM - 4:00 PM ET on weekdays."""
        from src.services.autonomous_trader import MarketHoursService
        
        service = MarketHoursService()
        
        # Monday at 10:00 AM ET - should be open
        monday_10am = datetime(2025, 12, 15, 10, 0, 0)  # Monday
        assert service.is_stock_market_open(monday_10am) is True
        
    def test_stock_market_is_closed_before_open(self):
        """Stock market should be closed before 9:30 AM ET."""
        from src.services.autonomous_trader import MarketHoursService
        
        service = MarketHoursService()
        
        # Monday at 9:00 AM ET - should be closed
        monday_9am = datetime(2025, 12, 15, 9, 0, 0)
        assert service.is_stock_market_open(monday_9am) is False
        
    def test_stock_market_is_closed_after_close(self):
        """Stock market should be closed after 4:00 PM ET."""
        from src.services.autonomous_trader import MarketHoursService
        
        service = MarketHoursService()
        
        # Monday at 5:00 PM ET - should be closed
        monday_5pm = datetime(2025, 12, 15, 17, 0, 0)
        assert service.is_stock_market_open(monday_5pm) is False
        
    def test_stock_market_is_closed_on_weekends(self):
        """Stock market should be closed on weekends."""
        from src.services.autonomous_trader import MarketHoursService
        
        service = MarketHoursService()
        
        # Saturday at 10:00 AM ET - should be closed
        saturday = datetime(2025, 12, 13, 10, 0, 0)  # Saturday
        assert service.is_stock_market_open(saturday) is False
        
    def test_crypto_market_is_always_open(self):
        """Crypto markets are open 24/7."""
        from src.services.autonomous_trader import MarketHoursService
        
        service = MarketHoursService()
        
        # Any time should be open for crypto
        saturday_3am = datetime(2025, 12, 13, 3, 0, 0)
        assert service.is_crypto_market_open(saturday_3am) is True


class TestAutonomousTradingConfig:
    """Test autonomous trading configuration."""
    
    def test_default_config_has_safety_limits(self):
        """Default config should have conservative safety limits."""
        from src.services.autonomous_trader import TradingConfig
        
        config = TradingConfig()
        
        assert config.max_position_size_pct <= 0.10  # Max 10% per position
        assert config.max_daily_loss_pct <= 0.05  # Max 5% daily loss
        assert config.min_confidence_threshold >= 0.6  # At least 60% confidence
        
    def test_config_can_enable_paper_trading(self):
        """Config should support paper trading mode."""
        from src.services.autonomous_trader import TradingConfig
        
        config = TradingConfig(paper_trading=True)
        
        assert config.paper_trading is True
        
    def test_config_can_set_trading_assets(self):
        """Config should allow specifying which assets to trade."""
        from src.services.autonomous_trader import TradingConfig
        
        config = TradingConfig(
            crypto_symbols=["BTC-USD", "ETH-USD"],
            stock_symbols=["AAPL", "TSLA"]
        )
        
        assert "BTC-USD" in config.crypto_symbols
        assert "AAPL" in config.stock_symbols


class TestAutonomousTrader:
    """Test the autonomous trading service."""
    
    @pytest.fixture
    def mock_coinbase_agent(self):
        """Create a mock Coinbase agent."""
        agent = MagicMock()
        agent.process = AsyncMock(return_value="Order placed")
        agent.get_balances = AsyncMock(return_value={"USD": 10000, "BTC": 0.1})
        return agent
    
    @pytest.fixture
    def mock_schwab_agent(self):
        """Create a mock Schwab agent."""
        agent = MagicMock()
        agent.process = AsyncMock(return_value="Order placed")
        agent.get_account_info = AsyncMock(return_value={"balance": 50000})
        return agent
    
    @pytest.fixture
    def mock_finrl_agent(self):
        """Create a mock FinRL agent."""
        agent = MagicMock()
        agent.finrl_service = MagicMock()
        agent.finrl_service.get_trading_signal = MagicMock(return_value=MagicMock(
            action="buy",
            confidence=0.85,
            symbol="BTC-USD",
            price=90000,
            reasoning="Strong upward momentum"
        ))
        return agent
    
    @pytest.mark.asyncio
    async def test_trader_initializes_with_agents(
        self, mock_coinbase_agent, mock_schwab_agent, mock_finrl_agent
    ):
        """Trader should initialize with required agents."""
        from src.services.autonomous_trader import AutonomousTrader, TradingConfig
        
        config = TradingConfig(paper_trading=True)
        trader = AutonomousTrader(
            config=config,
            coinbase_agent=mock_coinbase_agent,
            schwab_agent=mock_schwab_agent,
            finrl_agent=mock_finrl_agent
        )
        
        assert trader.config.paper_trading is True
        assert trader.is_running is False
        
    @pytest.mark.asyncio
    async def test_trader_can_start_and_stop(
        self, mock_coinbase_agent, mock_schwab_agent, mock_finrl_agent
    ):
        """Trader should be able to start and stop the trading loop."""
        from src.services.autonomous_trader import AutonomousTrader, TradingConfig
        
        config = TradingConfig(paper_trading=True, trading_interval_seconds=0.1)
        trader = AutonomousTrader(
            config=config,
            coinbase_agent=mock_coinbase_agent,
            schwab_agent=mock_schwab_agent,
            finrl_agent=mock_finrl_agent
        )
        
        # Start trading
        task = asyncio.create_task(trader.start())
        await asyncio.sleep(0.2)  # Let it run briefly
        
        assert trader.is_running is True
        
        # Stop trading
        await trader.stop()
        assert trader.is_running is False
        
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
    @pytest.mark.asyncio
    async def test_trader_generates_signals(
        self, mock_coinbase_agent, mock_schwab_agent, mock_finrl_agent
    ):
        """Trader should generate trading signals from FinRL."""
        from src.services.autonomous_trader import AutonomousTrader, TradingConfig
        
        config = TradingConfig(
            paper_trading=True,
            crypto_symbols=["BTC-USD"]
        )
        trader = AutonomousTrader(
            config=config,
            coinbase_agent=mock_coinbase_agent,
            schwab_agent=mock_schwab_agent,
            finrl_agent=mock_finrl_agent
        )
        
        signals = await trader.generate_signals()
        
        assert len(signals) > 0
        assert signals[0].symbol == "BTC-USD"
        
    @pytest.mark.asyncio
    async def test_trader_respects_confidence_threshold(
        self, mock_coinbase_agent, mock_schwab_agent, mock_finrl_agent
    ):
        """Trader should only act on signals above confidence threshold."""
        from src.services.autonomous_trader import AutonomousTrader, TradingConfig
        
        # Set high confidence threshold
        config = TradingConfig(
            paper_trading=True,
            min_confidence_threshold=0.95,
            crypto_symbols=["BTC-USD"]
        )
        
        # Signal has 0.85 confidence (below threshold)
        trader = AutonomousTrader(
            config=config,
            coinbase_agent=mock_coinbase_agent,
            schwab_agent=mock_schwab_agent,
            finrl_agent=mock_finrl_agent
        )
        
        signals = await trader.generate_signals()
        actionable = trader.filter_actionable_signals(signals)
        
        # Should not act on 0.85 confidence when threshold is 0.95
        assert len(actionable) == 0


class TestPredictiveModels:
    """Test predictive model integration."""
    
    def test_regression_model_can_predict(self):
        """Regression model should make predictions from features."""
        from src.services.predictive_models import RegressionPredictor
        
        predictor = RegressionPredictor()
        
        # Simple features: [price, volume, rsi, macd]
        features = [90000, 1000000, 45, 0.5]
        prediction = predictor.predict(features)
        
        assert prediction is not None
        assert "direction" in prediction  # up/down/neutral
        assert "confidence" in prediction
        
    def test_model_uses_historical_data(self):
        """Model should be trainable on historical data."""
        from src.services.predictive_models import RegressionPredictor
        
        predictor = RegressionPredictor()
        
        # Historical data: list of (features, target) tuples
        training_data = [
            ([90000, 1000000, 45, 0.5], 1),  # price went up
            ([89000, 900000, 35, -0.5], -1),  # price went down
            ([91000, 1100000, 55, 1.0], 1),  # price went up
        ]
        
        predictor.train(training_data)
        
        assert predictor.is_trained is True
