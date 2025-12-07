"""FinRL Service - Deep Reinforcement Learning for Trading Decisions.

This module provides a FinRL-based trading decision service that uses
real-time WebSocket data from Coinbase to make buy/sell/hold recommendations.

Uses stable-baselines3 algorithms: PPO, A2C, DDPG, SAC, TD3
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal, Any
from dataclasses import dataclass, field
import json
import os

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger("mister_risker.finrl")


@dataclass
class TradingSignal:
    """A trading signal from the FinRL model."""
    action: Literal["buy", "sell", "hold"]
    confidence: float  # 0.0 to 1.0
    symbol: str
    price: float
    reasoning: str
    model_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "symbol": self.symbol,
            "price": self.price,
            "reasoning": self.reasoning,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat()
        }


class CryptoTradingEnv(gym.Env):
    """Custom Gymnasium environment for crypto trading.
    
    Uses OHLCV candle data to simulate trading.
    State: [open, high, low, close, volume, position, cash, portfolio_value]
    Action: 0=sell, 1=hold, 2=buy (continuous: -1 to 1)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self, 
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,  # 0.1% Coinbase fee
        max_position: float = 1.0,  # Max fraction of portfolio in crypto
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position = max_position
        
        # State space: OHLCV + technical indicators + position info
        # [open, high, low, close, volume, rsi, macd, position, cash_ratio]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        
        # Action space: continuous from -1 (full sell) to 1 (full buy)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self.reset()
    
    def _calculate_indicators(self, idx: int) -> tuple[float, float]:
        """Calculate RSI and MACD at current index."""
        # Simple RSI calculation (14-period)
        if idx < 14:
            rsi = 50.0
        else:
            closes = self.df['close'].iloc[max(0, idx-14):idx+1].values
            deltas = np.diff(closes)
            gains = np.maximum(deltas, 0)
            losses = np.abs(np.minimum(deltas, 0))
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Simple MACD (12, 26 EMA difference normalized)
        if idx < 26:
            macd = 0.0
        else:
            closes = self.df['close'].iloc[max(0, idx-26):idx+1].values
            ema12 = np.mean(closes[-12:])
            ema26 = np.mean(closes)
            macd = (ema12 - ema26) / ema26 * 100  # Percentage
        
        return rsi, macd
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        row = self.df.iloc[self.current_step]
        rsi, macd = self._calculate_indicators(self.current_step)
        
        # Normalize prices by dividing by current close
        close = row['close']
        
        obs = np.array([
            row['open'] / close,
            row['high'] / close,
            row['low'] / close,
            1.0,  # close normalized to 1
            row['volume'] / (self.df['volume'].mean() + 1e-10),  # Normalized volume
            rsi / 100.0,  # RSI normalized to 0-1
            macd / 10.0,  # MACD normalized
            self.position,  # Current position (0-1)
            self.cash / self.initial_balance,  # Cash ratio
        ], dtype=np.float32)
        
        return obs
    
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_balance
        self.position = 0.0  # Fraction of portfolio in crypto
        self.crypto_held = 0.0
        self.portfolio_value = self.initial_balance
        self.trades = []
        
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        action_value = float(action[0])  # -1 to 1
        
        current_price = self.df.iloc[self.current_step]['close']
        
        # Calculate portfolio value before action
        prev_portfolio = self.cash + self.crypto_held * current_price
        
        # Execute action
        if action_value > 0.1:  # Buy
            buy_amount = action_value * self.cash * self.max_position
            fee = buy_amount * self.transaction_fee
            crypto_bought = (buy_amount - fee) / current_price
            self.cash -= buy_amount
            self.crypto_held += crypto_bought
            self.trades.append(("buy", current_price, crypto_bought))
            
        elif action_value < -0.1:  # Sell
            sell_fraction = abs(action_value)
            crypto_to_sell = sell_fraction * self.crypto_held
            sell_value = crypto_to_sell * current_price
            fee = sell_value * self.transaction_fee
            self.cash += sell_value - fee
            self.crypto_held -= crypto_to_sell
            self.trades.append(("sell", current_price, crypto_to_sell))
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Calculate new portfolio value
        if not done:
            new_price = self.df.iloc[self.current_step]['close']
        else:
            new_price = current_price
        
        self.portfolio_value = self.cash + self.crypto_held * new_price
        self.position = (self.crypto_held * new_price) / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Reward: percentage change in portfolio value
        reward = (self.portfolio_value - prev_portfolio) / prev_portfolio * 100
        
        # Add small penalty for excessive trading
        if abs(action_value) > 0.1:
            reward -= 0.01
        
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "crypto_held": self.crypto_held,
            "position": self.position,
            "num_trades": len(self.trades)
        }
        
        return self._get_obs(), reward, done, False, info
    
    def render(self):
        print(f"Step {self.current_step}: Portfolio=${self.portfolio_value:.2f}, "
              f"Position={self.position:.2%}, Trades={len(self.trades)}")


class FinRLService:
    """Service for FinRL-based trading decisions.
    
    Integrates with Mister Risker to provide AI-powered trading signals
    using deep reinforcement learning.
    """
    
    SUPPORTED_ALGORITHMS = ["PPO", "A2C", "SAC", "TD3"]
    MODEL_DIR = Path("models/finrl")
    
    def __init__(
        self,
        websocket_service: Optional[Any] = None,
        default_algorithm: str = "PPO",
        model_dir: Optional[Path] = None
    ):
        """Initialize FinRL Service.
        
        Args:
            websocket_service: CoinbaseWebSocketService for real-time data
            default_algorithm: Default DRL algorithm (PPO, A2C, SAC, TD3)
            model_dir: Directory to save/load trained models
        """
        self.websocket_service = websocket_service
        self.default_algorithm = default_algorithm
        self.model_dir = model_dir or self.MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: dict[str, Any] = {}  # symbol -> trained model
        self.training_history: dict[str, list] = {}
        
        logger.info(f"FinRL Service initialized (algorithm={default_algorithm})")
    
    def _get_historical_data(self, symbol: str, periods: int = 500) -> pd.DataFrame:
        """Get historical candle data for training.
        
        Uses WebSocket candle history or generates synthetic data for training.
        """
        # Try to get data from WebSocket service
        if self.websocket_service:
            candle = self.websocket_service.get_latest_candle(symbol)
            if candle:
                # For now, generate synthetic training data around the current price
                # In production, you'd fetch historical data from an API
                current_price = float(candle.get('close', 50000))
                return self._generate_training_data(current_price, periods)
        
        # Fallback: generate synthetic training data
        return self._generate_training_data(50000.0, periods)
    
    def _generate_training_data(self, base_price: float, periods: int = 500) -> pd.DataFrame:
        """Generate synthetic OHLCV data for training.
        
        Uses random walk with mean reversion to simulate realistic price action.
        In production, replace with actual historical data from Coinbase/Yahoo/etc.
        """
        np.random.seed(42)  # Reproducible for demo
        
        prices = [base_price]
        volumes = []
        
        for i in range(periods - 1):
            # Random walk with mean reversion
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            mean_reversion = (base_price - prices[-1]) / base_price * 0.1
            new_price = prices[-1] * (1 + change + mean_reversion)
            prices.append(max(new_price, base_price * 0.5))  # Floor at 50% of base
            volumes.append(np.random.uniform(1, 10))
        
        volumes.append(np.random.uniform(1, 10))
        
        # Create OHLCV dataframe
        data = []
        for i, price in enumerate(prices):
            volatility = price * 0.01  # 1% intraday volatility
            high = price + np.random.uniform(0, volatility)
            low = price - np.random.uniform(0, volatility)
            open_price = low + np.random.uniform(0, high - low)
            
            data.append({
                'timestamp': datetime.now() - timedelta(hours=periods-i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volumes[i]
            })
        
        return pd.DataFrame(data)
    
    def train_model(
        self,
        symbol: str,
        algorithm: str = None,
        total_timesteps: int = 10000,
        df: pd.DataFrame = None
    ) -> dict:
        """Train a DRL model for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            algorithm: DRL algorithm to use
            total_timesteps: Training steps
            df: Historical OHLCV data (optional, will fetch if not provided)
        
        Returns:
            Training results dict
        """
        algorithm = algorithm or self.default_algorithm
        
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use one of {self.SUPPORTED_ALGORITHMS}")
        
        logger.info(f"Training {algorithm} model for {symbol}...")
        
        # Get training data
        if df is None:
            df = self._get_historical_data(symbol)
        
        # Create environment
        env = CryptoTradingEnv(df)
        env = DummyVecEnv([lambda: env])
        
        # Create model based on algorithm
        if algorithm == "PPO":
            model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4)
        elif algorithm == "A2C":
            model = A2C("MlpPolicy", env, verbose=0, learning_rate=7e-4)
        elif algorithm == "SAC":
            model = SAC("MlpPolicy", env, verbose=0, learning_rate=3e-4)
        elif algorithm == "TD3":
            model = TD3("MlpPolicy", env, verbose=0, learning_rate=1e-3)
        
        # Train
        start_time = datetime.now()
        model.learn(total_timesteps=total_timesteps)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save model
        model_path = self.model_dir / f"{symbol.replace('-', '_')}_{algorithm}.zip"
        model.save(str(model_path))
        
        # Store in memory
        self.models[symbol] = {
            "model": model,
            "algorithm": algorithm,
            "trained_at": datetime.now(),
            "timesteps": total_timesteps
        }
        
        logger.info(f"Model trained and saved to {model_path} ({training_time:.1f}s)")
        
        return {
            "symbol": symbol,
            "algorithm": algorithm,
            "model_path": str(model_path),
            "training_time": training_time,
            "total_timesteps": total_timesteps
        }
    
    def load_model(self, symbol: str, algorithm: str = None) -> bool:
        """Load a pre-trained model from disk."""
        algorithm = algorithm or self.default_algorithm
        model_path = self.model_dir / f"{symbol.replace('-', '_')}_{algorithm}.zip"
        
        if not model_path.exists():
            logger.warning(f"No model found at {model_path}")
            return False
        
        try:
            if algorithm == "PPO":
                model = PPO.load(str(model_path))
            elif algorithm == "A2C":
                model = A2C.load(str(model_path))
            elif algorithm == "SAC":
                model = SAC.load(str(model_path))
            elif algorithm == "TD3":
                model = TD3.load(str(model_path))
            else:
                return False
            
            self.models[symbol] = {
                "model": model,
                "algorithm": algorithm,
                "loaded_at": datetime.now()
            }
            
            logger.info(f"Loaded {algorithm} model for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_trading_signal(
        self,
        symbol: str,
        current_candle: dict = None
    ) -> TradingSignal:
        """Get a trading signal from the trained model.
        
        Args:
            symbol: Trading symbol
            current_candle: Current OHLCV candle data
        
        Returns:
            TradingSignal with buy/sell/hold recommendation
        """
        # Get current candle from WebSocket if not provided
        if current_candle is None and self.websocket_service:
            current_candle = self.websocket_service.get_latest_candle(symbol)
        
        if current_candle is None:
            return TradingSignal(
                action="hold",
                confidence=0.0,
                symbol=symbol,
                price=0.0,
                reasoning="No market data available",
                model_name="none"
            )
        
        current_price = float(current_candle.get('close', 0))
        
        # Check if we have a trained model
        if symbol not in self.models:
            # Try to load from disk
            if not self.load_model(symbol):
                # Train a new model with quick training
                logger.info(f"No model for {symbol}, training quick model...")
                self.train_model(symbol, total_timesteps=5000)
        
        model_info = self.models.get(symbol)
        if not model_info:
            return TradingSignal(
                action="hold",
                confidence=0.0,
                symbol=symbol,
                price=current_price,
                reasoning="Failed to train model",
                model_name="none"
            )
        
        model = model_info["model"]
        algorithm = model_info["algorithm"]
        
        # Create observation from current candle
        obs = self._candle_to_observation(current_candle)
        
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        action_value = float(action[0])
        
        # Convert action to trading signal
        if action_value > 0.3:
            action_type = "buy"
            confidence = min(abs(action_value), 1.0)
            reasoning = f"{algorithm} model recommends buying with {confidence:.0%} conviction based on current price action and technical indicators"
        elif action_value < -0.3:
            action_type = "sell"
            confidence = min(abs(action_value), 1.0)
            reasoning = f"{algorithm} model recommends selling with {confidence:.0%} conviction based on current market conditions"
        else:
            action_type = "hold"
            confidence = 1.0 - abs(action_value)
            reasoning = f"{algorithm} model recommends holding - market conditions are neutral"
        
        return TradingSignal(
            action=action_type,
            confidence=confidence,
            symbol=symbol,
            price=current_price,
            reasoning=reasoning,
            model_name=algorithm
        )
    
    def _candle_to_observation(self, candle: dict) -> np.ndarray:
        """Convert a candle dict to model observation."""
        close = float(candle.get('close', 1))
        open_price = float(candle.get('open', close))
        high = float(candle.get('high', close))
        low = float(candle.get('low', close))
        volume = float(candle.get('volume', 1))
        
        # Create normalized observation matching CryptoTradingEnv
        obs = np.array([
            open_price / close,
            high / close,
            low / close,
            1.0,  # close normalized
            volume / 5.0,  # Rough normalization
            0.5,  # RSI placeholder (would need history)
            0.0,  # MACD placeholder
            0.5,  # Assume 50% position
            0.5,  # Assume 50% cash
        ], dtype=np.float32)
        
        return obs.reshape(1, -1)
    
    def get_all_signals(self) -> list[TradingSignal]:
        """Get trading signals for all monitored symbols."""
        if not self.websocket_service:
            return []
        
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ZEC-USD"]
        signals = []
        
        for symbol in symbols:
            candle = self.websocket_service.get_latest_candle(symbol)
            if candle:
                signal = self.get_trading_signal(symbol, candle)
                signals.append(signal)
        
        return signals
    
    def get_portfolio_recommendation(self) -> dict:
        """Get overall portfolio recommendation based on all signals."""
        signals = self.get_all_signals()
        
        if not signals:
            return {
                "recommendation": "No data available",
                "signals": [],
                "summary": "Unable to generate recommendations - no market data"
            }
        
        buy_signals = [s for s in signals if s.action == "buy"]
        sell_signals = [s for s in signals if s.action == "sell"]
        hold_signals = [s for s in signals if s.action == "hold"]
        
        # Generate summary
        if len(buy_signals) > len(sell_signals):
            overall = "BULLISH"
            summary = f"Market outlook is bullish. {len(buy_signals)} buy signals detected."
        elif len(sell_signals) > len(buy_signals):
            overall = "BEARISH"
            summary = f"Market outlook is bearish. {len(sell_signals)} sell signals detected."
        else:
            overall = "NEUTRAL"
            summary = "Market outlook is neutral. Consider holding positions."
        
        return {
            "recommendation": overall,
            "buy_count": len(buy_signals),
            "sell_count": len(sell_signals),
            "hold_count": len(hold_signals),
            "signals": [s.to_dict() for s in signals],
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
