"""Predictive Models for Autonomous Trading.

Implements regression and classification models for price prediction
based on ISLR (Introduction to Statistical Learning) principles.

These models complement FinRL's deep RL approach with traditional
statistical methods for robust predictions.
"""

from dataclasses import dataclass
from typing import Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A prediction from a model."""
    direction: str  # "up", "down", "neutral"
    confidence: float  # 0.0 to 1.0
    expected_change_pct: float  # Expected % change
    features_used: list[str]


class RegressionPredictor:
    """Linear regression model for price prediction.
    
    Uses features like price, volume, RSI, MACD to predict
    future price direction.
    """
    
    def __init__(self):
        self.weights: Optional[np.ndarray] = None
        self.is_trained = False
        self.feature_names = ["price", "volume", "rsi", "macd"]
    
    def predict(self, features: list[float]) -> dict:
        """Make a prediction from input features.
        
        Args:
            features: List of [price, volume, rsi, macd]
            
        Returns:
            Dict with 'direction' and 'confidence'
        """
        if not self.is_trained:
            # Use simple heuristics if not trained
            return self._heuristic_predict(features)
        
        X = np.array(features).reshape(1, -1)
        prediction = X @ self.weights
        
        if prediction > 0.5:
            direction = "up"
        elif prediction < -0.5:
            direction = "down"
        else:
            direction = "neutral"
        
        # Confidence based on prediction magnitude
        confidence = min(abs(prediction[0]) / 2.0, 1.0)
        
        return {
            "direction": direction,
            "confidence": float(confidence),
            "expected_change_pct": float(prediction[0]) * 0.01,
            "features_used": self.feature_names
        }
    
    def _heuristic_predict(self, features: list[float]) -> dict:
        """Simple heuristic prediction when model isn't trained."""
        price, volume, rsi, macd = features
        
        # Simple RSI-based prediction
        if rsi < 30:
            direction = "up"  # Oversold
            confidence = (30 - rsi) / 30
        elif rsi > 70:
            direction = "down"  # Overbought
            confidence = (rsi - 70) / 30
        else:
            direction = "neutral"
            confidence = 0.5
        
        # MACD adjustment
        if macd > 0 and direction != "down":
            confidence = min(confidence + 0.1, 1.0)
        elif macd < 0 and direction != "up":
            confidence = min(confidence + 0.1, 1.0)
        
        return {
            "direction": direction,
            "confidence": confidence,
            "expected_change_pct": confidence * 0.02 * (1 if direction == "up" else -1),
            "features_used": self.feature_names
        }
    
    def train(self, training_data: list[tuple[list[float], int]]):
        """Train the model on historical data.
        
        Args:
            training_data: List of (features, target) tuples
                          where target is 1 (up), -1 (down), or 0 (neutral)
        """
        if len(training_data) < 3:
            logger.warning("Not enough training data")
            return
        
        X = np.array([item[0] for item in training_data])
        y = np.array([item[1] for item in training_data])
        
        # Simple OLS regression: weights = (X'X)^-1 X'y
        try:
            XtX = X.T @ X
            XtX_inv = np.linalg.inv(XtX + 0.01 * np.eye(X.shape[1]))  # Ridge regularization
            self.weights = XtX_inv @ X.T @ y
            self.is_trained = True
            logger.info("Model trained successfully")
        except np.linalg.LinAlgError:
            logger.error("Failed to train model - singular matrix")


class EnsemblePredictor:
    """Ensemble of multiple predictors for robust predictions.
    
    Combines regression, momentum, and FinRL signals.
    """
    
    def __init__(self):
        self.regression = RegressionPredictor()
        self.weights = {
            "regression": 0.3,
            "momentum": 0.2,
            "finrl": 0.5  # Give more weight to deep RL
        }
    
    def predict(
        self,
        features: list[float],
        finrl_signal: Optional[dict] = None
    ) -> dict:
        """Make an ensemble prediction.
        
        Args:
            features: Price features
            finrl_signal: Optional signal from FinRL agent
            
        Returns:
            Ensemble prediction with confidence
        """
        predictions = []
        
        # Regression prediction
        reg_pred = self.regression.predict(features)
        predictions.append({
            "source": "regression",
            "direction": reg_pred["direction"],
            "confidence": reg_pred["confidence"],
            "weight": self.weights["regression"]
        })
        
        # Momentum prediction (simple price/volume momentum)
        mom_pred = self._momentum_predict(features)
        predictions.append({
            "source": "momentum",
            "direction": mom_pred["direction"],
            "confidence": mom_pred["confidence"],
            "weight": self.weights["momentum"]
        })
        
        # FinRL prediction if available
        if finrl_signal:
            predictions.append({
                "source": "finrl",
                "direction": finrl_signal.get("action", "hold"),
                "confidence": finrl_signal.get("confidence", 0.5),
                "weight": self.weights["finrl"]
            })
        
        return self._aggregate_predictions(predictions)
    
    def _momentum_predict(self, features: list[float]) -> dict:
        """Simple momentum-based prediction."""
        _, volume, rsi, macd = features
        
        # Volume and MACD momentum
        if macd > 1.0:
            direction = "up"
            confidence = min(macd / 3.0, 1.0)
        elif macd < -1.0:
            direction = "down"
            confidence = min(abs(macd) / 3.0, 1.0)
        else:
            direction = "neutral"
            confidence = 0.5
        
        return {"direction": direction, "confidence": confidence}
    
    def _aggregate_predictions(self, predictions: list[dict]) -> dict:
        """Aggregate multiple predictions into one."""
        direction_scores = {"up": 0.0, "down": 0.0, "neutral": 0.0}
        
        for pred in predictions:
            direction = pred["direction"]
            if direction == "buy":
                direction = "up"
            elif direction == "sell":
                direction = "down"
            elif direction == "hold":
                direction = "neutral"
            
            score = pred["confidence"] * pred["weight"]
            direction_scores[direction] = direction_scores.get(direction, 0) + score
        
        # Find dominant direction
        best_direction = max(direction_scores, key=direction_scores.get)
        total_weight = sum(p["weight"] for p in predictions)
        confidence = direction_scores[best_direction] / total_weight if total_weight > 0 else 0.5
        
        return {
            "direction": best_direction,
            "confidence": confidence,
            "predictions": predictions
        }


class FeatureExtractor:
    """Extract features from market data for predictions."""
    
    @staticmethod
    def extract_features(market_data: dict) -> list[float]:
        """Extract prediction features from market data.
        
        Args:
            market_data: Dict with price, volume, indicators
            
        Returns:
            Feature vector [price, volume, rsi, macd]
        """
        return [
            market_data.get("price", 0),
            market_data.get("volume", 0),
            market_data.get("rsi", 50),  # Default neutral RSI
            market_data.get("macd", 0)
        ]
    
    @staticmethod
    def calculate_rsi(prices: list[float], period: int = 14) -> float:
        """Calculate RSI from price history."""
        if len(prices) < period + 1:
            return 50.0  # Neutral if not enough data
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    @staticmethod
    def calculate_macd(
        prices: list[float],
        fast_period: int = 12,
        slow_period: int = 26
    ) -> float:
        """Calculate MACD from price history."""
        if len(prices) < slow_period:
            return 0.0
        
        prices_arr = np.array(prices)
        
        # Exponential moving averages
        fast_ema = FeatureExtractor._ema(prices_arr, fast_period)
        slow_ema = FeatureExtractor._ema(prices_arr, slow_period)
        
        macd = fast_ema - slow_ema
        return float(macd)
    
    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> float:
        """Calculate exponential moving average."""
        alpha = 2 / (period + 1)
        weights = (1 - alpha) ** np.arange(min(len(prices), period))
        weights = weights[::-1]
        weights /= weights.sum()
        
        return float(np.dot(prices[-len(weights):], weights))
