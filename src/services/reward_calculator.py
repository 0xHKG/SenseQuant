"""Reward signal calculation for teacher-student adaptive learning (US-028 Phase 7 Initiative 2).

This module computes reward signals based on realized returns after student predictions
are generated. The reward signals are used to adapt training through sample weighting.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class RewardCalculator:
    """Calculate reward signals from predictions and realized returns."""

    def __init__(
        self,
        reward_horizon_days: int = 5,
        reward_clip_min: float = -2.0,
        reward_clip_max: float = 2.0,
        reward_log_path: Path | None = None,
    ):
        """Initialize reward calculator.

        Args:
            reward_horizon_days: Number of days to look ahead for realized returns
            reward_clip_min: Minimum reward value (clip negative rewards)
            reward_clip_max: Maximum reward value (clip positive rewards)
            reward_log_path: Path to reward_history.jsonl file
        """
        self.reward_horizon_days = reward_horizon_days
        self.reward_clip_min = reward_clip_min
        self.reward_clip_max = reward_clip_max
        self.reward_log_path = reward_log_path

        if reward_log_path:
            reward_log_path.parent.mkdir(parents=True, exist_ok=True)

    def calculate_reward(
        self,
        prediction: int,
        actual_return: float,
        return_magnitude: float | None = None,
    ) -> tuple[float, float]:
        """Calculate reward signal for a single prediction.

        Reward logic:
        - Correct direction prediction: +1 × |return|
        - Incorrect direction prediction: -1 × |return|
        - Clip to [reward_clip_min, reward_clip_max]

        Args:
            prediction: Predicted class (0=down, 1=neutral, 2=up)
            actual_return: Realized return (positive or negative)
            return_magnitude: Optional override for return magnitude (uses abs(actual_return) if None)

        Returns:
            Tuple of (raw_reward, clipped_reward)
        """
        # Convert prediction to direction (-1, 0, +1)
        if prediction == 0:
            pred_direction = -1  # Down
        elif prediction == 1:
            pred_direction = 0  # Neutral
        else:  # prediction == 2
            pred_direction = 1  # Up

        # Determine actual direction
        if actual_return < -0.001:  # Small threshold for noise
            actual_direction = -1
        elif actual_return > 0.001:
            actual_direction = 1
        else:
            actual_direction = 0

        # Calculate magnitude (use provided or absolute return)
        magnitude = return_magnitude if return_magnitude is not None else abs(actual_return)

        # Calculate raw reward
        if pred_direction == 0:
            # Neutral predictions always get zero reward (no signal)
            raw_reward = 0.0
        elif actual_direction == 0:
            # Actual neutral return: no strong signal
            raw_reward = 0.0
        elif pred_direction == actual_direction:
            # Correct prediction: positive reward scaled by magnitude
            raw_reward = 1.0 * magnitude
        else:
            # Incorrect prediction: negative reward scaled by magnitude
            raw_reward = -1.0 * magnitude

        # Clip reward
        clipped_reward = np.clip(raw_reward, self.reward_clip_min, self.reward_clip_max)

        return float(raw_reward), float(clipped_reward)

    def log_reward_entry(
        self,
        symbol: str,
        window: str,
        timestamp: str,
        prediction: int,
        actual_return: float,
        raw_reward: float,
        clipped_reward: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log reward entry to reward_history.jsonl.

        Args:
            symbol: Stock symbol
            window: Training window identifier
            timestamp: Prediction timestamp (ISO format)
            prediction: Predicted class
            actual_return: Realized return
            raw_reward: Raw reward before clipping
            clipped_reward: Clipped reward
            metadata: Additional metadata (e.g., teacher_batch_id, confidence)
        """
        if not self.reward_log_path:
            logger.warning("No reward log path configured, skipping reward logging")
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction_timestamp": timestamp,
            "symbol": symbol,
            "window": window,
            "prediction": prediction,
            "actual_return": actual_return,
            "raw_reward": raw_reward,
            "clipped_reward": clipped_reward,
        }

        if metadata:
            entry.update(metadata)

        # Atomic append write
        with open(self.reward_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def calculate_batch_rewards(
        self,
        predictions_df: pd.DataFrame,
        price_data: pd.DataFrame,
        prediction_col: str = "prediction",
        timestamp_col: str = "timestamp",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """Calculate rewards for a batch of predictions.

        Args:
            predictions_df: DataFrame with predictions (columns: timestamp, prediction, symbol)
            price_data: DataFrame with price data (columns: timestamp, close, symbol)
            prediction_col: Name of prediction column
            timestamp_col: Name of timestamp column
            close_col: Name of close price column

        Returns:
            DataFrame with added reward columns (raw_reward, clipped_reward, actual_return)
        """
        results = []

        # Ensure both dataframes are sorted by timestamp
        predictions_df = predictions_df.sort_values(timestamp_col)
        price_data = price_data.sort_values(timestamp_col)

        for _idx, row in predictions_df.iterrows():
            pred_timestamp = pd.to_datetime(row[timestamp_col])
            symbol = row.get("symbol", "UNKNOWN")

            # Find future price (reward_horizon_days ahead)
            future_prices = price_data[
                (price_data[timestamp_col] > pred_timestamp)
                & (price_data["symbol"] == symbol)
            ]

            if len(future_prices) >= self.reward_horizon_days:
                # Get price at horizon
                future_price = future_prices.iloc[self.reward_horizon_days - 1][close_col]
                current_price = row.get(close_col, None)

                if current_price is not None and current_price > 0:
                    actual_return = (future_price - current_price) / current_price
                else:
                    # If current price not in predictions_df, look it up in price_data
                    current_row = price_data[
                        (price_data[timestamp_col] == pred_timestamp)
                        & (price_data["symbol"] == symbol)
                    ]
                    if len(current_row) > 0:
                        current_price = current_row.iloc[0][close_col]
                        actual_return = (future_price - current_price) / current_price
                    else:
                        actual_return = 0.0  # Fallback

                raw_reward, clipped_reward = self.calculate_reward(
                    row[prediction_col], actual_return
                )

                results.append(
                    {
                        "raw_reward": raw_reward,
                        "clipped_reward": clipped_reward,
                        "actual_return": actual_return,
                    }
                )
            else:
                # Not enough future data
                results.append(
                    {
                        "raw_reward": 0.0,
                        "clipped_reward": 0.0,
                        "actual_return": 0.0,
                    }
                )

        # Create results dataframe
        rewards_df = pd.DataFrame(results)

        # Concatenate with original predictions
        result_df = pd.concat([predictions_df.reset_index(drop=True), rewards_df], axis=1)

        return result_df

    def aggregate_reward_metrics(
        self, rewards: list[float] | pd.Series | np.ndarray
    ) -> dict[str, float]:
        """Aggregate reward metrics for reporting.

        Args:
            rewards: List or array of reward values

        Returns:
            Dictionary with mean, cumulative, volatility, min, max
        """
        rewards_array = np.array(rewards)

        if len(rewards_array) == 0:
            return {
                "mean_reward": 0.0,
                "cumulative_reward": 0.0,
                "reward_volatility": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "positive_rewards": 0,
                "negative_rewards": 0,
            }

        return {
            "mean_reward": float(np.mean(rewards_array)),
            "cumulative_reward": float(np.sum(rewards_array)),
            "reward_volatility": float(np.std(rewards_array)),
            "min_reward": float(np.min(rewards_array)),
            "max_reward": float(np.max(rewards_array)),
            "positive_rewards": int(np.sum(rewards_array > 0)),
            "negative_rewards": int(np.sum(rewards_array < 0)),
        }

    def compute_sample_weights(
        self,
        rewards: np.ndarray | pd.Series,
        mode: str = "linear",
        scale: float = 1.0,
    ) -> np.ndarray:
        """Compute sample weights from rewards for adaptive training.

        Args:
            rewards: Array of reward values
            mode: Weighting mode ("linear", "exponential", "none")
            scale: Scaling factor for weight computation

        Returns:
            Array of sample weights (normalized to sum to len(rewards))
        """
        rewards_array = np.array(rewards)

        if mode == "none" or len(rewards_array) == 0:
            # No weighting, uniform weights
            return np.ones(len(rewards_array))

        if mode == "linear":
            # Linear weighting: weight = 1 + scale * reward
            weights = 1.0 + scale * rewards_array
            # Ensure non-negative
            weights = np.maximum(weights, 0.01)  # Minimum weight

        elif mode == "exponential":
            # Exponential weighting: weight = exp(scale * reward)
            weights = np.exp(scale * rewards_array)

        else:
            raise ValueError(f"Unknown weighting mode: {mode}")

        # Normalize weights to sum to len(rewards) (maintains original scale)
        weights = weights / np.sum(weights) * len(rewards_array)

        return weights
