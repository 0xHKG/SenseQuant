"""Domain types for SenseQuant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# Type aliases
OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]
OrderStatus = Literal["NEW", "PLACED", "FILLED", "REJECTED", "CANCELLED"]
SignalDirection = Literal["LONG", "SHORT", "FLAT"]


@dataclass
class Bar:
    """OHLCV bar."""

    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Signal:
    """Trading signal."""

    symbol: str
    direction: SignalDirection
    strength: float
    meta: dict[str, Any] | None = None


@dataclass
class Order:
    """Order request."""

    symbol: str
    side: OrderSide
    qty: int
    order_type: OrderType = "MARKET"
    price: float | None = None


@dataclass
class OrderResponse:
    """Order response from broker."""

    order_id: str
    status: OrderStatus
    raw: dict[str, Any] | None = None


@dataclass
class Position:
    """Open position."""

    symbol: str
    qty: int
    avg_price: float
    entry_fees: float = 0.0
    exit_fees: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for Teacher model training."""

    symbol: str
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    label_window_days: int = 5
    label_threshold_pct: float = 0.02
    train_split: float = 0.8
    random_seed: int = 42
    model_params: dict[str, Any] | None = None


@dataclass
class TrainingResult:
    """Results from Teacher model training."""

    model_path: str
    labels_path: str
    importance_path: str
    metadata_path: str
    metrics: dict[str, float]
    feature_count: int
    train_samples: int
    val_samples: int


@dataclass
class StudentConfig:
    """Configuration for Student model training."""

    teacher_metadata_path: str
    teacher_labels_path: str
    decision_threshold: float = 0.5
    random_seed: int = 42
    incremental: bool = False  # Incremental vs full retrain


@dataclass
class PredictionResult:
    """Result from Student model prediction."""

    symbol: str
    probability: float  # Probability of profitable trade [0, 1]
    decision: int  # Binary decision (0 or 1)
    confidence: float  # Distance from threshold
    features_used: list[str]
    model_version: str
    metadata: dict[str, Any]


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    symbols: list[str]
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    strategy: Literal["intraday", "swing", "both"] = "swing"
    initial_capital: float = 1000000.0
    data_source: Literal["breeze", "csv", "teacher"] = "breeze"
    random_seed: int = 42
    csv_path: str | None = None  # Path to CSV file if data_source="csv"
    teacher_labels_path: str | None = None  # Path to teacher labels if data_source="teacher"
    resolution: Literal["1day", "1minute", "5minute", "15minute"] = "1day"  # US-018: Bar resolution


@dataclass
class BacktestResult:
    """Results from backtesting run.

    Attributes:
        config: Backtest configuration
        metrics: Financial metrics (Sharpe, return, drawdown, etc.)
        equity_curve: Equity over time
        trades: Trade history
        metadata: Additional metadata
        summary_path: Path to summary file
        equity_path: Path to equity curve file
        trades_path: Path to trades file
        accuracy_metrics: Accuracy metrics from telemetry (US-019)
        telemetry_dir: Directory containing prediction traces (US-019)
    """

    config: BacktestConfig
    metrics: dict[str, float]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metadata: dict[str, Any]
    summary_path: str
    equity_path: str
    trades_path: str
    accuracy_metrics: Any | None = None  # AccuracyMetrics from accuracy_analyzer (US-019)
    telemetry_dir: Path | None = None  # Telemetry storage directory (US-019)


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""

    symbols: list[str]
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    strategy: Literal["intraday", "swing", "both"] = "swing"
    initial_capital: float = 1000000.0
    search_space: dict[str, Any] | None = None  # Parameter ranges to optimize
    search_type: Literal["grid", "random"] = "grid"
    n_samples: int = 100  # For random search
    objective_metric: str = "sharpe_ratio"  # Metric to maximize
    random_seed: int = 42
    data_source: Literal["breeze", "csv", "teacher"] = "breeze"
    csv_path: str | None = None
    teacher_labels_path: str | None = None

    def __post_init__(self) -> None:
        """Initialize search_space if None."""
        if self.search_space is None:
            self.search_space = {}


@dataclass
class OptimizationCandidate:
    """Single parameter combination evaluated during optimization."""

    candidate_id: int
    parameters: dict[str, Any]
    backtest_result: BacktestResult | None = None
    score: float | None = None  # Objective metric value
    error: str | None = None  # Error message if failed
    elapsed_time: float = 0.0  # Execution time in seconds


@dataclass
class OptimizationResult:
    """Results from parameter optimization run."""

    config: OptimizationConfig
    candidates: list[OptimizationCandidate]
    best_candidate: OptimizationCandidate | None
    total_candidates: int
    successful_candidates: int
    failed_candidates: int
    total_time: float  # Total optimization time in seconds
    metadata: dict[str, Any]
    summary_path: str
    ranked_results_path: str
    best_config_path: str
