"""Accuracy analysis and telemetry for trading predictions.

This module provides comprehensive tools for tracking, analyzing, and visualizing
trading prediction accuracy. It supports real-time telemetry logging with buffering,
compression, and file rotation, plus detailed accuracy metrics computation and
publication-ready visualizations.

Key Components:
    - PredictionTrace: Single prediction record with features and metadata
    - TelemetryWriter: High-performance trace logger with buffering and rotation
    - AccuracyAnalyzer: Metrics computation and visualization engine
    - AccuracyMetrics: Comprehensive accuracy metrics dataclass

Usage Example:
    ```python
    from pathlib import Path
    from datetime import datetime
    from src.services.accuracy_analyzer import (
        PredictionTrace,
        TelemetryWriter,
        AccuracyAnalyzer,
    )

    # 1. Log predictions during trading
    with TelemetryWriter(Path("./telemetry"), format="csv", compression=True) as writer:
        trace = PredictionTrace(
            timestamp=datetime.now(),
            symbol="RELIANCE",
            strategy="intraday",
            predicted_direction="LONG",
            actual_direction="LONG",
            predicted_confidence=0.85,
            entry_price=2500.0,
            exit_price=2520.0,
            holding_period_minutes=45,
            realized_return_pct=0.8,
            features={"rsi": 65.0, "macd": 12.5},
            metadata={"reason": "momentum_signal"},
        )
        writer.write_trace(trace)

    # 2. Analyze accuracy
    analyzer = AccuracyAnalyzer()
    traces = analyzer.load_traces(Path("./telemetry/predictions_*.csv.gz"))
    metrics = analyzer.compute_metrics(traces)

    # 3. Generate reports and visualizations
    analyzer.export_report(metrics, Path("./reports/accuracy.json"))
    analyzer.plot_confusion_matrix(metrics, Path("./reports/confusion_matrix.png"))
    analyzer.plot_return_distribution(traces, Path("./reports/returns.png"))
    ```

Features:
    - CSV and JSONL output formats with optional gzip compression
    - Buffered writes with configurable buffer size
    - Automatic file rotation based on size threshold
    - Atomic writes to prevent data corruption
    - Classification metrics: precision, recall, F1, confusion matrix
    - Financial metrics: Sharpe ratio, max drawdown, profit factor
    - Publication-ready matplotlib/seaborn plots
    - Comprehensive error handling and logging
"""

from __future__ import annotations

import csv
import gzip
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Try to import seaborn for better plots, but fallback to matplotlib
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning(
        "seaborn not available, using matplotlib for plots",
        extra={"component": "accuracy_analyzer"},
    )


@dataclass
class PredictionTrace:
    """Single prediction trace for accuracy analysis."""

    timestamp: datetime
    symbol: str
    strategy: str  # intraday/swing
    predicted_direction: str  # LONG/SHORT/NOOP
    actual_direction: str  # LONG/SHORT/NOOP
    predicted_confidence: float
    entry_price: float
    exit_price: float
    holding_period_minutes: int
    realized_return_pct: float
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "strategy": self.strategy,
            "predicted_direction": self.predicted_direction,
            "actual_direction": self.actual_direction,
            "predicted_confidence": self.predicted_confidence,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "holding_period_minutes": self.holding_period_minutes,
            "realized_return_pct": self.realized_return_pct,
            "features": self.features,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionTrace:
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data["symbol"],
            strategy=data["strategy"],
            predicted_direction=data["predicted_direction"],
            actual_direction=data["actual_direction"],
            predicted_confidence=data["predicted_confidence"],
            entry_price=data["entry_price"],
            exit_price=data["exit_price"],
            holding_period_minutes=data["holding_period_minutes"],
            realized_return_pct=data["realized_return_pct"],
            features=data.get("features", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics for prediction analysis."""

    precision: dict[str, float]  # per direction: LONG, SHORT, NOOP
    recall: dict[str, float]  # per direction
    f1_score: dict[str, float]  # per direction
    confusion_matrix: np.ndarray  # 3x3 matrix
    hit_ratio: float  # Fraction of correct predictions
    win_rate: float  # Fraction of profitable trades
    avg_return: float  # Average return per trade (%)
    sharpe_ratio: float  # Risk-adjusted return metric
    max_drawdown: float  # Maximum peak-to-trough decline (%)
    profit_factor: float  # Ratio of gross profit to gross loss
    total_trades: int
    avg_holding_minutes: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "hit_ratio": self.hit_ratio,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_holding_minutes": self.avg_holding_minutes,
        }


class TelemetryWriter:
    """Writer for prediction traces with buffering, compression, and rotation."""

    def __init__(
        self,
        output_dir: Path,
        format: Literal["csv", "jsonl"] = "csv",
        compression: bool = True,
        buffer_size: int = 100,
        max_file_size_mb: int = 50,
    ) -> None:
        """Initialize telemetry writer.

        Args:
            output_dir: Directory to write telemetry files
            format: Output format ("csv" or "jsonl")
            compression: Enable gzip compression
            buffer_size: Number of traces to buffer before writing
            max_file_size_mb: Maximum file size before rotation
        """
        self.output_dir = Path(output_dir)
        self.format = format
        self.compression = compression
        self.buffer_size = buffer_size
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize buffer and file tracking
        self.buffer: list[PredictionTrace] = []
        self.file_index = 0
        self.current_file_path: Path | None = None
        self.written_count = 0

        logger.info(
            "Initialized TelemetryWriter",
            extra={
                "component": "telemetry",
                "output_dir": str(self.output_dir),
                "format": self.format,
                "compression": self.compression,
            },
        )

    def _get_next_file_path(self) -> Path:
        """Generate next file path with rotation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = f".{self.format}"
        if self.compression:
            ext += ".gz"

        filename = f"predictions_{timestamp}_{self.file_index:04d}{ext}"
        return self.output_dir / filename

    def write_trace(self, trace: PredictionTrace) -> None:
        """Buffer and write trace.

        Args:
            trace: Prediction trace to write
        """
        self.buffer.append(trace)

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered traces to disk."""
        if not self.buffer:
            return

        try:
            # Check if rotation is needed
            self.rotate_if_needed()

            # Get or create file path
            if self.current_file_path is None:
                self.current_file_path = self._get_next_file_path()

            # Write to temp file first (atomic write)
            temp_fd, temp_path = tempfile.mkstemp(dir=self.output_dir, prefix=".tmp_predictions_")

            try:
                if self.format == "csv":
                    self._write_csv(temp_path)
                elif self.format == "jsonl":
                    self._write_jsonl(temp_path)

                # Atomic rename
                temp_path_obj = Path(temp_path)
                temp_path_obj.rename(self.current_file_path)

                self.written_count += len(self.buffer)
                logger.debug(
                    f"Flushed {len(self.buffer)} traces to {self.current_file_path}",
                    extra={
                        "component": "telemetry",
                        "traces_count": len(self.buffer),
                        "file": str(self.current_file_path),
                    },
                )

            finally:
                # Clean up temp file if it still exists
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

            # Clear buffer
            self.buffer.clear()

        except Exception as e:
            logger.error(
                f"Failed to flush traces: {e}",
                extra={"component": "telemetry", "error": str(e)},
            )
            raise

    def _write_csv(self, file_path: str) -> None:
        """Write buffer to CSV file.

        Args:
            file_path: Path to write CSV file
        """
        # Check if the target file (not temp) exists for append mode
        target_exists = self.current_file_path is not None and self.current_file_path.exists()

        open_func = gzip.open if self.compression else open
        mode = "wt"  # Always write mode for temp file

        with open_func(file_path, mode, encoding="utf-8") as f:
            # Define CSV headers
            headers = [
                "timestamp",
                "symbol",
                "strategy",
                "predicted_direction",
                "actual_direction",
                "predicted_confidence",
                "entry_price",
                "exit_price",
                "holding_period_minutes",
                "realized_return_pct",
                "features_json",
                "metadata_json",
            ]

            writer = csv.DictWriter(f, fieldnames=headers)

            # If target file exists, read existing data first
            if target_exists:
                # Read existing CSV content
                with open_func(self.current_file_path, "rt", encoding="utf-8") as existing:
                    existing_content = existing.read()
                    f.write(existing_content)
            else:
                # Write headers for new file
                writer.writeheader()

            # Write new traces
            for trace in self.buffer:
                row = {
                    "timestamp": trace.timestamp.isoformat(),
                    "symbol": trace.symbol,
                    "strategy": trace.strategy,
                    "predicted_direction": trace.predicted_direction,
                    "actual_direction": trace.actual_direction,
                    "predicted_confidence": trace.predicted_confidence,
                    "entry_price": trace.entry_price,
                    "exit_price": trace.exit_price,
                    "holding_period_minutes": trace.holding_period_minutes,
                    "realized_return_pct": trace.realized_return_pct,
                    "features_json": json.dumps(trace.features),
                    "metadata_json": json.dumps(trace.metadata),
                }
                writer.writerow(row)

    def _write_jsonl(self, file_path: str) -> None:
        """Write buffer to JSONL file.

        Args:
            file_path: Path to write JSONL file
        """
        # Check if the target file (not temp) exists for append mode
        target_exists = self.current_file_path is not None and self.current_file_path.exists()

        open_func = gzip.open if self.compression else open
        mode = "wt"  # Always write mode for temp file

        with open_func(file_path, mode, encoding="utf-8") as f:
            # If target file exists, read and write existing content first
            if target_exists:
                with open_func(self.current_file_path, "rt", encoding="utf-8") as existing:
                    existing_content = existing.read()
                    f.write(existing_content)

            # Write new traces
            for trace in self.buffer:
                json_line = json.dumps(trace.to_dict())
                f.write(json_line + "\n")

    def rotate_if_needed(self) -> None:
        """Check file size and rotate if needed."""
        if self.current_file_path is None:
            return

        if not self.current_file_path.exists():
            return

        file_size = self.current_file_path.stat().st_size

        if file_size >= self.max_file_size_bytes:
            logger.info(
                f"Rotating telemetry file (size={file_size / 1024 / 1024:.2f}MB)",
                extra={
                    "component": "telemetry",
                    "old_file": str(self.current_file_path),
                },
            )
            self.file_index += 1
            self.current_file_path = None

    def close(self) -> None:
        """Flush and close writer."""
        try:
            self.flush()
            logger.info(
                f"Closed TelemetryWriter (total written: {self.written_count})",
                extra={"component": "telemetry", "written_count": self.written_count},
            )
        except Exception as e:
            logger.error(
                f"Error closing TelemetryWriter: {e}",
                extra={"component": "telemetry", "error": str(e)},
            )

    def __enter__(self) -> TelemetryWriter:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


class AccuracyAnalyzer:
    """Analyzer for prediction accuracy and performance metrics."""

    def __init__(self) -> None:
        """Initialize accuracy analyzer."""
        logger.info(
            "Initialized AccuracyAnalyzer",
            extra={"component": "accuracy_analyzer"},
        )

    def load_traces(
        self,
        path: Path | list[Path],
        strategy: str | None = None,
    ) -> list[PredictionTrace]:
        """Load traces from CSV or JSONL file(s) with optional strategy filtering.

        Args:
            path: Path to trace file or list of paths to load from multiple files/directories
            strategy: Optional strategy filter ("intraday", "swing", "both", or None for all)

        Returns:
            List of PredictionTrace objects, optionally filtered by strategy

        Raises:
            ValueError: If file format is unsupported or file is malformed
            FileNotFoundError: If file does not exist
        """
        # Handle both single path and list of paths
        paths = [path] if isinstance(path, Path) else path
        all_traces: list[PredictionTrace] = []

        for p in paths:
            if not p.exists():
                # If it's a directory, find all CSV/JSONL files
                if p.parent.exists() and p.parent.is_dir():
                    # Use glob to find matching files
                    matched_files = list(p.parent.glob(p.name))
                    if not matched_files:
                        logger.warning(
                            f"No files found matching pattern: {p}",
                            extra={"component": "accuracy_analyzer"},
                        )
                        continue
                    for matched_file in matched_files:
                        all_traces.extend(self._load_single_file(matched_file))
                else:
                    raise FileNotFoundError(f"Trace file not found: {p}")
            elif p.is_dir():
                # Load all CSV/JSONL files from directory
                for pattern in ["*.csv", "*.csv.gz", "*.jsonl", "*.jsonl.gz"]:
                    for file_path in p.glob(pattern):
                        all_traces.extend(self._load_single_file(file_path))
            else:
                all_traces.extend(self._load_single_file(p))

        # Apply strategy filter if specified
        if strategy and strategy.lower() != "both":
            original_count = len(all_traces)
            all_traces = [t for t in all_traces if t.strategy.lower() == strategy.lower()]
            logger.info(
                f"Filtered traces by strategy={strategy}: {original_count} -> {len(all_traces)}",
                extra={"component": "accuracy_analyzer", "strategy": strategy},
            )

        logger.info(
            f"Loaded {len(all_traces)} traces total",
            extra={"component": "accuracy_analyzer", "traces_count": len(all_traces)},
        )

        return all_traces

    def _load_single_file(self, path: Path) -> list[PredictionTrace]:
        """Load traces from a single CSV or JSONL file.

        Args:
            path: Path to trace file

        Returns:
            List of PredictionTrace objects

        Raises:
            ValueError: If file format is unsupported or file is malformed
        """
        traces: list[PredictionTrace] = []

        try:
            # Detect format from extension
            if path.suffix == ".gz":
                # Compressed file - check inner extension
                inner_suffix = path.stem.split(".")[-1]
                if inner_suffix == "csv":
                    traces = self._load_csv(path, compressed=True)
                elif inner_suffix == "jsonl":
                    traces = self._load_jsonl(path, compressed=True)
                else:
                    raise ValueError(f"Unsupported compressed format: {inner_suffix}")
            elif path.suffix == ".csv":
                traces = self._load_csv(path, compressed=False)
            elif path.suffix == ".jsonl":
                traces = self._load_jsonl(path, compressed=False)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            logger.debug(
                f"Loaded {len(traces)} traces from {path}",
                extra={"component": "accuracy_analyzer", "traces_count": len(traces)},
            )

            return traces

        except Exception as e:
            logger.error(
                f"Failed to load traces from {path}: {e}",
                extra={"component": "accuracy_analyzer", "error": str(e)},
            )
            raise

    def _load_csv(self, path: Path, compressed: bool) -> list[PredictionTrace]:
        """Load traces from CSV file.

        Args:
            path: Path to CSV file
            compressed: Whether file is gzip compressed

        Returns:
            List of PredictionTrace objects
        """
        traces: list[PredictionTrace] = []
        open_func = gzip.open if compressed else open

        with open_func(path, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trace = PredictionTrace(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    symbol=row["symbol"],
                    strategy=row["strategy"],
                    predicted_direction=row["predicted_direction"],
                    actual_direction=row["actual_direction"],
                    predicted_confidence=float(row["predicted_confidence"]),
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    holding_period_minutes=int(row["holding_period_minutes"]),
                    realized_return_pct=float(row["realized_return_pct"]),
                    features=json.loads(row["features_json"]),
                    metadata=json.loads(row["metadata_json"]),
                )
                traces.append(trace)

        return traces

    def _load_jsonl(self, path: Path, compressed: bool) -> list[PredictionTrace]:
        """Load traces from JSONL file.

        Args:
            path: Path to JSONL file
            compressed: Whether file is gzip compressed

        Returns:
            List of PredictionTrace objects
        """
        traces: list[PredictionTrace] = []
        open_func = gzip.open if compressed else open

        with open_func(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                trace = PredictionTrace.from_dict(data)
                traces.append(trace)

        return traces

    def compute_metrics(self, traces: list[PredictionTrace]) -> AccuracyMetrics:
        """Compute comprehensive accuracy metrics.

        Args:
            traces: List of prediction traces

        Returns:
            AccuracyMetrics with computed metrics

        Raises:
            ValueError: If traces is empty or contains invalid data
        """
        if not traces:
            raise ValueError("Cannot compute metrics on empty trace list")

        logger.info(
            f"Computing metrics for {len(traces)} traces",
            extra={"component": "accuracy_analyzer", "traces_count": len(traces)},
        )

        # Extract predictions and actuals
        y_pred = [t.predicted_direction for t in traces]
        y_true = [t.actual_direction for t in traces]

        # Define label order for consistent metrics
        labels = ["LONG", "SHORT", "NOOP"]

        # Compute classification metrics
        try:
            precision_per_class = precision_score(
                y_true, y_pred, labels=labels, average=None, zero_division=0.0
            )
            recall_per_class = recall_score(
                y_true, y_pred, labels=labels, average=None, zero_division=0.0
            )
            f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0.0)
            conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

            precision_dict = {
                label: float(score)
                for label, score in zip(labels, precision_per_class, strict=False)
            }
            recall_dict = {
                label: float(score) for label, score in zip(labels, recall_per_class, strict=False)
            }
            f1_dict = {
                label: float(score) for label, score in zip(labels, f1_per_class, strict=False)
            }

        except Exception as e:
            logger.warning(
                f"Error computing classification metrics: {e}",
                extra={"component": "accuracy_analyzer", "error": str(e)},
            )
            # Fallback to zeros
            precision_dict = dict.fromkeys(labels, 0.0)
            recall_dict = dict.fromkeys(labels, 0.0)
            f1_dict = dict.fromkeys(labels, 0.0)
            conf_matrix = np.zeros((3, 3), dtype=int)

        # Hit ratio (accuracy)
        hit_ratio = sum(p == a for p, a in zip(y_pred, y_true, strict=False)) / len(traces)

        # Return-based metrics
        returns = [t.realized_return_pct for t in traces]
        profitable_trades = sum(1 for r in returns if r > 0)
        win_rate = profitable_trades / len(traces) if traces else 0.0
        avg_return = float(np.mean(returns))

        # Sharpe ratio
        sharpe_ratio = self._compute_sharpe_ratio(traces)

        # Max drawdown
        max_drawdown = self._compute_max_drawdown(traces)

        # Profit factor
        profit_factor = self._compute_profit_factor(returns)

        # Holding period
        avg_holding_minutes = float(np.mean([t.holding_period_minutes for t in traces]))

        metrics = AccuracyMetrics(
            precision=precision_dict,
            recall=recall_dict,
            f1_score=f1_dict,
            confusion_matrix=conf_matrix,
            hit_ratio=hit_ratio,
            win_rate=win_rate,
            avg_return=avg_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            total_trades=len(traces),
            avg_holding_minutes=avg_holding_minutes,
        )

        logger.info(
            "Metrics computed",
            extra={
                "component": "accuracy_analyzer",
                "hit_ratio": hit_ratio,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
            },
        )

        return metrics

    def compute_comparative_metrics(self, traces: list[PredictionTrace]) -> dict[str, Any]:
        """Compute comparative metrics between intraday and swing strategies.

        Args:
            traces: List of prediction traces (mixed strategies)

        Returns:
            Dictionary with per-strategy metrics and comparison analysis:
            {
                "intraday": AccuracyMetrics,
                "swing": AccuracyMetrics,
                "comparison": {
                    "precision_delta": float,
                    "recall_delta": float,
                    "f1_delta": float,
                    "sharpe_delta": float,
                    "win_rate_delta": float,
                    "avg_return_delta": float,
                    "better_strategy": str,
                    "better_strategy_reason": str,
                    "intraday_trades": int,
                    "swing_trades": int,
                    "statistical_significance": dict,
                }
            }

        Raises:
            ValueError: If traces don't contain both strategies
        """
        if not traces:
            raise ValueError("Cannot compute comparative metrics on empty trace list")

        # Split traces by strategy
        intraday_traces = [t for t in traces if t.strategy.lower() == "intraday"]
        swing_traces = [t for t in traces if t.strategy.lower() == "swing"]

        if not intraday_traces or not swing_traces:
            raise ValueError(
                f"Need both intraday and swing traces for comparison. "
                f"Found: {len(intraday_traces)} intraday, {len(swing_traces)} swing"
            )

        logger.info(
            f"Computing comparative metrics: {len(intraday_traces)} intraday, {len(swing_traces)} swing",
            extra={"component": "accuracy_analyzer"},
        )

        # Compute metrics for each strategy
        intraday_metrics = self.compute_metrics(intraday_traces)
        swing_metrics = self.compute_metrics(swing_traces)

        # Compute deltas (intraday - swing)
        precision_delta = intraday_metrics.precision.get("LONG", 0.0) - swing_metrics.precision.get(
            "LONG", 0.0
        )
        recall_delta = intraday_metrics.recall.get("LONG", 0.0) - swing_metrics.recall.get(
            "LONG", 0.0
        )
        f1_delta = intraday_metrics.f1_score.get("LONG", 0.0) - swing_metrics.f1_score.get(
            "LONG", 0.0
        )
        sharpe_delta = intraday_metrics.sharpe_ratio - swing_metrics.sharpe_ratio
        win_rate_delta = intraday_metrics.win_rate - swing_metrics.win_rate
        avg_return_delta = intraday_metrics.avg_return - swing_metrics.avg_return

        # Determine better strategy (composite score)
        intraday_score = (
            intraday_metrics.sharpe_ratio * 0.4
            + intraday_metrics.precision.get("LONG", 0.0) * 0.3
            + intraday_metrics.win_rate * 0.3
        )
        swing_score = (
            swing_metrics.sharpe_ratio * 0.4
            + swing_metrics.precision.get("LONG", 0.0) * 0.3
            + swing_metrics.win_rate * 0.3
        )

        if intraday_score > swing_score:
            better_strategy = "intraday"
            better_strategy_reason = (
                f"Intraday has higher composite score ({intraday_score:.3f} vs {swing_score:.3f})"
            )
        else:
            better_strategy = "swing"
            better_strategy_reason = (
                f"Swing has higher composite score ({swing_score:.3f} vs {intraday_score:.3f})"
            )

        # Statistical significance test (t-test on returns)
        intraday_returns = [t.realized_return_pct for t in intraday_traces]
        swing_returns = [t.realized_return_pct for t in swing_traces]

        try:
            from scipy import stats

            t_stat, p_value = stats.ttest_ind(intraday_returns, swing_returns)
            statistical_significance = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_at_5pct": p_value < 0.05,
                "interpretation": (
                    "Returns are significantly different at 5% level"
                    if p_value < 0.05
                    else "No significant difference in returns"
                ),
            }
        except ImportError:
            logger.warning(
                "scipy not available, skipping statistical significance test",
                extra={"component": "accuracy_analyzer"},
            )
            statistical_significance = {
                "error": "scipy not installed",
                "interpretation": "Statistical test unavailable",
            }

        comparison = {
            "precision_delta": precision_delta,
            "recall_delta": recall_delta,
            "f1_delta": f1_delta,
            "sharpe_delta": sharpe_delta,
            "win_rate_delta": win_rate_delta,
            "avg_return_delta": avg_return_delta,
            "better_strategy": better_strategy,
            "better_strategy_reason": better_strategy_reason,
            "intraday_trades": len(intraday_traces),
            "swing_trades": len(swing_traces),
            "statistical_significance": statistical_significance,
        }

        result = {
            "intraday": intraday_metrics,
            "swing": swing_metrics,
            "comparison": comparison,
        }

        logger.info(
            f"Comparative analysis complete: better_strategy={better_strategy}",
            extra={
                "component": "accuracy_analyzer",
                "better_strategy": better_strategy,
                "sharpe_delta": sharpe_delta,
            },
        )

        return result

    def _compute_sharpe_ratio(self, traces: list[PredictionTrace]) -> float:
        """Compute annualized Sharpe ratio.

        Args:
            traces: List of prediction traces

        Returns:
            Sharpe ratio (annualized)
        """
        if not traces:
            return 0.0

        returns = [t.realized_return_pct / 100.0 for t in traces]  # Convert to decimal

        if len(returns) < 2:
            return 0.0

        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns, ddof=1))

        if std_return == 0:
            return 0.0

        # Determine annualization factor based on strategy
        # Intraday: ~390 trading minutes/day * 252 days = 98,280 minutes/year
        # Swing: ~252 trading days/year
        strategy_counts = {"intraday": 0, "swing": 0}
        for trace in traces:
            if trace.strategy in strategy_counts:
                strategy_counts[trace.strategy] += 1

        # Use dominant strategy for annualization
        if strategy_counts["intraday"] > strategy_counts["swing"]:
            # Intraday: annualize based on minutes
            avg_holding = np.mean([t.holding_period_minutes for t in traces])
            if avg_holding > 0:
                trades_per_year = (252 * 390) / avg_holding
            else:
                trades_per_year = 252  # Fallback
        else:
            # Swing: annualize based on daily returns
            trades_per_year = 252

        sharpe = mean_return / std_return * np.sqrt(trades_per_year)

        return float(sharpe)

    def _compute_max_drawdown(self, traces: list[PredictionTrace]) -> float:
        """Compute maximum drawdown from cumulative returns.

        Args:
            traces: List of prediction traces

        Returns:
            Maximum drawdown as percentage
        """
        if not traces:
            return 0.0

        # Sort traces by timestamp
        sorted_traces = sorted(traces, key=lambda t: t.timestamp)

        # Compute cumulative returns
        cumulative = 0.0
        cumulative_series = []

        for trace in sorted_traces:
            cumulative += trace.realized_return_pct
            cumulative_series.append(cumulative)

        if not cumulative_series:
            return 0.0

        # Convert to numpy array
        cumulative_array = np.array(cumulative_series)

        # Compute running maximum
        running_max = np.maximum.accumulate(cumulative_array)

        # Compute drawdowns
        drawdowns = running_max - cumulative_array

        # Maximum drawdown
        max_dd = float(np.max(drawdowns))

        return max_dd

    def _compute_profit_factor(self, returns: list[float]) -> float:
        """Compute profit factor.

        Args:
            returns: List of return percentages

        Returns:
            Profit factor (ratio of gross profit to gross loss)
        """
        if not returns:
            return 0.0

        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def export_report(self, metrics: AccuracyMetrics, output_path: Path) -> None:
        """Export metrics report to JSON.

        Args:
            metrics: Computed accuracy metrics
            output_path: Path to save JSON report
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            report = {
                "generated_at": datetime.now().isoformat(),
                "metrics": metrics.to_dict(),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            logger.info(
                f"Exported metrics report to {output_path}",
                extra={"component": "accuracy_analyzer", "output_path": str(output_path)},
            )

        except Exception as e:
            logger.error(
                f"Failed to export report: {e}",
                extra={"component": "accuracy_analyzer", "error": str(e)},
            )
            raise

    def plot_confusion_matrix(self, metrics: AccuracyMetrics, output_path: Path) -> None:
        """Plot and save confusion matrix heatmap.

        Args:
            metrics: Computed accuracy metrics
            output_path: Path to save plot image
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot heatmap
            labels = ["LONG", "SHORT", "NOOP"]

            if HAS_SEABORN:
                # Use seaborn for better visuals
                sns.heatmap(
                    metrics.confusion_matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels,
                    cbar_kws={"label": "Count"},
                    ax=ax,
                )
            else:
                # Fallback to matplotlib imshow
                im = ax.imshow(metrics.confusion_matrix, cmap="Blues", aspect="auto")

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Count", rotation=270, labelpad=20)

                # Set ticks and labels
                ax.set_xticks(np.arange(len(labels)))
                ax.set_yticks(np.arange(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)

                # Add text annotations
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        text = ax.text(
                            j,
                            i,
                            int(metrics.confusion_matrix[i, j]),
                            ha="center",
                            va="center",
                            color="white"
                            if metrics.confusion_matrix[i, j] > metrics.confusion_matrix.max() / 2
                            else "black",
                            fontsize=14,
                            fontweight="bold",
                        )

            plt.title("Prediction Confusion Matrix", fontsize=16, fontweight="bold")
            plt.xlabel("Predicted Direction", fontsize=12)
            plt.ylabel("Actual Direction", fontsize=12)
            plt.tight_layout()

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(
                f"Saved confusion matrix plot to {output_path}",
                extra={"component": "accuracy_analyzer", "output_path": str(output_path)},
            )

        except Exception as e:
            logger.error(
                f"Failed to plot confusion matrix: {e}",
                extra={"component": "accuracy_analyzer", "error": str(e)},
            )
            raise

    def plot_return_distribution(self, traces: list[PredictionTrace], output_path: Path) -> None:
        """Plot and save return distribution histogram.

        Args:
            traces: List of prediction traces
            output_path: Path to save plot image
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            returns = [t.realized_return_pct for t in traces]

            # Create figure
            plt.figure(figsize=(12, 6))

            # Plot histogram
            plt.hist(
                returns,
                bins=50,
                edgecolor="black",
                alpha=0.7,
                color="steelblue",
            )

            # Add vertical line at zero
            plt.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Break-even")

            # Add mean line
            mean_return = np.mean(returns)
            plt.axvline(
                x=mean_return,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_return:.2f}%",
            )

            plt.title("Return Distribution", fontsize=16, fontweight="bold")
            plt.xlabel("Return (%)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(
                f"Saved return distribution plot to {output_path}",
                extra={"component": "accuracy_analyzer", "output_path": str(output_path)},
            )

        except Exception as e:
            logger.error(
                f"Failed to plot return distribution: {e}",
                extra={"component": "accuracy_analyzer", "error": str(e)},
            )
            raise

    def plot_feature_importance(
        self, feature_importances: dict[str, float], output_path: Path
    ) -> None:
        """Plot and save feature importance bar chart.

        Args:
            feature_importances: Dictionary mapping feature names to importance scores
            output_path: Path to save plot image
        """
        try:
            if not feature_importances:
                logger.warning(
                    "No feature importances provided, skipping plot",
                    extra={"component": "accuracy_analyzer"},
                )
                return

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Sort features by importance
            sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

            # Take top 20 features if more than 20
            if len(sorted_features) > 20:
                sorted_features = sorted_features[:20]

            features, importances = zip(*sorted_features, strict=False)

            # Create figure
            plt.figure(figsize=(12, 8))

            # Plot horizontal bar chart
            y_pos = np.arange(len(features))
            plt.barh(y_pos, importances, color="steelblue", edgecolor="black")

            plt.yticks(y_pos, features)
            plt.xlabel("Importance Score", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.title("Feature Importance (Top 20)", fontsize=16, fontweight="bold")
            plt.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(
                f"Saved feature importance plot to {output_path}",
                extra={"component": "accuracy_analyzer", "output_path": str(output_path)},
            )

        except Exception as e:
            logger.error(
                f"Failed to plot feature importance: {e}",
                extra={"component": "accuracy_analyzer", "error": str(e)},
            )
            raise
