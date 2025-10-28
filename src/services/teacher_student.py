"""Teacher-Student learning framework for SenseQuant.

This module implements the Teacher service that trains on historical data
to generate labels and model artifacts for downstream Student models.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

# Try to import LightGBM, fall back to sklearn GradientBoosting if not available
try:
    import lightgbm as lgb  # type: ignore[import-untyped]

    LIGHTGBM_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-untyped]

    LIGHTGBM_AVAILABLE = False
    logger.warning(
        "LightGBM not available, falling back to sklearn GradientBoostingClassifier",
        extra={"component": "teacher"},
    )

from src.adapters.breeze_client import BreezeClient
from src.domain.features import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    calculate_sma,
    calculate_vwap,
)
from src.domain.types import PredictionResult, StudentConfig, TrainingConfig, TrainingResult


class TeacherLabeler:
    """Offline Teacher service for label generation and model training.

    The Teacher trains on historical data with full feature set to generate
    high-quality labels and predictions that can be used to train a lightweight
    Student model for real-time inference.

    Workflow:
    1. Load historical OHLCV data
    2. Generate features using shared feature library
    3. Generate forward-looking labels (binary: profitable/unprofitable)
    4. Train LightGBM classifier
    5. Export artifacts (model, labels, importance, metadata)
    """

    def __init__(
        self,
        config: TrainingConfig,
        client: BreezeClient | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize Teacher with training configuration.

        Args:
            config: Training configuration (symbol, dates, labeling params)
            client: Optional BreezeClient for data loading (None for testing)
            output_dir: Output directory for artifacts (US-028 Phase 6o)
        """
        self.config = config
        self.client = client
        self.model: Any = None  # LightGBM or sklearn model
        self.output_dir = output_dir or Path("data/models")  # US-028 Phase 6o

        logger.info(
            f"Initialized TeacherLabeler for {config.symbol}",
            extra={
                "component": "teacher",
                "symbol": config.symbol,
                "date_range": [config.start_date, config.end_date],
            },
        )

    def load_historical_data(self) -> pd.DataFrame:
        """Load historical OHLCV bars for training.

        Returns:
            DataFrame with columns: ts, open, high, low, close, volume

        Raises:
            RuntimeError: If client is not provided or data loading fails
        """
        if self.client is None:
            raise RuntimeError("BreezeClient required for data loading")

        logger.info(
            f"Loading historical data for {self.config.symbol}",
            extra={
                "component": "teacher",
                "symbol": self.config.symbol,
                "start": self.config.start_date,
                "end": self.config.end_date,
            },
        )

        start_ts = pd.Timestamp(self.config.start_date, tz="Asia/Kolkata")
        end_ts = pd.Timestamp(self.config.end_date, tz="Asia/Kolkata")

        bars = self.client.historical_bars(
            symbol=self.config.symbol,
            interval="1day",
            start=start_ts,
            end=end_ts,
        )

        if not bars:
            raise RuntimeError(f"No data returned for {self.config.symbol}")

        df = pd.DataFrame(
            [
                {
                    "ts": bar.ts,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ]
        )

        logger.info(
            f"Loaded {len(df)} bars for {self.config.symbol}",
            extra={
                "component": "teacher",
                "symbol": self.config.symbol,
                "rows": len(df),
                "date_range": [str(df["ts"].min()), str(df["ts"].max())],
            },
        )

        return df

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate feature matrix using shared feature library.

        Computes all 9 technical indicators and drops rows with NaN values.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with original columns + feature columns (NaN rows dropped)

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ["ts", "open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy().sort_values("ts").reset_index(drop=True)

        logger.info(
            "Generating features",
            extra={"component": "teacher", "input_rows": len(df)},
        )

        # Calculate all indicators using feature library
        df["sma_20"] = calculate_sma(df["close"], period=20)
        df["sma_50"] = calculate_sma(df["close"], period=50)
        df["ema_12"] = calculate_ema(df["close"], period=12)
        df["ema_26"] = calculate_ema(df["close"], period=26)
        df["rsi_14"] = calculate_rsi(df["close"], period=14)
        df["atr_14"] = calculate_atr(df["high"], df["low"], df["close"], period=14)
        df["vwap"] = calculate_vwap(df["high"], df["low"], df["close"], df["volume"])

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df["close"], period=20)
        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower

        # MACD
        macd_line, macd_signal, macd_histogram = calculate_macd(df["close"])
        df["macd_line"] = macd_line
        df["macd_signal"] = macd_signal
        df["macd_histogram"] = macd_histogram

        # ADX
        df["adx_14"] = calculate_adx(df["high"], df["low"], df["close"], period=14)

        # OBV
        df["obv"] = calculate_obv(df["close"], df["volume"])

        # Drop rows with NaN (warm-up period for indicators)
        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)

        logger.info(
            "Features generated",
            extra={
                "component": "teacher",
                "initial_rows": initial_rows,
                "valid_rows": len(df),
                "dropped_rows": initial_rows - len(df),
            },
        )

        return df

    def generate_labels(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Generate forward-looking binary labels.

        Labels are based on forward N-day returns:
        - Label = 1 if forward_return > threshold (profitable)
        - Label = 0 otherwise (unprofitable or neutral)

        Args:
            df: DataFrame with features and close prices

        Returns:
            Tuple of (df_with_labels, labels_series)
            - df is trimmed to remove rows without forward lookahead
            - labels_series has same index as trimmed df
        """
        df = df.copy()

        logger.info(
            "Generating labels",
            extra={
                "component": "teacher",
                "window_days": self.config.label_window_days,
                "threshold_pct": self.config.label_threshold_pct,
            },
        )

        # Calculate forward N-day return
        df["forward_close"] = df["close"].shift(-self.config.label_window_days)
        df["forward_return"] = (df["forward_close"] - df["close"]) / df["close"]

        # Generate binary label
        df["label"] = (df["forward_return"] > self.config.label_threshold_pct).astype(int)

        # Drop rows without forward lookahead (last N rows)
        initial_rows = len(df)
        df = df.dropna(subset=["forward_close"]).reset_index(drop=True)

        # Get label distribution
        label_counts = df["label"].value_counts().to_dict()

        logger.info(
            "Labels generated",
            extra={
                "component": "teacher",
                "initial_rows": initial_rows,
                "labeled_rows": len(df),
                "dropped_rows": initial_rows - len(df),
                "label_distribution": label_counts,
                "class_balance": label_counts.get(1, 0) / len(df) if len(df) > 0 else 0,
            },
        )

        return df, df["label"]

    def train(self, df: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
        """Train LightGBM model and return training results.

        Args:
            df: Feature DataFrame
            labels: Binary labels (0 or 1)

        Returns:
            Dictionary with trained model, metrics, and feature importance

        Raises:
            ValueError: If insufficient samples for training (US-028 Phase 6h)
        """
        # Define feature columns (exclude metadata and target)
        exclude_cols = [
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "forward_close",
            "forward_return",
            "label",
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]  # noqa: N806
        y = labels

        # US-028 Phase 6h: Check minimum sample threshold
        min_samples = 20  # Configurable minimum for training
        if len(X) < min_samples:
            raise ValueError(
                f"Insufficient samples for training: {len(X)} < {min_samples} minimum. "
                f"Consider increasing window size or reducing forecast horizon."
            )

        # US-028 Phase 6h: Dynamic train/val split for small datasets
        # For datasets < 40 samples, use smaller train split to ensure validation has >= 2 samples
        if len(X) < 40:
            # Ensure validation gets at least 2 samples minimum
            train_split = max(0.6, 1.0 - 2.0 / len(X))
            logger.warning(
                f"Small dataset detected ({len(X)} samples). Using reduced train_split={train_split:.2f} "
                f"to ensure sufficient validation samples.",
                extra={
                    "component": "teacher",
                    "total_samples": len(X),
                    "adjusted_train_split": train_split,
                },
            )
        else:
            train_split = self.config.train_split

        logger.info(
            "Preparing train/validation split",
            extra={
                "component": "teacher",
                "total_samples": len(X),
                "features": len(feature_cols),
                "train_split": train_split,
            },
        )

        # Stratified train/val split (if possible)
        # Check if stratification is possible (need at least 2 samples per class)
        class_counts = y.value_counts()
        can_stratify = (class_counts >= 2).all()

        if can_stratify:
            X_train, X_val, y_train, y_val = train_test_split(  # noqa: N806
                X,
                y,
                train_size=train_split,
                random_state=self.config.random_seed,
                stratify=y,
            )
        else:
            logger.warning(
                "Cannot stratify split due to insufficient samples per class. Using random split.",
                extra={"component": "teacher", "class_counts": class_counts.to_dict()},
            )
            X_train, X_val, y_train, y_val = train_test_split(  # noqa: N806
                X,
                y,
                train_size=train_split,
                random_state=self.config.random_seed,
                stratify=None,
            )

        logger.info(
            f"Split complete: train={len(X_train)}, val={len(X_val)}",
            extra={
                "component": "teacher",
                "train_samples": len(X_train),
                "val_samples": len(X_val),
            },
        )

        # Configure model
        model_params = self.config.model_params or {}

        if LIGHTGBM_AVAILABLE:
            # Use LightGBM with CUDA and optimized params for maximum accuracy
            # 2x NVIDIA RTX A6000 GPUs - MANDATORY CUDA USAGE
            default_params = {
                "objective": "binary",
                "device": "cuda",  # MANDATORY GPU (2x A6000)
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
                "num_leaves": 127,  # High complexity for pattern recognition
                "max_depth": 9,  # Deep trees for non-linear patterns
                "learning_rate": 0.01,  # Slow learning for precision
                "n_estimators": 500,  # Many boosting rounds with early stopping
                "min_child_samples": 20,  # Regularization
                "subsample": 0.8,  # Row sampling / bagging
                "colsample_bytree": 0.8,  # Feature sampling
                "is_unbalance": True,  # Handle class imbalance
                "random_state": self.config.random_seed,
                "verbose": -1,
            }
            params = {**default_params, **model_params}

            logger.info(
                "Training LightGBM classifier",
                extra={"component": "teacher", "params": params},
            )

            model = lgb.LGBMClassifier(**params)
        else:
            # Use sklearn GradientBoostingClassifier as fallback
            default_params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.05,
                "random_state": self.config.random_seed,
            }
            params = {**default_params, **model_params}

            logger.info(
                "Training sklearn GradientBoostingClassifier",
                extra={"component": "teacher", "params": params},
            )

            model = GradientBoostingClassifier(**params)

        # Train model with early stopping
        if LIGHTGBM_AVAILABLE:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
        else:
            model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        metrics = {
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
            "val_precision": float(precision_score(y_val, y_val_pred, zero_division=0)),
            "val_recall": float(recall_score(y_val, y_val_pred, zero_division=0)),
            "val_f1": float(f1_score(y_val, y_val_pred, zero_division=0)),
            "val_auc": float(roc_auc_score(y_val, y_val_proba)),
        }

        logger.info(
            "Training complete",
            extra={
                "component": "teacher",
                "metrics": metrics,
            },
        )

        # Feature importance
        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        importance_df["rank"] = range(1, len(importance_df) + 1)

        self.model = model

        return {
            "model": model,
            "metrics": metrics,
            "importance": importance_df,
            "feature_cols": feature_cols,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }

    def save_artifacts(
        self,
        model: Any,  # LightGBM or sklearn model
        df_labeled: pd.DataFrame,
        importance: pd.DataFrame,
        metrics: dict[str, float],
        feature_count: int,
        train_samples: int,
        val_samples: int,
    ) -> TrainingResult:
        """Save all training artifacts to data/models/.

        Args:
            model: Trained LightGBM model
            df_labeled: DataFrame with labels and forward returns
            importance: Feature importance DataFrame
            metrics: Training metrics dict
            feature_count: Number of features used
            train_samples: Number of training samples
            val_samples: Number of validation samples

        Returns:
            TrainingResult with paths to all saved artifacts
        """
        # US-028 Phase 6o: Use output_dir with standardized filenames
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use standardized filenames (not timestamp-based) for batch training
        model_path = self.output_dir / "model.pkl"
        labels_path = self.output_dir / "labels.csv.gz"
        importance_path = self.output_dir / "feature_importance.csv"
        metadata_path = self.output_dir / "metadata.json"

        logger.info(
            "Saving artifacts",
            extra={
                "component": "teacher",
                "model_path": str(model_path),
                "labels_path": str(labels_path),
                "importance_path": str(importance_path),
                "metadata_path": str(metadata_path),
            },
        )

        # Save model (pickle)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save labels CSV with features for Student consumption
        # Include all columns except the temporary forward_close
        save_cols = [col for col in df_labeled.columns if col != "forward_close"]
        labels_df = df_labeled[save_cols].copy()
        if "symbol" not in labels_df.columns:
            labels_df["symbol"] = self.config.symbol
        # Reorder: ts, symbol, features..., label, forward_return
        col_order = (
            ["ts", "symbol"]
            + [c for c in save_cols if c not in ["ts", "symbol", "label", "forward_return"]]
            + ["label", "forward_return"]
        )
        labels_df = labels_df[[c for c in col_order if c in labels_df.columns]]
        # US-028 Phase 6o: Save as compressed CSV
        labels_df.to_csv(labels_path, index=False, compression="gzip")

        # Save feature importance CSV
        importance.to_csv(importance_path, index=False)

        # Save metadata JSON
        label_dist = df_labeled["label"].value_counts().to_dict()

        # Get feature columns from labeled DataFrame
        exclude_cols = [
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "forward_close",
            "forward_return",
            "label",
            "symbol",
        ]
        feature_cols = [col for col in df_labeled.columns if col not in exclude_cols]

        metadata = {
            "training_date": datetime.now().isoformat(),
            "symbol": self.config.symbol,
            "date_range": [self.config.start_date, self.config.end_date],
            "total_rows": len(df_labeled),
            "train_rows": train_samples,
            "val_rows": val_samples,
            "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
            "features": feature_cols,  # Add feature list for Student consumption
            "config": {
                "label_window_days": self.config.label_window_days,
                "label_threshold_pct": self.config.label_threshold_pct,
                "train_split": self.config.train_split,
                "random_seed": self.config.random_seed,
            },
            "metrics": metrics,
            "feature_count": feature_count,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Artifacts saved successfully",
            extra={
                "component": "teacher",
                "files": [
                    str(model_path),
                    str(labels_path),
                    str(importance_path),
                    str(metadata_path),
                ],
            },
        )

        return TrainingResult(
            model_path=str(model_path),
            labels_path=str(labels_path),
            importance_path=str(importance_path),
            metadata_path=str(metadata_path),
            metrics=metrics,
            feature_count=feature_count,
            train_samples=train_samples,
            val_samples=val_samples,
        )

    def run_full_pipeline(self) -> TrainingResult:
        """Run complete Teacher training pipeline.

        Steps:
        1. Load historical data
        2. Generate features
        3. Generate labels
        4. Train model
        5. Save artifacts

        Returns:
            TrainingResult with paths and metrics
        """
        logger.info(
            "Starting Teacher training pipeline",
            extra={"component": "teacher", "symbol": self.config.symbol},
        )

        # Step 1: Load data
        df = self.load_historical_data()

        # Step 2: Generate features
        df_features = self.generate_features(df)

        # Step 3: Generate labels
        df_labeled, labels = self.generate_labels(df_features)

        # Step 4: Train model
        training_output = self.train(df_labeled, labels)

        # Step 5: Save artifacts
        result = self.save_artifacts(
            model=training_output["model"],
            df_labeled=df_labeled,
            importance=training_output["importance"],
            metrics=training_output["metrics"],
            feature_count=len(training_output["feature_cols"]),
            train_samples=training_output["train_samples"],
            val_samples=training_output["val_samples"],
        )

        logger.info(
            "Teacher training pipeline complete",
            extra={
                "component": "teacher",
                "symbol": self.config.symbol,
                "val_accuracy": result.metrics["val_accuracy"],
                "val_f1": result.metrics["val_f1"],
            },
        )

        return result

    @staticmethod
    def log_batch_metadata(
        metadata_file: Path,
        batch_id: str,
        symbol: str,
        date_range: dict[str, str],
        artifacts_path: str,
        metrics: dict[str, float] | None,
        status: str,
        error: str | None = None,
        sentiment_snapshot_path: str | None = None,
    ) -> None:
        """Log batch training metadata to JSON Lines file (US-024 Phases 1-3).

        Args:
            metadata_file: Path to JSON Lines metadata file
            batch_id: Unique batch identifier
            symbol: Stock symbol
            date_range: Dict with 'start' and 'end' date strings
            artifacts_path: Path to training artifacts
            metrics: Training metrics dict (precision, recall, f1, etc.)
            status: Training status ('success' or 'failed')
            error: Error message if status is 'failed'
            sentiment_snapshot_path: Path to sentiment snapshot directory (Phase 3)
        """
        import json
        from datetime import datetime

        metadata = {
            "batch_id": batch_id,
            "symbol": symbol,
            "date_range": date_range,
            "artifacts_path": artifacts_path,
            "metrics": metrics,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }

        if error:
            metadata["error"] = error

        # US-024 Phase 3: Add sentiment snapshot reference
        if sentiment_snapshot_path is not None:
            metadata["sentiment_snapshot_path"] = sentiment_snapshot_path
            metadata["sentiment_available"] = True
        else:
            metadata["sentiment_available"] = False

        # Append to JSON Lines file
        with open(metadata_file, "a") as f:
            f.write(json.dumps(metadata) + "\n")

        logger.debug(
            f"Logged batch metadata for {symbol}",
            extra={
                "component": "teacher",
                "batch_id": batch_id,
                "symbol": symbol,
                "status": status,
            },
        )

    @staticmethod
    def load_batch_metadata(metadata_file: Path) -> list[dict]:
        """Load batch metadata from JSON Lines file (US-024).

        Args:
            metadata_file: Path to JSON Lines metadata file

        Returns:
            List of metadata dicts
        """
        import json

        if not metadata_file.exists():
            return []

        metadata_list = []
        with open(metadata_file) as f:
            for line in f:
                if line.strip():
                    metadata_list.append(json.loads(line))

        return metadata_list


class StudentModel:
    """Lightweight Student model for real-time inference.

    Student model learns from Teacher's labels using logistic regression
    for fast inference. Supports loading Teacher artifacts, training,
    prediction, and incremental retraining.
    """

    def __init__(self, config: StudentConfig) -> None:
        """Initialize Student with configuration.

        Args:
            config: Student configuration with Teacher artifact paths
        """
        from sklearn.linear_model import (  # type: ignore[import-untyped]
            LogisticRegression,
            SGDClassifier,
        )

        self.config = config
        self.model: Any = None
        self.feature_cols: list[str] = []
        self.teacher_metadata: dict[str, Any] | None = None
        self.model_version: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize model (use SGD for incremental, LogReg for batch)
        if config.incremental:
            self._base_model = SGDClassifier(
                loss="log_loss",  # For logistic regression
                random_state=config.random_seed,
                max_iter=1000,
            )
        else:
            self._base_model = LogisticRegression(
                random_state=config.random_seed,
                max_iter=1000,
                solver="lbfgs",
            )

        logger.info(
            "Initialized StudentModel",
            extra={
                "component": "student",
                "teacher_metadata_path": config.teacher_metadata_path,
                "decision_threshold": config.decision_threshold,
            },
        )

    def load_teacher_artifacts(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load Teacher's labels and metadata.

        Returns:
            Tuple of (labels_df, labels_series)

        Raises:
            FileNotFoundError: If Teacher artifacts don't exist
            ValueError: If artifacts are malformed
        """
        logger.info(
            "Loading Teacher artifacts",
            extra={
                "component": "student",
                "metadata_path": self.config.teacher_metadata_path,
                "labels_path": self.config.teacher_labels_path,
            },
        )

        # Load metadata
        metadata_path = Path(self.config.teacher_metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Teacher metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            self.teacher_metadata = json.load(f)

        # Extract feature list from metadata
        # Teacher metadata doesn't store feature list explicitly, so we need to derive it
        # from the labels CSV columns (exclude ts, symbol, label, forward_return)
        logger.info(
            f"Loaded Teacher metadata: {self.teacher_metadata.get('symbol', 'unknown')}",
            extra={
                "component": "student",
                "teacher_date": self.teacher_metadata.get("training_date"),
                "total_rows": self.teacher_metadata.get("total_rows"),
            },
        )

        # Load labels
        labels_path = Path(self.config.teacher_labels_path)
        if not labels_path.exists():
            raise FileNotFoundError(f"Teacher labels not found: {labels_path}")

        labels_df = pd.read_csv(labels_path)

        # Validate minimal structure (only label is strictly required)
        if "label" not in labels_df.columns:
            raise ValueError("Teacher labels missing 'label' column")

        logger.info(
            f"Loaded {len(labels_df)} labels from Teacher",
            extra={
                "component": "student",
                "rows": len(labels_df),
                "label_distribution": labels_df["label"].value_counts().to_dict(),
                "columns": list(labels_df.columns),
            },
        )

        return labels_df, labels_df["label"]

    def train(self, df_features: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
        """Train Student on Teacher's labels.

        Args:
            df_features: DataFrame with features (same as Teacher used)
            labels: Binary labels from Teacher

        Returns:
            Dictionary with training metrics and feature info
        """
        # Extract feature columns (exclude metadata)
        exclude_cols = [
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "forward_close",
            "forward_return",
            "label",
            "symbol",
        ]
        self.feature_cols = [col for col in df_features.columns if col not in exclude_cols]

        X = df_features[self.feature_cols]  # noqa: N806
        y = labels

        logger.info(
            "Training Student model",
            extra={
                "component": "student",
                "samples": len(X),
                "features": len(self.feature_cols),
                "feature_list": self.feature_cols,
            },
        )

        # Train logistic regression
        if self.config.incremental and self.model is not None:
            # Partial fit for incremental learning
            logger.info("Incremental training mode", extra={"component": "student"})
            self.model.partial_fit(X, y, classes=[0, 1])
        else:
            # Full training
            self.model = self._base_model
            self.model.fit(X, y)

        # Calculate training metrics
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        metrics = {
            "train_accuracy": float(accuracy_score(y, y_pred)),
            "train_precision": float(precision_score(y, y_pred, zero_division=0)),
            "train_recall": float(recall_score(y, y_pred, zero_division=0)),
            "train_f1": float(f1_score(y, y_pred, zero_division=0)),
            "train_auc": float(roc_auc_score(y, y_proba)),
        }

        logger.info(
            "Student training complete",
            extra={"component": "student", "metrics": metrics},
        )

        return {
            "metrics": metrics,
            "feature_cols": self.feature_cols,
            "samples": len(X),
        }

    def predict(self, features: pd.DataFrame) -> list[PredictionResult]:
        """Generate predictions for feature DataFrame.

        Args:
            features: DataFrame with feature columns

        Returns:
            List of PredictionResult objects
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Validate feature compatibility
        missing_features = [col for col in self.feature_cols if col not in features.columns]
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}. "
                f"Expected features: {self.feature_cols}"
            )

        # Extract features
        X = features[self.feature_cols]  # noqa: N806

        # Generate predictions
        probabilities = self.model.predict_proba(X)[:, 1]
        decisions = (probabilities >= self.config.decision_threshold).astype(int)

        # Build results
        results = []
        for prob, dec in zip(probabilities, decisions, strict=True):
            confidence = abs(prob - self.config.decision_threshold)
            results.append(
                PredictionResult(
                    symbol="",  # Will be filled by caller
                    probability=float(prob),
                    decision=int(dec),
                    confidence=float(confidence),
                    features_used=self.feature_cols,
                    model_version=self.model_version,
                    metadata={
                        "decision_threshold": self.config.decision_threshold,
                        "feature_count": len(self.feature_cols),
                    },
                )
            )

        return results

    def predict_single(self, features: dict[str, float], symbol: str = "") -> PredictionResult:
        """Generate prediction for single observation.

        Args:
            features: Dictionary of feature name -> value
            symbol: Stock symbol (optional)

        Returns:
            PredictionResult with prediction and metadata
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Validate all features present
        missing_features = [feat for feat in self.feature_cols if feat not in features]
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}. "
                f"Expected features: {self.feature_cols}"
            )

        # Build feature vector
        X_single = [features[feat] for feat in self.feature_cols]  # noqa: N806

        # Reshape for sklearn
        import numpy as np

        X_array = np.array([X_single])  # noqa: N806

        # Predict
        probability = float(self.model.predict_proba(X_array)[0, 1])
        decision = int(probability >= self.config.decision_threshold)
        confidence = abs(probability - self.config.decision_threshold)

        return PredictionResult(
            symbol=symbol,
            probability=probability,
            decision=decision,
            confidence=confidence,
            features_used=self.feature_cols,
            model_version=self.model_version,
            metadata={
                "decision_threshold": self.config.decision_threshold,
                "feature_count": len(self.feature_cols),
            },
        )

    def save(self, model_path: str | None = None, metadata_path: str | None = None) -> None:
        """Save Student model and metadata.

        Args:
            model_path: Path to save model (auto-generated if None)
            metadata_path: Path to save metadata (auto-generated if None)
        """
        if self.model is None:
            raise RuntimeError("No model to save. Call train() first.")

        # Auto-generate paths if not provided
        if model_path is None or metadata_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            models_dir = Path("data/models")
            models_dir.mkdir(parents=True, exist_ok=True)

            if model_path is None:
                model_path = str(models_dir / f"student_model_{timestamp}.pkl")
            if metadata_path is None:
                metadata_path = str(models_dir / f"student_metadata_{timestamp}.json")

        logger.info(
            "Saving Student model",
            extra={
                "component": "student",
                "model_path": model_path,
                "metadata_path": metadata_path,
            },
        )

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Save metadata
        metadata = {
            "model_version": self.model_version,
            "model_type": "logistic_regression",
            "symbol": self.teacher_metadata.get("symbol", "unknown")
            if self.teacher_metadata
            else "unknown",
            "training_date": datetime.now().isoformat(),
            "features": self.feature_cols,  # Use 'features' for consistency with Teacher
            "feature_cols": self.feature_cols,  # Keep for backward compatibility
            "feature_count": len(self.feature_cols),
            "decision_threshold": self.config.decision_threshold,
            "metrics": {},  # Student doesn't track detailed metrics in save, but field expected
            "teacher_reference": {
                "metadata_path": self.config.teacher_metadata_path,
                "labels_path": self.config.teacher_labels_path,
            },
            "teacher_metadata": self.teacher_metadata,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Student artifacts saved",
            extra={"component": "student", "files": [model_path, metadata_path]},
        )

    def load(self, model_path: str, metadata_path: str) -> None:
        """Load existing Student model.

        Args:
            model_path: Path to saved model
            metadata_path: Path to saved metadata

        Raises:
            FileNotFoundError: If model or metadata not found
        """
        logger.info(
            "Loading Student model",
            extra={
                "component": "student",
                "model_path": model_path,
                "metadata_path": metadata_path,
            },
        )

        # Load metadata
        metadata_file = Path(metadata_path)
        if not metadata_file.exists():
            raise FileNotFoundError(f"Student metadata not found: {metadata_path}")

        with open(metadata_file) as f:
            metadata = json.load(f)

        self.model_version = metadata["model_version"]
        self.feature_cols = metadata["feature_cols"]
        self.teacher_metadata = metadata.get("teacher_metadata", {})

        # Load model
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Student model not found: {model_path}")

        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

        logger.info(
            f"Student model loaded: version={self.model_version}, features={len(self.feature_cols)}",
            extra={
                "component": "student",
                "model_version": self.model_version,
                "features": len(self.feature_cols),
            },
        )

    def validate_features(self, features: pd.DataFrame) -> bool:
        """Validate feature compatibility with Teacher.

        Args:
            features: DataFrame with features

        Returns:
            True if features are compatible, False otherwise
        """
        if not self.feature_cols:
            logger.warning("No feature columns defined", extra={"component": "student"})
            return False

        missing_features = [col for col in self.feature_cols if col not in features.columns]

        if missing_features:
            logger.warning(
                f"Missing features: {missing_features}",
                extra={"component": "student", "missing": missing_features},
            )
            return False

        return True

    @staticmethod
    def log_batch_metadata(
        metadata_file: Path,
        batch_id: str,
        symbol: str,
        teacher_run_id: str,
        teacher_artifacts_path: str,
        student_artifacts_path: str,
        metrics: dict[str, float] | None,
        promotion_checklist_path: str | None,
        status: str,
        error: str | None = None,
        sentiment_snapshot_path: str | None = None,
        incremental: bool = False,
        reward_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log student batch training metadata to JSON Lines file (US-024 Phases 2-4, US-028 Phase 7 Initiative 2).

        Args:
            metadata_file: Path to JSON Lines metadata file
            batch_id: Unique batch identifier
            symbol: Stock symbol
            teacher_run_id: ID of corresponding teacher run
            teacher_artifacts_path: Path to teacher artifacts
            student_artifacts_path: Path to student artifacts
            metrics: Student model metrics dict (precision, recall, f1, etc.)
            promotion_checklist_path: Path to promotion checklist file
            status: Training status ('success' or 'failed')
            error: Error message if status is 'failed'
            sentiment_snapshot_path: Path to sentiment snapshot directory (Phase 3)
            incremental: Whether this is an incremental run (Phase 4)
            reward_metrics: Reward signal metrics (Phase 7 Initiative 2)
        """
        import json
        from datetime import datetime

        metadata = {
            "batch_id": batch_id,
            "symbol": symbol,
            "teacher_run_id": teacher_run_id,
            "teacher_artifacts_path": teacher_artifacts_path,
            "student_artifacts_path": student_artifacts_path,
            "metrics": metrics,
            "promotion_checklist_path": promotion_checklist_path,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "incremental": incremental,  # US-024 Phase 4
        }

        if error:
            metadata["error"] = error

        # US-024 Phase 3: Add sentiment snapshot reference
        if sentiment_snapshot_path is not None:
            metadata["sentiment_snapshot_path"] = sentiment_snapshot_path
            metadata["sentiment_available"] = True
        else:
            metadata["sentiment_available"] = False

        # US-028 Phase 7 Initiative 2: Add reward metrics
        if reward_metrics is not None:
            metadata["reward_metrics"] = reward_metrics
            metadata["reward_loop_enabled"] = True
        else:
            metadata["reward_loop_enabled"] = False

        # Append to JSON Lines file
        with open(metadata_file, "a") as f:
            f.write(json.dumps(metadata) + "\n")

        logger.debug(
            f"Logged student batch metadata for {symbol}",
            extra={
                "component": "student",
                "batch_id": batch_id,
                "symbol": symbol,
                "status": status,
            },
        )

    @staticmethod
    def load_batch_metadata(metadata_file: Path) -> list[dict]:
        """Load student batch metadata from JSON Lines file (US-024 Phase 2).

        Args:
            metadata_file: Path to JSON Lines metadata file

        Returns:
            List of metadata dicts
        """
        import json

        if not metadata_file.exists():
            return []

        metadata_list = []
        with open(metadata_file) as f:
            for line in f:
                if line.strip():
                    metadata_list.append(json.loads(line))

        return metadata_list

    @staticmethod
    def summarize_batch_results(metadata_list: list[dict]) -> dict[str, Any]:
        """Summarize student batch results (US-024 Phase 2).

        Args:
            metadata_list: List of student metadata dicts

        Returns:
            Summary dict with success rate and avg metrics
        """
        if not metadata_list:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "avg_f1": 0.0,
            }

        successful = [m for m in metadata_list if m.get("status") == "success"]

        # Calculate average metrics
        precisions = [m["metrics"]["precision"] for m in successful if m.get("metrics")]
        recalls = [m["metrics"]["recall"] for m in successful if m.get("metrics")]
        f1s = [m["metrics"]["f1"] for m in successful if m.get("metrics") and "f1" in m["metrics"]]

        return {
            "total": len(metadata_list),
            "successful": len(successful),
            "failed": len(metadata_list) - len(successful),
            "success_rate": len(successful) / len(metadata_list) if metadata_list else 0.0,
            "avg_precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "avg_recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "avg_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        }


class StudentModelPromoter:
    """Helper for safe student model promotion to production (US-021 Phase 2).

    Validates candidate model, checks promotion criteria, and performs atomic
    promotion with backup and rollback capabilities.

    Usage:
        from src.app.config import settings
        promoter = StudentModelPromoter(settings)
        result = promoter.validate_promotion(
            model_path="data/models/20250112_143000/student/student_model.pkl"
        )
        if result["can_promote"]:
            promoter.promote_model(model_path)
    """

    def __init__(self, settings: Any) -> None:
        """Initialize promoter with settings.

        Args:
            settings: Application settings with promotion thresholds
        """
        self.settings = settings
        logger.info("StudentModelPromoter initialized", extra={"component": "promoter"})

    def validate_promotion(self, model_path: str) -> dict[str, Any]:
        """Validate that model can be safely promoted (US-021 Phase 2).

        Args:
            model_path: Path to candidate student model

        Returns:
            Dictionary with validation results:
            {
                "can_promote": bool,
                "errors": list[str],
                "warnings": list[str],
                "checklist_found": bool,
                "criteria_passed": bool,
                "recommendation": "PROMOTE" | "REJECT"
            }
        """
        errors = []
        warnings = []
        can_promote = True
        checklist_found = False
        criteria_passed = False
        recommendation = "REJECT"

        logger.info(
            f"Validating promotion for model: {model_path}", extra={"component": "promoter"}
        )

        # 1. Check model file exists
        model_file = Path(model_path)
        if not model_file.exists():
            errors.append(f"Model file not found: {model_path}")
            can_promote = False
        else:
            # Try to load model to verify it's valid
            try:
                with open(model_file, "rb") as f:
                    _model = pickle.load(f)
                logger.info("Model loaded successfully", extra={"component": "promoter"})
            except Exception as e:
                errors.append(f"Failed to load model: {e}")
                can_promote = False

        # 2. Check metadata file exists
        model_dir = model_file.parent
        metadata_file = model_dir / "metadata.json"
        if not metadata_file.exists():
            errors.append(f"Metadata file not found: {metadata_file}")
            can_promote = False
        else:
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)

                # Verify metadata structure
                if "timestamp" not in metadata:
                    warnings.append("Metadata missing timestamp")
                if "model_type" not in metadata:
                    warnings.append("Metadata missing model_type")

                logger.info(
                    f"Metadata loaded: {metadata.get('timestamp', 'unknown')}",
                    extra={"component": "promoter"},
                )
            except Exception as e:
                errors.append(f"Failed to load metadata: {e}")
                can_promote = False

        # 3. Check promotion checklist exists
        checklist_file = model_dir / "promotion_checklist.json"
        if not checklist_file.exists():
            warnings.append(
                f"Promotion checklist not found: {checklist_file} (run with --validate flag)"
            )
        else:
            checklist_found = True
            try:
                with open(checklist_file) as f:
                    checklist = json.load(f)

                # Verify checklist structure
                if "validation_results" not in checklist:
                    errors.append("Checklist missing validation_results")
                    can_promote = False
                elif "recommendation" not in checklist:
                    errors.append("Checklist missing recommendation")
                    can_promote = False
                else:
                    recommendation = checklist["recommendation"]
                    validation_results = checklist["validation_results"]

                    # Check if all criteria passed
                    criteria_passed = validation_results.get("all_criteria_pass", False)

                    if recommendation != "PROMOTE":
                        errors.append(f"Checklist recommendation: {recommendation}")
                        can_promote = False
                    elif not criteria_passed:
                        errors.append("Validation criteria not met")
                        can_promote = False

                    logger.info(
                        f"Checklist: {recommendation}, criteria_passed={criteria_passed}",
                        extra={"component": "promoter"},
                    )
            except Exception as e:
                errors.append(f"Failed to load promotion checklist: {e}")
                can_promote = False

        # 4. Check production directory exists
        production_path = Path(self.settings.student_model_path)
        production_dir = production_path.parent
        if not production_dir.exists():
            warnings.append(f"Production directory will be created: {production_dir}")

        result = {
            "can_promote": can_promote,
            "errors": errors,
            "warnings": warnings,
            "checklist_found": checklist_found,
            "criteria_passed": criteria_passed,
            "recommendation": recommendation,
        }

        if can_promote:
            logger.info("✓ Promotion validation passed", extra={"component": "promoter"})
        else:
            logger.warning(
                f"✗ Promotion validation failed: {errors}",
                extra={"component": "promoter", "errors": errors},
            )

        return result

    def promote_model(self, model_path: str, dry_run: bool = False) -> dict[str, Any]:
        """Promote model to production with atomic copy and backup (US-021 Phase 2).

        Args:
            model_path: Path to candidate student model
            dry_run: If True, only simulate promotion without actual file operations

        Returns:
            Dictionary with promotion results:
            {
                "success": bool,
                "message": str,
                "backup_path": str | None,
                "production_path": str
            }
        """
        import shutil

        model_file = Path(model_path)
        production_path = Path(self.settings.student_model_path)
        production_dir = production_path.parent

        result = {
            "success": False,
            "message": "",
            "backup_path": None,
            "production_path": str(production_path),
        }

        logger.info(
            f"Promoting model: {model_path} → {production_path} (dry_run={dry_run})",
            extra={"component": "promoter"},
        )

        # 1. Create production directory if needed
        if not dry_run:
            production_dir.mkdir(parents=True, exist_ok=True)

        # 2. Backup existing production model (if exists)
        backup_path = None
        if production_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = production_dir.parent / "archive"
            if not dry_run:
                archive_dir.mkdir(parents=True, exist_ok=True)
            backup_path = archive_dir / f"student_baseline_{timestamp}.pkl"

            if not dry_run:
                shutil.copy2(production_path, backup_path)
                logger.info(
                    f"✓ Backed up existing model: {backup_path}", extra={"component": "promoter"}
                )
            else:
                logger.info(
                    f"[DRY RUN] Would backup: {backup_path}", extra={"component": "promoter"}
                )

            result["backup_path"] = str(backup_path)

        # 3. Copy candidate model to production (atomic)
        if not dry_run:
            try:
                shutil.copy2(model_file, production_path)
                logger.info(
                    f"✓ Model promoted to production: {production_path}",
                    extra={"component": "promoter"},
                )
                result["success"] = True
                result["message"] = f"Model successfully promoted to {production_path}"
            except Exception as e:
                logger.error(
                    f"✗ Promotion failed: {e}",
                    extra={"component": "promoter"},
                    exc_info=True,
                )
                result["success"] = False
                result["message"] = f"Promotion failed: {e}"

                # Restore backup if copy failed
                if backup_path and Path(backup_path).exists():
                    try:
                        shutil.copy2(backup_path, production_path)
                        logger.info(
                            "✓ Restored backup after failed promotion",
                            extra={"component": "promoter"},
                        )
                    except Exception as restore_error:
                        logger.error(
                            f"✗ Failed to restore backup: {restore_error}",
                            extra={"component": "promoter"},
                        )
        else:
            logger.info(
                f"[DRY RUN] Would copy: {model_file} → {production_path}",
                extra={"component": "promoter"},
            )
            result["success"] = True
            result["message"] = f"[DRY RUN] Would promote model to {production_path}"

        return result

    def rollback_model(self, archive_date: str) -> dict[str, Any]:
        """Rollback to archived baseline model (US-021 Phase 2).

        Args:
            archive_date: Date tag of archived baseline (format: YYYYMMDD_HHMMSS)

        Returns:
            Dictionary with rollback results:
            {
                "success": bool,
                "message": str,
                "restored_from": str | None
            }
        """
        import shutil

        production_path = Path(self.settings.student_model_path)
        production_dir = production_path.parent
        archive_dir = production_dir.parent / "archive"

        result = {"success": False, "message": "", "restored_from": None}

        logger.info(f"Rolling back to archive: {archive_date}", extra={"component": "promoter"})

        # Find archived model
        archive_file = archive_dir / f"student_baseline_{archive_date}.pkl"

        if not archive_file.exists():
            result["message"] = f"Archive not found: {archive_file}"
            logger.error(result["message"], extra={"component": "promoter"})
            return result

        # Copy archived model to production
        try:
            shutil.copy2(archive_file, production_path)
            result["success"] = True
            result["message"] = f"Rolled back to {archive_file}"
            result["restored_from"] = str(archive_file)
            logger.info(f"✓ Rollback successful: {archive_file}", extra={"component": "promoter"})
        except Exception as e:
            result["message"] = f"Rollback failed: {e}"
            logger.error(result["message"], extra={"component": "promoter"}, exc_info=True)

        return result

    # US-021 Phase 3: Automated Rollback

    def should_rollback(
        self, monitoring_alerts: list[dict[str, Any]], confirmation_hours: int | None = None
    ) -> dict[str, Any]:
        """Determine if automatic rollback should be triggered (US-021 Phase 3).

        Args:
            monitoring_alerts: List of recent monitoring alerts
            confirmation_hours: Hours to wait for confirmation (uses setting if None)

        Returns:
            Dictionary with rollback decision:
            {
                "should_rollback": bool,
                "reason": str,
                "trigger_alerts": list[dict],
                "earliest_alert_time": str | None,
                "confirmation_period_met": bool
            }
        """
        result = {
            "should_rollback": False,
            "reason": "",
            "trigger_alerts": [],
            "earliest_alert_time": None,
            "confirmation_period_met": False,
        }

        if not monitoring_alerts:
            result["reason"] = "No monitoring alerts"
            return result

        # Filter for student model degradation alerts
        degradation_alerts = [
            alert
            for alert in monitoring_alerts
            if alert.get("rule")
            in [
                "student_model_precision_degradation",
                "student_model_hit_ratio_degradation",
            ]
            and not alert.get("acknowledged", False)
        ]

        if not degradation_alerts:
            result["reason"] = "No student model degradation alerts"
            return result

        result["trigger_alerts"] = degradation_alerts

        # Get earliest alert time
        timestamps = [alert["timestamp"] for alert in degradation_alerts if "timestamp" in alert]
        if timestamps:
            earliest = min(timestamps)
            result["earliest_alert_time"] = earliest

            # Check confirmation period
            from datetime import datetime

            earliest_dt = datetime.fromisoformat(earliest)
            hours_since_alert = (datetime.now() - earliest_dt).total_seconds() / 3600

            confirmation_hours = (
                confirmation_hours or self.settings.student_auto_rollback_confirmation_hours
            )

            if hours_since_alert >= confirmation_hours:
                result["confirmation_period_met"] = True
                result["should_rollback"] = True
                result["reason"] = (
                    f"Degradation confirmed: {len(degradation_alerts)} alert(s) "
                    f"over {hours_since_alert:.1f}h (threshold: {confirmation_hours}h)"
                )
            else:
                result["reason"] = (
                    f"Degradation detected but confirmation period not met: "
                    f"{hours_since_alert:.1f}h < {confirmation_hours}h"
                )
        else:
            result["reason"] = "No valid timestamps in alerts"

        logger.info(
            f"Rollback decision: {result['should_rollback']} - {result['reason']}",
            extra={"component": "promoter", "degradation_alerts": len(degradation_alerts)},
        )

        return result

    def execute_auto_rollback(
        self, monitoring_service: Any, reason: str | None = None
    ) -> dict[str, Any]:
        """Execute automatic rollback after degradation confirmation (US-021 Phase 3).

        Args:
            monitoring_service: MonitoringService instance for alert tracking
            reason: Optional reason for rollback

        Returns:
            Dictionary with rollback execution result
        """
        result = {"success": False, "message": "", "rollback_result": None}

        logger.warning(
            f"Executing automatic rollback: {reason or 'Performance degradation detected'}",
            extra={"component": "promoter"},
        )

        # Get recent alerts from monitoring service
        alerts = (
            [alert.to_dict() for alert in monitoring_service.student_alerts[-20:]]
            if hasattr(monitoring_service, "student_alerts")
            else []
        )

        # Check if rollback should proceed
        rollback_decision = self.should_rollback(alerts)

        if not rollback_decision["should_rollback"]:
            result["message"] = f"Rollback not triggered: {rollback_decision['reason']}"
            logger.info(result["message"], extra={"component": "promoter"})
            return result

        # Find latest archived baseline
        production_path = Path(self.settings.student_model_path)
        production_dir = production_path.parent
        archive_dir = production_dir.parent / "archive"

        if not archive_dir.exists():
            result["message"] = "No archive directory found"
            logger.error(result["message"], extra={"component": "promoter"})
            return result

        # Get most recent baseline archive
        archive_files = sorted(archive_dir.glob("student_baseline_*.pkl"), reverse=True)

        if not archive_files:
            result["message"] = "No archived baselines found"
            logger.error(result["message"], extra={"component": "promoter"})
            return result

        latest_archive = archive_files[0]
        archive_date = latest_archive.stem.replace("student_baseline_", "")

        logger.info(
            f"Rolling back to most recent baseline: {latest_archive}",
            extra={"component": "promoter"},
        )

        # Execute rollback
        rollback_result = self.rollback_model(archive_date)

        if rollback_result["success"]:
            result["success"] = True
            result["message"] = (
                f"Automatic rollback executed successfully: {rollback_result['message']}"
            )
            result["rollback_result"] = rollback_result

            # Log rollback event
            rollback_log_path = Path("logs/alerts/rollback_events.jsonl")
            rollback_log_path.parent.mkdir(parents=True, exist_ok=True)

            rollback_event = {
                "timestamp": datetime.now().isoformat(),
                "reason": reason or rollback_decision["reason"],
                "archive_restored": str(latest_archive),
                "trigger_alerts": rollback_decision["trigger_alerts"],
                "confirmation_hours": self.settings.student_auto_rollback_confirmation_hours,
            }

            try:
                with open(rollback_log_path, "a") as f:
                    json.dump(rollback_event, f)
                    f.write("\n")
            except Exception as e:
                logger.error(
                    f"Failed to log rollback event: {e}",
                    extra={"component": "promoter"},
                )

            logger.warning(
                f"✓ Automatic rollback completed: {latest_archive}",
                extra={"component": "promoter", "reason": reason},
            )
        else:
            result["message"] = f"Automatic rollback failed: {rollback_result['message']}"
            result["rollback_result"] = rollback_result
            logger.error(result["message"], extra={"component": "promoter"})

        return result
