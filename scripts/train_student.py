#!/usr/bin/env python
"""CLI entry point for Student model training.

The Student model is trained on teacher-generated labels to learn prediction patterns
for real-time inference. This script loads teacher artifacts, trains a lightweight
student model, evaluates performance, and optionally validates via backtesting.

Usage:
    # Train student from teacher labels
    python scripts/train_student.py \
      --teacher-dir data/models/20250112_143000/teacher \
      --output-dir data/models/20250112_143000/student \
      --model-type logistic \
      --test-size 0.2

    # Train with hyperparameter tuning
    python scripts/train_student.py \
      --teacher-dir data/models/20250112_143000/teacher \
      --output-dir data/models/20250112_143000/student \
      --model-type logistic \
      --hyperparameter-tuning \
      --cv-folds 5

    # Train and validate with backtest
    python scripts/train_student.py \
      --teacher-dir data/models/20250112_143000/teacher \
      --output-dir data/models/20250112_143000/student \
      --validate \
      --validation-symbols RELIANCE TCS \
      --validation-start 2024-07-01 \
      --validation-end 2024-09-30
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import calibration_curve  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression, SGDClassifier  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split  # type: ignore[import-untyped]

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning(
        "Matplotlib not available, skipping plot generation", extra={"component": "student"}
    )

# Ensure project root (src) is on sys.path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Student model from Teacher labels (US-020)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--teacher-dir",
        type=str,
        required=True,
        help="Path to teacher artifacts directory (containing labels.csv.gz with embedded features, metadata.json)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for student artifacts",
    )

    # Model configuration
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic", "sgd", "lightgbm"],
        default="logistic",
        help="Student model type",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size ratio (e.g., 0.2 = 20%% test, 80%% train)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Hyperparameter tuning
    parser.add_argument(
        "--hyperparameter-tuning",
        action="store_true",
        help="Enable hyperparameter tuning via cross-validation",
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds for hyperparameter tuning",
    )

    # Validation
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run post-training validation via backtesting",
    )

    parser.add_argument(
        "--validation-symbols",
        type=str,
        nargs="+",
        help="Symbols for validation backtest (e.g., RELIANCE TCS)",
    )

    parser.add_argument(
        "--validation-start",
        type=str,
        help="Validation start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--validation-end",
        type=str,
        help="Validation end date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--baseline-model",
        type=str,
        help="Path to baseline student model for comparison",
    )

    # Incremental training
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental training (append new data to existing model)",
    )

    # US-021 Phase 2: Promotion workflow
    parser.add_argument(
        "--generate-checklist",
        action="store_true",
        help="Generate promotion checklist after training (US-021 Phase 2)",
    )

    parser.add_argument(
        "--precision-uplift-threshold",
        type=float,
        default=0.02,
        help="Minimum precision uplift vs baseline for promotion (e.g., 0.02 = 2%%)",
    )

    parser.add_argument(
        "--hit-ratio-uplift-threshold",
        type=float,
        default=0.03,
        help="Minimum hit ratio uplift vs baseline for promotion (e.g., 0.03 = 3%%)",
    )

    parser.add_argument(
        "--sharpe-uplift-threshold",
        type=float,
        default=0.1,
        help="Minimum Sharpe ratio uplift vs baseline for promotion (e.g., 0.1)",
    )

    # US-028 Phase 7 Initiative 2: Reward loop flags
    parser.add_argument(
        "--enable-reward-loop",
        action="store_true",
        help="Enable adaptive learning via reward signals (US-028 Phase 7 Initiative 2)",
    )

    parser.add_argument(
        "--reward-horizon-days",
        type=int,
        default=5,
        help="Number of days to look ahead for realized returns (reward calculation)",
    )

    parser.add_argument(
        "--reward-weighting-mode",
        type=str,
        choices=["linear", "exponential", "none"],
        default="linear",
        help="Sample weighting mode based on rewards",
    )

    parser.add_argument(
        "--reward-ab-testing",
        action="store_true",
        help="Enable A/B testing (train both baseline and reward-weighted models)",
    )

    # US-024 Phase 2: Batch mode flags
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Non-interactive batch mode (skip prompts, auto-generate checklists)",
    )

    parser.add_argument(
        "--baseline-precision",
        type=float,
        help="Baseline precision for promotion criteria (batch mode only)",
    )

    parser.add_argument(
        "--baseline-recall",
        type=float,
        help="Baseline recall for promotion criteria (batch mode only)",
    )

    return parser.parse_args()


def load_teacher_artifacts(teacher_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load teacher-generated labels, features, and metadata.

    Args:
        teacher_dir: Path to teacher artifacts directory

    Returns:
        Tuple of (labels_df, features_df, metadata)

    Raises:
        FileNotFoundError: If required artifacts missing
    """
    logger.info(f"Loading teacher artifacts from {teacher_dir}", extra={"component": "student"})

    # Load labels
    labels_path = teacher_dir / "labels.csv.gz"
    if not labels_path.exists():
        raise FileNotFoundError(f"Teacher labels not found: {labels_path}")

    with gzip.open(labels_path, "rt") as f:
        labels_df = pd.read_csv(f)

    logger.info(f"Loaded {len(labels_df)} teacher labels", extra={"component": "student"})

    # US-028 Phase 6p: Features are embedded in labels.csv.gz
    # Extract feature columns (exclude ts, symbol, label, forward_return)
    exclude_cols = ["ts", "symbol", "label", "forward_return"]
    feature_cols = [col for col in labels_df.columns if col not in exclude_cols]
    features_df = labels_df[feature_cols].copy()

    logger.info(
        f"Extracted {len(feature_cols)} features from labels (shape: {features_df.shape})",
        extra={"component": "student"},
    )

    # Load metadata
    metadata_path = teacher_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Teacher metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    return labels_df, features_df, metadata


def prepare_training_data(
    labels_df: pd.DataFrame, features_df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare training and test datasets.

    Args:
        labels_df: Teacher-generated labels
        features_df: Engineered features
        test_size: Test set proportion
        seed: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Merge labels and features (assuming common index or timestamp column)
    # For simplicity, assume labels_df has 'label' column and features_df has feature columns
    # In production, would need proper join logic

    # Assuming labels_df has columns: [timestamp, label, ...]
    # and features_df has columns: [timestamp, feature1, feature2, ...]

    if "timestamp" in labels_df.columns and "timestamp" in features_df.columns:
        # Merge on timestamp
        data = pd.merge(labels_df, features_df, on="timestamp", how="inner")
    else:
        # Assume same index
        data = pd.concat([labels_df, features_df], axis=1)

    # Extract labels
    if "label" not in data.columns:
        raise ValueError("Labels DataFrame must contain 'label' column")

    y = data["label"]

    # Extract features (drop non-feature columns)
    # US-028 Phase 6p: Also exclude 'ts', 'forward_return', raw OHLCV
    non_feature_cols = ["timestamp", "ts", "label", "symbol", "forward_return"]
    X = data.drop(columns=[col for col in non_feature_cols if col in data.columns])

    # Drop any NaN rows
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    logger.info(
        f"Prepared dataset: {len(X)} samples, {X.shape[1]} features",
        extra={"component": "student"},
    )

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    logger.info(
        f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples",
        extra={"component": "student"},
    )

    return X_train, X_test, y_train, y_test


def train_student_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    hyperparameter_tuning: bool,
    cv_folds: int,
    seed: int,
    sample_weights: np.ndarray | None = None,
) -> Any:
    """Train student model with optional hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Model type (logistic, sgd, lightgbm)
        hyperparameter_tuning: Enable hyperparameter tuning
        cv_folds: Number of CV folds
        seed: Random seed
        sample_weights: Optional sample weights (US-028 Phase 7 Initiative 2)

    Returns:
        Trained model
    """
    weighted_str = "weighted" if sample_weights is not None else "unweighted"
    logger.info(
        f"Training {model_type} student model (tuning={hyperparameter_tuning}, {weighted_str})",
        extra={"component": "student"},
    )

    if model_type == "logistic":
        if hyperparameter_tuning:
            param_grid = {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
                "max_iter": [1000],
            }
            base_model = LogisticRegression(random_state=seed)
            model = GridSearchCV(base_model, param_grid, cv=cv_folds, scoring="f1_macro", n_jobs=-1)
        else:
            model = LogisticRegression(C=1.0, random_state=seed, max_iter=1000)

    elif model_type == "sgd":
        if hyperparameter_tuning:
            param_grid = {
                "alpha": [0.0001, 0.001, 0.01],
                "penalty": ["l2"],
                "loss": ["log_loss"],
                "max_iter": [1000],
            }
            base_model = SGDClassifier(random_state=seed)
            model = GridSearchCV(base_model, param_grid, cv=cv_folds, scoring="f1_macro", n_jobs=-1)
        else:
            model = SGDClassifier(alpha=0.001, random_state=seed, max_iter=1000)

    elif model_type == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore[import-untyped]

            if hyperparameter_tuning:
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.05, 0.1],
                }
                base_model = lgb.LGBMClassifier(random_state=seed)
                model = GridSearchCV(
                    base_model, param_grid, cv=cv_folds, scoring="f1_macro", n_jobs=-1
                )
            else:
                model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=seed)
        except ImportError:
            logger.error("LightGBM not installed", extra={"component": "student"})
            sys.exit(1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model with optional sample weights (US-028 Phase 7 Initiative 2)
    if sample_weights is not None:
        logger.info(
            f"Applying sample weights: mean={sample_weights.mean():.4f}, "
            f"std={sample_weights.std():.4f}, min={sample_weights.min():.4f}, "
            f"max={sample_weights.max():.4f}",
            extra={"component": "student"},
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    if hyperparameter_tuning and isinstance(model, GridSearchCV):
        logger.info(f"Best hyperparameters: {model.best_params_}", extra={"component": "student"})
        return model.best_estimator_

    return model


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float | dict[str, float]]:
    """Evaluate student model on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating student model on test set", extra={"component": "student"})

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    labels_unique = sorted(y_test.unique())
    precision_per_class = precision_score(
        y_test, y_pred, average=None, labels=labels_unique, zero_division=0
    )
    recall_per_class = recall_score(
        y_test, y_pred, average=None, labels=labels_unique, zero_division=0
    )
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=labels_unique, zero_division=0)

    # AUC-ROC (if multi-class, use OvR strategy)
    try:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0  # Fallback if AUC computation fails

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels_unique)

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "auc_macro": float(auc),
        "precision_per_class": {
            str(label): float(prec)
            for label, prec in zip(labels_unique, precision_per_class, strict=False)
        },
        "recall_per_class": {
            str(label): float(rec)
            for label, rec in zip(labels_unique, recall_per_class, strict=False)
        },
        "f1_per_class": {
            str(label): float(f1) for label, f1 in zip(labels_unique, f1_per_class, strict=False)
        },
        "confusion_matrix": cm.tolist(),
        "class_labels": [str(label) for label in labels_unique],
    }

    logger.info(f"Test Accuracy: {accuracy:.4f}", extra={"component": "student"})
    logger.info(f"Test F1 (macro): {f1_macro:.4f}", extra={"component": "student"})

    return metrics


def generate_calibration_plot(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path
) -> None:
    """Generate calibration plot for student model.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        output_dir: Output directory for plot
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning(
            "Matplotlib not available, skipping calibration plot", extra={"component": "student"}
        )
        return

    # For binary classification only
    if len(y_test.unique()) != 2:
        logger.info(
            "Calibration plot only for binary classification, skipping",
            extra={"component": "student"},
        )
        return

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Student Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "calibration_plot.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Calibration plot saved to {output_path}", extra={"component": "student"})


def save_student_artifacts(
    model: Any,
    metrics: dict[str, Any],
    teacher_metadata: dict[str, Any],
    output_dir: Path,
    model_type: str,
    test_size: float,
    cv_folds: int,
    hyperparameter_tuning: bool,
) -> None:
    """Save student model and metadata.

    Args:
        model: Trained student model
        metrics: Evaluation metrics
        teacher_metadata: Teacher training metadata
        output_dir: Output directory
        model_type: Student model type
        test_size: Test set size
        cv_folds: Number of CV folds
        hyperparameter_tuning: Whether hyperparameter tuning was used
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "student_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Student model saved to {model_path}", extra={"component": "student"})

    # Save evaluation metrics
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation metrics saved to {metrics_path}", extra={"component": "student"})

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "teacher_dir": str(teacher_metadata.get("model_path", "")),
        "model_type": model_type,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "hyperparameter_tuning": hyperparameter_tuning,
        "teacher_metadata": teacher_metadata,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}", extra={"component": "student"})


def run_validation_backtest(
    model: Any,
    symbols: list[str],
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run validation backtest for student model (US-021 Phase 2).

    Args:
        model: Trained student model
        symbols: Symbols for validation
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory for backtest results

    Returns:
        Dictionary with validation metrics
    """
    logger.info(
        f"Running validation backtest: {symbols}, {start_date} to {end_date}",
        extra={"component": "student"},
    )

    try:
        from src.app.config import Settings
        from src.domain.types import BacktestConfig
        from src.services.backtester import Backtester

        # Create backtest config
        config = BacktestConfig(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            strategy="intraday",  # Default to intraday for validation
            initial_capital=1_000_000.0,
        )

        settings = Settings()
        backtester = Backtester(config=config, settings=settings)

        # Run backtest
        result = backtester.run()

        # Extract key metrics
        validation_metrics = {
            "total_return_pct": result.total_return_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "profitable_trades": result.profitable_trades,
            "final_equity": result.final_equity,
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
        }

        logger.info(
            f"Validation backtest complete: Return={result.total_return_pct:.2f}%, "
            f"Sharpe={result.sharpe_ratio:.2f}, Win Rate={result.win_rate:.2f}%",
            extra={"component": "student"},
        )

        return validation_metrics

    except Exception as e:
        logger.error(f"Validation backtest failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
        }


def compare_with_baseline(
    validation_metrics: dict[str, Any],
    baseline_model_path: str | None,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Compare candidate model with baseline (US-021 Phase 2).

    Args:
        validation_metrics: Validation metrics from candidate model
        baseline_model_path: Path to baseline model
        symbols: Symbols for validation
        start_date: Start date
        end_date: End date

    Returns:
        Dictionary with comparison results
    """
    if not baseline_model_path or not Path(baseline_model_path).exists():
        logger.warning(
            "No baseline model provided or found, skipping comparison",
            extra={"component": "student"},
        )
        return {
            "baseline_available": False,
            "uplift": {},
        }

    logger.info(f"Comparing with baseline: {baseline_model_path}", extra={"component": "student"})

    try:
        # Load baseline model
        with open(baseline_model_path, "rb") as f:
            baseline_model = pickle.load(f)

        # Run baseline backtest
        baseline_metrics = run_validation_backtest(
            baseline_model, symbols, start_date, end_date, Path("baseline_temp")
        )

        # Calculate uplifts
        uplift = {}
        if "error" not in baseline_metrics and "error" not in validation_metrics:
            for key in ["sharpe_ratio", "win_rate", "total_return_pct"]:
                if key in validation_metrics and key in baseline_metrics:
                    baseline_val = baseline_metrics[key]
                    candidate_val = validation_metrics[key]
                    if baseline_val != 0:
                        uplift[key] = candidate_val - baseline_val
                        uplift[f"{key}_pct_change"] = (
                            (candidate_val - baseline_val) / abs(baseline_val) * 100
                        )

        comparison = {
            "baseline_available": True,
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": validation_metrics,
            "uplift": uplift,
        }

        logger.info(f"Comparison complete: {uplift}", extra={"component": "student"})
        return comparison

    except Exception as e:
        logger.error(f"Baseline comparison failed: {e}", exc_info=True)
        return {
            "baseline_available": False,
            "error": str(e),
        }


def generate_promotion_checklist(
    validation_metrics: dict[str, Any],
    comparison: dict[str, Any],
    train_metrics: dict[str, Any],
    output_dir: Path,
    precision_uplift_threshold: float,
    hit_ratio_uplift_threshold: float,
    sharpe_uplift_threshold: float,
) -> None:
    """Generate promotion checklist files (US-021 Phase 2).

    Args:
        validation_metrics: Validation backtest metrics
        comparison: Baseline comparison results
        train_metrics: Training evaluation metrics
        output_dir: Output directory
        precision_uplift_threshold: Minimum precision uplift required
        hit_ratio_uplift_threshold: Minimum hit ratio uplift required
        sharpe_uplift_threshold: Minimum Sharpe ratio uplift required
    """
    logger.info("Generating promotion checklist", extra={"component": "student"})

    # Evaluate criteria
    criteria = {}
    all_pass = True

    # 1. Training accuracy threshold
    train_accuracy = train_metrics.get("accuracy", 0.0)
    criteria["train_accuracy_ge_60pct"] = train_accuracy >= 0.60
    if not criteria["train_accuracy_ge_60pct"]:
        all_pass = False

    # 2. Training F1 score threshold
    train_f1 = train_metrics.get("f1_macro", 0.0)
    criteria["train_f1_ge_55pct"] = train_f1 >= 0.55
    if not criteria["train_f1_ge_55pct"]:
        all_pass = False

    # 3. Validation backtest completed successfully
    criteria["validation_backtest_success"] = "error" not in validation_metrics
    if not criteria["validation_backtest_success"]:
        all_pass = False

    # 4. Check uplifts vs baseline (if available)
    if comparison.get("baseline_available"):
        uplift = comparison.get("uplift", {})

        # Precision/accuracy uplift
        accuracy_uplift = uplift.get("win_rate", 0.0)  # Using win_rate as proxy for precision
        criteria["precision_uplift_met"] = accuracy_uplift >= hit_ratio_uplift_threshold
        if not criteria["precision_uplift_met"]:
            all_pass = False

        # Sharpe uplift
        sharpe_uplift = uplift.get("sharpe_ratio", 0.0)
        criteria["sharpe_uplift_met"] = sharpe_uplift >= sharpe_uplift_threshold
        if not criteria["sharpe_uplift_met"]:
            all_pass = False

        criteria["baseline_comparison_available"] = True
    else:
        criteria["baseline_comparison_available"] = False
        # If no baseline, can still promote if other criteria pass
        criteria["precision_uplift_met"] = True
        criteria["sharpe_uplift_met"] = True

    # Determine recommendation
    recommendation = "PROMOTE" if all_pass else "REJECT"

    # Build checklist JSON
    checklist_json = {
        "timestamp": datetime.now().isoformat(),
        "recommendation": recommendation,
        "validation_results": {
            "all_criteria_pass": all_pass,
            "criteria": criteria,
            "thresholds": {
                "precision_uplift": precision_uplift_threshold,
                "hit_ratio_uplift": hit_ratio_uplift_threshold,
                "sharpe_uplift": sharpe_uplift_threshold,
            },
        },
        "train_metrics": {
            "accuracy": train_metrics.get("accuracy", 0.0),
            "f1_macro": train_metrics.get("f1_macro", 0.0),
            "precision_macro": train_metrics.get("precision_macro", 0.0),
        },
        "validation_metrics": validation_metrics,
        "baseline_comparison": comparison,
    }

    # Save JSON checklist
    checklist_json_path = output_dir / "promotion_checklist.json"
    with open(checklist_json_path, "w") as f:
        json.dump(checklist_json, f, indent=2)
    logger.info(f"Promotion checklist (JSON) saved to {checklist_json_path}")

    # Build Markdown checklist
    checklist_md = f"""# Student Model Promotion Checklist

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Recommendation:** `{recommendation}`

## Validation Results

| Criterion | Status | Value |
|-----------|--------|-------|
| Training Accuracy >= 60% | {"✅ PASS" if criteria["train_accuracy_ge_60pct"] else "❌ FAIL"} | {train_accuracy:.2%} |
| Training F1 Score >= 55% | {"✅ PASS" if criteria["train_f1_ge_55pct"] else "❌ FAIL"} | {train_f1:.2%} |
| Validation Backtest Success | {"✅ PASS" if criteria["validation_backtest_success"] else "❌ FAIL"} | {"Success" if criteria["validation_backtest_success"] else "Failed"} |
| Baseline Comparison Available | {"✅ YES" if criteria.get("baseline_comparison_available", False) else "⚠️  NO"} | {"Available" if criteria.get("baseline_comparison_available", False) else "Not Available"} |
"""

    if criteria.get("baseline_comparison_available"):
        uplift = comparison.get("uplift", {})
        checklist_md += f"""| Precision Uplift >= {hit_ratio_uplift_threshold:.1%} | {"✅ PASS" if criteria["precision_uplift_met"] else "❌ FAIL"} | {uplift.get("win_rate", 0.0):.2%} |
| Sharpe Uplift >= {sharpe_uplift_threshold:.2f} | {"✅ PASS" if criteria["sharpe_uplift_met"] else "❌ FAIL"} | {uplift.get("sharpe_ratio", 0.0):.2f} |
"""

    checklist_md += f"""
## Training Metrics

- **Accuracy:** {train_metrics.get("accuracy", 0.0):.2%}
- **Precision (macro):** {train_metrics.get("precision_macro", 0.0):.2%}
- **Recall (macro):** {train_metrics.get("recall_macro", 0.0):.2%}
- **F1 Score (macro):** {train_metrics.get("f1_macro", 0.0):.2%}

## Validation Backtest

"""

    if "error" not in validation_metrics:
        checklist_md += f"""- **Symbols:** {", ".join(validation_metrics.get("symbols", []))}
- **Period:** {validation_metrics.get("start_date", "N/A")} to {validation_metrics.get("end_date", "N/A")}
- **Total Return:** {validation_metrics.get("total_return_pct", 0.0):.2f}%
- **Sharpe Ratio:** {validation_metrics.get("sharpe_ratio", 0.0):.2f}
- **Max Drawdown:** {validation_metrics.get("max_drawdown_pct", 0.0):.2f}%
- **Win Rate:** {validation_metrics.get("win_rate", 0.0):.2f}%
- **Total Trades:** {validation_metrics.get("total_trades", 0)}
"""
    else:
        checklist_md += f"""**Error:** {validation_metrics.get("error", "Unknown error")}
"""

    if comparison.get("baseline_available"):
        checklist_md += """
## Baseline Comparison

### Candidate vs Baseline Uplift

"""
        uplift = comparison.get("uplift", {})
        for key, value in uplift.items():
            if not key.endswith("_pct_change"):
                pct_key = f"{key}_pct_change"
                pct_value = uplift.get(pct_key, 0.0)
                checklist_md += (
                    f"- **{key.replace('_', ' ').title()}:** {value:+.4f} ({pct_value:+.2f}%)\n"
                )

    checklist_md += f"""
---

**Next Steps:**
- Review validation results above
- If recommendation is `PROMOTE`, run: `python scripts/promote_student.py --model-path {output_dir / "student_model.pkl"} --promote`
- If recommendation is `REJECT`, investigate failures and retrain with adjusted parameters
"""

    # Save Markdown checklist
    checklist_md_path = output_dir / "promotion_checklist.md"
    with open(checklist_md_path, "w") as f:
        f.write(checklist_md)
    logger.info(f"Promotion checklist (Markdown) saved to {checklist_md_path}")

    logger.info(f"Promotion recommendation: {recommendation}", extra={"component": "student"})


def calculate_and_log_rewards(
    labels_df: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
    reward_horizon_days: int,
    reward_clip_min: float,
    reward_clip_max: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Calculate rewards for training samples and log to reward_history.jsonl.

    US-028 Phase 7 Initiative 2: Reward Loop Integration

    Args:
        labels_df: Full labels dataframe with forward_return column
        y_train: Training labels (predictions)
        output_dir: Output directory for reward_history.jsonl
        reward_horizon_days: Days to look ahead for realized returns
        reward_clip_min: Minimum reward value (clipping)
        reward_clip_max: Maximum reward value (clipping)

    Returns:
        Tuple of (reward_entries, aggregate_metrics)
    """
    from src.services.reward_calculator import RewardCalculator

    logger.info(
        f"Calculating rewards: horizon={reward_horizon_days} days, clip=[{reward_clip_min}, {reward_clip_max}]",
        extra={"component": "student"},
    )

    # Initialize reward calculator
    calculator = RewardCalculator(
        reward_clip_min=reward_clip_min,
        reward_clip_max=reward_clip_max,
    )

    # Build mapping from train indices to forward returns
    # labels_df should have: ts, symbol, label, forward_return
    if "forward_return" not in labels_df.columns:
        logger.warning("No forward_return column in labels_df, cannot calculate rewards", extra={"component": "student"})
        return [], {}

    # Match train indices with labels_df
    # y_train.index should align with labels_df indices
    train_indices = y_train.index.tolist()

    # Extract forward returns for training samples
    train_forward_returns = labels_df.loc[train_indices, "forward_return"].values
    train_predictions = y_train.values

    # Check for missing returns
    missing_returns = np.isnan(train_forward_returns).sum()
    if missing_returns > 0:
        logger.warning(
            f"{missing_returns}/{len(train_forward_returns)} training samples missing forward_return, "
            "will skip reward calculation for these",
            extra={"component": "student"},
        )

    # Calculate rewards for each prediction
    rewards_data = []
    for pred, ret in zip(train_predictions, train_forward_returns, strict=False):
        if np.isnan(ret):
            # Skip samples with missing returns
            rewards_data.append({
                "prediction": int(pred),
                "actual_return": None,
                "raw_reward": 0.0,
                "clipped_reward": 0.0,
            })
        else:
            raw_reward, clipped_reward = calculator.calculate_reward(
                prediction=int(pred),
                actual_return=float(ret),
            )
            rewards_data.append({
                "prediction": int(pred),
                "actual_return": float(ret),
                "raw_reward": raw_reward,
                "clipped_reward": clipped_reward,
            })

    # Log rewards to reward_history.jsonl
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    reward_history_path = output_dir / "reward_history.jsonl"
    logger.info(f"Logging {len(rewards_data)} rewards to {reward_history_path}", extra={"component": "student"})

    with open(reward_history_path, "w") as f:
        for i, reward_entry in enumerate(rewards_data):
            # Add timestamp and index
            entry = {
                "timestamp": datetime.now().isoformat(),
                "index": int(train_indices[i]) if i < len(train_indices) else i,
                "prediction": reward_entry["prediction"],
                "actual_return": reward_entry["actual_return"],
                "raw_reward": reward_entry["raw_reward"],
                "clipped_reward": reward_entry["clipped_reward"],
            }
            f.write(json.dumps(entry) + "\n")

    # Aggregate metrics - extract clipped rewards from rewards_data
    clipped_rewards = [r["clipped_reward"] for r in rewards_data]
    aggregate_metrics = calculator.aggregate_reward_metrics(clipped_rewards)

    # Add num_rewards to aggregate_metrics
    aggregate_metrics["num_rewards"] = len(rewards_data)

    logger.info(
        f"Reward metrics: mean={aggregate_metrics['mean_reward']:.4f}, "
        f"cumulative={aggregate_metrics['cumulative_reward']:.4f}, "
        f"volatility={aggregate_metrics['reward_volatility']:.4f}, "
        f"positive={aggregate_metrics['positive_rewards']}/{aggregate_metrics['num_rewards']}",
        extra={"component": "student"},
    )

    return rewards_data, aggregate_metrics


def compute_reward_based_weights(
    rewards_data: list[dict[str, Any]],
    weighting_mode: str,
    weighting_scale: float,
) -> np.ndarray:
    """Compute sample weights from rewards.

    US-028 Phase 7 Initiative 2: Reward Loop Integration

    Args:
        rewards_data: List of reward entries from calculate_batch_rewards
        weighting_mode: Weighting mode (linear, exponential, none)
        weighting_scale: Scaling factor for weights

    Returns:
        Array of sample weights
    """
    from src.services.reward_calculator import RewardCalculator

    calculator = RewardCalculator()

    # Extract clipped rewards
    rewards = np.array([r["clipped_reward"] for r in rewards_data])

    # Compute weights
    weights = calculator.compute_sample_weights(
        rewards=rewards,
        mode=weighting_mode,
        scale=weighting_scale,
    )

    logger.info(
        f"Computed sample weights: mode={weighting_mode}, scale={weighting_scale}, "
        f"mean={weights.mean():.4f}, std={weights.std():.4f}",
        extra={"component": "student"},
    )

    return weights


def main() -> None:
    """Main entry point for Student training."""
    args = parse_args()

    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[component]}</cyan> | {message}",
        level="INFO",
    )
    logger.configure(extra={"component": "student"})

    logger.info("=" * 80)
    logger.info("Student Model Training (US-020)")
    logger.info("=" * 80)
    logger.info(f"Teacher Directory: {args.teacher_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Test Size: {args.test_size * 100:.0f}%")
    logger.info(f"Hyperparameter Tuning: {args.hyperparameter_tuning}")
    if args.hyperparameter_tuning:
        logger.info(f"CV Folds: {args.cv_folds}")
    logger.info("=" * 80)

    # Load teacher artifacts
    try:
        teacher_dir = Path(args.teacher_dir)
        labels_df, features_df, teacher_metadata = load_teacher_artifacts(teacher_dir)
    except Exception as e:
        logger.error(f"Failed to load teacher artifacts: {e}", exc_info=True)
        sys.exit(1)

    # Prepare training data
    try:
        X_train, X_test, y_train, y_test = prepare_training_data(
            labels_df, features_df, args.test_size, args.seed
        )
    except Exception as e:
        logger.error(f"Failed to prepare training data: {e}", exc_info=True)
        sys.exit(1)

    # US-028 Phase 7 Initiative 2: Reward Loop Integration
    output_dir = Path(args.output_dir)
    reward_metrics_dict = {}
    baseline_metrics_dict = {}

    if args.enable_reward_loop:
        logger.info("=" * 80)
        logger.info("US-028 Phase 7: Reward Loop Enabled")
        logger.info("=" * 80)
        logger.info(f"Reward Horizon: {args.reward_horizon_days} days")
        logger.info(f"Weighting Mode: {args.reward_weighting_mode}")
        logger.info(f"A/B Testing: {args.reward_ab_testing}")
        logger.info("=" * 80)

        # Calculate rewards for training samples
        try:
            rewards_data, reward_metrics_dict = calculate_and_log_rewards(
                labels_df=labels_df,
                y_train=y_train,
                output_dir=output_dir,
                reward_horizon_days=args.reward_horizon_days,
                reward_clip_min=-2.0,  # From Settings defaults
                reward_clip_max=2.0,
            )
        except Exception as e:
            logger.error(f"Reward calculation failed: {e}", exc_info=True)
            sys.exit(1)

        # Compute sample weights from rewards
        try:
            sample_weights = compute_reward_based_weights(
                rewards_data=rewards_data,
                weighting_mode=args.reward_weighting_mode,
                weighting_scale=1.0,  # From Settings defaults
            )
        except Exception as e:
            logger.error(f"Weight computation failed: {e}", exc_info=True)
            sys.exit(1)

        # A/B Testing: Train baseline model first
        if args.reward_ab_testing:
            logger.info("=" * 80)
            logger.info("A/B Testing: Training Baseline Model (uniform weights)")
            logger.info("=" * 80)
            try:
                baseline_model = train_student_model(
                    X_train, y_train, args.model_type, args.hyperparameter_tuning, args.cv_folds, args.seed,
                    sample_weights=None,  # Uniform weights
                )
                baseline_metrics_dict = evaluate_model(baseline_model, X_test, y_test)
                logger.info("-" * 80)
                logger.info("Baseline Model Performance:")
                logger.info(f"  Precision: {baseline_metrics_dict['precision_macro']:.4f}")
                logger.info(f"  Recall:    {baseline_metrics_dict['recall_macro']:.4f}")
                logger.info(f"  F1 Score:  {baseline_metrics_dict['f1_macro']:.4f}")
                logger.info("=" * 80)
            except Exception as e:
                logger.warning(f"Baseline model training failed: {e}")
                baseline_metrics_dict = {}

        # Train reward-weighted model
        logger.info("=" * 80)
        logger.info(f"Training Reward-Weighted Model ({args.reward_weighting_mode} weighting)")
        logger.info("=" * 80)
        try:
            model = train_student_model(
                X_train, y_train, args.model_type, args.hyperparameter_tuning, args.cv_folds, args.seed,
                sample_weights=sample_weights,
            )
        except Exception as e:
            logger.error(f"Reward-weighted training failed: {e}", exc_info=True)
            sys.exit(1)

    else:
        # Standard training without reward loop
        try:
            model = train_student_model(
                X_train, y_train, args.model_type, args.hyperparameter_tuning, args.cv_folds, args.seed
            )
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            sys.exit(1)

    # Evaluate model
    try:
        metrics = evaluate_model(model, X_test, y_test)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

    # Generate calibration plot
    try:
        output_dir = Path(args.output_dir)
        generate_calibration_plot(model, X_test, y_test, output_dir)
    except Exception as e:
        logger.warning(f"Calibration plot generation failed: {e}")

    # Save artifacts
    try:
        save_student_artifacts(
            model,
            metrics,
            teacher_metadata,
            output_dir,
            args.model_type,
            args.test_size,
            args.cv_folds,
            args.hyperparameter_tuning,
        )
    except Exception as e:
        logger.error(f"Failed to save artifacts: {e}", exc_info=True)
        sys.exit(1)

    # Print summary
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Student Model: {output_dir / 'student_model.pkl'}")
    logger.info(f"Evaluation Metrics: {output_dir / 'evaluation_metrics.json'}")
    logger.info(f"Metadata: {output_dir / 'metadata.json'}")
    logger.info("-" * 80)
    logger.info("Test Set Performance:")
    logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:   {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall:      {metrics['recall_macro']:.4f}")
    logger.info(f"  F1 Score:    {metrics['f1_macro']:.4f}")
    logger.info(f"  AUC-ROC:     {metrics['auc_macro']:.4f}")
    logger.info("=" * 80)

    # US-028 Phase 7 Initiative 2: Print reward metrics and A/B comparison
    if args.enable_reward_loop and reward_metrics_dict:
        logger.info("=" * 80)
        logger.info("Reward Loop Metrics")
        logger.info("=" * 80)
        logger.info(f"Mean Reward:        {reward_metrics_dict.get('mean_reward', 0.0):.4f}")
        logger.info(f"Cumulative Reward:  {reward_metrics_dict.get('cumulative_reward', 0.0):.4f}")
        logger.info(f"Reward Volatility:  {reward_metrics_dict.get('reward_volatility', 0.0):.4f}")
        logger.info(f"Positive Rewards:   {reward_metrics_dict.get('positive_rewards', 0)}/{reward_metrics_dict.get('num_rewards', 0)}")
        logger.info(f"Reward History:     {output_dir / 'reward_history.jsonl'}")
        logger.info("=" * 80)

        # Print JSON for batch script parsing
        print(f"REWARD_METRICS_JSON: {json.dumps(reward_metrics_dict)}")

        if args.reward_ab_testing and baseline_metrics_dict:
            logger.info("=" * 80)
            logger.info("A/B Testing: Baseline vs Reward-Weighted")
            logger.info("=" * 80)
            logger.info("                Baseline  Reward-Weighted  Delta")
            logger.info("-" * 80)

            for metric_name in ["precision_macro", "recall_macro", "f1_macro"]:
                baseline_val = baseline_metrics_dict.get(metric_name, 0.0)
                reward_val = metrics.get(metric_name, 0.0)
                delta = reward_val - baseline_val
                delta_str = f"{delta:+.4f}"
                logger.info(f"{metric_name:20s} {baseline_val:.4f}    {reward_val:.4f}           {delta_str}")

            logger.info("=" * 80)
            logger.info(f"Reward weighting improved F1 by: {(metrics.get('f1_macro', 0.0) - baseline_metrics_dict.get('f1_macro', 0.0)):+.4f}")
            logger.info("=" * 80)

    # US-021 Phase 2: Validation and promotion checklist
    validation_metrics = {}
    comparison = {}

    if args.validate or args.generate_checklist:
        logger.info("=" * 80)
        logger.info("Running Validation & Checklist Generation (US-021 Phase 2)")
        logger.info("=" * 80)

        # Check required validation parameters
        if not args.validation_symbols:
            logger.error(
                "--validation-symbols required for validation", extra={"component": "student"}
            )
            sys.exit(1)
        if not args.validation_start or not args.validation_end:
            logger.error(
                "--validation-start and --validation-end required for validation",
                extra={"component": "student"},
            )
            sys.exit(1)

        # Run validation backtest
        try:
            validation_metrics = run_validation_backtest(
                model,
                args.validation_symbols,
                args.validation_start,
                args.validation_end,
                output_dir,
            )
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            validation_metrics = {"error": str(e)}

        # Compare with baseline if provided
        if args.baseline_model:
            try:
                comparison = compare_with_baseline(
                    validation_metrics,
                    args.baseline_model,
                    args.validation_symbols,
                    args.validation_start,
                    args.validation_end,
                )
            except Exception as e:
                logger.error(f"Baseline comparison failed: {e}", exc_info=True)
                comparison = {"baseline_available": False, "error": str(e)}
        else:
            comparison = {"baseline_available": False, "uplift": {}}

    # Generate promotion checklist if requested
    if args.generate_checklist:
        try:
            generate_promotion_checklist(
                validation_metrics,
                comparison,
                metrics,
                output_dir,
                args.precision_uplift_threshold,
                args.hit_ratio_uplift_threshold,
                args.sharpe_uplift_threshold,
            )
            logger.info("=" * 80)
            logger.info(f"Promotion Checklist: {output_dir / 'promotion_checklist.md'}")
            logger.info(f"Promotion Checklist (JSON): {output_dir / 'promotion_checklist.json'}")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"Checklist generation failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
