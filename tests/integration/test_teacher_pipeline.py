"""Integration tests for Teacher training pipeline."""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.domain.types import Bar, TrainingConfig
from src.services.teacher_student import TeacherLabeler


@pytest.fixture
def large_sample_data() -> pd.DataFrame:
    """Create larger sample dataset for realistic training."""
    dates = pd.date_range("2023-01-01", periods=250, freq="D")  # ~1 year of trading days

    # Create realistic price data with trends and volatility
    base_price = 100.0
    prices = []
    for i in range(250):
        # Add trend + noise
        trend = i * 0.1
        noise = (i % 10 - 5) * 0.5
        price = base_price + trend + noise
        prices.append(price)

    return pd.DataFrame(
        {
            "ts": dates,
            "open": [p - 1.0 for p in prices],
            "high": [p + 2.0 for p in prices],
            "low": [p - 2.0 for p in prices],
            "close": prices,
            "volume": [1000 + i * 10 + (i % 7) * 50 for i in range(250)],
        }
    )


@pytest.fixture
def mock_client_large_data(large_sample_data: pd.DataFrame) -> MagicMock:
    """Create mock client that returns large sample dataset."""
    client = MagicMock()

    bars = [
        Bar(
            ts=row["ts"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=int(row["volume"]),
        )
        for _, row in large_sample_data.iterrows()
    ]

    client.historical_bars.return_value = bars
    return client


def test_full_training_pipeline(mock_client_large_data: MagicMock) -> None:
    """Test complete Teacher training pipeline end-to-end."""
    config = TrainingConfig(
        symbol="RELIANCE",
        start_date="2023-01-01",
        end_date="2023-12-31",
        label_window_days=5,
        label_threshold_pct=0.02,
        train_split=0.8,
        random_seed=42,
    )

    teacher = TeacherLabeler(config, client=mock_client_large_data)

    # Run full pipeline
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.services.teacher_student.Path") as mock_path:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            mock_path.return_value = models_dir

            result = teacher.run_full_pipeline()

            # Verify result structure
            assert result.model_path.endswith(".pkl")
            assert result.labels_path.endswith(".csv")
            assert result.importance_path.endswith(".csv")
            assert result.metadata_path.endswith(".json")

            # Verify metrics exist and are reasonable
            assert "val_accuracy" in result.metrics
            assert "val_f1" in result.metrics
            assert "val_auc" in result.metrics

            # Metrics should be between 0 and 1
            assert 0 <= result.metrics["val_accuracy"] <= 1
            assert 0 <= result.metrics["val_f1"] <= 1
            assert 0 <= result.metrics["val_auc"] <= 1

            # Verify sample counts
            assert result.train_samples > 0
            assert result.val_samples > 0
            assert result.feature_count > 0


def test_trained_model_predictions(mock_client_large_data: MagicMock) -> None:
    """Test that trained model can make predictions on new data."""
    config = TrainingConfig(
        symbol="RELIANCE",
        start_date="2023-01-01",
        end_date="2023-12-31",
        random_seed=42,
    )

    teacher = TeacherLabeler(config, client=mock_client_large_data)

    # Load and prepare data
    df = teacher.load_historical_data()
    df_features = teacher.generate_features(df)
    df_labeled, labels = teacher.generate_labels(df_features)

    # Train model
    result = teacher.train(df_labeled, labels)
    model = result["model"]

    # Prepare test features (exclude metadata columns)
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
    feature_cols = [col for col in df_labeled.columns if col not in exclude_cols]
    X_test = df_labeled[feature_cols].head(10)  # noqa: N806

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Verify prediction shape and values
    assert len(predictions) == 10
    assert predictions.shape == (10,)
    assert probabilities.shape == (10, 2)  # Binary classification

    # Predictions should be 0 or 1
    assert set(predictions).issubset({0, 1})

    # Probabilities should sum to 1 for each sample
    assert all(abs(prob.sum() - 1.0) < 0.001 for prob in probabilities)


def test_artifact_completeness(mock_client_large_data: MagicMock) -> None:
    """Test that all artifacts are created and contain expected data."""
    config = TrainingConfig(
        symbol="RELIANCE",
        start_date="2023-01-01",
        end_date="2023-12-31",
        random_seed=42,
    )

    teacher = TeacherLabeler(config, client=mock_client_large_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.services.teacher_student.Path") as mock_path:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            mock_path.return_value = models_dir

            result = teacher.run_full_pipeline()

            # Verify all files exist
            assert Path(result.model_path).exists()
            assert Path(result.labels_path).exists()
            assert Path(result.importance_path).exists()
            assert Path(result.metadata_path).exists()

            # Verify model can be loaded
            with open(result.model_path, "rb") as f:
                loaded_model = pickle.load(f)
            assert loaded_model is not None
            assert hasattr(loaded_model, "predict")

            # Verify labels CSV structure
            labels_df = pd.read_csv(result.labels_path)
            assert "ts" in labels_df.columns
            assert "symbol" in labels_df.columns
            assert "label" in labels_df.columns
            assert "forward_return" in labels_df.columns
            assert len(labels_df) > 0
            assert (labels_df["symbol"] == "RELIANCE").all()

            # Verify importance CSV structure
            importance_df = pd.read_csv(result.importance_path)
            assert "feature" in importance_df.columns
            assert "importance" in importance_df.columns
            assert "rank" in importance_df.columns
            assert len(importance_df) > 0
            # Importance should be sorted by rank
            assert (importance_df["rank"] == range(1, len(importance_df) + 1)).all()

            # Verify metadata JSON structure
            with open(result.metadata_path) as f:
                metadata = json.load(f)
            assert metadata["symbol"] == "RELIANCE"
            assert "metrics" in metadata
            assert "config" in metadata
            assert "label_distribution" in metadata


def test_reproducible_training(mock_client_large_data: MagicMock) -> None:
    """Test that same seed produces identical results."""
    config = TrainingConfig(
        symbol="RELIANCE",
        start_date="2023-01-01",
        end_date="2023-12-31",
        random_seed=42,  # Fixed seed
    )

    # First training run
    teacher1 = TeacherLabeler(config, client=mock_client_large_data)
    df1 = teacher1.load_historical_data()
    df_features1 = teacher1.generate_features(df1)
    df_labeled1, labels1 = teacher1.generate_labels(df_features1)
    result1 = teacher1.train(df_labeled1, labels1)

    # Second training run with same seed
    teacher2 = TeacherLabeler(config, client=mock_client_large_data)
    df2 = teacher2.load_historical_data()
    df_features2 = teacher2.generate_features(df2)
    df_labeled2, labels2 = teacher2.generate_labels(df_features2)
    result2 = teacher2.train(df_labeled2, labels2)

    # Results should be identical
    assert result1["metrics"]["val_accuracy"] == result2["metrics"]["val_accuracy"]
    assert result1["metrics"]["val_f1"] == result2["metrics"]["val_f1"]
    assert result1["metrics"]["val_auc"] == result2["metrics"]["val_auc"]
    assert result1["train_samples"] == result2["train_samples"]
    assert result1["val_samples"] == result2["val_samples"]


def test_different_label_configurations(mock_client_large_data: MagicMock) -> None:
    """Test training with different labeling configurations."""
    # Configuration with longer window
    config_long = TrainingConfig(
        symbol="RELIANCE",
        start_date="2023-01-01",
        end_date="2023-12-31",
        label_window_days=10,  # Longer window
        label_threshold_pct=0.03,  # Higher threshold
        random_seed=42,
    )

    teacher_long = TeacherLabeler(config_long, client=mock_client_large_data)
    df_long = teacher_long.load_historical_data()
    df_features_long = teacher_long.generate_features(df_long)
    df_labeled_long, labels_long = teacher_long.generate_labels(df_features_long)

    # Configuration with shorter window
    config_short = TrainingConfig(
        symbol="RELIANCE",
        start_date="2023-01-01",
        end_date="2023-12-31",
        label_window_days=3,  # Shorter window
        label_threshold_pct=0.01,  # Lower threshold
        random_seed=42,
    )

    teacher_short = TeacherLabeler(config_short, client=mock_client_large_data)
    df_short = teacher_short.load_historical_data()
    df_features_short = teacher_short.generate_features(df_short)
    df_labeled_short, labels_short = teacher_short.generate_labels(df_features_short)

    # Longer window should have fewer samples (more rows dropped for lookahead)
    assert len(df_labeled_long) < len(df_labeled_short)

    # Higher threshold should produce fewer positive labels
    assert labels_long.sum() <= labels_short.sum()


def test_feature_importance_ranking(mock_client_large_data: MagicMock) -> None:
    """Test that feature importance is properly ranked."""
    config = TrainingConfig(
        symbol="RELIANCE",
        start_date="2023-01-01",
        end_date="2023-12-31",
        random_seed=42,
    )

    teacher = TeacherLabeler(config, client=mock_client_large_data)
    df = teacher.load_historical_data()
    df_features = teacher.generate_features(df)
    df_labeled, labels = teacher.generate_labels(df_features)
    result = teacher.train(df_labeled, labels)

    importance = result["importance"]

    # Verify importance is sorted descending
    importance_values = importance["importance"].tolist()
    assert importance_values == sorted(importance_values, reverse=True)

    # Verify ranks are sequential
    assert importance["rank"].tolist() == list(range(1, len(importance) + 1))

    # Top feature should have highest importance
    assert importance.iloc[0]["rank"] == 1
    assert importance.iloc[0]["importance"] >= importance.iloc[-1]["importance"]
