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
            assert result.labels_path.endswith(".csv.gz")
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


def test_batch_trainer_skips_insufficient_future_data() -> None:
    """Test that BatchTrainer skips windows with insufficient future data."""
    from datetime import datetime, timedelta
    from pathlib import Path
    from unittest.mock import MagicMock

    from src.app.config import Settings
    from scripts.train_teacher_batch import BatchTrainer

    # Create mock settings
    settings = MagicMock(spec=Settings)
    settings.historical_data_output_dir = "data/historical"
    settings.batch_training_output_dir = "data/models"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create batch trainer
        trainer = BatchTrainer(
            settings=settings,
            output_dir=output_dir,
            resume=False,
            incremental=False,
            workers=1,
        )

        # Create a mock task that requires data beyond what's available
        # Assume latest available data is 2024-11-30
        task = {
            "symbol": "RELIANCE",
            "start_date": "2024-09-01",
            "end_date": "2024-12-31",  # Would need data through 2025-03-31 (90 days after)
            "window_label": "RELIANCE_2024Q4",
            "artifacts_path": str(output_dir / "RELIANCE_2024Q4"),
        }

        # Mock get_latest_available_timestamp to return November 30, 2024
        original_method = trainer.get_latest_available_timestamp
        trainer.get_latest_available_timestamp = MagicMock(
            return_value=datetime(2024, 11, 30)
        )

        # Check if window should be skipped
        should_skip, reason = trainer.should_skip_window_insufficient_data(
            task, forecast_horizon=90
        )

        # Restore original method
        trainer.get_latest_available_timestamp = original_method

        # Should skip because end_date (2024-12-31) + 90 days = 2025-03-31 > 2024-11-30
        assert should_skip is True
        assert "Insufficient future data" in reason
        assert "2024-11-30" in reason

        # Now test a task that has sufficient data
        task_sufficient = {
            "symbol": "RELIANCE",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",  # Would need data through 2024-06-29 (90 days after)
            "window_label": "RELIANCE_2024Q1",
            "artifacts_path": str(output_dir / "RELIANCE_2024Q1"),
        }

        trainer.get_latest_available_timestamp = MagicMock(
            return_value=datetime(2024, 11, 30)
        )

        should_skip2, reason2 = trainer.should_skip_window_insufficient_data(
            task_sufficient, forecast_horizon=90
        )

        # Should NOT skip because 2024-03-31 + 90 days = 2024-06-29 < 2024-11-30
        assert should_skip2 is False
        assert reason2 == ""


def test_batch_trainer_deterministic_window_labels() -> None:
    """Test that window labels are deterministic and include explicit dates (US-028 Phase 6e)."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from src.app.config import Settings
    from scripts.train_teacher_batch import BatchTrainer

    settings = MagicMock(spec=Settings)
    settings.batch_training_output_dir = "data/models"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        trainer = BatchTrainer(settings=settings, output_dir=output_dir,
                              resume=False, incremental=False, workers=1)

        # Generate windows that might overlap calendar quarters
        tasks = trainer.generate_training_windows(
            symbols=["RELIANCE"],
            start_date="2024-01-15",  # Mid-month start
            end_date="2024-06-15",
            window_days=90
        )

        # Verify we have tasks
        assert len(tasks) > 0

        # Check each label format
        for task in tasks:
            label = task["window_label"]
            start = task["start_date"]
            end = task["end_date"]

            # Label should follow SYMBOL_YYYY-MM-DD_to_YYYY-MM-DD format
            assert label == f"RELIANCE_{start}_to_{end}"

            # Label should contain explicit dates
            assert start in label
            assert end in label
            assert "_to_" in label

        # Verify labels are unique
        labels = [task["window_label"] for task in tasks]
        assert len(labels) == len(set(labels)), "Window labels are not unique!"

        # Verify labels don't use old quarter format
        for label in labels:
            assert "Q1" not in label
            assert "Q2" not in label
            assert "Q3" not in label
            assert "Q4" not in label


def test_batch_trainer_error_reporting_with_traceback() -> None:
    """Test that training errors include full exception details and traceback (US-028 Phase 6e)."""
    import subprocess
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from src.app.config import Settings
    from scripts.train_teacher_batch import BatchTrainer

    settings = MagicMock(spec=Settings)
    settings.batch_training_output_dir = "data/models"
    settings.parallel_retry_limit = 1  # No retries for faster test

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        trainer = BatchTrainer(settings=settings, output_dir=output_dir,
                              resume=False, incremental=False, workers=1)

        task = {
            "symbol": "TEST",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "window_label": "TEST_2024-01-01_to_2024-03-31",
            "artifacts_path": str(output_dir / "TEST_2024-01-01_to_2024-03-31"),
        }

        # Test Case 1: subprocess failure with stderr
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ValueError: Training data has zero samples"
        mock_result.stdout = "DEBUG: Loading data...\nERROR: No valid samples found"

        with patch("subprocess.run", return_value=mock_result):
            result = trainer.train_window(task, forecast_horizon=7)

        assert result["status"] == "failed"
        assert "error" in result
        assert "ValueError" in result["error"] or "zero samples" in result["error"]
        assert "error_detail" in result
        assert result["error_detail"]["exit_code"] == 1
        assert "stderr" in result["error_detail"]

        # Test Case 2: Exception with traceback
        def raise_custom_error(*args, **kwargs):
            raise RuntimeError("Mock training failure for testing")

        with patch("subprocess.run", side_effect=raise_custom_error):
            result = trainer.train_window(task, forecast_horizon=7)

        assert result["status"] == "failed"
        assert "error" in result
        assert "RuntimeError" in result["error"]
        assert "Mock training failure" in result["error"]
        assert "error_detail" in result
        assert "traceback" in result["error_detail"]
        assert "RuntimeError" in result["error_detail"]["traceback"]

        # Test Case 3: Timeout
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 600)):
            result = trainer.train_window(task, forecast_horizon=7)

        assert result["status"] == "failed"
        assert "error" in result
        assert "timeout" in result["error"].lower()
        assert "error_detail" in result
        assert "timeout_seconds" in result["error_detail"]
        assert result["error_detail"]["timeout_seconds"] == 600


def test_batch_trainer_skips_zero_sample_windows() -> None:
    """Test that BatchTrainer skips windows with zero samples after filtering (US-028 Phase 6f)."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from src.app.config import Settings
    from scripts.train_teacher_batch import BatchTrainer

    settings = MagicMock(spec=Settings)
    settings.batch_training_output_dir = "data/models"
    settings.parallel_retry_limit = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        trainer = BatchTrainer(settings=settings, output_dir=output_dir,
                              resume=False, incremental=False, workers=1)

        task = {
            "symbol": "TEST",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "window_label": "TEST_2024-01-01_to_2024-03-31",
            "artifacts_path": str(output_dir / "TEST_2024-01-01_to_2024-03-31"),
        }

        # Mock subprocess that succeeds but returns zero samples
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """
2025-10-14 02:00:00 | INFO     | Training Complete!
Training Samples: 0
Validation Samples: 0
TEACHER_DIAGNOSTICS: {"sample_counts": {"train_samples": 0, "val_samples": 0, "total_samples": 0, "feature_count": 50}}
"""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = trainer.train_window(task, forecast_horizon=7)

        # Should be marked as skipped (not failed)
        assert result["status"] == "skipped"
        assert "reason" in result
        assert "Insufficient samples" in result["reason"]
        assert "0 total samples" in result["reason"]

        # Should include sample counts in result
        assert "sample_counts" in result
        assert result["sample_counts"]["total_samples"] == 0
        assert result["sample_counts"]["train_samples"] == 0
        assert result["sample_counts"]["val_samples"] == 0

        # Should not have metrics (no model trained)
        assert result["metrics"] is None


def test_batch_trainer_includes_sample_diagnostics_on_success() -> None:
    """Test that successful training includes sample count diagnostics (US-028 Phase 6f)."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from src.app.config import Settings
    from scripts.train_teacher_batch import BatchTrainer

    settings = MagicMock(spec=Settings)
    settings.batch_training_output_dir = "data/models"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        trainer = BatchTrainer(settings=settings, output_dir=output_dir,
                              resume=False, incremental=False, workers=1)

        task = {
            "symbol": "TEST",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "window_label": "TEST_2024-01-01_to_2024-03-31",
            "artifacts_path": str(output_dir / "TEST_2024-01-01_to_2024-03-31"),
        }

        # Mock successful training with sample counts
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """
2025-10-14 02:00:00 | INFO     | Training Complete!
Training Samples: 800
Validation Samples: 200
Precision: 0.85
TEACHER_DIAGNOSTICS: {"sample_counts": {"train_samples": 800, "val_samples": 200, "total_samples": 1000, "feature_count": 50}}
"""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = trainer.train_window(task, forecast_horizon=7)

        # Should succeed
        assert result["status"] == "success"

        # Should include sample counts
        assert "sample_counts" in result
        assert result["sample_counts"]["total_samples"] == 1000
        assert result["sample_counts"]["train_samples"] == 800
        assert result["sample_counts"]["val_samples"] == 200
        assert result["sample_counts"]["feature_count"] == 50

        # Should have metrics
        assert result["metrics"] is not None


def test_batch_trainer_skips_insufficient_samples_minimum_threshold() -> None:
    """Test that BatchTrainer skips windows with < 20 samples (US-028 Phase 6h)."""
    from unittest.mock import MagicMock, patch

    from scripts.train_teacher_batch import BatchTrainer
    from src.app.config import Settings

    settings = Settings()
    trainer = BatchTrainer(
        output_dir=Path("/tmp/test_batch"),
        workers=1,
        settings=settings,
    )

    # Create a test task
    task = {
        "symbol": "TEST",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "window_label": "TEST_2024-01-01_to_2024-03-31",
        "artifacts_path": "/tmp/test_batch/TEST_2024-01-01_to_2024-03-31",
    }

    # Mock subprocess that returns exit code 2 (skip) with insufficient samples message
    mock_result = MagicMock()
    mock_result.returncode = 2
    mock_result.stdout = """
Window skipped: Insufficient samples for training: 15 < 20 minimum.
TEACHER_SKIP: {"status": "skipped", "reason": "Insufficient samples for training: 15 < 20 minimum. Consider increasing window size or reducing forecast horizon."}
"""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = trainer.train_window(task, forecast_horizon=7)

    # Should be marked as skipped
    assert result["status"] == "skipped"
    assert "Insufficient samples for training" in result["reason"]
    assert result["sample_counts"] is None
    assert result["metrics"] is None


def test_batch_trainer_recognizes_exit_code_2_as_skip() -> None:
    """Test that BatchTrainer treats exit code 2 as skip, not failure (US-028 Phase 6h)."""
    from unittest.mock import MagicMock, patch

    from scripts.train_teacher_batch import BatchTrainer
    from src.app.config import Settings

    settings = Settings()
    trainer = BatchTrainer(
        output_dir=Path("/tmp/test_batch"),
        workers=1,
        settings=settings,
    )

    task = {
        "symbol": "TEST",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "window_label": "TEST_2024-01-01_to_2024-03-31",
        "artifacts_path": "/tmp/test_batch/TEST_2024-01-01_to_2024-03-31",
    }

    # Mock subprocess with exit code 2
    mock_result = MagicMock()
    mock_result.returncode = 2
    mock_result.stdout = 'TEACHER_SKIP: {"status": "skipped", "reason": "Test skip"}'
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = trainer.train_window(task, forecast_horizon=7)

    # Should be skipped, NOT failed
    assert result["status"] == "skipped"
    assert result["status"] != "failed"
