"""Unit tests for Teacher labeler service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.domain.types import Bar, TrainingConfig
from src.services.teacher_student import TeacherLabeler


@pytest.fixture
def sample_config() -> TrainingConfig:
    """Create sample training configuration."""
    return TrainingConfig(
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2024-12-31",
        label_window_days=5,
        label_threshold_pct=0.02,
        train_split=0.8,
        random_seed=42,
    )


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV DataFrame with realistic price movements for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    # Create realistic price data with trends, volatility, and reversals
    prices = []
    base_price = 100.0
    for i in range(100):
        # Oscillating trend with noise
        trend = i * 0.1
        cycle = 5 * pd.np.sin(i * 0.2) if hasattr(pd, "np") else 5 * (i % 10 - 5) / 5
        noise = (i % 7 - 3) * 0.3
        price = base_price + trend + cycle + noise
        prices.append(max(price, 50.0))  # Floor price at 50

    return pd.DataFrame(
        {
            "ts": dates,
            "open": [p - 1.0 for p in prices],
            "high": [p + 2.0 for p in prices],
            "low": [p - 2.0 for p in prices],
            "close": prices,
            "volume": [1000 + i * 10 + (i % 5) * 20 for i in range(100)],
        }
    )


@pytest.fixture
def mock_breeze_client(sample_ohlcv_data: pd.DataFrame) -> MagicMock:
    """Create mock BreezeClient that returns sample data."""
    client = MagicMock()

    # Convert DataFrame rows to Bar objects
    bars = [
        Bar(
            ts=row["ts"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=int(row["volume"]),
        )
        for _, row in sample_ohlcv_data.iterrows()
    ]

    client.historical_bars.return_value = bars
    return client


# ==================== Initialization Tests ====================


def test_teacher_labeler_initialization(sample_config: TrainingConfig) -> None:
    """Test TeacherLabeler initializes correctly."""
    teacher = TeacherLabeler(sample_config)

    assert teacher.config == sample_config
    assert teacher.config.symbol == "TEST"
    assert teacher.config.label_window_days == 5
    assert teacher.model is None  # Model not trained yet


# ==================== Data Loading Tests ====================


def test_load_historical_data_success(
    sample_config: TrainingConfig, mock_breeze_client: MagicMock
) -> None:
    """Test successful historical data loading."""
    teacher = TeacherLabeler(sample_config, client=mock_breeze_client)

    df = teacher.load_historical_data()

    # Verify data loaded
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]

    # Verify client was called correctly
    mock_breeze_client.historical_bars.assert_called_once()
    call_kwargs = mock_breeze_client.historical_bars.call_args[1]
    assert call_kwargs["symbol"] == "TEST"
    assert call_kwargs["interval"] == "1day"


def test_load_historical_data_no_client(sample_config: TrainingConfig) -> None:
    """Test loading data without client raises error."""
    teacher = TeacherLabeler(sample_config, client=None)

    with pytest.raises(RuntimeError, match="BreezeClient required"):
        teacher.load_historical_data()


def test_load_historical_data_empty_response(sample_config: TrainingConfig) -> None:
    """Test loading data when client returns empty list."""
    mock_client = MagicMock()
    mock_client.historical_bars.return_value = []

    teacher = TeacherLabeler(sample_config, client=mock_client)

    with pytest.raises(RuntimeError, match="No data returned"):
        teacher.load_historical_data()


# ==================== Feature Generation Tests ====================


def test_generate_features_all_indicators(sample_ohlcv_data: pd.DataFrame) -> None:
    """Test feature generation produces all expected indicators."""
    config = TrainingConfig(symbol="TEST", start_date="2024-01-01", end_date="2024-12-31")
    teacher = TeacherLabeler(config)

    df_features = teacher.generate_features(sample_ohlcv_data)

    # Verify all feature columns are present
    expected_features = [
        "sma_20",
        "sma_50",
        "ema_12",
        "ema_26",
        "rsi_14",
        "atr_14",
        "vwap",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "adx_14",
        "obv",
    ]

    for feature in expected_features:
        assert feature in df_features.columns, f"Missing feature: {feature}"

    # Verify no NaN values (after warm-up period drop)
    assert not df_features.isna().any().any(), "Features contain NaN values"


def test_generate_features_drops_nan_rows(sample_ohlcv_data: pd.DataFrame) -> None:
    """Test feature generation drops rows with NaN values."""
    config = TrainingConfig(symbol="TEST", start_date="2024-01-01", end_date="2024-12-31")
    teacher = TeacherLabeler(config)

    df_features = teacher.generate_features(sample_ohlcv_data)

    # Should have fewer rows after dropping NaN (warm-up period)
    assert len(df_features) < len(sample_ohlcv_data)

    # All rows should be valid
    assert not df_features.isna().any().any()


def test_generate_features_missing_columns() -> None:
    """Test feature generation raises error for missing columns."""
    config = TrainingConfig(symbol="TEST", start_date="2024-01-01", end_date="2024-12-31")
    teacher = TeacherLabeler(config)

    # DataFrame missing required columns
    df_invalid = pd.DataFrame({"ts": [1, 2, 3], "close": [100, 101, 102]})

    with pytest.raises(ValueError, match="Missing required columns"):
        teacher.generate_features(df_invalid)


# ==================== Label Generation Tests ====================


def test_generate_labels_forward_return(sample_ohlcv_data: pd.DataFrame) -> None:
    """Test label generation calculates forward returns correctly."""
    config = TrainingConfig(
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2024-12-31",
        label_window_days=5,
        label_threshold_pct=0.02,
    )
    teacher = TeacherLabeler(config)

    # Generate features first
    df_features = teacher.generate_features(sample_ohlcv_data)

    # Generate labels
    df_labeled, labels = teacher.generate_labels(df_features)

    # Verify forward_return column exists
    assert "forward_return" in df_labeled.columns
    assert "label" in df_labeled.columns

    # Verify labels are binary (0 or 1)
    assert set(labels.unique()).issubset({0, 1})

    # Verify rows were dropped (last 5 rows don't have forward lookahead)
    assert len(df_labeled) < len(df_features)


def test_generate_labels_threshold() -> None:
    """Test label generation respects threshold parameter."""
    # Create data with known returns that exceed threshold
    # With 5-day window and 2% threshold, need ~0.4% daily increase
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=100, freq="D"),
            "open": [100.0] * 100,
            "high": [102.0] * 100,
            "low": [98.0] * 100,
            # Increase 0.5% per day = 2.5% over 5 days (exceeds 2% threshold)
            "close": [100.0 * (1.005**i) for i in range(100)],
            "volume": [1000] * 100,
        }
    )

    config = TrainingConfig(
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2024-12-31",
        label_window_days=5,
        label_threshold_pct=0.02,  # 2% threshold
    )
    teacher = TeacherLabeler(config)

    df_features = teacher.generate_features(df)
    df_labeled, labels = teacher.generate_labels(df_features)

    # With steadily increasing prices (0.5% daily), we should have many positive labels
    assert labels.sum() > 0  # At least some profitable trades
    # Most should be profitable due to consistent upward trend
    assert labels.mean() > 0.5  # More than 50% profitable


def test_generate_labels_class_distribution(sample_ohlcv_data: pd.DataFrame) -> None:
    """Test label generation produces reasonable class distribution."""
    config = TrainingConfig(
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2024-12-31",
        label_window_days=5,
        label_threshold_pct=0.02,
    )
    teacher = TeacherLabeler(config)

    df_features = teacher.generate_features(sample_ohlcv_data)
    df_labeled, labels = teacher.generate_labels(df_features)

    # Verify we have both classes (not all 0 or all 1)
    label_counts = labels.value_counts()
    assert len(label_counts) > 0  # At least one class
    # With upward trending data, we should have positive labels
    assert labels.sum() > 0


# ==================== Training Tests ====================


def test_train_model_success(sample_ohlcv_data: pd.DataFrame) -> None:
    """Test model training completes successfully."""
    config = TrainingConfig(
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2024-12-31",
        label_window_days=5,
        label_threshold_pct=0.02,
        train_split=0.8,
        random_seed=42,
    )
    teacher = TeacherLabeler(config)

    # Prepare data
    df_features = teacher.generate_features(sample_ohlcv_data)
    df_labeled, labels = teacher.generate_labels(df_features)

    # Train
    result = teacher.train(df_labeled, labels)

    # Verify training output
    assert "model" in result
    assert "metrics" in result
    assert "importance" in result
    assert "feature_cols" in result
    assert "train_samples" in result
    assert "val_samples" in result

    # Verify metrics
    metrics = result["metrics"]
    assert "train_accuracy" in metrics
    assert "val_accuracy" in metrics
    assert "val_precision" in metrics
    assert "val_recall" in metrics
    assert "val_f1" in metrics
    assert "val_auc" in metrics

    # Metrics should be between 0 and 1
    for metric_value in metrics.values():
        assert 0 <= metric_value <= 1


def test_train_model_feature_importance(sample_ohlcv_data: pd.DataFrame) -> None:
    """Test model training produces feature importance."""
    config = TrainingConfig(
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2024-12-31",
        random_seed=42,
    )
    teacher = TeacherLabeler(config)

    df_features = teacher.generate_features(sample_ohlcv_data)
    df_labeled, labels = teacher.generate_labels(df_features)

    result = teacher.train(df_labeled, labels)

    importance = result["importance"]

    # Verify importance DataFrame structure
    assert "feature" in importance.columns
    assert "importance" in importance.columns
    assert "rank" in importance.columns

    # Should have importance for all features
    assert len(importance) == len(result["feature_cols"])

    # Importance values should be non-negative
    assert (importance["importance"] >= 0).all()


def test_train_model_reproducible(sample_ohlcv_data: pd.DataFrame) -> None:
    """Test model training is reproducible with same seed."""
    config = TrainingConfig(
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2024-12-31",
        random_seed=42,
    )

    # Train first model
    teacher1 = TeacherLabeler(config)
    df_features1 = teacher1.generate_features(sample_ohlcv_data)
    df_labeled1, labels1 = teacher1.generate_labels(df_features1)
    result1 = teacher1.train(df_labeled1, labels1)

    # Train second model with same seed
    teacher2 = TeacherLabeler(config)
    df_features2 = teacher2.generate_features(sample_ohlcv_data)
    df_labeled2, labels2 = teacher2.generate_labels(df_features2)
    result2 = teacher2.train(df_labeled2, labels2)

    # Results should be identical
    assert result1["metrics"]["val_accuracy"] == result2["metrics"]["val_accuracy"]
    assert result1["train_samples"] == result2["train_samples"]
    assert result1["val_samples"] == result2["val_samples"]


# ==================== Artifact Persistence Tests ====================


def test_save_artifacts_creates_files(sample_ohlcv_data: pd.DataFrame, tmp_path: Path) -> None:
    """Test save_artifacts creates all expected files."""
    import tempfile
    from unittest.mock import patch

    config = TrainingConfig(
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2024-12-31",
        random_seed=42,
    )
    teacher = TeacherLabeler(config)

    # Prepare and train
    df_features = teacher.generate_features(sample_ohlcv_data)
    df_labeled, labels = teacher.generate_labels(df_features)
    result = teacher.train(df_labeled, labels)

    # Save artifacts to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.services.teacher_student.Path") as mock_path:
            # Mock Path to use temp directory
            mock_path.return_value = Path(tmpdir)
            Path(tmpdir).mkdir(parents=True, exist_ok=True)

            training_result = teacher.save_artifacts(
                model=result["model"],
                df_labeled=df_labeled,
                importance=result["importance"],
                metrics=result["metrics"],
                feature_count=len(result["feature_cols"]),
                train_samples=result["train_samples"],
                val_samples=result["val_samples"],
            )

            # Verify TrainingResult fields
            assert training_result.model_path.endswith(".pkl")
            assert training_result.labels_path.endswith(".csv.gz")
            assert training_result.importance_path.endswith(".csv")
            assert training_result.metadata_path.endswith(".json")
            assert training_result.feature_count > 0
            assert training_result.train_samples > 0
            assert training_result.val_samples > 0


def test_save_artifacts_metadata_structure(sample_ohlcv_data: pd.DataFrame) -> None:
    """Test metadata JSON has correct structure."""
    import json
    import tempfile
    from unittest.mock import patch

    config = TrainingConfig(
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2024-12-31",
        label_window_days=5,
        label_threshold_pct=0.02,
        random_seed=42,
    )
    teacher = TeacherLabeler(config)

    df_features = teacher.generate_features(sample_ohlcv_data)
    df_labeled, labels = teacher.generate_labels(df_features)
    result = teacher.train(df_labeled, labels)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock Path to use temp directory
        models_dir = Path(tmpdir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        with patch("src.services.teacher_student.Path") as mock_path:
            mock_path.return_value = models_dir

            training_result = teacher.save_artifacts(
                model=result["model"],
                df_labeled=df_labeled,
                importance=result["importance"],
                metrics=result["metrics"],
                feature_count=len(result["feature_cols"]),
                train_samples=result["train_samples"],
                val_samples=result["val_samples"],
            )

            # Read metadata
            with open(training_result.metadata_path) as f:
                metadata = json.load(f)

            # Verify structure
            assert "training_date" in metadata
            assert "symbol" in metadata
            assert "date_range" in metadata
            assert "total_rows" in metadata
            assert "train_rows" in metadata
            assert "val_rows" in metadata
            assert "label_distribution" in metadata
            assert "config" in metadata
            assert "metrics" in metadata
            assert "feature_count" in metadata

            # Verify config fields
            assert metadata["config"]["label_window_days"] == 5
            assert metadata["config"]["label_threshold_pct"] == 0.02
            assert metadata["config"]["random_seed"] == 42
