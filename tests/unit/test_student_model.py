"""Unit tests for StudentModel class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.domain.types import PredictionResult, StudentConfig
from src.services.teacher_student import StudentModel


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def teacher_artifacts(temp_dir: Path) -> tuple[Path, Path]:
    """Create mock Teacher artifacts."""
    # Create labels CSV
    labels_path = temp_dir / "teacher_labels.csv"
    labels_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="D"),
            "close": [100.0 + i * 0.5 for i in range(100)],
            "sma20": [100.0 + i * 0.4 for i in range(100)],
            "rsi14": [50.0 + (i % 20 - 10) for i in range(100)],
            "atr14": [2.0 + (i % 10) * 0.1 for i in range(100)],
            "label": [i % 2 for i in range(100)],  # Alternating labels for balance
        }
    )
    labels_df.to_csv(labels_path, index=False)

    # Create metadata JSON
    metadata_path = temp_dir / "teacher_metadata.json"
    metadata = {
        "symbol": "TEST",
        "training_date": "2025-01-15T10:00:00",
        "features": ["sma20", "rsi14", "atr14"],
        "model_type": "lightgbm",
        "config": {
            "label_window_days": 5,
            "label_threshold_pct": 0.02,
            "train_split": 0.8,
        },
        "metrics": {
            "train_accuracy": 0.85,
            "val_accuracy": 0.80,
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return metadata_path, labels_path


def test_student_initialization(teacher_artifacts: tuple[Path, Path]) -> None:
    """Verifies StudentModel initialization."""
    metadata_path, labels_path = teacher_artifacts

    config = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
        decision_threshold=0.6,
        random_seed=42,
    )

    student = StudentModel(config)

    assert student.config == config
    assert student.model is None  # Not trained yet
    assert student.feature_cols == []
    assert student.teacher_metadata is None


def test_load_teacher_artifacts(teacher_artifacts: tuple[Path, Path]) -> None:
    """Tests artifact loading."""
    metadata_path, labels_path = teacher_artifacts

    config = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
    )

    student = StudentModel(config)
    labels_df, labels = student.load_teacher_artifacts()

    assert len(labels_df) == 100
    assert len(labels) == 100
    assert "label" in labels_df.columns
    assert student.teacher_metadata is not None
    assert student.teacher_metadata["symbol"] == "TEST"
    assert student.teacher_metadata["features"] == ["sma20", "rsi14", "atr14"]


def test_load_teacher_artifacts_missing_file(temp_dir: Path) -> None:
    """Tests error handling for missing artifacts."""
    config = StudentConfig(
        teacher_metadata_path=str(temp_dir / "missing.json"),
        teacher_labels_path=str(temp_dir / "missing.csv"),
    )

    student = StudentModel(config)

    with pytest.raises(FileNotFoundError):
        student.load_teacher_artifacts()


def test_train_student_model(teacher_artifacts: tuple[Path, Path]) -> None:
    """Tests Student training."""
    metadata_path, labels_path = teacher_artifacts

    config = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
        random_seed=42,
    )

    student = StudentModel(config)
    labels_df, labels = student.load_teacher_artifacts()

    # Train on features
    features_df = labels_df[["sma20", "rsi14", "atr14"]]
    result = student.train(features_df, labels)

    assert student.model is not None
    assert student.feature_cols == ["sma20", "rsi14", "atr14"]
    assert "metrics" in result
    assert "train_accuracy" in result["metrics"]
    assert 0.0 <= result["metrics"]["train_accuracy"] <= 1.0


def test_train_with_class_imbalance(teacher_artifacts: tuple[Path, Path]) -> None:
    """Tests training with imbalanced classes."""
    metadata_path, labels_path = teacher_artifacts

    config = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
        random_seed=42,
    )

    student = StudentModel(config)
    labels_df, _ = student.load_teacher_artifacts()

    # Create imbalanced labels (90% class 0, 10% class 1)
    imbalanced_labels = pd.Series([0] * 90 + [1] * 10)
    features_df = labels_df[["sma20", "rsi14", "atr14"]]

    result = student.train(features_df, imbalanced_labels)

    # Should still train successfully
    assert student.model is not None
    assert "metrics" in result
    assert "train_accuracy" in result["metrics"]


def test_predict_single(teacher_artifacts: tuple[Path, Path]) -> None:
    """Tests single prediction."""
    metadata_path, labels_path = teacher_artifacts

    config = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
        decision_threshold=0.5,
        random_seed=42,
    )

    student = StudentModel(config)
    labels_df, labels = student.load_teacher_artifacts()
    features_df = labels_df[["sma20", "rsi14", "atr14"]]
    student.train(features_df, labels)

    # Make prediction
    test_features = {"sma20": 110.0, "rsi14": 55.0, "atr14": 2.5}
    result = student.predict_single(test_features, symbol="TEST")

    assert isinstance(result, PredictionResult)
    assert result.symbol == "TEST"
    assert 0.0 <= result.probability <= 1.0
    assert result.decision in [0, 1]
    assert result.confidence >= 0.0
    assert result.features_used == ["sma20", "rsi14", "atr14"]
    assert result.model_version is not None
    assert isinstance(result.metadata, dict)


def test_predict_single_missing_features(teacher_artifacts: tuple[Path, Path]) -> None:
    """Tests graceful handling of missing features."""
    metadata_path, labels_path = teacher_artifacts

    config = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
    )

    student = StudentModel(config)
    labels_df, labels = student.load_teacher_artifacts()
    features_df = labels_df[["sma20", "rsi14", "atr14"]]
    student.train(features_df, labels)

    # Missing one feature
    incomplete_features = {"sma20": 110.0, "rsi14": 55.0}

    with pytest.raises(ValueError, match="Missing required features"):
        student.predict_single(incomplete_features, symbol="TEST")


def test_predict_batch(teacher_artifacts: tuple[Path, Path]) -> None:
    """Tests batch predictions."""
    metadata_path, labels_path = teacher_artifacts

    config = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
        decision_threshold=0.5,
        random_seed=42,
    )

    student = StudentModel(config)
    labels_df, labels = student.load_teacher_artifacts()
    features_df = labels_df[["sma20", "rsi14", "atr14"]]
    student.train(features_df, labels)

    # Make batch predictions
    test_batch = pd.DataFrame(
        {
            "sma20": [110.0, 115.0, 120.0],
            "rsi14": [55.0, 60.0, 65.0],
            "atr14": [2.5, 2.8, 3.0],
        }
    )

    results = student.predict(test_batch)

    assert len(results) == 3
    for result in results:
        assert isinstance(result, PredictionResult)
        assert result.symbol == ""  # Batch predictions don't set symbol
        assert 0.0 <= result.probability <= 1.0
        assert result.decision in [0, 1]


def test_save_and_load(teacher_artifacts: tuple[Path, Path], temp_dir: Path) -> None:
    """Tests persistence."""
    metadata_path, labels_path = teacher_artifacts

    config = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
        random_seed=42,
    )

    # Train original model
    student = StudentModel(config)
    labels_df, labels = student.load_teacher_artifacts()
    features_df = labels_df[["sma20", "rsi14", "atr14"]]
    student.train(features_df, labels)

    # Save model
    model_path = temp_dir / "student_model.pkl"
    student_metadata_path = temp_dir / "student_metadata.json"
    student.save(str(model_path), str(student_metadata_path))

    assert model_path.exists()
    assert student_metadata_path.exists()

    # Load model into new instance
    new_student = StudentModel(config)
    new_student.load(str(model_path), str(student_metadata_path))

    assert new_student.model is not None
    assert new_student.feature_cols == ["sma20", "rsi14", "atr14"]

    # Verify predictions match
    test_features = {"sma20": 110.0, "rsi14": 55.0, "atr14": 2.5}
    original_result = student.predict_single(test_features, symbol="TEST")
    loaded_result = new_student.predict_single(test_features, symbol="TEST")

    assert original_result.probability == loaded_result.probability
    assert original_result.decision == loaded_result.decision


def test_incremental_retraining(teacher_artifacts: tuple[Path, Path]) -> None:
    """Tests incremental updates."""
    metadata_path, labels_path = teacher_artifacts

    config = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
        incremental=True,
        random_seed=42,
    )

    student = StudentModel(config)
    labels_df, labels = student.load_teacher_artifacts()
    features_df = labels_df[["sma20", "rsi14", "atr14"]]

    # Initial training
    student.train(features_df[:80], labels[:80])

    # Incremental update with new data
    new_features = features_df[80:]
    new_labels = labels[80:]
    result = student.train(new_features, new_labels)

    assert student.model is not None
    assert "metrics" in result
    assert "train_accuracy" in result["metrics"]


def test_validate_features(teacher_artifacts: tuple[Path, Path]) -> None:
    """Tests feature compatibility validation."""
    metadata_path, labels_path = teacher_artifacts

    config = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
    )

    student = StudentModel(config)
    labels_df, labels = student.load_teacher_artifacts()
    features_df = labels_df[["sma20", "rsi14", "atr14"]]
    student.train(features_df, labels)

    # Valid features
    valid_df = pd.DataFrame(
        {
            "sma20": [110.0],
            "rsi14": [55.0],
            "atr14": [2.5],
        }
    )
    assert student.validate_features(valid_df) is True

    # Missing feature
    invalid_df = pd.DataFrame(
        {
            "sma20": [110.0],
            "rsi14": [55.0],
        }
    )
    assert student.validate_features(invalid_df) is False

    # Extra feature is okay
    extra_df = pd.DataFrame(
        {
            "sma20": [110.0],
            "rsi14": [55.0],
            "atr14": [2.5],
            "extra": [99.0],
        }
    )
    assert student.validate_features(extra_df) is True


def test_schema_mismatch(teacher_artifacts: tuple[Path, Path]) -> None:
    """Tests error handling for schema mismatches."""
    metadata_path, labels_path = teacher_artifacts

    config = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
    )

    student = StudentModel(config)
    labels_df, labels = student.load_teacher_artifacts()
    features_df = labels_df[["sma20", "rsi14", "atr14"]]
    student.train(features_df, labels)

    # Try to predict with wrong features
    wrong_features = pd.DataFrame(
        {
            "wrong_col1": [110.0],
            "wrong_col2": [55.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required features"):
        student.predict(wrong_features)


def test_decision_threshold_custom(teacher_artifacts: tuple[Path, Path]) -> None:
    """Tests custom decision threshold."""
    metadata_path, labels_path = teacher_artifacts

    # High threshold (0.8) should require higher confidence
    config_high = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
        decision_threshold=0.8,
        random_seed=42,
    )

    student_high = StudentModel(config_high)
    labels_df, labels = student_high.load_teacher_artifacts()
    features_df = labels_df[["sma20", "rsi14", "atr14"]]
    student_high.train(features_df, labels)

    # Low threshold (0.2) should be more permissive
    config_low = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
        decision_threshold=0.2,
        random_seed=42,
    )

    student_low = StudentModel(config_low)
    student_low.load_teacher_artifacts()
    student_low.train(features_df, labels)

    # Same features should give different decisions
    test_features = {"sma20": 110.0, "rsi14": 55.0, "atr14": 2.5}
    result_high = student_high.predict_single(test_features, symbol="TEST")
    result_low = student_low.predict_single(test_features, symbol="TEST")

    # Probabilities should be the same
    assert result_high.probability == result_low.probability

    # Confidence calculation should differ based on threshold
    assert abs(result_high.confidence - abs(result_high.probability - 0.8)) < 0.01
    assert abs(result_low.confidence - abs(result_low.probability - 0.2)) < 0.01


def test_model_not_trained_error(teacher_artifacts: tuple[Path, Path]) -> None:
    """Tests error when trying to predict without training."""
    metadata_path, labels_path = teacher_artifacts

    config = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
    )

    student = StudentModel(config)

    # Try to predict without training
    test_features = {"sma20": 110.0, "rsi14": 55.0, "atr14": 2.5}

    with pytest.raises(ValueError, match="Model not trained"):
        student.predict_single(test_features, symbol="TEST")


def test_reproducibility_with_seed(teacher_artifacts: tuple[Path, Path]) -> None:
    """Tests reproducibility with random seed."""
    metadata_path, labels_path = teacher_artifacts

    config1 = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
        random_seed=42,
    )

    config2 = StudentConfig(
        teacher_metadata_path=str(metadata_path),
        teacher_labels_path=str(labels_path),
        random_seed=42,
    )

    # Train two models with same seed
    student1 = StudentModel(config1)
    labels_df, labels = student1.load_teacher_artifacts()
    features_df = labels_df[["sma20", "rsi14", "atr14"]]
    student1.train(features_df, labels)

    student2 = StudentModel(config2)
    student2.load_teacher_artifacts()
    student2.train(features_df, labels)

    # Predictions should be identical
    test_features = {"sma20": 110.0, "rsi14": 55.0, "atr14": 2.5}
    result1 = student1.predict_single(test_features, symbol="TEST")
    result2 = student2.predict_single(test_features, symbol="TEST")

    assert result1.probability == result2.probability
    assert result1.decision == result2.decision
