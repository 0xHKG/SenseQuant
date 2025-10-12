"""Integration tests for Teacher-Student pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

from src.domain.types import Bar, StudentConfig, TrainingConfig
from src.services.teacher_student import StudentModel, TeacherLabeler


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_breeze_client() -> MagicMock:
    """Create mock Breeze client."""
    client = MagicMock()
    client.authenticate = MagicMock()
    client.historical_bars = MagicMock()
    return client


@pytest.fixture
def sample_bars() -> list[Bar]:
    """Generate realistic sample bars."""
    ist = pytz.timezone("Asia/Kolkata")
    base_date = pd.Timestamp("2025-01-01", tz=ist)
    bars = []

    for i in range(100):
        ts = base_date + pd.Timedelta(days=i)
        # Create oscillating price pattern with trend
        base_price = 100.0
        trend = i * 0.1
        cycle = 5 * ((i % 20) - 10) / 10  # Oscillation
        noise = ((i * 7) % 13 - 6) * 0.3
        price = base_price + trend + cycle + noise

        bars.append(
            Bar(
                ts=ts,
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=10000 + (i % 10) * 1000,
            )
        )

    return bars


def test_full_teacher_student_pipeline(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    temp_dir: Path,
) -> None:
    """End-to-end Teacher → Student pipeline."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    with patch("src.services.teacher_student.BreezeClient", return_value=mock_breeze_client):
        # Step 1: Train Teacher
        teacher_config = TrainingConfig(
            symbol="TEST",
            start_date="2025-01-01",
            end_date="2025-04-10",
            label_window_days=5,
            label_threshold_pct=0.02,
            train_split=0.8,
            random_seed=42,
        )

        teacher = TeacherLabeler(teacher_config, client=mock_breeze_client)
        teacher_result = teacher.run_full_pipeline()

        # Verify Teacher artifacts created
        assert Path(teacher_result.model_path).exists()
        assert Path(teacher_result.labels_path).exists()
        assert Path(teacher_result.metadata_path).exists()
        assert Path(teacher_result.importance_path).exists()

        # Step 2: Train Student from Teacher artifacts
        student_config = StudentConfig(
            teacher_metadata_path=teacher_result.metadata_path,
            teacher_labels_path=teacher_result.labels_path,
            decision_threshold=0.5,
            random_seed=42,
        )

        student = StudentModel(student_config)
        labels_df, labels = student.load_teacher_artifacts()

        # Get feature columns from Teacher metadata
        features_df = labels_df[student.teacher_metadata["features"]]
        student_result = student.train(features_df, labels)

        # Verify Student trained successfully
        assert student.model is not None
        assert "metrics" in student_result
        assert "train_accuracy" in student_result["metrics"]

        # Step 3: Save Student artifacts
        student_model_path = temp_dir / "student_model.pkl"
        student_metadata_path = temp_dir / "student_metadata.json"
        student.save(str(student_model_path), str(student_metadata_path))

        assert student_model_path.exists()
        assert student_metadata_path.exists()

        # Step 4: Load Student and make predictions
        new_student = StudentModel(student_config)
        new_student.load(str(student_model_path), str(student_metadata_path))

        # Make prediction with current bar features
        test_features = {
            col: features_df[col].iloc[-1] for col in student.teacher_metadata["features"]
        }
        result = new_student.predict_single(test_features, symbol="TEST")

        assert result.symbol == "TEST"
        assert 0.0 <= result.probability <= 1.0
        assert result.decision in [0, 1]


def test_student_predictions_accuracy(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    temp_dir: Path,
) -> None:
    """Validates prediction quality."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    with patch("src.services.teacher_student.BreezeClient", return_value=mock_breeze_client):
        # Train Teacher
        teacher_config = TrainingConfig(
            symbol="TEST",
            start_date="2025-01-01",
            end_date="2025-04-10",
            label_window_days=5,
            label_threshold_pct=0.02,
            train_split=0.8,
            random_seed=42,
        )

        teacher = TeacherLabeler(teacher_config, client=mock_breeze_client)
        teacher_result = teacher.run_full_pipeline()

        # Train Student
        student_config = StudentConfig(
            teacher_metadata_path=teacher_result.metadata_path,
            teacher_labels_path=teacher_result.labels_path,
            decision_threshold=0.5,
            random_seed=42,
        )

        student = StudentModel(student_config)
        labels_df, labels = student.load_teacher_artifacts()
        features_df = labels_df[student.teacher_metadata["features"]]
        student_result = student.train(features_df, labels)

        # Student should achieve reasonable accuracy
        assert student_result["metrics"]["train_accuracy"] > 0.5  # Better than random

        # Make batch predictions
        test_batch = features_df.tail(10)
        results = student.predict(test_batch)

        # Verify all predictions valid
        assert len(results) == 10
        for result in results:
            assert 0.0 <= result.probability <= 1.0
            assert result.decision in [0, 1]
            assert result.confidence >= 0.0


def test_student_artifact_completeness(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    temp_dir: Path,
) -> None:
    """Ensures all files created with correct schema."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    with patch("src.services.teacher_student.BreezeClient", return_value=mock_breeze_client):
        # Train Teacher
        teacher_config = TrainingConfig(
            symbol="TEST",
            start_date="2025-01-01",
            end_date="2025-04-10",
            random_seed=42,
        )

        teacher = TeacherLabeler(teacher_config, client=mock_breeze_client)
        teacher_result = teacher.run_full_pipeline()

        # Train and save Student
        student_config = StudentConfig(
            teacher_metadata_path=teacher_result.metadata_path,
            teacher_labels_path=teacher_result.labels_path,
            random_seed=42,
        )

        student = StudentModel(student_config)
        labels_df, labels = student.load_teacher_artifacts()
        features_df = labels_df[student.teacher_metadata["features"]]
        student.train(features_df, labels)

        student_model_path = temp_dir / "student_model.pkl"
        student_metadata_path = temp_dir / "student_metadata.json"
        student.save(str(student_model_path), str(student_metadata_path))

        # Verify all artifacts exist
        assert student_model_path.exists()
        assert student_metadata_path.exists()

        # Load and verify metadata schema
        import json

        with open(student_metadata_path) as f:
            metadata = json.load(f)

        # Required fields
        assert "symbol" in metadata
        assert "training_date" in metadata
        assert "features" in metadata
        assert "decision_threshold" in metadata
        assert "model_type" in metadata
        assert "teacher_reference" in metadata
        assert "metrics" in metadata

        # Features should match Teacher
        assert metadata["features"] == student.teacher_metadata["features"]

        # Teacher reference should point to original artifacts
        assert metadata["teacher_reference"]["metadata_path"] == teacher_result.metadata_path
        assert metadata["teacher_reference"]["labels_path"] == teacher_result.labels_path


def test_walk_forward_retraining(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    temp_dir: Path,
) -> None:
    """Tests incremental learning."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    with patch("src.services.teacher_student.BreezeClient", return_value=mock_breeze_client):
        # Train Teacher on initial period
        teacher_config = TrainingConfig(
            symbol="TEST",
            start_date="2025-01-01",
            end_date="2025-04-10",
            random_seed=42,
        )

        teacher = TeacherLabeler(teacher_config, client=mock_breeze_client)
        teacher_result = teacher.run_full_pipeline()

        # Train initial Student
        student_config = StudentConfig(
            teacher_metadata_path=teacher_result.metadata_path,
            teacher_labels_path=teacher_result.labels_path,
            incremental=True,
            random_seed=42,
        )

        student = StudentModel(student_config)
        labels_df, labels = student.load_teacher_artifacts()
        features_df = labels_df[student.teacher_metadata["features"]]

        # Initial training on 80% of data
        split_idx = int(len(features_df) * 0.8)
        initial_result = student.train(features_df[:split_idx], labels[:split_idx])

        assert student.model is not None
        assert "metrics" in initial_result
        assert "train_accuracy" in initial_result["metrics"]

        # Incremental update with remaining 20%
        incremental_result = student.train(features_df[split_idx:], labels[split_idx:])

        assert "metrics" in incremental_result
        assert "train_accuracy" in incremental_result["metrics"]

        # Model should still make valid predictions
        test_features = {
            col: features_df[col].iloc[-1] for col in student.teacher_metadata["features"]
        }
        result = student.predict_single(test_features, symbol="TEST")

        assert 0.0 <= result.probability <= 1.0
        assert result.decision in [0, 1]


def test_student_feature_compatibility_check(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    temp_dir: Path,
) -> None:
    """Tests feature validation between Teacher and Student."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    with patch("src.services.teacher_student.BreezeClient", return_value=mock_breeze_client):
        # Train Teacher
        teacher_config = TrainingConfig(
            symbol="TEST",
            start_date="2025-01-01",
            end_date="2025-04-10",
            random_seed=42,
        )

        teacher = TeacherLabeler(teacher_config, client=mock_breeze_client)
        teacher_result = teacher.run_full_pipeline()

        # Train Student
        student_config = StudentConfig(
            teacher_metadata_path=teacher_result.metadata_path,
            teacher_labels_path=teacher_result.labels_path,
        )

        student = StudentModel(student_config)
        labels_df, labels = student.load_teacher_artifacts()
        features_df = labels_df[student.teacher_metadata["features"]]
        student.train(features_df, labels)

        # Valid features
        valid_batch = features_df.tail(5)
        assert student.validate_features(valid_batch) is True

        # Missing feature
        invalid_batch = features_df.tail(5).drop(columns=[features_df.columns[0]])
        assert student.validate_features(invalid_batch) is False

        # Extra features are okay
        extra_batch = features_df.tail(5).copy()
        extra_batch["extra_feature"] = 99.0
        assert student.validate_features(extra_batch) is True


def test_teacher_student_reproducibility(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    temp_dir: Path,
) -> None:
    """Tests reproducibility of full pipeline with same seed."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    with patch("src.services.teacher_student.BreezeClient", return_value=mock_breeze_client):
        # Run pipeline twice with same seed
        results = []
        for run in range(2):
            run_dir = temp_dir / f"run_{run}"
            run_dir.mkdir()

            # Train Teacher
            teacher_config = TrainingConfig(
                symbol="TEST",
                start_date="2025-01-01",
                end_date="2025-04-10",
                random_seed=42,  # Same seed
            )

            teacher = TeacherLabeler(teacher_config, client=mock_breeze_client)
            teacher_result = teacher.run_full_pipeline()

            # Train Student
            student_config = StudentConfig(
                teacher_metadata_path=teacher_result.metadata_path,
                teacher_labels_path=teacher_result.labels_path,
                random_seed=42,  # Same seed
            )

            student = StudentModel(student_config)
            labels_df, labels = student.load_teacher_artifacts()
            features_df = labels_df[student.teacher_metadata["features"]]
            student.train(features_df, labels)

            # Make prediction
            test_features = {
                col: features_df[col].iloc[-1] for col in student.teacher_metadata["features"]
            }
            result = student.predict_single(test_features, symbol="TEST")
            results.append(result)

        # Both runs should produce identical predictions
        assert results[0].probability == results[1].probability
        assert results[0].decision == results[1].decision
        assert results[0].confidence == results[1].confidence
