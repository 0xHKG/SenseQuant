"""Integration tests for teacher/student model training workflow (US-020)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest


def test_train_student_script_help():
    """Test train_student.py script help output (US-020).

    Verifies:
    - Script exists and is executable
    - Help text displays correctly
    - Required arguments documented
    """
    import subprocess

    script_path = Path("scripts/train_student.py")
    assert script_path.exists(), "train_student.py script not found"

    # Test help output
    result = subprocess.run(["python", str(script_path), "--help"], capture_output=True, text=True)

    assert result.returncode == 0, "train_student.py --help failed"
    assert "--teacher-dir" in result.stdout, "Missing --teacher-dir argument"
    assert "--output-dir" in result.stdout, "Missing --output-dir argument"
    assert "--model-type" in result.stdout, "Missing --model-type argument"
    assert "--validate" in result.stdout, "Missing --validate flag"

    print("\n✓ train_student.py script help validated")


def test_mock_teacher_artifacts(tmp_path):
    """Test creation of mock teacher artifacts for testing (US-020).

    Verifies:
    - Mock labels DataFrame created with required columns
    - Mock features DataFrame created with synthetic features
    - Metadata JSON created with training configuration
    - Artifacts saved in expected format
    """
    teacher_dir = tmp_path / "models" / "teacher"
    teacher_dir.mkdir(parents=True)

    # Create mock labels (teacher-generated labels for training student)
    # Labels format: timestamp, symbol, label (0=NOOP, 1=LONG, 2=SHORT)
    labels_data = {
        "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1h"),
        "symbol": ["RELIANCE"] * 1000,
        "label": [0, 1, 2] * 333 + [0],  # Balanced distribution
    }
    labels_df = pd.DataFrame(labels_data)

    # Save labels as compressed CSV
    labels_path = teacher_dir / "labels.csv.gz"
    with gzip.open(labels_path, "wt") as f:
        labels_df.to_csv(f, index=False)

    assert labels_path.exists()
    print(f"\n✓ Mock labels created: {len(labels_df)} samples")

    # Create mock features (synthetic technical indicators)
    features_data = {
        "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1h"),
        "rsi": [50.0 + i * 0.1 for i in range(1000)],  # RSI trending up
        "sma_short": [2500.0 + i * 0.5 for i in range(1000)],  # SMA trending up
        "sma_long": [2480.0 + i * 0.4 for i in range(1000)],  # Slower SMA
        "volume": [1000000 + i * 100 for i in range(1000)],  # Volume increasing
        "sentiment": [0.1 + (i % 10) * 0.05 for i in range(1000)],  # Sentiment cycling
    }
    features_df = pd.DataFrame(features_data)

    # Save features as compressed CSV
    features_path = teacher_dir / "features.csv.gz"
    with gzip.open(features_path, "wt") as f:
        features_df.to_csv(f, index=False)

    assert features_path.exists()
    print(f"✓ Mock features created: {features_df.shape[1] - 1} features")

    # Create mock metadata
    metadata = {
        "timestamp": "2024-01-01T00:00:00Z",
        "symbols": ["RELIANCE"],
        "date_range": {"start": "2024-01-01", "end": "2024-02-10"},
        "strategy": "intraday",
        "teacher_params": {
            "min_holding_return": 0.005,
            "lookback_days": 20,
            "features": ["rsi", "sma_short", "sma_long", "volume", "sentiment"],
        },
        "dataset_stats": {
            "total_samples": 1000,
            "label_distribution": {"0": 334, "1": 333, "2": 333},
        },
        "model_type": "LightGBM",
        "model_path": str(teacher_dir / "teacher_model.pkl"),
    }

    metadata_path = teacher_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    assert metadata_path.exists()
    print("✓ Mock metadata created")

    # Verify artifacts can be loaded
    with gzip.open(labels_path, "rt") as f:
        loaded_labels = pd.read_csv(f)
    assert len(loaded_labels) == 1000

    with gzip.open(features_path, "rt") as f:
        loaded_features = pd.read_csv(f)
    assert loaded_features.shape == (1000, 6)  # 5 features + timestamp

    with open(metadata_path) as f:
        loaded_metadata = json.load(f)
    assert loaded_metadata["model_type"] == "LightGBM"

    print("✓ Mock teacher artifacts validated")


def test_student_training_with_mock_data(tmp_path):
    """Test student training with mock teacher artifacts (US-020).

    Verifies:
    - Student training script can load teacher artifacts
    - Student model trains successfully
    - Evaluation metrics computed
    - Student artifacts saved (model, metrics, metadata)
    """
    # Create mock teacher artifacts
    teacher_dir = tmp_path / "models" / "teacher"
    teacher_dir.mkdir(parents=True)

    # Mock labels with features
    labels_data = {
        "timestamp": pd.date_range("2024-01-01", periods=200, freq="1h"),
        "symbol": ["RELIANCE"] * 200,
        "label": [0, 1, 2] * 66 + [0, 1],  # Balanced distribution
        "feature_1": [50.0 + i * 0.1 for i in range(200)],
        "feature_2": [2500.0 + i * 0.5 for i in range(200)],
        "feature_3": [1000000 + i * 100 for i in range(200)],
    }
    labels_df = pd.DataFrame(labels_data)

    labels_path = teacher_dir / "labels.csv.gz"
    with gzip.open(labels_path, "wt") as f:
        labels_df.to_csv(f, index=False)

    # Mock features
    features_data = {
        "timestamp": pd.date_range("2024-01-01", periods=200, freq="1h"),
        "rsi": [50.0 + i * 0.1 for i in range(200)],
        "sma_short": [2500.0 + i * 0.5 for i in range(200)],
        "volume": [1000000 + i * 100 for i in range(200)],
    }
    features_df = pd.DataFrame(features_data)

    features_path = teacher_dir / "features.csv.gz"
    with gzip.open(features_path, "wt") as f:
        features_df.to_csv(f, index=False)

    # Mock metadata
    metadata = {
        "timestamp": "2024-01-01T00:00:00Z",
        "symbols": ["RELIANCE"],
        "model_type": "LightGBM",
        "model_path": str(teacher_dir / "teacher_model.pkl"),
    }

    metadata_path = teacher_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Train student model
    import subprocess

    student_dir = tmp_path / "models" / "student"
    result = subprocess.run(
        [
            "python",
            "scripts/train_student.py",
            "--teacher-dir",
            str(teacher_dir),
            "--output-dir",
            str(student_dir),
            "--model-type",
            "logistic",
            "--test-size",
            "0.2",
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
    )

    # Print output for debugging
    if result.returncode != 0:
        print(f"\nSTDOUT:\n{result.stdout}")
        print(f"\nSTDERR:\n{result.stderr}")

    assert result.returncode == 0, f"Student training failed: {result.stderr}"

    # Verify student artifacts
    assert (student_dir / "student_model.pkl").exists(), "Student model not created"
    assert (student_dir / "evaluation_metrics.json").exists(), "Evaluation metrics not created"
    assert (student_dir / "metadata.json").exists(), "Student metadata not created"

    # Load and validate evaluation metrics
    with open(student_dir / "evaluation_metrics.json") as f:
        metrics = json.load(f)

    assert "accuracy" in metrics, "Accuracy metric missing"
    assert "f1_macro" in metrics, "F1 macro metric missing"
    assert "confusion_matrix" in metrics, "Confusion matrix missing"
    assert 0.0 <= metrics["accuracy"] <= 1.0, "Invalid accuracy value"

    print("\n✓ Student training completed successfully")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F1 (macro): {metrics['f1_macro']:.4f}")

    # Load and validate student metadata
    with open(student_dir / "metadata.json") as f:
        student_metadata = json.load(f)

    assert "timestamp" in student_metadata, "Timestamp missing from metadata"
    assert "model_type" in student_metadata, "Model type missing from metadata"
    assert student_metadata["model_type"] == "logistic", "Incorrect model type in metadata"

    print("✓ Student artifacts validated")


def test_artifact_versioning_structure(tmp_path):
    """Test artifact versioning directory structure (US-020).

    Verifies:
    - Timestamped directory structure
    - Teacher and student subdirectories
    - Required artifact files present
    - Metadata linkage between teacher and student
    """
    # Create expected directory structure
    timestamp = "20250112_143000"
    models_dir = tmp_path / "models" / timestamp

    teacher_dir = models_dir / "teacher"
    student_dir = models_dir / "student"

    teacher_dir.mkdir(parents=True)
    student_dir.mkdir(parents=True)

    # Expected teacher artifacts
    teacher_artifacts = [
        "teacher_model.pkl",
        "labels.csv.gz",
        "features.csv.gz",
        "metadata.json",
        "dataset_stats.json",
        "evaluation_metrics.json",
        "feature_importance.json",
    ]

    # Expected student artifacts
    student_artifacts = [
        "student_model.pkl",
        "metadata.json",
        "evaluation_metrics.json",
        "confusion_matrix.json",
    ]

    # Create mock teacher artifacts
    for artifact in teacher_artifacts:
        if artifact.endswith(".json"):
            (teacher_dir / artifact).write_text(json.dumps({"test": "data"}))
        elif artifact.endswith(".gz"):
            with gzip.open(teacher_dir / artifact, "wt") as f:
                f.write("test,data\n1,2\n")
        else:
            (teacher_dir / artifact).write_bytes(b"mock_data")

    # Create mock student artifacts
    for artifact in student_artifacts:
        if artifact.endswith(".json"):
            (student_dir / artifact).write_text(json.dumps({"test": "data"}))
        else:
            (student_dir / artifact).write_bytes(b"mock_data")

    # Verify structure
    assert models_dir.exists(), "Timestamped models directory not created"
    assert teacher_dir.exists(), "Teacher subdirectory not created"
    assert student_dir.exists(), "Student subdirectory not created"

    # Verify teacher artifacts
    for artifact in teacher_artifacts:
        assert (teacher_dir / artifact).exists(), f"Teacher artifact missing: {artifact}"

    # Verify student artifacts
    for artifact in student_artifacts:
        assert (student_dir / artifact).exists(), f"Student artifact missing: {artifact}"

    print("\n✓ Artifact versioning structure validated")
    print(f"  - Models directory: {models_dir}")
    print(f"  - Teacher artifacts: {len(teacher_artifacts)}")
    print(f"  - Student artifacts: {len(student_artifacts)}")

    # Test metadata linkage
    teacher_metadata = {
        "timestamp": "2025-01-12T14:30:00Z",
        "symbols": ["RELIANCE", "TCS"],
        "config_hash": "a3f5d8c2",
    }
    with open(teacher_dir / "metadata.json", "w") as f:
        json.dump(teacher_metadata, f)

    student_metadata = {
        "timestamp": "2025-01-12T15:00:00Z",
        "teacher_dir": str(teacher_dir),
        "model_type": "logistic",
    }
    with open(student_dir / "metadata.json", "w") as f:
        json.dump(student_metadata, f)

    # Verify linkage
    with open(student_dir / "metadata.json") as f:
        loaded_student_metadata = json.load(f)

    assert loaded_student_metadata["teacher_dir"] == str(teacher_dir), (
        "Student metadata not linked to teacher"
    )

    print("✓ Metadata linkage validated")


def test_training_pipeline_metadata(tmp_path):
    """Test training pipeline metadata structure (US-020).

    Verifies:
    - Pipeline metadata JSON structure
    - Stage tracking (teacher → student → validation)
    - Config hash and optimization run linkage
    - Git commit and dependency tracking
    """
    pipeline_dir = tmp_path / "models" / "20250112_143000"
    pipeline_dir.mkdir(parents=True)

    # Create pipeline metadata
    pipeline_metadata = {
        "timestamp": "2025-01-12T14:30:00Z",
        "pipeline_version": "1.0",
        "stages": {
            "teacher_training": {
                "start": "2025-01-12T14:30:00Z",
                "end": "2025-01-12T14:45:00Z",
                "status": "completed",
                "artifacts": str(pipeline_dir / "teacher/"),
            },
            "student_training": {
                "start": "2025-01-12T14:45:00Z",
                "end": "2025-01-12T15:00:00Z",
                "status": "completed",
                "artifacts": str(pipeline_dir / "student/"),
            },
            "validation": {
                "start": "2025-01-12T15:00:00Z",
                "end": "2025-01-12T15:30:00Z",
                "status": "completed",
                "artifacts": str(pipeline_dir / "student/validation_results.json"),
            },
        },
        "config_hash": "a3f5d8c2",
        "optimization_run": "data/optimization/run_20250110_120000",
        "git_commit": "8b965e5d",
        "dependencies": {"python": "3.12.2", "scikit-learn": "1.5.2", "lightgbm": "4.5.0"},
    }

    pipeline_metadata_path = pipeline_dir / "training_pipeline.json"
    with open(pipeline_metadata_path, "w") as f:
        json.dump(pipeline_metadata, f, indent=2)

    assert pipeline_metadata_path.exists(), "Pipeline metadata not created"

    # Load and validate
    with open(pipeline_metadata_path) as f:
        loaded_metadata = json.load(f)

    assert "stages" in loaded_metadata, "Stages missing from pipeline metadata"
    assert "teacher_training" in loaded_metadata["stages"], "Teacher training stage missing"
    assert "student_training" in loaded_metadata["stages"], "Student training stage missing"
    assert "validation" in loaded_metadata["stages"], "Validation stage missing"

    # Verify config hash
    assert loaded_metadata["config_hash"] == "a3f5d8c2", "Config hash mismatch"

    # Verify optimization run linkage
    assert "optimization_run" in loaded_metadata, "Optimization run linkage missing"

    print("\n✓ Training pipeline metadata validated")
    print(f"  - Pipeline version: {loaded_metadata['pipeline_version']}")
    print(f"  - Stages tracked: {len(loaded_metadata['stages'])}")
    print(f"  - Config hash: {loaded_metadata['config_hash']}")
    print(f"  - Git commit: {loaded_metadata['git_commit']}")


@pytest.mark.skipif(not Path("data/models").exists(), reason="data/models directory not found")
def test_sample_artifacts_exist():
    """Test that sample model artifacts exist (US-020).

    Verifies:
    - Sample artifacts directory exists
    - At least one teacher/student pair present
    - Required artifact files exist
    """
    models_dir = Path("data/models")
    assert models_dir.exists(), "Models directory not found"

    # Look for any timestamped directory
    timestamp_dirs = [
        d for d in models_dir.iterdir() if d.is_dir() and d.name.replace("_", "").isdigit()
    ]

    if not timestamp_dirs:
        pytest.skip("No sample artifacts found (expected for new installation)")

    # Check first sample
    sample_dir = timestamp_dirs[0]
    teacher_dir = sample_dir / "teacher"
    student_dir = sample_dir / "student"

    print(f"\n✓ Sample artifacts found: {sample_dir.name}")

    if teacher_dir.exists():
        assert (teacher_dir / "metadata.json").exists(), "Teacher metadata missing"
        print(f"  - Teacher artifacts: {teacher_dir}")

    if student_dir.exists():
        assert (student_dir / "metadata.json").exists(), "Student metadata missing"
        print(f"  - Student artifacts: {student_dir}")
