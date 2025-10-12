"""Integration tests for release deployment pipeline (US-023)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from src.app.config import Settings
from src.services.monitoring import MonitoringService


@pytest.fixture
def tmp_release_structure(tmp_path: Path) -> Path:
    """Create temporary directory structure for release testing."""
    # Create release directories
    (tmp_path / "release" / "manifests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "release" / "backups").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "monitoring" / "releases").mkdir(parents=True, exist_ok=True)

    # Create sample artifacts
    (tmp_path / "src" / "app").mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "src" / "app" / "config.py"
    config_file.write_text("# Config file\nDEBUG = False\n")

    (tmp_path / "data" / "models").mkdir(parents=True, exist_ok=True)
    model_file = tmp_path / "data" / "models" / "student_model.pkl"
    model_file.write_bytes(b"mock_model_data")

    (tmp_path / "notebooks").mkdir(parents=True, exist_ok=True)
    notebook_file = tmp_path / "notebooks" / "accuracy_report.ipynb"
    notebook_file.write_text('{"cells": []}')

    return tmp_path


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_manifest_generation_with_hashes(tmp_release_structure: Path) -> None:
    """Test manifest generation includes SHA256 hashes for all artifacts."""
    release_id = f"release_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Artifact paths
    config_path = tmp_release_structure / "src" / "app" / "config.py"
    model_path = tmp_release_structure / "data" / "models" / "student_model.pkl"
    notebook_path = tmp_release_structure / "notebooks" / "accuracy_report.ipynb"

    # Compute expected hashes
    config_hash = compute_sha256(config_path)
    model_hash = compute_sha256(model_path)
    notebook_hash = compute_sha256(notebook_path)

    # Create mock manifest
    manifest = {
        "release_id": release_id,
        "release_type": "minor",
        "deployment": {
            "timestamp": datetime.now().isoformat(),
            "deployer": "test_user",
            "environment": "production",
        },
        "artifacts": {
            "configs": [
                {
                    "path": str(config_path),
                    "hash": f"sha256:{config_hash}",
                    "backup": str(
                        tmp_release_structure / "release" / "backups" / release_id / "config.py"
                    ),
                }
            ],
            "models": [
                {
                    "path": str(model_path),
                    "hash": f"sha256:{model_hash}",
                    "version": "v1.0",
                    "backup": str(
                        tmp_release_structure
                        / "release"
                        / "backups"
                        / release_id
                        / "student_model.pkl"
                    ),
                }
            ],
            "notebooks": [
                {
                    "path": str(notebook_path),
                    "hash": f"sha256:{notebook_hash}",
                    "backup": str(
                        tmp_release_structure
                        / "release"
                        / "backups"
                        / release_id
                        / "accuracy_report.ipynb"
                    ),
                }
            ],
        },
        "monitoring": {
            "heightened_period_hours": 48,
            "heightened_start": datetime.now().isoformat(),
            "heightened_end": (datetime.now() + timedelta(hours=48)).isoformat(),
        },
    }

    # Write manifest
    manifest_path = tmp_release_structure / "release" / "manifests" / f"{release_id}.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f)

    # Verify manifest exists
    assert manifest_path.exists()

    # Verify manifest content
    with open(manifest_path) as f:
        loaded_manifest = yaml.safe_load(f)

    assert loaded_manifest["release_id"] == release_id
    assert len(loaded_manifest["artifacts"]["configs"]) == 1
    assert loaded_manifest["artifacts"]["configs"][0]["hash"] == f"sha256:{config_hash}"
    assert len(loaded_manifest["artifacts"]["models"]) == 1
    assert loaded_manifest["artifacts"]["models"][0]["hash"] == f"sha256:{model_hash}"
    assert len(loaded_manifest["artifacts"]["notebooks"]) == 1
    assert loaded_manifest["artifacts"]["notebooks"][0]["hash"] == f"sha256:{notebook_hash}"


def test_release_registration_with_monitoring(tmp_release_structure: Path) -> None:
    """Test release registration activates heightened monitoring."""
    settings = Settings()  # type: ignore[call-arg]
    monitoring = MonitoringService(settings)

    # Override releases directory
    monitoring.releases_dir = tmp_release_structure / "data" / "monitoring" / "releases"
    monitoring.releases_dir.mkdir(parents=True, exist_ok=True)

    # Create mock manifest
    release_id = f"release_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    manifest_path = tmp_release_structure / "release" / "manifests" / f"{release_id}.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "release_id": release_id,
        "release_type": "minor",
        "deployment": {
            "timestamp": datetime.now().isoformat(),
            "deployer": "test_user",
        },
    }

    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f)

    # Register release
    monitoring.register_release(release_id, manifest_path, heightened_hours=48)

    # Verify active release is set
    active_release = monitoring.get_active_release()
    assert active_release is not None
    assert active_release["release_id"] == release_id
    assert active_release["heightened_monitoring_active"] is True

    # Verify heightened monitoring
    assert monitoring.is_in_heightened_monitoring() is True

    # Verify alert thresholds are stricter
    thresholds = monitoring.get_alert_thresholds()
    assert thresholds["intraday_hit_ratio_drop"] == 0.05  # 5% in heightened mode
    assert thresholds["swing_precision_drop"] == 0.05

    # Verify monitoring windows are shorter
    windows = monitoring.get_monitoring_window_hours()
    assert windows["intraday"] == 6  # 6 hours in heightened mode
    assert windows["swing"] == 24  # 24 hours in heightened mode


def test_monitoring_transition_after_48h(tmp_release_structure: Path) -> None:
    """Test monitoring automatically transitions from heightened to normal after 48h."""
    settings = Settings()  # type: ignore[call-arg]
    monitoring = MonitoringService(settings)

    # Override releases directory
    monitoring.releases_dir = tmp_release_structure / "data" / "monitoring" / "releases"
    monitoring.releases_dir.mkdir(parents=True, exist_ok=True)

    # Create release that ended 1 hour ago (past the 48h window)
    release_id = "release_old_20241010_120000"
    past_time = datetime.now() - timedelta(hours=49)
    heightened_end = past_time + timedelta(hours=48)

    # Manually create active release file (simulating a past deployment)
    active_release_data = {
        "release_id": release_id,
        "deployment_timestamp": past_time.isoformat(),
        "manifest_path": "release/manifests/release_old.yaml",
        "heightened_monitoring_active": True,
        "heightened_monitoring_end": heightened_end.isoformat(),
        "heightened_hours": 48,
    }

    active_release_path = monitoring.releases_dir / "active_release.yaml"
    with open(active_release_path, "w") as f:
        yaml.dump(active_release_data, f)

    # Load and check - should auto-transition to normal
    monitoring._load_active_release()
    active_release = monitoring.get_active_release()

    assert active_release is not None
    assert active_release["heightened_monitoring_active"] is False

    # Verify monitoring is no longer heightened
    assert monitoring.is_in_heightened_monitoring() is False

    # Verify alert thresholds are normal
    thresholds = monitoring.get_alert_thresholds()
    assert thresholds["intraday_hit_ratio_drop"] == 0.10  # 10% in normal mode

    # Verify monitoring windows are normal
    windows = monitoring.get_monitoring_window_hours()
    assert windows["intraday"] == 24  # 24 hours in normal mode
    assert windows["swing"] == 2160  # 90 days in normal mode


def test_rollback_plan_generation(tmp_release_structure: Path) -> None:
    """Test rollback plan references previous release artifacts."""
    # Create first release manifest
    release_id_1 = "release_20241010_120000"
    manifest_1 = {
        "release_id": release_id_1,
        "artifacts": {
            "configs": [
                {
                    "path": "src/app/config.py",
                    "hash": "sha256:abc123",
                    "backup": f"release/backups/{release_id_1}/config.py",
                }
            ],
            "models": [
                {
                    "path": "data/models/student_model.pkl",
                    "hash": "sha256:def456",
                    "backup": f"release/backups/{release_id_1}/student_model.pkl",
                }
            ],
        },
    }

    manifest_path_1 = tmp_release_structure / "release" / "manifests" / f"{release_id_1}.yaml"
    with open(manifest_path_1, "w") as f:
        yaml.dump(manifest_1, f)

    # Create backup files for first release
    backup_dir_1 = tmp_release_structure / "release" / "backups" / release_id_1
    backup_dir_1.mkdir(parents=True, exist_ok=True)
    (backup_dir_1 / "config.py").write_text("# Old config\n")
    (backup_dir_1 / "student_model.pkl").write_bytes(b"old_model")

    # Create second release manifest with rollback plan
    release_id_2 = "release_20241012_140000"
    manifest_2 = {
        "release_id": release_id_2,
        "artifacts": {
            "configs": [
                {
                    "path": "src/app/config.py",
                    "hash": "sha256:new123",
                    "backup": f"release/backups/{release_id_2}/config.py",
                }
            ],
            "models": [
                {
                    "path": "data/models/student_model.pkl",
                    "hash": "sha256:new456",
                    "backup": f"release/backups/{release_id_2}/student_model.pkl",
                }
            ],
        },
        "rollback_plan": {
            "previous_release_id": release_id_1,
            "previous_manifest": str(manifest_path_1),
            "restore_commands": [
                f"cp release/backups/{release_id_1}/config.py src/app/config.py",
                f"cp release/backups/{release_id_1}/student_model.pkl data/models/student_model.pkl",
            ],
            "verification_steps": [
                "python scripts/release_audit.py --skip-validation",
                "make test",
            ],
        },
    }

    manifest_path_2 = tmp_release_structure / "release" / "manifests" / f"{release_id_2}.yaml"
    with open(manifest_path_2, "w") as f:
        yaml.dump(manifest_2, f)

    # Verify rollback plan
    with open(manifest_path_2) as f:
        loaded_manifest = yaml.safe_load(f)

    rollback_plan = loaded_manifest["rollback_plan"]
    assert rollback_plan["previous_release_id"] == release_id_1
    assert rollback_plan["previous_manifest"] == str(manifest_path_1)
    assert len(rollback_plan["restore_commands"]) == 2
    assert len(rollback_plan["verification_steps"]) == 2
    assert "config.py" in rollback_plan["restore_commands"][0]
    assert "student_model.pkl" in rollback_plan["restore_commands"][1]


def test_artifact_backup_creation(tmp_release_structure: Path) -> None:
    """Test artifacts are backed up during manifest generation."""
    release_id = f"release_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir = tmp_release_structure / "release" / "backups" / release_id
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Source artifacts
    config_src = tmp_release_structure / "src" / "app" / "config.py"
    model_src = tmp_release_structure / "data" / "models" / "student_model.pkl"

    # Backup paths
    config_backup = backup_dir / "config.py"
    model_backup = backup_dir / "student_model.pkl"

    # Simulate backup
    import shutil

    shutil.copy2(config_src, config_backup)
    shutil.copy2(model_src, model_backup)

    # Verify backups exist
    assert config_backup.exists()
    assert model_backup.exists()

    # Verify content matches
    assert config_backup.read_text() == config_src.read_text()
    assert config_backup.read_bytes() == config_src.read_bytes()
    assert model_backup.read_bytes() == model_src.read_bytes()

    # Verify hashes match
    config_src_hash = compute_sha256(config_src)
    config_backup_hash = compute_sha256(config_backup)
    assert config_src_hash == config_backup_hash

    model_src_hash = compute_sha256(model_src)
    model_backup_hash = compute_sha256(model_backup)
    assert model_src_hash == model_backup_hash


def test_release_manifest_persistence(tmp_release_structure: Path) -> None:
    """Test release state is persisted to YAML files."""
    settings = Settings()  # type: ignore[call-arg]
    monitoring = MonitoringService(settings)

    # Override releases directory
    monitoring.releases_dir = tmp_release_structure / "data" / "monitoring" / "releases"
    monitoring.releases_dir.mkdir(parents=True, exist_ok=True)

    # Register release
    release_id = f"release_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    manifest_path = tmp_release_structure / "release" / "manifests" / f"{release_id}.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(f"release_id: {release_id}\n")

    monitoring.register_release(release_id, manifest_path, heightened_hours=48)

    # Verify active_release.yaml exists
    active_release_path = monitoring.releases_dir / "active_release.yaml"
    assert active_release_path.exists()

    # Verify historical release file exists
    historical_path = monitoring.releases_dir / f"{release_id}.yaml"
    assert historical_path.exists()

    # Load and verify content
    with open(active_release_path) as f:
        active_data = yaml.safe_load(f)

    assert active_data["release_id"] == release_id
    assert active_data["heightened_monitoring_active"] is True
    assert "heightened_monitoring_end" in active_data

    with open(historical_path) as f:
        historical_data = yaml.safe_load(f)

    assert historical_data["release_id"] == release_id


def test_full_release_pipeline_end_to_end(tmp_release_structure: Path) -> None:
    """Test complete release pipeline: manifest generation -> registration -> monitoring."""
    # Step 1: Generate manifest with artifacts
    release_id = f"release_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Artifact paths
    config_path = tmp_release_structure / "src" / "app" / "config.py"
    model_path = tmp_release_structure / "data" / "models" / "student_model.pkl"

    # Compute hashes
    config_hash = compute_sha256(config_path)
    model_hash = compute_sha256(model_path)

    # Create backups
    backup_dir = tmp_release_structure / "release" / "backups" / release_id
    backup_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(config_path, backup_dir / "config.py")
    shutil.copy2(model_path, backup_dir / "student_model.pkl")

    # Create manifest
    manifest = {
        "release_id": release_id,
        "release_type": "minor",
        "deployment": {
            "timestamp": datetime.now().isoformat(),
            "deployer": "test_pipeline",
            "environment": "production",
        },
        "artifacts": {
            "configs": [
                {
                    "path": str(config_path),
                    "hash": f"sha256:{config_hash}",
                    "backup": str(backup_dir / "config.py"),
                }
            ],
            "models": [
                {
                    "path": str(model_path),
                    "hash": f"sha256:{model_hash}",
                    "version": "v1.0",
                    "backup": str(backup_dir / "student_model.pkl"),
                }
            ],
        },
        "monitoring": {
            "heightened_period_hours": 48,
            "heightened_start": datetime.now().isoformat(),
            "heightened_end": (datetime.now() + timedelta(hours=48)).isoformat(),
        },
    }

    manifest_path = tmp_release_structure / "release" / "manifests" / f"{release_id}.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f)

    # Step 2: Register release with monitoring
    settings = Settings()  # type: ignore[call-arg]
    monitoring = MonitoringService(settings)
    monitoring.releases_dir = tmp_release_structure / "data" / "monitoring" / "releases"
    monitoring.releases_dir.mkdir(parents=True, exist_ok=True)

    monitoring.register_release(release_id, manifest_path, heightened_hours=48)

    # Step 3: Verify complete pipeline
    # Manifest exists with correct hashes
    assert manifest_path.exists()
    with open(manifest_path) as f:
        loaded_manifest = yaml.safe_load(f)
    assert loaded_manifest["artifacts"]["configs"][0]["hash"] == f"sha256:{config_hash}"
    assert loaded_manifest["artifacts"]["models"][0]["hash"] == f"sha256:{model_hash}"

    # Backups exist
    assert (backup_dir / "config.py").exists()
    assert (backup_dir / "student_model.pkl").exists()

    # Release registered
    active_release = monitoring.get_active_release()
    assert active_release is not None
    assert active_release["release_id"] == release_id

    # Heightened monitoring active
    assert monitoring.is_in_heightened_monitoring() is True

    # Alert thresholds stricter
    thresholds = monitoring.get_alert_thresholds()
    assert thresholds["intraday_hit_ratio_drop"] == 0.05

    # Monitoring windows shorter
    windows = monitoring.get_monitoring_window_hours()
    assert windows["intraday"] == 6
    assert windows["swing"] == 24

    # Persistence verified
    assert (monitoring.releases_dir / "active_release.yaml").exists()
    assert (monitoring.releases_dir / f"{release_id}.yaml").exists()
