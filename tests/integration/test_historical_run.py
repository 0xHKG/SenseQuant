"""Integration tests for historical model training execution & promotion (US-028)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.state_manager import StateManager


@pytest.fixture
def temp_state_file(tmp_path: Path) -> Path:
    """Create temporary state file."""
    return tmp_path / "test_state.json"


@pytest.fixture
def mock_orchestrator(tmp_path: Path, temp_state_file: Path):
    """Create mock historical run orchestrator."""
    # Import after adding to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    from run_historical_training import HistoricalRunOrchestrator

    orchestrator = HistoricalRunOrchestrator(
        symbols=["RELIANCE", "TCS"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        skip_fetch=True,
        dryrun=True,
    )

    # Override directories to use temp path
    orchestrator.model_dir = tmp_path / "models" / orchestrator.run_id
    orchestrator.audit_dir = tmp_path / "release" / f"audit_{orchestrator.run_id}"
    orchestrator.model_dir.mkdir(parents=True)
    orchestrator.audit_dir.mkdir(parents=True)

    # Override state manager
    orchestrator.state_mgr = StateManager(state_file=temp_state_file)

    return orchestrator


# ============================================================================
# Orchestration Tests
# ============================================================================


def test_historical_run_orchestration(mock_orchestrator) -> None:
    """Test full pipeline execution (dryrun mode)."""
    # Execute pipeline
    success = mock_orchestrator.run()

    # Verify success
    assert success is True

    # Verify all phases executed
    phases = mock_orchestrator.results["phases"]
    assert "data_ingestion" in phases
    assert "teacher_training" in phases
    assert "student_training" in phases
    assert "model_validation" in phases
    assert "statistical_tests" in phases
    assert "release_audit" in phases
    assert "promotion_briefing" in phases


def test_historical_run_directory_structure(mock_orchestrator) -> None:
    """Test artifact directory structure is created."""
    # Execute pipeline
    mock_orchestrator.run()

    # Verify model directory structure
    assert mock_orchestrator.model_dir.exists()
    assert (mock_orchestrator.model_dir / "teacher_runs.json").exists()
    assert (mock_orchestrator.model_dir / "student_runs.json").exists()

    # Verify audit directory structure
    assert mock_orchestrator.audit_dir.exists()
    assert (mock_orchestrator.audit_dir / "validation_summary.json").exists()
    assert (mock_orchestrator.audit_dir / "stat_tests.json").exists()
    assert (mock_orchestrator.audit_dir / "manifest.yaml").exists()
    assert (mock_orchestrator.audit_dir / "promotion_briefing.md").exists()
    assert (mock_orchestrator.audit_dir / "promotion_briefing.json").exists()


def test_historical_run_promotion_briefing(mock_orchestrator) -> None:
    """Test promotion briefing generation."""
    # Execute pipeline
    mock_orchestrator.run()

    # Load briefing
    briefing_md = (mock_orchestrator.audit_dir / "promotion_briefing.md").read_text()
    briefing_json_file = mock_orchestrator.audit_dir / "promotion_briefing.json"
    with open(briefing_json_file) as f:
        briefing_json = json.load(f)

    # Verify Markdown briefing
    assert "Promotion Briefing" in briefing_md
    assert mock_orchestrator.run_id in briefing_md
    assert "Training Summary" in briefing_md
    assert "Validation Results" in briefing_md
    assert "Risk Assessment" in briefing_md
    assert "Next Steps" in briefing_md

    # Verify JSON briefing
    assert briefing_json["run_id"] == mock_orchestrator.run_id
    assert briefing_json["status"] == "ready-for-review"
    assert "training" in briefing_json
    assert "validation" in briefing_json
    assert "statistical_tests" in briefing_json
    assert "risk_assessment" in briefing_json
    assert "artifacts" in briefing_json


def test_historical_run_state_manager(temp_state_file: Path) -> None:
    """Test candidate run tracking in state manager."""
    state_mgr = StateManager(state_file=temp_state_file)

    # Record candidate run
    state_mgr.record_candidate_run(
        run_id="live_candidate_20251012_153000",
        timestamp=datetime.now().isoformat(),
        status="ready-for-review",
        training={
            "symbols": ["RELIANCE", "TCS"],
            "teacher": {"runs_completed": 12, "avg_precision": 0.82},
            "student": {"total_samples": 25000, "accuracy": 0.84},
        },
        validation={"passed": True, "reports": ["accuracy_report.html"]},
        statistical_tests={"sharpe_ratio": 1.45, "bootstrap_significant": True},
        artifacts={
            "model_dir": "data/models/live_candidate_20251012_153000",
            "audit_dir": "release/audit_live_candidate_20251012_153000",
        },
    )

    # Verify latest candidate
    latest = state_mgr.get_latest_candidate_run()
    assert latest is not None
    assert latest["run_id"] == "live_candidate_20251012_153000"
    assert latest["status"] == "ready-for-review"

    # Verify get all candidates
    candidates = state_mgr.get_candidate_runs()
    assert len(candidates) == 1
    assert candidates[0]["run_id"] == "live_candidate_20251012_153000"

    # Verify filtered by status
    ready_candidates = state_mgr.get_candidate_runs(status="ready-for-review")
    assert len(ready_candidates) == 1

    approved_candidates = state_mgr.get_candidate_runs(status="approved")
    assert len(approved_candidates) == 0


def test_historical_run_candidate_approval(temp_state_file: Path) -> None:
    """Test candidate run approval workflow."""
    state_mgr = StateManager(state_file=temp_state_file)

    # Record candidate
    run_id = "live_candidate_20251012_153000"
    state_mgr.record_candidate_run(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        status="ready-for-review",
        training={},
        validation={},
        statistical_tests={},
        artifacts={},
    )

    # Approve candidate
    state_mgr.approve_candidate_run(run_id, approved_by="test-user")

    # Verify approval
    latest = state_mgr.get_latest_candidate_run()
    assert latest is not None
    assert latest["status"] == "approved"
    assert latest["approved_by"] == "test-user"
    assert "approved_at" in latest

    # Verify filtered by approved status
    approved_candidates = state_mgr.get_candidate_runs(status="approved")
    assert len(approved_candidates) == 1


def test_historical_run_metrics_loading(mock_orchestrator) -> None:
    """Test metrics loading from artifact files."""
    # Create mock artifacts
    mock_orchestrator._create_mock_teacher_runs()
    mock_orchestrator._create_mock_student_runs()
    mock_orchestrator._create_mock_validation_summary()
    mock_orchestrator._create_mock_stat_tests()

    # Load metrics
    teacher_metrics = mock_orchestrator._load_teacher_metrics()
    student_metrics = mock_orchestrator._load_student_metrics()
    validation_metrics = mock_orchestrator._load_validation_metrics()
    stat_metrics = mock_orchestrator._load_stat_metrics()

    # Verify teacher metrics
    assert teacher_metrics["runs_completed"] > 0
    assert "avg_precision" in teacher_metrics
    assert "avg_recall" in teacher_metrics
    assert "avg_f1" in teacher_metrics

    # Verify student metrics
    assert student_metrics["total_samples"] == 25000
    assert student_metrics["accuracy"] == 0.84

    # Verify validation metrics
    assert validation_metrics["passed"] is True
    assert len(validation_metrics["reports"]) > 0

    # Verify stat metrics
    assert stat_metrics["walk_forward_passed"] is True
    assert stat_metrics["bootstrap_significant"] is True
    assert stat_metrics["sharpe_ratio"] > 0


# ============================================================================
# Phase Tests
# ============================================================================


def test_historical_run_phase_1_data_ingestion(mock_orchestrator) -> None:
    """Test phase 1 (data ingestion) execution."""
    success = mock_orchestrator._run_phase_1_data_ingestion()

    assert success is True
    assert "data_ingestion" in mock_orchestrator.results["phases"]
    assert mock_orchestrator.results["phases"]["data_ingestion"]["status"] in [
        "success",
        "skipped",
    ]


def test_historical_run_phase_2_teacher_training(mock_orchestrator) -> None:
    """Test phase 2 (teacher training) execution."""
    success = mock_orchestrator._run_phase_2_teacher_training()

    assert success is True
    assert "teacher_training" in mock_orchestrator.results["phases"]

    # Verify teacher_runs.json created
    assert (mock_orchestrator.model_dir / "teacher_runs.json").exists()


def test_historical_run_phase_3_student_training(mock_orchestrator) -> None:
    """Test phase 3 (student training) execution."""
    # First run teacher training to create dependencies
    mock_orchestrator._run_phase_2_teacher_training()

    success = mock_orchestrator._run_phase_3_student_training()

    assert success is True
    assert "student_training" in mock_orchestrator.results["phases"]

    # Verify student_runs.json created
    assert (mock_orchestrator.model_dir / "student_runs.json").exists()


def test_historical_run_phase_4_validation(mock_orchestrator) -> None:
    """Test phase 4 (validation) execution."""
    success = mock_orchestrator._run_phase_4_model_validation()

    assert success is True
    assert "model_validation" in mock_orchestrator.results["phases"]

    # Verify validation_summary.json created
    assert (mock_orchestrator.audit_dir / "validation_summary.json").exists()


def test_historical_run_phase_5_statistical_tests(mock_orchestrator) -> None:
    """Test phase 5 (statistical tests) execution."""
    success = mock_orchestrator._run_phase_5_statistical_tests()

    assert success is True
    assert "statistical_tests" in mock_orchestrator.results["phases"]

    # Verify stat_tests.json created
    assert (mock_orchestrator.audit_dir / "stat_tests.json").exists()


def test_historical_run_phase_6_release_audit(mock_orchestrator) -> None:
    """Test phase 6 (release audit) execution."""
    success = mock_orchestrator._run_phase_6_release_audit()

    assert success is True
    assert "release_audit" in mock_orchestrator.results["phases"]

    # Verify manifest.yaml created
    assert (mock_orchestrator.audit_dir / "manifest.yaml").exists()


def test_historical_run_phase_7_promotion_briefing(mock_orchestrator) -> None:
    """Test phase 7 (promotion briefing) execution."""
    # Create dependencies
    mock_orchestrator._create_mock_teacher_runs()
    mock_orchestrator._create_mock_student_runs()
    mock_orchestrator._create_mock_validation_summary()
    mock_orchestrator._create_mock_stat_tests()

    success = mock_orchestrator._run_phase_7_promotion_briefing()

    assert success is True
    assert "promotion_briefing" in mock_orchestrator.results["phases"]

    # Verify briefing files created
    assert (mock_orchestrator.audit_dir / "promotion_briefing.md").exists()
    assert (mock_orchestrator.audit_dir / "promotion_briefing.json").exists()


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_historical_run_phase_failure_handling(mock_orchestrator) -> None:
    """Test graceful handling of phase failures."""
    # Mock a phase to fail
    with patch.object(mock_orchestrator, "_run_phase_2_teacher_training", return_value=False):
        success = mock_orchestrator.run()

        # Verify pipeline stopped
        assert success is False


def test_historical_run_partial_metrics(mock_orchestrator) -> None:
    """Test handling of missing metrics files."""
    # Don't create mock files

    # Load metrics (should return empty dicts, not crash)
    teacher_metrics = mock_orchestrator._load_teacher_metrics()
    student_metrics = mock_orchestrator._load_student_metrics()
    validation_metrics = mock_orchestrator._load_validation_metrics()
    stat_metrics = mock_orchestrator._load_stat_metrics()

    # Verify empty dicts returned
    assert isinstance(teacher_metrics, dict)
    assert isinstance(student_metrics, dict)
    assert isinstance(validation_metrics, dict)
    assert isinstance(stat_metrics, dict)
