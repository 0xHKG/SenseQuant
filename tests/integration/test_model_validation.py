"""Integration tests for US-025: Model Validation workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.app.config import Settings
from src.services.state_manager import StateManager


@pytest.fixture
def tmp_validation_dir(tmp_path: Path) -> Path:
    """Create temporary validation directory."""
    val_dir = tmp_path / "validation"
    val_dir.mkdir()
    return val_dir


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for testing."""
    return Settings()  # type: ignore[call-arg]


def test_validation_run_state_tracking(tmp_validation_dir: Path) -> None:
    """Test StateManager tracks validation runs (US-025)."""
    state_file = tmp_validation_dir / "validation_runs.json"
    manager = StateManager(state_file)

    # Record validation run
    manager.record_validation_run(
        run_id="validation_20251012_180000",
        timestamp="2025-10-12T18:00:00+05:30",
        symbols=["RELIANCE", "TCS"],
        date_range={"start": "2024-01-01", "end": "2024-12-31"},
        status="completed",
        dryrun=True,
        results={"teacher_results": {"status": "success"}},
    )

    # Verify run recorded
    run = manager.get_validation_run("validation_20251012_180000")
    assert run is not None
    assert run["run_id"] == "validation_20251012_180000"
    assert run["status"] == "completed"
    assert run["dryrun"] is True
    assert len(run["symbols"]) == 2

    # Get all runs
    all_runs = manager.get_validation_runs()
    assert len(all_runs) == 1

    # Filter by status
    completed_runs = manager.get_validation_runs(status="completed")
    assert len(completed_runs) == 1

    # Filter by dryrun
    dryrun_runs = manager.get_validation_runs(dryrun=True)
    assert len(dryrun_runs) == 1


def test_validation_runner_dryrun_mode(tmp_validation_dir: Path) -> None:
    """Test validation runner in dryrun mode (US-025)."""
    import sys

    sys.path.insert(0, "scripts")

    try:
        from run_model_validation import ModelValidationRunner

        # Create runner in dryrun mode
        runner = ModelValidationRunner(
            run_id="test_validation_001",
            symbols=["RELIANCE"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            dryrun=True,
            skip_optimizer=True,
            skip_reports=True,
        )

        # Run validation
        results = runner.run()

        # Verify results
        assert results["status"] == "completed"
        assert results["dryrun"] is True
        assert results["teacher_results"]["status"] == "skipped"
        assert results["student_results"]["status"] == "skipped"

    finally:
        sys.path.pop(0)


def test_validation_directory_structure(tmp_validation_dir: Path) -> None:
    """Test validation creates proper directory structure (US-025)."""
    import sys

    sys.path.insert(0, "scripts")

    try:
        from run_model_validation import ModelValidationRunner

        run_id = "test_validation_002"

        # Create runner
        runner = ModelValidationRunner(
            run_id=run_id,
            symbols=["RELIANCE", "TCS"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            dryrun=True,
            skip_optimizer=True,
            skip_reports=True,
        )

        # Verify directories created
        assert runner.models_dir.exists()
        assert runner.optimization_dir.exists()
        assert runner.release_dir.exists()
        assert runner.reports_dir.exists()

    finally:
        sys.path.pop(0)


def test_validation_summary_generation(tmp_validation_dir: Path) -> None:
    """Test validation generates summary files (US-025)."""
    import sys

    sys.path.insert(0, "scripts")

    try:
        from run_model_validation import ModelValidationRunner

        run_id = "test_validation_003"

        # Create runner
        runner = ModelValidationRunner(
            run_id=run_id,
            symbols=["RELIANCE"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            dryrun=True,
            skip_optimizer=True,
            skip_reports=True,
        )

        # Run validation
        runner.run()

        # Verify summary files created
        summary_json = runner.release_dir / "validation_summary.json"
        summary_md = runner.release_dir / "validation_summary.md"

        assert summary_json.exists()
        assert summary_md.exists()

        # Verify JSON content
        import json

        with open(summary_json) as f:
            summary = json.load(f)
            assert summary["run_id"] == run_id
            assert summary["status"] == "completed"
            assert summary["dryrun"] is True

        # Verify Markdown content
        md_content = summary_md.read_text()
        assert run_id in md_content
        assert "Validation Summary" in md_content

    finally:
        sys.path.pop(0)


def test_validation_runner_error_handling(tmp_validation_dir: Path) -> None:
    """Test validation runner handles errors gracefully (US-025)."""
    import sys

    sys.path.insert(0, "scripts")

    try:
        from run_model_validation import ModelValidationRunner

        # Create runner that will fail (invalid dates)
        runner = ModelValidationRunner(
            run_id="test_validation_004",
            symbols=["INVALID_SYMBOL"],
            start_date="invalid-date",
            end_date="invalid-date",
            dryrun=True,
            skip_optimizer=True,
            skip_reports=True,
        )

        # Run should complete even with invalid input (in dryrun mode)
        results = runner.run()
        assert results["status"] in ["completed", "failed"]

    finally:
        sys.path.pop(0)
