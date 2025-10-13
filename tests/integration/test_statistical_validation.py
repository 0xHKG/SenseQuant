"""Integration tests for statistical validation (US-026)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.state_manager import StateManager


@pytest.fixture
def tmp_statistical_dir(tmp_path: Path) -> Path:
    """Create temporary directory for statistical validation tests."""
    stat_dir = tmp_path / "statistical_validation"
    stat_dir.mkdir()

    # Change to temp directory
    import os

    old_cwd = os.getcwd()
    os.chdir(stat_dir)

    yield stat_dir

    # Restore original directory
    os.chdir(old_cwd)


def test_state_manager_statistical_validation(tmp_statistical_dir: Path) -> None:
    """Test StateManager tracks statistical validation runs (US-026)."""
    state_file = tmp_statistical_dir / "validation_runs.json"
    manager = StateManager(state_file)

    # Record statistical validation
    manager.record_statistical_validation(
        run_id="validation_20251012_180000",
        timestamp="2025-10-12T18:00:00+05:30",
        status="completed",
        walk_forward_results={
            "num_folds": 4,
            "aggregate": {"student": {"accuracy": {"mean": 0.84, "std": 0.02, "cv": 0.024}}},
        },
        bootstrap_results={
            "n_iterations": 1000,
            "results": {
                "student_accuracy": {
                    "mean": 0.84,
                    "ci_lower": 0.81,
                    "ci_upper": 0.87,
                }
            },
        },
        benchmark_comparison={
            "benchmark": "NIFTY_50",
            "alpha": 0.015,
            "beta": 0.82,
            "information_ratio": 0.45,
        },
    )

    # Verify validation recorded
    validation = manager.get_statistical_validation("validation_20251012_180000")
    assert validation is not None
    assert validation["status"] == "completed"
    assert validation["walk_forward_results"]["num_folds"] == 4
    assert validation["bootstrap_results"]["n_iterations"] == 1000
    assert validation["benchmark_comparison"]["benchmark"] == "NIFTY_50"

    # Verify last benchmark comparison updated
    last_bench = manager.get_last_benchmark_comparison()
    assert last_bench is not None
    assert last_bench["benchmark"] == "NIFTY_50"
    assert last_bench["alpha"] == 0.015
    assert last_bench["beta"] == 0.82


def test_statistical_validator_dryrun(tmp_statistical_dir: Path) -> None:
    """Test statistical validator in dryrun mode (US-026)."""
    import sys
    from pathlib import Path as PathLib

    # Add scripts directory to path
    scripts_dir = PathLib(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    try:
        from run_statistical_tests import StatisticalValidator

        # Create mock validation summary
        release_dir = Path("release/audit_validation_test_001")
        release_dir.mkdir(parents=True, exist_ok=True)

        validation_summary = {
            "run_id": "validation_test_001",
            "status": "completed",
            "symbols": ["RELIANCE"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "dryrun": False,
        }

        with open(release_dir / "validation_summary.json", "w") as f:
            json.dump(validation_summary, f)

        # Run statistical validator in dryrun
        validator = StatisticalValidator(
            run_id="validation_test_001",
            bootstrap_iterations=100,
            dryrun=True,
        )

        results = validator.run()

        # Verify dryrun behavior
        assert results["status"] == "skipped"
        assert results["reason"] == "dryrun"

    finally:
        sys.path.pop(0)


def test_statistical_validator_walk_forward_cv(tmp_statistical_dir: Path) -> None:
    """Test walk-forward cross-validation generates correct results (US-026)."""
    import sys

    # Add scripts directory to path
    from pathlib import Path as PathLib

    scripts_dir = PathLib(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    try:
        from run_statistical_tests import StatisticalValidator

        # Create mock validation summary
        release_dir = Path("release/audit_validation_test_002")
        release_dir.mkdir(parents=True, exist_ok=True)

        validation_summary = {
            "run_id": "validation_test_002",
            "status": "completed",
            "symbols": ["RELIANCE", "TCS"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "dryrun": False,
        }

        with open(release_dir / "validation_summary.json", "w") as f:
            json.dump(validation_summary, f)

        # Run statistical validator
        validator = StatisticalValidator(
            run_id="validation_test_002",
            bootstrap_iterations=100,
            dryrun=False,
        )

        results = validator.run()

        # Verify walk-forward CV results
        assert results["status"] == "completed"
        assert "walk_forward_cv" in results
        assert results["walk_forward_cv"]["num_folds"] == 4
        assert "aggregate" in results["walk_forward_cv"]
        assert "student" in results["walk_forward_cv"]["aggregate"]

        # Check student accuracy metrics
        student_acc = results["walk_forward_cv"]["aggregate"]["student"]["accuracy"]
        assert "mean" in student_acc
        assert "std" in student_acc
        assert "cv" in student_acc
        assert 0.0 <= student_acc["mean"] <= 1.0
        assert student_acc["std"] >= 0.0
        assert student_acc["cv"] >= 0.0

    finally:
        sys.path.pop(0)


def test_statistical_validator_bootstrap_tests(tmp_statistical_dir: Path) -> None:
    """Test bootstrap significance tests compute confidence intervals (US-026)."""
    import sys

    # Add scripts directory to path
    from pathlib import Path as PathLib

    scripts_dir = PathLib(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    try:
        from run_statistical_tests import StatisticalValidator

        # Create mock validation summary
        release_dir = Path("release/audit_validation_test_003")
        release_dir.mkdir(parents=True, exist_ok=True)

        validation_summary = {
            "run_id": "validation_test_003",
            "status": "completed",
            "symbols": ["RELIANCE"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "dryrun": False,
        }

        with open(release_dir / "validation_summary.json", "w") as f:
            json.dump(validation_summary, f)

        # Run statistical validator
        validator = StatisticalValidator(
            run_id="validation_test_003",
            bootstrap_iterations=500,
            confidence_level=0.95,
            dryrun=False,
        )

        results = validator.run()

        # Verify bootstrap results
        assert "bootstrap_tests" in results
        bootstrap = results["bootstrap_tests"]
        assert bootstrap["n_iterations"] == 500
        assert bootstrap["confidence_level"] == 0.95

        # Check confidence intervals
        assert "results" in bootstrap
        assert "student_accuracy" in bootstrap["results"]

        acc_ci = bootstrap["results"]["student_accuracy"]
        assert "ci_lower" in acc_ci
        assert "ci_upper" in acc_ci
        assert "mean" in acc_ci
        assert "std" in acc_ci

        # Verify CI bounds are sensible
        assert acc_ci["ci_lower"] < acc_ci["mean"]
        assert acc_ci["mean"] < acc_ci["ci_upper"]
        assert 0.0 <= acc_ci["ci_lower"] <= 1.0
        assert 0.0 <= acc_ci["ci_upper"] <= 1.0

    finally:
        sys.path.pop(0)


def test_statistical_validator_sharpe_comparison(tmp_statistical_dir: Path) -> None:
    """Test Sharpe/Sortino ratio comparisons (US-026)."""
    import sys

    # Add scripts directory to path
    from pathlib import Path as PathLib

    scripts_dir = PathLib(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    try:
        from run_statistical_tests import StatisticalValidator

        # Create mock validation summary
        release_dir = Path("release/audit_validation_test_004")
        release_dir.mkdir(parents=True, exist_ok=True)

        validation_summary = {
            "run_id": "validation_test_004",
            "status": "completed",
            "symbols": ["RELIANCE"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "dryrun": False,
        }

        with open(release_dir / "validation_summary.json", "w") as f:
            json.dump(validation_summary, f)

        # Run statistical validator
        validator = StatisticalValidator(
            run_id="validation_test_004",
            dryrun=False,
        )

        results = validator.run()

        # Verify Sharpe comparison
        assert "sharpe_comparison" in results
        sharpe = results["sharpe_comparison"]

        # Check baseline metrics
        assert "baseline" in sharpe
        assert "sharpe_ratio" in sharpe["baseline"]
        assert "sortino_ratio" in sharpe["baseline"]

        # Check strategy metrics
        assert "strategy" in sharpe
        assert "sharpe_ratio" in sharpe["strategy"]
        assert "sortino_ratio" in sharpe["strategy"]

        # Check delta
        assert "delta" in sharpe
        assert "sharpe_delta" in sharpe["delta"]
        assert "sortino_delta" in sharpe["delta"]

        # Check significance test
        assert "significance" in sharpe
        assert "p_value" in sharpe["significance"]
        assert "reject_null" in sharpe["significance"]

    finally:
        sys.path.pop(0)


def test_statistical_validator_benchmark_comparison(tmp_statistical_dir: Path) -> None:
    """Test benchmark comparison against NIFTY 50 (US-026)."""
    import sys

    # Add scripts directory to path
    from pathlib import Path as PathLib

    scripts_dir = PathLib(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    try:
        from run_statistical_tests import StatisticalValidator

        # Create mock validation summary
        release_dir = Path("release/audit_validation_test_005")
        release_dir.mkdir(parents=True, exist_ok=True)

        validation_summary = {
            "run_id": "validation_test_005",
            "status": "completed",
            "symbols": ["RELIANCE", "TCS"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "dryrun": False,
        }

        with open(release_dir / "validation_summary.json", "w") as f:
            json.dump(validation_summary, f)

        # Run statistical validator
        validator = StatisticalValidator(
            run_id="validation_test_005",
            benchmark="NIFTY_50",
            dryrun=False,
        )

        results = validator.run()

        # Verify benchmark comparison
        assert "benchmark_comparison" in results
        bench = results["benchmark_comparison"]

        # Check benchmark metrics
        assert bench["benchmark"] == "NIFTY_50"
        assert "benchmark_return" in bench
        assert "strategy_return" in bench

        # Check risk metrics
        assert "alpha" in bench
        assert "beta" in bench
        assert "information_ratio" in bench
        assert "tracking_error" in bench
        assert "z_score" in bench
        assert "correlation" in bench

        # Check relative performance
        assert "relative_performance" in bench
        assert "excess_return" in bench["relative_performance"]
        assert "significant" in bench["relative_performance"]
        assert "p_value" in bench["relative_performance"]

    finally:
        sys.path.pop(0)


def test_statistical_validator_updates_validation_summary(
    tmp_statistical_dir: Path,
) -> None:
    """Test statistical validator updates validation_summary.json (US-026)."""
    import sys

    # Add scripts directory to path
    from pathlib import Path as PathLib

    scripts_dir = PathLib(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    try:
        from run_statistical_tests import StatisticalValidator

        # Create mock validation summary
        release_dir = Path("release/audit_validation_test_006")
        release_dir.mkdir(parents=True, exist_ok=True)

        validation_summary = {
            "run_id": "validation_test_006",
            "status": "completed",
            "symbols": ["RELIANCE"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "dryrun": False,
            "promotion_recommendation": {
                "approved": True,
                "reason": "Accuracy thresholds met",
            },
        }

        summary_path = release_dir / "validation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(validation_summary, f)

        # Run statistical validator
        validator = StatisticalValidator(
            run_id="validation_test_006",
            dryrun=False,
        )

        validator.run()

        # Verify validation summary was updated
        with open(summary_path) as f:
            updated_summary = json.load(f)

        # Check statistical validation section added
        assert "statistical_validation" in updated_summary
        stat_val = updated_summary["statistical_validation"]

        assert stat_val["status"] == "completed"
        assert "walk_forward_cv" in stat_val
        assert "confidence_intervals" in stat_val
        assert "significance_tests" in stat_val
        assert "risk_adjusted_performance" in stat_val
        assert "benchmark_comparison" in stat_val

        # Check promotion recommendation updated
        promo = updated_summary["promotion_recommendation"]
        assert "statistical_confidence" in promo
        assert "risk_assessment" in promo

    finally:
        sys.path.pop(0)


def test_statistical_validator_generates_stat_tests_json(
    tmp_statistical_dir: Path,
) -> None:
    """Test statistical validator generates stat_tests.json file (US-026)."""
    import sys

    # Add scripts directory to path
    from pathlib import Path as PathLib

    scripts_dir = PathLib(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    try:
        from run_statistical_tests import StatisticalValidator

        # Create mock validation summary
        release_dir = Path("release/audit_validation_test_007")
        release_dir.mkdir(parents=True, exist_ok=True)

        validation_summary = {
            "run_id": "validation_test_007",
            "status": "completed",
            "symbols": ["RELIANCE"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "dryrun": False,
        }

        with open(release_dir / "validation_summary.json", "w") as f:
            json.dump(validation_summary, f)

        # Run statistical validator
        validator = StatisticalValidator(
            run_id="validation_test_007",
            dryrun=False,
        )

        validator.run()

        # Verify stat_tests.json created
        stat_tests_path = release_dir / "stat_tests.json"
        assert stat_tests_path.exists()

        # Load and verify contents
        with open(stat_tests_path) as f:
            stat_tests = json.load(f)

        assert stat_tests["run_id"] == "validation_test_007"
        assert stat_tests["status"] == "completed"
        assert "timestamp" in stat_tests
        assert "walk_forward_cv" in stat_tests
        assert "bootstrap_tests" in stat_tests
        assert "hypothesis_tests" in stat_tests
        assert "sharpe_comparison" in stat_tests
        assert "benchmark_comparison" in stat_tests

    finally:
        sys.path.pop(0)
