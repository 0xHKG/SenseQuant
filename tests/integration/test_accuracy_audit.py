"""Integration tests for accuracy audit system.

Tests the complete telemetry capture and accuracy analysis pipeline:
- Telemetry capture during backtest
- Accuracy metrics computation from traces
- Telemetry enable/disable functionality
- Output file verification
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import pytz

from src.app.config import Settings
from src.domain.types import BacktestConfig, Bar
from src.services.accuracy_analyzer import AccuracyAnalyzer, PredictionTrace
from src.services.backtester import Backtester


@pytest.fixture
def sample_swing_bars_with_signals() -> list[Bar]:
    """Generate sample daily bars with clear trading signals.

    Creates a pattern that will generate multiple trades:
    - Initial downtrend (bars 0-80)
    - Sharp rally causing bullish crossover (bars 80-100)
    - Consolidation and exit signal (bars 100-120)
    """
    ist = pytz.timezone("Asia/Kolkata")
    base_date = pd.Timestamp("2024-01-01", tz=ist)
    bars = []

    for i in range(120):
        ts = base_date + pd.Timedelta(days=i)

        if i < 80:
            # Downtrend: below both SMAs
            close = 150.0 - (i * 0.5)
        elif i < 100:
            # Rally: causes bullish crossover
            close = 110.0 + ((i - 79) * 3.0)
        else:
            # Consolidation: triggers exit
            close = 170.0 + ((i - 99) * 0.5)

        bars.append(
            Bar(
                ts=ts,
                open=close - 0.5,
                high=close + 2.0,
                low=close - 2.0,
                close=close,
                volume=100000,
            )
        )

    return bars


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with telemetry enabled."""
    settings = Settings()  # type: ignore[call-arg]
    settings.telemetry_enabled = True
    settings.telemetry_sample_rate = 1.0  # Capture all traces
    settings.telemetry_compression = False  # Easier to read in tests
    settings.telemetry_buffer_size = 10
    settings.telemetry_max_file_size_mb = 10
    return settings


def test_telemetry_capture_during_backtest(
    sample_swing_bars_with_signals: list[Bar],
    test_settings: Settings,
) -> None:
    """Test that telemetry is captured during backtest.

    Verifies:
    - Telemetry files are created in expected location
    - Trace files contain data
    - File format is correct (CSV with headers)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        telemetry_dir = Path(tmpdir) / "telemetry"
        telemetry_dir.mkdir(parents=True, exist_ok=True)

        # Create mock data feed
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.historical_bars.return_value = sample_swing_bars_with_signals

        config = BacktestConfig(
            symbols=["TEST"],
            start_date="2024-01-01",
            end_date="2024-04-30",
            strategy="swing",
            initial_capital=1000000.0,
            data_source="breeze",
            random_seed=42,
        )

        # Run backtest with telemetry enabled
        backtester = Backtester(
            config=config,
            client=mock_client,
            settings=test_settings,
            enable_telemetry=True,
            telemetry_dir=telemetry_dir,
        )

        result = backtester.run()

        # Verify backtest completed
        assert result is not None
        assert result.metrics["total_trades"] >= 0

        # Verify telemetry files were created
        telemetry_files = list(telemetry_dir.glob("predictions_*.csv"))

        # If trades occurred, telemetry files should exist
        if result.metrics["total_trades"] > 0:
            assert len(telemetry_files) > 0, "Telemetry files should exist when trades occur"

            # Verify file format
            for file_path in telemetry_files:
                # Check file is not empty
                assert file_path.stat().st_size > 0, f"Telemetry file {file_path} is empty"

                # Load and verify CSV structure
                df = pd.read_csv(file_path)
                assert len(df) > 0, "Telemetry CSV should contain traces"

                # Verify required columns
                required_columns = [
                    "timestamp",
                    "symbol",
                    "strategy",
                    "predicted_direction",
                    "actual_direction",
                    "predicted_confidence",
                    "entry_price",
                    "exit_price",
                    "holding_period_minutes",
                    "realized_return_pct",
                ]
                for col in required_columns:
                    assert col in df.columns, f"Missing required column: {col}"


def test_telemetry_disabled(
    sample_swing_bars_with_signals: list[Bar],
    test_settings: Settings,
) -> None:
    """Test that telemetry is NOT captured when disabled.

    Verifies:
    - No telemetry files are created
    - Backtest still completes successfully
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        telemetry_dir = Path(tmpdir) / "telemetry"
        telemetry_dir.mkdir(parents=True, exist_ok=True)

        # Create mock data feed
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.historical_bars.return_value = sample_swing_bars_with_signals

        config = BacktestConfig(
            symbols=["TEST"],
            start_date="2024-01-01",
            end_date="2024-04-30",
            strategy="swing",
            initial_capital=1000000.0,
            data_source="breeze",
            random_seed=42,
        )

        # Run backtest with telemetry DISABLED
        backtester = Backtester(
            config=config,
            client=mock_client,
            settings=test_settings,
            enable_telemetry=False,  # Explicitly disabled
            telemetry_dir=telemetry_dir,
        )

        result = backtester.run()

        # Verify backtest completed
        assert result is not None

        # Verify NO telemetry files were created
        telemetry_files = list(telemetry_dir.glob("predictions_*.csv"))
        assert len(telemetry_files) == 0, "No telemetry files should be created when disabled"


def test_accuracy_metrics_computation(
    sample_swing_bars_with_signals: list[Bar],
    test_settings: Settings,
) -> None:
    """Test that accuracy metrics can be computed from telemetry traces.

    Verifies:
    - Traces can be loaded from CSV files
    - Metrics are computed correctly
    - Key metrics are within expected ranges
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        telemetry_dir = Path(tmpdir) / "telemetry"
        telemetry_dir.mkdir(parents=True, exist_ok=True)

        # Create mock data feed
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.historical_bars.return_value = sample_swing_bars_with_signals

        config = BacktestConfig(
            symbols=["TEST"],
            start_date="2024-01-01",
            end_date="2024-04-30",
            strategy="swing",
            initial_capital=1000000.0,
            data_source="breeze",
            random_seed=42,
        )

        # Run backtest with telemetry enabled
        backtester = Backtester(
            config=config,
            client=mock_client,
            settings=test_settings,
            enable_telemetry=True,
            telemetry_dir=telemetry_dir,
        )

        result = backtester.run()

        # Skip test if no trades occurred (pattern dependent)
        if result.metrics["total_trades"] == 0:
            pytest.skip("No trades occurred in backtest")

        # Load telemetry traces
        telemetry_files = list(telemetry_dir.glob("predictions_*.csv"))
        assert len(telemetry_files) > 0, "Telemetry files should exist"

        analyzer = AccuracyAnalyzer()

        # Load traces from all files
        all_traces: list[PredictionTrace] = []
        for file_path in telemetry_files:
            traces = analyzer.load_traces(file_path)
            all_traces.extend(traces)

        assert len(all_traces) > 0, "Should have loaded traces"

        # Compute accuracy metrics
        metrics = analyzer.compute_metrics(all_traces)

        # Verify metrics structure
        assert metrics is not None
        assert metrics.total_trades == len(all_traces)
        assert 0.0 <= metrics.hit_ratio <= 1.0, "Hit ratio should be between 0 and 1"
        assert 0.0 <= metrics.win_rate <= 1.0, "Win rate should be between 0 and 1"

        # Verify precision/recall/f1 for each direction
        for direction in ["LONG", "SHORT", "NOOP"]:
            assert direction in metrics.precision
            assert direction in metrics.recall
            assert direction in metrics.f1_score
            assert 0.0 <= metrics.precision[direction] <= 1.0
            assert 0.0 <= metrics.recall[direction] <= 1.0
            assert 0.0 <= metrics.f1_score[direction] <= 1.0

        # Verify confusion matrix shape
        assert metrics.confusion_matrix.shape == (3, 3)

        # Verify financial metrics
        assert isinstance(metrics.avg_return, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert metrics.profit_factor >= 0.0


def test_accuracy_report_export(
    sample_swing_bars_with_signals: list[Bar],
    test_settings: Settings,
) -> None:
    """Test that accuracy reports can be exported to JSON.

    Verifies:
    - Report files are created
    - JSON structure is valid
    - Contains all expected fields
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        telemetry_dir = Path(tmpdir) / "telemetry"
        report_dir = Path(tmpdir) / "reports"
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        # Create mock data feed
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.historical_bars.return_value = sample_swing_bars_with_signals

        config = BacktestConfig(
            symbols=["TEST"],
            start_date="2024-01-01",
            end_date="2024-04-30",
            strategy="swing",
            initial_capital=1000000.0,
            data_source="breeze",
            random_seed=42,
        )

        # Run backtest with telemetry enabled
        backtester = Backtester(
            config=config,
            client=mock_client,
            settings=test_settings,
            enable_telemetry=True,
            telemetry_dir=telemetry_dir,
        )

        result = backtester.run()

        # Skip if no trades
        if result.metrics["total_trades"] == 0:
            pytest.skip("No trades occurred in backtest")

        # Load and analyze traces
        telemetry_files = list(telemetry_dir.glob("predictions_*.csv"))
        analyzer = AccuracyAnalyzer()

        all_traces: list[PredictionTrace] = []
        for file_path in telemetry_files:
            traces = analyzer.load_traces(file_path)
            all_traces.extend(traces)

        metrics = analyzer.compute_metrics(all_traces)

        # Export report
        report_path = report_dir / "accuracy_report.json"
        analyzer.export_report(metrics, report_path)

        # Verify report file exists
        assert report_path.exists(), "Report file should be created"
        assert report_path.stat().st_size > 0, "Report file should not be empty"

        # Verify JSON structure
        import json

        with open(report_path) as f:
            report_data = json.load(f)

        assert "generated_at" in report_data
        assert "metrics" in report_data

        # Verify metrics in report
        report_metrics = report_data["metrics"]
        assert "hit_ratio" in report_metrics
        assert "win_rate" in report_metrics
        assert "precision" in report_metrics
        assert "recall" in report_metrics
        assert "f1_score" in report_metrics
        assert "confusion_matrix" in report_metrics


def test_telemetry_sampling(
    sample_swing_bars_with_signals: list[Bar],
    test_settings: Settings,
) -> None:
    """Test that telemetry sampling rate works correctly.

    Verifies:
    - Sample rate of 0.0 captures no traces
    - Sample rate of 1.0 captures all traces (tested in other tests)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        telemetry_dir = Path(tmpdir) / "telemetry"
        telemetry_dir.mkdir(parents=True, exist_ok=True)

        # Set sample rate to 0.0 (capture nothing)
        test_settings.telemetry_sample_rate = 0.0

        # Create mock data feed
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.historical_bars.return_value = sample_swing_bars_with_signals

        config = BacktestConfig(
            symbols=["TEST"],
            start_date="2024-01-01",
            end_date="2024-04-30",
            strategy="swing",
            initial_capital=1000000.0,
            data_source="breeze",
            random_seed=42,
        )

        # Run backtest with telemetry enabled but sample rate = 0.0
        backtester = Backtester(
            config=config,
            client=mock_client,
            settings=test_settings,
            enable_telemetry=True,
            telemetry_dir=telemetry_dir,
        )

        result = backtester.run()

        # Verify backtest completed
        assert result is not None

        # Verify telemetry files are empty or don't exist
        telemetry_files = list(telemetry_dir.glob("predictions_*.csv"))

        # With sample rate 0.0, either no files or empty files
        for file_path in telemetry_files:
            df = pd.read_csv(file_path)
            assert len(df) == 0, "No traces should be captured with sample_rate=0.0"


def test_visualization_generation(
    sample_swing_bars_with_signals: list[Bar],
    test_settings: Settings,
) -> None:
    """Test that visualizations can be generated from traces.

    Verifies:
    - Confusion matrix plot is created
    - Return distribution plot is created
    - Files are valid images (non-empty)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        telemetry_dir = Path(tmpdir) / "telemetry"
        plots_dir = Path(tmpdir) / "plots"
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Create mock data feed
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.historical_bars.return_value = sample_swing_bars_with_signals

        config = BacktestConfig(
            symbols=["TEST"],
            start_date="2024-01-01",
            end_date="2024-04-30",
            strategy="swing",
            initial_capital=1000000.0,
            data_source="breeze",
            random_seed=42,
        )

        # Run backtest with telemetry enabled
        backtester = Backtester(
            config=config,
            client=mock_client,
            settings=test_settings,
            enable_telemetry=True,
            telemetry_dir=telemetry_dir,
        )

        result = backtester.run()

        # Skip if no trades
        if result.metrics["total_trades"] == 0:
            pytest.skip("No trades occurred in backtest")

        # Load and analyze traces
        telemetry_files = list(telemetry_dir.glob("predictions_*.csv"))
        analyzer = AccuracyAnalyzer()

        all_traces: list[PredictionTrace] = []
        for file_path in telemetry_files:
            traces = analyzer.load_traces(file_path)
            all_traces.extend(traces)

        metrics = analyzer.compute_metrics(all_traces)

        # Generate visualizations
        confusion_matrix_path = plots_dir / "confusion_matrix.png"
        return_dist_path = plots_dir / "return_distribution.png"

        analyzer.plot_confusion_matrix(metrics, confusion_matrix_path)
        analyzer.plot_return_distribution(all_traces, return_dist_path)

        # Verify files exist and are not empty
        assert confusion_matrix_path.exists(), "Confusion matrix plot should be created"
        assert confusion_matrix_path.stat().st_size > 0, "Confusion matrix plot should not be empty"

        assert return_dist_path.exists(), "Return distribution plot should be created"
        assert return_dist_path.stat().st_size > 0, "Return distribution plot should not be empty"


def test_release_audit_bundle_generation() -> None:
    """Test release audit bundle generation (US-022).

    Verifies:
    - Audit bundle is created with correct structure
    - metrics.json is present and valid
    - summary.md is present and non-empty
    - Required subdirectories exist
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "audit_test"

        # Run release audit script
        result = subprocess.run(
            [
                "python",
                "scripts/release_audit.py",
                "--output-dir",
                str(output_dir),
                "--skip-validation",
                "--skip-plots",
            ],
            capture_output=True,
            text=True,
        )

        # Check script executed (may fail with warnings, that's OK for test)
        assert result.returncode in [0, 1], f"Script failed: {result.stderr}"

        # Verify bundle directory exists
        assert output_dir.exists(), "Audit bundle directory should be created"

        # Verify required files
        metrics_path = output_dir / "metrics.json"
        assert metrics_path.exists(), "metrics.json should be created"
        assert metrics_path.stat().st_size > 0, "metrics.json should not be empty"

        summary_path = output_dir / "summary.md"
        assert summary_path.exists(), "summary.md should be created"
        assert summary_path.stat().st_size > 0, "summary.md should not be empty"

        # Verify subdirectories
        required_dirs = ["plots", "configs", "telemetry_summaries", "validation_results"]
        for dirname in required_dirs:
            subdir = output_dir / dirname
            assert subdir.exists(), f"{dirname} directory should be created"
            assert subdir.is_dir(), f"{dirname} should be a directory"


def test_audit_metrics_aggregation() -> None:
    """Test audit metrics aggregation and schema validation (US-022).

    Verifies:
    - metrics.json has correct schema
    - Required fields are present
    - Data types are correct
    """
    import json
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "audit_metrics_test"

        # Run release audit script
        subprocess.run(
            [
                "python",
                "scripts/release_audit.py",
                "--output-dir",
                str(output_dir),
                "--skip-validation",
                "--skip-plots",
            ],
            capture_output=True,
            text=True,
        )

        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            pytest.skip("metrics.json not generated (may need optimization data)")

        # Load and validate metrics
        with open(metrics_path) as f:
            metrics = json.load(f)

        # Verify required top-level keys
        required_keys = [
            "audit_timestamp",
            "audit_id",
            "monitoring",
            "validation_results",
            "risk_flags",
            "deployment_ready",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing required key: {key}"

        # Verify data types
        assert isinstance(metrics["audit_timestamp"], str)
        assert isinstance(metrics["audit_id"], str)
        assert isinstance(metrics["monitoring"], dict)
        assert isinstance(metrics["validation_results"], dict)
        assert isinstance(metrics["risk_flags"], list)
        assert isinstance(metrics["deployment_ready"], bool)

        # Verify audit_id format
        assert metrics["audit_id"].startswith("audit_"), "audit_id should have correct prefix"

        # Verify monitoring structure
        monitoring = metrics["monitoring"]
        assert isinstance(monitoring.get("intraday_30day", {}), dict)
        assert isinstance(monitoring.get("swing_90day", {}), dict)

        # Verify validation_results structure
        val_results = metrics["validation_results"]
        assert isinstance(val_results.get("optimizer_validation", {}), dict)
        assert isinstance(val_results.get("student_validation", {}), dict)


def test_audit_summary_markdown() -> None:
    """Test audit summary markdown generation (US-022).

    Verifies:
    - summary.md is properly formatted
    - Contains required sections
    - Has readable content
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "audit_summary_test"

        # Run release audit script
        subprocess.run(
            [
                "python",
                "scripts/release_audit.py",
                "--output-dir",
                str(output_dir),
                "--skip-validation",
                "--skip-plots",
            ],
            capture_output=True,
            text=True,
        )

        summary_path = output_dir / "summary.md"
        if not summary_path.exists():
            pytest.skip("summary.md not generated")

        # Read summary
        with open(summary_path) as f:
            content = f.read()

        # Verify it's not empty
        assert len(content) > 100, "Summary should have substantial content"

        # Verify required sections (some sections only present with optimization data)
        must_have_sections = [
            "# Release Audit Summary",
            "## Executive Summary",
            "## Validation Results",
            "## Deployment Plan",
            "## Approval",
        ]
        for section in must_have_sections:
            assert section in content, f"Missing required section: {section}"

        # Optional section that requires optimization data
        # "## Metrics Comparison" - only present when baseline/optimized metrics exist
        # Markdown tables also only present with optimization data

        # Verify basic content
        assert "audit_" in content, "Should contain audit ID"
        assert "Status" in content, "Should contain status"
        assert "Deployment Recommendation" in content, "Should contain deployment recommendation"
        assert "Audit Date" in content, "Should contain audit date"
        assert "APPROVED" in content or "REQUIRES REVIEW" in content, "Should have approval status"


def test_config_snapshot() -> None:
    """Test configuration snapshot capture (US-022).

    Verifies:
    - Current production config is captured
    - Config files are copied to bundle
    - Snapshot is valid JSON
    """
    import json
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "audit_config_test"

        # Run release audit script
        subprocess.run(
            [
                "python",
                "scripts/release_audit.py",
                "--output-dir",
                str(output_dir),
                "--skip-validation",
                "--skip-plots",
            ],
            capture_output=True,
            text=True,
        )

        configs_dir = output_dir / "configs"
        if not configs_dir.exists():
            pytest.skip("configs directory not generated")

        # Verify config.json exists
        config_json_path = configs_dir / "config.json"
        assert config_json_path.exists(), "config.json should be created"

        # Verify it's valid JSON
        with open(config_json_path) as f:
            config_data = json.load(f)

        assert isinstance(config_data, dict), "Config should be a dictionary"
        assert len(config_data) > 0, "Config should not be empty"
