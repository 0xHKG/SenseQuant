"""Integration tests for backtester telemetry capture."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import pytz

from src.domain.types import BacktestConfig, Bar
from src.services.accuracy_analyzer import AccuracyAnalyzer
from src.services.backtester import Backtester


@pytest.fixture
def mock_breeze_client() -> MagicMock:
    """Create mock Breeze client."""
    client = MagicMock()
    client.authenticate = MagicMock()
    client.historical_bars = MagicMock()
    return client


@pytest.fixture
def sample_swing_bars() -> list[Bar]:
    """Generate sample daily bars with bullish crossover pattern."""
    ist = pytz.timezone("Asia/Kolkata")
    base_date = pd.Timestamp("2024-01-01", tz=ist)
    bars = []

    # Create 120 days of data with clear patterns
    for i in range(120):
        ts = base_date + pd.Timedelta(days=i)

        if i < 80:
            # Downtrend
            close = 150.0 - (i * 0.5)
        elif i < 100:
            # Rally (will trigger LONG entry)
            close = 110.0 + ((i - 79) * 3.0)
        else:
            # Consolidation (will trigger exit)
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


def test_backtest_telemetry_capture(
    mock_breeze_client: MagicMock,
    sample_swing_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test that telemetry is captured during backtest."""
    mock_breeze_client.historical_bars.return_value = sample_swing_bars

    config = BacktestConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        initial_capital=1000000.0,
        data_source="breeze",
        random_seed=42,
    )

    # Create telemetry directory
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    from src.app.config import settings

    # Run backtest with telemetry enabled
    backtester = Backtester(
        config=config,
        client=mock_breeze_client,
        settings=settings,
        enable_telemetry=True,
        telemetry_dir=telemetry_dir,
    )
    result = backtester.run()

    # Verify backtest completed successfully
    assert result.config == config
    assert "total_trades" in result.metrics

    # Verify telemetry files were created
    telemetry_files = list(telemetry_dir.glob("predictions_*.csv*"))
    assert len(telemetry_files) > 0, "No telemetry files were created"

    # Load and verify telemetry traces
    analyzer = AccuracyAnalyzer()
    traces = []
    for telemetry_file in telemetry_files:
        try:
            file_traces = analyzer.load_traces(telemetry_file)
            traces.extend(file_traces)
        except Exception as e:
            pytest.fail(f"Failed to load telemetry file {telemetry_file}: {e}")

    # If trades occurred, verify traces were captured
    if result.metrics["total_trades"] > 0:
        assert len(traces) > 0, "Expected telemetry traces but none were found"

        # Verify trace structure
        for trace in traces:
            assert trace.symbol == "TEST"
            assert trace.strategy == "swing"
            assert trace.predicted_direction in ["LONG", "SHORT"]
            assert trace.actual_direction in ["LONG", "SHORT", "NOOP"]
            assert 0.0 <= trace.predicted_confidence <= 1.0
            assert trace.entry_price > 0
            assert trace.exit_price > 0
            assert trace.holding_period_minutes >= 0
            assert isinstance(trace.metadata, dict)
            assert "exit_reason" in trace.metadata


def test_backtest_telemetry_disabled(
    mock_breeze_client: MagicMock,
    sample_swing_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test that telemetry is NOT captured when disabled."""
    mock_breeze_client.historical_bars.return_value = sample_swing_bars

    config = BacktestConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        initial_capital=1000000.0,
        data_source="breeze",
        random_seed=42,
    )

    # Create telemetry directory (but won't be used)
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    from src.app.config import settings

    # Run backtest with telemetry DISABLED
    backtester = Backtester(
        config=config,
        client=mock_breeze_client,
        settings=settings,
        enable_telemetry=False,  # Explicitly disabled
        telemetry_dir=telemetry_dir,
    )
    result = backtester.run()

    # Verify backtest completed successfully
    assert result.config == config
    assert "total_trades" in result.metrics

    # Verify NO telemetry files were created
    telemetry_files = list(telemetry_dir.glob("predictions_*.csv*"))
    assert len(telemetry_files) == 0, "Telemetry files were created when disabled"


def test_backtest_telemetry_sample_rate(
    mock_breeze_client: MagicMock,
    sample_swing_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test that telemetry respects sample rate."""
    mock_breeze_client.historical_bars.return_value = sample_swing_bars

    config = BacktestConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        initial_capital=1000000.0,
        data_source="breeze",
        random_seed=42,
    )

    # Create telemetry directory
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    from src.app.config import Settings

    # Create a copy of settings and override telemetry parameters
    test_settings = Settings()
    # Directly set attributes to bypass any default overrides
    test_settings.telemetry_sample_rate = 0.0
    test_settings.telemetry_storage_path = str(telemetry_dir)
    test_settings.telemetry_compression = False
    test_settings.telemetry_buffer_size = 10
    test_settings.telemetry_max_file_size_mb = 10

    # Run backtest with telemetry enabled but 0% sample rate
    backtester = Backtester(
        config=config,
        client=mock_breeze_client,
        settings=test_settings,
        enable_telemetry=True,
        telemetry_dir=telemetry_dir,
    )
    result = backtester.run()

    # Verify backtest completed successfully
    assert result.config == config

    # With 0% sample rate, no traces should be captured
    # (even if files are created, they should be empty or have only headers)
    analyzer = AccuracyAnalyzer()
    telemetry_files = list(telemetry_dir.glob("predictions_*.csv*"))

    total_traces = 0
    for telemetry_file in telemetry_files:
        try:
            traces = analyzer.load_traces(telemetry_file)
            total_traces += len(traces)
        except Exception:
            # Empty file or no data is expected
            pass

    # With 0% sample rate, we expect 0 traces
    assert total_traces == 0, f"Expected 0 traces with 0% sample rate, got {total_traces}"


def test_backtest_telemetry_backward_compatibility(
    mock_breeze_client: MagicMock,
    sample_swing_bars: list[Bar],
) -> None:
    """Test that telemetry parameters are optional and backward compatible."""
    mock_breeze_client.historical_bars.return_value = sample_swing_bars

    config = BacktestConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        initial_capital=1000000.0,
        data_source="breeze",
        random_seed=42,
    )

    from src.app.config import settings

    # Run backtest WITHOUT telemetry parameters (default behavior)
    backtester = Backtester(
        config=config,
        client=mock_breeze_client,
        settings=settings,
    )
    result = backtester.run()

    # Verify backtest completes successfully
    assert result.config == config
    assert "total_trades" in result.metrics
    assert result.metrics["total_trades"] >= 0


def test_backtest_telemetry_trace_contents(
    mock_breeze_client: MagicMock,
    sample_swing_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test that telemetry traces contain expected data."""
    mock_breeze_client.historical_bars.return_value = sample_swing_bars

    config = BacktestConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        initial_capital=1000000.0,
        data_source="breeze",
        random_seed=42,
    )

    # Create telemetry directory
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    from src.app.config import Settings

    # Create settings with 100% sample rate to ensure traces are captured
    test_settings = Settings()
    test_settings.telemetry_sample_rate = 1.0
    test_settings.telemetry_storage_path = str(telemetry_dir)
    test_settings.telemetry_compression = False
    test_settings.telemetry_buffer_size = 1  # Flush immediately
    test_settings.telemetry_max_file_size_mb = 10

    # Run backtest with telemetry enabled
    backtester = Backtester(
        config=config,
        client=mock_breeze_client,
        settings=test_settings,
        enable_telemetry=True,
        telemetry_dir=telemetry_dir,
    )
    result = backtester.run()

    # Skip test if no trades occurred
    if result.metrics["total_trades"] == 0:
        pytest.skip("No trades occurred in backtest")

    # Load telemetry traces
    analyzer = AccuracyAnalyzer()
    telemetry_files = list(telemetry_dir.glob("predictions_*.csv*"))
    assert len(telemetry_files) > 0

    traces = []
    for telemetry_file in telemetry_files:
        traces.extend(analyzer.load_traces(telemetry_file))

    assert len(traces) > 0

    # Verify trace contents
    for trace in traces:
        # Check required fields
        assert trace.symbol is not None
        assert trace.strategy == "swing"
        assert trace.predicted_direction in ["LONG", "SHORT"]
        assert trace.actual_direction in ["LONG", "SHORT", "NOOP"]
        assert trace.entry_price > 0
        assert trace.exit_price > 0

        # Check metadata fields
        assert "exit_reason" in trace.metadata
        assert "entry_fees" in trace.metadata
        assert "exit_fees" in trace.metadata
        assert "total_fees" in trace.metadata
        assert "position_value" in trace.metadata
        assert "gross_pnl" in trace.metadata

        # Verify actual_direction logic
        if trace.realized_return_pct > 0.5:
            assert trace.actual_direction == "LONG"
        elif trace.realized_return_pct < -0.5:
            assert trace.actual_direction == "SHORT"
        else:
            assert trace.actual_direction == "NOOP"
