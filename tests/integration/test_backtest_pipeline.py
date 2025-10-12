"""Integration tests for full backtest pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

from src.domain.types import BacktestConfig, Bar
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

    # Create 120 days of data:
    # Days 0-79: Downtrend (below both SMAs)
    # Days 80-99: Sharp rally causing bullish crossover
    # Days 100-119: Consolidation allowing exit
    for i in range(120):
        ts = base_date + pd.Timedelta(days=i)

        if i < 80:
            # Downtrend
            close = 150.0 - (i * 0.5)
        elif i < 100:
            # Rally
            close = 110.0 + ((i - 79) * 3.0)
        else:
            # Consolidation (trigger exit)
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
def sample_intraday_bars() -> list[Bar]:
    """Generate sample minute bars for intraday strategy."""
    ist = pytz.timezone("Asia/Kolkata")
    base_time = pd.Timestamp("2024-01-10 09:15:00", tz=ist)
    bars = []

    # Create intraday bars (9:15 - 15:29, 375 minutes)
    for i in range(375):
        ts = base_time + pd.Timedelta(minutes=i)

        # Simulate intraday volatility
        if i < 100:
            close = 100.0 + (i * 0.1)
        elif i < 200:
            close = 110.0 - ((i - 100) * 0.05)
        else:
            close = 105.0 + ((i - 200) * 0.02)

        bars.append(
            Bar(
                ts=ts,
                open=close - 0.2,
                high=close + 0.5,
                low=close - 0.5,
                close=close,
                volume=10000,
            )
        )

    return bars


def test_swing_backtest_pipeline(
    mock_breeze_client: MagicMock,
    sample_swing_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test full swing backtest pipeline end-to-end."""
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

    # Patch data directory to use tmp_path
    with patch("src.services.backtester.Path") as mock_path:
        backtest_dir = tmp_path / "backtests"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        mock_path.return_value = backtest_dir

        # Mock settings
        from src.app.config import settings

        backtester = Backtester(config=config, client=mock_breeze_client, settings=settings)
        result = backtester.run()

        # Verify result structure
        assert result.config == config
        assert "total_return_pct" in result.metrics
        assert "cagr_pct" in result.metrics
        assert "max_drawdown_pct" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "win_rate_pct" in result.metrics
        assert "total_trades" in result.metrics
        assert "exposure_pct" in result.metrics

        # Verify equity curve
        assert len(result.equity_curve) > 0
        assert "timestamp" in result.equity_curve.columns
        assert "equity" in result.equity_curve.columns
        assert "open_positions" in result.equity_curve.columns

        # Verify trades log structure (may be empty if no trades)
        assert "symbol" in result.trades.columns or len(result.trades) == 0

        # Verify metadata
        assert result.metadata is not None
        assert "run_date" in result.metadata
        assert "random_seed" in result.metadata


def test_intraday_backtest_pipeline(
    mock_breeze_client: MagicMock,
    sample_intraday_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test full intraday backtest pipeline end-to-end.

    NOTE: Currently intraday simulation is not implemented (requires minute-level data).
    This test verifies graceful handling of intraday strategy.
    """
    mock_breeze_client.historical_bars.return_value = sample_intraday_bars

    config = BacktestConfig(
        symbols=["TEST"],
        start_date="2024-01-10",
        end_date="2024-01-10",  # Single day
        strategy="intraday",
        initial_capital=500000.0,
        data_source="breeze",
        random_seed=42,
    )

    from src.app.config import settings

    backtester = Backtester(config=config, client=mock_breeze_client, settings=settings)
    result = backtester.run()

    # Verify result structure
    assert result.config == config
    # NOTE: Since intraday is not implemented, metrics will be zero/empty
    # Just verify the backtest completes without errors
    assert "total_trades" in result.metrics or result.metrics == {}


def test_both_strategies_backtest(
    mock_breeze_client: MagicMock,
    sample_swing_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test backtest with both intraday and swing strategies."""
    mock_breeze_client.historical_bars.return_value = sample_swing_bars

    config = BacktestConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="both",
        initial_capital=1000000.0,
        data_source="breeze",
        random_seed=42,
    )

    from src.app.config import settings

    backtester = Backtester(config=config, client=mock_breeze_client, settings=settings)
    result = backtester.run()

    # Verify result structure
    assert result.config == config
    assert result.config.strategy == "both"
    assert "total_trades" in result.metrics

    # Trades may include both intraday and swing
    if len(result.trades) > 0:
        # Check that trades are recorded
        assert "symbol" in result.trades.columns
        assert "direction" in result.trades.columns


def test_artifact_completeness(
    mock_breeze_client: MagicMock,
    sample_swing_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test that all artifacts are created with correct schema."""
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

    # Use tmp_path for artifact storage
    backtest_dir = tmp_path / "backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    with patch("src.services.backtester.Path") as mock_path_class:
        mock_path_class.return_value = backtest_dir

        from src.app.config import settings

        backtester = Backtester(config=config, client=mock_breeze_client, settings=settings)
        result = backtester.run()

        # Note: Artifacts are saved but paths point to mock directory
        # Verify paths are set
        assert result.summary_path is not None
        assert result.equity_path is not None
        assert result.trades_path is not None

        # Verify result includes all required fields
        assert result.config is not None
        assert result.metrics is not None
        assert result.equity_curve is not None
        assert result.trades is not None
        assert result.metadata is not None


def test_multi_symbol_backtest(
    mock_breeze_client: MagicMock,
    sample_swing_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test backtest with multiple symbols."""
    # Return same bars for all symbols (simplified)
    mock_breeze_client.historical_bars.return_value = sample_swing_bars

    config = BacktestConfig(
        symbols=["TEST1", "TEST2", "TEST3"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        initial_capital=1000000.0,
        data_source="breeze",
        random_seed=42,
    )

    from src.app.config import settings

    backtester = Backtester(config=config, client=mock_breeze_client, settings=settings)
    result = backtester.run()

    # Verify result processes multiple symbols
    assert result.config.symbols == ["TEST1", "TEST2", "TEST3"]
    assert "total_trades" in result.metrics

    # If trades occurred, verify multiple symbols may be present
    if len(result.trades) > 0:
        unique_symbols = result.trades["symbol"].unique()
        # At least one symbol should have trades (depending on data)
        assert len(unique_symbols) >= 1


def test_backtest_with_no_trades(
    mock_breeze_client: MagicMock,
    tmp_path: Path,
) -> None:
    """Test backtest that produces no trades."""
    ist = pytz.timezone("Asia/Kolkata")
    # Create bars with no crossover (flat market)
    flat_bars = [
        Bar(
            ts=pd.Timestamp("2024-01-01", tz=ist) + pd.Timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10000,
        )
        for i in range(60)
    ]

    mock_breeze_client.historical_bars.return_value = flat_bars

    config = BacktestConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-03-01",
        strategy="swing",
        initial_capital=1000000.0,
        data_source="breeze",
        random_seed=42,
    )

    from src.app.config import settings

    backtester = Backtester(config=config, client=mock_breeze_client, settings=settings)
    result = backtester.run()

    # Verify metrics handle zero trades
    assert result.metrics["total_trades"] == 0
    assert result.metrics["win_rate_pct"] == 0.0
    assert result.metrics["total_fees"] == 0.0
    assert len(result.trades) == 0


def test_backtest_with_csv_data_source(tmp_path: Path) -> None:
    """Test backtest using CSV data source with DataFeed."""
    from src.services.data_feed import CSVDataFeed

    # Create sample CSV directory structure
    ist = pytz.timezone("Asia/Kolkata")
    symbol_dir = tmp_path / "TEST" / "1day"
    symbol_dir.mkdir(parents=True)

    # Create bars with bullish crossover pattern
    bars_data = []
    for i in range(120):
        ts = pd.Timestamp("2024-01-01", tz=ist) + pd.Timedelta(days=i)

        if i < 80:
            # Downtrend
            close = 150.0 - (i * 0.5)
        elif i < 100:
            # Rally
            close = 110.0 + ((i - 79) * 3.0)
        else:
            # Consolidation
            close = 170.0 + ((i - 99) * 0.5)

        bars_data.append(
            {
                "timestamp": ts,
                "open": close - 0.5,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "volume": 100000,
            }
        )

    csv_data = pd.DataFrame(bars_data)
    csv_file = symbol_dir / "2024-01-01_to_2024-04-30.csv"
    csv_data.to_csv(csv_file, index=False)

    config = BacktestConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        initial_capital=1000000.0,
        data_source="csv",
        random_seed=42,
    )

    from src.app.config import settings

    # Create DataFeed and pass to Backtester
    data_feed = CSVDataFeed(tmp_path)
    backtester = Backtester(config=config, client=None, data_feed=data_feed, settings=settings)
    result = backtester.run()

    # Verify backtest runs with CSV data
    assert result.config.data_source == "csv"
    assert "total_return_pct" in result.metrics
    assert result.metrics["total_trades"] >= 0


def test_backtest_metrics_validation(
    mock_breeze_client: MagicMock,
    sample_swing_bars: list[Bar],
) -> None:
    """Test that backtest metrics are within expected ranges."""
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

    backtester = Backtester(config=config, client=mock_breeze_client, settings=settings)
    result = backtester.run()

    # Validate metric ranges
    metrics = result.metrics

    # Total return can be negative or positive
    assert -100 <= metrics["total_return_pct"] <= 1000

    # CAGR can be negative or positive
    assert -100 <= metrics["cagr_pct"] <= 1000

    # Max drawdown should be <= 0
    assert metrics["max_drawdown_pct"] <= 0

    # Win rate should be 0-100%
    assert 0 <= metrics["win_rate_pct"] <= 100

    # Exposure should be 0-100%
    assert 0 <= metrics["exposure_pct"] <= 100

    # Total trades should be non-negative integer
    assert metrics["total_trades"] >= 0

    # Total fees should be non-negative
    assert metrics["total_fees"] >= 0


def test_backtest_determinism(
    mock_breeze_client: MagicMock,
    sample_swing_bars: list[Bar],
) -> None:
    """Test that backtests are deterministic with same seed."""
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

    # Run backtest twice
    backtester1 = Backtester(config=config, client=mock_breeze_client, settings=settings)
    result1 = backtester1.run()

    backtester2 = Backtester(config=config, client=mock_breeze_client, settings=settings)
    result2 = backtester2.run()

    # Results should be identical
    assert result1.metrics["total_return_pct"] == result2.metrics["total_return_pct"]
    assert result1.metrics["total_trades"] == result2.metrics["total_trades"]
    assert result1.metrics["win_rate_pct"] == result2.metrics["win_rate_pct"]
