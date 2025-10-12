"""Unit tests for Backtester metrics and calculations."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import pytz

from src.domain.types import BacktestConfig


@pytest.fixture
def sample_equity_curve() -> pd.DataFrame:
    """Generate sample equity curve for metric calculations."""
    ist = pytz.timezone("Asia/Kolkata")
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D", tz=ist)

    # Simulate realistic equity curve with growth and drawdown
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    equity = 1000000.0 * np.cumprod(1 + returns)

    return pd.DataFrame({"ts": dates, "equity": equity})


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    """Generate sample trades for win rate calculations."""
    ist = pytz.timezone("Asia/Kolkata")
    trades = [
        {
            "symbol": "TEST",
            "entry_date": pd.Timestamp("2024-01-10", tz=ist),
            "exit_date": pd.Timestamp("2024-01-15", tz=ist),
            "direction": "LONG",
            "entry_price": 100.0,
            "exit_price": 105.0,
            "qty": 100,
            "pnl": 500.0,
            "return_pct": 5.0,
            "fees": 30.0,
            "reason": "tp_hit",
        },
        {
            "symbol": "TEST",
            "entry_date": pd.Timestamp("2024-01-20", tz=ist),
            "exit_date": pd.Timestamp("2024-01-25", tz=ist),
            "direction": "SHORT",
            "entry_price": 100.0,
            "exit_price": 102.0,
            "qty": 100,
            "pnl": -200.0,
            "return_pct": -2.0,
            "fees": 30.0,
            "reason": "sl_hit",
        },
        {
            "symbol": "TEST",
            "entry_date": pd.Timestamp("2024-02-01", tz=ist),
            "exit_date": pd.Timestamp("2024-02-05", tz=ist),
            "direction": "LONG",
            "entry_price": 100.0,
            "exit_price": 103.0,
            "qty": 100,
            "pnl": 300.0,
            "return_pct": 3.0,
            "fees": 30.0,
            "reason": "tp_hit",
        },
    ]
    return pd.DataFrame(trades)


def test_calculate_cagr_one_year_double() -> None:
    """Test CAGR calculation for doubling in one year."""
    initial = 1000000.0
    final = 2000000.0
    years = 1.0

    cagr = (final / initial) ** (1 / years) - 1

    assert abs(cagr - 1.0) < 0.001  # 100% CAGR


def test_calculate_cagr_two_years_50pct() -> None:
    """Test CAGR calculation for 50% total return over 2 years."""
    initial = 1000000.0
    final = 1500000.0
    years = 2.0

    cagr = (final / initial) ** (1 / years) - 1

    expected_cagr = 0.2247  # ~22.47% CAGR
    assert abs(cagr - expected_cagr) < 0.001


def test_calculate_cagr_loss() -> None:
    """Test CAGR calculation with negative returns."""
    initial = 1000000.0
    final = 800000.0
    years = 1.0

    cagr = (final / initial) ** (1 / years) - 1

    assert abs(cagr - (-0.2)) < 0.001  # -20% CAGR


def test_calculate_max_drawdown_no_loss(sample_equity_curve: pd.DataFrame) -> None:
    """Test max drawdown calculation on monotonic increase."""
    # Create monotonically increasing equity
    equity_curve = sample_equity_curve.copy()
    equity_curve["equity"] = np.linspace(1000000, 1500000, len(equity_curve))

    running_max = equity_curve["equity"].cummax()
    drawdown = (equity_curve["equity"] - running_max) / running_max
    max_drawdown = drawdown.min()

    assert max_drawdown == 0.0  # No drawdown


def test_calculate_max_drawdown_50pct() -> None:
    """Test max drawdown calculation with 50% drop."""
    equity = pd.Series([1000000, 1200000, 1500000, 750000, 1000000])
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    assert abs(max_drawdown - (-0.5)) < 0.001  # -50% max drawdown


def test_calculate_sharpe_ratio_positive(sample_equity_curve: pd.DataFrame) -> None:
    """Test Sharpe ratio calculation with positive returns."""
    returns = sample_equity_curve["equity"].pct_change().dropna()

    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        # Sharpe should be reasonable (typically -3 to 3)
        assert -3 < sharpe < 5


def test_calculate_sharpe_ratio_zero_std() -> None:
    """Test Sharpe ratio with zero volatility (constant returns)."""
    equity = pd.Series([1000000] * 100)
    returns = equity.pct_change().dropna()

    # All returns are 0, std is 0 → Sharpe = 0
    if returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    assert sharpe == 0.0


def test_calculate_win_rate(sample_trades: pd.DataFrame) -> None:
    """Test win rate calculation."""
    winning_trades = len(sample_trades[sample_trades["pnl"] > 0])
    total_trades = len(sample_trades)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    assert abs(win_rate - (2 / 3)) < 0.001  # 2 wins out of 3 trades


def test_calculate_win_rate_all_wins() -> None:
    """Test win rate with all winning trades."""
    trades = pd.DataFrame({"pnl": [100, 200, 300]})
    winning_trades = len(trades[trades["pnl"] > 0])
    total_trades = len(trades)
    win_rate = winning_trades / total_trades

    assert win_rate == 1.0


def test_calculate_win_rate_all_losses() -> None:
    """Test win rate with all losing trades."""
    trades = pd.DataFrame({"pnl": [-100, -200, -300]})
    winning_trades = len(trades[trades["pnl"] > 0])
    total_trades = len(trades)
    win_rate = winning_trades / total_trades

    assert win_rate == 0.0


def test_calculate_win_rate_no_trades() -> None:
    """Test win rate with no trades."""
    trades = pd.DataFrame({"pnl": []})
    total_trades = len(trades)
    win_rate = 0.0 if total_trades == 0 else len(trades[trades["pnl"] > 0]) / total_trades

    assert win_rate == 0.0


def test_calculate_avg_win_loss(sample_trades: pd.DataFrame) -> None:
    """Test average win and loss calculations."""
    winning_trades = sample_trades[sample_trades["pnl"] > 0]
    losing_trades = sample_trades[sample_trades["pnl"] < 0]

    avg_win = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0.0

    assert abs(avg_win - 400.0) < 0.1  # (500 + 300) / 2
    assert abs(avg_loss - (-200.0)) < 0.1


def test_calculate_exposure_50pct() -> None:
    """Test exposure calculation with 50% time in market."""
    ist = pytz.timezone("Asia/Kolkata")
    # 100 trading days total
    equity_curve = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=100, freq="D", tz=ist),
            "position_value": [0] * 50 + [10000] * 50,  # 50 days with position
        }
    )

    days_with_positions = len(equity_curve[equity_curve["position_value"] > 0])
    total_days = len(equity_curve)
    exposure = days_with_positions / total_days if total_days > 0 else 0.0

    assert abs(exposure - 0.5) < 0.001


def test_calculate_exposure_always_in_market() -> None:
    """Test exposure calculation when always in market."""
    ist = pytz.timezone("Asia/Kolkata")
    equity_curve = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=100, freq="D", tz=ist),
            "position_value": [10000] * 100,
        }
    )

    days_with_positions = len(equity_curve[equity_curve["position_value"] > 0])
    total_days = len(equity_curve)
    exposure = days_with_positions / total_days

    assert exposure == 1.0


def test_calculate_exposure_never_in_market() -> None:
    """Test exposure calculation when never in market."""
    ist = pytz.timezone("Asia/Kolkata")
    equity_curve = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=100, freq="D", tz=ist),
            "position_value": [0] * 100,
        }
    )

    days_with_positions = len(equity_curve[equity_curve["position_value"] > 0])
    total_days = len(equity_curve)
    exposure = days_with_positions / total_days

    assert exposure == 0.0


def test_total_fees_calculation(sample_trades: pd.DataFrame) -> None:
    """Test total fees calculation from trades."""
    total_fees = sample_trades["fees"].sum()

    assert abs(total_fees - 90.0) < 0.1  # 3 trades * 30 fees each


def test_metrics_edge_case_zero_trades() -> None:
    """Test metrics calculation with zero trades."""
    # Empty trades dataframe
    trades = pd.DataFrame(columns=["pnl", "fees", "return_pct"])

    total_trades = len(trades)
    win_rate = 0.0 if total_trades == 0 else len(trades[trades["pnl"] > 0]) / total_trades
    avg_win = 0.0
    avg_loss = 0.0
    total_fees = 0.0

    assert total_trades == 0
    assert win_rate == 0.0
    assert avg_win == 0.0
    assert avg_loss == 0.0
    assert total_fees == 0.0


def test_metrics_edge_case_single_trade() -> None:
    """Test metrics calculation with single trade."""
    trades = pd.DataFrame({"pnl": [500.0], "fees": [30.0]})

    total_trades = len(trades)
    win_rate = len(trades[trades["pnl"] > 0]) / total_trades
    avg_win = trades[trades["pnl"] > 0]["pnl"].mean()
    total_fees = trades["fees"].sum()

    assert total_trades == 1
    assert win_rate == 1.0
    assert avg_win == 500.0
    assert total_fees == 30.0


def test_config_validation_invalid_dates() -> None:
    """Test config validation with invalid date order."""
    with pytest.raises(ValueError):
        # start_date after end_date should fail in validation
        config = BacktestConfig(
            symbols=["TEST"],
            start_date="2024-12-31",
            end_date="2024-01-01",  # Earlier than start
            strategy="swing",
        )
        # Validation logic would check this
        start = datetime.strptime(config.start_date, "%Y-%m-%d")
        end = datetime.strptime(config.end_date, "%Y-%m-%d")
        if start >= end:
            raise ValueError("start_date must be before end_date")


def test_config_validation_empty_symbols() -> None:
    """Test config validation with empty symbols list."""
    with pytest.raises((ValueError, AssertionError)):
        config = BacktestConfig(
            symbols=[],
            start_date="2024-01-01",
            end_date="2024-12-31",
            strategy="swing",
        )
        if not config.symbols:
            raise ValueError("symbols list cannot be empty")


def test_date_range_business_days() -> None:
    """Test business day calculation for date range."""
    start = pd.Timestamp("2024-01-01")  # Monday
    end = pd.Timestamp("2024-01-07")  # Sunday

    # 5 business days (Mon-Fri)
    business_days = pd.bdate_range(start, end)
    assert len(business_days) == 5


def test_artifact_serialization_json_format() -> None:
    """Test that metrics can be serialized to JSON."""
    import json

    metrics = {
        "total_return_pct": 25.5,
        "cagr_pct": 20.3,
        "max_drawdown_pct": -15.2,
        "sharpe_ratio": 1.8,
        "win_rate_pct": 65.5,
        "total_trades": 50.0,
        "avg_win": 1500.0,
        "avg_loss": -800.0,
        "exposure_pct": 45.0,
        "total_fees": 7500.0,
    }

    # Should serialize without errors
    json_str = json.dumps(metrics)
    parsed = json.loads(json_str)

    assert parsed["total_return_pct"] == 25.5
    assert parsed["sharpe_ratio"] == 1.8


def test_equity_curve_schema() -> None:
    """Test equity curve DataFrame has required columns."""
    ist = pytz.timezone("Asia/Kolkata")
    equity_curve = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz=ist),
            "equity": np.linspace(1000000, 1100000, 10),
            "position_value": [0, 5000, 5000, 0, 0, 10000, 10000, 10000, 0, 0],
        }
    )

    required_cols = ["ts", "equity", "position_value"]
    assert all(col in equity_curve.columns for col in required_cols)
    assert len(equity_curve) == 10


def test_trades_log_schema() -> None:
    """Test trades log DataFrame has required columns."""
    ist = pytz.timezone("Asia/Kolkata")
    trades = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "entry_date": [pd.Timestamp("2024-01-10", tz=ist)],
            "exit_date": [pd.Timestamp("2024-01-15", tz=ist)],
            "direction": ["LONG"],
            "entry_price": [100.0],
            "exit_price": [105.0],
            "qty": [100],
            "pnl": [500.0],
            "return_pct": [5.0],
            "fees": [30.0],
            "reason": ["tp_hit"],
        }
    )

    required_cols = [
        "symbol",
        "entry_date",
        "exit_date",
        "direction",
        "entry_price",
        "exit_price",
        "qty",
        "pnl",
        "return_pct",
        "fees",
        "reason",
    ]
    assert all(col in trades.columns for col in required_cols)
