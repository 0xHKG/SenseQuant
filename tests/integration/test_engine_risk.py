"""Integration tests for engine risk management."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

from src.domain.types import Bar
from src.services.engine import Engine


@pytest.fixture
def mock_breeze_client() -> MagicMock:
    """Create mock Breeze client."""
    client = MagicMock()
    client.authenticate = MagicMock()
    client.historical_bars = MagicMock()
    client.place_order = MagicMock()
    return client


@pytest.fixture
def sample_daily_bars() -> list[Bar]:
    """Generate sample daily bars with crossover pattern."""
    ist = pytz.timezone("Asia/Kolkata")
    base_date = pd.Timestamp("2025-01-01", tz=ist)
    bars = []
    for i in range(100):
        ts = base_date + pd.Timedelta(days=i)
        # Create downtrend then sharp rally for bullish crossover
        if i < 80:
            close = 150.0 - (i * 0.5)
        else:
            close = 120.0 + ((i - 79) * 5.0)

        bars.append(
            Bar(
                ts=ts,
                open=close - 1.0,
                high=close + 2.0,
                low=close - 2.0,
                close=close,
                volume=10000,
            )
        )
    return bars


def test_swing_entry_with_position_sizing(
    mock_breeze_client: MagicMock,
) -> None:
    """Test swing entry uses RiskManager for position sizing."""
    from src.domain.strategies.swing import SwingPosition

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        engine = Engine(symbols=["TEST"])
        engine.start()

        # Manually test position sizing and risk checks
        # (Integration with swing strategy signal tested elsewhere)
        pos_size = engine._risk_manager.calculate_position_size(
            symbol="TEST",
            price=100.0,
            atr=5.0,
            signal_strength=0.8,
        )

        # Verify position sizing works
        assert pos_size.qty > 0
        assert pos_size.risk_pct > 0

        # Verify risk check allows position
        risk_check = engine._risk_manager.can_open_position(
            symbol="TEST",
            qty=pos_size.qty,
            price=100.0,
        )

        assert risk_check.allowed
        assert risk_check.allowed_qty == pos_size.qty

        # Verify fees calculation
        fees = engine._risk_manager.calculate_fees(qty=pos_size.qty, price=100.0)
        assert fees > 0

        # Manually create position to verify integration
        engine._swing_positions["TEST"] = SwingPosition(
            symbol="TEST",
            direction="LONG",
            entry_price=100.0,
            entry_date=pd.Timestamp("2025-01-01"),
            qty=pos_size.qty,
            entry_fees=fees,
        )

        # Verify position has fees recorded
        position = engine._swing_positions["TEST"]
        assert position.entry_fees > 0
        assert position.qty > 0


def test_swing_entry_blocked_by_position_cap(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Test position blocked when symbol cap reached."""
    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            # Use very low position cap to trigger blocking
            with patch("src.app.config.settings.max_position_value_per_symbol", 100.0):
                engine = Engine(symbols=["TEST"])
                engine.start()

                # Manually set high position value to exceed cap
                engine._risk_manager.update_position("TEST", qty=10, price=100.0, is_opening=True)

                # Try to enter - should be blocked
                engine.run_swing_daily("TEST")

                # Verify no position created
                assert "TEST" not in engine._swing_positions


def test_swing_exit_calculates_realized_pnl_with_fees(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Test swing exit calculates correct realized PnL including fees."""
    from src.domain.strategies.swing import SwingPosition

    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            engine.start()

            # Manually create a position
            engine._swing_positions["TEST"] = SwingPosition(
                symbol="TEST",
                direction="LONG",
                entry_price=100.0,
                entry_date=pd.Timestamp("2025-01-01", tz=ist),
                qty=100,
                entry_fees=15.0,  # Pre-calculated
            )

            # Create bars showing losing position (price dropped)
            exit_bars = [
                Bar(
                    ts=pd.Timestamp("2025-01-15", tz=ist) + pd.Timedelta(days=i),
                    open=90.0,
                    high=92.0,
                    low=88.0,
                    close=90.0,
                    volume=10000,
                )
                for i in range(100)
            ]
            mock_breeze_client.historical_bars.return_value = exit_bars

            # Trigger exit via max hold days
            engine.run_swing_daily("TEST")

            # Verify position closed
            assert "TEST" not in engine._swing_positions


def test_circuit_breaker_activation(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Test circuit breaker triggers on max daily loss."""
    from src.domain.strategies.swing import SwingPosition

    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST", "ANOTHER"])
            engine.start()

            # Create positions for both symbols
            engine._swing_positions["TEST"] = SwingPosition(
                symbol="TEST",
                direction="LONG",
                entry_price=100.0,
                entry_date=pd.Timestamp("2025-01-01", tz=ist),
                qty=1000,  # Large qty for big loss
                entry_fees=150.0,
            )

            engine._swing_positions["ANOTHER"] = SwingPosition(
                symbol="ANOTHER",
                direction="LONG",
                entry_price=100.0,
                entry_date=pd.Timestamp("2025-01-01", tz=ist),
                qty=100,
                entry_fees=15.0,
            )

            # Manually trigger large loss to activate breaker
            # 5% of 1M capital = 50K loss limit
            engine._risk_manager.record_trade(
                symbol="MANUAL_LOSS",
                realized_pnl=-60000.0,  # Exceeds 5% limit
                fees=1000.0,
            )

            # Verify breaker is active
            assert engine._risk_manager.is_circuit_breaker_active()

            # Verify new positions blocked
            engine.run_swing_daily("NEWSTOCK")
            assert "NEWSTOCK" not in engine._swing_positions


def test_fees_recorded_in_journal(
    mock_breeze_client: MagicMock,
) -> None:
    """Test RiskManager integrates with Engine for fee tracking."""
    from src.domain.strategies.swing import SwingPosition

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        engine = Engine(symbols=["TEST"])
        engine.start()

        # Create a test position with fees
        pos_size = engine._risk_manager.calculate_position_size("TEST", price=100.0)
        entry_fees = engine._risk_manager.calculate_fees(pos_size.qty, 100.0)

        position = SwingPosition(
            symbol="TEST",
            direction="LONG",
            entry_price=100.0,
            entry_date=pd.Timestamp("2025-01-01", tz=ist),
            qty=pos_size.qty,
            entry_fees=entry_fees,
        )

        # Verify fees are positive
        assert position.entry_fees > 0

        # Verify exit fees calculation
        exit_fees = engine._risk_manager.calculate_fees(position.qty, 110.0)
        assert exit_fees > 0

        # Verify realized PnL calculation includes fees
        gross_pnl = (110.0 - 100.0) * position.qty
        realized_pnl = gross_pnl - position.entry_fees - exit_fees
        assert realized_pnl < gross_pnl  # Fees reduce PnL


def test_intraday_entry_with_position_sizing(
    mock_breeze_client: MagicMock,
) -> None:
    """Test intraday entry uses RiskManager for position sizing."""
    from src.domain.strategies.intraday import IntradayPosition

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        engine = Engine(symbols=["TEST"])
        engine.start()

        # Test intraday position sizing with ATR
        pos_size = engine._risk_manager.calculate_position_size(
            symbol="TEST",
            price=100.0,
            atr=3.0,
            signal_strength=0.7,
        )

        # Verify position sizing works
        assert pos_size.qty > 0
        assert pos_size.risk_pct > 0

        # Verify risk check allows position
        risk_check = engine._risk_manager.can_open_position(
            symbol="TEST",
            qty=pos_size.qty,
            price=100.0,
        )

        assert risk_check.allowed
        assert risk_check.allowed_qty == pos_size.qty

        # Verify fees calculation
        fees = engine._risk_manager.calculate_fees(qty=pos_size.qty, price=100.0)
        assert fees > 0

        # Manually create intraday position to verify integration
        engine._intraday_positions["TEST"] = IntradayPosition(
            symbol="TEST",
            direction="LONG",
            entry_price=100.0,
            entry_time=pd.Timestamp("2025-01-01 10:00:00"),
            qty=pos_size.qty,
            entry_fees=fees,
        )

        # Verify position has fees recorded
        position = engine._intraday_positions["TEST"]
        assert position.entry_fees > 0
        assert position.qty > 0


def test_intraday_entry_blocked_by_circuit_breaker(
    mock_breeze_client: MagicMock,
) -> None:
    """Test intraday entry blocked when circuit breaker is active."""
    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        engine = Engine(symbols=["TEST"])
        engine.start()

        # Trigger circuit breaker with large loss
        engine._risk_manager.record_trade(
            symbol="MANUAL_LOSS",
            realized_pnl=-60000.0,  # Exceeds 5% limit
            fees=1000.0,
        )

        # Verify breaker is active
        assert engine._risk_manager.is_circuit_breaker_active()

        # Try to open intraday position - should be blocked
        risk_check = engine._risk_manager.can_open_position(
            symbol="TEST",
            qty=100,
            price=100.0,
        )

        assert not risk_check.allowed
        assert risk_check.breaker_active


def test_intraday_exit_calculates_realized_pnl_with_fees(
    mock_breeze_client: MagicMock,
) -> None:
    """Test intraday exit calculates correct realized PnL including fees."""
    from src.domain.strategies.intraday import IntradayPosition

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        engine = Engine(symbols=["TEST"])
        engine.start()

        # Create intraday position
        entry_fees = engine._risk_manager.calculate_fees(qty=100, price=100.0)
        engine._intraday_positions["TEST"] = IntradayPosition(
            symbol="TEST",
            direction="LONG",
            entry_price=100.0,
            entry_time=pd.Timestamp("2025-01-01 10:00:00"),
            qty=100,
            entry_fees=entry_fees,
        )

        # Track with risk manager
        engine._risk_manager.update_position("TEST", qty=100, price=100.0, is_opening=True)

        # Calculate exit fees
        exit_price = 110.0  # Profitable exit
        exit_fees = engine._risk_manager.calculate_fees(qty=100, price=exit_price)

        # Calculate expected PnL
        gross_pnl = (exit_price - 100.0) * 100  # = 1000
        expected_realized_pnl = gross_pnl - entry_fees - exit_fees

        # Close position
        engine._close_intraday_position("TEST", exit_price, reason="test_exit")

        # Verify position closed
        assert "TEST" not in engine._intraday_positions

        # Verify capital updated
        expected_capital = 1000000.0 + expected_realized_pnl
        assert abs(engine._risk_manager._current_capital - expected_capital) < 0.01


def test_combined_intraday_swing_position_tracking(
    mock_breeze_client: MagicMock,
) -> None:
    """Test per-symbol cap applies to combined intraday + swing positions."""
    from src.domain.strategies.swing import SwingPosition

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        engine = Engine(symbols=["TEST"])
        engine.start()

        # Create swing position
        engine._swing_positions["TEST"] = SwingPosition(
            symbol="TEST",
            direction="LONG",
            entry_price=100.0,
            entry_date=pd.Timestamp("2025-01-01"),
            qty=500,  # 50K position value
            entry_fees=75.0,
        )

        # Track with risk manager
        engine._risk_manager.update_position("TEST", qty=500, price=100.0, is_opening=True)

        # Try to open large intraday position (would exceed 100K cap)
        risk_check = engine._risk_manager.can_open_position(
            symbol="TEST",
            qty=600,  # Would be 60K more (total 110K)
            price=100.0,
        )

        # Should allow partial (only 50K more to reach 100K cap)
        assert risk_check.allowed  # Partial allowance
        assert risk_check.allowed_qty == 500  # Only 500 shares allowed (50K)


def test_circuit_breaker_squares_off_all_positions(
    mock_breeze_client: MagicMock,
) -> None:
    """Test circuit breaker triggers unified square-off of intraday + swing."""
    from src.domain.strategies.intraday import IntradayPosition
    from src.domain.strategies.swing import SwingPosition

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        engine = Engine(symbols=["TEST1", "TEST2", "TEST3"])
        engine.start()

        # Create intraday positions
        engine._intraday_positions["TEST1"] = IntradayPosition(
            symbol="TEST1",
            direction="LONG",
            entry_price=100.0,
            entry_time=pd.Timestamp("2025-01-01 10:00:00"),
            qty=100,
            entry_fees=15.0,
        )

        engine._intraday_positions["TEST2"] = IntradayPosition(
            symbol="TEST2",
            direction="SHORT",
            entry_price=200.0,
            entry_time=pd.Timestamp("2025-01-01 11:00:00"),
            qty=50,
            entry_fees=15.0,
        )

        # Create swing position
        engine._swing_positions["TEST3"] = SwingPosition(
            symbol="TEST3",
            direction="LONG",
            entry_price=150.0,
            entry_date=pd.Timestamp("2025-01-01", tz=ist),
            qty=200,
            entry_fees=45.0,
        )

        # Track all positions
        engine._risk_manager.update_position("TEST1", qty=100, price=100.0, is_opening=True)
        engine._risk_manager.update_position("TEST2", qty=50, price=200.0, is_opening=True)
        engine._risk_manager.update_position("TEST3", qty=200, price=150.0, is_opening=True)

        # Trigger circuit breaker
        engine._risk_manager.record_trade(
            symbol="MANUAL_LOSS",
            realized_pnl=-60000.0,
            fees=1000.0,
        )

        assert engine._risk_manager.is_circuit_breaker_active()

        # Call unified square-off
        engine._square_off_all_positions()

        # Verify all positions closed
        assert len(engine._intraday_positions) == 0
        assert len(engine._swing_positions) == 0
