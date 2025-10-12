"""Unit tests for risk manager."""

from __future__ import annotations

from src.services.risk_manager import RiskManager


def test_fixed_fractional_sizing() -> None:
    """Test fixed fractional position sizing."""
    risk_mgr = RiskManager(
        starting_capital=1000000.0,
        mode="FIXED_FRACTIONAL",
        risk_per_trade_pct=1.0,
    )

    # 1% of 1M = 10K, at price 100 = 100 shares
    pos_size = risk_mgr.calculate_position_size("RELIANCE", price=100.0)

    assert pos_size.qty == 100
    assert pos_size.risk_pct == 1.0
    assert "fixed_fractional" in pos_size.rationale
    assert len(pos_size.warnings) == 0


def test_fixed_fractional_with_signal_strength() -> None:
    """Test position sizing scales with signal strength."""
    risk_mgr = RiskManager(
        starting_capital=1000000.0,
        mode="FIXED_FRACTIONAL",
        risk_per_trade_pct=2.0,
    )

    # 50% signal strength -> 1% risk (half of 2%)
    # 1% of 1M = 10K, at price 100 = 100 shares
    pos_size = risk_mgr.calculate_position_size("TCS", price=100.0, signal_strength=0.5)

    assert pos_size.qty == 100  # 1% of 1M / 100
    assert pos_size.risk_pct == 1.0


def test_atr_based_sizing() -> None:
    """Test ATR-based position sizing."""
    risk_mgr = RiskManager(
        starting_capital=1000000.0,
        mode="ATR_BASED",
        risk_per_trade_pct=1.0,
        atr_multiplier=2.0,
    )

    # ATR = 5, multiplier = 2, stop distance = 10
    # 1% of 1M = 10K, at stop distance 10 = 1000 shares
    pos_size = risk_mgr.calculate_position_size("INFY", price=100.0, atr=5.0)

    assert pos_size.qty == 1000
    assert "atr_based" in pos_size.rationale
    assert len(pos_size.warnings) == 0


def test_atr_fallback_when_missing() -> None:
    """Test ATR mode falls back to fixed fractional when ATR not provided."""
    risk_mgr = RiskManager(
        starting_capital=1000000.0,
        mode="ATR_BASED",
        risk_per_trade_pct=1.0,
    )

    # No ATR provided -> fallback to fixed fractional
    pos_size = risk_mgr.calculate_position_size("HDFC", price=100.0, atr=0.0)

    assert pos_size.qty == 100  # 1% of 1M / 100
    assert "fallback" in pos_size.rationale
    assert len(pos_size.warnings) > 0
    assert "ATR not provided" in pos_size.warnings[0]


def test_minimum_position_size() -> None:
    """Test minimum 1 share enforced."""
    risk_mgr = RiskManager(
        starting_capital=10000.0,  # Small capital
        mode="FIXED_FRACTIONAL",
        risk_per_trade_pct=0.1,  # Very small risk
    )

    # 0.1% of 10K = 10, at price 100 = 0.1 shares -> rounds to 1
    pos_size = risk_mgr.calculate_position_size("EXPENSIVE", price=100.0)

    assert pos_size.qty >= 1
    assert "minimum" in pos_size.warnings[0].lower()


def test_position_cap_full_block() -> None:
    """Test position blocked when symbol cap already reached."""
    risk_mgr = RiskManager(
        starting_capital=1000000.0,
        max_position_value_per_symbol=100000.0,
    )

    # Open position worth 100K
    risk_mgr.update_position("RELIANCE", qty=1000, price=100.0, is_opening=True)

    # Try to open more - should be blocked
    risk_check = risk_mgr.can_open_position("RELIANCE", qty=100, price=100.0)

    assert not risk_check.allowed
    assert risk_check.allowed_qty == 0
    assert "cap_reached" in risk_check.reason


def test_position_cap_partial_allowance() -> None:
    """Test position partially allowed when approaching symbol cap."""
    risk_mgr = RiskManager(
        starting_capital=1000000.0,
        max_position_value_per_symbol=100000.0,
    )

    # Open position worth 90K
    risk_mgr.update_position("TCS", qty=900, price=100.0, is_opening=True)

    # Try to add 20K more (200 shares) - only 100 shares allowed (10K)
    risk_check = risk_mgr.can_open_position("TCS", qty=200, price=100.0)

    assert risk_check.allowed
    assert risk_check.allowed_qty == 100  # Only 10K room left
    assert "partial" in risk_check.reason


def test_position_cap_full_allowance() -> None:
    """Test position fully allowed when within cap."""
    risk_mgr = RiskManager(
        starting_capital=1000000.0,
        max_position_value_per_symbol=100000.0,
    )

    # No existing position
    risk_check = risk_mgr.can_open_position("INFY", qty=500, price=100.0)

    assert risk_check.allowed
    assert risk_check.allowed_qty == 500
    assert "within_limits" in risk_check.reason
    assert not risk_check.breaker_active


def test_circuit_breaker_activation() -> None:
    """Test circuit breaker triggers on max daily loss."""
    risk_mgr = RiskManager(
        starting_capital=1000000.0,
        max_daily_loss_pct=5.0,  # 50K loss limit
    )

    # Record losing trade
    risk_mgr.record_trade("RELIANCE", realized_pnl=-60000.0, fees=1000.0)

    assert risk_mgr.is_circuit_breaker_active()


def test_circuit_breaker_blocks_new_positions() -> None:
    """Test circuit breaker blocks new positions."""
    risk_mgr = RiskManager(
        starting_capital=1000000.0,
        max_daily_loss_pct=5.0,
    )

    # Trigger circuit breaker
    risk_mgr.record_trade("TCS", realized_pnl=-60000.0, fees=1000.0)

    # Try to open position - should be blocked
    risk_check = risk_mgr.can_open_position("INFY", qty=100, price=100.0)

    assert not risk_check.allowed
    assert risk_check.breaker_active
    assert "circuit_breaker" in risk_check.reason


def test_circuit_breaker_reset() -> None:
    """Test circuit breaker resets on new day."""
    from datetime import date, timedelta
    from unittest.mock import patch

    risk_mgr = RiskManager(
        starting_capital=1000000.0,
        max_daily_loss_pct=5.0,
    )

    # Trigger breaker
    risk_mgr.record_trade("RELIANCE", realized_pnl=-60000.0, fees=1000.0)
    assert risk_mgr.is_circuit_breaker_active()

    # Reset on new day
    tomorrow = date.today() + timedelta(days=1)
    with patch("src.services.risk_manager.date") as mock_date:
        mock_date.today.return_value = tomorrow
        risk_mgr.reset_daily()

    assert not risk_mgr.is_circuit_breaker_active()


def test_fee_calculation() -> None:
    """Test fee calculation includes trading fees and slippage."""
    risk_mgr = RiskManager(
        starting_capital=1000000.0,
        trading_fee_bps=10.0,  # 0.1%
        slippage_bps=5.0,  # 0.05%
    )

    # 100 shares * 100 price = 10K notional
    # Total fees: 15 bps = 0.15% = 15 INR
    fees = risk_mgr.calculate_fees(qty=100, price=100.0)

    assert abs(fees - 15.0) < 0.01  # Allow floating point tolerance


def test_position_tracking_open_close() -> None:
    """Test position tracking updates correctly."""
    risk_mgr = RiskManager(starting_capital=1000000.0)

    # Open position
    risk_mgr.update_position("RELIANCE", qty=100, price=100.0, is_opening=True)
    assert risk_mgr.get_current_position_value("RELIANCE") == 10000.0

    # Close position
    risk_mgr.update_position("RELIANCE", qty=100, price=100.0, is_opening=False)
    assert risk_mgr.get_current_position_value("RELIANCE") == 0.0


def test_position_tracking_multiple_symbols() -> None:
    """Test multiple symbols tracked independently."""
    risk_mgr = RiskManager(starting_capital=1000000.0)

    risk_mgr.update_position("RELIANCE", qty=100, price=100.0, is_opening=True)
    risk_mgr.update_position("TCS", qty=200, price=50.0, is_opening=True)

    assert risk_mgr.get_current_position_value("RELIANCE") == 10000.0
    assert risk_mgr.get_current_position_value("TCS") == 10000.0


def test_capital_updates_with_pnl() -> None:
    """Test capital updates after realized PnL."""
    risk_mgr = RiskManager(starting_capital=1000000.0)

    # Profitable trade
    risk_mgr.record_trade("RELIANCE", realized_pnl=5000.0, fees=100.0)

    stats = risk_mgr.get_daily_stats()
    assert stats["current_capital"] == 1005000.0
    assert stats["daily_realized_pnl"] == 5000.0


def test_daily_stats() -> None:
    """Test daily statistics reporting."""
    risk_mgr = RiskManager(starting_capital=1000000.0)

    risk_mgr.record_trade("RELIANCE", realized_pnl=2000.0, fees=100.0)
    risk_mgr.record_trade("TCS", realized_pnl=-1000.0, fees=50.0)

    stats = risk_mgr.get_daily_stats()

    assert stats["daily_realized_pnl"] == 1000.0
    assert stats["total_fees"] == 150.0
    assert stats["current_capital"] == 1001000.0
    assert abs(stats["capital_change_pct"] - 0.1) < 0.01
