"""Risk management and position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from loguru import logger


@dataclass
class PositionSize:
    """Position sizing result with rationale."""

    qty: int
    risk_pct: float
    rationale: str
    warnings: list[str]


@dataclass
class RiskCheck:
    """Risk check result for position opening."""

    allowed: bool
    allowed_qty: int
    reason: str
    breaker_active: bool


class RiskManager:
    """
    Risk management with position sizing, caps, and circuit breaker.

    Supports:
    - Fixed fractional position sizing
    - ATR-based volatility sizing
    - Per-symbol position caps
    - Global circuit breaker on daily loss
    - Trading fees and slippage tracking
    """

    def __init__(
        self,
        starting_capital: float,
        mode: str = "FIXED_FRACTIONAL",
        risk_per_trade_pct: float = 1.0,
        atr_multiplier: float = 2.0,
        max_position_value_per_symbol: float = 100000.0,
        max_daily_loss_pct: float = 5.0,
        trading_fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ) -> None:
        """
        Initialize risk manager.

        Args:
            starting_capital: Starting capital in INR
            mode: Position sizing mode ("FIXED_FRACTIONAL" or "ATR_BASED")
            risk_per_trade_pct: Risk per trade as percentage of capital
            atr_multiplier: ATR multiplier for stop-loss (default: 2.0)
            max_position_value_per_symbol: Max position value per symbol in INR
            max_daily_loss_pct: Max daily loss before circuit breaker triggers
            trading_fee_bps: Trading fees in basis points (1 bps = 0.01%)
            slippage_bps: Slippage in basis points
        """
        self._starting_capital = starting_capital
        self._current_capital = starting_capital
        self._mode = mode
        self._risk_per_trade_pct = risk_per_trade_pct
        self._atr_multiplier = atr_multiplier
        self._max_position_value = max_position_value_per_symbol
        self._max_daily_loss_pct = max_daily_loss_pct
        self._trading_fee_bps = trading_fee_bps
        self._slippage_bps = slippage_bps

        # Track positions and PnL
        self._positions: dict[str, float] = {}  # symbol -> position value
        self._daily_realized_pnl: float = 0.0
        self._total_fees: float = 0.0
        self._circuit_breaker_active: bool = False
        self._last_reset_date: date | None = None

        logger.info(
            f"RiskManager initialized: capital={starting_capital:,.0f}, "
            f"mode={mode}, risk={risk_per_trade_pct}%",
            extra={
                "component": "risk_manager",
                "starting_capital": starting_capital,
                "mode": mode,
            },
        )

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        atr: float = 0.0,
        signal_strength: float = 1.0,
    ) -> PositionSize:
        """
        Calculate position size based on risk parameters.

        Args:
            symbol: Stock symbol
            price: Current price
            atr: Average True Range (required for ATR_BASED mode)
            signal_strength: Signal confidence [0.0, 1.0], scales risk

        Returns:
            PositionSize with qty, rationale, and warnings
        """
        warnings: list[str] = []

        # Adjust risk by signal strength
        effective_risk_pct = self._risk_per_trade_pct * signal_strength

        if self._mode == "FIXED_FRACTIONAL":
            # Fixed fractional: qty = (capital * risk%) / price
            risk_amount = self._current_capital * (effective_risk_pct / 100.0)
            qty = int(risk_amount / price)
            rationale = f"fixed_fractional_{effective_risk_pct:.1f}%"

            logger.debug(
                f"Fixed fractional sizing for {symbol}: "
                f"capital={self._current_capital:,.0f}, "
                f"risk={effective_risk_pct:.2f}%, qty={qty}",
                extra={
                    "component": "risk_manager",
                    "symbol": symbol,
                    "mode": "fixed_fractional",
                    "qty": qty,
                },
            )

        elif self._mode == "ATR_BASED":
            # ATR-based: qty = (capital * risk%) / (ATR * multiplier)
            if atr <= 0:
                warnings.append("ATR not provided, falling back to fixed fractional")
                risk_amount = self._current_capital * (effective_risk_pct / 100.0)
                qty = int(risk_amount / price)
                rationale = "fallback_fixed_fractional"
            else:
                stop_loss_distance = atr * self._atr_multiplier
                risk_amount = self._current_capital * (effective_risk_pct / 100.0)
                qty = int(risk_amount / stop_loss_distance)
                rationale = f"atr_based_atr={atr:.2f}_mult={self._atr_multiplier}"

                logger.debug(
                    f"ATR-based sizing for {symbol}: "
                    f"atr={atr:.2f}, stop_dist={stop_loss_distance:.2f}, qty={qty}",
                    extra={
                        "component": "risk_manager",
                        "symbol": symbol,
                        "mode": "atr_based",
                        "atr": atr,
                        "qty": qty,
                    },
                )

        else:
            warnings.append(f"Unknown sizing mode: {self._mode}")
            qty = 1  # Safe fallback
            rationale = "fallback_single_share"

        # Ensure minimum 1 share
        if qty < 1:
            qty = 1
            warnings.append("Position size too small, using 1 share minimum")

        return PositionSize(
            qty=qty,
            risk_pct=effective_risk_pct,
            rationale=rationale,
            warnings=warnings,
        )

    def can_open_position(
        self,
        symbol: str,
        qty: int,
        price: float,
    ) -> RiskCheck:
        """
        Check if position can be opened within risk limits.

        Args:
            symbol: Stock symbol
            qty: Requested quantity
            price: Current price

        Returns:
            RiskCheck with allowed flag, adjusted qty, and reason
        """
        # Check circuit breaker first
        if self._circuit_breaker_active:
            logger.warning(
                f"Position blocked for {symbol}: circuit breaker active",
                extra={"component": "risk_manager", "symbol": symbol},
            )
            return RiskCheck(
                allowed=False,
                allowed_qty=0,
                reason="circuit_breaker_active",
                breaker_active=True,
            )

        # Calculate position value
        requested_value = qty * price
        current_position_value = self._positions.get(symbol, 0.0)
        total_value = current_position_value + requested_value

        # Check per-symbol cap
        if total_value > self._max_position_value:
            # Calculate max allowable qty
            available_value = self._max_position_value - current_position_value
            if available_value <= 0:
                logger.warning(
                    f"Position fully blocked for {symbol}: cap reached "
                    f"(current={current_position_value:,.0f}, "
                    f"max={self._max_position_value:,.0f})",
                    extra={
                        "component": "risk_manager",
                        "symbol": symbol,
                        "current_value": current_position_value,
                    },
                )
                return RiskCheck(
                    allowed=False,
                    allowed_qty=0,
                    reason=f"symbol_cap_reached_{self._max_position_value:,.0f}",
                    breaker_active=False,
                )

            # Partial allowance
            allowed_qty = int(available_value / price)
            if allowed_qty < 1:
                allowed_qty = 0

            logger.warning(
                f"Position partially allowed for {symbol}: "
                f"requested={qty}, allowed={allowed_qty} "
                f"(cap={self._max_position_value:,.0f})",
                extra={
                    "component": "risk_manager",
                    "symbol": symbol,
                    "requested_qty": qty,
                    "allowed_qty": allowed_qty,
                },
            )
            return RiskCheck(
                allowed=allowed_qty > 0,
                allowed_qty=allowed_qty,
                reason=f"partial_capped_to_{allowed_qty}",
                breaker_active=False,
            )

        # Position allowed
        logger.debug(
            f"Position allowed for {symbol}: qty={qty}, value={requested_value:,.0f}",
            extra={
                "component": "risk_manager",
                "symbol": symbol,
                "qty": qty,
                "value": requested_value,
            },
        )
        return RiskCheck(
            allowed=True,
            allowed_qty=qty,
            reason="within_limits",
            breaker_active=False,
        )

    def is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker has been triggered."""
        return self._circuit_breaker_active

    def record_trade(
        self,
        symbol: str,
        realized_pnl: float,
        fees: float,
    ) -> None:
        """
        Record realized PnL and check circuit breaker.

        Args:
            symbol: Stock symbol
            realized_pnl: Realized PnL (net of fees)
            fees: Total fees paid
        """
        self._daily_realized_pnl += realized_pnl
        self._total_fees += fees
        self._current_capital += realized_pnl

        logger.info(
            f"Trade recorded for {symbol}: pnl={realized_pnl:,.2f}, "
            f"fees={fees:,.2f}, daily_pnl={self._daily_realized_pnl:,.2f}",
            extra={
                "component": "risk_manager",
                "symbol": symbol,
                "realized_pnl": realized_pnl,
                "daily_pnl": self._daily_realized_pnl,
            },
        )

        # Check circuit breaker
        max_loss = self._starting_capital * (self._max_daily_loss_pct / 100.0)
        if self._daily_realized_pnl < -max_loss:
            self._circuit_breaker_active = True
            logger.error(
                f"CIRCUIT BREAKER TRIGGERED! Daily loss: {self._daily_realized_pnl:,.2f} "
                f"exceeds limit: {-max_loss:,.2f} ({self._max_daily_loss_pct}%)",
                extra={
                    "component": "risk_manager",
                    "daily_pnl": self._daily_realized_pnl,
                    "max_loss": max_loss,
                    "breaker_active": True,
                },
            )

    def calculate_fees(self, qty: int, price: float) -> float:
        """
        Calculate total fees (trading + slippage).

        Args:
            qty: Quantity
            price: Price per share

        Returns:
            Total fees in INR
        """
        notional_value = qty * price
        total_bps = self._trading_fee_bps + self._slippage_bps
        fees = notional_value * (total_bps / 10000.0)  # Convert bps to fraction
        return fees

    def get_current_position_value(self, symbol: str) -> float:
        """Get current position value for a symbol."""
        return self._positions.get(symbol, 0.0)

    def update_position(
        self,
        symbol: str,
        qty: int,
        price: float,
        is_opening: bool,
    ) -> None:
        """
        Update position tracking.

        Args:
            symbol: Stock symbol
            qty: Quantity (positive for long, negative for short)
            price: Price per share
            is_opening: True if opening, False if closing
        """
        value = abs(qty) * price

        if is_opening:
            self._positions[symbol] = self._positions.get(symbol, 0.0) + value
        else:
            # Closing - reduce or clear position
            current = self._positions.get(symbol, 0.0)
            new_value = max(0.0, current - value)
            if new_value > 0:
                self._positions[symbol] = new_value
            elif symbol in self._positions:
                del self._positions[symbol]

        logger.debug(
            f"Position updated for {symbol}: "
            f"action={'open' if is_opening else 'close'}, "
            f"value={value:,.0f}, "
            f"total={self._positions.get(symbol, 0.0):,.0f}",
            extra={
                "component": "risk_manager",
                "symbol": symbol,
                "is_opening": is_opening,
                "position_value": self._positions.get(symbol, 0.0),
            },
        )

    def reset_daily(self) -> None:
        """Reset daily PnL and circuit breaker."""
        today = date.today()

        if self._last_reset_date != today:
            logger.info(
                f"Daily reset: prev_pnl={self._daily_realized_pnl:,.2f}, "
                f"total_fees={self._total_fees:,.2f}",
                extra={
                    "component": "risk_manager",
                    "daily_pnl": self._daily_realized_pnl,
                    "total_fees": self._total_fees,
                },
            )

            self._daily_realized_pnl = 0.0
            self._total_fees = 0.0
            self._circuit_breaker_active = False
            self._last_reset_date = today

    def get_daily_stats(self) -> dict[str, float]:
        """Get daily statistics."""
        return {
            "daily_realized_pnl": self._daily_realized_pnl,
            "total_fees": self._total_fees,
            "current_capital": self._current_capital,
            "capital_change_pct": (
                (self._current_capital - self._starting_capital) / self._starting_capital * 100.0
            ),
            "breaker_active": float(self._circuit_breaker_active),
        }
