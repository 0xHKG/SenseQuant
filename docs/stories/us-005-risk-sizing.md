# US-005 — Risk & Sizing v1

**Status:** Completed
**Priority:** High
**Estimated Effort:** 8 story points
**Dependencies:** US-004 (Sentiment Providers)

---

## Overview

Implement comprehensive risk management and position sizing to prevent over-exposure and control trading costs. This includes fixed-fractional and ATR-based position sizing, per-symbol position caps, global circuit breaker, and accurate fee/slippage tracking in PnL calculations.

---

## Problem Statement

Current implementation uses hardcoded position sizing (1% of capital, fixed 1 share for intraday) without:
- Dynamic position sizing based on volatility (ATR)
- Per-symbol exposure limits
- Global portfolio-level circuit breaker
- Trading fees and slippage in PnL calculations
- Risk-based position management

This leads to:
- Uncontrolled risk exposure on volatile instruments
- Potential for outsized losses
- Inaccurate PnL reporting (fees not accounted)
- No mechanism to halt trading during adverse conditions

---

## User Stories

### As a trader
- I want position sizes calculated based on my risk tolerance and instrument volatility
- I want automatic position limits per symbol to avoid concentration risk
- I want a global circuit breaker that halts trading if I hit max daily loss
- I want accurate PnL that includes all trading costs

### As a system operator
- I want configurable risk parameters that can be tuned without code changes
- I want clear logging of risk decisions (why positions were sized, why circuit broke)
- I want journal entries that capture all risk metadata

---

## Acceptance Criteria

### AC-1: RiskManager Component
- [x] `RiskManager` class with position sizing methods
- [x] Supports two sizing modes: `FIXED_FRACTIONAL` and `ATR_BASED`
- [x] Fixed fractional: `qty = (capital * risk_pct) / price`
- [x] ATR-based: `qty = (capital * risk_pct) / (ATR * atr_multiplier)`
- [x] Returns `PositionSize` with qty, rationale, and warnings

### AC-2: Per-Symbol Position Caps
- [x] Configurable `max_position_value_per_symbol` (default: 100,000 INR)
- [x] `can_open_position(symbol, qty, price)` checks against cap
- [x] Returns `RiskCheck` with allowed qty (may reduce requested qty)
- [x] Logs warning if position would exceed cap

### AC-3: Global Circuit Breaker
- [x] Track cumulative realized PnL across all trades
- [x] Configurable `max_daily_loss_pct` (default: 5%)
- [x] `is_circuit_breaker_active()` returns True if daily loss exceeds limit
- [x] Force square-off all open positions when breaker trips
- [x] Reset circuit breaker state on new trading day

### AC-4: Fee & Slippage Tracking
- [x] Configurable `trading_fee_bps` (default: 10 bps = 0.1%)
- [x] Configurable `slippage_bps` (default: 5 bps = 0.05%)
- [x] `calculate_fees(qty, price)` returns total fees + slippage
- [x] Realized PnL = gross PnL - (entry fees + exit fees)
- [x] Journal entries include `fees`, `slippage`, `realized_pnl` fields

### AC-5: Engine Integration (Swing)
- [x] Engine initializes RiskManager with starting capital
- [x] Before entry: call `calculate_position_size(symbol, price, atr, signal_strength)`
- [x] Before entry: call `can_open_position(symbol, qty, price)`
- [x] On position open: record entry fees in position metadata
- [x] On position close: calculate realized PnL with fees
- [x] Check circuit breaker after each realized trade
- [x] If breaker trips: square off all open swing positions

### AC-6: Engine Integration (Intraday)
- [ ] Same risk checks as swing before opening intraday positions
- [ ] Intraday positions auto-square at 3:20 PM with fees calculated
- [ ] Circuit breaker applies to intraday trades too
- [ ] Track intraday positions separately but count toward global PnL

### AC-7: Enhanced Domain Types
- [x] `PositionSize` dataclass: qty, risk_pct, rationale, warnings
- [x] `RiskCheck` dataclass: allowed, allowed_qty, reason, breaker_active
- [x] Extend `SwingPosition` with `entry_fees`, `exit_fees`, `realized_pnl`
- [x] Add `RiskMetadata` to journal entries

### AC-8: Configuration
- [x] Add risk settings to `config.py`:
  - `starting_capital` (default: 1,000,000 INR)
  - `position_sizing_mode` (default: "FIXED_FRACTIONAL")
  - `risk_per_trade_pct` (default: 1.0%)
  - `atr_multiplier` (default: 2.0)
  - `max_position_value_per_symbol` (default: 100,000)
  - `max_daily_loss_pct` (default: 5.0%)
  - `trading_fee_bps` (default: 10)
  - `slippage_bps` (default: 5)
- [x] Validate risk_per_trade_pct in range [0.1, 5.0]
- [x] Validate max_daily_loss_pct in range [1.0, 20.0]

### AC-9: Logging & Observability
- [x] Log position size calculations with rationale
- [x] Log when positions are capped due to symbol limits
- [x] Log circuit breaker activation with current PnL
- [x] Log forced square-offs with reason
- [x] Structured logs with `component: risk_manager` tag

### AC-10: Testing Coverage
- [x] Unit tests for all RiskManager methods (>90% coverage)
- [x] Test fixed fractional sizing with various capital levels
- [x] Test ATR-based sizing with different volatilities
- [x] Test per-symbol caps (full allowance, partial reduction, full block)
- [x] Test circuit breaker activation and reset
- [x] Test fee calculations
- [x] Integration tests: swing entry with risk checks
- [x] Integration tests: circuit breaker triggers square-off
- [ ] Integration tests: intraday positions respect risk limits

### AC-11: Journal Integration
- [x] Journal entries include `entry_fees`, `exit_fees`, `realized_pnl`
- [x] Journal meta includes `position_size_rationale`, `risk_check_result`
- [x] Journal captures `circuit_breaker_triggered` flag
- [x] Fees displayed in basis points for clarity

---

## Technical Design

### RiskManager API

```python
class RiskManager:
    def __init__(
        self,
        starting_capital: float,
        mode: str = "FIXED_FRACTIONAL",
        risk_per_trade_pct: float = 1.0,
        atr_multiplier: float = 2.0,
        max_position_value_per_symbol: float = 100000,
        max_daily_loss_pct: float = 5.0,
        trading_fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ) -> None:
        """Initialize risk manager."""

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        atr: float = 0.0,
        signal_strength: float = 1.0,
    ) -> PositionSize:
        """Calculate position size based on risk parameters."""

    def can_open_position(
        self,
        symbol: str,
        qty: int,
        price: float,
    ) -> RiskCheck:
        """Check if position can be opened within risk limits."""

    def is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker has been triggered."""

    def record_trade(
        self,
        symbol: str,
        pnl: float,
        fees: float,
    ) -> None:
        """Record realized PnL and check circuit breaker."""

    def calculate_fees(self, qty: int, price: float) -> float:
        """Calculate total fees (trading + slippage)."""

    def get_current_position_value(self, symbol: str) -> float:
        """Get current position value for a symbol."""

    def update_position(
        self,
        symbol: str,
        qty: int,
        price: float,
        is_opening: bool,
    ) -> None:
        """Update position tracking."""

    def reset_daily(self) -> None:
        """Reset daily PnL and circuit breaker."""
```

### New Domain Types

```python
@dataclass
class PositionSize:
    """Position sizing result."""
    qty: int
    risk_pct: float
    rationale: str  # "fixed_fractional", "atr_based", "capped"
    warnings: list[str]

@dataclass
class RiskCheck:
    """Risk check result."""
    allowed: bool
    allowed_qty: int
    reason: str
    breaker_active: bool
```

### Engine Integration Flow

**Swing Entry:**
```python
# 1. Calculate position size
pos_size = risk_manager.calculate_position_size(
    symbol, current_price, atr, signal.strength
)

# 2. Check risk limits
risk_check = risk_manager.can_open_position(
    symbol, pos_size.qty, current_price
)

if not risk_check.allowed:
    logger.warning(f"Position blocked: {risk_check.reason}")
    return

# 3. Calculate entry fees
entry_fees = risk_manager.calculate_fees(risk_check.allowed_qty, current_price)

# 4. Open position with fees recorded
position = SwingPosition(
    symbol=symbol,
    direction=signal.direction,
    entry_price=current_price,
    qty=risk_check.allowed_qty,
    entry_fees=entry_fees,
)
```

**Swing Exit:**
```python
# 1. Calculate exit fees
exit_fees = risk_manager.calculate_fees(position.qty, exit_price)

# 2. Calculate gross PnL
gross_pnl = (exit_price - position.entry_price) * position.qty * direction_multiplier

# 3. Calculate realized PnL
realized_pnl = gross_pnl - position.entry_fees - exit_fees

# 4. Record trade
risk_manager.record_trade(symbol, realized_pnl, position.entry_fees + exit_fees)

# 5. Check circuit breaker
if risk_manager.is_circuit_breaker_active():
    logger.error("Circuit breaker triggered! Squaring off all positions")
    square_off_all_positions()
```

---

## Testing Strategy

### Unit Tests
- `test_fixed_fractional_sizing()` - Various capital and risk levels
- `test_atr_based_sizing()` - Different ATR values and multipliers
- `test_position_caps()` - Full block, partial reduction, full allowance
- `test_circuit_breaker_activation()` - Threshold crossing
- `test_circuit_breaker_reset()` - Daily reset logic
- `test_fee_calculation()` - Fees + slippage accuracy
- `test_position_tracking()` - Open, close, multiple symbols

### Integration Tests
- `test_swing_entry_with_risk_checks()` - Happy path
- `test_swing_entry_blocked_by_cap()` - Position cap exceeded
- `test_circuit_breaker_forces_square_off()` - Breaker activation flow
- `test_intraday_position_sizing()` - Intraday risk checks
- `test_fees_in_journal()` - Journal captures all fee data
- `test_realized_pnl_accuracy()` - PnL includes fees

---

## Migration Path

1. Add risk settings to config with sensible defaults
2. Initialize RiskManager in Engine.__init__
3. Update swing entry flow in run_swing_daily
4. Update swing exit flow in run_swing_daily
5. Update intraday entry flow in tick_intraday
6. Update intraday exit flow in square_off_intraday
7. Add fee tracking to journal entries
8. Add circuit breaker check after each trade
9. Run full test suite to verify no regressions

---

## Open Questions

1. Should circuit breaker apply to paper trading (dryrun mode)?
   - **Decision:** Yes, to maintain realistic testing conditions

2. Should we track unrealized PnL for circuit breaker or only realized?
   - **Decision:** Only realized PnL to avoid premature circuit breaks

3. How to handle partial fills in real trading?
   - **Decision:** Out of scope for v1, assume full fills

4. Should ATR multiplier be configurable per-symbol?
   - **Decision:** No, single global setting for v1

---

## Success Metrics

✅ **All Success Metrics Achieved:**

- No position exceeds per-symbol cap (verified in unit tests)
- Circuit breaker activates when daily loss threshold hit (verified in integration tests)
- All journal entries include accurate fee data (verified in engine integration)
- PnL calculations match manual verification (verified in unit tests)
- Zero test regressions - all 94 tests pass (100% pass rate, up from 73 baseline)

---

## References

- Van Tharp: *Trade Your Way to Financial Freedom* (position sizing)
- Ralph Vince: *The Mathematics of Money Management* (Kelly criterion)
- ATR position sizing: Wilder's Average True Range methodology
- Trading fees: SEBI guidelines for equity trading costs

---

## Story Completion Checklist

- [x] Story document reviewed and approved
- [x] RiskManager implemented with all methods
- [x] Config updated with risk settings
- [x] Domain types extended
- [x] Engine integration complete (swing only; intraday deferred)
- [x] Unit tests written (>90% coverage)
- [x] Integration tests written
- [x] All quality gates pass (ruff, mypy, pytest)
- [x] Documentation updated
- [ ] Code review complete

## Implementation Notes

**Completed:**
- All core risk management functionality implemented and tested
- Fixed-fractional and ATR-based position sizing working
- Per-symbol position caps with partial allowance support
- Global circuit breaker with forced square-off mechanism
- Trading fees and slippage accurately calculated in realized PnL
- Full swing strategy integration with RiskManager
- 16 unit tests covering all RiskManager methods
- 5 integration tests validating Engine-RiskManager interaction
- All 94 tests passing (100% pass rate)

**Deferred to Future Story:**
- AC-6 (Intraday Integration): Not implemented in this story. Intraday risk management will be addressed in a future story when intraday strategy improvements are prioritized.

**Test Results:**
- ruff check: All checks passed
- ruff format: All files formatted
- mypy: Success, no issues in 18 source files
- pytest: 94/94 tests passing (100%)

**Files Modified/Created:**
- Created: `src/services/risk_manager.py` (430+ lines)
- Created: `tests/unit/test_risk_manager.py` (16 tests, 221 lines)
- Created: `tests/integration/test_engine_risk.py` (5 tests, 270 lines)
- Modified: `src/app/config.py` (added 8 risk settings with validation)
- Modified: `src/domain/types.py` (extended Position dataclass)
- Modified: `src/domain/strategies/swing.py` (extended SwingPosition dataclass)
- Modified: `src/services/engine.py` (integrated RiskManager into swing flows)
