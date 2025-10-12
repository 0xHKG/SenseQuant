# US-003 — Swing Strategy v1 (Daily Bars)

## Goal
Implement a **daily-bar** swing strategy with configurable signals (SMA crossover), exits (SL/TP), and max-hold rules. Run post-close in the engine. Journal all entries/exits with reasons. Dry-run must never place real orders but fully log/journal intents.

## Scope Files
- `src/domain/strategies/swing.py` (NEW - complete rewrite)
- `src/services/engine.py` (ADD daily_swing method + position tracking)
- `src/app/config.py` (ADD SWING_* settings)
- `tests/unit/test_swing.py` (NEW - 12+ tests)
- `tests/integration/test_swing_engine.py` (NEW - 5+ integration tests)

---

## 1. Configuration Settings (Pydantic v2)

**File**: `src/app/config.py`

Add to `Settings` class with sensible defaults and validation aliases:

```python
# Swing Strategy
swing_bar_interval: Literal["1day"] = Field("1day", validation_alias="SWING_BAR_INTERVAL")
swing_sma_fast: int = Field(20, validation_alias="SWING_SMA_FAST", ge=5, le=100)
swing_sma_slow: int = Field(50, validation_alias="SWING_SMA_SLOW", ge=20, le=200)
swing_rsi_period: int = Field(14, validation_alias="SWING_RSI_PERIOD", ge=7, le=30)
swing_sl_pct: float = Field(0.03, validation_alias="SWING_SL_PCT", ge=0.01, le=0.10)  # 3%
swing_tp_pct: float = Field(0.06, validation_alias="SWING_TP_PCT", ge=0.02, le=0.20)  # 6%
swing_max_hold_days: int = Field(15, validation_alias="SWING_MAX_HOLD_DAYS", ge=2, le=30)
swing_feature_lookback_days: int = Field(120, validation_alias="SWING_FEATURE_LOOKBACK_DAYS", ge=60, le=365)
```

**Validation**:
- Ensure `swing_sma_slow > swing_sma_fast` (add field_validator)
- Ensure `swing_tp_pct > swing_sl_pct`

---

## 2. Strategy Implementation

**File**: `src/domain/strategies/swing.py`

### 2.1 Imports and Types

```python
"""Swing trading strategy with SMA crossover and position management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from src.app.config import Settings
from src.domain.types import Position, Signal, SignalDirection
```

### 2.2 Position State Helper

```python
@dataclass
class SwingPosition:
    """Swing position state for tracking open trades."""
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    entry_date: pd.Timestamp
    qty: int

    def days_held(self, current_date: pd.Timestamp) -> int:
        """Calculate days held (business days approximation)."""
        return (current_date - self.entry_date).days
```

### 2.3 compute_features Function

```python
def compute_features(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    Compute technical indicators for swing strategy (daily bars).

    Adds columns: sma_fast, sma_slow, rsi, valid.
    If insufficient data, sets valid=False for all rows.

    Args:
        df: DataFrame with OHLC columns (open, high, low, close, volume)
            Expected daily frequency with tz-aware timestamps
        settings: Application settings with indicator periods

    Returns:
        DataFrame with additional feature columns

    Raises:
        ValueError: If required OHLC columns are missing
    """
    required_cols = ["open", "high", "low", "close", "ts"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df.sort_values("ts")  # Ensure chronological order

    # Check if we have enough data for the longest period indicator
    min_required = max(settings.swing_sma_fast, settings.swing_sma_slow, settings.swing_rsi_period)

    if len(df) < min_required:
        logger.warning(
            "Insufficient data for swing features",
            extra={"component": "swing", "rows": len(df), "required": min_required},
        )
        df["sma_fast"] = None
        df["sma_slow"] = None
        df["rsi"] = None
        df["valid"] = False
        return df

    # Simple Moving Averages
    df["sma_fast"] = df["close"].rolling(window=settings.swing_sma_fast).mean()
    df["sma_slow"] = df["close"].rolling(window=settings.swing_sma_slow).mean()

    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=settings.swing_rsi_period).mean()  # type: ignore[operator]
    loss = (-delta.where(delta < 0, 0)).rolling(window=settings.swing_rsi_period).mean()  # type: ignore[operator]
    rs = gain / loss.replace(0, 1e-10)  # Avoid division by zero
    df["rsi"] = 100 - (100 / (1 + rs))

    # Mark rows with valid indicators (non-NaN for all features)
    df["valid"] = df["sma_fast"].notna() & df["sma_slow"].notna() & df["rsi"].notna()

    logger.debug(
        "Swing features computed",
        extra={"component": "swing", "rows": len(df), "valid_rows": df["valid"].sum()},
    )

    return df
```

### 2.4 signal Function

```python
def signal(
    df: pd.DataFrame,
    settings: Settings,
    *,
    position: SwingPosition | None = None,
) -> Signal:
    """
    Generate swing trading signal from feature DataFrame.

    Entry Logic (no open position):
    - LONG if sma_fast crosses ABOVE sma_slow today (yesterday: fast <= slow, today: fast > slow)
    - SHORT if sma_fast crosses BELOW sma_slow today (yesterday: fast >= slow, today: fast < slow)
    - Otherwise FLAT

    Exit Logic (open position):
    - Check gap at open: if open price triggers SL/TP → exit at open ("gap_exit")
    - Check close price: SL/TP/max_hold → exit at close
    - Returns exit signal with negative qty

    Args:
        df: DataFrame with computed features (from compute_features)
        settings: Application settings with thresholds
        position: Current open position (None if no position)

    Returns:
        Signal with direction LONG/SHORT/FLAT, qty, and meta with reason
    """
    required_features = ["ts", "open", "close", "sma_fast", "sma_slow", "valid"]
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Filter to valid rows only
    valid_df = df[df["valid"]]
    if valid_df.empty:
        logger.warning("No valid feature rows for swing signal", extra={"component": "swing"})
        return Signal(
            symbol="",
            direction="FLAT",
            strength=0.0,
            meta={"reason": "insufficient", "strategy": "swing"},
        )

    if len(valid_df) < 2:
        return Signal(
            symbol="",
            direction="FLAT",
            strength=0.0,
            meta={"reason": "insufficient", "strategy": "swing"},
        )

    # Get today (last row) and yesterday (second-to-last row)
    today = valid_df.iloc[-1]
    yesterday = valid_df.iloc[-2]

    today_date = pd.Timestamp(today["ts"])
    today_open = float(today["open"])
    today_close = float(today["close"])
    sma_fast_today = float(today["sma_fast"])
    sma_slow_today = float(today["sma_slow"])
    sma_fast_yesterday = float(yesterday["sma_fast"])
    sma_slow_yesterday = float(yesterday["sma_slow"])

    # === EXIT LOGIC (if position exists) ===
    if position is not None:
        days_held = position.days_held(today_date)

        # Check gap exit at open
        if position.direction == "LONG":
            pnl_open = (today_open - position.entry_price) / position.entry_price
            pnl_close = (today_close - position.entry_price) / position.entry_price

            # Gap down to SL at open
            if pnl_open <= -settings.swing_sl_pct:
                return Signal(
                    symbol=position.symbol,
                    direction="FLAT",
                    strength=1.0,
                    meta={
                        "reason": "gap_exit_sl",
                        "strategy": "swing",
                        "exit_price": today_open,
                        "pnl": pnl_open,
                        "days_held": days_held,
                    },
                )
            # Gap up to TP at open
            if pnl_open >= settings.swing_tp_pct:
                return Signal(
                    symbol=position.symbol,
                    direction="FLAT",
                    strength=1.0,
                    meta={
                        "reason": "gap_exit_tp",
                        "strategy": "swing",
                        "exit_price": today_open,
                        "pnl": pnl_open,
                        "days_held": days_held,
                    },
                )

            # Check SL/TP at close
            if pnl_close <= -settings.swing_sl_pct:
                return Signal(
                    symbol=position.symbol,
                    direction="FLAT",
                    strength=1.0,
                    meta={
                        "reason": "sl_hit",
                        "strategy": "swing",
                        "exit_price": today_close,
                        "pnl": pnl_close,
                        "days_held": days_held,
                    },
                )
            if pnl_close >= settings.swing_tp_pct:
                return Signal(
                    symbol=position.symbol,
                    direction="FLAT",
                    strength=1.0,
                    meta={
                        "reason": "tp_hit",
                        "strategy": "swing",
                        "exit_price": today_close,
                        "pnl": pnl_close,
                        "days_held": days_held,
                    },
                )

        elif position.direction == "SHORT":
            pnl_open = (position.entry_price - today_open) / position.entry_price
            pnl_close = (position.entry_price - today_close) / position.entry_price

            # Gap up to SL at open (short loses when price rises)
            if pnl_open <= -settings.swing_sl_pct:
                return Signal(
                    symbol=position.symbol,
                    direction="FLAT",
                    strength=1.0,
                    meta={
                        "reason": "gap_exit_sl",
                        "strategy": "swing",
                        "exit_price": today_open,
                        "pnl": pnl_open,
                        "days_held": days_held,
                    },
                )
            # Gap down to TP at open (short wins when price falls)
            if pnl_open >= settings.swing_tp_pct:
                return Signal(
                    symbol=position.symbol,
                    direction="FLAT",
                    strength=1.0,
                    meta={
                        "reason": "gap_exit_tp",
                        "strategy": "swing",
                        "exit_price": today_open,
                        "pnl": pnl_open,
                        "days_held": days_held,
                    },
                )

            # Check SL/TP at close
            if pnl_close <= -settings.swing_sl_pct:
                return Signal(
                    symbol=position.symbol,
                    direction="FLAT",
                    strength=1.0,
                    meta={
                        "reason": "sl_hit",
                        "strategy": "swing",
                        "exit_price": today_close,
                        "pnl": pnl_close,
                        "days_held": days_held,
                    },
                )
            if pnl_close >= settings.swing_tp_pct:
                return Signal(
                    symbol=position.symbol,
                    direction="FLAT",
                    strength=1.0,
                    meta={
                        "reason": "tp_hit",
                        "strategy": "swing",
                        "exit_price": today_close,
                        "pnl": pnl_close,
                        "days_held": days_held,
                    },
                )

        # Check max hold
        if days_held >= settings.swing_max_hold_days:
            pnl = (
                (today_close - position.entry_price) / position.entry_price
                if position.direction == "LONG"
                else (position.entry_price - today_close) / position.entry_price
            )
            return Signal(
                symbol=position.symbol,
                direction="FLAT",
                strength=1.0,
                meta={
                    "reason": "max_hold",
                    "strategy": "swing",
                    "exit_price": today_close,
                    "pnl": pnl,
                    "days_held": days_held,
                },
            )

        # Hold position (no exit signal)
        return Signal(
            symbol=position.symbol,
            direction="FLAT",
            strength=0.0,
            meta={"reason": "hold", "strategy": "swing", "days_held": days_held},
        )

    # === ENTRY LOGIC (no position) ===
    direction: SignalDirection = "FLAT"
    reason = "noop"
    strength = 0.0

    # Bullish crossover: fast crosses above slow
    if sma_fast_yesterday <= sma_slow_yesterday and sma_fast_today > sma_slow_today:
        direction = "LONG"
        reason = "bull_cross"
        strength = 0.8
        logger.info(
            "Bullish crossover detected",
            extra={
                "component": "swing",
                "sma_fast": sma_fast_today,
                "sma_slow": sma_slow_today,
            },
        )

    # Bearish crossunder: fast crosses below slow
    elif sma_fast_yesterday >= sma_slow_yesterday and sma_fast_today < sma_slow_today:
        direction = "SHORT"
        reason = "bear_cross"
        strength = 0.8
        logger.info(
            "Bearish crossunder detected",
            extra={
                "component": "swing",
                "sma_fast": sma_fast_today,
                "sma_slow": sma_slow_today,
            },
        )

    meta = {
        "reason": reason,
        "strategy": "swing",
        "sma_fast": sma_fast_today,
        "sma_slow": sma_slow_today,
        "close": today_close,
    }

    return Signal(symbol="", direction=direction, strength=strength, meta=meta)
```

---

## 3. Engine Integration

**File**: `src/services/engine.py`

### 3.1 Add Position Tracking

```python
# In Engine.__init__
self._swing_positions: dict[str, SwingPosition] = {}  # symbol -> position
```

### 3.2 Implement run_swing_daily Method

```python
def run_swing_daily(self, symbol: str) -> None:
    """
    Process daily swing evaluation for symbol.

    Runs post-market close (typically 16:00 IST). Fetches last N daily bars,
    computes features, generates entry/exit signals, and journals decisions.

    Holiday-safe: skips if no new bar available today.

    Args:
        symbol: Stock symbol to process
    """
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)

    # Fetch historical bars for feature computation
    lookback_days = settings.swing_feature_lookback_days
    start_ts = pd.Timestamp(now_ist - timedelta(days=lookback_days))
    end_ts = pd.Timestamp(now_ist)

    try:
        bars = self.client.historical_bars(
            symbol=symbol,
            interval=settings.swing_bar_interval,
            start=start_ts,
            end=end_ts,
        )
    except Exception as e:
        logger.error(
            f"Failed to fetch daily bars for {symbol}: {e}",
            extra={"component": "engine", "symbol": symbol, "error": str(e)},
        )
        return

    if not bars:
        logger.warning(
            f"No daily bars returned for {symbol}",
            extra={"component": "engine", "symbol": symbol},
        )
        return

    # Convert bars to DataFrame
    df = pd.DataFrame(
        [
            {
                "ts": bar.ts,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )

    # Holiday check: ensure we have today's bar
    today_date = now_ist.date()
    last_bar_date = pd.Timestamp(df.iloc[-1]["ts"]).date()
    if last_bar_date < today_date:
        logger.info(
            f"No new bar for {symbol} today (holiday or weekend)",
            extra={"component": "engine", "symbol": symbol, "last_bar": str(last_bar_date)},
        )
        return

    # Compute features
    try:
        from src.domain.strategies.swing import SwingPosition, compute_features, signal

        df_features = compute_features(df, settings)
    except Exception as e:
        logger.error(
            f"Feature computation failed for {symbol}: {e}",
            extra={"component": "engine", "symbol": symbol, "error": str(e)},
        )
        return

    # Get current position (if any)
    current_position = self._swing_positions.get(symbol)

    # Generate signal
    try:
        sig = signal(df_features, settings, position=current_position)
    except Exception as e:
        logger.error(
            f"Swing signal generation failed for {symbol}: {e}",
            extra={"component": "engine", "symbol": symbol, "error": str(e)},
        )
        return

    # Process signal
    if sig.direction == "FLAT" and sig.meta and sig.meta.get("reason") == "hold":
        # Continue holding, no action needed
        logger.debug(
            f"Holding swing position for {symbol}",
            extra={"component": "engine", "symbol": symbol},
        )
        return

    # Exit signal
    if sig.direction == "FLAT" and current_position is not None:
        exit_price = sig.meta.get("exit_price", df_features.iloc[-1]["close"]) if sig.meta else df_features.iloc[-1]["close"]
        pnl = sig.meta.get("pnl", 0.0) if sig.meta else 0.0
        reason = sig.meta.get("reason", "exit") if sig.meta else "exit"

        # Place exit order (or journal in dryrun)
        side = "SELL" if current_position.direction == "LONG" else "BUY"

        if settings.mode == "dryrun":
            self.journal.log(
                symbol=symbol,
                action=side,
                qty=current_position.qty,
                price=exit_price,
                pnl=pnl * current_position.entry_price * current_position.qty,
                reason=reason,
                mode=settings.mode,
                order_id="DRYRUN",
                status="EXIT",
                strategy="swing",
                meta_json=str(sig.meta) if sig.meta else "",
            )
        else:
            # Live mode: place real exit order
            try:
                response = self.client.place_order(
                    symbol=symbol,
                    side=side,  # type: ignore[arg-type]
                    qty=current_position.qty,
                    order_type="MARKET",
                )
                self.journal.log(
                    symbol=symbol,
                    action=side,
                    qty=current_position.qty,
                    price=exit_price,
                    pnl=pnl * current_position.entry_price * current_position.qty,
                    reason=reason,
                    mode=settings.mode,
                    order_id=response.order_id,
                    status=response.status,
                    strategy="swing",
                    meta_json=str(sig.meta) if sig.meta else "",
                )
            except Exception as e:
                logger.error(
                    f"Failed to place exit order for {symbol}: {e}",
                    extra={"component": "engine", "symbol": symbol, "error": str(e)},
                )

        # Clear position
        del self._swing_positions[symbol]
        logger.info(
            f"Swing exit for {symbol}: {reason}",
            extra={"component": "engine", "symbol": symbol, "pnl": pnl},
        )
        return

    # Entry signal
    if sig.direction in ("LONG", "SHORT") and current_position is None:
        entry_price = df_features.iloc[-1]["close"]
        qty = 10  # Fixed qty for v1 (Risk module US-005 will handle sizing)

        if settings.mode == "dryrun":
            self.journal.log(
                symbol=symbol,
                action=sig.direction,
                qty=qty,
                price=entry_price,
                pnl=0.0,
                reason=sig.meta.get("reason", "entry") if sig.meta else "entry",
                mode=settings.mode,
                order_id="DRYRUN",
                status="ENTRY",
                strategy="swing",
                meta_json=str(sig.meta) if sig.meta else "",
            )
        else:
            # Live mode: place real entry order
            side = "BUY" if sig.direction == "LONG" else "SELL"
            try:
                response = self.client.place_order(
                    symbol=symbol,
                    side=side,  # type: ignore[arg-type]
                    qty=qty,
                    order_type="MARKET",
                )
                self.journal.log(
                    symbol=symbol,
                    action=sig.direction,
                    qty=qty,
                    price=entry_price,
                    pnl=0.0,
                    reason=sig.meta.get("reason", "entry") if sig.meta else "entry",
                    mode=settings.mode,
                    order_id=response.order_id,
                    status=response.status,
                    strategy="swing",
                    meta_json=str(sig.meta) if sig.meta else "",
                )
            except Exception as e:
                logger.error(
                    f"Failed to place entry order for {symbol}: {e}",
                    extra={"component": "engine", "symbol": symbol, "error": str(e)},
                )
                return

        # Create position tracking
        self._swing_positions[symbol] = SwingPosition(
            symbol=symbol,
            direction=sig.direction,  # type: ignore[arg-type]
            entry_price=entry_price,
            entry_date=pd.Timestamp(df_features.iloc[-1]["ts"]),
            qty=qty,
        )
        logger.info(
            f"Swing entry for {symbol}: {sig.direction}",
            extra={"component": "engine", "symbol": symbol, "price": entry_price},
        )
```

---

## 4. Unit Tests

**File**: `tests/unit/test_swing.py`

Test cases (12 minimum):

1. `test_compute_features_success` - Valid daily data → features computed
2. `test_compute_features_insufficient_data` - Too few bars → valid=False
3. `test_compute_features_missing_columns` - Raises ValueError
4. `test_signal_bullish_crossover_long_entry` - fast crosses above slow → LONG
5. `test_signal_bearish_crossunder_short_entry` - fast crosses below slow → SHORT
6. `test_signal_no_cross_noop` - No crossover → FLAT
7. `test_signal_stop_loss_trigger_long` - Position hits SL → exit
8. `test_signal_take_profit_trigger_long` - Position hits TP → exit
9. `test_signal_max_hold_exit_reached` - Days held ≥ MAX_HOLD → exit
10. `test_signal_gap_exit_at_open_tp` - Gap up at open exceeds TP → gap_exit
11. `test_signal_gap_exit_at_open_sl` - Gap down at open hits SL → gap_exit
12. `test_signal_insufficient_data_safe` - Empty valid_df → FLAT with reason

---

## 5. Integration Tests

**File**: `tests/integration/test_swing_engine.py`

Test scenarios (5 minimum):

1. `test_run_swing_daily_entry_on_crossover` - Mock bars → bullish cross → journal entry
2. `test_run_swing_daily_tp_exit` - Open position → TP hit → journal exit
3. `test_run_swing_daily_holiday_skip` - Last bar is yesterday → skip processing
4. `test_run_swing_daily_gap_exit` - Gap at open exceeds SL → journal gap_exit
5. `test_run_swing_daily_dryrun_no_orders` - MODE=dryrun → no place_order calls

---

## Acceptance Criteria (GIVEN-WHEN-THEN)

### AC1: Configuration Settings Validated
**GIVEN** Settings loaded from .env
**WHEN** SWING_SMA_SLOW ≤ SWING_SMA_FAST
**THEN** Validation error raised

**GIVEN** All SWING_* settings present
**WHEN** Settings() instantiated
**THEN** All swing parameters accessible with correct types

### AC2: Feature Computation Robust
**GIVEN** Daily OHLCV DataFrame with 100 rows
**WHEN** compute_features() called
**THEN** sma_fast, sma_slow, rsi, valid columns added; valid rows > 0

**GIVEN** Daily OHLCV DataFrame with 10 rows (insufficient)
**WHEN** compute_features() called
**THEN** All valid=False, no exceptions raised

### AC3: Crossover Entry Signals Deterministic
**GIVEN** Fixture where yesterday fast ≤ slow, today fast > slow
**WHEN** signal() called with no position
**THEN** Returns Signal(direction="LONG", meta["reason"]="bull_cross")

**GIVEN** Fixture where yesterday fast ≥ slow, today fast < slow
**WHEN** signal() called with no position
**THEN** Returns Signal(direction="SHORT", meta["reason"]="bear_cross")

**GIVEN** Fixture with no crossover
**WHEN** signal() called with no position
**THEN** Returns Signal(direction="FLAT", meta["reason"]="noop")

### AC4: Stop-Loss Exit Enforced
**GIVEN** Open LONG position with entry_price=100
**AND** Today's close=96.5 (PnL = -3.5% < -3% SL)
**WHEN** signal() called with position
**THEN** Returns Signal(direction="FLAT", meta["reason"]="sl_hit")

**GIVEN** Open SHORT position with entry_price=100
**AND** Today's close=104 (PnL = -4% < -3% SL)
**WHEN** signal() called with position
**THEN** Returns Signal(direction="FLAT", meta["reason"]="sl_hit")

### AC5: Take-Profit Exit Enforced
**GIVEN** Open LONG position with entry_price=100
**AND** Today's close=107 (PnL = +7% > +6% TP)
**WHEN** signal() called with position
**THEN** Returns Signal(direction="FLAT", meta["reason"]="tp_hit")

### AC6: Max-Hold Exit Enforced
**GIVEN** Open LONG position held for 16 days
**AND** SWING_MAX_HOLD_DAYS=15
**WHEN** signal() called with position
**THEN** Returns Signal(direction="FLAT", meta["reason"]="max_hold")

### AC7: Gap Exits at Open
**GIVEN** Open LONG position with entry_price=100
**AND** Today's open=92 (gap down -8% triggers SL at open)
**WHEN** signal() called with position
**THEN** Returns Signal(direction="FLAT", meta["reason"]="gap_exit_sl", meta["exit_price"]=92)

**GIVEN** Open SHORT position with entry_price=100
**AND** Today's open=93 (gap down +7% profit triggers TP at open)
**WHEN** signal() called with position
**THEN** Returns Signal(direction="FLAT", meta["reason"]="gap_exit_tp", meta["exit_price"]=93)

### AC8: Engine Daily Flow Holiday-Safe
**GIVEN** run_swing_daily() called on Saturday
**AND** Last bar date is Friday
**WHEN** Engine checks today's bar
**THEN** Logs "No new bar for {symbol} today (holiday or weekend)", skips processing

### AC9: Engine Journals All Decisions
**GIVEN** Bullish crossover detected
**WHEN** run_swing_daily() completes
**THEN** Journal entry with action=LONG, strategy="swing", reason="bull_cross"

**GIVEN** TP hit on open position
**WHEN** run_swing_daily() completes
**THEN** Journal entry with action=SELL (if LONG), strategy="swing", reason="tp_hit", pnl>0

### AC10: Dry-Run Mode No Live Orders
**GIVEN** MODE=dryrun, bullish crossover signal
**WHEN** run_swing_daily() called
**THEN** client.place_order() NOT called; journal entry written with order_id="DRYRUN"

### AC11: Logging Structured
**GIVEN** Any swing operation
**WHEN** Logs emitted
**THEN** All logs include extra={"component": "swing"} or {"component": "engine"}; no secrets logged

### AC12: Quality Gates Pass
**GIVEN** Implementation complete
**WHEN** Run `ruff check .`
**THEN** All checks pass

**WHEN** Run `ruff format --check .`
**THEN** All files already formatted

**WHEN** Run `mypy src`
**THEN** Success: no issues found

**WHEN** Run `pytest -q`
**THEN** All tests pass; coverage on src/domain/strategies/swing.py ≥ 75%

---

## Done Checklist

- [ ] Settings added to src/app/config.py with validation
- [ ] SwingPosition dataclass implemented
- [ ] compute_features() implemented with full typing/docstrings
- [ ] signal() implemented with entry + exit logic
- [ ] run_swing_daily() added to engine
- [ ] Position tracking (_swing_positions dict) added to Engine
- [ ] 12+ unit tests in tests/unit/test_swing.py
- [ ] 5+ integration tests in tests/integration/test_swing_engine.py
- [ ] All ACs verified with test cases
- [ ] ruff check passes
- [ ] ruff format passes
- [ ] mypy src passes
- [ ] pytest passes with ≥75% coverage on swing.py
- [ ] Structured logging verified (no secrets)
- [ ] Journal entries verified for all decisions

---

**Estimated Effort**: 4-6 hours (implementation + tests + verification)
