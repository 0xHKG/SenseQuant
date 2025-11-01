"""Swing trading strategy with SMA crossover and position management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

from src.app.config import Settings
from src.domain.features import (
    calculate_adx,
    calculate_atr,
    calculate_ema,
    calculate_obv,
    calculate_rsi,
    calculate_sma,
)
from src.domain.support_resistance import (
    calculate_52week_levels,
    calculate_anchored_vwap,
    calculate_swing_highs_lows,
)
from src.domain.types import Signal, SignalDirection


@dataclass
class SwingPosition:
    """Swing position state for tracking open trades."""

    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    entry_date: pd.Timestamp
    qty: int
    entry_fees: float = 0.0
    exit_fees: float = 0.0
    realized_pnl: float = 0.0

    def days_held(self, current_date: pd.Timestamp) -> int:
        """Calculate days held (business days approximation)."""
        return (current_date - self.entry_date).days


def compute_features(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    Compute technical indicators for swing strategy (daily bars).

    Adds columns:
    - Momentum: sma_fast, sma_slow, rsi, ema20, atr14, adx14, obv
    - Support/Resistance: 52w_high, 52w_low, dist_from_52w_high, dist_from_52w_low,
      range_position, anchored_vwap, vwap_upper_1sd, vwap_lower_1sd, vwap_upper_2sd,
      vwap_lower_2sd, dist_from_vwap, is_swing_high, is_swing_low, last_swing_high,
      last_swing_low, bars_since_swing_high, bars_since_swing_low
    - valid: Boolean flag indicating sufficient data for core features

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
    required_cols = ["open", "high", "low", "close", "volume", "ts"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df.sort_values("ts")  # Ensure chronological order

    # Check if we have enough data for the longest period indicator
    min_required = max(
        settings.swing_sma_fast,
        settings.swing_sma_slow,
        settings.swing_rsi_period,
    )

    if len(df) < min_required:
        logger.warning(
            "Insufficient data for swing features",
            extra={
                "component": "swing",
                "rows": len(df),
                "required": min_required,
            },
        )
        df["sma_fast"] = None
        df["sma_slow"] = None
        df["rsi"] = None
        df["ema20"] = None
        df["atr14"] = None
        df["adx14"] = None
        df["obv"] = None
        df["valid"] = False
        return df

    # Calculate indicators using feature library
    df["sma_fast"] = calculate_sma(df["close"], period=settings.swing_sma_fast)
    df["sma_slow"] = calculate_sma(df["close"], period=settings.swing_sma_slow)
    df["rsi"] = calculate_rsi(df["close"], period=settings.swing_rsi_period)
    df["ema20"] = calculate_ema(df["close"], period=20)
    df["atr14"] = calculate_atr(df["high"], df["low"], df["close"], period=14)
    df["adx14"] = calculate_adx(df["high"], df["low"], df["close"], period=14)
    df["obv"] = calculate_obv(df["close"], df["volume"])

    # Calculate support/resistance levels (US-028 Phase 7 - long-horizon analytics)
    # 52-week high/low levels
    levels_52w = calculate_52week_levels(df["high"], df["low"], window_days=252)
    df = pd.concat([df, levels_52w], axis=1)

    # Anchored VWAP (1-year lookback)
    vwap_anchored = calculate_anchored_vwap(
        df["high"],
        df["low"],
        df["close"],
        df["volume"],
        lookback_days=252,
    )
    df = pd.concat([df, vwap_anchored], axis=1)

    # Swing highs/lows
    swing_levels = calculate_swing_highs_lows(df["high"], df["low"], lookback_left=5, lookback_right=5)
    df = pd.concat([df, swing_levels], axis=1)

    # Mark rows with valid indicators (non-NaN for core features)
    df["valid"] = df["sma_fast"].notna() & df["sma_slow"].notna() & df["rsi"].notna()

    logger.debug(
        "Swing features computed",
        extra={
            "component": "swing",
            "rows": len(df),
            "valid_rows": df["valid"].sum(),
        },
    )

    return df


def _check_iv_position_sizing(
    market_features: dict[str, float],
    settings: Settings,
) -> tuple[float, dict[str, float | bool]]:
    """Check IV-based position sizing adjustment (US-029 Phase 3).

    Args:
        market_features: Market features dictionary
        settings: Application settings

    Returns:
        Tuple of (size_multiplier, metadata_dict)
    """
    if not settings.swing_iv_position_sizing_enabled:
        return 1.0, {"enabled": False}

    iv_rank = market_features.get("opt_iv_rank", 50.0)
    size_multiplier = 1.0

    # Reduce size in elevated volatility
    if iv_rank > 70:
        size_multiplier = 0.7  # 30% reduction
    # Increase size in compressed volatility
    elif iv_rank < 30:
        size_multiplier = 1.2  # 20% increase

    return size_multiplier, {
        "enabled": True,
        "iv_rank": iv_rank,
        "size_multiplier": size_multiplier,
        "adjustment_pct": (size_multiplier - 1.0) * 100,
    }


def _check_macro_correlation_filter(
    market_features: dict[str, float],
    settings: Settings,
) -> tuple[bool, dict[str, float | bool | str]]:
    """Check macro correlation filtering (US-029 Phase 3).

    Args:
        market_features: Market features dictionary
        settings: Application settings

    Returns:
        Tuple of (passed, metadata_dict)
    """
    if not settings.swing_macro_correlation_filter_enabled:
        return True, {"enabled": False}

    correlation = market_features.get("macro_correlation_nifty", 0.5)
    beta = market_features.get("macro_beta_nifty", 1.0)

    # Block if low correlation (stock-specific risk)
    if abs(correlation) < 0.3:
        return False, {
            "enabled": True,
            "passed": False,
            "reason": "low_correlation",
            "correlation": correlation,
            "beta": beta,
        }

    return True, {
        "enabled": True,
        "passed": True,
        "correlation": correlation,
        "beta": beta,
        "high_beta": beta > 1.5,
    }


def _apply_beta_strength_adjustment(
    strength: float,
    market_features: dict[str, float],
    settings: Settings,
) -> tuple[float, dict[str, float | bool]]:
    """Apply beta-based strength adjustment (US-029 Phase 3).

    Args:
        strength: Base signal strength
        market_features: Market features dictionary
        settings: Application settings

    Returns:
        Tuple of (adjusted_strength, metadata_dict)
    """
    if not settings.swing_macro_correlation_filter_enabled:
        return strength, {"enabled": False}

    beta = market_features.get("macro_beta_nifty", 1.0)

    # Reduce strength for high beta stocks (amplified moves)
    if beta > 1.5:
        adjusted_strength = strength * 0.8  # 20% reduction
        return adjusted_strength, {
            "enabled": True,
            "beta": beta,
            "adjustment": -0.2,
            "original_strength": strength,
            "adjusted_strength": adjusted_strength,
        }

    return strength, {"enabled": True, "beta": beta, "adjustment": 0.0}


def signal(
    df: pd.DataFrame,
    settings: Settings,
    *,
    position: SwingPosition | None = None,
    sentiment_score: float = 0.0,
    sentiment_meta: dict[str, bool | float | str] | None = None,
    market_features: dict[str, float] | None = None,
) -> Signal:
    """
    Generate swing trading signal from feature DataFrame with sentiment gating.

    Entry Logic (no open position):
    - LONG if sma_fast crosses ABOVE sma_slow today (yesterday: fast <= slow, today: fast > slow)
      - Suppressed if sentiment < sentiment_gate_threshold
      - Confidence boosted if sentiment > sentiment_boost_threshold
    - SHORT if sma_fast crosses BELOW sma_slow today (yesterday: fast >= slow, today: fast < slow)
      - Suppressed if sentiment > -sentiment_gate_threshold
      - Confidence boosted if sentiment < -sentiment_boost_threshold
    - Market feature gating (US-029 Phase 3, optional):
      * Macro correlation filter: block if correlation too low
      * Beta adjustment: reduce strength for high beta stocks
      * IV position sizing: adjust position size based on IV rank
    - Otherwise FLAT

    Exit Logic (open position):
    - Check gap at open: if open price triggers SL/TP → exit at open ("gap_exit")
    - Check close price: SL/TP/max_hold → exit at close
    - Returns exit signal with FLAT direction

    Args:
        df: DataFrame with computed features (from compute_features)
        settings: Application settings with thresholds
        position: Current open position (None if no position)
        sentiment_score: Sentiment score [-1.0, 1.0] (default: 0.0 = neutral)
        sentiment_meta: Metadata from sentiment cache (provider, cache_hit, etc.)
        market_features: Optional market features dict (US-029 Phase 3)

    Returns:
        Signal with direction LONG/SHORT/FLAT, strength, and meta with reason
    """
    if sentiment_meta is None:
        sentiment_meta = {}
    required_features = ["ts", "open", "close", "sma_fast", "sma_slow", "valid"]
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Filter to valid rows only
    valid_df = df[df["valid"]]
    if valid_df.empty:
        logger.warning(
            "No valid feature rows for swing signal",
            extra={"component": "swing"},
        )
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
            meta={
                "reason": "hold",
                "strategy": "swing",
                "days_held": days_held,
            },
        )

    # === ENTRY LOGIC (no position) ===
    direction: SignalDirection = "FLAT"
    reason = "noop"
    strength = 0.0

    # Bullish crossover: fast crosses above slow
    if sma_fast_yesterday <= sma_slow_yesterday and sma_fast_today > sma_slow_today:
        # SENTIMENT GATING
        if sentiment_score < settings.sentiment_gate_threshold:
            logger.info(
                "Bullish signal suppressed by negative sentiment",
                extra={
                    "component": "swing",
                    "sentiment": sentiment_score,
                    "threshold": settings.sentiment_gate_threshold,
                },
            )
            return Signal(
                symbol="",
                direction="FLAT",
                strength=0.0,
                meta={
                    "reason": "sentiment_gate",
                    "strategy": "swing",
                    "sentiment": sentiment_score,
                    "sentiment_source": sentiment_meta.get("provider", "unknown"),
                },
            )

        direction = "LONG"
        reason = "bull_cross"
        strength = 0.8

        # SENTIMENT BOOSTING
        if sentiment_score > settings.sentiment_boost_threshold:
            original_strength = strength
            strength *= settings.sentiment_boost_multiplier
            logger.info(
                "Bullish signal boosted by positive sentiment",
                extra={
                    "component": "swing",
                    "sentiment": sentiment_score,
                    "original_strength": original_strength,
                    "boosted_strength": strength,
                },
            )

        logger.info(
            "Bullish crossover detected",
            extra={
                "component": "swing",
                "sma_fast": sma_fast_today,
                "sma_slow": sma_slow_today,
                "sentiment": sentiment_score,
            },
        )

    # Bearish crossunder: fast crosses below slow
    elif sma_fast_yesterday >= sma_slow_yesterday and sma_fast_today < sma_slow_today:
        # SENTIMENT GATING (inverse for SHORT)
        if sentiment_score > -settings.sentiment_gate_threshold:
            logger.info(
                "Bearish signal suppressed by positive sentiment",
                extra={
                    "component": "swing",
                    "sentiment": sentiment_score,
                    "threshold": -settings.sentiment_gate_threshold,
                },
            )
            return Signal(
                symbol="",
                direction="FLAT",
                strength=0.0,
                meta={
                    "reason": "sentiment_gate",
                    "strategy": "swing",
                    "sentiment": sentiment_score,
                    "sentiment_source": sentiment_meta.get("provider", "unknown"),
                },
            )

        direction = "SHORT"
        reason = "bear_cross"
        strength = 0.8

        # SENTIMENT BOOSTING (inverse for SHORT)
        if sentiment_score < -settings.sentiment_boost_threshold:
            original_strength = strength
            strength *= settings.sentiment_boost_multiplier
            logger.info(
                "Bearish signal boosted by negative sentiment",
                extra={
                    "component": "swing",
                    "sentiment": sentiment_score,
                    "original_strength": original_strength,
                    "boosted_strength": strength,
                },
            )

        logger.info(
            "Bearish crossunder detected",
            extra={
                "component": "swing",
                "sma_fast": sma_fast_today,
                "sma_slow": sma_slow_today,
                "sentiment": sentiment_score,
            },
        )

    # US-029 Phase 3: Apply market feature gates (if provided and enabled)
    feature_checks = {}
    if market_features is not None and direction != "FLAT":
        # Check macro correlation filter
        correlation_passed, correlation_meta = _check_macro_correlation_filter(
            market_features, settings
        )
        feature_checks["macro_correlation"] = correlation_meta
        if not correlation_passed:
            logger.info(
                f"{direction} signal blocked by macro correlation filter",
                extra={"component": "swing", "correlation_meta": correlation_meta},
            )
            direction = "FLAT"
            strength = 0.0
            reason = "blocked_by_macro_correlation"

        # Apply beta strength adjustment (if signal still active)
        if direction != "FLAT":
            strength, beta_meta = _apply_beta_strength_adjustment(
                strength, market_features, settings
            )
            feature_checks["beta_adjustment"] = beta_meta

        # Check IV position sizing
        size_multiplier, iv_sizing_meta = _check_iv_position_sizing(market_features, settings)
        feature_checks["iv_position_sizing"] = iv_sizing_meta

    meta = {
        "reason": reason,
        "strategy": "swing",
        "sma_fast": sma_fast_today,
        "sma_slow": sma_slow_today,
        "close": today_close,
        "sentiment": sentiment_score,
        "sentiment_source": sentiment_meta.get("provider", "unknown"),
    }

    # Add feature checks if any were performed
    if feature_checks:
        meta["feature_checks"] = feature_checks

    return Signal(symbol="", direction=direction, strength=strength, meta=meta)
