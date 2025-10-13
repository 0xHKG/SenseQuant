"""Intraday trading strategy with technical indicators and sentiment gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

from src.app.config import Settings
from src.domain.features import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    calculate_vwap,
)
from src.domain.types import Signal, SignalDirection


@dataclass
class IntradayPosition:
    """Intraday position with fee tracking."""

    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    entry_time: pd.Timestamp
    qty: int
    entry_fees: float = 0.0
    exit_fees: float = 0.0
    realized_pnl: float = 0.0


def compute_features(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    Compute technical indicators for intraday strategy.

    Adds columns: sma20, ema50, rsi14, atr14, vwap, bb_upper, bb_middle, bb_lower,
    macd_line, macd_signal, macd_histogram, valid.
    If insufficient data, sets valid=False for all rows.

    Args:
        df: DataFrame with OHLC columns (open, high, low, close, volume)
        settings: Application settings with indicator periods

    Returns:
        DataFrame with additional feature columns

    Raises:
        ValueError: If required OHLC columns are missing
    """
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Check if we have enough data for the longest period indicator
    min_required = max(
        settings.intraday_sma_period,
        settings.intraday_ema_period,
        settings.intraday_rsi_period,
        settings.intraday_atr_period,
    )

    if len(df) < min_required:
        logger.warning(
            "Insufficient data for features",
            extra={
                "component": "intraday",
                "rows": len(df),
                "required": min_required,
            },
        )
        df["sma20"] = None
        df["ema50"] = None
        df["rsi14"] = None
        df["atr14"] = None
        df["vwap"] = None
        df["bb_upper"] = None
        df["bb_middle"] = None
        df["bb_lower"] = None
        df["macd_line"] = None
        df["macd_signal"] = None
        df["macd_histogram"] = None
        df["valid"] = False
        return df

    # Calculate indicators using feature library
    df["sma20"] = calculate_sma(df["close"], period=settings.intraday_sma_period)
    df["ema50"] = calculate_ema(df["close"], period=settings.intraday_ema_period)
    df["rsi14"] = calculate_rsi(df["close"], period=settings.intraday_rsi_period)
    df["atr14"] = calculate_atr(
        df["high"], df["low"], df["close"], period=settings.intraday_atr_period
    )
    df["vwap"] = calculate_vwap(df["high"], df["low"], df["close"], df["volume"])

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df["close"], period=20, num_std=2.0)
    df["bb_upper"] = bb_upper
    df["bb_middle"] = bb_middle
    df["bb_lower"] = bb_lower

    # MACD
    macd_line, macd_signal, macd_histogram = calculate_macd(
        df["close"], fast_period=12, slow_period=26, signal_period=9
    )
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_histogram"] = macd_histogram

    # Mark rows with valid indicators (non-NaN for core features)
    df["valid"] = (
        df["sma20"].notna() & df["ema50"].notna() & df["rsi14"].notna() & df["atr14"].notna()
    )

    logger.debug(
        "Features computed",
        extra={
            "component": "intraday",
            "rows": len(df),
            "valid_rows": df["valid"].sum(),
        },
    )

    return df


def _check_spread_filter(
    market_features: dict[str, float],
    settings: Settings,
) -> tuple[bool, dict[str, float | bool]]:
    """Check if order book spread is acceptable (US-029 Phase 3).

    Args:
        market_features: Market features dictionary
        settings: Application settings

    Returns:
        Tuple of (passed, metadata_dict)
    """
    if not settings.intraday_spread_filter_enabled:
        return True, {"enabled": False}

    spread_pct = market_features.get("ob_spread_pct", 0.0)
    passed = spread_pct <= settings.intraday_max_spread_pct

    return passed, {
        "enabled": True,
        "passed": passed,
        "spread_pct": spread_pct,
        "threshold": settings.intraday_max_spread_pct,
    }


def _check_iv_gate(
    market_features: dict[str, float],
    settings: Settings,
    direction: SignalDirection,
) -> tuple[bool, dict[str, float | bool | str]]:
    """Check IV-based gating rules (US-029 Phase 3).

    Args:
        market_features: Market features dictionary
        settings: Application settings
        direction: Signal direction to check

    Returns:
        Tuple of (passed, metadata_dict)
    """
    if not settings.intraday_iv_adjustment_enabled:
        return True, {"enabled": False}

    iv_percentile = market_features.get("opt_iv_percentile", 50.0)
    skew = market_features.get("opt_skew_put_call", 0.0)

    # Block LONG in high IV, SHORT in low IV
    if direction == "LONG" and iv_percentile > 80:
        return False, {
            "enabled": True,
            "passed": False,
            "reason": "iv_too_high",
            "iv_percentile": iv_percentile,
            "skew": skew,
        }
    if direction == "SHORT" and iv_percentile < 20:
        return False, {
            "enabled": True,
            "passed": False,
            "reason": "iv_too_low",
            "iv_percentile": iv_percentile,
            "skew": skew,
        }

    return True, {"enabled": True, "passed": True, "iv_percentile": iv_percentile, "skew": skew}


def _check_macro_regime(
    market_features: dict[str, float],
    settings: Settings,
    direction: SignalDirection,
) -> tuple[bool, dict[str, float | bool | str]]:
    """Check macro regime gating rules (US-029 Phase 3).

    Args:
        market_features: Market features dictionary
        settings: Application settings
        direction: Signal direction to check

    Returns:
        Tuple of (passed, metadata_dict)
    """
    if not settings.intraday_macro_regime_filter_enabled:
        return True, {"enabled": False}

    trend_regime = market_features.get("macro_trend_regime", 0)
    vol_regime = market_features.get("macro_volatility_regime", 1)

    # Block LONG in downtrend, SHORT in uptrend
    if direction == "LONG" and trend_regime == -1:
        return False, {
            "enabled": True,
            "passed": False,
            "reason": "macro_downtrend",
            "trend_regime": trend_regime,
            "vol_regime": vol_regime,
        }
    if direction == "SHORT" and trend_regime == 1:
        return False, {
            "enabled": True,
            "passed": False,
            "reason": "macro_uptrend",
            "trend_regime": trend_regime,
            "vol_regime": vol_regime,
        }

    # Block all in high volatility regime
    if vol_regime == 2:
        return False, {
            "enabled": True,
            "passed": False,
            "reason": "high_volatility_regime",
            "trend_regime": trend_regime,
            "vol_regime": vol_regime,
        }

    return True, {
        "enabled": True,
        "passed": True,
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
    }


def _apply_market_pressure_adjustment(
    strength: float,
    market_features: dict[str, float],
    settings: Settings,
    direction: SignalDirection,
) -> tuple[float, dict[str, float | bool]]:
    """Apply market pressure-based signal strength adjustment (US-029 Phase 3).

    Args:
        strength: Base signal strength
        market_features: Market features dictionary
        settings: Application settings
        direction: Signal direction

    Returns:
        Tuple of (adjusted_strength, metadata_dict)
    """
    if not settings.intraday_market_pressure_adjustment_enabled:
        return strength, {"enabled": False}

    pressure = market_features.get("ob_market_pressure", 0.0)
    adjustment = 0.0

    if direction == "LONG" and pressure > 0:
        adjustment = min(0.2, pressure / 1000.0)
    elif direction == "SHORT" and pressure < 0:
        adjustment = min(0.2, -pressure / 1000.0)

    adjusted_strength = min(1.0, strength + adjustment)

    return adjusted_strength, {
        "enabled": True,
        "pressure": pressure,
        "adjustment": adjustment,
        "original_strength": strength,
        "adjusted_strength": adjusted_strength,
    }


def signal(
    df: pd.DataFrame,
    settings: Settings,
    *,
    sentiment: float | None = None,
    market_features: dict[str, float] | None = None,
) -> Signal:
    """
    Generate intraday trading signal from feature DataFrame.

    Logic:
    - LONG if close > sma20 AND rsi > INTRADAY_LONG_RSI_MIN
    - SHORT if close < sma20 AND rsi < INTRADAY_SHORT_RSI_MAX
    - Sentiment gating:
      * Block LONG if sentiment < SENTIMENT_NEG_LIMIT
      * Block SHORT if sentiment > SENTIMENT_POS_LIMIT
    - Market feature gating (US-029 Phase 3, optional):
      * Spread filter: block if spread too wide
      * IV gate: block LONG if IV too high, SHORT if IV too low
      * Macro regime: block LONG in downtrend, SHORT in uptrend, all in high vol
      * Market pressure: adjust signal strength based on order book pressure
    - Otherwise FLAT

    Args:
        df: DataFrame with computed features (from compute_features)
        settings: Application settings with thresholds
        sentiment: Sentiment score in [-1.0, 1.0] or None (default 0.0)
        market_features: Optional market features dict (US-029 Phase 3)

    Returns:
        Signal with direction LONG/SHORT/FLAT and strength

    Raises:
        ValueError: If required feature columns are missing or no valid rows
    """
    required_features = ["close", "sma20", "rsi14", "valid"]
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Filter to valid rows only
    valid_df = df[df["valid"]]
    if valid_df.empty:
        logger.warning(
            "No valid feature rows for signal generation",
            extra={"component": "intraday"},
        )
        return Signal(symbol="", direction="FLAT", strength=0.0, meta={"reason": "no_valid_data"})

    # Get latest valid row
    last_row = valid_df.iloc[-1]
    close = float(last_row["close"])
    sma20 = float(last_row["sma20"])
    rsi14 = float(last_row["rsi14"])

    # Default sentiment to 0.0 if None
    sentiment_val = sentiment if sentiment is not None else 0.0

    # Initialize direction
    direction: SignalDirection = "FLAT"
    strength = 0.0
    reason = "no_signal"

    # LONG signal logic
    if close > sma20 and rsi14 > settings.intraday_long_rsi_min:
        # Check sentiment gate for LONG
        if sentiment_val < settings.sentiment_neg_limit:
            reason = "long_blocked_by_negative_sentiment"
            logger.info(
                "LONG signal blocked by sentiment",
                extra={
                    "component": "intraday",
                    "sentiment": sentiment_val,
                    "threshold": settings.sentiment_neg_limit,
                },
            )
        else:
            direction = "LONG"
            strength = 0.7
            reason = "long_sma_rsi_sentiment_ok"

    # SHORT signal logic
    elif close < sma20 and rsi14 < settings.intraday_short_rsi_max:
        # Check sentiment gate for SHORT
        if sentiment_val > settings.sentiment_pos_limit:
            reason = "short_blocked_by_positive_sentiment"
            logger.info(
                "SHORT signal blocked by sentiment",
                extra={
                    "component": "intraday",
                    "sentiment": sentiment_val,
                    "threshold": settings.sentiment_pos_limit,
                },
            )
        else:
            direction = "SHORT"
            strength = 0.7
            reason = "short_sma_rsi_sentiment_ok"

    # US-029 Phase 3: Apply market feature gates (if provided and enabled)
    feature_checks = {}
    if market_features is not None and direction != "FLAT":
        # Check spread filter first (blocks all signals if spread too wide)
        spread_passed, spread_meta = _check_spread_filter(market_features, settings)
        feature_checks["spread_filter"] = spread_meta
        if not spread_passed:
            logger.info(
                f"{direction} signal blocked by spread filter",
                extra={"component": "intraday", "spread_meta": spread_meta},
            )
            direction = "FLAT"
            strength = 0.0
            reason = "blocked_by_spread_filter"

        # Check IV gate
        if direction != "FLAT":
            iv_passed, iv_meta = _check_iv_gate(market_features, settings, direction)
            feature_checks["iv_gate"] = iv_meta
            if not iv_passed:
                logger.info(
                    f"{direction} signal blocked by IV gate",
                    extra={"component": "intraday", "iv_meta": iv_meta},
                )
                direction = "FLAT"
                strength = 0.0
                reason = f"blocked_by_iv_gate: {iv_meta.get('reason', 'unknown')}"

        # Check macro regime
        if direction != "FLAT":
            macro_passed, macro_meta = _check_macro_regime(market_features, settings, direction)
            feature_checks["macro_regime"] = macro_meta
            if not macro_passed:
                logger.info(
                    f"{direction} signal blocked by macro regime",
                    extra={"component": "intraday", "macro_meta": macro_meta},
                )
                direction = "FLAT"
                strength = 0.0
                reason = f"blocked_by_macro_regime: {macro_meta.get('reason', 'unknown')}"

        # Apply market pressure adjustment (if signal still active)
        if direction != "FLAT":
            strength, pressure_meta = _apply_market_pressure_adjustment(
                strength, market_features, settings, direction
            )
            feature_checks["market_pressure"] = pressure_meta

    # Build meta snapshot
    meta = {
        "close": close,
        "sma20": sma20,
        "rsi14": rsi14,
        "sentiment": sentiment_val,
        "reason": reason,
    }

    # Add feature checks if any were performed
    if feature_checks:
        meta["feature_checks"] = feature_checks

    logger.info(
        f"Signal generated: {direction}",
        extra={
            "component": "intraday",
            "direction": direction,
            "strength": strength,
            "meta": meta,
        },
    )

    return Signal(symbol="", direction=direction, strength=strength, meta=meta)
