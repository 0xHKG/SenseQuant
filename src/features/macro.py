"""Macro economic feature engineering (US-029 Phase 2).

Transforms raw macro indicator data into normalized trading features:
- Regime features (volatility, trend, liquidity regimes)
- Correlation features (rolling correlation with indices)
- Momentum features (MA crossovers, rate of change)
- Breadth features (market breadth indicators)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def calculate_regime_features(
    macro_df: pd.DataFrame,
    volatility_percentiles: tuple[float, float] = (33, 67),
) -> pd.DataFrame:
    """Calculate regime classification features.

    Args:
        macro_df: DataFrame with columns [date, indicator, value]
        volatility_percentiles: Percentile thresholds for regime classification

    Returns:
        DataFrame with columns [date, macro_volatility_regime,
                               macro_trend_regime, macro_liquidity_regime]
    """
    if macro_df.empty:
        logger.warning("Empty macro DataFrame for regime features")
        return pd.DataFrame()

    # Calculate rolling volatility for VIX or similar volatility indicator
    vix_data = macro_df[macro_df["indicator"].str.contains("VIX|vix", na=False)]

    if vix_data.empty:
        logger.warning("No VIX data found for regime features")
        return pd.DataFrame()

    features = []

    for date, group in vix_data.groupby("date"):
        try:
            vix_value = group["value"].iloc[0]

            # Volatility regime classification
            # Low: < 15, Medium: 15-25, High: > 25 (India VIX typical ranges)
            if vix_value < 15:
                vol_regime = "low"
            elif vix_value < 25:
                vol_regime = "medium"
            else:
                vol_regime = "high"

            # Trend regime (based on MA crossovers)
            # Simplified: using rate of change
            trend_regime = "sideways"  # Placeholder

            # Liquidity regime (based on spreads)
            liquidity_regime = "normal"  # Placeholder

            features.append(
                {
                    "date": date,
                    "macro_volatility_regime": vol_regime,
                    "macro_trend_regime": trend_regime,
                    "macro_liquidity_regime": liquidity_regime,
                }
            )

        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Error calculating regime features for {date}: {e}")
            continue

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    return df


def calculate_correlation_features(
    macro_df: pd.DataFrame,
    symbol_data: pd.DataFrame | None = None,
    window: int = 30,
) -> pd.DataFrame:
    """Calculate rolling correlation features with indices.

    Args:
        macro_df: DataFrame with columns [date, indicator, value]
        symbol_data: Optional symbol price data for correlation
        window: Rolling window in days

    Returns:
        DataFrame with columns [date, macro_correlation_nifty,
                               macro_correlation_vix, macro_beta_nifty]
    """
    if macro_df.empty:
        logger.warning("Empty macro DataFrame for correlation features")
        return pd.DataFrame()

    # Extract NIFTY data
    nifty_data = macro_df[macro_df["indicator"].str.contains("NIFTY|nifty", na=False)]

    if nifty_data.empty:
        logger.warning("No NIFTY data found for correlation features")
        return pd.DataFrame()

    # Pivot to get time series
    nifty_ts = nifty_data.pivot_table(index="date", columns="indicator", values="value").iloc[:, 0]
    nifty_ts = nifty_ts.sort_index()

    features = []

    for date in nifty_ts.index:
        try:
            # Rolling correlation with NIFTY (self-correlation = 1 for NIFTY itself)
            corr_nifty = 1.0  # Placeholder when symbol_data not provided

            # Rolling correlation with VIX
            corr_vix = 0.0  # Placeholder

            # Beta with respect to NIFTY
            beta_nifty = 1.0  # Placeholder

            features.append(
                {
                    "date": date,
                    "macro_correlation_nifty": corr_nifty,
                    "macro_correlation_vix": corr_vix,
                    "macro_beta_nifty": beta_nifty,
                }
            )

        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Error calculating correlation features for {date}: {e}")
            continue

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    return df


def calculate_momentum_features(
    macro_df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 50,
) -> pd.DataFrame:
    """Calculate momentum indicators from macro data.

    Args:
        macro_df: DataFrame with columns [date, indicator, value]
        short_window: Short MA window
        long_window: Long MA window

    Returns:
        DataFrame with columns [date, macro_ma_crossover, macro_roc,
                               macro_momentum_score, macro_volatility_mom]
    """
    if macro_df.empty:
        logger.warning("Empty macro DataFrame for momentum features")
        return pd.DataFrame()

    # Extract index data (NIFTY)
    index_data = macro_df[macro_df["indicator"].str.contains("NIFTY|nifty", na=False)]

    if index_data.empty:
        logger.warning("No index data found for momentum features")
        return pd.DataFrame()

    # Pivot to time series
    index_ts = index_data.pivot_table(index="date", columns="indicator", values="value").iloc[:, 0]
    index_ts = index_ts.sort_index()

    # Calculate moving averages
    ma_short = index_ts.rolling(window=short_window).mean()
    ma_long = index_ts.rolling(window=long_window).mean()

    # Calculate ROC
    roc = index_ts.pct_change(periods=short_window)

    # Calculate rolling volatility
    rolling_vol = index_ts.rolling(window=short_window).std()

    features = []

    for date in index_ts.index:
        try:
            # MA crossover signal
            if pd.notna(ma_short.loc[date]) and pd.notna(ma_long.loc[date]):
                if ma_short.loc[date] > ma_long.loc[date]:
                    ma_crossover = 1  # Bullish
                elif ma_short.loc[date] < ma_long.loc[date]:
                    ma_crossover = -1  # Bearish
                else:
                    ma_crossover = 0  # Neutral
            else:
                ma_crossover = 0

            # Rate of change
            roc_value = roc.loc[date] if pd.notna(roc.loc[date]) else 0

            # Momentum score (composite of ROC and MA crossover)
            momentum_score = (ma_crossover + np.sign(roc_value)) / 2

            # Volatility momentum (expanding/contracting)
            if pd.notna(rolling_vol.loc[date]):
                vol_pct = rolling_vol.pct_change().loc[date]
                volatility_mom = vol_pct if pd.notna(vol_pct) else 0
            else:
                volatility_mom = 0

            features.append(
                {
                    "date": date,
                    "macro_ma_crossover": ma_crossover,
                    "macro_roc": roc_value,
                    "macro_momentum_score": momentum_score,
                    "macro_volatility_mom": volatility_mom,
                }
            )

        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Error calculating momentum features for {date}: {e}")
            continue

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Fill NaN with 0
    df = df.fillna(0)

    return df


def calculate_breadth_features(
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate market breadth indicators.

    Note: This is a placeholder. Real implementation would require
    advance-decline data for constituent stocks.

    Args:
        macro_df: DataFrame with columns [date, indicator, value]

    Returns:
        DataFrame with columns [date, macro_advance_decline_ratio,
                               macro_new_highs_lows, macro_breadth_momentum]
    """
    if macro_df.empty:
        logger.warning("Empty macro DataFrame for breadth features")
        return pd.DataFrame()

    # Placeholder: Real implementation needs constituent data
    features = []

    unique_dates = macro_df["date"].unique()

    for date in unique_dates:
        features.append(
            {
                "date": date,
                "macro_advance_decline_ratio": 1.0,  # Placeholder
                "macro_new_highs_lows": 0.0,  # Placeholder
                "macro_breadth_momentum": 0.0,  # Placeholder
            }
        )

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    return df


def compute_all_macro_features(
    macro_df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 50,
    correlation_window: int = 30,
) -> pd.DataFrame:
    """Compute all macro features in one call.

    Args:
        macro_df: DataFrame with columns [date, indicator, value, change, change_pct]
        short_window: Short MA window for momentum
        long_window: Long MA window for momentum
        correlation_window: Window for correlation features

    Returns:
        DataFrame with all macro features, indexed by date
    """
    if macro_df.empty:
        logger.warning("Empty macro DataFrame for macro features")
        return pd.DataFrame()

    # Compute individual feature sets
    regime_features = calculate_regime_features(macro_df)
    correlation_features = calculate_correlation_features(macro_df, window=correlation_window)
    momentum_features = calculate_momentum_features(macro_df, short_window, long_window)
    breadth_features = calculate_breadth_features(macro_df)

    # Merge all features
    features = pd.DataFrame()

    if not regime_features.empty:
        features = regime_features
    if not correlation_features.empty:
        if features.empty:
            features = correlation_features
        else:
            features = features.join(correlation_features, how="outer")
    if not momentum_features.empty:
        if features.empty:
            features = momentum_features
        else:
            features = features.join(momentum_features, how="outer")
    if not breadth_features.empty:
        if features.empty:
            features = breadth_features
        else:
            features = features.join(breadth_features, how="outer")

    # Fill NaN
    features = features.ffill().bfill()

    if not features.empty:
        logger.info(
            f"Computed {len(features)} macro feature rows with {len(features.columns)} features"
        )

    return features
