"""Support/Resistance Analytics for Multi-Year Technical Analysis.

This module provides long-horizon support/resistance level calculations for swing
trading strategies, including 52-week high/low bands, anchored VWAP, and volume
profile analysis.

All functions accept pandas Series/DataFrame as input and return pandas Series
or DataFrames with calculated levels. All functions handle edge cases gracefully
and return NaN for periods where calculation is not possible.

Usage:
    >>> import pandas as pd
    >>> from src.domain.support_resistance import (
    ...     calculate_52week_levels,
    ...     calculate_anchored_vwap,
    ...     calculate_volume_profile_levels,
    ... )
    >>> # Calculate 52-week high/low levels
    >>> high = pd.Series([...])
    >>> low = pd.Series([...])
    >>> levels = calculate_52week_levels(high, low, window_days=252)
"""

from __future__ import annotations

import pandas as pd


def calculate_52week_levels(
    high: pd.Series,
    low: pd.Series,
    window_days: int = 252,
) -> pd.DataFrame:
    """Calculate 52-week (or custom window) high/low support/resistance levels.

    Identifies the highest high and lowest low within a rolling window, which
    serve as key resistance and support levels respectively. These levels are
    commonly used by swing traders to identify breakout/breakdown opportunities.

    Args:
        high: High price series
        low: Low price series
        window_days: Lookback window in trading days (default 252 ≈ 52 weeks)

    Returns:
        DataFrame with columns:
        - '52w_high': Rolling maximum high (resistance level)
        - '52w_low': Rolling minimum low (support level)
        - 'dist_from_52w_high': Distance from current high to 52w high (%)
        - 'dist_from_52w_low': Distance from current low to 52w low (%)
        - 'range_position': Position within 52w range (0-1, where 1 = at high)

    Example:
        >>> high = pd.Series([100, 105, 110, 108, 112] * 60)
        >>> low = pd.Series([95, 100, 105, 103, 107] * 60)
        >>> levels = calculate_52week_levels(high, low, window_days=252)
        >>> levels[['52w_high', '52w_low']].iloc[-1]
        52w_high    112.0
        52w_low      95.0
        Name: 299, dtype: float64
    """
    # Calculate rolling 52-week high and low
    week_52_high = high.rolling(window=window_days, min_periods=window_days).max()
    week_52_low = low.rolling(window=window_days, min_periods=window_days).min()

    # Calculate percentage distance from current level to 52w levels
    # Negative = below level, Positive = above level
    dist_from_high = ((high - week_52_high) / week_52_high.replace(0, 1e-10)) * 100
    dist_from_low = ((low - week_52_low) / week_52_low.replace(0, 1e-10)) * 100

    # Calculate position within 52-week range (0 = at low, 1 = at high)
    # This helps identify if price is near support or resistance
    range_width = week_52_high - week_52_low
    range_position = (high - week_52_low) / range_width.replace(0, 1e-10)
    range_position = range_position.clip(0, 1)  # Clamp to [0, 1]

    # Combine into DataFrame
    result = pd.DataFrame(
        {
            "52w_high": week_52_high,
            "52w_low": week_52_low,
            "dist_from_52w_high": dist_from_high,
            "dist_from_52w_low": dist_from_low,
            "range_position": range_position,
        },
        index=high.index,
    )

    return result


def calculate_anchored_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    anchor_date: pd.Timestamp | None = None,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Calculate Anchored VWAP with standard deviation bands.

    Anchored VWAP is VWAP calculated from a specific anchor point (e.g., earnings
    date, pivot high/low, year start) rather than cumulative from session start.
    It serves as a dynamic support/resistance level with bands showing volatility.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        anchor_date: Start date for VWAP calculation (default: series start)
        lookback_days: Alternative to anchor_date - lookback N days from end

    Returns:
        DataFrame with columns:
        - 'anchored_vwap': Volume-weighted average price from anchor
        - 'vwap_upper_1sd': Upper band (VWAP + 1 std dev)
        - 'vwap_lower_1sd': Lower band (VWAP - 1 std dev)
        - 'vwap_upper_2sd': Upper band (VWAP + 2 std dev)
        - 'vwap_lower_2sd': Lower band (VWAP - 2 std dev)
        - 'dist_from_vwap': Distance from close to VWAP (%)

    Example:
        >>> high = pd.Series([102, 104, 103, 105, 107])
        >>> low = pd.Series([98, 100, 99, 101, 103])
        >>> close = pd.Series([100, 102, 101, 103, 105])
        >>> volume = pd.Series([1000, 1100, 1050, 1200, 1150])
        >>> vwap = calculate_anchored_vwap(high, low, close, volume, lookback_days=5)
        >>> vwap['anchored_vwap'].iloc[-1] > 100
        True
    """
    # Calculate typical price
    typical_price = (high + low + close) / 3

    # Determine anchor point
    if anchor_date is not None:
        # Filter data from anchor_date onwards
        mask = high.index >= anchor_date
        typical_price = typical_price[mask]
        volume = volume[mask]
    elif lookback_days is not None:
        # Use last N days
        typical_price = typical_price.iloc[-lookback_days:]
        volume = volume.iloc[-lookback_days:]

    # Calculate anchored VWAP
    cumulative_tp_volume = (typical_price * volume).cumsum()
    cumulative_volume = volume.cumsum()
    anchored_vwap = cumulative_tp_volume / cumulative_volume.replace(0, 1e-10)

    # Calculate standard deviation of typical price from VWAP
    # Weighted by volume for consistency
    squared_diff = ((typical_price - anchored_vwap) ** 2) * volume
    cumulative_squared_diff = squared_diff.cumsum()
    variance = cumulative_squared_diff / cumulative_volume.replace(0, 1e-10)
    std_dev = variance**0.5

    # Calculate bands
    vwap_upper_1sd = anchored_vwap + std_dev
    vwap_lower_1sd = anchored_vwap - std_dev
    vwap_upper_2sd = anchored_vwap + (2 * std_dev)
    vwap_lower_2sd = anchored_vwap - (2 * std_dev)

    # Calculate distance from close to VWAP (%)
    # Use original close series aligned with anchored VWAP index
    close_aligned = close.reindex(anchored_vwap.index)
    dist_from_vwap = ((close_aligned - anchored_vwap) / anchored_vwap.replace(0, 1e-10)) * 100

    # Combine into DataFrame
    result = pd.DataFrame(
        {
            "anchored_vwap": anchored_vwap,
            "vwap_upper_1sd": vwap_upper_1sd,
            "vwap_lower_1sd": vwap_lower_1sd,
            "vwap_upper_2sd": vwap_upper_2sd,
            "vwap_lower_2sd": vwap_lower_2sd,
            "dist_from_vwap": dist_from_vwap,
        },
        index=anchored_vwap.index,
    )

    # Reindex to original series index (fill forward for missing values)
    result = result.reindex(close.index, method="ffill")

    return result


def calculate_volume_profile_levels(
    close: pd.Series,
    volume: pd.Series,
    lookback_days: int = 252,
    num_bins: int = 50,
    top_n_levels: int = 5,
) -> pd.DataFrame:
    """Calculate Volume Profile support/resistance levels.

    Volume Profile identifies price levels with high traded volume, which act as
    significant support/resistance zones. This implementation uses a rolling
    window to identify current relevant levels.

    Args:
        close: Close price series
        volume: Volume series
        lookback_days: Lookback window for volume profile (default 252 ≈ 1 year)
        num_bins: Number of price bins for volume histogram (default 50)
        top_n_levels: Number of top volume levels to return (default 5)

    Returns:
        DataFrame with columns:
        - 'vp_level_1' to 'vp_level_N': Top N volume profile price levels
        - 'vp_volume_1' to 'vp_volume_N': Corresponding volume at each level
        - 'dist_to_nearest_level': Distance to nearest volume profile level (%)
        - 'nearest_level_price': Price of nearest volume profile level
        - 'poc': Point of Control (price level with highest volume)

    Example:
        >>> close = pd.Series([100 + i % 10 for i in range(300)])
        >>> volume = pd.Series([1000 + (i % 5) * 100 for i in range(300)])
        >>> vp = calculate_volume_profile_levels(close, volume, lookback_days=252)
        >>> vp['poc'].iloc[-1] > 0  # Point of Control exists
        True
    """
    # Initialize result DataFrame
    result = pd.DataFrame(index=close.index)

    # Calculate rolling volume profile for each row
    for i in range(len(close)):
        # Get lookback window
        start_idx = max(0, i - lookback_days + 1)
        window_close = close.iloc[start_idx : i + 1]
        window_volume = volume.iloc[start_idx : i + 1]

        if len(window_close) < lookback_days // 2:
            # Insufficient data - set NaN
            result.loc[close.index[i], "poc"] = float("nan")
            for n in range(1, top_n_levels + 1):
                result.loc[close.index[i], f"vp_level_{n}"] = float("nan")
                result.loc[close.index[i], f"vp_volume_{n}"] = float("nan")
            result.loc[close.index[i], "dist_to_nearest_level"] = float("nan")
            result.loc[close.index[i], "nearest_level_price"] = float("nan")
            continue

        # Create price bins
        price_min = window_close.min()
        price_max = window_close.max()
        if price_max == price_min:
            # No price movement - set single level
            result.loc[close.index[i], "poc"] = price_min
            result.loc[close.index[i], "vp_level_1"] = price_min
            result.loc[close.index[i], "vp_volume_1"] = window_volume.sum()
            for n in range(2, top_n_levels + 1):
                result.loc[close.index[i], f"vp_level_{n}"] = float("nan")
                result.loc[close.index[i], f"vp_volume_{n}"] = float("nan")
            result.loc[close.index[i], "dist_to_nearest_level"] = 0.0
            result.loc[close.index[i], "nearest_level_price"] = price_min
            continue

        # Bin prices and aggregate volume
        bins = pd.cut(window_close, bins=num_bins, include_lowest=True)
        volume_by_price = window_volume.groupby(bins).sum().sort_values(ascending=False)

        # Get top N levels
        top_levels = volume_by_price.head(top_n_levels)

        # Extract midpoint of each bin as the level price
        for n, (bin_interval, vol) in enumerate(top_levels.items(), start=1):
            level_price = (bin_interval.left + bin_interval.right) / 2
            result.loc[close.index[i], f"vp_level_{n}"] = level_price
            result.loc[close.index[i], f"vp_volume_{n}"] = vol

        # Fill remaining levels with NaN if fewer than top_n_levels exist
        for n in range(len(top_levels) + 1, top_n_levels + 1):
            result.loc[close.index[i], f"vp_level_{n}"] = float("nan")
            result.loc[close.index[i], f"vp_volume_{n}"] = float("nan")

        # Point of Control (POC) is the level with highest volume
        poc_bin = volume_by_price.index[0]
        poc_price = (poc_bin.left + poc_bin.right) / 2
        result.loc[close.index[i], "poc"] = poc_price

        # Calculate distance to nearest level
        current_price = close.iloc[i]
        level_prices = [
            result.loc[close.index[i], f"vp_level_{n}"]
            for n in range(1, top_n_levels + 1)
            if not pd.isna(result.loc[close.index[i], f"vp_level_{n}"])
        ]

        if level_prices:
            distances = [abs(current_price - lp) for lp in level_prices]
            nearest_idx = distances.index(min(distances))
            nearest_level = level_prices[nearest_idx]
            dist_pct = ((current_price - nearest_level) / nearest_level) * 100

            result.loc[close.index[i], "dist_to_nearest_level"] = dist_pct
            result.loc[close.index[i], "nearest_level_price"] = nearest_level
        else:
            result.loc[close.index[i], "dist_to_nearest_level"] = float("nan")
            result.loc[close.index[i], "nearest_level_price"] = float("nan")

    return result


def calculate_swing_highs_lows(
    high: pd.Series,
    low: pd.Series,
    lookback_left: int = 5,
    lookback_right: int = 5,
) -> pd.DataFrame:
    """Identify swing highs and swing lows as support/resistance levels.

    A swing high is a peak where the high is greater than N bars to the left
    and N bars to the right. A swing low is a trough where the low is less than
    N bars to the left and right. These pivots act as key support/resistance.

    Args:
        high: High price series
        low: Low price series
        lookback_left: Number of bars to the left for comparison (default 5)
        lookback_right: Number of bars to the right for comparison (default 5)

    Returns:
        DataFrame with columns:
        - 'is_swing_high': Boolean, True if bar is a swing high
        - 'is_swing_low': Boolean, True if bar is a swing low
        - 'last_swing_high': Price of most recent swing high
        - 'last_swing_low': Price of most recent swing low
        - 'bars_since_swing_high': Number of bars since last swing high
        - 'bars_since_swing_low': Number of bars since last swing low

    Example:
        >>> high = pd.Series([100, 105, 110, 108, 107, 112, 109, 108])
        >>> low = pd.Series([95, 100, 105, 103, 102, 107, 104, 103])
        >>> swings = calculate_swing_highs_lows(high, low, lookback_left=2, lookback_right=2)
        >>> swings['is_swing_high'].sum() >= 0  # At least one swing high
        True
    """
    # Initialize result DataFrame
    is_swing_high = pd.Series(False, index=high.index)
    is_swing_low = pd.Series(False, index=low.index)

    # Identify swing highs and lows
    for i in range(lookback_left, len(high) - lookback_right):
        # Check if current high is greater than left and right neighbors
        left_highs = high.iloc[i - lookback_left : i]
        right_highs = high.iloc[i + 1 : i + 1 + lookback_right]
        current_high = high.iloc[i]

        if (current_high > left_highs.max()) and (current_high > right_highs.max()):
            is_swing_high.iloc[i] = True

        # Check if current low is less than left and right neighbors
        left_lows = low.iloc[i - lookback_left : i]
        right_lows = low.iloc[i + 1 : i + 1 + lookback_right]
        current_low = low.iloc[i]

        if (current_low < left_lows.min()) and (current_low < right_lows.min()):
            is_swing_low.iloc[i] = True

    # Track last swing high/low and bars since
    last_swing_high = pd.Series(float("nan"), index=high.index)
    last_swing_low = pd.Series(float("nan"), index=low.index)
    bars_since_swing_high = pd.Series(float("nan"), index=high.index)
    bars_since_swing_low = pd.Series(float("nan"), index=low.index)

    last_high_price = float("nan")
    last_low_price = float("nan")
    last_high_idx = -1
    last_low_idx = -1

    for i in range(len(high)):
        # Update last swing high
        if is_swing_high.iloc[i]:
            last_high_price = high.iloc[i]
            last_high_idx = i

        last_swing_high.iloc[i] = last_high_price
        if last_high_idx >= 0:
            bars_since_swing_high.iloc[i] = i - last_high_idx

        # Update last swing low
        if is_swing_low.iloc[i]:
            last_low_price = low.iloc[i]
            last_low_idx = i

        last_swing_low.iloc[i] = last_low_price
        if last_low_idx >= 0:
            bars_since_swing_low.iloc[i] = i - last_low_idx

    # Combine into DataFrame
    result = pd.DataFrame(
        {
            "is_swing_high": is_swing_high,
            "is_swing_low": is_swing_low,
            "last_swing_high": last_swing_high,
            "last_swing_low": last_swing_low,
            "bars_since_swing_high": bars_since_swing_high,
            "bars_since_swing_low": bars_since_swing_low,
        },
        index=high.index,
    )

    return result
