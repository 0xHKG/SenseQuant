"""Technical indicator library for trading strategies.

This module provides reusable technical indicator functions for use across
all trading strategies. All functions accept pandas Series as input and return
pandas Series (or tuples of Series) as output.

All indicators handle edge cases (insufficient data, NaNs, zeros) gracefully
and return NaN for periods where calculation is not possible.
"""

from __future__ import annotations

import pandas as pd


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        series: Price series (typically close prices)
        period: Number of periods for the moving average

    Returns:
        Series with SMA values (NaN for initial periods)

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105])
        >>> sma = calculate_sma(prices, period=3)
        >>> sma.iloc[-1]  # Average of last 3 prices
        103.0
    """
    return series.rolling(window=period, min_periods=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average.

    EMA gives more weight to recent prices compared to SMA.

    Args:
        series: Price series (typically close prices)
        period: Number of periods for the moving average (span parameter)

    Returns:
        Series with EMA values (NaN for initial periods)

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105])
        >>> ema = calculate_ema(prices, period=3)
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    Values range from 0 to 100, with 70+ indicating overbought and 30- indicating oversold.

    Args:
        series: Price series (typically close prices)
        period: Number of periods for RSI calculation (default 14)

    Returns:
        Series with RSI values (0-100 range, NaN for initial periods)

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> rsi = calculate_rsi(prices, period=6)
        >>> 0 <= rsi.iloc[-1] <= 100
        True
    """
    # Calculate price changes
    delta = series.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0.0)  # type: ignore[operator]
    loss = -delta.where(delta < 0, 0.0)  # type: ignore[operator]

    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Calculate relative strength (RS)
    # Avoid division by zero by replacing 0 with small value
    rs = avg_gain / avg_loss.replace(0, 1e-10)

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range (ATR).

    ATR is a volatility indicator that measures the average range of price movement.
    Higher ATR indicates higher volatility.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods for ATR calculation (default 14)

    Returns:
        Series with ATR values (NaN for initial periods)

    Example:
        >>> high = pd.Series([102, 104, 103, 105, 107])
        >>> low = pd.Series([98, 100, 99, 101, 103])
        >>> close = pd.Series([100, 102, 101, 103, 105])
        >>> atr = calculate_atr(high, low, close, period=3)
    """
    # Calculate true range components
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()

    # True range is the maximum of the three components
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR is the moving average of true range
    atr = true_range.rolling(window=period, min_periods=period).mean()

    return atr


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate Volume-Weighted Average Price (VWAP).

    VWAP is the average price weighted by volume. It's commonly used as an
    intraday benchmark for institutional traders.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series

    Returns:
        Series with VWAP values

    Example:
        >>> high = pd.Series([102, 104, 103])
        >>> low = pd.Series([98, 100, 99])
        >>> close = pd.Series([100, 102, 101])
        >>> volume = pd.Series([1000, 1100, 1050])
        >>> vwap = calculate_vwap(high, low, close, volume)
    """
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3

    # VWAP = cumulative(typical_price * volume) / cumulative(volume)
    cumulative_tp_volume = (typical_price * volume).cumsum()
    cumulative_volume = volume.cumsum()

    # Avoid division by zero
    vwap = cumulative_tp_volume / cumulative_volume.replace(0, 1e-10)

    return vwap


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Bollinger Bands consist of a middle band (SMA) and upper/lower bands that are
    standard deviations away from the middle band. They measure volatility and
    provide dynamic support/resistance levels.

    Args:
        series: Price series (typically close prices)
        period: Number of periods for the moving average (default 20)
        num_std: Number of standard deviations for bands (default 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band) Series

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> upper, middle, lower = calculate_bollinger_bands(prices, period=5)
        >>> upper.iloc[-1] > middle.iloc[-1] > lower.iloc[-1]
        True
    """
    # Middle band is the SMA
    middle_band = series.rolling(window=period, min_periods=period).mean()

    # Calculate standard deviation
    std = series.rolling(window=period, min_periods=period).std()

    # Upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)

    return upper_band, middle_band, lower_band


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).

    MACD is a trend-following momentum indicator that shows the relationship
    between two exponential moving averages.

    Args:
        series: Price series (typically close prices)
        fast_period: Period for fast EMA (default 12)
        slow_period: Period for slow EMA (default 26)
        signal_period: Period for signal line EMA (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram) Series

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106] * 5)
        >>> macd, signal, histogram = calculate_macd(prices)
        >>> histogram.iloc[-1] == macd.iloc[-1] - signal.iloc[-1]
        True
    """
    # Calculate fast and slow EMAs
    ema_fast = series.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()

    # MACD line is the difference between fast and slow EMAs
    macd_line = ema_fast - ema_slow

    # Signal line is the EMA of the MACD line
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()

    # Histogram is the difference between MACD line and signal line
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average Directional Index (ADX).

    ADX measures trend strength (not direction) on a scale of 0-100.
    Values above 25 indicate a strong trend, while values below 20 indicate a weak trend.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods for ADX calculation (default 14)

    Returns:
        Series with ADX values (0-100 range, NaN for initial periods)

    Example:
        >>> high = pd.Series([102, 104, 103, 105, 107] * 5)
        >>> low = pd.Series([98, 100, 99, 101, 103] * 5)
        >>> close = pd.Series([100, 102, 101, 103, 105] * 5)
        >>> adx = calculate_adx(high, low, close, period=14)
    """
    # Calculate +DM and -DM (Directional Movement)
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)  # type: ignore[operator]
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)  # type: ignore[operator]

    # Calculate ATR
    atr = calculate_atr(high, low, close, period=period)

    # Calculate +DI and -DI (Directional Indicators)
    # Use smoothed +DM and -DM
    plus_dm_smooth = plus_dm.rolling(window=period, min_periods=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period, min_periods=period).sum()

    plus_di = 100 * (plus_dm_smooth / atr.replace(0, 1e-10))
    minus_di = 100 * (minus_dm_smooth / atr.replace(0, 1e-10))

    # Calculate DX (Directional Index)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)

    # ADX is the smoothed DX
    adx = dx.rolling(window=period, min_periods=period).mean()

    return adx


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume (OBV).

    OBV is a cumulative volume-based indicator that shows the relationship
    between volume and price changes. It's used to confirm trends and identify
    potential reversals.

    Args:
        close: Close price series
        volume: Volume series

    Returns:
        Series with OBV values (cumulative volume)

    Example:
        >>> close = pd.Series([100, 102, 101, 103, 105])
        >>> volume = pd.Series([1000, 1100, 1050, 1200, 1150])
        >>> obv = calculate_obv(close, volume)
        >>> isinstance(obv.iloc[-1], (int, float))
        True
    """
    # Calculate price direction (1 for up, -1 for down, 0 for unchanged)
    price_change = close.diff()
    direction = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # OBV is cumulative sum of (direction * volume)
    obv = (direction * volume).cumsum()

    return obv


# =====================================================================
# US-029 Phase 2: Market Data Feature Integration
# =====================================================================


def compute_order_book_features(
    symbol: str,
    from_date,
    to_date,
    lookback_window: int = 60,
) -> pd.DataFrame:
    """Compute order book features for symbol/date range (US-029 Phase 2).

    Returns empty DataFrame if no order book data available.
    Logs warning if data missing.

    Args:
        symbol: Stock symbol
        from_date: Start date (datetime)
        to_date: End date (datetime)
        lookback_window: Lookback window in seconds

    Returns:
        DataFrame with order book features, indexed by timestamp
    """
    from datetime import datetime

    from loguru import logger

    from src.features.order_book import compute_all_order_book_features
    from src.services.data_feed import load_order_book_snapshots

    # Convert dates if needed
    if not isinstance(from_date, datetime):
        from_date = pd.to_datetime(from_date)
    if not isinstance(to_date, datetime):
        to_date = pd.to_datetime(to_date)

    # Load raw data
    snapshots = load_order_book_snapshots(symbol, from_date, to_date)
    if not snapshots:
        logger.warning(f"No order book data for {symbol} ({from_date} to {to_date})")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(snapshots)

    # Compute features
    features = compute_all_order_book_features(df, lookback_window)

    return features


def compute_options_features(
    symbol: str,
    from_date,
    to_date,
    iv_lookback_days: int = 30,
) -> pd.DataFrame:
    """Compute options features for symbol/date range (US-029 Phase 2).

    Returns empty DataFrame if no options data available.

    Args:
        symbol: Stock symbol (e.g., "NIFTY", "BANKNIFTY")
        from_date: Start date (datetime)
        to_date: End date (datetime)
        iv_lookback_days: IV lookback period in days

    Returns:
        DataFrame with options features, indexed by date
    """
    from datetime import datetime, timedelta

    from loguru import logger

    from src.features.options import compute_all_options_features
    from src.services.data_feed import load_options_chain

    # Convert dates if needed
    if not isinstance(from_date, datetime):
        from_date = pd.to_datetime(from_date)
    if not isinstance(to_date, datetime):
        to_date = pd.to_datetime(to_date)

    # Load options chains for date range
    all_chains = []
    current_date = from_date

    while current_date <= to_date:
        chain = load_options_chain(symbol, current_date)
        if chain:
            # Convert chain to DataFrame rows
            for option in chain.get("options", []):
                all_chains.append(
                    {
                        "date": chain["date"],
                        "strike": option["strike"],
                        "expiry": option["expiry"],
                        "call_iv": option["call"].get("iv", 0),
                        "put_iv": option["put"].get("iv", 0),
                        "call_volume": option["call"].get("volume", 0),
                        "put_volume": option["put"].get("volume", 0),
                        "call_oi": option["call"].get("oi", 0),
                        "put_oi": option["put"].get("oi", 0),
                        "underlying_price": chain["underlying_price"],
                    }
                )

        current_date += timedelta(days=1)

    if not all_chains:
        logger.warning(f"No options data for {symbol} ({from_date} to {to_date})")
        return pd.DataFrame()

    # Convert to DataFrame
    chain_df = pd.DataFrame(all_chains)
    chain_df["date"] = pd.to_datetime(chain_df["date"])

    # Compute features
    features = compute_all_options_features(chain_df, iv_lookback_days)

    return features


def compute_macro_features(
    indicators: list[str],
    from_date,
    to_date,
    short_window: int = 10,
    long_window: int = 50,
) -> pd.DataFrame:
    """Compute macro features for indicators/date range (US-029 Phase 2).

    Args:
        indicators: List of macro indicator names
        from_date: Start date (datetime)
        to_date: End date (datetime)
        short_window: Short MA window
        long_window: Long MA window

    Returns:
        DataFrame with macro features, indexed by date
    """
    from datetime import datetime

    from loguru import logger

    from src.features.macro import compute_all_macro_features
    from src.services.data_feed import load_macro_data

    # Convert dates if needed
    if not isinstance(from_date, datetime):
        from_date = pd.to_datetime(from_date)
    if not isinstance(to_date, datetime):
        to_date = pd.to_datetime(to_date)

    # Load macro data for all indicators
    all_macro_data = []

    for indicator in indicators:
        df = load_macro_data(indicator, from_date, to_date)
        if not df.empty:
            df["indicator"] = indicator
            all_macro_data.append(df)

    if not all_macro_data:
        logger.warning(f"No macro data for indicators {indicators}")
        return pd.DataFrame()

    # Combine all macro data
    macro_df = pd.concat(all_macro_data, ignore_index=True)

    # Compute features
    features = compute_all_macro_features(macro_df, short_window, long_window)

    return features


def compute_all_market_features(
    symbol: str,
    from_date,
    to_date,
    include_order_book: bool = False,
    include_options: bool = False,
    include_macro: bool = False,
    macro_indicators: list[str] | None = None,
) -> pd.DataFrame:
    """Unified interface to compute all available market features (US-029 Phase 2).

    Args:
        symbol: Stock symbol
        from_date: Start date
        to_date: End date
        include_order_book: Whether to include order book features
        include_options: Whether to include options features
        include_macro: Whether to include macro features
        macro_indicators: List of macro indicators (if None, uses defaults)

    Returns:
        DataFrame with all enabled market features
    """
    from loguru import logger

    features_dict = {}

    # Order book features
    if include_order_book:
        ob_features = compute_order_book_features(symbol, from_date, to_date)
        if not ob_features.empty:
            features_dict["order_book"] = ob_features
            logger.info(f"Loaded {len(ob_features)} order book feature rows")

    # Options features
    if include_options:
        opt_features = compute_options_features(symbol, from_date, to_date)
        if not opt_features.empty:
            features_dict["options"] = opt_features
            logger.info(f"Loaded {len(opt_features)} options feature rows")

    # Macro features
    if include_macro:
        if macro_indicators is None:
            macro_indicators = ["NIFTY50", "INDIAVIX", "USDINR"]

        macro_features = compute_macro_features(macro_indicators, from_date, to_date)
        if not macro_features.empty:
            features_dict["macro"] = macro_features
            logger.info(f"Loaded {len(macro_features)} macro feature rows")

    # Merge all features
    if not features_dict:
        logger.warning("No market features available")
        return pd.DataFrame()

    # Start with first available feature set
    combined = None
    for _feature_type, features in features_dict.items():
        if combined is None:
            combined = features
        else:
            # Join on index (timestamp or date)
            combined = combined.join(features, how="outer")

    # Fill NaN with forward/backward fill
    if combined is not None:
        combined = combined.ffill().bfill()

    logger.info(f"Computed {len(combined)} total market feature rows")

    return combined if combined is not None else pd.DataFrame()
