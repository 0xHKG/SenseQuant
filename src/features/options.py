"""Options chain feature engineering (US-029 Phase 2).

Transforms raw options chain data into normalized trading features:
- IV features (percentile, ATM/OTM IV, IV rank)
- Skew features (put-call IV skew, strike-to-strike skew)
- Volume/OI features (put-call ratios, total volume ratios)
- Greeks aggregates (delta, gamma, vega across strikes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def calculate_iv_features(
    chain_df: pd.DataFrame,
    iv_lookback_days: int = 30,
) -> pd.DataFrame:
    """Calculate implied volatility features from options chain.

    Args:
        chain_df: DataFrame with columns [date, strike, expiry, call_iv, put_iv, underlying_price]
        iv_lookback_days: Lookback period for IV percentile calculation

    Returns:
        DataFrame with columns [date, opt_iv_atm, opt_iv_percentile, opt_iv_rank,
                               opt_iv_otm_call, opt_iv_otm_put]
    """
    if chain_df.empty:
        logger.warning("Empty chain DataFrame for IV features")
        return pd.DataFrame()

    features = []

    # Group by date
    for date, group in chain_df.groupby("date"):
        try:
            underlying_price = group["underlying_price"].iloc[0]

            # Find ATM strike (closest to underlying price)
            group["strike_diff"] = abs(group["strike"] - underlying_price)
            atm_idx = group["strike_diff"].idxmin()
            atm_row = group.loc[atm_idx]

            # ATM IV (average of call and put)
            call_iv_atm = atm_row.get("call_iv", 0)
            put_iv_atm = atm_row.get("put_iv", 0)
            iv_atm = (call_iv_atm + put_iv_atm) / 2

            # OTM IVs (5% OTM)
            otm_call_strike = underlying_price * 1.05
            otm_put_strike = underlying_price * 0.95

            # Find nearest strikes
            group["otm_call_diff"] = abs(group["strike"] - otm_call_strike)
            group["otm_put_diff"] = abs(group["strike"] - otm_put_strike)

            otm_call_idx = group["otm_call_diff"].idxmin()
            otm_put_idx = group["otm_put_diff"].idxmin()

            iv_otm_call = group.loc[otm_call_idx].get("call_iv", 0)
            iv_otm_put = group.loc[otm_put_idx].get("put_iv", 0)

            # IV percentile and rank (requires historical context)
            # For now, use simple rolling calculation
            iv_percentile = 50.0  # Placeholder - would need historical IV data
            iv_rank = 50.0  # Placeholder - would need historical IV data

            features.append(
                {
                    "date": date,
                    "opt_iv_atm": iv_atm,
                    "opt_iv_percentile": iv_percentile,
                    "opt_iv_rank": iv_rank,
                    "opt_iv_otm_call": iv_otm_call,
                    "opt_iv_otm_put": iv_otm_put,
                }
            )

        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Error calculating IV features for {date}: {e}")
            continue

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    return df


def calculate_skew_features(
    chain_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate IV skew features from options chain.

    Args:
        chain_df: DataFrame with columns [date, strike, call_iv, put_iv, underlying_price]

    Returns:
        DataFrame with columns [date, opt_skew_put_call, opt_skew_25delta,
                               opt_skew_slope, opt_skew_curvature]
    """
    if chain_df.empty:
        logger.warning("Empty chain DataFrame for skew features")
        return pd.DataFrame()

    features = []

    for date, group in chain_df.groupby("date"):
        try:
            underlying_price = group["underlying_price"].iloc[0]

            # ATM skew (put IV - call IV at ATM)
            group["strike_diff"] = abs(group["strike"] - underlying_price)
            atm_idx = group["strike_diff"].idxmin()
            atm_row = group.loc[atm_idx]

            call_iv_atm = atm_row.get("call_iv", 0)
            put_iv_atm = atm_row.get("put_iv", 0)
            skew_put_call = put_iv_atm - call_iv_atm

            # 25-delta skew (approximation using OTM options)
            # 25-delta roughly corresponds to 10-15% OTM
            otm_call_strike = underlying_price * 1.10
            otm_put_strike = underlying_price * 0.90

            group["call_25d_diff"] = abs(group["strike"] - otm_call_strike)
            group["put_25d_diff"] = abs(group["strike"] - otm_put_strike)

            call_25d_idx = group["call_25d_diff"].idxmin()
            put_25d_idx = group["put_25d_diff"].idxmin()

            call_iv_25d = group.loc[call_25d_idx].get("call_iv", 0)
            put_iv_25d = group.loc[put_25d_idx].get("put_iv", 0)
            skew_25delta = put_iv_25d - call_iv_25d

            # IV curve slope (linear regression of IV vs moneyness)
            group["moneyness"] = group["strike"] / underlying_price
            group["avg_iv"] = (group["call_iv"] + group["put_iv"]) / 2

            if len(group) >= 3:
                # Simple linear regression
                x = group["moneyness"].values
                y = group["avg_iv"].values

                # Remove NaN
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]

                if len(x) >= 2:
                    slope = np.polyfit(x, y, 1)[0]
                else:
                    slope = 0
            else:
                slope = 0

            # IV smile curvature (quadratic fit)
            if len(group) >= 4:
                try:
                    curvature = np.polyfit(x, y, 2)[0]  # a in ax^2 + bx + c
                except (np.linalg.LinAlgError, TypeError):
                    curvature = 0
            else:
                curvature = 0

            features.append(
                {
                    "date": date,
                    "opt_skew_put_call": skew_put_call,
                    "opt_skew_25delta": skew_25delta,
                    "opt_skew_slope": slope,
                    "opt_skew_curvature": curvature,
                }
            )

        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.warning(f"Error calculating skew features for {date}: {e}")
            continue

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    return df


def calculate_volume_features(
    chain_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate volume and open interest features.

    Args:
        chain_df: DataFrame with columns [date, call_volume, put_volume, call_oi, put_oi]

    Returns:
        DataFrame with columns [date, opt_put_call_ratio_volume,
                               opt_put_call_ratio_oi, opt_total_volume_ratio]
    """
    if chain_df.empty:
        logger.warning("Empty chain DataFrame for volume features")
        return pd.DataFrame()

    features = []

    for date, group in chain_df.groupby("date"):
        try:
            # Aggregate volume and OI across all strikes
            total_call_volume = group["call_volume"].sum()
            total_put_volume = group["put_volume"].sum()
            total_call_oi = group["call_oi"].sum()
            total_put_oi = group["put_oi"].sum()

            # Put-call ratios
            pcr_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 1.0
            pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

            # Total options volume ratio (relative to underlying)
            # Note: would need underlying volume for accurate ratio
            # Using placeholder calculation
            total_options_volume = total_call_volume + total_put_volume
            total_volume_ratio = total_options_volume / 1000000  # Normalized placeholder

            features.append(
                {
                    "date": date,
                    "opt_put_call_ratio_volume": pcr_volume,
                    "opt_put_call_ratio_oi": pcr_oi,
                    "opt_total_volume_ratio": total_volume_ratio,
                }
            )

        except (KeyError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating volume features for {date}: {e}")
            continue

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    return df


def calculate_greeks_aggregates(
    chain_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate aggregate Greeks across strikes.

    Note: This is a placeholder implementation. Real Greeks require
    Black-Scholes calculations. For now, we return mock values.

    Args:
        chain_df: DataFrame with options chain data

    Returns:
        DataFrame with columns [date, opt_aggregate_delta,
                               opt_aggregate_gamma, opt_aggregate_vega]
    """
    if chain_df.empty:
        logger.warning("Empty chain DataFrame for Greeks aggregates")
        return pd.DataFrame()

    features = []

    for date, group in chain_df.groupby("date"):
        try:
            # Placeholder: Real implementation would calculate Black-Scholes Greeks
            # For now, return normalized aggregates based on volume/OI
            total_call_oi = group["call_oi"].sum()
            total_put_oi = group["put_oi"].sum()

            # Aggregate delta (approximation: net directional exposure)
            # Positive = bullish, negative = bearish
            aggregate_delta = (total_call_oi - total_put_oi) / (total_call_oi + total_put_oi)

            # Aggregate gamma (approximation: curvature exposure)
            # Higher when ATM options have high OI
            aggregate_gamma = 0.5  # Placeholder

            # Aggregate vega (approximation: volatility exposure)
            # Higher when longer-dated options have high OI
            aggregate_vega = 0.5  # Placeholder

            features.append(
                {
                    "date": date,
                    "opt_aggregate_delta": aggregate_delta,
                    "opt_aggregate_gamma": aggregate_gamma,
                    "opt_aggregate_vega": aggregate_vega,
                }
            )

        except (KeyError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating Greeks aggregates for {date}: {e}")
            continue

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    return df


def compute_all_options_features(
    chain_df: pd.DataFrame,
    iv_lookback_days: int = 30,
) -> pd.DataFrame:
    """Compute all options features in one call.

    Args:
        chain_df: DataFrame with columns [date, strike, expiry, call_iv, put_iv,
                                         call_volume, put_volume, call_oi, put_oi,
                                         underlying_price]
        iv_lookback_days: Lookback period for IV calculations

    Returns:
        DataFrame with all options features, indexed by date
    """
    if chain_df.empty:
        logger.warning("Empty chain DataFrame for options features")
        return pd.DataFrame()

    # Compute individual feature sets
    iv_features = calculate_iv_features(chain_df, iv_lookback_days)
    skew_features = calculate_skew_features(chain_df)
    volume_features = calculate_volume_features(chain_df)
    greeks_features = calculate_greeks_aggregates(chain_df)

    # Merge all features
    if iv_features.empty:
        return pd.DataFrame()

    features = iv_features
    if not skew_features.empty:
        features = features.join(skew_features, how="outer")
    if not volume_features.empty:
        features = features.join(volume_features, how="outer")
    if not greeks_features.empty:
        features = features.join(greeks_features, how="outer")

    # Fill NaN
    features = features.ffill().bfill()

    logger.info(
        f"Computed {len(features)} options feature rows with {len(features.columns)} features"
    )

    return features
