"""Order book feature engineering (US-029 Phase 2).

Transforms raw order book snapshots into normalized trading features:
- Spread features (bid-ask spread metrics)
- Depth imbalance (order book skew at multiple levels)
- Order flow metrics (buying/selling pressure)
- Liquidity features (volume-weighted prices, market impact)
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


def calculate_spread_features(
    snapshots: pd.DataFrame,
    lookback_window: int = 60,
) -> pd.DataFrame:
    """Calculate bid-ask spread features from order book snapshots.

    Args:
        snapshots: DataFrame with columns [timestamp, bids, asks]
                  bids/asks are lists of dicts with price/quantity/orders
        lookback_window: Lookback window in seconds for time-weighted metrics

    Returns:
        DataFrame with columns [timestamp, ob_spread_abs, ob_spread_rel,
                               ob_spread_pct, ob_time_weighted_spread]
        Indexed by timestamp
    """
    if snapshots.empty:
        logger.warning("Empty snapshots DataFrame for spread features")
        return pd.DataFrame()

    features = []

    for _, row in snapshots.iterrows():
        try:
            bids = row["bids"]
            asks = row["asks"]

            if not bids or not asks:
                continue

            # Best bid/ask
            best_bid = bids[0]["price"]
            best_ask = asks[0]["price"]

            # Calculate spread metrics
            spread_abs = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            spread_rel = spread_abs / mid_price if mid_price > 0 else 0
            spread_pct = (spread_abs / mid_price) * 100 if mid_price > 0 else 0

            features.append(
                {
                    "timestamp": row["timestamp"],
                    "ob_spread_abs": spread_abs,
                    "ob_spread_rel": spread_rel,
                    "ob_spread_pct": spread_pct,
                }
            )

        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Error calculating spread features: {e}")
            continue

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # Calculate time-weighted spread over lookback window
    if len(df) > 1:
        df["ob_time_weighted_spread"] = (
            df["ob_spread_abs"].rolling(window=f"{lookback_window}s").mean()
        )
    else:
        df["ob_time_weighted_spread"] = df["ob_spread_abs"]

    return df


def calculate_depth_imbalance(
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate order book depth imbalance features.

    Args:
        snapshots: DataFrame with columns [timestamp, bids, asks]

    Returns:
        DataFrame with columns [timestamp, ob_depth_imbalance_l1,
                               ob_depth_imbalance_l5, ob_volume_weighted_price_bid,
                               ob_volume_weighted_price_ask]
    """
    if snapshots.empty:
        logger.warning("Empty snapshots DataFrame for depth imbalance")
        return pd.DataFrame()

    features = []

    for _, row in snapshots.iterrows():
        try:
            bids = row["bids"]
            asks = row["asks"]

            if not bids or not asks:
                continue

            # Level 1 imbalance (best bid/ask)
            bid_qty_l1 = bids[0]["quantity"]
            ask_qty_l1 = asks[0]["quantity"]
            total_qty_l1 = bid_qty_l1 + ask_qty_l1
            imbalance_l1 = (bid_qty_l1 - ask_qty_l1) / total_qty_l1 if total_qty_l1 > 0 else 0

            # Level 5 aggregate imbalance (sum of top 5 levels)
            bid_qty_l5 = sum(b["quantity"] for b in bids[:5])
            ask_qty_l5 = sum(a["quantity"] for a in asks[:5])
            total_qty_l5 = bid_qty_l5 + ask_qty_l5
            imbalance_l5 = (bid_qty_l5 - ask_qty_l5) / total_qty_l5 if total_qty_l5 > 0 else 0

            # Volume-weighted prices (top 5 levels)
            vwap_bid = (
                sum(b["price"] * b["quantity"] for b in bids[:5]) / bid_qty_l5
                if bid_qty_l5 > 0
                else bids[0]["price"]
            )

            vwap_ask = (
                sum(a["price"] * a["quantity"] for a in asks[:5]) / ask_qty_l5
                if ask_qty_l5 > 0
                else asks[0]["price"]
            )

            features.append(
                {
                    "timestamp": row["timestamp"],
                    "ob_depth_imbalance_l1": imbalance_l1,
                    "ob_depth_imbalance_l5": imbalance_l5,
                    "ob_volume_weighted_price_bid": vwap_bid,
                    "ob_volume_weighted_price_ask": vwap_ask,
                }
            )

        except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating depth imbalance: {e}")
            continue

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    return df


def calculate_order_flow_metrics(
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate order flow and market pressure metrics.

    Args:
        snapshots: DataFrame with columns [timestamp, bids, asks]

    Returns:
        DataFrame with columns [timestamp, ob_order_flow_ratio,
                               ob_market_pressure, ob_liquidity_imbalance,
                               ob_effective_spread]
    """
    if snapshots.empty:
        logger.warning("Empty snapshots DataFrame for order flow metrics")
        return pd.DataFrame()

    features = []

    for _, row in snapshots.iterrows():
        try:
            bids = row["bids"]
            asks = row["asks"]

            if not bids or not asks:
                continue

            # Order flow ratio (number of orders on bid vs ask side)
            bid_orders = sum(b.get("orders", 1) for b in bids[:5])
            ask_orders = sum(a.get("orders", 1) for a in asks[:5])
            order_flow_ratio = bid_orders / ask_orders if ask_orders > 0 else 1.0

            # Market pressure (weighted by price distance from mid)
            best_bid = bids[0]["price"]
            best_ask = asks[0]["price"]
            mid_price = (best_bid + best_ask) / 2

            bid_pressure = sum(
                b["quantity"] * (1 - abs(b["price"] - mid_price) / mid_price) for b in bids[:5]
            )

            ask_pressure = sum(
                a["quantity"] * (1 - abs(a["price"] - mid_price) / mid_price) for a in asks[:5]
            )

            total_pressure = bid_pressure + ask_pressure
            market_pressure = (
                (bid_pressure - ask_pressure) / total_pressure if total_pressure > 0 else 0
            )

            # Liquidity imbalance (total quantity difference)
            bid_liquidity = sum(b["quantity"] for b in bids[:5])
            ask_liquidity = sum(a["quantity"] for a in asks[:5])
            total_liquidity = bid_liquidity + ask_liquidity
            liquidity_imbalance = (
                (bid_liquidity - ask_liquidity) / total_liquidity if total_liquidity > 0 else 0
            )

            # Effective spread (considering depth)
            # Approximation: spread adjusted for available liquidity
            spread_abs = best_ask - best_bid
            effective_spread = spread_abs * (1 + abs(liquidity_imbalance))

            features.append(
                {
                    "timestamp": row["timestamp"],
                    "ob_order_flow_ratio": order_flow_ratio,
                    "ob_market_pressure": market_pressure,
                    "ob_liquidity_imbalance": liquidity_imbalance,
                    "ob_effective_spread": effective_spread,
                }
            )

        except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating order flow metrics: {e}")
            continue

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    return df


def calculate_liquidity_features(
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate liquidity and market impact features.

    Args:
        snapshots: DataFrame with columns [timestamp, bids, asks]

    Returns:
        DataFrame with columns [timestamp, ob_bid_liquidity, ob_ask_liquidity,
                               ob_total_liquidity, ob_market_impact_estimate]
    """
    if snapshots.empty:
        logger.warning("Empty snapshots DataFrame for liquidity features")
        return pd.DataFrame()

    features = []

    for _, row in snapshots.iterrows():
        try:
            bids = row["bids"]
            asks = row["asks"]

            if not bids or not asks:
                continue

            # Total liquidity at each side (top 5 levels)
            bid_liquidity = sum(b["quantity"] for b in bids[:5])
            ask_liquidity = sum(a["quantity"] for a in asks[:5])
            total_liquidity = bid_liquidity + ask_liquidity

            # Market impact estimate (Kyle's lambda approximation)
            # Impact = price_change / volume
            # Approximation: spread / (2 * sqrt(total_liquidity))
            best_bid = bids[0]["price"]
            best_ask = asks[0]["price"]
            spread = best_ask - best_bid

            market_impact = spread / (2 * (total_liquidity**0.5)) if total_liquidity > 0 else 0

            features.append(
                {
                    "timestamp": row["timestamp"],
                    "ob_bid_liquidity": bid_liquidity,
                    "ob_ask_liquidity": ask_liquidity,
                    "ob_total_liquidity": total_liquidity,
                    "ob_market_impact_estimate": market_impact,
                }
            )

        except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating liquidity features: {e}")
            continue

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    return df


def compute_all_order_book_features(
    snapshots: pd.DataFrame,
    lookback_window: int = 60,
) -> pd.DataFrame:
    """Compute all order book features in one call.

    Args:
        snapshots: DataFrame with columns [timestamp, bids, asks]
        lookback_window: Lookback window in seconds

    Returns:
        DataFrame with all order book features, indexed by timestamp
    """
    if snapshots.empty:
        logger.warning("Empty snapshots DataFrame for order book features")
        return pd.DataFrame()

    # Compute individual feature sets
    spread_features = calculate_spread_features(snapshots, lookback_window)
    depth_features = calculate_depth_imbalance(snapshots)
    flow_features = calculate_order_flow_metrics(snapshots)
    liquidity_features = calculate_liquidity_features(snapshots)

    # Merge all features
    if spread_features.empty:
        return pd.DataFrame()

    features = spread_features
    if not depth_features.empty:
        features = features.join(depth_features, how="outer")
    if not flow_features.empty:
        features = features.join(flow_features, how="outer")
    if not liquidity_features.empty:
        features = features.join(liquidity_features, how="outer")

    # Fill NaN with forward fill then backward fill
    features = features.ffill().bfill()

    logger.info(
        f"Computed {len(features)} order book feature rows with {len(features.columns)} features"
    )

    return features
