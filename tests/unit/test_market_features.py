"""Unit tests for market data feature engineering (US-029 Phase 2).

Tests feature computation functions with deterministic toy data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.macro import (
    calculate_momentum_features,
    calculate_regime_features,
    compute_all_macro_features,
)
from src.features.options import (
    calculate_iv_features,
    calculate_skew_features,
    calculate_volume_features,
    compute_all_options_features,
)
from src.features.order_book import (
    calculate_depth_imbalance,
    calculate_liquidity_features,
    calculate_order_flow_metrics,
    calculate_spread_features,
    compute_all_order_book_features,
)

# ==================================================================
# Order Book Feature Tests
# ==================================================================


def test_calculate_spread_features():
    """Test spread feature calculation with toy data."""
    snapshots = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-15 09:15:00",
                "bids": [
                    {"price": 2450.0, "quantity": 1000, "orders": 5},
                    {"price": 2449.5, "quantity": 1500, "orders": 7},
                ],
                "asks": [
                    {"price": 2451.0, "quantity": 800, "orders": 4},
                    {"price": 2451.5, "quantity": 1200, "orders": 6},
                ],
            },
            {
                "timestamp": "2025-01-15 09:16:00",
                "bids": [
                    {"price": 2451.0, "quantity": 1100, "orders": 6},
                    {"price": 2450.5, "quantity": 1400, "orders": 8},
                ],
                "asks": [
                    {"price": 2452.0, "quantity": 900, "orders": 5},
                    {"price": 2452.5, "quantity": 1100, "orders": 7},
                ],
            },
        ]
    )

    features = calculate_spread_features(snapshots, lookback_window=60)

    assert not features.empty
    assert "ob_spread_abs" in features.columns
    assert "ob_spread_rel" in features.columns
    assert "ob_spread_pct" in features.columns
    assert len(features) == 2

    # Check values
    assert features["ob_spread_abs"].iloc[0] == 1.0  # 2451 - 2450
    assert features["ob_spread_rel"].iloc[0] == pytest.approx(1.0 / 2450.5, rel=1e-4)


def test_calculate_depth_imbalance():
    """Test depth imbalance calculation."""
    snapshots = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-15 09:15:00",
                "bids": [
                    {"price": 2450.0, "quantity": 1200, "orders": 5},
                    {"price": 2449.5, "quantity": 1500, "orders": 7},
                ],
                "asks": [
                    {"price": 2451.0, "quantity": 800, "orders": 4},
                    {"price": 2451.5, "quantity": 1000, "orders": 6},
                ],
            }
        ]
    )

    features = calculate_depth_imbalance(snapshots)

    assert not features.empty
    assert "ob_depth_imbalance_l1" in features.columns
    assert "ob_depth_imbalance_l5" in features.columns

    # L1 imbalance: (1200 - 800) / (1200 + 800) = 400 / 2000 = 0.2
    assert features["ob_depth_imbalance_l1"].iloc[0] == pytest.approx(0.2, rel=1e-4)


def test_calculate_order_flow_metrics():
    """Test order flow metrics calculation."""
    snapshots = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-15 09:15:00",
                "bids": [
                    {"price": 2450.0, "quantity": 1000, "orders": 5},
                ],
                "asks": [
                    {"price": 2451.0, "quantity": 800, "orders": 4},
                ],
            }
        ]
    )

    features = calculate_order_flow_metrics(snapshots)

    assert not features.empty
    assert "ob_order_flow_ratio" in features.columns
    assert "ob_market_pressure" in features.columns
    assert "ob_liquidity_imbalance" in features.columns

    # Order flow ratio: 5 / 4 = 1.25
    assert features["ob_order_flow_ratio"].iloc[0] == pytest.approx(1.25, rel=1e-4)


def test_calculate_liquidity_features():
    """Test liquidity features calculation."""
    snapshots = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-15 09:15:00",
                "bids": [
                    {"price": 2450.0, "quantity": 1000, "orders": 5},
                    {"price": 2449.5, "quantity": 1500, "orders": 7},
                ],
                "asks": [
                    {"price": 2451.0, "quantity": 800, "orders": 4},
                    {"price": 2451.5, "quantity": 1200, "orders": 6},
                ],
            }
        ]
    )

    features = calculate_liquidity_features(snapshots)

    assert not features.empty
    assert "ob_bid_liquidity" in features.columns
    assert "ob_ask_liquidity" in features.columns
    assert "ob_total_liquidity" in features.columns
    assert "ob_market_impact_estimate" in features.columns

    # Total liquidity: 1000 + 1500 + 800 + 1200 = 4500
    assert features["ob_total_liquidity"].iloc[0] == 4500


def test_compute_all_order_book_features():
    """Test complete order book feature computation."""
    snapshots = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-15 09:15:00",
                "bids": [
                    {"price": 2450.0, "quantity": 1000, "orders": 5},
                    {"price": 2449.5, "quantity": 1500, "orders": 7},
                ],
                "asks": [
                    {"price": 2451.0, "quantity": 800, "orders": 4},
                    {"price": 2451.5, "quantity": 1200, "orders": 6},
                ],
            }
        ]
    )

    features = compute_all_order_book_features(snapshots)

    assert not features.empty
    # Check all feature categories present
    assert "ob_spread_abs" in features.columns
    assert "ob_depth_imbalance_l1" in features.columns
    assert "ob_order_flow_ratio" in features.columns
    assert "ob_bid_liquidity" in features.columns


def test_order_book_features_empty_input():
    """Test order book features with empty input."""
    features = calculate_spread_features(pd.DataFrame())
    assert features.empty


# ==================================================================
# Options Feature Tests
# ==================================================================


def test_calculate_iv_features():
    """Test IV feature calculation with toy data."""
    chain_df = pd.DataFrame(
        [
            {
                "date": "2025-01-15",
                "strike": 21000,
                "expiry": "2025-01-30",
                "call_iv": 0.18,
                "put_iv": 0.17,
                "underlying_price": 21500,
            },
            {
                "date": "2025-01-15",
                "strike": 21500,
                "expiry": "2025-01-30",
                "call_iv": 0.16,
                "put_iv": 0.16,
                "underlying_price": 21500,
            },
            {
                "date": "2025-01-15",
                "strike": 22000,
                "expiry": "2025-01-30",
                "call_iv": 0.15,
                "put_iv": 0.18,
                "underlying_price": 21500,
            },
        ]
    )

    features = calculate_iv_features(chain_df)

    assert not features.empty
    assert "opt_iv_atm" in features.columns
    assert "opt_iv_percentile" in features.columns
    assert "opt_iv_otm_call" in features.columns
    assert "opt_iv_otm_put" in features.columns


def test_calculate_skew_features():
    """Test IV skew calculation."""
    chain_df = pd.DataFrame(
        [
            {
                "date": "2025-01-15",
                "strike": 21000,
                "call_iv": 0.16,
                "put_iv": 0.18,
                "underlying_price": 21500,
            },
            {
                "date": "2025-01-15",
                "strike": 21500,
                "call_iv": 0.15,
                "put_iv": 0.17,
                "underlying_price": 21500,
            },
            {
                "date": "2025-01-15",
                "strike": 22000,
                "call_iv": 0.14,
                "put_iv": 0.19,
                "underlying_price": 21500,
            },
        ]
    )

    features = calculate_skew_features(chain_df)

    assert not features.empty
    assert "opt_skew_put_call" in features.columns
    assert "opt_skew_25delta" in features.columns
    assert "opt_skew_slope" in features.columns


def test_calculate_volume_features():
    """Test options volume feature calculation."""
    chain_df = pd.DataFrame(
        [
            {
                "date": "2025-01-15",
                "call_volume": 15000,
                "put_volume": 10000,
                "call_oi": 125000,
                "put_oi": 95000,
            },
            {
                "date": "2025-01-15",
                "call_volume": 12000,
                "put_volume": 8000,
                "call_oi": 100000,
                "put_oi": 80000,
            },
        ]
    )

    features = calculate_volume_features(chain_df)

    assert not features.empty
    assert "opt_put_call_ratio_volume" in features.columns
    assert "opt_put_call_ratio_oi" in features.columns

    # PCR volume: (10000 + 8000) / (15000 + 12000) = 18000 / 27000 = 0.6667
    assert features["opt_put_call_ratio_volume"].iloc[0] == pytest.approx(0.6667, rel=1e-3)


def test_compute_all_options_features():
    """Test complete options feature computation."""
    chain_df = pd.DataFrame(
        [
            {
                "date": "2025-01-15",
                "strike": 21500,
                "expiry": "2025-01-30",
                "call_iv": 0.16,
                "put_iv": 0.17,
                "call_volume": 15000,
                "put_volume": 10000,
                "call_oi": 125000,
                "put_oi": 95000,
                "underlying_price": 21500,
            }
        ]
    )

    features = compute_all_options_features(chain_df)

    assert not features.empty
    # Check feature categories
    assert "opt_iv_atm" in features.columns
    assert "opt_skew_put_call" in features.columns
    assert "opt_put_call_ratio_volume" in features.columns


# ==================================================================
# Macro Feature Tests
# ==================================================================


def test_calculate_regime_features():
    """Test regime classification with toy data."""
    macro_df = pd.DataFrame(
        [
            {"date": "2025-01-15", "indicator": "INDIAVIX", "value": 12.5},
            {"date": "2025-01-16", "indicator": "INDIAVIX", "value": 18.0},
            {"date": "2025-01-17", "indicator": "INDIAVIX", "value": 27.5},
        ]
    )

    features = calculate_regime_features(macro_df)

    assert not features.empty
    assert "macro_volatility_regime" in features.columns

    # Check regime classification
    assert features["macro_volatility_regime"].iloc[0] == "low"  # VIX < 15
    assert features["macro_volatility_regime"].iloc[1] == "medium"  # 15 <= VIX < 25
    assert features["macro_volatility_regime"].iloc[2] == "high"  # VIX >= 25


def test_calculate_momentum_features():
    """Test momentum indicators."""
    macro_df = pd.DataFrame(
        [
            {"date": "2025-01-01", "indicator": "NIFTY50", "value": 21000},
            {"date": "2025-01-02", "indicator": "NIFTY50", "value": 21100},
            {"date": "2025-01-03", "indicator": "NIFTY50", "value": 21200},
            {"date": "2025-01-04", "indicator": "NIFTY50", "value": 21300},
            {"date": "2025-01-05", "indicator": "NIFTY50", "value": 21400},
        ]
    )

    features = calculate_momentum_features(macro_df, short_window=2, long_window=3)

    assert not features.empty
    assert "macro_ma_crossover" in features.columns
    assert "macro_roc" in features.columns
    assert "macro_momentum_score" in features.columns


def test_compute_all_macro_features():
    """Test complete macro feature computation."""
    macro_df = pd.DataFrame(
        [
            {
                "date": "2025-01-15",
                "indicator": "NIFTY50",
                "value": 21500,
                "change": 150,
                "change_pct": 0.7,
            },
            {
                "date": "2025-01-15",
                "indicator": "INDIAVIX",
                "value": 15.5,
                "change": 0.5,
                "change_pct": 3.3,
            },
            {
                "date": "2025-01-16",
                "indicator": "NIFTY50",
                "value": 21600,
                "change": 100,
                "change_pct": 0.5,
            },
            {
                "date": "2025-01-16",
                "indicator": "INDIAVIX",
                "value": 16.0,
                "change": 0.5,
                "change_pct": 3.2,
            },
        ]
    )

    features = compute_all_macro_features(macro_df, short_window=1, long_window=2)

    assert not features.empty
    # Check feature categories present
    assert "macro_volatility_regime" in features.columns or "macro_ma_crossover" in features.columns


# ==================================================================
# Edge Case Tests
# ==================================================================


def test_features_with_single_snapshot():
    """Test features with single snapshot (edge case)."""
    snapshots = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-15 09:15:00",
                "bids": [{"price": 2450.0, "quantity": 1000, "orders": 5}],
                "asks": [{"price": 2451.0, "quantity": 800, "orders": 4}],
            }
        ]
    )

    features = calculate_spread_features(snapshots)
    assert not features.empty
    assert len(features) == 1


def test_features_with_zero_volume():
    """Test features with zero volume (edge case)."""
    chain_df = pd.DataFrame(
        [
            {
                "date": "2025-01-15",
                "call_volume": 0,
                "put_volume": 0,
                "call_oi": 0,
                "put_oi": 0,
            }
        ]
    )

    features = calculate_volume_features(chain_df)
    assert not features.empty
    # Should handle division by zero gracefully


def test_features_numerical_stability():
    """Test features don't produce inf/nan."""
    snapshots = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-15 09:15:00",
                "bids": [{"price": 2450.0, "quantity": 1000, "orders": 5}],
                "asks": [{"price": 2451.0, "quantity": 800, "orders": 4}],
            }
        ]
    )

    features = compute_all_order_book_features(snapshots)

    # Check no inf or nan values
    assert not features.isnull().any().any()
    assert not np.isinf(features.select_dtypes(include=[np.number])).any().any()
