"""Unit tests for market data providers (US-029 Phase 4)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from src.adapters.market_data_providers import (
    BreezeOptionsProvider,
    BreezeOrderBookProvider,
    MacroIndicatorProvider,
    MacroIndicatorSnapshot,
    OptionsChainSnapshot,
    OrderBookSnapshot,
    create_macro_provider,
    create_options_provider,
    create_order_book_provider,
)
from src.app.config import Settings

# ============================================================================
# Order Book Provider Tests
# ============================================================================


def test_order_book_provider_dryrun() -> None:
    """Test order book provider in dryrun mode returns mock data."""
    settings = Settings()
    provider = BreezeOrderBookProvider(settings, dry_run=True)

    snapshot = provider.fetch("RELIANCE", depth_levels=5)

    assert isinstance(snapshot, OrderBookSnapshot)
    assert snapshot.symbol == "RELIANCE"
    assert len(snapshot.bids) == 5
    assert len(snapshot.asks) == 5
    assert snapshot.metadata["dryrun"] is True
    assert snapshot.metadata["source"] == "stub"

    # Verify bid/ask structure
    for bid in snapshot.bids:
        assert "price" in bid
        assert "quantity" in bid
        assert "orders" in bid
        assert bid["price"] > 0
        assert bid["quantity"] > 0

    for ask in snapshot.asks:
        assert "price" in ask
        assert "quantity" in ask
        assert "orders" in ask
        assert ask["price"] > 0
        assert ask["quantity"] > 0

    # Verify spread (ask > bid)
    assert snapshot.asks[0]["price"] > snapshot.bids[0]["price"]


def test_order_book_provider_deterministic() -> None:
    """Test order book provider generates deterministic mock data."""
    settings = Settings()
    provider = BreezeOrderBookProvider(settings, dry_run=True)

    snapshot1 = provider.fetch("RELIANCE", depth_levels=3)
    snapshot2 = provider.fetch("RELIANCE", depth_levels=3)

    # Same symbol should generate same prices (deterministic)
    assert snapshot1.bids[0]["price"] == snapshot2.bids[0]["price"]
    assert snapshot1.asks[0]["price"] == snapshot2.asks[0]["price"]


def test_order_book_provider_different_symbols() -> None:
    """Test order book provider generates different data for different symbols."""
    settings = Settings()
    provider = BreezeOrderBookProvider(settings, dry_run=True)

    snapshot_rel = provider.fetch("RELIANCE", depth_levels=5)
    snapshot_tcs = provider.fetch("TCS", depth_levels=5)

    # Different symbols should have different prices
    assert snapshot_rel.bids[0]["price"] != snapshot_tcs.bids[0]["price"]


def test_order_book_provider_depth_levels() -> None:
    """Test order book provider respects depth_levels parameter."""
    settings = Settings()
    provider = BreezeOrderBookProvider(settings, dry_run=True)

    snapshot_3 = provider.fetch("RELIANCE", depth_levels=3)
    snapshot_10 = provider.fetch("RELIANCE", depth_levels=10)

    assert len(snapshot_3.bids) == 3
    assert len(snapshot_3.asks) == 3
    assert len(snapshot_10.bids) == 10
    assert len(snapshot_10.asks) == 10


def test_order_book_provider_factory_dryrun_default() -> None:
    """Test factory creates provider in dryrun mode when order_book_enabled=False."""
    settings = Settings()
    assert settings.order_book_enabled is False

    provider = create_order_book_provider(settings)

    assert provider.dry_run is True


def test_order_book_provider_factory_live_mode() -> None:
    """Test factory creates provider in live mode when order_book_enabled=True."""
    settings = Settings()
    settings.order_book_enabled = True

    # Mock BreezeClient to avoid real API calls
    with patch("src.adapters.market_data_providers.BreezeClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        provider = create_order_book_provider(settings)

        assert provider.dry_run is False


# ============================================================================
# Options Provider Tests
# ============================================================================


def test_options_provider_dryrun() -> None:
    """Test options provider in dryrun mode returns mock data."""
    settings = Settings()
    provider = BreezeOptionsProvider(settings, dry_run=True)

    snapshot = provider.fetch("NIFTY", date="2025-01-15")

    assert isinstance(snapshot, OptionsChainSnapshot)
    assert snapshot.symbol == "NIFTY"
    assert snapshot.date == "2025-01-15"
    assert snapshot.underlying_price > 0
    assert len(snapshot.options) > 0
    assert snapshot.metadata["dryrun"] is True
    assert snapshot.metadata["source"] == "stub"

    # Verify options structure
    for option in snapshot.options:
        assert "strike" in option
        assert "expiry" in option
        assert "call" in option
        assert "put" in option

        # Verify call structure
        call = option["call"]
        assert "last_price" in call
        assert "bid" in call
        assert "ask" in call
        assert "volume" in call
        assert "oi" in call
        assert "iv" in call
        assert call["iv"] > 0

        # Verify put structure
        put = option["put"]
        assert "last_price" in put
        assert "bid" in put
        assert "ask" in put
        assert "volume" in put
        assert "oi" in put
        assert "iv" in put
        assert put["iv"] > 0


def test_options_provider_deterministic() -> None:
    """Test options provider generates deterministic mock data."""
    settings = Settings()
    provider = BreezeOptionsProvider(settings, dry_run=True)

    snapshot1 = provider.fetch("NIFTY", date="2025-01-15")
    snapshot2 = provider.fetch("NIFTY", date="2025-01-15")

    # Same symbol should generate same underlying price
    assert snapshot1.underlying_price == snapshot2.underlying_price

    # Same number of options
    assert len(snapshot1.options) == len(snapshot2.options)


def test_options_provider_different_symbols() -> None:
    """Test options provider generates different data for different symbols."""
    settings = Settings()
    provider = BreezeOptionsProvider(settings, dry_run=True)

    snapshot_nifty = provider.fetch("NIFTY", date="2025-01-15")
    snapshot_bank = provider.fetch("BANKNIFTY", date="2025-01-15")

    # Different symbols should have different underlying prices
    assert snapshot_nifty.underlying_price != snapshot_bank.underlying_price


def test_options_provider_metadata() -> None:
    """Test options provider includes metadata."""
    settings = Settings()
    provider = BreezeOptionsProvider(settings, dry_run=True)

    snapshot = provider.fetch("NIFTY", date="2025-01-15")

    assert "total_strikes" in snapshot.metadata
    assert "expiries" in snapshot.metadata
    assert snapshot.metadata["total_strikes"] > 0
    assert len(snapshot.metadata["expiries"]) > 0


def test_options_provider_factory_dryrun_default() -> None:
    """Test factory creates provider in dryrun mode when options_enabled=False."""
    settings = Settings()
    assert settings.options_enabled is False

    provider = create_options_provider(settings)

    assert provider.dry_run is True


def test_options_provider_factory_live_mode() -> None:
    """Test factory creates provider in live mode when options_enabled=True."""
    settings = Settings()
    settings.options_enabled = True

    # Mock BreezeClient to avoid real API calls
    with patch("src.adapters.market_data_providers.BreezeClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        provider = create_options_provider(settings)

        assert provider.dry_run is False


# ============================================================================
# Macro Indicator Provider Tests
# ============================================================================


def test_macro_provider_dryrun_nifty() -> None:
    """Test macro provider in dryrun mode for NIFTY50."""
    settings = Settings()
    provider = MacroIndicatorProvider(settings, dry_run=True)

    snapshot = provider.fetch("NIFTY50", date="2025-01-15")

    assert isinstance(snapshot, MacroIndicatorSnapshot)
    assert snapshot.indicator == "NIFTY50"
    assert snapshot.date == "2025-01-15"
    assert snapshot.value > 0
    assert snapshot.change is not None
    assert snapshot.change_pct is not None
    assert snapshot.metadata["dryrun"] is True
    assert snapshot.metadata["source"] == "stub"


def test_macro_provider_dryrun_all_indicators() -> None:
    """Test macro provider supports all configured indicators."""
    settings = Settings()
    provider = MacroIndicatorProvider(settings, dry_run=True)

    indicators = ["NIFTY50", "BANKNIFTY", "INDIAVIX", "USDINR", "IN10Y"]

    for indicator in indicators:
        snapshot = provider.fetch(indicator, date="2025-01-15")

        assert snapshot.indicator == indicator
        assert snapshot.value > 0
        assert snapshot.metadata["source"] == "stub"


def test_macro_provider_deterministic() -> None:
    """Test macro provider generates deterministic mock data."""
    settings = Settings()
    provider = MacroIndicatorProvider(settings, dry_run=True)

    snapshot1 = provider.fetch("NIFTY50", date="2025-01-15")
    snapshot2 = provider.fetch("NIFTY50", date="2025-01-15")

    # Same indicator and date should generate same value
    assert snapshot1.value == snapshot2.value
    assert snapshot1.change == snapshot2.change


def test_macro_provider_different_dates() -> None:
    """Test macro provider generates different data for different dates."""
    settings = Settings()
    provider = MacroIndicatorProvider(settings, dry_run=True)

    snapshot_jan = provider.fetch("NIFTY50", date="2025-01-15")
    snapshot_feb = provider.fetch("NIFTY50", date="2025-02-15")

    # Different dates should generate different values (due to date hash)
    assert snapshot_jan.value != snapshot_feb.value


def test_macro_provider_factory_dryrun_default() -> None:
    """Test factory creates provider in dryrun mode when macro_enabled=False."""
    settings = Settings()
    assert settings.macro_enabled is False

    provider = create_macro_provider(settings)

    assert provider.dry_run is True


def test_macro_provider_factory_live_mode() -> None:
    """Test factory creates provider in live mode when macro_enabled=True."""
    settings = Settings()
    settings.macro_enabled = True

    provider = create_macro_provider(settings)

    # Should be in live mode (will fall back to mock if yfinance not available)
    assert provider.dry_run is False


def test_macro_provider_change_calculation() -> None:
    """Test macro provider includes change and change_pct."""
    settings = Settings()
    provider = MacroIndicatorProvider(settings, dry_run=True)

    snapshot = provider.fetch("NIFTY50", date="2025-01-15")

    assert snapshot.change is not None
    assert snapshot.change_pct is not None
    assert isinstance(snapshot.change, float)
    assert isinstance(snapshot.change_pct, float)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_order_book_provider_zero_depth() -> None:
    """Test order book provider handles zero depth gracefully."""
    settings = Settings()
    provider = BreezeOrderBookProvider(settings, dry_run=True)

    snapshot = provider.fetch("RELIANCE", depth_levels=0)

    assert len(snapshot.bids) == 0
    assert len(snapshot.asks) == 0


def test_options_provider_default_date() -> None:
    """Test options provider uses current date when date=None."""
    settings = Settings()
    provider = BreezeOptionsProvider(settings, dry_run=True)

    snapshot = provider.fetch("NIFTY", date=None)

    # Should use today's date
    expected_date = datetime.now().strftime("%Y-%m-%d")
    assert snapshot.date == expected_date


def test_macro_provider_default_date() -> None:
    """Test macro provider uses current date when date=None."""
    settings = Settings()
    provider = MacroIndicatorProvider(settings, dry_run=True)

    snapshot = provider.fetch("NIFTY50", date=None)

    # Should use today's date
    expected_date = datetime.now().strftime("%Y-%m-%d")
    assert snapshot.date == expected_date


def test_order_book_snapshot_timestamp() -> None:
    """Test order book snapshot includes valid timestamp."""
    settings = Settings()
    provider = BreezeOrderBookProvider(settings, dry_run=True)

    snapshot = provider.fetch("RELIANCE")

    assert isinstance(snapshot.timestamp, datetime)
    assert snapshot.timestamp <= datetime.now()


def test_options_snapshot_timestamp() -> None:
    """Test options snapshot includes valid timestamp."""
    settings = Settings()
    provider = BreezeOptionsProvider(settings, dry_run=True)

    snapshot = provider.fetch("NIFTY")

    assert isinstance(snapshot.timestamp, datetime)
    assert snapshot.timestamp <= datetime.now()


def test_macro_snapshot_timestamp() -> None:
    """Test macro snapshot includes valid timestamp."""
    settings = Settings()
    provider = MacroIndicatorProvider(settings, dry_run=True)

    snapshot = provider.fetch("NIFTY50")

    assert isinstance(snapshot.timestamp, datetime)
    assert snapshot.timestamp <= datetime.now()


# ============================================================================
# Factory Tests
# ============================================================================


def test_factory_explicit_dryrun_override() -> None:
    """Test factory respects explicit dry_run parameter."""
    settings = Settings()
    settings.order_book_enabled = True

    # Override to dryrun despite enabled=True
    provider = create_order_book_provider(settings, dry_run=True)

    assert provider.dry_run is True


def test_factory_explicit_live_override() -> None:
    """Test factory respects explicit dry_run=False parameter."""
    settings = Settings()
    settings.order_book_enabled = False

    # Mock BreezeClient to avoid real API calls
    with patch("src.adapters.market_data_providers.BreezeClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Override to live mode despite enabled=False
        provider = create_order_book_provider(settings, dry_run=False)

        assert provider.dry_run is False


def test_factory_with_existing_client() -> None:
    """Test factory accepts existing BreezeClient instance."""
    settings = Settings()
    settings.order_book_enabled = True

    mock_client = MagicMock()
    provider = create_order_book_provider(settings, client=mock_client, dry_run=False)

    assert provider.client == mock_client
    assert provider.dry_run is False
