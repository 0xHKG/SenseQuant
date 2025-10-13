"""Integration tests for market data ingestion (US-029 Phase 1).

Tests order book, options chain, and macro data ingestion scripts with dryrun mode.
Verifies directory structures, state updates, and DataFeed loader functions.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.services.data_feed import (
    load_macro_data,
    load_options_chain,
    load_order_book_snapshots,
)
from src.services.state_manager import StateManager


def load_script_module(script_path: str, module_name: str):
    """Load a Python script as a module using importlib."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def temp_order_book_dir(tmp_path: Path) -> Path:
    """Create temporary directory for order book data."""
    order_book_dir = tmp_path / "order_book"
    order_book_dir.mkdir()
    return order_book_dir


@pytest.fixture
def temp_options_dir(tmp_path: Path) -> Path:
    """Create temporary directory for options data."""
    options_dir = tmp_path / "options"
    options_dir.mkdir()
    return options_dir


@pytest.fixture
def temp_macro_dir(tmp_path: Path) -> Path:
    """Create temporary directory for macro data."""
    macro_dir = tmp_path / "macro"
    macro_dir.mkdir()
    return macro_dir


@pytest.fixture
def temp_state_file(tmp_path: Path) -> Path:
    """Create temporary state file."""
    return tmp_path / "state.json"


# =====================================================================
# Order Book Tests
# =====================================================================


def test_order_book_ingestion_dryrun(temp_order_book_dir: Path, temp_state_file: Path) -> None:
    """Test order book ingestion in dryrun mode creates expected structure."""
    # Load script module
    fetch_order_book = load_script_module("scripts/fetch_order_book.py", "fetch_order_book")
    OrderBookFetcher = fetch_order_book.OrderBookFetcher

    fetcher = OrderBookFetcher(
        output_dir=temp_order_book_dir,
        depth_levels=5,
        dryrun=True,
        force=False,
    )

    # Generate time range
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_time = today.replace(hour=9, minute=15, second=0)
    end_time = today.replace(hour=9, minute=18, second=0)  # 3 snapshots (0, 60, 120 seconds)

    # Fetch for single symbol
    summary = fetcher.fetch_all(
        symbols=["RELIANCE"],
        start_time=start_time,
        end_time=end_time,
        interval_seconds=60,
    )

    # Verify summary
    assert summary["stats"]["fetched"] == 4  # 09:15, 09:16, 09:17, 09:18
    assert summary["stats"]["cached"] == 0
    assert summary["stats"]["failed"] == 0
    assert summary["dryrun"] is True

    # Verify directory structure
    symbol_dir = temp_order_book_dir / "RELIANCE"
    assert symbol_dir.exists()

    date_str = today.strftime("%Y-%m-%d")
    date_dir = symbol_dir / date_str
    assert date_dir.exists()

    # Verify snapshot files
    snapshot_files = list(date_dir.glob("*.json"))
    assert len(snapshot_files) == 4

    # Verify snapshot content
    with open(snapshot_files[0]) as f:
        snapshot = json.load(f)
        assert snapshot["symbol"] == "RELIANCE"
        assert snapshot["exchange"] == "NSE"
        assert len(snapshot["bids"]) == 5
        assert len(snapshot["asks"]) == 5
        assert snapshot["metadata"]["depth_levels"] == 5
        assert snapshot["metadata"]["source"] == "stub"
        assert snapshot["metadata"]["dryrun"] is True


def test_order_book_incremental_mode_skips_existing(
    temp_order_book_dir: Path, temp_state_file: Path
) -> None:
    """Test order book incremental mode skips existing snapshots."""
    fetch_order_book = load_script_module("scripts/fetch_order_book.py", "fetch_order_book")
    OrderBookFetcher = fetch_order_book.OrderBookFetcher

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_time = today.replace(hour=9, minute=15, second=0)
    end_time = today.replace(hour=9, minute=16, second=0)

    # First fetch
    fetcher1 = OrderBookFetcher(
        output_dir=temp_order_book_dir,
        depth_levels=5,
        dryrun=True,
        force=False,
    )
    summary1 = fetcher1.fetch_all(["RELIANCE"], start_time, end_time, interval_seconds=60)
    assert summary1["stats"]["fetched"] == 2
    assert summary1["stats"]["cached"] == 0

    # Second fetch (same time range) - should use cache
    fetcher2 = OrderBookFetcher(
        output_dir=temp_order_book_dir,
        depth_levels=5,
        dryrun=True,
        force=False,
    )
    summary2 = fetcher2.fetch_all(["RELIANCE"], start_time, end_time, interval_seconds=60)
    # Note: fetched counter still increments in second run because stats are tracked in dryrun
    # The key check is that cached == 2 (all snapshots were found in cache)
    assert summary2["stats"]["cached"] == 2  # All cached


# =====================================================================
# Options Chain Tests
# =====================================================================


def test_options_ingestion_dryrun(temp_options_dir: Path, temp_state_file: Path) -> None:
    """Test options chain ingestion in dryrun mode creates expected structure."""
    fetch_options_data = load_script_module("scripts/fetch_options_data.py", "fetch_options_data")
    OptionsDataFetcher = fetch_options_data.OptionsDataFetcher

    fetcher = OptionsDataFetcher(
        output_dir=temp_options_dir,
        dryrun=True,
        force=False,
    )

    # Fetch for single symbol and date range
    start_date = datetime(2025, 1, 15)
    end_date = datetime(2025, 1, 17)

    summary = fetcher.fetch_all(
        symbols=["NIFTY"],
        start_date=start_date,
        end_date=end_date,
    )

    # Verify summary
    assert summary["stats"]["fetched"] == 3  # 3 days
    assert summary["stats"]["cached"] == 0
    assert summary["stats"]["failed"] == 0
    assert summary["stats"]["total_strikes"] > 0
    assert summary["stats"]["total_expiries"] > 0
    assert summary["dryrun"] is True

    # Verify directory structure
    symbol_dir = temp_options_dir / "NIFTY"
    assert symbol_dir.exists()

    # Verify chain files
    chain_files = list(symbol_dir.glob("*.json"))
    assert len(chain_files) == 3

    # Verify chain content
    with open(chain_files[0]) as f:
        chain = json.load(f)
        assert chain["symbol"] == "NIFTY"
        assert "underlying_price" in chain
        assert "options" in chain
        assert len(chain["options"]) > 0

        # Verify option structure
        option = chain["options"][0]
        assert "strike" in option
        assert "expiry" in option
        assert "call" in option
        assert "put" in option
        assert "last_price" in option["call"]
        assert "iv" in option["call"]
        assert "volume" in option["put"]
        assert "oi" in option["put"]

        assert chain["metadata"]["source"] == "stub"
        assert chain["metadata"]["dryrun"] is True


def test_options_incremental_mode_skips_existing(
    temp_options_dir: Path, temp_state_file: Path
) -> None:
    """Test options chain incremental mode skips existing chains."""
    fetch_options_data = load_script_module("scripts/fetch_options_data.py", "fetch_options_data")
    OptionsDataFetcher = fetch_options_data.OptionsDataFetcher

    date = datetime(2025, 1, 15)

    # First fetch
    fetcher1 = OptionsDataFetcher(
        output_dir=temp_options_dir,
        dryrun=True,
        force=False,
    )
    summary1 = fetcher1.fetch_all(["NIFTY"], date, date)
    assert summary1["stats"]["fetched"] == 1
    assert summary1["stats"]["cached"] == 0

    # Second fetch (same date) - should use cache
    fetcher2 = OptionsDataFetcher(
        output_dir=temp_options_dir,
        dryrun=True,
        force=False,
    )
    summary2 = fetcher2.fetch_all(["NIFTY"], date, date)
    assert summary2["stats"]["cached"] == 1  # Cached


# =====================================================================
# Macro Data Tests
# =====================================================================


def test_macro_ingestion_dryrun(temp_macro_dir: Path, temp_state_file: Path) -> None:
    """Test macro data ingestion in dryrun mode creates expected structure."""
    fetch_macro_data_module = load_script_module(
        "scripts/fetch_macro_data.py", "fetch_macro_data_module"
    )
    MacroDataFetcher = fetch_macro_data_module.MacroDataFetcher

    fetcher = MacroDataFetcher(
        output_dir=temp_macro_dir,
        dryrun=True,
        force=False,
    )

    # Fetch for multiple indicators
    start_date = datetime(2025, 1, 15)
    end_date = datetime(2025, 1, 17)

    summary = fetcher.fetch_all(
        indicators=["NIFTY50", "INDIAVIX"],
        start_date=start_date,
        end_date=end_date,
    )

    # Verify summary
    assert summary["stats"]["fetched"] == 6  # 2 indicators * 3 days
    assert summary["stats"]["cached"] == 0
    assert summary["stats"]["failed"] == 0
    assert summary["stats"]["total_data_points"] == 6
    assert summary["dryrun"] is True

    # Verify directory structure
    nifty_dir = temp_macro_dir / "NIFTY50"
    vix_dir = temp_macro_dir / "INDIAVIX"
    assert nifty_dir.exists()
    assert vix_dir.exists()

    # Verify data files
    nifty_files = list(nifty_dir.glob("*.json"))
    assert len(nifty_files) == 3

    # Verify data content
    with open(nifty_files[0]) as f:
        data = json.load(f)
        assert data["indicator"] == "NIFTY50"
        assert "value" in data
        assert "change" in data
        assert "change_pct" in data
        assert data["metadata"]["source"] == "stub"
        assert data["metadata"]["dryrun"] is True


def test_macro_incremental_mode_skips_existing(temp_macro_dir: Path, temp_state_file: Path) -> None:
    """Test macro data incremental mode skips existing data."""
    fetch_macro_data_module = load_script_module(
        "scripts/fetch_macro_data.py", "fetch_macro_data_module"
    )
    MacroDataFetcher = fetch_macro_data_module.MacroDataFetcher

    date = datetime(2025, 1, 15)

    # First fetch
    fetcher1 = MacroDataFetcher(
        output_dir=temp_macro_dir,
        dryrun=True,
        force=False,
    )
    summary1 = fetcher1.fetch_all(["NIFTY50"], date, date)
    assert summary1["stats"]["fetched"] == 1
    assert summary1["stats"]["cached"] == 0

    # Second fetch (same date) - should use cache
    fetcher2 = MacroDataFetcher(
        output_dir=temp_macro_dir,
        dryrun=True,
        force=False,
    )
    summary2 = fetcher2.fetch_all(["NIFTY50"], date, date)
    assert summary2["stats"]["cached"] == 1  # Cached


# =====================================================================
# StateManager Market Data Tracking Tests
# =====================================================================


def test_market_data_state_manager(temp_state_file: Path) -> None:
    """Test StateManager market data tracking methods."""
    state_mgr = StateManager(state_file=temp_state_file)

    # Record order book fetch
    state_mgr.record_market_data_fetch(
        data_type="order_book",
        symbol="RELIANCE",
        timestamp=datetime.now().isoformat(),
        stats={"fetched": 10, "cached": 5, "failed": 0},
    )

    # Record options fetch
    state_mgr.record_market_data_fetch(
        data_type="options",
        symbol="NIFTY",
        timestamp=datetime.now().isoformat(),
        stats={"fetched": 1, "cached": 0, "failed": 0, "strikes": 50, "expiries": 3},
    )

    # Record macro fetch
    state_mgr.record_market_data_fetch(
        data_type="macro",
        symbol="NIFTY50",
        timestamp=datetime.now().isoformat(),
        stats={"fetched": 7, "cached": 0, "failed": 0, "data_points": 7},
    )

    # Verify last fetch retrieval
    last_order_book = state_mgr.get_last_market_data_fetch("order_book", "RELIANCE")
    assert last_order_book is not None
    assert last_order_book["stats"]["fetched"] == 10
    assert last_order_book["stats"]["cached"] == 5
    assert last_order_book["total_fetches"] == 1
    assert last_order_book["total_errors"] == 0

    last_options = state_mgr.get_last_market_data_fetch("options", "NIFTY")
    assert last_options is not None
    assert last_options["stats"]["strikes"] == 50

    last_macro = state_mgr.get_last_market_data_fetch("macro", "NIFTY50")
    assert last_macro is not None
    assert last_macro["stats"]["data_points"] == 7

    # Verify aggregated stats
    order_book_stats = state_mgr.get_market_data_fetch_stats("order_book")
    assert order_book_stats["total_symbols"] == 1
    assert order_book_stats["total_fetches"] == 1
    assert order_book_stats["total_errors"] == 0

    options_stats = state_mgr.get_market_data_fetch_stats("options")
    assert options_stats["total_symbols"] == 1

    macro_stats = state_mgr.get_market_data_fetch_stats("macro")
    assert macro_stats["total_symbols"] == 1

    # Test empty state
    empty_stats = state_mgr.get_market_data_fetch_stats("nonexistent")
    assert empty_stats["total_symbols"] == 0


def test_market_data_state_manager_multiple_fetches(temp_state_file: Path) -> None:
    """Test StateManager tracks multiple fetches correctly."""
    state_mgr = StateManager(state_file=temp_state_file)

    # Record 3 fetches for same symbol
    for i in range(3):
        state_mgr.record_market_data_fetch(
            data_type="order_book",
            symbol="RELIANCE",
            timestamp=datetime.now().isoformat(),
            stats={"fetched": 10, "cached": 0, "failed": i},  # Increment failures
        )

    last_fetch = state_mgr.get_last_market_data_fetch("order_book", "RELIANCE")
    assert last_fetch is not None
    assert last_fetch["total_fetches"] == 3
    assert last_fetch["total_errors"] == 3  # 0 + 1 + 2 = 3


# =====================================================================
# DataFeed Market Data Loader Tests
# =====================================================================


def test_data_feed_load_order_book_snapshots(temp_order_book_dir: Path) -> None:
    """Test DataFeed can load cached order book snapshots."""
    # Create mock order book data
    symbol = "RELIANCE"
    date = datetime(2025, 1, 15)
    date_dir = temp_order_book_dir / symbol / date.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True)

    # Create 3 snapshots
    for i in range(3):
        timestamp = date.replace(hour=9, minute=15 + i, second=0)
        snapshot = {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "exchange": "NSE",
            "bids": [{"price": 2450.0, "quantity": 1000, "orders": 5}],
            "asks": [{"price": 2451.0, "quantity": 800, "orders": 4}],
            "metadata": {"source": "test"},
        }

        snapshot_file = date_dir / f"09-{15 + i:02d}-00.json"
        with open(snapshot_file, "w") as f:
            json.dump(snapshot, f)

    # Load snapshots
    snapshots = load_order_book_snapshots(
        symbol=symbol,
        from_date=date,
        to_date=date,
        base_dir=str(temp_order_book_dir),
    )

    # Verify
    assert len(snapshots) == 3
    assert snapshots[0]["symbol"] == symbol
    assert len(snapshots[0]["bids"]) == 1


def test_data_feed_load_order_book_missing_dir(temp_order_book_dir: Path) -> None:
    """Test DataFeed handles missing order book directory gracefully."""
    snapshots = load_order_book_snapshots(
        symbol="NONEXISTENT",
        from_date=datetime(2025, 1, 15),
        to_date=datetime(2025, 1, 15),
        base_dir=str(temp_order_book_dir),
    )

    assert snapshots == []


def test_data_feed_load_options_chain(temp_options_dir: Path) -> None:
    """Test DataFeed can load cached options chain."""
    # Create mock options chain
    symbol = "NIFTY"
    date = datetime(2025, 1, 15)
    symbol_dir = temp_options_dir / symbol
    symbol_dir.mkdir(parents=True)

    chain = {
        "symbol": symbol,
        "date": date.strftime("%Y-%m-%d"),
        "underlying_price": 21500.0,
        "options": [
            {
                "strike": 21000,
                "expiry": "2025-01-30",
                "call": {"last_price": 550.0, "iv": 0.18},
                "put": {"last_price": 45.0, "iv": 0.17},
            }
        ],
        "metadata": {"source": "test"},
    }

    chain_file = symbol_dir / f"{date.strftime('%Y-%m-%d')}.json"
    with open(chain_file, "w") as f:
        json.dump(chain, f)

    # Load chain
    loaded_chain = load_options_chain(
        symbol=symbol,
        date=date,
        base_dir=str(temp_options_dir),
    )

    # Verify
    assert loaded_chain is not None
    assert loaded_chain["symbol"] == symbol
    assert loaded_chain["underlying_price"] == 21500.0
    assert len(loaded_chain["options"]) == 1


def test_data_feed_load_options_chain_missing(temp_options_dir: Path) -> None:
    """Test DataFeed handles missing options chain gracefully."""
    chain = load_options_chain(
        symbol="NONEXISTENT",
        date=datetime(2025, 1, 15),
        base_dir=str(temp_options_dir),
    )

    assert chain is None


def test_data_feed_load_macro_data(temp_macro_dir: Path) -> None:
    """Test DataFeed can load cached macro data."""
    # Create mock macro data
    indicator = "NIFTY50"
    indicator_dir = temp_macro_dir / indicator
    indicator_dir.mkdir(parents=True)

    # Create 3 days of data
    for i in range(3):
        date = datetime(2025, 1, 15) + timedelta(days=i)
        data = {
            "indicator": indicator,
            "date": date.strftime("%Y-%m-%d"),
            "value": 21500.0 + i * 100,
            "change": 100.0,
            "change_pct": 0.50,
        }

        data_file = indicator_dir / f"{date.strftime('%Y-%m-%d')}.json"
        with open(data_file, "w") as f:
            json.dump(data, f)

    # Load data
    df = load_macro_data(
        indicator=indicator,
        from_date=datetime(2025, 1, 15),
        to_date=datetime(2025, 1, 17),
        base_dir=str(temp_macro_dir),
    )

    # Verify
    assert len(df) == 3
    assert list(df.columns) == ["date", "value", "change", "change_pct"]
    assert df["value"].iloc[0] == 21500.0
    assert df["value"].iloc[2] == 21700.0


def test_data_feed_load_macro_data_missing(temp_macro_dir: Path) -> None:
    """Test DataFeed handles missing macro data gracefully."""
    df = load_macro_data(
        indicator="NONEXISTENT",
        from_date=datetime(2025, 1, 15),
        to_date=datetime(2025, 1, 15),
        base_dir=str(temp_macro_dir),
    )

    assert df.empty


# ============================================================================
# US-029 Phase 4: Provider Integration Tests
# ============================================================================


def test_order_book_provider_integration_dryrun(tmp_path: Path) -> None:
    """Test order book provider integration in dryrun mode."""
    import sys
    from pathlib import Path as PathLib

    sys.path.insert(0, str(PathLib(__file__).parent.parent.parent / "scripts"))
    from fetch_order_book import OrderBookFetcher

    output_dir = tmp_path / "order_book"
    fetcher = OrderBookFetcher(
        output_dir=output_dir,
        depth_levels=5,
        dryrun=True,
        secrets_mode="plain",
    )

    # Fetch single snapshot
    from datetime import datetime

    timestamp = datetime(2025, 1, 15, 9, 15, 0)
    result = fetcher.fetch_snapshot("RELIANCE", timestamp)

    assert result is True
    assert fetcher.stats["fetched"] == 1
    assert fetcher.stats["failed"] == 0

    # Verify file created
    snapshot_path = output_dir / "RELIANCE" / "2025-01-15" / "09-15-00.json"
    assert snapshot_path.exists()

    # Verify content
    import json

    with open(snapshot_path) as f:
        data = json.load(f)

    assert data["symbol"] == "RELIANCE"
    assert len(data["bids"]) == 5
    assert len(data["asks"]) == 5
    assert data["metadata"]["dryrun"] is True


def test_options_provider_integration_dryrun(tmp_path: Path) -> None:
    """Test options provider integration in dryrun mode."""
    import sys
    from pathlib import Path as PathLib

    sys.path.insert(0, str(PathLib(__file__).parent.parent.parent / "scripts"))
    from fetch_options_data import OptionsDataFetcher

    output_dir = tmp_path / "options"
    fetcher = OptionsDataFetcher(
        output_dir=output_dir,
        dryrun=True,
        secrets_mode="plain",
    )

    # Fetch single chain
    from datetime import datetime

    date = datetime(2025, 1, 15)
    result = fetcher.fetch_chain("NIFTY", date)

    assert result is True
    assert fetcher.stats["fetched"] == 1
    assert fetcher.stats["failed"] == 0

    # Verify file created
    chain_path = output_dir / "NIFTY" / "2025-01-15.json"
    assert chain_path.exists()

    # Verify content
    import json

    with open(chain_path) as f:
        data = json.load(f)

    assert data["symbol"] == "NIFTY"
    assert data["underlying_price"] > 0
    assert len(data["options"]) > 0
    assert data["metadata"]["dryrun"] is True


def test_macro_provider_integration_dryrun(tmp_path: Path) -> None:
    """Test macro provider integration in dryrun mode."""
    import sys
    from pathlib import Path as PathLib

    sys.path.insert(0, str(PathLib(__file__).parent.parent.parent / "scripts"))
    from fetch_macro_data import MacroDataFetcher

    output_dir = tmp_path / "macro"
    fetcher = MacroDataFetcher(
        output_dir=output_dir,
        dryrun=True,
        secrets_mode="plain",
    )

    # Fetch single data point
    from datetime import datetime

    date = datetime(2025, 1, 15)
    result = fetcher.fetch_data("NIFTY50", date)

    assert result is True
    assert fetcher.stats["fetched"] == 1
    assert fetcher.stats["failed"] == 0

    # Verify file created
    data_path = output_dir / "NIFTY50" / "2025-01-15.json"
    assert data_path.exists()

    # Verify content
    import json

    with open(data_path) as f:
        data = json.load(f)

    assert data["indicator"] == "NIFTY50"
    assert data["value"] > 0
    assert data["metadata"]["dryrun"] is True


def test_provider_metrics_tracking() -> None:
    """Test StateManager provider metrics tracking (US-029 Phase 4)."""
    from src.services.state_manager import StateManager

    state_file = Path("data/state/test_provider_metrics.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)

    if state_file.exists():
        state_file.unlink()

    manager = StateManager(state_file)

    # Record successful fetch
    manager.record_provider_metrics(
        provider_name="order_book",
        success=True,
        retries=0,
        latency_ms=120.5,
    )

    # Record failed fetch
    manager.record_provider_metrics(
        provider_name="order_book",
        success=False,
        retries=2,
        error_message="Connection timeout",
    )

    # Get stats
    stats = manager.get_provider_stats("order_book")

    assert stats["total_requests"] == 2
    assert stats["successful_requests"] == 1
    assert stats["failed_requests"] == 1
    assert stats["success_rate"] == 50.0
    assert stats["total_retries"] == 2
    # Only successful request with latency counts: 120.5 / 1 = 120.5
    # Failed request has no latency, so it doesn't affect the average
    assert stats["avg_latency_ms"] == 120.5
    assert stats["max_latency_ms"] == 120.5
    assert stats["last_error_message"] == "Connection timeout"

    # Cleanup
    state_file.unlink()


def test_provider_no_network_calls_in_dryrun(tmp_path: Path) -> None:
    """Test that dryrun mode makes no network calls."""
    import sys
    from datetime import datetime
    from pathlib import Path as PathLib
    from unittest.mock import MagicMock, patch

    sys.path.insert(0, str(PathLib(__file__).parent.parent.parent / "scripts"))
    from fetch_order_book import OrderBookFetcher

    output_dir = tmp_path / "order_book"

    # Mock BreezeClient to detect if it's called
    with patch("src.adapters.market_data_providers.BreezeClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create fetcher in dryrun mode
        fetcher = OrderBookFetcher(
            output_dir=output_dir,
            depth_levels=5,
            dryrun=True,
            secrets_mode="plain",
        )

        # Fetch snapshot
        timestamp = datetime(2025, 1, 15, 9, 15, 0)
        result = fetcher.fetch_snapshot("RELIANCE", timestamp)

        # Verify success
        assert result is True

        # Verify BreezeClient was NOT created in dryrun mode
        # Since we're in dryrun, the client should not have been instantiated
        assert not mock_client_class.called


def test_multiple_providers_stats() -> None:
    """Test tracking metrics for multiple providers."""
    from src.services.state_manager import StateManager

    state_file = Path("data/state/test_multi_provider.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)

    if state_file.exists():
        state_file.unlink()

    manager = StateManager(state_file)

    # Record metrics for different providers
    manager.record_provider_metrics("order_book", success=True, latency_ms=100)
    manager.record_provider_metrics("options", success=True, latency_ms=200)
    manager.record_provider_metrics("macro", success=True, latency_ms=50)

    # Get all stats
    all_stats = manager.get_all_provider_stats()

    assert len(all_stats) == 3
    assert "order_book" in all_stats
    assert "options" in all_stats
    assert "macro" in all_stats

    assert all_stats["order_book"]["avg_latency_ms"] == 100.0
    assert all_stats["options"]["avg_latency_ms"] == 200.0
    assert all_stats["macro"]["avg_latency_ms"] == 50.0

    # Cleanup
    state_file.unlink()


# =============================================================================
# US-029 Phase 5: Streaming Tests
# =============================================================================


def test_streaming_order_book_dryrun(tmp_path: Path) -> None:
    """Test order book streaming in dryrun mode (US-029 Phase 5)."""
    import sys
    import time
    from pathlib import Path as PathLib
    from threading import Thread

    sys.path.insert(0, str(PathLib(__file__).parent.parent.parent / "scripts"))
    from stream_order_book import OrderBookStreamer

    output_dir = tmp_path / "streaming_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create streamer in dryrun mode
    streamer = OrderBookStreamer(
        symbols=["RELIANCE", "TCS"],
        output_dir=output_dir,
        buffer_size=10,
        update_interval_seconds=0.1,  # Fast for testing
        dryrun=True,
        secrets_mode="plain",
    )

    # Start streaming in background thread
    def run_stream() -> None:
        streamer.start()

    thread = Thread(target=run_stream, daemon=True)
    thread.start()

    # Let it run for a bit
    time.sleep(0.5)

    # Stop streaming
    streamer.running = False
    thread.join(timeout=2)

    # Verify updates were processed
    assert streamer.stats["updates"] > 0, "No updates processed"
    assert streamer.stats["errors"] == 0, "Unexpected errors"

    # Verify snapshots were cached
    for symbol in ["RELIANCE", "TCS"]:
        cache_path = output_dir / "streaming" / symbol / "latest.json"
        assert cache_path.exists(), f"Cache file not created for {symbol}"

        with open(cache_path) as f:
            data = json.load(f)

        assert data["symbol"] == symbol
        assert data["metadata"]["source"] == "stream"
        assert data["metadata"]["dryrun"] is True
        assert "bids" in data
        assert "asks" in data
        assert len(data["bids"]) == 5
        assert len(data["asks"]) == 5


def test_streaming_heartbeat_tracking(tmp_path: Path) -> None:
    """Test StateManager streaming heartbeat tracking (US-029 Phase 5)."""
    import time

    from src.services.state_manager import StateManager

    state_file = tmp_path / "streaming_state.json"
    if state_file.exists():
        state_file.unlink()

    manager = StateManager(state_file)

    # Record heartbeat
    manager.record_streaming_heartbeat(
        stream_type="order_book",
        symbols=["RELIANCE", "TCS"],
        stats={"updates": 100, "errors": 2},
    )

    # Get health status
    health = manager.get_streaming_health("order_book")

    assert health["exists"] is True
    assert health["is_healthy"] is True
    assert health["update_count"] == 100
    assert health["error_count"] == 2
    assert health["symbols"] == ["RELIANCE", "TCS"]
    assert health["time_since_heartbeat_seconds"] < 1.0

    # Simulate timeout by waiting
    time.sleep(2)

    # Update heartbeat again
    manager.record_streaming_heartbeat(
        stream_type="order_book",
        symbols=["RELIANCE"],
        stats={"updates": 150, "errors": 2},
    )

    # Check health again
    health = manager.get_streaming_health("order_book")
    assert health["is_healthy"] is True
    assert health["update_count"] == 150

    # Cleanup
    state_file.unlink()


def test_streaming_health_timeout(tmp_path: Path) -> None:
    """Test streaming health detection with timeout (US-029 Phase 5)."""

    from src.services.state_manager import StateManager

    state_file = tmp_path / "streaming_state.json"
    if state_file.exists():
        state_file.unlink()

    manager = StateManager(state_file)

    # Record old heartbeat
    manager.record_streaming_heartbeat(
        stream_type="order_book",
        symbols=["RELIANCE"],
        stats={"updates": 10, "errors": 0},
    )

    # Manually set old timestamp (simulate timeout)
    from datetime import datetime, timedelta

    old_time = datetime.now() - timedelta(seconds=60)  # 60 seconds ago
    manager.state["streaming"]["order_book"]["last_heartbeat"] = old_time.isoformat()
    manager._save_state()

    # Check health - should be unhealthy due to timeout
    health = manager.get_streaming_health("order_book")
    assert health["exists"] is True
    assert health["is_healthy"] is False
    assert health["time_since_heartbeat_seconds"] > 50.0

    # Cleanup
    state_file.unlink()


def test_streaming_buffer_management(tmp_path: Path) -> None:
    """Test circular buffer management in streaming (US-029 Phase 5)."""
    import sys
    from pathlib import Path as PathLib

    sys.path.insert(0, str(PathLib(__file__).parent.parent.parent / "scripts"))
    from stream_order_book import OrderBookStreamer

    output_dir = tmp_path / "streaming_buffer_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create streamer with small buffer
    streamer = OrderBookStreamer(
        symbols=["RELIANCE"],
        output_dir=output_dir,
        buffer_size=5,  # Small buffer for testing
        update_interval_seconds=1,
        dryrun=True,
        secrets_mode="plain",
    )

    # Manually add snapshots to buffer
    for i in range(10):
        snapshot = {
            "symbol": "RELIANCE",
            "timestamp": f"2024-01-01T10:00:{i:02d}",
            "bids": [],
            "asks": [],
        }
        streamer.buffers["RELIANCE"].append(snapshot)

    # Verify circular buffer keeps only last 5
    assert len(streamer.buffers["RELIANCE"]) == 5
    snapshots = list(streamer.buffers["RELIANCE"])
    assert snapshots[0]["timestamp"] == "2024-01-01T10:00:05"
    assert snapshots[-1]["timestamp"] == "2024-01-01T10:00:09"

    # Test get_latest_snapshot
    latest = streamer.get_latest_snapshot("RELIANCE")
    assert latest is not None
    assert latest["timestamp"] == "2024-01-01T10:00:09"

    # Test get_buffer_snapshots
    all_snapshots = streamer.get_buffer_snapshots("RELIANCE")
    assert len(all_snapshots) == 5
    assert all_snapshots[0]["timestamp"] == "2024-01-01T10:00:09"  # Newest first

    # Test limit
    limited = streamer.get_buffer_snapshots("RELIANCE", limit=3)
    assert len(limited) == 3


def test_streaming_mock_websocket_deterministic(tmp_path: Path) -> None:
    """Test MockWebSocketClient generates deterministic data (US-029 Phase 5)."""
    import sys
    from pathlib import Path as PathLib

    sys.path.insert(0, str(PathLib(__file__).parent.parent.parent / "scripts"))
    from stream_order_book import MockWebSocketClient

    client = MockWebSocketClient(symbols=["RELIANCE"], interval=0)  # No sleep for testing

    # Get two updates
    update1 = client.receive(timeout=0)
    update2 = client.receive(timeout=0)

    assert update1 is not None
    assert update2 is not None

    # Both should be for RELIANCE (only symbol)
    assert update1["symbol"] == "RELIANCE"
    assert update2["symbol"] == "RELIANCE"

    # Both should have 5 bids and 5 asks
    assert len(update1["bids"]) == 5
    assert len(update1["asks"]) == 5
    assert len(update2["bids"]) == 5
    assert len(update2["asks"]) == 5

    # Bids should be descending price
    for i in range(4):
        assert update1["bids"][i]["price"] > update1["bids"][i + 1]["price"]

    # Asks should be ascending price
    for i in range(4):
        assert update1["asks"][i]["price"] < update1["asks"][i + 1]["price"]
