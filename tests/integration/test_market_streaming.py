"""Integration tests for market data streaming (US-029 Phase 5b)."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.data_feed import get_latest_order_book, get_order_book_history
from src.services.monitoring import MonitoringService
from src.services.state_manager import StateManager


def test_data_feed_get_latest_order_book(tmp_path: Path) -> None:
    """Test DataFeed get_latest_order_book helper (US-029 Phase 5b)."""
    streaming_dir = tmp_path / "streaming"
    symbol_dir = streaming_dir / "RELIANCE"
    symbol_dir.mkdir(parents=True, exist_ok=True)

    # Create mock latest.json
    snapshot = {
        "symbol": "RELIANCE",
        "timestamp": "2025-01-13T10:30:00",
        "bids": [
            {"price": 2500.0, "quantity": 1000, "orders": 3},
            {"price": 2499.5, "quantity": 800, "orders": 2},
        ],
        "asks": [
            {"price": 2500.5, "quantity": 900, "orders": 2},
            {"price": 2501.0, "quantity": 700, "orders": 1},
        ],
        "metadata": {"source": "stream", "dryrun": True},
    }

    latest_file = symbol_dir / "latest.json"
    with open(latest_file, "w") as f:
        json.dump(snapshot, f)

    # Test get_latest_order_book
    result = get_latest_order_book("RELIANCE", streaming_cache_dir=streaming_dir)

    assert result is not None
    assert result["symbol"] == "RELIANCE"
    assert len(result["bids"]) == 2
    assert len(result["asks"]) == 2
    assert result["bids"][0]["price"] == 2500.0
    assert result["asks"][0]["price"] == 2500.5


def test_data_feed_get_latest_order_book_fallback(tmp_path: Path) -> None:
    """Test DataFeed fallback to CSV cache (US-029 Phase 5b)."""
    streaming_dir = tmp_path / "streaming"
    csv_dir = tmp_path / "csv_cache"

    # Create CSV fallback snapshot
    symbol_dir = csv_dir / "RELIANCE" / "2025-01-13"
    symbol_dir.mkdir(parents=True, exist_ok=True)

    fallback_snapshot = {
        "symbol": "RELIANCE",
        "timestamp": "2025-01-13T09:00:00",
        "bids": [{"price": 2490.0, "quantity": 500, "orders": 1}],
        "asks": [{"price": 2491.0, "quantity": 400, "orders": 1}],
        "metadata": {"source": "csv_fallback"},
    }

    snapshot_file = symbol_dir / "snapshot_090000.json"
    with open(snapshot_file, "w") as f:
        json.dump(fallback_snapshot, f)

    # Test fallback (no streaming cache)
    result = get_latest_order_book(
        "RELIANCE", streaming_cache_dir=streaming_dir, fallback_csv_dir=csv_dir
    )

    assert result is not None
    assert result["symbol"] == "RELIANCE"
    assert result["metadata"]["source"] == "csv_fallback"


def test_data_feed_get_order_book_history(tmp_path: Path) -> None:
    """Test DataFeed get_order_book_history helper (US-029 Phase 5b)."""
    streaming_dir = tmp_path / "streaming"
    symbol_dir = streaming_dir / "RELIANCE"
    symbol_dir.mkdir(parents=True, exist_ok=True)

    # Create mock latest.json
    snapshot = {
        "symbol": "RELIANCE",
        "timestamp": "2025-01-13T10:30:00",
        "bids": [{"price": 2500.0, "quantity": 1000, "orders": 3}],
        "asks": [{"price": 2500.5, "quantity": 900, "orders": 2}],
        "metadata": {"source": "stream"},
    }

    latest_file = symbol_dir / "latest.json"
    with open(latest_file, "w") as f:
        json.dump(snapshot, f)

    # Test get_order_book_history
    history = get_order_book_history("RELIANCE", limit=10, streaming_cache_dir=streaming_dir)

    assert len(history) == 1
    assert history[0]["symbol"] == "RELIANCE"
    assert history[0]["timestamp"] == "2025-01-13T10:30:00"


def test_state_manager_buffer_metadata(tmp_path: Path) -> None:
    """Test StateManager buffer metadata persistence (US-029 Phase 5b)."""
    state_file = tmp_path / "streaming_state.json"
    manager = StateManager(state_file)

    # Record heartbeat with buffer metadata
    buffer_metadata = {
        "buffer_lengths": {"RELIANCE": 50, "TCS": 30},
        "last_snapshot_times": {
            "RELIANCE": "2025-01-13T10:30:00",
            "TCS": "2025-01-13T10:29:45",
        },
        "total_capacity": 100,
    }

    manager.record_streaming_heartbeat(
        stream_type="order_book",
        symbols=["RELIANCE", "TCS"],
        stats={"updates": 1000, "errors": 5},
        buffer_metadata=buffer_metadata,
    )

    # Get health and verify buffer metadata included
    health = manager.get_streaming_health("order_book")

    assert health["exists"] is True
    assert health["is_healthy"] is True
    assert health["buffer_lengths"] == {"RELIANCE": 50, "TCS": 30}
    assert health["last_snapshot_times"]["RELIANCE"] == "2025-01-13T10:30:00"
    assert health["total_capacity"] == 100
    assert "buffer_utilization_pct" in health
    # (50 + 30) / (100 * 2) * 100 = 40%
    assert health["buffer_utilization_pct"] == 40.0


def test_monitoring_service_streaming_health_check(tmp_path: Path) -> None:
    """Test MonitoringService streaming health check (US-029 Phase 5b)."""
    from src.app.config import Settings

    settings = Settings()  # type: ignore
    monitoring = MonitoringService(settings)

    state_file = tmp_path / "streaming_state.json"
    state_manager = StateManager(state_file)

    # Record healthy heartbeat
    state_manager.record_streaming_heartbeat(
        stream_type="order_book",
        symbols=["RELIANCE"],
        stats={"updates": 100, "errors": 0},
        buffer_metadata={
            "buffer_lengths": {"RELIANCE": 20},
            "last_snapshot_times": {"RELIANCE": datetime.now().isoformat()},
            "total_capacity": 100,
        },
    )

    # Check streaming health
    results = monitoring.check_streaming_health(state_manager)

    assert len(results) == 1
    assert results[0].check_name == "streaming_order_book"
    assert results[0].status == "OK"
    assert "healthy" in results[0].message.lower()


def test_monitoring_service_streaming_lag_alert(tmp_path: Path) -> None:
    """Test MonitoringService detects streaming lag (US-029 Phase 5b)."""
    from src.app.config import Settings

    settings = Settings()  # type: ignore
    monitoring = MonitoringService(settings)

    state_file = tmp_path / "streaming_state.json"
    state_manager = StateManager(state_file)

    # Record old heartbeat (simulates lag)
    state_manager.record_streaming_heartbeat(
        stream_type="order_book",
        symbols=["RELIANCE"],
        stats={"updates": 50, "errors": 2},
    )

    # Manually set old timestamp
    old_time = datetime.now() - timedelta(seconds=60)  # 60 seconds ago
    state_manager.state["streaming"]["order_book"]["last_heartbeat"] = old_time.isoformat()
    state_manager._save_state()

    # Check streaming health - should detect unhealthy state
    results = monitoring.check_streaming_health(state_manager)

    assert len(results) == 1
    assert results[0].check_name == "streaming_order_book"
    assert results[0].status == "ERROR"
    assert "unhealthy" in results[0].message.lower() or "lag" in results[0].message.lower()


def test_monitoring_service_streaming_high_buffer_utilization(tmp_path: Path) -> None:
    """Test MonitoringService warns on high buffer utilization (US-029 Phase 5b)."""
    from src.app.config import Settings

    settings = Settings()  # type: ignore
    monitoring = MonitoringService(settings)

    state_file = tmp_path / "streaming_state.json"
    state_manager = StateManager(state_file)

    # Record heartbeat with high buffer utilization
    state_manager.record_streaming_heartbeat(
        stream_type="order_book",
        symbols=["RELIANCE"],
        stats={"updates": 1000, "errors": 0},
        buffer_metadata={
            "buffer_lengths": {"RELIANCE": 95},  # 95/100 = 95% utilization
            "last_snapshot_times": {"RELIANCE": datetime.now().isoformat()},
            "total_capacity": 100,
        },
    )

    # Check streaming health
    results = monitoring.check_streaming_health(state_manager)

    assert len(results) == 1
    # Should be WARNING due to high buffer utilization (>90%)
    assert results[0].status == "WARNING"
    assert "buffer" in results[0].message.lower()


def test_streaming_integration_with_buffer_metadata(tmp_path: Path) -> None:
    """Test end-to-end streaming with buffer metadata (US-029 Phase 5b)."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    from stream_order_book import OrderBookStreamer

    output_dir = tmp_path / "streaming_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create streamer
    streamer = OrderBookStreamer(
        symbols=["RELIANCE", "TCS"],
        output_dir=output_dir,
        buffer_size=10,
        update_interval_seconds=0.05,  # Fast for testing
        dryrun=True,
        secrets_mode="plain",
    )

    # Start streaming in background
    def run_stream() -> None:
        streamer.start()

    thread = Thread(target=run_stream, daemon=True)
    thread.start()

    # Let it run briefly
    time.sleep(0.3)

    # Stop streaming
    streamer.running = False
    thread.join(timeout=2)

    # Verify buffer metadata was collected
    metadata = streamer._get_buffer_metadata()

    assert "buffer_lengths" in metadata
    assert "last_snapshot_times" in metadata
    assert "total_capacity" in metadata
    assert metadata["total_capacity"] == 10

    # Both symbols should have some data
    assert "RELIANCE" in metadata["buffer_lengths"]
    assert "TCS" in metadata["buffer_lengths"]
    assert metadata["buffer_lengths"]["RELIANCE"] > 0
    assert metadata["buffer_lengths"]["TCS"] > 0

    # Verify DataFeed can read the cached snapshots
    reliance_snapshot = get_latest_order_book(
        "RELIANCE", streaming_cache_dir=output_dir / "streaming"
    )
    assert reliance_snapshot is not None
    assert reliance_snapshot["symbol"] == "RELIANCE"


def test_monitoring_service_no_active_streams(tmp_path: Path) -> None:
    """Test MonitoringService handles no active streams gracefully (US-029 Phase 5b)."""
    from src.app.config import Settings

    settings = Settings()  # type: ignore
    monitoring = MonitoringService(settings)

    state_file = tmp_path / "empty_state.json"
    state_manager = StateManager(state_file)

    # Check with no streaming data
    results = monitoring.check_streaming_health(state_manager)

    assert len(results) == 1
    assert results[0].check_name == "streaming"
    assert results[0].status == "OK"
    assert "no active streams" in results[0].message.lower()
