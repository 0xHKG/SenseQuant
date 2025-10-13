# US-029 Phase 5: Real-Time Streaming & Background Ingestion

**Status**: SPECIFICATION (Ready for Implementation)
**Prerequisites**: Phase 4 Complete ✅
**Estimated Effort**: 3-5 days (senior developer)

---

## Executive Summary

Phase 5 adds real-time market data streaming capabilities and background ingestion automation to SenseQuant. This enables:
- **Real-time order book updates** via Breeze WebSocket
- **Automated background ingestion** for options/macro data
- **Streaming buffer integration** for live strategy execution
- **Enhanced monitoring** with heartbeat tracking and alerts

**Safety**: All streaming features disabled by default, dryrun mode required for testing.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SenseQuant Phase 5                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │  Breeze WSS  │───────▶│ stream_order │                  │
│  │  (Real-time) │        │   _book.py   │                  │
│  └──────────────┘        └──────┬───────┘                  │
│                                  │                           │
│  ┌──────────────┐               │                           │
│  │ Scheduled    │               ▼                           │
│  │ fetch_*.py   │        ┌──────────────┐                  │
│  │ (Background) │───────▶│  DataFeed    │                  │
│  └──────────────┘        │  (Streaming  │                  │
│                          │   Buffers)   │                  │
│  ┌──────────────┐        └──────┬───────┘                  │
│  │ StateManager │◀──────────────┘                          │
│  │ (Metrics +   │                                           │
│  │  Heartbeat)  │        ┌──────────────┐                  │
│  └──────┬───────┘        │ Monitoring   │                  │
│         │                │ Service      │                  │
│         └───────────────▶│ (Alerts)     │                  │
│                          └──────────────┘                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Component 1: Real-Time Streaming (`stream_order_book.py`)

### Purpose
Stream order book updates via Breeze WebSocket and cache snapshots for live strategies.

### Implementation Specification

```python
#!/usr/bin/env python3
"""Stream order book data via Breeze WebSocket (US-029 Phase 5).

Features:
- Real-time order book updates via WebSocket
- Configurable update interval and buffer size
- Graceful shutdown on SIGTERM/SIGINT
- State tracking for heartbeat monitoring
- Dryrun mode with deterministic mock stream

Usage:
    # Dryrun mode (mock WebSocket)
    python scripts/stream_order_book.py --dryrun

    # Live mode (requires Breeze credentials)
    python scripts/stream_order_book.py --symbols RELIANCE TCS --interval 1
"""

class OrderBookStreamer:
    """Manages real-time order book streaming via WebSocket."""

    def __init__(
        self,
        symbols: list[str],
        output_dir: Path,
        buffer_size: int = 100,
        update_interval_seconds: int = 1,
        dryrun: bool = False,
        secrets_mode: str = "plain",
    ):
        """Initialize streamer.

        Args:
            symbols: Stock symbols to stream
            output_dir: Cache directory for snapshots
            buffer_size: Max snapshots to buffer per symbol
            update_interval_seconds: Snapshot update interval
            dryrun: If True, use mock WebSocket
            secrets_mode: Secrets mode for credentials
        """
        self.symbols = symbols
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.update_interval = update_interval_seconds
        self.dryrun = dryrun

        # Statistics
        self.stats = {
            "updates": 0,
            "errors": 0,
            "last_heartbeat": None,
        }

        # Circular buffers (per symbol)
        self.buffers: dict[str, deque] = {
            symbol: deque(maxlen=buffer_size) for symbol in symbols
        }

        # Initialize WebSocket connection
        self.ws_client = self._create_websocket_client(secrets_mode)

        # State manager for heartbeat tracking
        self.state_manager = StateManager("data/state/streaming.json")

        # Shutdown flag
        self.running = False

    def _create_websocket_client(self, secrets_mode: str):
        """Create WebSocket client (Breeze or mock)."""
        if self.dryrun:
            return MockWebSocketClient(self.symbols, self.update_interval)
        else:
            # Real Breeze WebSocket client
            secrets = SecretsManager(mode=secrets_mode)
            api_key = secrets.get_secret("BREEZE_API_KEY", "")
            # ... create real Breeze WS client
            return BreezeWebSocketClient(api_key, ...)

    def start(self):
        """Start streaming (blocking)."""
        self.running = True

        # Register signal handlers
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        logger.info(f"Starting order book stream for {len(self.symbols)} symbols")

        # Subscribe to symbols
        self.ws_client.subscribe(self.symbols)

        # Main streaming loop
        while self.running:
            try:
                # Read from WebSocket
                update = self.ws_client.receive(timeout=5)

                if update:
                    self._process_update(update)

                # Record heartbeat
                self.state_manager.record_streaming_heartbeat(
                    stream_type="order_book",
                    symbols=self.symbols,
                    stats=self.stats,
                )

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                self.stats["errors"] += 1
                time.sleep(1)  # Back off on error

        # Cleanup
        self.ws_client.disconnect()
        logger.info("Order book stream stopped")

    def _process_update(self, update: dict):
        """Process WebSocket update and cache snapshot."""
        symbol = update["symbol"]

        # Create snapshot
        snapshot = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "bids": update["bids"],
            "asks": update["asks"],
            "metadata": {
                "source": "stream",
                "dryrun": self.dryrun,
            },
        }

        # Add to buffer
        if symbol in self.buffers:
            self.buffers[symbol].append(snapshot)

        # Write to cache
        cache_path = self._get_cache_path(symbol)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(snapshot, f)

        self.stats["updates"] += 1
        self.stats["last_heartbeat"] = datetime.now().isoformat()

    def _get_cache_path(self, symbol: str) -> Path:
        """Get path to latest snapshot cache."""
        return self.output_dir / "streaming" / symbol / "latest.json"

    def _shutdown_handler(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Shutdown signal received, stopping stream...")
        self.running = False


class MockWebSocketClient:
    """Mock WebSocket client for dryrun mode."""

    def __init__(self, symbols: list[str], interval: int):
        self.symbols = symbols
        self.interval = interval
        self.counter = 0

    def subscribe(self, symbols: list[str]):
        logger.info(f"[DRYRUN] Subscribed to {len(symbols)} symbols")

    def receive(self, timeout: int = 5):
        """Generate deterministic mock update."""
        time.sleep(self.interval)

        # Round-robin through symbols
        symbol = self.symbols[self.counter % len(self.symbols)]
        self.counter += 1

        # Generate mock order book
        base_price = 2000 + (hash(symbol) % 1000) + (self.counter % 10)

        return {
            "symbol": symbol,
            "bids": [
                {"price": base_price - i * 0.5, "quantity": 1000, "orders": 3}
                for i in range(1, 6)
            ],
            "asks": [
                {"price": base_price + i * 0.5, "quantity": 800, "orders": 2}
                for i in range(1, 6)
            ],
        }

    def disconnect(self):
        logger.info("[DRYRUN] WebSocket disconnected")
```

### Configuration

```python
# src/app/config.py additions

streaming_enabled: bool = Field(
    False, validation_alias="STREAMING_ENABLED"
)  # Master switch for streaming

streaming_buffer_size: int = Field(
    100, validation_alias="STREAMING_BUFFER_SIZE", ge=10, le=1000
)  # Max snapshots per symbol

streaming_update_interval_seconds: int = Field(
    1, validation_alias="STREAMING_UPDATE_INTERVAL_SECONDS", ge=1, le=60
)  # Update frequency

streaming_heartbeat_timeout_seconds: int = Field(
    30, validation_alias="STREAMING_HEARTBEAT_TIMEOUT_SECONDS", ge=5, le=300
)  # Heartbeat timeout for alerts
```

### Safety Controls

1. **Disabled by Default**: `STREAMING_ENABLED=false`
2. **Dryrun Required**: Must test with `--dryrun` before live
3. **Graceful Shutdown**: SIGINT/SIGTERM handlers
4. **Heartbeat Monitoring**: StateManager tracks last update
5. **Buffer Limits**: Prevents memory exhaustion
6. **Error Recovery**: Automatic reconnection on WebSocket failure

---

## Component 2: Background Ingestion (Enhanced Fetch Scripts)

### Purpose
Run fetch scripts as background daemons with scheduled updates.

### Implementation Approach

**Option A: Simple Cron Integration** (Recommended)
```bash
# /etc/cron.d/sensequant-ingestion

# Fetch order book snapshots every 5 minutes during market hours
*/5 9-15 * * 1-5 cd /app && python scripts/fetch_order_book.py --incremental

# Fetch options chain daily at market close
35 15 * * 1-5 cd /app && python scripts/fetch_options_data.py --incremental

# Fetch macro data daily at 6 PM
0 18 * * 1-5 cd /app && python scripts/fetch_macro_data.py --incremental
```

**Option B: Daemon Mode** (Advanced)

Add `--daemon` flag to fetch scripts:

```python
# scripts/fetch_order_book.py additions

def run_daemon_mode(
    fetcher: OrderBookFetcher,
    symbols: list[str],
    interval_seconds: int,
    state_manager: StateManager,
):
    """Run fetcher in daemon mode with periodic updates."""
    running = True

    def shutdown_handler(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received")
        running = False

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    logger.info(f"Starting background ingestion daemon (interval={interval_seconds}s)")

    while running:
        try:
            # Calculate next run time
            now = datetime.now()

            # Only run during market hours (9:15 - 15:30)
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                logger.debug("Before market hours, sleeping...")
                time.sleep(60)
                continue

            if now.hour > 15 or (now.hour == 15 and now.minute > 30):
                logger.debug("After market hours, sleeping...")
                time.sleep(60)
                continue

            # Run fetch
            start_time = datetime.now()

            for symbol in symbols:
                fetcher.fetch_snapshot(symbol, now)

            # Record metrics
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            state_manager.record_provider_metrics(
                provider_name="order_book",
                success=True,
                latency_ms=latency_ms,
            )

            # Sleep until next interval
            time.sleep(interval_seconds)

        except Exception as e:
            logger.error(f"Daemon error: {e}")
            time.sleep(10)  # Back off on error

    logger.info("Background ingestion daemon stopped")
```

### Rate Limiting

```python
class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, requests_per_second: float):
        self.rate = requests_per_second
        self.last_request = 0.0

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        time_since_last = now - self.last_request
        min_interval = 1.0 / self.rate

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request = time.time()
```

---

## Component 3: DataFeed Streaming Integration

### Purpose
Expose streaming buffers and latest snapshots to strategies.

### Implementation Specification

```python
# src/services/data_feed.py additions

class DataFeed:
    """Existing DataFeed class..."""

    def get_latest_order_book(
        self, symbol: str, max_age_seconds: int = 5
    ) -> dict | None:
        """Get latest order book snapshot from streaming cache.

        Args:
            symbol: Stock symbol
            max_age_seconds: Maximum age of snapshot (staleness check)

        Returns:
            Latest order book snapshot or None if stale/unavailable
        """
        cache_path = Path("data/order_book/streaming") / symbol / "latest.json"

        if not cache_path.exists():
            return None

        # Load snapshot
        with open(cache_path) as f:
            snapshot = json.load(f)

        # Check staleness
        snapshot_time = datetime.fromisoformat(snapshot["timestamp"])
        age = (datetime.now() - snapshot_time).total_seconds()

        if age > max_age_seconds:
            logger.warning(
                f"Stale order book snapshot for {symbol}: {age:.1f}s old"
            )
            return None

        return snapshot

    def get_streaming_buffer(
        self, symbol: str, lookback_count: int = 10
    ) -> list[dict]:
        """Get recent order book snapshots from streaming buffer.

        Note: This requires the streaming buffer to persist snapshots.
        For now, we only keep latest.json. In production, consider
        using Redis or similar for time-series buffer.

        Args:
            symbol: Stock symbol
            lookback_count: Number of recent snapshots

        Returns:
            List of recent snapshots (newest first)
        """
        # Placeholder: Would read from Redis/buffer in production
        latest = self.get_latest_order_book(symbol)
        return [latest] if latest else []
```

---

## Component 4: Monitoring Integration

### Purpose
Track streaming health and alert on issues.

### StateManager Additions

```python
# src/services/state_manager.py additions

def record_streaming_heartbeat(
    self,
    stream_type: str,
    symbols: list[str],
    stats: dict[str, Any],
) -> None:
    """Record streaming heartbeat for monitoring.

    Args:
        stream_type: Type of stream ("order_book", etc.)
        symbols: Symbols being streamed
        stats: Stream statistics (updates, errors, etc.)
    """
    if "streaming_heartbeats" not in self.state:
        self.state["streaming_heartbeats"] = {}

    if stream_type not in self.state["streaming_heartbeats"]:
        self.state["streaming_heartbeats"][stream_type] = {
            "start_time": datetime.now().isoformat(),
            "last_heartbeat": None,
            "total_updates": 0,
            "total_errors": 0,
        }

    stream_state = self.state["streaming_heartbeats"][stream_type]

    # Update heartbeat
    stream_state["last_heartbeat"] = datetime.now().isoformat()
    stream_state["total_updates"] += stats.get("updates", 0)
    stream_state["total_errors"] += stats.get("errors", 0)
    stream_state["symbols"] = symbols

    self._save_state()

def get_streaming_health(self, stream_type: str) -> dict[str, Any]:
    """Get streaming health status.

    Returns health metrics including lag and uptime.
    """
    heartbeats = self.state.get("streaming_heartbeats", {})
    stream_state = heartbeats.get(stream_type)

    if not stream_state:
        return {"status": "not_running"}

    # Calculate lag
    last_heartbeat = stream_state.get("last_heartbeat")
    if last_heartbeat:
        last_time = datetime.fromisoformat(last_heartbeat)
        lag_seconds = (datetime.now() - last_time).total_seconds()
    else:
        lag_seconds = None

    # Calculate uptime
    start_time = datetime.fromisoformat(stream_state["start_time"])
    uptime_seconds = (datetime.now() - start_time).total_seconds()

    return {
        "status": "running" if lag_seconds and lag_seconds < 60 else "stale",
        "lag_seconds": lag_seconds,
        "uptime_seconds": uptime_seconds,
        "total_updates": stream_state["total_updates"],
        "total_errors": stream_state["total_errors"],
        "symbols": stream_state.get("symbols", []),
    }
```

### MonitoringService Additions

```python
# src/services/monitoring.py additions

def check_streaming_health(self) -> dict[str, Any]:
    """Check streaming health and generate alerts.

    Returns alert if heartbeat timeout exceeded.
    """
    alerts = []

    # Check order book stream
    health = self.state_manager.get_streaming_health("order_book")

    if health["status"] == "not_running":
        # Not an error if streaming is disabled
        pass
    elif health["status"] == "stale":
        alerts.append({
            "severity": "critical",
            "component": "streaming",
            "message": f"Order book stream heartbeat timeout: {health['lag_seconds']:.0f}s",
            "details": health,
        })

    return {"streaming_healthy": len(alerts) == 0, "alerts": alerts}
```

---

## Component 5: Integration Tests

### Test Specification

```python
# tests/integration/test_market_streaming.py

def test_websocket_streaming_dryrun(tmp_path: Path) -> None:
    """Test WebSocket streaming in dryrun mode."""
    # ... test implementation

def test_streaming_heartbeat_tracking(tmp_path: Path) -> None:
    """Test StateManager tracks streaming heartbeats."""
    # ... test implementation

def test_streaming_alert_on_timeout(tmp_path: Path) -> None:
    """Test MonitoringService alerts on heartbeat timeout."""
    # ... test implementation

def test_streaming_buffer_cache(tmp_path: Path) -> None:
    """Test streaming snapshots cached correctly."""
    # ... test implementation

def test_graceful_shutdown(tmp_path: Path) -> None:
    """Test streaming handles shutdown signals."""
    # ... test implementation
```

---

## Operational Procedures

### Starting Streaming

```bash
# Dryrun test
python scripts/stream_order_book.py --dryrun --symbols RELIANCE TCS

# Production (with systemd)
sudo systemctl start sensequant-streaming
sudo systemctl status sensequant-streaming
```

### Monitoring

```bash
# Check streaming health
python -c "from src.services.state_manager import StateManager; \
           sm = StateManager('data/state/streaming.json'); \
           print(sm.get_streaming_health('order_book'))"

# Check heartbeat lag
watch -n 5 'tail -1 data/state/streaming.json | jq .streaming_heartbeats.order_book.last_heartbeat'
```

### Troubleshooting

**Symptom**: Heartbeat timeout alerts
**Cause**: WebSocket disconnected or network issue
**Fix**: Check logs, restart streaming service

**Symptom**: High error count in stats
**Cause**: Invalid data from WebSocket
**Fix**: Check Breeze API status, update parser

---

## Safety Checklist

- [ ] `STREAMING_ENABLED=false` by default
- [ ] Dryrun mode tested before live
- [ ] Graceful shutdown handlers registered
- [ ] Heartbeat monitoring configured
- [ ] Buffer size limits enforced
- [ ] Rate limiting implemented
- [ ] Error recovery tested
- [ ] Logs configured for debugging
- [ ] Systemd service file created
- [ ] Monitoring alerts integrated

---

## Estimated Implementation Timeline

**Day 1**: Streaming script skeleton + mock WebSocket
**Day 2**: StateManager heartbeat + DataFeed integration
**Day 3**: Monitoring integration + alerts
**Day 4**: Integration tests + documentation
**Day 5**: Production deployment + validation

---

## Next Steps

1. Review this specification with the team
2. Prioritize components (streaming vs. background ingestion)
3. Implement in phases with testing after each
4. Deploy to staging environment first
5. Monitor for 1 week before production

---

**Status**: Ready for implementation when team capacity allows.
