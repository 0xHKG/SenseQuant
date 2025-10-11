# SenseQuant — Architecture (v1.1)

## 1. Module Map & Folder Layout

```
src/
├── adapters/              # External integrations
│   ├── breeze_client.py   # Breeze API wrapper (market data, orders)
│   └── sentiment_provider.py  # News/social sentiment feeds
├── domain/                # Core business logic
│   ├── strategies/
│   │   ├── base.py        # Abstract Strategy interface
│   │   ├── intraday.py    # IntradayStrategy (9:15–15:29 IST)
│   │   └── swing.py       # SwingStrategy (2–10 day holds)
│   ├── features.py        # TA indicators (SMA, RSI, VWAP, etc.)
│   └── models.py          # Data models (Signal, Position, Order, Bar)
├── services/              # Application services
│   ├── risk_manager.py    # Exposure caps, SL/TP, circuit-breaker
│   ├── position_sizer.py  # Kelly fraction / fixed notional
│   ├── execution.py       # Order placement, retry logic
│   └── teacher_student.py # EOD learning loop
├── app/                   # Application layer
│   ├── config.py          # Pydantic settings from .env
│   ├── engine.py          # Main event loop (live/backtest)
│   ├── logger.py          # Structured JSON logging
│   └── cli.py             # CLI entry (modes: live, backtest, dry-run)
└── main.py                # Entry point

tests/
├── unit/                  # Fast isolated tests
│   ├── test_breeze_client.py
│   ├── test_strategies.py
│   └── test_risk_manager.py
└── integration/           # End-to-end scenarios
    └── test_backtest.py

data/                      # Local cache
├── historical/            # Downloaded OHLCV CSVs
└── models/                # Trained student model weights

logs/                      # Runtime outputs
├── app.log                # Structured JSON logs
└── trades.csv             # Audit journal (timestamp, symbol, action, price, qty, P&L)
```

## 2. Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION (adapters)                                                 │
│    BreezeClient: get_historical() / subscribe_live()                    │
│    SentimentProvider: fetch_news() → sentiment_score(symbol)            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. FEATURE ENGINEERING (domain)                                         │
│    features.py: compute_indicators(bars) → {sma_50, rsi_14, vwap, ...} │
│    Merge sentiment_score into feature dict                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. SIGNAL GENERATION (domain/strategies)                                │
│    IntradayStrategy / SwingStrategy:                                    │
│      - consume features + sentiment                                     │
│      - output Signal(symbol, direction, confidence, reason)             │
│    Sentiment gating: if sentiment < -0.3 → suppress BUY signals         │
│    Sentiment boost: if sentiment > +0.5 → scale confidence by 1.2x      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. RISK & POSITION SIZING (services)                                    │
│    RiskManager:                                                         │
│      - check global exposure cap, daily loss limit                      │
│      - enforce circuit-breaker if triggered                             │
│      - attach SL/TP levels to signal                                    │
│    PositionSizer:                                                       │
│      - compute qty = f(confidence, available_capital, volatility)       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. EXECUTION (services/execution + adapters/breeze_client)              │
│    ExecutionService.place_order(order):                                 │
│      - if dry_run: log to console, return mock OrderID                  │
│      - else: BreezeClient.place_order() with retry on transient errors  │
│      - log response (order_id, status, timestamp)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. LOGGING & AUDIT (app/logger)                                         │
│    - Append to logs/app.log (JSON: {ts, level, component, msg})         │
│    - Append to logs/trades.csv (trade journal for analytics)            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. EOD TEACHER–STUDENT (services/teacher_student)                       │
│    - Teacher reviews day's trades + market context → generate labels    │
│    - Student retrains on expanding window → update params/weights       │
│    - Persist new config; rollback if validation loss degrades           │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3. Configuration Management (Pydantic)

**src/app/config.py**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Breeze API
    breeze_api_key: str
    breeze_api_secret: str
    breeze_session_token: str

    # Trading
    mode: str = "dry-run"  # "dry-run" | "live" | "backtest"
    symbols: list[str] = ["RELIANCE", "TCS", "INFY"]

    # Risk
    max_exposure_inr: float = 100000.0
    daily_loss_cap_inr: float = 5000.0
    default_stop_loss_pct: float = 2.0
    default_take_profit_pct: float = 4.0

    # Timing (IST)
    intraday_start: str = "09:15"
    intraday_cutoff: str = "15:29"

    # Teacher–Student
    teacher_retrain_days: int = 7
    student_model_path: str = "data/models/student.pkl"

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
```

## 4. Error Handling & Retries

### Breeze API (adapters/breeze_client.py)
```python
import tenacity

class BreezeClient:
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(f"Retry {retry_state.attempt_number}")
    )
    def get_historical(self, symbol, from_date, to_date, interval="1minute"):
        # Implements rate-limit backoff on HTTP 429
        ...

    def place_order(self, order: Order) -> OrderResponse:
        try:
            resp = self._breeze_api.place_order(...)
            return OrderResponse.parse(resp)
        except BreezeAPIError as e:
            if e.is_transient():  # network, 5xx
                # Retry once
                resp = self._breeze_api.place_order(...)
                return OrderResponse.parse(resp)
            else:
                logger.error(f"Non-retryable error: {e}")
                raise
```

### Live Feed Reconnection
- WebSocket disconnect → exponential backoff (2s, 4s, 8s, max 30s)
- Log reconnection attempts; alert if > 5 consecutive failures

## 5. Logging Policy

**Structured JSON format** (timestamp, level, component, message, context):
```json
{
  "timestamp": "2025-10-11T10:32:15.123+05:30",
  "level": "INFO",
  "component": "IntradayStrategy",
  "message": "Generated BUY signal",
  "context": {
    "symbol": "RELIANCE",
    "price": 2456.75,
    "confidence": 0.78,
    "sentiment": 0.42,
    "reason": "RSI oversold + positive sentiment"
  }
}
```

**Log Levels**:
- DEBUG: feature computations, intermediate calculations
- INFO: signals generated, orders placed, positions closed
- WARNING: risk limit approached, reconnection attempts
- ERROR: order failures, data feed errors
- CRITICAL: circuit-breaker triggered, unhandled exceptions

**Rotation**: daily at midnight IST; retain 30 days; compress after 7 days.

## 6. EOD Teacher–Student Process

**Objective**: Progressively refine student strategy parameters via walk-forward learning.

### Inputs (services/teacher_student.py)
1. **Historical window**: last N days of OHLCV + sentiment
2. **Trade journal**: logs/trades.csv (actual fills, P&L)
3. **Student params**: current thresholds (RSI levels, sentiment gates, position sizes)

### Teacher Logic (v1 — rule-based)
```python
def teacher_label(bars, trades) -> dict[str, float]:
    """
    Analyze EOD: label which signals were correct in hindsight.
    Returns: adjusted_params (e.g., rsi_buy_threshold: 28 → 25)
    """
    # Example rules:
    # - If BUY signal at RSI=30 led to +2% in 4h → reinforce threshold
    # - If sentiment > 0.5 but stock dropped → tighten sentiment gate
    # Output: {"rsi_buy": 25, "sentiment_gate": 0.6, "position_size_multiplier": 1.1}
```

### Student Update
- Retrain lightweight model (RandomForest / LightGBM) on teacher-labeled windows
- Validate on held-out week; compute Sharpe, win-rate
- **Rollback rule**: if validation Sharpe < 0.3 or win-rate < 45%, revert to previous params

### Outputs
- Updated `data/models/student.pkl` (model weights or param JSON)
- Log: `{"event": "student_update", "params_changed": {...}, "validation_sharpe": 0.78}`

### Schedule
- Runs daily at 16:00 IST (after market close)
- Walk-forward window: retrain every 7 days with expanding dataset

## 7. Non-Functional Constraints

### IST Timings (domain/strategies/intraday.py)
```python
def should_trade_intraday(now: datetime) -> bool:
    """Only allow intraday trades between 9:15–15:29 IST."""
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = now.astimezone(ist)
    start = now_ist.replace(hour=9, minute=15, second=0)
    cutoff = now_ist.replace(hour=15, minute=29, second=0)
    return start <= now_ist <= cutoff

def force_square_off(positions):
    """At 15:29 IST, close all intraday positions at market."""
    ...
```

### Memory Footprint
- Target: **< 300 MB RSS** under steady load
- Techniques:
  - Stream bars (don't load full historical arrays into memory)
  - Rolling window for features (keep last 200 bars in memory)
  - Periodic GC after EOD batch processing

### Graceful Shutdown (app/engine.py)
```python
def shutdown_handler(signum, frame):
    logger.info("Shutdown signal received")
    # 1. Stop accepting new signals
    # 2. Close all open positions (market orders)
    # 3. Flush logs/trades.csv
    # 4. Disconnect Breeze WebSocket
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)
```

## 8. Modes of Operation

### Live Mode (`mode=live`)
- Subscribe to Breeze WebSocket for live ticks
- Execute real orders via Breeze API
- Enforce all risk checks, timings, circuit-breakers

### Dry-Run Mode (`mode=dry-run`)
- Connect to live data feed
- Generate signals normally
- **Do NOT call Breeze place_order()** → log intended order to console
- Useful for paper-trading validation

### Backtest Mode (`mode=backtest`)
- Load historical bars from `data/historical/`
- Replay bar-by-bar; generate signals
- Simulate fills (assume market orders execute at next bar's open)
- Output metrics: total P&L, Sharpe, max drawdown, win-rate, avg hold time
- No live API calls

## 9. Traceability to PRD ACs

| PRD AC | Architecture Component | Implementation Notes |
|--------|------------------------|----------------------|
| **1. Intraday: open after 9:15, close by 15:29 IST** | `domain/strategies/intraday.py` | `should_trade_intraday()` guard; `force_square_off()` at 15:29 |
| **1. Swing: hold 2–10 days with SL/TP** | `domain/strategies/swing.py` | Signals tagged with `hold_days_target`; `RiskManager` enforces SL/TP |
| **2. Teacher generates labels** | `services/teacher_student.py` | `teacher_label()` analyzes EOD trades → param adjustments |
| **2. Student trains on teacher labels** | `services/teacher_student.py` | `retrain_student()` every N days; outputs `student.pkl` |
| **2. Walk-forward retrain every N days** | `services/teacher_student.py` | Scheduled at 16:00 IST; expanding window |
| **3. Ingest news/social feeds** | `adapters/sentiment_provider.py` | Pluggable provider (stub: simple NLP; later: API integrations) |
| **3. Compute sentiment score −1 to +1** | `adapters/sentiment_provider.py` | `sentiment_score(symbol) -> float` |
| **3. Gate/boost signals by sentiment** | `domain/strategies/*.py` | Logic in `_apply_sentiment_filter()` |
| **4. Fetch historical OHLCV with retry** | `adapters/breeze_client.py` | `@tenacity.retry` on `get_historical()` |
| **4. Subscribe to live market data; reconnect** | `adapters/breeze_client.py` | WebSocket reconnect with exponential backoff |
| **4. Place market/limit orders; retry once** | `adapters/breeze_client.py` + `services/execution.py` | Single retry on transient errors; log all responses |
| **4. Respect rate limits (429 backoff)** | `adapters/breeze_client.py` | Detect HTTP 429 → `time.sleep(backoff_seconds)` |
| **5. Global max exposure cap** | `services/risk_manager.py` | `check_exposure_limit()` before sizing |
| **5. Per-trade SL/TP enforced** | `services/risk_manager.py` | `attach_sl_tp(signal)` adds levels; monitor fills |
| **5. Daily loss cap → circuit-breaker** | `services/risk_manager.py` | Track cumulative P&L; `circuit_breaker_active` flag |
| **5. Exchange circuit-breaker awareness** | `services/risk_manager.py` | Listen for halt events from Breeze; pause trading |
| **6. Credentials/config in .env** | `app/config.py` | Pydantic `Settings` with `env_file=".env"` |
| **6. Structured JSON logs** | `app/logger.py` | JSON formatter; fields: timestamp, level, component, message, context |
| **6. Dry-run mode** | `app/engine.py` | `if config.mode == "dry-run": log_order()` (skip Breeze API) |
| **6. Backtest mode** | `app/engine.py` | Offline runner over `data/historical/` |
| **NF: ≤1 vCPU, <300MB RAM** | All modules | Stream processing; rolling windows; periodic GC |
| **NF: Graceful shutdown** | `app/engine.py` | Signal handlers close positions, flush logs |
| **NF: Modular code** | Folder layout | `adapters/`, `domain/`, `services/`, `app/` separation |
| **NF: ≥70% test coverage** | `tests/unit/`, `tests/integration/` | pytest; coverage.py reports |
| **NF: Docstrings** | All modules | Google-style docstrings for public functions |
| **Success: Live ticks → signals → orders** | `app/engine.py` | Main event loop orchestrates full pipeline |
| **Success: Backtest outputs Sharpe, max DD, win-rate** | `app/engine.py` (backtest mode) | `compute_metrics()` helper; print summary table |
| **Success: No unhandled exceptions in 24h** | Error handling + logging | Try/except at event loop level; log errors; continue |

---

**End of Architecture v1.1**
