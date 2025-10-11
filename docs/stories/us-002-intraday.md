# US-002 — Intraday Strategy v1

## Goal
Implement a minute-bar intraday strategy with configurable TA features and a sentiment gate, integrated with the engine. Enforce strict EOD square-off (≤ 15:29 IST). Dry-run must never place real orders but fully log/journal intents.

## Scope Files
- `src/domain/strategies/intraday.py`
- `src/services/engine.py`
- `src/app/config.py`
- `src/adapters/sentiment_provider.py` (used as a gate; stub acceptable)
- Tests:
  - `tests/unit/test_intraday.py`
  - `tests/integration/test_intraday_engine.py`

## Tasks

### 1) Config (Pydantic v2)
Add these to `Settings` (defaults shown):
- `INTRADAY_BAR_INTERVAL: Literal["1minute"] = "1minute"`
- `INTRADAY_FEATURE_LOOKBACK_MINUTES: int = 60`  (bars to fetch per tick)
- `INTRADAY_TICK_SECONDS: int = 5`  (engine sleep between ticks)
- `INTRADAY_SMA_PERIOD: int = 20`
- `INTRADAY_EMA_PERIOD: int = 50`
- `INTRADAY_RSI_PERIOD: int = 14`
- `INTRADAY_ATR_PERIOD: int = 14`
- `INTRADAY_LONG_RSI_MIN: int = 55`
- `INTRADAY_SHORT_RSI_MAX: int = 45`
- `SENTIMENT_POS_LIMIT: float = 0.15`
- `SENTIMENT_NEG_LIMIT: float = -0.15`

### 2) Features (minute bars)
In `src/domain/strategies/intraday.py`:
- `compute_features(df: pd.DataFrame, settings: Settings) -> pd.DataFrame`
  - Input columns: `ts, open, high, low, close, volume` (from adapter bars)
  - Compute: `sma20`, `ema50`, `rsi14`, `atr14`
  - Robust to insufficient rows → return df with required columns but last row flagged `valid=False`.

### 3) Signal logic (parameterized)
In `src/domain/strategies/intraday.py`:
- `signal(df: pd.DataFrame, settings: Settings, *, sentiment: float | None = None) -> Signal`
  - If `valid=False` or no data → `Signal(direction="FLAT", strength=0.0, meta={"reason": "insufficient"})`
  - Rules:
    - LONG if `close > sma20` and `rsi > INTRADAY_LONG_RSI_MIN`
    - SHORT if `close < sma20` and `rsi < INTRADAY_SHORT_RSI_MAX`
    - else FLAT
  - **Sentiment gate**:
    - If LONG and `sentiment < SENTIMENT_NEG_LIMIT` → force FLAT (`meta.reason="neg_sentiment_block"`)
    - If SHORT and `sentiment > SENTIMENT_POS_LIMIT` → force FLAT (`meta.reason="pos_sentiment_block"`)
  - Include `meta` with the last-row features snapshot (or a hash of it).

### 4) Engine integration
In `src/services/engine.py`:
- Add `square_off_intraday(symbol: str) -> None`: if mode=dryrun, log+journal intent; otherwise call `place_order` to close any open qty (assume flatting order).
- Extend loop to:
  - Each tick: fetch last `INTRADAY_FEATURE_LOOKBACK_MINUTES` via `historical_bars` for each symbol.
  - Build a DataFrame; call `compute_features` then `signal(...)` (pass sentiment from provider).
  - Emit **journal** entry on every decision (BUY/SELL/NOOP) with reason and features hash.
  - Respect NSE timing window 09:15–15:29 IST for taking new positions; always allow square-off before 15:29.
  - Sleep for `INTRADAY_TICK_SECONDS`.

### 5) Sentiment provider (stub ok)
In `src/adapters/sentiment_provider.py`:
- Add `get_sentiment(symbol: str) -> float`: default to 0.0 if no real feed; log component="sentiment".
- Return range [-1.0, 1.0].

### 6) Logging/Journal
- Component tags: "intraday", "engine", "sentiment".
- Journal fields: reuse the CSV schema from US-000; include `strategy="intraday"` and a concise `reason`.

## Acceptance Criteria

- **AC1 (Deterministic Signals):** Given fixture DataFrames, `signal()` returns expected LONG/SHORT/FLAT.
- **AC2 (Sentiment Gate):** LONG blocked if sentiment < NEG_LIMIT; SHORT blocked if sentiment > POS_LIMIT.
- **AC3 (EOD Square-off):** `square_off_intraday()` is called such that all intraday positions are closed ≤ 15:29 IST. In `MODE=dryrun`, only logs/journals, no network calls.
- **AC4 (Configurable):** Changing thresholds/periods in `Settings` changes decisions in tests.
- **AC5 (Engine Tick):** Integration test validates fetch→features→signal→journal flow; honors timing window.
- **AC6 (Quality):** `ruff`, `mypy`, `pytest` pass; strategy unit test coverage ≥ 75%.
- **AC7 (Logs/Journal):** Structured logs include component tags; journal entries created for decisions & square-off.

## Test Notes

- Unit (`tests/unit/test_intraday.py`):
  - Fixtures: small DataFrames with known TA values; sentiment values above/below thresholds.
  - Tests: LONG/SHORT/FLAT, gate blocks, insufficient data path.
- Integration (`tests/integration/test_intraday_engine.py`):
  - Mock adapter (historical_bars + place_order) and sentiment provider.
  - Simulate just one tick; assert journal writes and correct adapter invocations.
  - Time-window edges: 09:14 (no entries); 09:15 allowed; 15:29 square-off executed.

## Out of Scope
- Risk sizing & order qty (US-005)
- Backtester (US-006)
- Real sentiment feeds (US-004)
