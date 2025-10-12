# US-004 — Sentiment Providers v1

**Epic**: Sentiment Analysis Integration
**Priority**: High
**Estimate**: 3 days
**Dependencies**: US-001 (Breeze Adapter), US-002 (Intraday), US-003 (Swing)

---

## 1. Objective

Integrate sentiment analysis into trading strategies to gate (suppress) or boost signals based on news/social media sentiment. Implement provider abstraction with TTL caching and rate-limit guards to minimize external API calls.

---

## 2. Background

Current state:
- `src/adapters/sentiment_provider.py` exists but returns stub data
- Strategies (intraday, swing) do NOT consume sentiment scores
- No caching mechanism → would hit APIs on every signal generation

Required improvements:
- Abstract provider interface supporting multiple sources (news, Twitter, Reddit)
- TTL cache with configurable expiry (default: 1 hour for sentiment)
- Rate-limit guard to prevent API quota exhaustion
- Fallback to neutral sentiment (0.0) on errors with structured logging
- Integration into strategy signal generation with sentiment gating/boosting

---

## 3. Requirements

### 3.1 Functional Requirements

**FR-1**: Sentiment Provider Abstraction
- Abstract base class `SentimentProvider` with `get_sentiment(symbol: str) -> float`
- Returns score in range [-1.0, 1.0]: -1 (very negative) → 0 (neutral) → +1 (very positive)
- Implement `StubSentimentProvider` returning 0.0 (neutral) for v1
- Future providers: `NewsSentimentProvider`, `TwitterSentimentProvider`, `RedditSentimentProvider`

**FR-2**: TTL Cache with Rate-Limit Guard
- Cache sentiment scores with configurable TTL (default: 3600 seconds = 1 hour)
- Rate-limit guard: max N requests per minute per symbol (default: 10 req/min)
- Cache key: `f"sentiment:{symbol}:{provider_name}"`
- Cache hit → return cached value (log cache hit with TTL remaining)
- Cache miss → fetch from provider, cache result, return value
- Rate limit exceeded → return last cached value or 0.0 fallback

**FR-3**: Sentiment Gating in Strategies
- If sentiment < -0.3 → suppress BUY signals (log reason: "sentiment_gate")
- If sentiment > +0.5 → boost signal confidence by 1.2x (log reason: "sentiment_boost")
- Otherwise → no modification

**FR-4**: Structured Logging & Telemetry
- Log cache hits/misses with TTL remaining
- Log rate-limit events
- Log sentiment fallback events (errors, network failures)
- Include `sentiment_source` in signal meta and journal entries

**FR-5**: Configuration
- Add to `config.py`:
  - `sentiment_cache_ttl: int = 3600` (1 hour)
  - `sentiment_rate_limit_per_min: int = 10`
  - `sentiment_gate_threshold: float = -0.3`
  - `sentiment_boost_threshold: float = 0.5`
  - `sentiment_boost_multiplier: float = 1.2`

### 3.2 Non-Functional Requirements

**NFR-1**: Performance
- Cache lookup < 1ms
- No blocking I/O on cache hit
- Async-friendly design (optional for v1, prepare for v2)

**NFR-2**: Reliability
- Graceful degradation on provider failures → fallback to 0.0
- No crashes on network errors
- Cache survives across strategy invocations (in-memory for v1)

**NFR-3**: Observability
- Structured logs for cache telemetry
- Metrics: hit rate, miss rate, fallback rate, avg latency
- Include sentiment in journal CSV for post-trade analysis

---

## 4. Architecture

### 4.1 Module Structure

```
src/
├── adapters/
│   └── sentiment_provider.py         # Provider implementations (Stub, News, Twitter, etc.)
├── services/
│   └── sentiment/
│       ├── __init__.py                # Re-export SentimentCache
│       └── cache.py                   # TTL cache + rate-limit guard
├── domain/
│   └── strategies/
│       ├── intraday.py                # Updated: sentiment parameter in signal()
│       └── swing.py                   # Updated: sentiment parameter in signal()
└── services/
    └── engine.py                      # Updated: fetch sentiment, pass to strategies
```

### 4.2 Provider Abstraction

```python
# src/adapters/sentiment_provider.py
from abc import ABC, abstractmethod

class SentimentProvider(ABC):
    """Abstract base for sentiment providers."""

    @abstractmethod
    def get_sentiment(self, symbol: str) -> float:
        """
        Fetch sentiment score for symbol.

        Returns:
            float: Sentiment in range [-1.0, 1.0]
                   -1.0: Very negative
                    0.0: Neutral
                   +1.0: Very positive

        Raises:
            SentimentProviderError: On fetch failures
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/caching."""
        pass


class StubSentimentProvider(SentimentProvider):
    """Stub provider returning neutral sentiment (0.0) for v1."""

    def get_sentiment(self, symbol: str) -> float:
        """Always return neutral sentiment."""
        return 0.0

    @property
    def name(self) -> str:
        return "stub"
```

### 4.3 Cache Architecture

```python
# src/services/sentiment/cache.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import time

@dataclass
class CacheEntry:
    """Cached sentiment entry with TTL and metadata."""
    value: float
    timestamp: float
    provider: str


class SentimentCache:
    """TTL cache with rate-limit guard for sentiment providers."""

    def __init__(self, ttl_seconds: int = 3600, rate_limit_per_min: int = 10):
        self._cache: Dict[str, CacheEntry] = {}
        self._ttl = ttl_seconds
        self._rate_limit = rate_limit_per_min
        self._request_times: Dict[str, list] = {}  # symbol -> [timestamps]

    def get(
        self,
        symbol: str,
        provider: SentimentProvider,
        fallback: float = 0.0
    ) -> tuple[float, dict]:
        """
        Get sentiment with caching and rate-limit guard.

        Returns:
            (sentiment_score, metadata)
            metadata includes: cache_hit, ttl_remaining, rate_limited, etc.
        """
        cache_key = f"sentiment:{symbol}:{provider.name}"
        now = time.time()

        # Check cache
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            age = now - entry.timestamp
            if age < self._ttl:
                # Cache hit
                ttl_remaining = self._ttl - age
                logger.debug(
                    f"Sentiment cache hit for {symbol}",
                    extra={
                        "component": "sentiment_cache",
                        "symbol": symbol,
                        "provider": provider.name,
                        "ttl_remaining": ttl_remaining,
                    }
                )
                return entry.value, {
                    "cache_hit": True,
                    "ttl_remaining": ttl_remaining,
                    "provider": provider.name,
                }

        # Check rate limit
        if self._is_rate_limited(symbol):
            logger.warning(
                f"Sentiment rate limit exceeded for {symbol}",
                extra={"component": "sentiment_cache", "symbol": symbol}
            )
            # Return cached value if available, else fallback
            if cache_key in self._cache:
                return self._cache[cache_key].value, {
                    "cache_hit": False,
                    "rate_limited": True,
                    "provider": provider.name,
                }
            return fallback, {
                "cache_hit": False,
                "rate_limited": True,
                "fallback": True,
            }

        # Fetch from provider
        try:
            value = provider.get_sentiment(symbol)
            self._cache[cache_key] = CacheEntry(value, now, provider.name)
            self._record_request(symbol)

            logger.debug(
                f"Sentiment cache miss for {symbol}",
                extra={
                    "component": "sentiment_cache",
                    "symbol": symbol,
                    "provider": provider.name,
                    "value": value,
                }
            )

            return value, {
                "cache_hit": False,
                "provider": provider.name,
                "rate_limited": False,
            }
        except Exception as e:
            logger.error(
                f"Sentiment provider failed for {symbol}: {e}",
                extra={
                    "component": "sentiment_cache",
                    "symbol": symbol,
                    "provider": provider.name,
                    "error": str(e),
                }
            )
            # Return cached value if available, else fallback
            if cache_key in self._cache:
                return self._cache[cache_key].value, {
                    "cache_hit": True,
                    "stale": True,
                    "error": str(e),
                }
            return fallback, {
                "cache_hit": False,
                "fallback": True,
                "error": str(e),
            }

    def _is_rate_limited(self, symbol: str) -> bool:
        """Check if symbol has exceeded rate limit."""
        now = time.time()
        window_start = now - 60  # Last 60 seconds

        if symbol not in self._request_times:
            return False

        # Clean old requests
        self._request_times[symbol] = [
            t for t in self._request_times[symbol] if t > window_start
        ]

        return len(self._request_times[symbol]) >= self._rate_limit

    def _record_request(self, symbol: str):
        """Record a request timestamp for rate limiting."""
        if symbol not in self._request_times:
            self._request_times[symbol] = []
        self._request_times[symbol].append(time.time())
```

### 4.4 Strategy Integration

```python
# src/domain/strategies/swing.py (updated signal function)
def signal(
    df: pd.DataFrame,
    settings: Settings,
    position: SwingPosition | None = None,
    sentiment_score: float = 0.0,  # NEW PARAMETER
    sentiment_meta: dict | None = None,  # NEW PARAMETER
) -> Signal:
    """Generate swing signal with sentiment gating/boosting."""

    # ... existing crossover detection ...

    if bullish_crossover:
        # SENTIMENT GATING
        if sentiment_score < settings.sentiment_gate_threshold:
            logger.info(
                "Bullish signal suppressed by negative sentiment",
                extra={
                    "component": "swing",
                    "symbol": df.iloc[-1].get("symbol", "UNKNOWN"),
                    "sentiment": sentiment_score,
                }
            )
            return Signal(
                direction="FLAT",
                confidence=0.0,
                meta={
                    "reason": "sentiment_gate",
                    "sentiment": sentiment_score,
                    "sentiment_source": sentiment_meta.get("provider") if sentiment_meta else "unknown",
                }
            )

        # SENTIMENT BOOSTING
        confidence = 0.75
        if sentiment_score > settings.sentiment_boost_threshold:
            confidence *= settings.sentiment_boost_multiplier
            logger.info(
                "Bullish signal boosted by positive sentiment",
                extra={
                    "component": "swing",
                    "sentiment": sentiment_score,
                    "original_confidence": 0.75,
                    "boosted_confidence": confidence,
                }
            )

        return Signal(
            direction="LONG",
            confidence=confidence,
            meta={
                "reason": "bullish_crossover",
                "sentiment": sentiment_score,
                "sentiment_source": sentiment_meta.get("provider") if sentiment_meta else "unknown",
                "sma_fast": today_fast,
                "sma_slow": today_slow,
            }
        )
```

### 4.5 Engine Integration

```python
# src/services/engine.py (updated run_swing_daily)
def run_swing_daily(self, symbol: str) -> None:
    """Process daily swing evaluation with sentiment."""

    # ... existing bar fetching and feature computation ...

    # Fetch sentiment with caching
    sentiment_score, sentiment_meta = self._sentiment_cache.get(
        symbol,
        self._sentiment_provider,
        fallback=0.0
    )

    logger.debug(
        f"Sentiment for {symbol}: {sentiment_score:.2f}",
        extra={
            "component": "engine",
            "symbol": symbol,
            "sentiment": sentiment_score,
            **sentiment_meta,
        }
    )

    # Generate signal with sentiment
    sig = swing_signal(
        df_features,
        settings,
        position=current_position,
        sentiment_score=sentiment_score,
        sentiment_meta=sentiment_meta,
    )

    # ... existing entry/exit logic ...

    # Include sentiment in journal
    self.journal.log(
        symbol=symbol,
        action=sig.direction,
        qty=qty,
        price=entry_price,
        pnl=0.0,
        reason=sig.meta.get("reason", "entry"),
        mode=settings.mode,
        order_id="DRYRUN",
        status="ENTRY",
        strategy="swing",
        meta_json=str(sig.meta),  # Includes sentiment and sentiment_source
    )
```

---

## 5. Acceptance Criteria

### AC-1: Provider Abstraction
- [ ] `SentimentProvider` abstract base class with `get_sentiment()` method
- [ ] `StubSentimentProvider` returns 0.0 for all symbols
- [ ] Provider has `name` property for cache keys

### AC-2: TTL Cache
- [ ] `SentimentCache` stores sentiment with TTL (default 3600s)
- [ ] Cache hit returns cached value + metadata (cache_hit=True, ttl_remaining)
- [ ] Cache miss fetches from provider, caches result, returns value
- [ ] Expired entries trigger re-fetch

### AC-3: Rate-Limit Guard
- [ ] Rate limit: max 10 requests/minute per symbol (configurable)
- [ ] Exceeded limit returns cached value or fallback 0.0
- [ ] Rate-limit events logged with `rate_limited=True`

### AC-4: Fallback on Errors
- [ ] Provider exceptions caught → return cached value or 0.0
- [ ] Network errors logged with structured logging
- [ ] Stale cache served on errors (logged as `stale=True`)

### AC-5: Sentiment Gating
- [ ] Sentiment < -0.3 suppresses BUY signals (reason: "sentiment_gate")
- [ ] Log includes sentiment score and source

### AC-6: Sentiment Boosting
- [ ] Sentiment > +0.5 boosts confidence by 1.2x
- [ ] Log includes original and boosted confidence

### AC-7: Configuration
- [ ] Add to `config.py`: `sentiment_cache_ttl`, `sentiment_rate_limit_per_min`
- [ ] Add gating/boosting thresholds and multiplier

### AC-8: Engine Integration
- [ ] Engine fetches sentiment via cache before signal generation
- [ ] Sentiment passed to strategy `signal()` functions
- [ ] Sentiment included in journal meta (sentiment_source)

### AC-9: Structured Logging
- [ ] Cache hits/misses logged with component="sentiment_cache"
- [ ] Rate-limit events logged
- [ ] Fallback events logged with error details

### AC-10: Tests
- [ ] Unit tests: cache hit, miss, expiry, rate-limit
- [ ] Integration tests: engine sentiment flow, fallback path
- [ ] Verify journal includes sentiment_source

### AC-11: Quality Gates
- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `mypy src` passes
- [ ] `pytest -q` passes (all tests)

---

## 6. Testing Strategy

### 6.1 Unit Tests

**tests/unit/test_sentiment_cache.py**
```python
def test_cache_hit():
    """Cache returns cached value within TTL."""

def test_cache_miss():
    """Cache fetches from provider on miss."""

def test_cache_expiry():
    """Expired cache entry triggers re-fetch."""

def test_rate_limit():
    """Rate limit returns cached/fallback value."""

def test_provider_error_with_cache():
    """Provider error returns stale cache."""

def test_provider_error_without_cache():
    """Provider error returns fallback 0.0."""

def test_rate_limit_window():
    """Rate limit resets after 60 seconds."""
```

### 6.2 Integration Tests

**tests/integration/test_engine_sentiment.py**
```python
def test_engine_sentiment_gating():
    """Engine suppresses BUY on negative sentiment."""

def test_engine_sentiment_boosting():
    """Engine boosts confidence on positive sentiment."""

def test_engine_sentiment_cache_hit():
    """Second call uses cached sentiment (logs cache_hit=True)."""

def test_engine_sentiment_fallback():
    """Provider error falls back to 0.0 with structured logs."""

def test_journal_includes_sentiment():
    """Journal CSV includes sentiment and sentiment_source."""
```

---

## 7. Implementation Tasks

### Task 1: Configuration
- [ ] Add sentiment settings to `src/app/config.py`

### Task 2: Provider Abstraction
- [ ] Update `src/adapters/sentiment_provider.py` with abstract base
- [ ] Implement `StubSentimentProvider`

### Task 3: Cache Infrastructure
- [ ] Create `src/services/sentiment/__init__.py`
- [ ] Create `src/services/sentiment/cache.py` with TTL + rate-limit

### Task 4: Strategy Integration
- [ ] Update `src/domain/strategies/swing.py` signal() signature
- [ ] Add sentiment gating logic
- [ ] Add sentiment boosting logic
- [ ] Update `src/domain/strategies/intraday.py` similarly

### Task 5: Engine Integration
- [ ] Update `src/services/engine.py` to initialize cache + provider
- [ ] Fetch sentiment before signal generation
- [ ] Pass sentiment to strategies
- [ ] Include sentiment in journal

### Task 6: Unit Tests
- [ ] Create `tests/unit/test_sentiment_cache.py`

### Task 7: Integration Tests
- [ ] Create `tests/integration/test_engine_sentiment.py`

### Task 8: Quality Gates
- [ ] Run ruff, mypy, pytest
- [ ] Fix any issues

---

## 8. Verification Commands

```bash
# Linting
python -m ruff check .

# Formatting
python -m ruff format --check .

# Type checking
python -m mypy src

# Tests
python -m pytest -q

# Coverage (optional)
python -m pytest --cov=src/services/sentiment --cov=src/adapters/sentiment_provider -v
```

---

## 9. Future Enhancements (Out of Scope for v1)

- **US-004.1**: News Sentiment Provider (NewsAPI, Alpha Vantage)
- **US-004.2**: Twitter Sentiment Provider (Twitter API v2)
- **US-004.3**: Reddit Sentiment Provider (PRAW)
- **US-004.4**: Aggregated Sentiment (weighted average of multiple providers)
- **US-004.5**: Persistent cache (Redis, SQLite) for cross-process sharing
- **US-004.6**: Async sentiment fetching (background workers)

---

**End of US-004 Story**
