# US-015 — External Sentiment Provider Integration

**Status**: In Progress
**Priority**: High
**Complexity**: High
**Estimated Effort**: 8-12 hours

---

## Problem Statement

The current sentiment analysis implementation uses a stub provider that returns mock sentiment scores. For production trading, we need real-time sentiment data from multiple external sources (news APIs, social media, etc.) to inform trading decisions. The system lacks:

1. **Real Data Sources**: No integration with actual sentiment providers (NewsAPI, Twitter, Reddit, etc.)
2. **Reliability**: Single point of failure - if one provider fails, sentiment analysis fails
3. **Rate Limit Handling**: No exponential backoff or rate limit management for API calls
4. **Provider Diversity**: Cannot combine multiple sentiment signals for robust analysis
5. **Audit Trail**: No tracking of which providers contributed to sentiment scores
6. **Configuration Flexibility**: Cannot easily add/remove providers or adjust weights

This limits the system's ability to make informed trading decisions based on market sentiment.

---

## Objectives

1. **Pluggable Architecture**: Create abstract sentiment provider interface supporting multiple implementations
2. **Multiple Data Sources**: Integrate NewsAPI and Twitter as initial external providers
3. **Resilience**: Implement rate limiting, exponential backoff, and graceful degradation
4. **Hybrid Scoring**: Support weighted averaging of multiple provider scores
5. **Provider Registry**: Central registry managing provider lifecycle and fallback ordering
6. **Enhanced Caching**: Track provider-level statistics (hits/misses, latency, errors)
7. **Audit Trail**: Log per-provider contributions and persist last successful payloads
8. **Configuration**: Extend Settings with API credentials, endpoints, and provider weights
9. **Backward Compatibility**: Maintain existing sentiment interface for Engine integration

---

## Requirements

### FR-1: Abstract Sentiment Provider Interface

**Description**: Define abstract base class for all sentiment providers

**Acceptance Criteria**:
- [ ] Abstract `SentimentProvider` class with `get_sentiment()` method
- [ ] Standardized return format: `SentimentScore` with value (-1.0 to 1.0), confidence (0.0 to 1.0), source, timestamp
- [ ] Provider metadata: name, version, rate_limit_per_minute
- [ ] Health check method: `is_healthy()` returning provider availability status
- [ ] Async support: Optional async `get_sentiment_async()` for concurrent fetching

### FR-2: NewsAPI Provider Implementation

**Description**: Integrate NewsAPI REST API for news sentiment analysis

**Acceptance Criteria**:
- [ ] `NewsAPIProvider` class implementing `SentimentProvider`
- [ ] Fetch recent articles for given symbol/keyword
- [ ] Normalize JSON response into standardized `SentimentScore`
- [ ] Rate limiting: Configurable requests per minute (default: 100/min for free tier)
- [ ] Exponential backoff: Retry with 2^n second delays (max 3 retries)
- [ ] Error handling: Graceful degradation on API failures (log error, return None)
- [ ] Timeout: 5-second request timeout with configurable override
- [ ] API key validation: Raise configuration error if credentials missing

### FR-3: Twitter API Provider Implementation

**Description**: Integrate Twitter API v2 for social media sentiment analysis

**Acceptance Criteria**:
- [ ] `TwitterAPIProvider` class implementing `SentimentProvider`
- [ ] Search recent tweets for given symbol/cashtag
- [ ] Normalize JSON response into standardized `SentimentScore`
- [ ] Rate limiting: Configurable requests per minute (default: 450/min for essential tier)
- [ ] Exponential backoff: Retry with 2^n second delays (max 3 retries)
- [ ] Error handling: Graceful degradation on API failures
- [ ] Bearer token authentication
- [ ] Tweet filtering: Exclude retweets, filter by language (default: English)

### FR-4: Sentiment Provider Registry

**Description**: Central registry managing multiple providers with fallback logic

**Acceptance Criteria**:
- [ ] `SentimentProviderRegistry` class managing provider instances
- [ ] Register providers with unique names and weights
- [ ] Fallback ordering: Attempt providers in priority order until success
- [ ] Weighted averaging: Combine multiple provider scores using configured weights
- [ ] Provider health monitoring: Track success/failure rates per provider
- [ ] Circuit breaker: Temporarily disable providers after N consecutive failures (default: 5)
- [ ] Factory method: `create_registry_from_settings()` instantiating providers from config

### FR-5: Enhanced Settings Configuration

**Description**: Extend Settings with sentiment provider configuration

**Acceptance Criteria**:
- [ ] NewsAPI settings: `newsapi_api_key`, `newsapi_endpoint`, `newsapi_rate_limit_per_min`
- [ ] Twitter settings: `twitter_bearer_token`, `twitter_endpoint`, `twitter_rate_limit_per_min`
- [ ] Provider weights: `sentiment_provider_weights` (dict mapping provider names to weights)
- [ ] Fallback order: `sentiment_provider_fallback_order` (list of provider names)
- [ ] Circuit breaker: `sentiment_circuit_breaker_threshold` (failures before disable)
- [ ] Timeout: `sentiment_provider_timeout_seconds` (default: 5)
- [ ] Enable/disable: `sentiment_enable_newsapi`, `sentiment_enable_twitter`
- [ ] Validation: Ensure weights sum to 1.0, fallback order contains valid provider names

### FR-6: Enhanced Sentiment Cache

**Description**: Extend cache to track provider-level statistics and audit trail

**Acceptance Criteria**:
- [ ] Track per-provider stats: hits, misses, latency_ms, errors, last_success_ts
- [ ] Persist last successful payload per provider as JSON for audit
- [ ] Cache metadata includes: provider_name, fetch_timestamp, ttl_seconds
- [ ] Stats aggregation: Rolling window statistics (last 100 requests)
- [ ] Health metrics: Success rate, average latency per provider
- [ ] Export method: `get_provider_stats()` returning dict of provider statistics

### FR-7: Engine Integration

**Description**: Wire sentiment provider registry into Engine for live sentiment analysis

**Acceptance Criteria**:
- [ ] Engine constructor accepts optional `sentiment_registry` parameter
- [ ] Fallback to stub provider when registry not provided (backward compatibility)
- [ ] Log per-provider contributions: "Sentiment from NewsAPI: 0.45 (conf=0.8), Twitter: 0.62 (conf=0.9), Weighted: 0.54"
- [ ] Cache sentiment scores using enhanced cache with provider metadata
- [ ] Handle provider failures gracefully: Log warning, continue with available providers
- [ ] Expose sentiment stats via Engine: `get_sentiment_health()` method

### FR-8: Comprehensive Testing

**Description**: Unit and integration tests ensuring reliability

**Acceptance Criteria**:
- [ ] Unit tests for NewsAPI provider: Happy path, rate limit, timeout, error handling
- [ ] Unit tests for Twitter provider: Happy path, rate limit, authentication, error handling
- [ ] Unit tests for registry: Weighted averaging, fallback logic, circuit breaker
- [ ] Unit tests for enhanced cache: Provider stats tracking, audit trail persistence
- [ ] Integration test: Multi-provider sentiment feeding strategy decisions
- [ ] Integration test: Provider failure cascading to fallback
- [ ] Mock external APIs: No real API calls in tests
- [ ] Coverage target: 90%+ for new sentiment modules

---

## Architecture Design

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Engine                              │
│  - tick_intraday()                                          │
│  - run_swing_daily()                                        │
└────────────────┬────────────────────────────────────────────┘
                 │ uses
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              SentimentProviderRegistry                      │
│  - providers: dict[str, SentimentProvider]                  │
│  - weights: dict[str, float]                                │
│  - fallback_order: list[str]                                │
│  + get_sentiment(symbol) -> SentimentScore                  │
│  + get_provider_health() -> dict[str, HealthMetrics]        │
└────────────────┬────────────────────────────────────────────┘
                 │ manages
                 ▼
┌────────────────────────────────────────────────────────────┐
│              SentimentProvider (ABC)                       │
│  + get_sentiment(symbol) -> SentimentScore | None          │
│  + is_healthy() -> bool                                    │
│  + get_metadata() -> ProviderMetadata                      │
└────────────────┬───────────────────────────────────────────┘
                 │ implements
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐ │
│  │ NewsAPIProvider  │  │TwitterAPIProvider│  │StubProvider│ │
│  │ - api_key        │  │ - bearer_token   │  │ (testing)  │ │
│  │ - rate_limiter   │  │ - rate_limiter   │  │            │ │
│  │ - backoff_retry  │  │ - backoff_retry  │  │            │ │
│  └──────────────────┘  └──────────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────┘
                 │ writes to
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              EnhancedSentimentCache                         │
│  - cache: dict[str, CachedSentiment]                        │
│  - provider_stats: dict[str, ProviderStats]                 │
│  + get(symbol, provider) -> SentimentScore | None           │
│  + set(symbol, provider, score) -> None                     │
│  + get_provider_stats(provider) -> ProviderStats            │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

**Scenario 1: Multi-Provider Sentiment Fetch (Happy Path)**

1. Engine calls `sentiment_registry.get_sentiment("RELIANCE")`
2. Registry checks cache for recent sentiment (< TTL)
3. Cache miss → Registry iterates enabled providers (NewsAPI, Twitter)
4. NewsAPI fetches articles, normalizes sentiment: `SentimentScore(value=0.45, confidence=0.8, source="newsapi")`
5. Twitter fetches tweets, normalizes sentiment: `SentimentScore(value=0.62, confidence=0.9, source="twitter")`
6. Registry computes weighted average: `(0.45 * 0.6 + 0.62 * 0.4) / 1.0 = 0.518`
7. Registry returns `SentimentScore(value=0.518, confidence=0.85, source="hybrid[newsapi,twitter]")`
8. Cache stores result with provider metadata
9. Engine logs: "Sentiment for RELIANCE: 0.518 (NewsAPI: 0.45, Twitter: 0.62)"

**Scenario 2: Provider Failure with Fallback**

1. Engine calls `sentiment_registry.get_sentiment("TCS")`
2. Registry attempts NewsAPI (priority 1)
3. NewsAPI rate limit exceeded → Exponential backoff (2s, 4s, 8s) → All retries fail
4. Registry logs error, marks NewsAPI unhealthy, moves to next provider
5. Registry attempts Twitter (priority 2)
6. Twitter returns `SentimentScore(value=0.35, confidence=0.7, source="twitter")`
7. Registry returns Twitter score (no weighted average with single provider)
8. Cache stores result with metadata: `{"provider": "twitter", "fallback_reason": "newsapi_rate_limit"}`
9. Engine logs: "Sentiment for TCS: 0.35 (Twitter only, NewsAPI failed)"

**Scenario 3: Circuit Breaker Activation**

1. NewsAPI fails 5 consecutive requests (threshold reached)
2. Registry activates circuit breaker for NewsAPI (30-minute cooldown)
3. Future requests skip NewsAPI entirely until cooldown expires
4. Registry logs: "Circuit breaker OPEN for NewsAPI (5 consecutive failures)"
5. After 30 minutes, registry attempts half-open state (single test request)
6. If test succeeds → Circuit closes, NewsAPI re-enabled
7. If test fails → Circuit remains open, cooldown resets

---

## Implementation Plan

### Phase 1: Core Abstractions (2-3 hours)

1. **Create sentiment provider module structure**:
   ```
   src/services/sentiment/
   ├── __init__.py
   ├── base.py              # Abstract base classes
   ├── types.py             # SentimentScore, ProviderMetadata
   ├── registry.py          # SentimentProviderRegistry
   ├── cache.py             # Enhanced cache (extends existing)
   └── providers/
       ├── __init__.py
       ├── stub.py          # Existing stub provider (refactored)
       ├── news_api.py      # NewsAPI implementation
       └── twitter_api.py   # Twitter implementation
   ```

2. **Define core types** (`types.py`):
   ```python
   @dataclass
   class SentimentScore:
       value: float          # -1.0 to 1.0
       confidence: float     # 0.0 to 1.0
       source: str          # Provider name
       timestamp: datetime
       metadata: dict[str, Any] = field(default_factory=dict)

   @dataclass
   class ProviderMetadata:
       name: str
       version: str
       rate_limit_per_minute: int
       supports_async: bool

   @dataclass
   class ProviderStats:
       hits: int
       misses: int
       errors: int
       avg_latency_ms: float
       success_rate: float
       last_success_ts: datetime | None
   ```

3. **Implement abstract base** (`base.py`):
   ```python
   class SentimentProvider(ABC):
       @abstractmethod
       def get_sentiment(self, symbol: str) -> SentimentScore | None:
           """Fetch sentiment for symbol."""
           pass

       @abstractmethod
       def is_healthy(self) -> bool:
           """Check if provider is available."""
           pass

       @abstractmethod
       def get_metadata(self) -> ProviderMetadata:
           """Return provider metadata."""
           pass
   ```

### Phase 2: Rate Limiting & Retry Logic (1-2 hours)

1. **Implement rate limiter** (simple token bucket):
   ```python
   class RateLimiter:
       def __init__(self, requests_per_minute: int):
           self.capacity = requests_per_minute
           self.tokens = requests_per_minute
           self.last_refill = time.time()

       def acquire(self) -> bool:
           """Try to acquire token, return False if rate limit exceeded."""
           self._refill()
           if self.tokens > 0:
               self.tokens -= 1
               return True
           return False

       def _refill(self):
           now = time.time()
           elapsed = now - self.last_refill
           tokens_to_add = int(elapsed * self.capacity / 60)
           if tokens_to_add > 0:
               self.tokens = min(self.capacity, self.tokens + tokens_to_add)
               self.last_refill = now
   ```

2. **Implement exponential backoff decorator**:
   ```python
   def exponential_backoff(max_retries: int = 3, base_delay: float = 1.0):
       def decorator(func):
           def wrapper(*args, **kwargs):
               for attempt in range(max_retries + 1):
                   try:
                       return func(*args, **kwargs)
                   except Exception as e:
                       if attempt == max_retries:
                           raise
                       delay = base_delay * (2 ** attempt)
                       logging.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {delay}s...")
                       time.sleep(delay)
           return wrapper
       return decorator
   ```

### Phase 3: External Provider Implementations (3-4 hours)

1. **NewsAPI Provider** (`news_api.py`):
   ```python
   class NewsAPIProvider(SentimentProvider):
       def __init__(self, api_key: str, endpoint: str, rate_limit: int, timeout: int):
           self.api_key = api_key
           self.endpoint = endpoint
           self.rate_limiter = RateLimiter(rate_limit)
           self.timeout = timeout
           self._circuit_breaker = CircuitBreaker(threshold=5)

       @exponential_backoff(max_retries=3)
       def get_sentiment(self, symbol: str) -> SentimentScore | None:
           if not self.rate_limiter.acquire():
               raise RateLimitExceeded("NewsAPI rate limit exceeded")

           response = requests.get(
               f"{self.endpoint}/everything",
               params={"q": symbol, "apiKey": self.api_key, "sortBy": "publishedAt"},
               timeout=self.timeout
           )
           response.raise_for_status()

           articles = response.json().get("articles", [])
           if not articles:
               return None

           # Normalize sentiment from article titles/descriptions
           sentiment_value = self._analyze_articles(articles)
           return SentimentScore(
               value=sentiment_value,
               confidence=self._compute_confidence(len(articles)),
               source="newsapi",
               timestamp=datetime.now(timezone.utc)
           )

       def _analyze_articles(self, articles: list[dict]) -> float:
           """Placeholder: Use TextBlob or VADER for sentiment analysis."""
           # Simple keyword-based sentiment for MVP
           positive_keywords = ["surge", "profit", "growth", "bullish", "gain"]
           negative_keywords = ["loss", "decline", "bearish", "plunge", "crash"]

           score = 0.0
           for article in articles[:10]:  # Limit to recent 10
               text = f"{article.get('title', '')} {article.get('description', '')}".lower()
               score += sum(1 for kw in positive_keywords if kw in text)
               score -= sum(1 for kw in negative_keywords if kw in text)

           return max(-1.0, min(1.0, score / len(articles) * 0.2))  # Normalize
   ```

2. **Twitter API Provider** (`twitter_api.py`):
   ```python
   class TwitterAPIProvider(SentimentProvider):
       def __init__(self, bearer_token: str, endpoint: str, rate_limit: int, timeout: int):
           self.bearer_token = bearer_token
           self.endpoint = endpoint
           self.rate_limiter = RateLimiter(rate_limit)
           self.timeout = timeout

       @exponential_backoff(max_retries=3)
       def get_sentiment(self, symbol: str) -> SentimentScore | None:
           if not self.rate_limiter.acquire():
               raise RateLimitExceeded("Twitter rate limit exceeded")

           headers = {"Authorization": f"Bearer {self.bearer_token}"}
           response = requests.get(
               f"{self.endpoint}/tweets/search/recent",
               params={"query": f"${symbol} -is:retweet lang:en", "max_results": 100},
               headers=headers,
               timeout=self.timeout
           )
           response.raise_for_status()

           tweets = response.json().get("data", [])
           if not tweets:
               return None

           sentiment_value = self._analyze_tweets(tweets)
           return SentimentScore(
               value=sentiment_value,
               confidence=self._compute_confidence(len(tweets)),
               source="twitter",
               timestamp=datetime.now(timezone.utc)
           )

       def _analyze_tweets(self, tweets: list[dict]) -> float:
           """Analyze tweet sentiment using keyword matching."""
           # Similar to NewsAPI, use simple keyword matching for MVP
           # Production: Use fine-tuned BERT model for financial tweets
           pass
   ```

### Phase 4: Provider Registry (1-2 hours)

1. **Implement registry** (`registry.py`):
   ```python
   class SentimentProviderRegistry:
       def __init__(self):
           self.providers: dict[str, SentimentProvider] = {}
           self.weights: dict[str, float] = {}
           self.fallback_order: list[str] = []
           self._stats: dict[str, ProviderStats] = {}

       def register(self, name: str, provider: SentimentProvider, weight: float = 1.0):
           self.providers[name] = provider
           self.weights[name] = weight
           self._stats[name] = ProviderStats(0, 0, 0, 0.0, 1.0, None)

       def get_sentiment(self, symbol: str) -> SentimentScore | None:
           scores: list[tuple[float, float, str]] = []  # (value, weight, source)

           for provider_name in self.fallback_order:
               if provider_name not in self.providers:
                   continue

               provider = self.providers[provider_name]
               try:
                   start = time.time()
                   score = provider.get_sentiment(symbol)
                   latency = (time.time() - start) * 1000

                   if score:
                       scores.append((score.value, self.weights[provider_name], score.source))
                       self._update_stats(provider_name, success=True, latency=latency)
                       logging.info(f"{provider_name} sentiment: {score.value:.3f} (conf={score.confidence:.2f})")
                   else:
                       self._update_stats(provider_name, success=False, latency=latency)
               except Exception as e:
                   logging.error(f"{provider_name} failed: {e}")
                   self._update_stats(provider_name, success=False, error=True)

           if not scores:
               return None

           # Weighted average
           total_weight = sum(w for _, w, _ in scores)
           weighted_value = sum(v * w for v, w, _ in scores) / total_weight
           sources = [s for _, _, s in scores]

           return SentimentScore(
               value=weighted_value,
               confidence=self._compute_confidence(len(scores)),
               source=f"hybrid[{','.join(sources)}]",
               timestamp=datetime.now(timezone.utc)
           )
   ```

### Phase 5: Enhanced Cache & Settings (1 hour)

1. **Extend Settings** (append to `src/app/config.py`)
2. **Enhance cache** (modify existing `src/adapters/sentiment_provider.py` or create new cache module)

### Phase 6: Engine Integration (1 hour)

1. **Update Engine constructor** to accept optional `sentiment_registry`
2. **Update sentiment hooks** in `tick_intraday()` and `run_swing_daily()`
3. **Add logging** for per-provider contributions

### Phase 7: Testing (2-3 hours)

1. **Unit tests**: Mock external APIs, test rate limiting, backoff, weighted averaging
2. **Integration tests**: Multi-provider sentiment flow, fallback scenarios
3. **Run quality gates**: ruff, mypy, pytest

---

## Acceptance Criteria

### Functional Requirements

- [ ] FR-1: Abstract `SentimentProvider` interface with standardized `SentimentScore` return type
- [ ] FR-2: `NewsAPIProvider` with rate limiting, exponential backoff, error handling
- [ ] FR-3: `TwitterAPIProvider` with rate limiting, exponential backoff, authentication
- [ ] FR-4: `SentimentProviderRegistry` with weighted averaging, fallback logic, circuit breaker
- [ ] FR-5: Extended Settings with NewsAPI/Twitter credentials, weights, fallback order
- [ ] FR-6: Enhanced cache tracking provider-level stats and audit trail
- [ ] FR-7: Engine integration with registry, per-provider logging
- [ ] FR-8: Comprehensive unit and integration tests (90%+ coverage)

### Quality Gates

- [ ] **Code Quality**: `ruff check .` passes (zero errors)
- [ ] **Formatting**: `ruff format .` passes (all files formatted)
- [ ] **Type Safety**: `mypy src/` passes (zero errors)
- [ ] **Tests**: `pytest` passes (100% success rate, 90%+ coverage for new code)
- [ ] **Documentation**: Architecture doc updated with sentiment provider design

### Performance

- [ ] Sentiment fetch completes in < 2 seconds per provider (excluding network latency)
- [ ] Rate limiting prevents API quota exhaustion (respects provider limits)
- [ ] Circuit breaker prevents cascading failures (max 5 retries before disable)
- [ ] Cache reduces redundant API calls by 80%+ for frequently queried symbols

---

## Risks and Mitigations

### Risk 1: API Rate Limits

**Impact**: High
**Probability**: High
**Mitigation**:
- Implement robust rate limiting with token bucket algorithm
- Use exponential backoff to avoid hammering APIs during failures
- Cache aggressively (default TTL: 1 hour for news, 15 minutes for Twitter)
- Circuit breaker to disable misbehaving providers temporarily

### Risk 2: API Cost

**Impact**: Medium
**Probability**: Medium
**Mitigation**:
- Use free tiers during development (NewsAPI: 100 req/day, Twitter: 500k tweets/month)
- Make providers optional (enable_newsapi, enable_twitter flags)
- Implement request batching where supported
- Add cost tracking and alerts in monitoring

### Risk 3: Sentiment Analysis Accuracy

**Impact**: High
**Probability**: Medium
**Mitigation**:
- Use multiple providers for diversified signal (weighted averaging)
- Implement confidence scores (lower confidence for sparse data)
- Start with simple keyword-based sentiment for MVP, plan for BERT/GPT upgrade in v2
- Backtest sentiment impact on strategy performance

### Risk 4: Provider Downtime

**Impact**: Medium
**Probability**: Medium
**Mitigation**:
- Fallback ordering with graceful degradation
- Circuit breaker to avoid wasting time on down providers
- Stub provider as ultimate fallback (returns neutral sentiment)
- Log provider health metrics for monitoring

### Risk 5: Schema Changes

**Impact**: Low
**Probability**: Medium
**Mitigation**:
- Version provider implementations (NewsAPIProvider v1.0)
- Comprehensive JSON schema validation in tests
- Fail gracefully on unexpected response formats
- Monitor provider API changelog

---

## Future Enhancements (Post-MVP)

1. **Additional Providers**:
   - Reddit API (r/wallstreetbets sentiment)
   - Google Trends (search volume as sentiment proxy)
   - Financial news aggregators (Bloomberg, Reuters)

2. **Advanced Sentiment Analysis**:
   - Fine-tuned BERT model for financial text
   - Entity recognition (company names, products)
   - Event detection (earnings, mergers, scandals)

3. **Real-Time Sentiment**:
   - WebSocket streams for Twitter/news
   - Sub-second sentiment updates for HFT strategies

4. **Sentiment Momentum**:
   - Track sentiment velocity (rate of change)
   - Detect sentiment spikes as trading signals

5. **Portfolio-Level Sentiment**:
   - Aggregate sentiment across all held positions
   - Sector sentiment analysis

---

## References

- [NewsAPI Documentation](https://newsapi.org/docs)
- [Twitter API v2 Documentation](https://developer.twitter.com/en/docs/twitter-api)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Token Bucket Algorithm](https://en.wikipedia.org/wiki/Token_bucket)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
