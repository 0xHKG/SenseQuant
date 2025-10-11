# US-001 — Breeze Adapter Robustness

## Goal
Harden the Breeze API adapter with comprehensive error handling, retries, timeouts, normalization, and deterministic dry-run behavior.

## Context (PRD/Arch)
- Must support dry-run and live modes with no network calls in dry-run
- Implement retry logic with exponential backoff for transient failures
- Normalize all API responses to domain types (`Bar`, `OrderResponse`)
- Never log secrets; use structured logging with component tagging
- All public methods must be fully typed with docstring examples

## Scope Files
- `src/adapters/breeze_client.py`
- `tests/unit/test_breeze_client.py`

---

## Tasks

### 1. Error Taxonomy

**File: `src/adapters/breeze_client.py`**

Define exception hierarchy for classifying Breeze API errors:

```python
class BreezeError(Exception):
    """Base exception for all Breeze API errors."""
    def __init__(self, message: str, status_code: int | None = None, raw_response: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.raw_response = raw_response

    def is_transient(self) -> bool:
        """Override in subclasses to indicate if error is retryable."""
        return False


class BreezeAuthError(BreezeError):
    """Authentication failure (401, invalid credentials)."""
    pass


class BreezeRateLimitError(BreezeError):
    """Rate limit exceeded (HTTP 429)."""
    def is_transient(self) -> bool:
        return True

    def get_retry_after(self) -> int | None:
        """Extract 'Retry-After' header value in seconds if present."""
        if self.raw_response and "retry_after" in self.raw_response:
            try:
                return int(self.raw_response["retry_after"])
            except (ValueError, TypeError):
                pass
        return None


class BreezeTransientError(BreezeError):
    """Transient errors: network issues, 5xx server errors, timeouts."""
    def is_transient(self) -> bool:
        return True


class BreezeOrderRejectedError(BreezeError):
    """Order rejected by exchange (insufficient funds, invalid symbol, etc.)."""
    pass


def is_transient(e: Exception) -> bool:
    """
    Helper to classify if an exception is retryable.

    Args:
        e: Exception instance

    Returns:
        True if error is transient and should be retried

    Examples:
        >>> is_transient(BreezeTransientError("timeout"))
        True
        >>> is_transient(BreezeAuthError("invalid key"))
        False
    """
    if isinstance(e, BreezeError):
        return e.is_transient()
    if isinstance(e, (ConnectionError, TimeoutError)):
        return True
    return False
```

**Tasks:**
- [ ] Define `BreezeError` base class with `is_transient()` method
- [ ] Define subclasses: `BreezeAuthError`, `BreezeRateLimitError`, `BreezeTransientError`, `BreezeOrderRejectedError`
- [ ] Implement `get_retry_after()` in `BreezeRateLimitError`
- [ ] Add module-level `is_transient(e: Exception) -> bool` helper
- [ ] Add docstrings with examples for all exception classes

---

### 2. Retries & Timeouts (Tenacity)

**File: `src/adapters/breeze_client.py`**

Apply retry decorators to all network-bound methods:

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
from loguru import logger


class BreezeClient:
    DEFAULT_TIMEOUT = 30  # seconds

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(is_transient),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _call_with_retry(self, method: str, **kwargs) -> dict:
        """Internal wrapper for Breeze SDK calls with timeout and retry."""
        # Implementation details in full story
        ...
```

**Tasks:**
- [ ] Add `DEFAULT_TIMEOUT = 30` class constant
- [ ] Implement `_call_with_retry()` internal method with tenacity decorator
- [ ] Configure exponential backoff: multiplier=1, min=2s, max=30s, stop after 3 attempts
- [ ] Use `retry_if_exception(is_transient)` to only retry transient errors
- [ ] Implement `_handle_rate_limit()` to respect `Retry-After` header
- [ ] Add `before_sleep_log` to log retry attempts at WARNING level
- [ ] Parse Breeze SDK response status codes (401, 429, 5xx) and raise appropriate exceptions

---

### 3. Public Methods with Final Signatures

**File: `src/adapters/breeze_client.py`**

#### Method: `authenticate() -> None`
- Establish Breeze session
- Skip in dry-run mode
- Raise `BreezeAuthError` on 401

#### Method: `latest_price(symbol: str) -> float`
- Get last traded price
- Return 0.0 in dry-run or on error
- Retry transient errors

#### Method: `historical_bars(symbol: str, interval: Literal["1minute","5minute","1day"], start: pd.Timestamp, end: pd.Timestamp) -> list[Bar]`
- Fetch OHLCV bars
- Return empty list in dry-run
- Normalize to `Bar` DTOs with IST timezone

#### Method: `place_order(symbol: str, side: OrderSide, qty: int, order_type: OrderType = "MARKET", price: float | None = None) -> OrderResponse`
- Place order via Breeze API
- Return mock response in dry-run
- Retry transient errors once

**Tasks:**
- [ ] Implement `authenticate()` with dry-run short-circuit
- [ ] Implement `latest_price()` with safe fallback to 0.0
- [ ] Implement `historical_bars()` with `_normalize_bars()` call
- [ ] Implement `place_order()` with single retry logic for transient errors
- [ ] Implement `_place_order_impl()` internal helper
- [ ] Add comprehensive docstrings with args, returns, raises, examples for all methods

---

### 4. Bar Normalization

**File: `src/adapters/breeze_client.py`**

```python
def _normalize_bars(self, payload: dict | list, symbol: str, interval: str) -> list[Bar]:
    """
    Normalize Breeze API response to list of Bar DTOs.

    Ensures:
    - Timestamps are IST timezone-aware (pytz.timezone("Asia/Kolkata"))
    - Numeric fields (open, high, low, close, volume) are correct types
    - Empty/malformed data returns empty list (no crash)
    """
    # Implementation details...
```

**Tasks:**
- [ ] Implement `_normalize_bars()` with IST timezone localization
- [ ] Handle both dict and list payload formats
- [ ] Safe numeric conversion with try/except for each bar
- [ ] Return empty list on malformed data (no crash)
- [ ] Sort bars by timestamp ascending
- [ ] Add debug logging for normalization stats

---

### 5. Deterministic DRYRUN

**File: `src/adapters/breeze_client.py`**

```python
def _mock_order_response(
    self, symbol: str, side: OrderSide, qty: int, order_type: OrderType, price: float | None
) -> OrderResponse:
    """
    Generate deterministic mock order response for dry-run mode.

    Order ID is stable hash: "DRYRUN-" + hash(symbol, side, qty, timestamp_floor_minute)
    """
    # Floor timestamp to minute for stability
    now = datetime.now()
    ts_floor = now.replace(second=0, microsecond=0)

    # Create stable hash
    hash_input = f"{symbol}|{side}|{qty}|{ts_floor.isoformat()}"
    hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    order_id = f"DRYRUN-{hash_digest.upper()}"
    # ...
```

**Tasks:**
- [ ] Implement `_mock_order_response()` with deterministic hash
- [ ] Use SHA256 hash of `symbol|side|qty|timestamp_floor_minute`
- [ ] Floor timestamp to minute for stability
- [ ] Return `OrderResponse` with `status="FILLED"`
- [ ] Log dry-run order with all parameters (no secrets)

---

### 6. Structured Logging

**File: `src/adapters/breeze_client.py`**

Ensure all log statements follow these rules:

```python
# Good: component tagging + no secrets
logger.info("Breeze session established", extra={"component": "breeze"})

# Good: context with symbol info
logger.error(
    "Failed to fetch historical bars",
    extra={"component": "breeze", "symbol": symbol, "interval": interval}
)

# Bad: exposes API key
logger.debug(f"Authenticating with key {self.api_key}")  # NEVER DO THIS
```

**Tasks:**
- [ ] Add `extra={"component": "breeze"}` to all logger calls
- [ ] Never log `api_key`, `api_secret`, `session_token` in plaintext
- [ ] Redact secrets from exception messages if present
- [ ] Use structured fields: `symbol`, `interval`, `side`, `qty`, `order_id`
- [ ] Add audit trail for all orders: log before/after API call

---

### 7. Unit Tests

**File: `tests/unit/test_breeze_client.py`**

#### Required Test Cases (15+)

1. **test_authenticate_success**: Mock successful auth
2. **test_authenticate_invalid_creds**: 401 → `BreezeAuthError`
3. **test_latest_price_success**: Mock LTP response
4. **test_latest_price_transient_retry**: Timeout → retry → success (verify 2 calls)
5. **test_historical_bars_normalization**: Verify IST timezone, correct types
6. **test_historical_bars_empty_safe**: Empty response → empty list
7. **test_place_order_success**: Mock order response
8. **test_place_order_rejected**: 400 → `BreezeOrderRejectedError`
9. **test_place_order_transient_retry**: 500 → retry → success (verify 2 calls)
10. **test_rate_limit_retry**: 429 → retry → success
11. **test_rate_limit_respect_retry_after**: Verify `time.sleep()` with correct value
12. **test_dryrun_no_sdk_calls**: All methods skip SDK
13. **test_dryrun_deterministic_order_id**: Same inputs → same order_id
14. **test_is_transient_helper**: Test all exception types
15. **test_no_secrets_in_logs**: Use `caplog` to verify

**Tasks:**
- [ ] Create fixtures: `mock_breeze_sdk`, `client_live`, `client_dryrun`
- [ ] Test authenticate: success + 401 AuthError
- [ ] Test latest_price: success + transient retry
- [ ] Test historical_bars: normalization + IST timezone + empty result
- [ ] Test place_order: success + rejected + transient retry
- [ ] Test rate limit: retry + respect retry-after header
- [ ] Test DRYRUN: no SDK calls + deterministic order_id
- [ ] Test is_transient: all exception types
- [ ] Test no secrets in logs (use caplog fixture)
- [ ] Achieve >80% coverage on `src/adapters/breeze_client.py`

---

## Acceptance Criteria

### AC1: Error Taxonomy
- [ ] **GIVEN** Breeze API returns various error responses
- [ ] **WHEN** errors are classified
- [ ] **THEN** correct exception type is raised
- [ ] **AND** `is_transient()` correctly identifies retryable errors
- [ ] **AND** `BreezeRateLimitError.get_retry_after()` extracts header value

### AC2: Retries & Timeouts
- [ ] **GIVEN** Breeze API call encounters transient error
- [ ] **WHEN** method is invoked
- [ ] **THEN** retry is attempted with exponential backoff (2s, 4s, 8s...)
- [ ] **AND** max 3 attempts before raising exception
- [ ] **AND** non-transient errors are NOT retried
- [ ] **AND** per-call timeout is 30 seconds

### AC3: Rate Limit Handling
- [ ] **GIVEN** Breeze API returns HTTP 429
- [ ] **WHEN** retry is triggered
- [ ] **THEN** `Retry-After` header is respected if present
- [ ] **AND** default backoff is used if header absent
- [ ] **AND** warning is logged with component="breeze"

### AC4: Method Signatures
- [ ] **GIVEN** all public methods
- [ ] **WHEN** inspecting type hints
- [ ] **THEN** all parameters and return types are fully typed
- [ ] **AND** no `Any` leakage in public API
- [ ] **AND** docstrings include Args, Returns, Raises, Examples

### AC5: Bar Normalization
- [ ] **GIVEN** raw Breeze API response with OHLCV data
- [ ] **WHEN** `_normalize_bars()` is called
- [ ] **THEN** returns list of `Bar` objects
- [ ] **AND** timestamps are IST timezone-aware (`pytz.timezone("Asia/Kolkata")`)
- [ ] **AND** numeric fields (open, high, low, close, volume) have correct types
- [ ] **AND** malformed data returns empty list without crash

### AC6: Deterministic DRYRUN
- [ ] **GIVEN** client in dry-run mode
- [ ] **WHEN** `place_order()` is called with same inputs within same minute
- [ ] **THEN** order_id is identical (deterministic hash)
- [ ] **AND** order_id format is `"DRYRUN-{8_char_hex}"`
- [ ] **AND** status is `"FILLED"`
- [ ] **AND** no Breeze SDK calls are made

### AC7: Structured Logging
- [ ] **GIVEN** any Breeze client method is invoked
- [ ] **WHEN** logs are emitted
- [ ] **THEN** all logs include `extra={"component": "breeze"}`
- [ ] **AND** no secrets (api_key, api_secret, session_token) are logged
- [ ] **AND** structured fields (symbol, side, qty, order_id) are present
- [ ] **AND** errors are logged with exception type, not full message if sensitive

### AC8: Unit Tests
- [ ] **GIVEN** test suite in `tests/unit/test_breeze_client.py`
- [ ] **WHEN** `pytest` is run
- [ ] **THEN** all 15+ tests pass
- [ ] **AND** coverage on `src/adapters/breeze_client.py` is ≥80%
- [ ] **AND** no network calls are made (all SDK methods mocked)
- [ ] **AND** tests verify: success paths, failure paths, retries, dry-run, logging

### AC9: Quality Gates
- [ ] **GIVEN** implementation is complete
- [ ] **WHEN** running quality checks
- [ ] **THEN** `ruff check .` passes (exit 0)
- [ ] **AND** `mypy src` passes (exit 0)
- [ ] **AND** `pytest -q` passes (exit 0)
- [ ] **AND** no type errors, lint errors, or test failures

---

## Definition of Done

- [ ] All tasks checked off above
- [ ] Error taxonomy implemented with 5 exception classes
- [ ] Tenacity retry decorators applied to all network calls
- [ ] All 4 public methods implemented with full type hints and docstrings
- [ ] Bar normalization handles IST timezone and malformed data
- [ ] Deterministic dry-run order IDs implemented
- [ ] Structured logging with component tagging, no secrets
- [ ] 15+ unit tests pass with ≥80% coverage
- [ ] `ruff check .` passes
- [ ] `mypy src` passes
- [ ] `pytest` passes
- [ ] PR reviewed and approved

---

## Estimates

- Error taxonomy: 1h
- Retries & timeouts: 2h
- Public methods implementation: 3h
- Bar normalization: 1h
- Deterministic dry-run: 1h
- Structured logging audit: 1h
- Unit tests (15+ tests): 4h
- Fix lint/type/test issues: 1h
- **Total: ~14h (2 days)**

---

## Dependencies

- US-000 (folder structure, domain types) must be complete

---

## Risks

- **Risk**: Breeze SDK response format may differ from documentation
  - **Mitigation**: Add defensive parsing with fallbacks; log unexpected formats
- **Risk**: Retry logic may cause delays in fast-moving markets
  - **Mitigation**: Cap max retry time at 30s; fail fast on non-transient errors
- **Risk**: Timezone handling may have edge cases (DST transitions)
  - **Mitigation**: Always use `pytz.timezone("Asia/Kolkata")`; add tests for edge dates
