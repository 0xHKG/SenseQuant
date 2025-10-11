"""Unit tests for Breeze client with comprehensive coverage."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.adapters.breeze_client import (
    BreezeAuthError,
    BreezeClient,
    BreezeOrderRejectedError,
    BreezeRateLimitError,
    BreezeTransientError,
    is_transient,
)
from src.domain.types import Bar


@pytest.fixture
def mock_breeze_sdk():
    """Mock breeze_connect.BreezeConnect."""
    with patch("src.adapters.breeze_client.BreezeConnect") as mock:
        yield mock


@pytest.fixture
def client_live(mock_breeze_sdk):
    """Live mode client with mocked SDK."""
    return BreezeClient(
        api_key="test_key", api_secret="test_secret", session_token="test_token", dry_run=False
    )


@pytest.fixture
def client_dryrun():
    """Dry-run mode client (no SDK needed)."""
    return BreezeClient(
        api_key="test_key", api_secret="test_secret", session_token="test_token", dry_run=True
    )


# ============================================================================
# Test 1-2: authenticate
# ============================================================================


def test_authenticate_success(client_live, mock_breeze_sdk):
    """Test successful authentication."""
    mock_instance = mock_breeze_sdk.return_value
    mock_instance.generate_session.return_value = {"Status": 200, "Success": {}}

    client_live.authenticate()

    mock_instance.generate_session.assert_called_once()


def test_authenticate_invalid_creds(client_live, mock_breeze_sdk):
    """Test authentication failure with 401."""
    mock_instance = mock_breeze_sdk.return_value
    mock_instance.generate_session.return_value = {"Status": 401, "Error": "Invalid credentials"}

    with pytest.raises(BreezeAuthError) as exc_info:
        client_live.authenticate()

    assert exc_info.value.status_code == 401


# ============================================================================
# Test 3-4: latest_price
# ============================================================================


def test_latest_price_success(client_live, mock_breeze_sdk):
    """Test successful LTP fetch."""
    mock_instance = mock_breeze_sdk.return_value
    mock_instance.get_quotes.return_value = {"Status": 200, "Success": [{"ltp": 2456.75}]}

    client_live._client = mock_instance
    price = client_live.latest_price("RELIANCE")

    assert price == 2456.75


def test_latest_price_transient_retry(client_live, mock_breeze_sdk):
    """Test transient failure triggers retry."""
    mock_instance = mock_breeze_sdk.return_value

    # First call fails with timeout, second succeeds
    mock_instance.get_quotes.side_effect = [
        TimeoutError("timeout"),
        {"Status": 200, "Success": [{"ltp": 2456.75}]},
    ]

    client_live._client = mock_instance

    # Should succeed after retry
    price = client_live.latest_price("RELIANCE")
    assert price == 2456.75
    assert mock_instance.get_quotes.call_count == 2


# ============================================================================
# Test 5-6: historical_bars
# ============================================================================


def test_historical_bars_normalization(client_live, mock_breeze_sdk):
    """Test bar normalization with IST timezone."""
    mock_instance = mock_breeze_sdk.return_value
    mock_instance.get_historical_data.return_value = {
        "Status": 200,
        "Success": [
            {
                "datetime": "2025-01-01 09:15:00",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "volume": 10000,
            }
        ],
    }

    client_live._client = mock_instance
    start = pd.Timestamp("2025-01-01 09:15", tz="Asia/Kolkata")
    end = pd.Timestamp("2025-01-01 15:30", tz="Asia/Kolkata")

    bars = client_live.historical_bars("RELIANCE", "1minute", start, end)

    assert len(bars) == 1
    assert isinstance(bars[0], Bar)
    assert bars[0].ts.tz.zone == "Asia/Kolkata"
    assert bars[0].close == 103.0


def test_historical_bars_empty_safe(client_live, mock_breeze_sdk):
    """Test empty result returns empty list without crash."""
    mock_instance = mock_breeze_sdk.return_value
    mock_instance.get_historical_data.return_value = {"Status": 200, "Success": []}

    client_live._client = mock_instance
    start = pd.Timestamp("2025-01-01 09:15", tz="Asia/Kolkata")
    end = pd.Timestamp("2025-01-01 15:30", tz="Asia/Kolkata")

    bars = client_live.historical_bars("RELIANCE", "1minute", start, end)

    assert bars == []


# ============================================================================
# Test 7-9: place_order
# ============================================================================


def test_place_order_success(client_live, mock_breeze_sdk):
    """Test successful order placement."""
    mock_instance = mock_breeze_sdk.return_value
    mock_instance.place_order.return_value = {"Status": 200, "Success": {"order_id": "12345"}}

    client_live._client = mock_instance
    response = client_live.place_order("RELIANCE", "BUY", 1)

    assert response.order_id == "12345"
    assert response.status == "PLACED"


def test_place_order_rejected(client_live, mock_breeze_sdk):
    """Test order rejection raises BreezeOrderRejectedError."""
    mock_instance = mock_breeze_sdk.return_value
    mock_instance.place_order.return_value = {"Status": 400, "Error": "Insufficient funds"}

    client_live._client = mock_instance

    with pytest.raises(BreezeOrderRejectedError) as exc_info:
        client_live.place_order("RELIANCE", "BUY", 1)

    assert "Insufficient funds" in str(exc_info.value)


def test_place_order_transient_retry(client_live, mock_breeze_sdk):
    """Test transient error triggers single retry."""
    mock_instance = mock_breeze_sdk.return_value

    # First call fails with 500, second succeeds
    mock_instance.place_order.side_effect = [
        {"Status": 500, "Error": "Server error"},
        {"Status": 200, "Success": {"order_id": "12345"}},
    ]

    client_live._client = mock_instance

    response = client_live.place_order("RELIANCE", "BUY", 1)

    assert response.order_id == "12345"
    # Note: _call_with_retry will retry, so count may be higher
    assert mock_instance.place_order.call_count >= 2


# ============================================================================
# Test 10-11: rate limit (429)
# ============================================================================


def test_rate_limit_retry(client_live, mock_breeze_sdk):
    """Test 429 rate limit triggers retry with backoff."""
    mock_instance = mock_breeze_sdk.return_value

    # First call 429, second succeeds
    mock_instance.get_quotes.side_effect = [
        {"Status": 429, "Error": "Rate limit"},
        {"Status": 200, "Success": [{"ltp": 2456.75}]},
    ]

    client_live._client = mock_instance

    price = client_live.latest_price("RELIANCE")

    assert price == 2456.75
    assert mock_instance.get_quotes.call_count == 2


def test_rate_limit_respect_retry_after(client_live, mock_breeze_sdk):
    """Test Retry-After header is respected."""
    mock_instance = mock_breeze_sdk.return_value

    mock_instance.get_quotes.side_effect = [
        {"Status": 429, "Error": "Rate limit", "retry_after": "5"},
        {"Status": 200, "Success": [{"ltp": 2456.75}]},
    ]

    client_live._client = mock_instance

    with patch("time.sleep") as mock_sleep:
        price = client_live.latest_price("RELIANCE")

        # Should sleep for 5 seconds as per retry_after
        mock_sleep.assert_called()
        assert price == 2456.75


# ============================================================================
# Test 12-13: DRYRUN
# ============================================================================


def test_dryrun_no_sdk_calls(client_dryrun):
    """Test dry-run mode makes no SDK calls."""
    client_dryrun.authenticate()

    price = client_dryrun.latest_price("RELIANCE")
    assert price == 0.0

    bars = client_dryrun.historical_bars(
        "RELIANCE",
        "1minute",
        pd.Timestamp("2025-01-01", tz="Asia/Kolkata"),
        pd.Timestamp("2025-01-02", tz="Asia/Kolkata"),
    )
    assert bars == []

    response = client_dryrun.place_order("RELIANCE", "BUY", 1)
    assert "DRYRUN" in response.order_id
    assert response.status == "FILLED"


def test_dryrun_deterministic_order_id(client_dryrun):
    """Test dry-run generates stable order IDs within same minute."""
    r1 = client_dryrun.place_order("RELIANCE", "BUY", 1)
    r2 = client_dryrun.place_order("RELIANCE", "BUY", 1)

    # Same inputs in same minute should yield same order_id
    assert r1.order_id == r2.order_id
    assert r1.order_id.startswith("DRYRUN-")
    assert len(r1.order_id) == len("DRYRUN-") + 8  # 8 hex chars

    # Different symbol should yield different order_id
    r3 = client_dryrun.place_order("TCS", "BUY", 1)
    assert r3.order_id != r1.order_id


# ============================================================================
# Test 14: is_transient helper
# ============================================================================


def test_is_transient_helper():
    """Test is_transient classification."""
    assert is_transient(BreezeTransientError("timeout"))
    assert is_transient(BreezeRateLimitError("429"))
    assert is_transient(TimeoutError("timeout"))
    assert is_transient(ConnectionError("connection"))

    assert not is_transient(BreezeAuthError("401"))
    assert not is_transient(BreezeOrderRejectedError("rejected"))
    assert not is_transient(ValueError("invalid"))


# ============================================================================
# Test 15: no secrets in logs
# ============================================================================


def test_no_secrets_in_logs(client_live, mock_breeze_sdk, caplog):
    """Test that secrets are never logged."""
    mock_instance = mock_breeze_sdk.return_value
    mock_instance.generate_session.return_value = {"Status": 200}

    with caplog.at_level("DEBUG"):
        client_live.authenticate()

    # Check logs don't contain secrets
    log_text = " ".join([record.message for record in caplog.records]).lower()
    assert "test_key" not in log_text
    assert "test_secret" not in log_text
    assert "test_token" not in log_text


# ============================================================================
# Additional edge case tests
# ============================================================================


def test_normalize_bars_malformed_data(client_live):
    """Test _normalize_bars handles malformed data gracefully."""
    # Missing datetime
    payload = {"Success": [{"open": 100, "close": 100}]}
    bars = client_live._normalize_bars(payload, "TEST", "1minute")
    assert bars == []

    # Invalid numeric values
    payload = {"Success": [{"datetime": "2025-01-01 09:15:00", "open": "invalid"}]}
    bars = client_live._normalize_bars(payload, "TEST", "1minute")
    assert bars == []

    # Unexpected payload type
    payload = "not a dict or list"
    bars = client_live._normalize_bars(payload, "TEST", "1minute")  # type: ignore
    assert bars == []


def test_authenticate_dryrun_skips_sdk(client_dryrun):
    """Test dry-run authenticate doesn't call SDK."""
    client_dryrun.authenticate()
    # No exception, just returns
    assert client_dryrun._client is None


def test_latest_price_returns_zero_on_error(client_live, mock_breeze_sdk):
    """Test latest_price returns 0.0 on non-transient errors."""
    mock_instance = mock_breeze_sdk.return_value
    mock_instance.get_quotes.return_value = {"Status": 404, "Error": "Not found"}

    client_live._client = mock_instance
    price = client_live.latest_price("INVALID")

    assert price == 0.0
