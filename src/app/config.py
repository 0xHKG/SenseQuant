"""Application configuration using Pydantic v2."""

from __future__ import annotations

import json
from typing import Any, Literal

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from repo root robustly
# Use override=True to ensure .env values take precedence over stale shell environment
load_dotenv(find_dotenv(), override=True)


class Settings(BaseSettings):  # type: ignore[misc]
    """Application settings loaded from environment variables."""

    # Breeze creds with explicit env aliases
    breeze_api_key: str = Field("", validation_alias="BREEZE_API_KEY")
    breeze_api_secret: str = Field("", validation_alias="BREEZE_API_SECRET")
    breeze_session_token: str = Field("", validation_alias="BREEZE_SESSION_TOKEN")

    # Trading
    symbols: Any = Field(default=["RELIANCE"], validation_alias="SYMBOLS")
    mode: Literal["dryrun", "live", "backtest"] = Field("dryrun", validation_alias="MODE")

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v: Any) -> list[str]:
        """Parse symbols from various input formats."""
        # Return default on None or empty string
        if v is None or v == "":
            return ["RELIANCE"]

        # If string, parse as JSON or CSV
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return ["RELIANCE"]
            # Try JSON parsing if starts with '['
            if v.startswith("["):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except (json.JSONDecodeError, ValueError):
                    pass
            # Parse as CSV
            return [s.strip() for s in v.split(",") if s.strip()]

        # If already a sequence, normalize to list[str]
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            return [str(item) for item in v]

        # Fallback: convert to string and return as single-item list
        return [str(v)]

    # Risk
    max_position_value: float = Field(50_000.0, validation_alias="MAX_POSITION_VALUE")
    per_trade_risk_pct: float = Field(0.01, validation_alias="PER_TRADE_RISK_PCT")

    # Intraday Strategy
    intraday_bar_interval: Literal["1minute"] = Field(
        "1minute", validation_alias="INTRADAY_BAR_INTERVAL"
    )
    intraday_feature_lookback_minutes: int = Field(
        60, validation_alias="INTRADAY_FEATURE_LOOKBACK_MINUTES"
    )
    intraday_tick_seconds: int = Field(5, validation_alias="INTRADAY_TICK_SECONDS")
    intraday_sma_period: int = Field(20, validation_alias="INTRADAY_SMA_PERIOD")
    intraday_ema_period: int = Field(50, validation_alias="INTRADAY_EMA_PERIOD")
    intraday_rsi_period: int = Field(14, validation_alias="INTRADAY_RSI_PERIOD")
    intraday_atr_period: int = Field(14, validation_alias="INTRADAY_ATR_PERIOD")
    intraday_long_rsi_min: int = Field(55, validation_alias="INTRADAY_LONG_RSI_MIN")
    intraday_short_rsi_max: int = Field(45, validation_alias="INTRADAY_SHORT_RSI_MAX")

    # Sentiment Analysis (Legacy)
    sentiment_pos_limit: float = Field(0.15, validation_alias="SENTIMENT_POS_LIMIT")
    sentiment_neg_limit: float = Field(-0.15, validation_alias="SENTIMENT_NEG_LIMIT")

    # Sentiment Provider Configuration
    sentiment_enable_newsapi: bool = Field(False, validation_alias="SENTIMENT_ENABLE_NEWSAPI")
    sentiment_newsapi_api_key: str = Field("", validation_alias="SENTIMENT_NEWSAPI_API_KEY")
    sentiment_newsapi_endpoint: str = Field(
        "https://newsapi.org/v2", validation_alias="SENTIMENT_NEWSAPI_ENDPOINT"
    )
    sentiment_newsapi_rate_limit: int = Field(
        100, validation_alias="SENTIMENT_NEWSAPI_RATE_LIMIT", ge=1, le=1000
    )

    sentiment_enable_twitter: bool = Field(False, validation_alias="SENTIMENT_ENABLE_TWITTER")
    sentiment_twitter_bearer_token: str = Field(
        "", validation_alias="SENTIMENT_TWITTER_BEARER_TOKEN"
    )
    sentiment_twitter_endpoint: str = Field(
        "https://api.twitter.com/2", validation_alias="SENTIMENT_TWITTER_ENDPOINT"
    )
    sentiment_twitter_rate_limit: int = Field(
        450, validation_alias="SENTIMENT_TWITTER_RATE_LIMIT", ge=1, le=1000
    )

    sentiment_provider_timeout: int = Field(
        5, validation_alias="SENTIMENT_PROVIDER_TIMEOUT", ge=1, le=30
    )
    sentiment_circuit_breaker_threshold: int = Field(
        5, validation_alias="SENTIMENT_CIRCUIT_BREAKER_THRESHOLD", ge=1, le=20
    )
    sentiment_circuit_breaker_cooldown: int = Field(
        30, validation_alias="SENTIMENT_CIRCUIT_BREAKER_COOLDOWN", ge=1, le=120
    )

    # Provider weights and fallback order (JSON strings)
    sentiment_provider_weights: str = Field(
        '{"newsapi": 0.6, "twitter": 0.4}', validation_alias="SENTIMENT_PROVIDER_WEIGHTS"
    )
    sentiment_provider_fallback_order: str = Field(
        '["newsapi", "twitter"]', validation_alias="SENTIMENT_PROVIDER_FALLBACK_ORDER"
    )

    # Swing Strategy
    swing_bar_interval: Literal["1day"] = Field("1day", validation_alias="SWING_BAR_INTERVAL")
    swing_sma_fast: int = Field(20, validation_alias="SWING_SMA_FAST", ge=5, le=100)
    swing_sma_slow: int = Field(50, validation_alias="SWING_SMA_SLOW", ge=20, le=200)
    swing_rsi_period: int = Field(14, validation_alias="SWING_RSI_PERIOD", ge=7, le=30)
    swing_sl_pct: float = Field(0.03, validation_alias="SWING_SL_PCT", ge=0.01, le=0.10)
    swing_tp_pct: float = Field(0.06, validation_alias="SWING_TP_PCT", ge=0.02, le=0.20)
    swing_max_hold_days: int = Field(15, validation_alias="SWING_MAX_HOLD_DAYS", ge=2, le=30)
    swing_feature_lookback_days: int = Field(
        120, validation_alias="SWING_FEATURE_LOOKBACK_DAYS", ge=60, le=365
    )

    # Sentiment Analysis
    sentiment_cache_ttl: int = Field(
        3600, validation_alias="SENTIMENT_CACHE_TTL", ge=60, le=86400
    )  # 1 hour default, min 1 min, max 24 hours
    sentiment_rate_limit_per_min: int = Field(
        10, validation_alias="SENTIMENT_RATE_LIMIT_PER_MIN", ge=1, le=100
    )
    sentiment_gate_threshold: float = Field(
        -0.3, validation_alias="SENTIMENT_GATE_THRESHOLD", ge=-1.0, le=0.0
    )  # Suppress BUY if sentiment < this
    sentiment_boost_threshold: float = Field(
        0.5, validation_alias="SENTIMENT_BOOST_THRESHOLD", ge=0.0, le=1.0
    )  # Boost confidence if sentiment > this
    sentiment_boost_multiplier: float = Field(
        1.2, validation_alias="SENTIMENT_BOOST_MULTIPLIER", ge=1.0, le=2.0
    )  # Multiply confidence by this factor

    # Risk Management & Position Sizing
    starting_capital: float = Field(
        1000000.0, validation_alias="STARTING_CAPITAL", ge=10000.0, le=100000000.0
    )  # Starting capital in INR
    position_sizing_mode: str = Field(
        "FIXED_FRACTIONAL", validation_alias="POSITION_SIZING_MODE"
    )  # FIXED_FRACTIONAL or ATR_BASED
    risk_per_trade_pct: float = Field(
        1.0, validation_alias="RISK_PER_TRADE_PCT", ge=0.1, le=5.0
    )  # Risk per trade as % of capital
    atr_multiplier: float = Field(
        2.0, validation_alias="ATR_MULTIPLIER", ge=1.0, le=5.0
    )  # ATR multiplier for stop-loss
    max_position_value_per_symbol: float = Field(
        100000.0, validation_alias="MAX_POSITION_VALUE_PER_SYMBOL", ge=1000.0, le=10000000.0
    )  # Max position value per symbol in INR
    max_daily_loss_pct: float = Field(
        5.0, validation_alias="MAX_DAILY_LOSS_PCT", ge=1.0, le=20.0
    )  # Max daily loss before circuit breaker
    trading_fee_bps: float = Field(
        10.0, validation_alias="TRADING_FEE_BPS", ge=0.0, le=100.0
    )  # Trading fees in basis points
    slippage_bps: float = Field(
        5.0, validation_alias="SLIPPAGE_BPS", ge=0.0, le=50.0
    )  # Slippage in basis points

    # Student Inference (Deprecated - use student_model_enabled below)
    # Kept for backward compatibility with existing code
    enable_student_inference: bool = Field(
        False, validation_alias="ENABLE_STUDENT_INFERENCE"
    )  # Enable Student model predictions (deprecated)
    student_metadata_path: str = Field(
        "", validation_alias="STUDENT_METADATA_PATH"
    )  # Path to Student metadata file

    # Monitoring & Alerts
    enable_monitoring: bool = Field(
        True, validation_alias="ENABLE_MONITORING"
    )  # Enable monitoring and alerts
    monitoring_heartbeat_interval: int = Field(
        60, validation_alias="MONITORING_HEARTBEAT_INTERVAL", ge=10, le=3600
    )  # Heartbeat interval in seconds
    monitoring_alert_recipients: list[str] = Field(
        default_factory=list, validation_alias="MONITORING_ALERT_RECIPIENTS"
    )  # Alert recipients (email/telegram handles)
    monitoring_max_sentiment_failures: int = Field(
        5, validation_alias="MONITORING_MAX_SENTIMENT_FAILURES", ge=1, le=100
    )  # Max sentiment failures per window before alert
    monitoring_sentiment_failure_window: int = Field(
        3600, validation_alias="MONITORING_SENTIMENT_FAILURE_WINDOW", ge=60, le=86400
    )  # Sentiment failure window in seconds
    monitoring_artifact_staleness_hours: int = Field(
        24, validation_alias="MONITORING_ARTIFACT_STALENESS_HOURS", ge=1, le=168
    )  # Hours before artifacts considered stale
    monitoring_heartbeat_lapse_seconds: int = Field(
        300, validation_alias="MONITORING_HEARTBEAT_LAPSE_SECONDS", ge=30, le=3600
    )  # Seconds before heartbeat considered lapsed
    monitoring_daily_loss_alert_pct: float = Field(
        4.0, validation_alias="MONITORING_DAILY_LOSS_ALERT_PCT", ge=0.5, le=10.0
    )  # Daily loss % threshold for alert

    # Monitoring v2 — Aggregation
    monitoring_aggregation_interval_seconds: int = Field(
        300, validation_alias="MONITORING_AGGREGATION_INTERVAL_SECONDS", ge=60, le=3600
    )  # Interval for metric rollups (5 min default)
    monitoring_enable_aggregation: bool = Field(
        True, validation_alias="MONITORING_ENABLE_AGGREGATION"
    )  # Enable metric aggregation

    # Monitoring v2 — Retention
    monitoring_max_raw_metrics: int = Field(
        100, validation_alias="MONITORING_MAX_RAW_METRICS", ge=50, le=1000
    )  # Max raw metrics to keep in memory
    monitoring_max_archive_days: int = Field(
        30, validation_alias="MONITORING_MAX_ARCHIVE_DAYS", ge=1, le=365
    )  # Max days to keep archived metrics
    monitoring_archive_interval_hours: int = Field(
        24, validation_alias="MONITORING_ARCHIVE_INTERVAL_HOURS", ge=1, le=168
    )  # Interval for archival operations

    # Monitoring v2 — Performance Tracking
    monitoring_enable_performance_tracking: bool = Field(
        True, validation_alias="MONITORING_ENABLE_PERFORMANCE_TRACKING"
    )  # Enable performance metrics
    monitoring_performance_alert_threshold_ms: float = Field(
        1000.0, validation_alias="MONITORING_PERFORMANCE_ALERT_THRESHOLD_MS", ge=100.0, le=10000.0
    )  # Alert if latency exceeds this (ms)

    # Monitoring v2 — Email Alerts
    monitoring_enable_email_alerts: bool = Field(
        False, validation_alias="MONITORING_ENABLE_EMAIL_ALERTS"
    )  # Enable email alert delivery
    monitoring_email_smtp_host: str = Field(
        "smtp.gmail.com", validation_alias="MONITORING_EMAIL_SMTP_HOST"
    )  # SMTP server host
    monitoring_email_smtp_port: int = Field(
        587, validation_alias="MONITORING_EMAIL_SMTP_PORT", ge=1, le=65535
    )  # SMTP server port
    monitoring_email_smtp_user: str = Field(
        "", validation_alias="MONITORING_EMAIL_SMTP_USER"
    )  # SMTP username
    monitoring_email_smtp_password: str = Field(
        "", validation_alias="MONITORING_EMAIL_SMTP_PASSWORD"
    )  # SMTP password
    monitoring_email_from: str = Field(
        "", validation_alias="MONITORING_EMAIL_FROM"
    )  # From email address
    monitoring_email_to: list[str] = Field(
        default_factory=list, validation_alias="MONITORING_EMAIL_TO"
    )  # Recipient email addresses

    # Monitoring v2 — Slack Alerts
    monitoring_enable_slack_alerts: bool = Field(
        False, validation_alias="MONITORING_ENABLE_SLACK_ALERTS"
    )  # Enable Slack webhook alerts
    monitoring_slack_webhook_url: str = Field(
        "", validation_alias="MONITORING_SLACK_WEBHOOK_URL"
    )  # Slack webhook URL

    # Monitoring v2 — Webhook Alerts
    monitoring_enable_webhook_alerts: bool = Field(
        False, validation_alias="MONITORING_ENABLE_WEBHOOK_ALERTS"
    )  # Enable generic webhook alerts
    monitoring_webhook_url: str = Field(
        "", validation_alias="MONITORING_WEBHOOK_URL"
    )  # Webhook URL
    monitoring_webhook_headers: dict[str, str] = Field(
        default_factory=dict, validation_alias="MONITORING_WEBHOOK_HEADERS"
    )  # Custom headers for webhook

    # Monitoring v2 — Acknowledgement
    monitoring_ack_ttl_seconds: int = Field(
        86400, validation_alias="MONITORING_ACK_TTL_SECONDS", ge=3600, le=604800
    )  # Acknowledgement TTL (1 day default)

    # Data Feed & Historical Data
    data_feed_source: Literal["csv", "breeze", "hybrid"] = Field(
        "hybrid", validation_alias="DATA_FEED_SOURCE"
    )  # Data source: csv (local only), breeze (API only), hybrid (API + cache)
    data_feed_enable_cache: bool = Field(
        True, validation_alias="DATA_FEED_ENABLE_CACHE"
    )  # Enable caching of Breeze API responses to CSV
    data_feed_csv_directory: str = Field(
        "data/historical", validation_alias="DATA_FEED_CSV_DIRECTORY"
    )  # Base directory for CSV cache
    data_feed_cache_compression: bool = Field(
        False, validation_alias="DATA_FEED_CACHE_COMPRESSION"
    )  # Enable gzip compression for cached CSV files

    # Accuracy Audit & Telemetry
    telemetry_enabled: bool = Field(False, validation_alias="TELEMETRY_ENABLED")
    telemetry_storage_path: str = Field("data/analytics", validation_alias="TELEMETRY_STORAGE_PATH")
    telemetry_sample_rate: float = Field(
        1.0, validation_alias="TELEMETRY_SAMPLE_RATE", ge=0.0, le=1.0
    )  # 0.0-1.0 (1.0 = 100% capture rate)
    telemetry_include_features: bool = Field(
        True, validation_alias="TELEMETRY_INCLUDE_FEATURES"
    )  # Include feature values in traces
    telemetry_max_file_size_mb: int = Field(
        100, validation_alias="TELEMETRY_MAX_FILE_SIZE_MB", ge=1, le=1000
    )  # Max file size before rotation
    telemetry_compression: bool = Field(
        False, validation_alias="TELEMETRY_COMPRESSION"
    )  # Enable gzip compression for telemetry files
    telemetry_buffer_size: int = Field(
        100, validation_alias="TELEMETRY_BUFFER_SIZE", ge=1, le=10000
    )  # Number of traces to buffer before writing

    # Batch Backtesting
    batch_parallel_workers: int = Field(
        4, validation_alias="BATCH_PARALLEL_WORKERS", ge=1, le=32
    )  # Number of parallel workers for batch execution
    batch_progress_bar: bool = Field(
        True, validation_alias="BATCH_PROGRESS_BAR"
    )  # Show progress bar during batch execution

    # Live Telemetry (US-018)
    live_telemetry_enabled: bool = Field(
        False, validation_alias="LIVE_TELEMETRY_ENABLED"
    )  # Enable telemetry in live trading mode
    live_telemetry_throttle_seconds: int = Field(
        60, validation_alias="LIVE_TELEMETRY_THROTTLE_SECONDS", ge=10, le=3600
    )  # Minimum seconds between telemetry writes in live mode
    live_telemetry_sample_rate: float = Field(
        0.1, validation_alias="LIVE_TELEMETRY_SAMPLE_RATE", ge=0.0, le=1.0
    )  # Sampling rate for live telemetry (default 10%)

    # Minute Bar Data (US-018)
    minute_data_enabled: bool = Field(
        True, validation_alias="MINUTE_DATA_ENABLED"
    )  # Enable minute-level bar data support
    minute_data_cache_dir: str = Field(
        "data/market_data", validation_alias="MINUTE_DATA_CACHE_DIR"
    )  # Directory for cached minute bar data
    minute_data_resolution: Literal["1m", "5m", "15m"] = Field(
        "1m", validation_alias="MINUTE_DATA_RESOLUTION"
    )  # Minute bar resolution
    minute_data_market_hours_start: str = Field(
        "09:15", validation_alias="MINUTE_DATA_MARKET_HOURS_START"
    )  # Market opening time (HH:MM)
    minute_data_market_hours_end: str = Field(
        "15:30", validation_alias="MINUTE_DATA_MARKET_HOURS_END"
    )  # Market closing time (HH:MM)

    # Dashboard Live Mode (US-018)
    dashboard_live_threshold_minutes: int = Field(
        5, validation_alias="DASHBOARD_LIVE_THRESHOLD_MINUTES", ge=1, le=60
    )  # Minutes before telemetry considered stale
    dashboard_rolling_window_trades: int = Field(
        100, validation_alias="DASHBOARD_ROLLING_WINDOW_TRADES", ge=10, le=1000
    )  # Number of recent trades for rolling metrics

    @field_validator("position_sizing_mode")
    @classmethod
    def validate_position_sizing_mode(cls, v: str) -> str:
        """Ensure position sizing mode is valid."""
        if v not in ["FIXED_FRACTIONAL", "ATR_BASED"]:
            raise ValueError(
                f"position_sizing_mode must be 'FIXED_FRACTIONAL' or 'ATR_BASED', got '{v}'"
            )
        return v

    @field_validator("swing_sma_slow")
    @classmethod
    def validate_swing_sma_slow(cls, v: int, info: Any) -> int:
        """Ensure swing_sma_slow > swing_sma_fast."""
        if "swing_sma_fast" in info.data and v <= info.data["swing_sma_fast"]:
            raise ValueError(
                f"swing_sma_slow ({v}) must be > swing_sma_fast ({info.data['swing_sma_fast']})"
            )
        return v

    @field_validator("swing_tp_pct")
    @classmethod
    def validate_swing_tp_pct(cls, v: float, info: Any) -> float:
        """Ensure swing_tp_pct > swing_sl_pct."""
        if "swing_sl_pct" in info.data and v <= info.data["swing_sl_pct"]:
            raise ValueError(
                f"swing_tp_pct ({v}) must be > swing_sl_pct ({info.data['swing_sl_pct']})"
            )
        return v

    def get_sentiment_provider_weights(self) -> dict[str, float]:
        """Parse sentiment provider weights from JSON string.

        Returns:
            Dictionary mapping provider names to weights
        """
        try:
            weights = json.loads(self.sentiment_provider_weights)
            if not isinstance(weights, dict):
                raise ValueError("sentiment_provider_weights must be a JSON object")
            return weights
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in sentiment_provider_weights: {e}") from e

    def get_sentiment_provider_fallback_order(self) -> list[str]:
        """Parse sentiment provider fallback order from JSON string.

        Returns:
            List of provider names in fallback order
        """
        try:
            order = json.loads(self.sentiment_provider_fallback_order)
            if not isinstance(order, list):
                raise ValueError("sentiment_provider_fallback_order must be a JSON array")
            return order
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in sentiment_provider_fallback_order: {e}") from e

    # Student Model Configuration (US-021)
    student_model_enabled: bool = Field(
        False, validation_alias="STUDENT_MODEL_ENABLED"
    )  # Master switch for live scoring
    student_model_path: str = Field(
        "data/models/production/student_model.pkl", validation_alias="STUDENT_MODEL_PATH"
    )
    student_model_version: str = Field(
        "", validation_alias="STUDENT_MODEL_VERSION"
    )  # Optional version tag
    student_model_confidence_threshold: float = Field(
        0.6, validation_alias="STUDENT_MODEL_CONFIDENCE_THRESHOLD", ge=0.0, le=1.0
    )

    # Validation Thresholds for Promotion (US-021)
    promotion_min_precision_uplift: float = Field(
        0.02, validation_alias="PROMOTION_MIN_PRECISION_UPLIFT", ge=0.0, le=1.0
    )  # +2% vs baseline
    promotion_min_hit_ratio_uplift: float = Field(
        0.02, validation_alias="PROMOTION_MIN_HIT_RATIO_UPLIFT", ge=0.0, le=1.0
    )  # +2% vs baseline
    promotion_min_sharpe_uplift: float = Field(
        0.05, validation_alias="PROMOTION_MIN_SHARPE_UPLIFT", ge=0.0, le=5.0
    )  # +5% vs baseline
    promotion_require_all_criteria: bool = Field(
        True, validation_alias="PROMOTION_REQUIRE_ALL_CRITERIA"
    )  # All thresholds must pass

    # Student Model Monitoring (US-021 Phase 3)
    student_monitoring_enabled: bool = Field(
        False, validation_alias="STUDENT_MONITORING_ENABLED"
    )  # Enable student model performance monitoring
    student_monitoring_window_hours: int = Field(
        24, validation_alias="STUDENT_MONITORING_WINDOW_HOURS", ge=1, le=168
    )  # Rolling window for metrics (hours)
    student_monitoring_min_samples: int = Field(
        100, validation_alias="STUDENT_MONITORING_MIN_SAMPLES", ge=10, le=10000
    )  # Minimum samples before alerting
    student_monitoring_precision_drop_threshold: float = Field(
        0.10, validation_alias="STUDENT_MONITORING_PRECISION_DROP_THRESHOLD", ge=0.01, le=0.50
    )  # Alert if precision drops by this % (e.g., 0.10 = 10%)
    student_monitoring_hit_ratio_drop_threshold: float = Field(
        0.10, validation_alias="STUDENT_MONITORING_HIT_RATIO_DROP_THRESHOLD", ge=0.01, le=0.50
    )  # Alert if hit ratio drops by this % (e.g., 0.10 = 10%)
    student_monitoring_alert_cooldown_hours: int = Field(
        6, validation_alias="STUDENT_MONITORING_ALERT_COOLDOWN_HOURS", ge=1, le=48
    )  # Hours between repeat alerts
    student_auto_rollback_enabled: bool = Field(
        False, validation_alias="STUDENT_AUTO_ROLLBACK_ENABLED"
    )  # Enable automatic rollback on degradation
    student_auto_rollback_confirmation_hours: int = Field(
        12, validation_alias="STUDENT_AUTO_ROLLBACK_CONFIRMATION_HOURS", ge=1, le=72
    )  # Hours to wait before auto-rollback (confirmation period)

    # =====================================================================
    # US-024: Historical Data Ingestion Configuration
    # =====================================================================
    # US-028 Phase 7 Initiative 1: Symbol universe management
    historical_data_symbols_mode: str | None = Field(
        default=None,
        validation_alias="HISTORICAL_DATA_SYMBOLS_MODE",
    )  # Symbol mode: "nifty100", "metals_etfs", "pilot" (overrides historical_data_symbols)
    historical_data_symbols: list[str] = Field(
        default=["RELIANCE", "TCS", "INFY"],
        validation_alias="HISTORICAL_DATA_SYMBOLS",
    )  # Symbols to download historical data for
    historical_data_intervals: list[str] = Field(
        default=["1minute", "5minute", "1day"],
        validation_alias="HISTORICAL_DATA_INTERVALS",
    )  # Time intervals to download (1minute, 5minute, 1hour, 1day)
    historical_data_start_date: str = Field(
        "2024-01-01", validation_alias="HISTORICAL_DATA_START_DATE"
    )  # Start date for historical data (YYYY-MM-DD)
    historical_data_end_date: str = Field(
        "2024-12-31", validation_alias="HISTORICAL_DATA_END_DATE"
    )  # End date for historical data (YYYY-MM-DD)
    historical_data_output_dir: str = Field(
        "data/historical", validation_alias="HISTORICAL_DATA_OUTPUT_DIR"
    )  # Directory to store historical OHLCV CSVs
    historical_data_retry_limit: int = Field(
        3, validation_alias="HISTORICAL_DATA_RETRY_LIMIT", ge=1, le=10
    )  # Number of retry attempts for failed API calls
    historical_data_retry_backoff_seconds: int = Field(
        2, validation_alias="HISTORICAL_DATA_RETRY_BACKOFF_SECONDS", ge=1, le=30
    )  # Base delay for exponential backoff (seconds)

    # US-028: Chunked Historical Data Ingestion with Rate Limiting
    historical_chunk_days: int = Field(
        90, validation_alias="HISTORICAL_CHUNK_DAYS", ge=1, le=365
    )  # Max days per API chunk request (prevents timeout/overload)
    breeze_rate_limit_requests_per_minute: int = Field(
        30, validation_alias="BREEZE_RATE_LIMIT_REQUESTS_PER_MINUTE", ge=1, le=100
    )  # Max Breeze API requests per minute (conservative default)
    breeze_rate_limit_delay_seconds: float = Field(
        2.0, validation_alias="BREEZE_RATE_LIMIT_DELAY_SECONDS", ge=0.1, le=10.0
    )  # Delay between chunk requests to respect rate limits

    # US-028 Phase 7 Initiative 1: Symbol batch processing
    historical_symbol_batch_size: int = Field(
        10, validation_alias="HISTORICAL_SYMBOL_BATCH_SIZE", ge=1, le=50
    )  # Process symbols in batches to avoid API spikes (Phase 7 Initiative 1)

    # =====================================================================
    # US-024: Batch Training Configuration
    # =====================================================================
    batch_training_enabled: bool = Field(
        False, validation_alias="BATCH_TRAINING_ENABLED"
    )  # Enable batch teacher training
    batch_training_window_days: int = Field(
        180, validation_alias="BATCH_TRAINING_WINDOW_DAYS", ge=30, le=365
    )  # Training window size in days (US-028 Phase 6h: increased from 90 to 180 for sufficient samples after feature warm-up)
    batch_training_forecast_horizon_days: int = Field(
        7, validation_alias="BATCH_TRAINING_FORECAST_HORIZON_DAYS", ge=1, le=30
    )  # Forecast horizon for teacher labels in days
    batch_training_min_samples: int = Field(
        20, validation_alias="BATCH_TRAINING_MIN_SAMPLES", ge=10, le=100
    )  # Minimum labeled samples required for training (US-028 Phase 6h)
    batch_training_output_dir: str = Field(
        "data/models", validation_alias="BATCH_TRAINING_OUTPUT_DIR"
    )  # Directory to store batch training artifacts
    batch_training_parallel_workers: int = Field(
        1, validation_alias="BATCH_TRAINING_PARALLEL_WORKERS", ge=1, le=16
    )  # Number of parallel workers (1 = sequential)

    # =====================================================================
    # US-024 Phase 2: Student Batch Training Configuration
    # =====================================================================
    student_batch_enabled: bool = Field(
        False, validation_alias="STUDENT_BATCH_ENABLED"
    )  # Enable batch student training
    student_batch_baseline_precision: float = Field(
        0.60, validation_alias="STUDENT_BATCH_BASELINE_PRECISION", ge=0.0, le=1.0
    )  # Baseline precision for student model promotion
    student_batch_baseline_recall: float = Field(
        0.55, validation_alias="STUDENT_BATCH_BASELINE_RECALL", ge=0.0, le=1.0
    )  # Baseline recall for student model promotion
    student_batch_output_dir: str = Field(
        "data/models", validation_alias="STUDENT_BATCH_OUTPUT_DIR"
    )  # Directory to store student batch training artifacts
    student_batch_promotion_enabled: bool = Field(
        True, validation_alias="STUDENT_BATCH_PROMOTION_ENABLED"
    )  # Enable automatic promotion checklist generation

    # US-028 Phase 7 Initiative 2: Teacher-Student Reward Loop Configuration
    reward_loop_enabled: bool = Field(
        False, validation_alias="REWARD_LOOP_ENABLED"
    )  # Enable adaptive learning via reward signals (US-028 Phase 7 Initiative 2)
    reward_horizon_days: int = Field(
        5, validation_alias="REWARD_HORIZON_DAYS", ge=1, le=30
    )  # Number of days to look ahead for realized returns
    reward_clip_min: float = Field(
        -2.0, validation_alias="REWARD_CLIP_MIN"
    )  # Minimum reward value (clip negative rewards)
    reward_clip_max: float = Field(
        2.0, validation_alias="REWARD_CLIP_MAX"
    )  # Maximum reward value (clip positive rewards)
    reward_weighting_mode: str = Field(
        "linear", validation_alias="REWARD_WEIGHTING_MODE"
    )  # Sample weighting mode: "linear", "exponential", "none"
    reward_weighting_scale: float = Field(
        1.0, validation_alias="REWARD_WEIGHTING_SCALE", ge=0.0, le=10.0
    )  # Scaling factor for reward-based sample weights
    reward_ab_testing_enabled: bool = Field(
        False, validation_alias="REWARD_AB_TESTING_ENABLED"
    )  # Enable A/B testing (baseline vs reward-weighted training)

    # US-028 Phase 7 Initiative 3: Black-Swan Stress Test Configuration
    stress_tests_enabled: bool = Field(
        False, validation_alias="STRESS_TESTS_ENABLED"
    )  # Enable Phase 8 stress testing against historical crisis periods
    stress_test_severity_filter: list[str] = Field(
        ["extreme", "high"], validation_alias="STRESS_TEST_SEVERITY_FILTER"
    )  # Severity levels to test (extreme, high, medium, low)
    stress_test_specific_periods: list[str] | None = Field(
        None, validation_alias="STRESS_TEST_SPECIFIC_PERIODS"
    )  # Specific period IDs to test (None = use severity filter)

    # US-024 Phase 3: Sentiment Snapshot Configuration
    sentiment_snapshot_enabled: bool = Field(
        False, validation_alias="SENTIMENT_SNAPSHOT_ENABLED"
    )  # Enable sentiment snapshot ingestion (disabled by default)
    sentiment_snapshot_providers: list[str] = Field(
        ["stub"], validation_alias="SENTIMENT_SNAPSHOT_PROVIDERS"
    )  # Sentiment providers to use (e.g., ["newsapi", "twitter", "stub"])
    sentiment_snapshot_output_dir: str = Field(
        "data/sentiment", validation_alias="SENTIMENT_SNAPSHOT_OUTPUT_DIR"
    )  # Directory to store sentiment snapshots
    sentiment_snapshot_retry_limit: int = Field(
        3, validation_alias="SENTIMENT_SNAPSHOT_RETRY_LIMIT", ge=1, le=10
    )  # Maximum retry attempts for sentiment fetch
    sentiment_snapshot_retry_backoff_seconds: int = Field(
        2, validation_alias="SENTIMENT_SNAPSHOT_RETRY_BACKOFF_SECONDS", ge=1, le=60
    )  # Base backoff delay in seconds for retries
    sentiment_snapshot_max_per_day: int = Field(
        100, validation_alias="SENTIMENT_SNAPSHOT_MAX_PER_DAY", ge=1
    )  # Maximum sentiment fetches per day (rate limiting)

    # US-024 Phase 4: Incremental Daily Updates Configuration
    incremental_enabled: bool = Field(
        False, validation_alias="INCREMENTAL_ENABLED"
    )  # Enable incremental daily updates (disabled by default)
    incremental_lookback_days: int = Field(
        30, validation_alias="INCREMENTAL_LOOKBACK_DAYS", ge=1, le=365
    )  # Lookback window for incremental fetch (days)
    incremental_cron_schedule: str = Field(
        "0 18 * * 1-5", validation_alias="INCREMENTAL_CRON_SCHEDULE"
    )  # Cron schedule for incremental runs (Mon-Fri at 6PM)

    # US-024 Phase 5: Distributed Training & Scheduled Automation Configuration
    parallel_workers: int = Field(
        1, validation_alias="PARALLEL_WORKERS", ge=1, le=32
    )  # Number of parallel workers for batch training (1 = sequential)
    parallel_retry_limit: int = Field(
        3, validation_alias="PARALLEL_RETRY_LIMIT", ge=1, le=10
    )  # Max retry attempts for failed batch tasks
    parallel_retry_backoff_seconds: int = Field(
        5, validation_alias="PARALLEL_RETRY_BACKOFF_SECONDS", ge=1, le=60
    )  # Backoff between retries (seconds)
    batch_training_max_failure_rate: float = Field(
        0.15, validation_alias="BATCH_TRAINING_MAX_FAILURE_RATE", ge=0.0, le=1.0
    )  # Max acceptable failure rate (0.15 = 15%) before batch exit code 1

    # Teacher Model GPU Tuning Parameters (US-028 Phase 7 GPU Optimization)
    teacher_gpu_platform_id: int = Field(
        0, validation_alias="TEACHER_GPU_PLATFORM_ID", ge=0, le=7
    )  # OpenCL platform ID (usually 0)
    teacher_gpu_device_id: int = Field(
        0, validation_alias="TEACHER_GPU_DEVICE_ID", ge=0, le=7
    )  # GPU device ID (0 for first GPU, 1 for second, etc.)
    teacher_gpu_use_dp: bool = Field(
        False, validation_alias="TEACHER_GPU_USE_DP"
    )  # Use double precision on GPU (slower but more accurate)
    teacher_num_leaves: int = Field(
        127, validation_alias="TEACHER_NUM_LEAVES", ge=2, le=1024
    )  # Max number of leaves in tree (higher = more complex)
    teacher_max_depth: int = Field(
        9, validation_alias="TEACHER_MAX_DEPTH", ge=1, le=20
    )  # Maximum tree depth (higher = more complex, slower)
    teacher_learning_rate: float = Field(
        0.01, validation_alias="TEACHER_LEARNING_RATE", ge=0.001, le=1.0
    )  # Learning rate for gradient boosting
    teacher_n_estimators: int = Field(
        500, validation_alias="TEACHER_N_ESTIMATORS", ge=10, le=5000
    )  # Number of boosting rounds (with early stopping)
    teacher_min_child_samples: int = Field(
        20, validation_alias="TEACHER_MIN_CHILD_SAMPLES", ge=1, le=1000
    )  # Minimum samples per leaf (regularization)
    teacher_subsample: float = Field(
        0.8, validation_alias="TEACHER_SUBSAMPLE", ge=0.1, le=1.0
    )  # Row sampling fraction (bagging)
    teacher_colsample_bytree: float = Field(
        0.8, validation_alias="TEACHER_COLSAMPLE_BYTREE", ge=0.1, le=1.0
    )  # Feature sampling fraction per tree

    scheduled_pipeline_skip_fetch: bool = Field(
        False, validation_alias="SCHEDULED_PIPELINE_SKIP_FETCH"
    )  # Skip data fetch phase in scheduled pipeline
    scheduled_pipeline_skip_teacher: bool = Field(
        False, validation_alias="SCHEDULED_PIPELINE_SKIP_TEACHER"
    )  # Skip teacher training phase in scheduled pipeline
    scheduled_pipeline_skip_student: bool = Field(
        False, validation_alias="SCHEDULED_PIPELINE_SKIP_STUDENT"
    )  # Skip student training phase in scheduled pipeline

    # US-024 Phase 6: Data Quality Dashboard & Alerts Configuration
    data_quality_scan_enabled: bool = Field(
        False, validation_alias="DATA_QUALITY_SCAN_ENABLED"
    )  # Enable automatic data quality scanning (disabled by default)
    data_quality_scan_after_fetch: bool = Field(
        True, validation_alias="DATA_QUALITY_SCAN_AFTER_FETCH"
    )  # Run quality scan after data fetch
    data_quality_alert_threshold_missing_files: int = Field(
        10, validation_alias="DATA_QUALITY_ALERT_THRESHOLD_MISSING_FILES", ge=0
    )  # Alert if missing files exceed this count
    data_quality_alert_threshold_duplicate_timestamps: int = Field(
        100, validation_alias="DATA_QUALITY_ALERT_THRESHOLD_DUPLICATE_TIMESTAMPS", ge=0
    )  # Alert if duplicate timestamps exceed this count
    data_quality_alert_threshold_zero_volume: int = Field(
        50, validation_alias="DATA_QUALITY_ALERT_THRESHOLD_ZERO_VOLUME", ge=0
    )  # Alert if zero-volume bars exceed this count
    data_quality_dashboard_enabled: bool = Field(
        False, validation_alias="DATA_QUALITY_DASHBOARD_ENABLED"
    )  # Enable Streamlit dashboard (requires streamlit package)

    # =====================================================================
    # US-029: Order Book, Options, and Macro Data Integration (Phase 1)
    # =====================================================================

    # Order Book Configuration
    order_book_enabled: bool = Field(
        False, validation_alias="ORDER_BOOK_ENABLED"
    )  # Enable order book depth snapshot ingestion (disabled by default)
    order_book_provider: str = Field(
        "stub", validation_alias="ORDER_BOOK_PROVIDER"
    )  # Order book data provider: "stub", "breeze", "websocket"
    order_book_endpoint: str = Field(
        "", validation_alias="ORDER_BOOK_ENDPOINT"
    )  # Provider API endpoint URL
    order_book_depth_levels: int = Field(
        5, validation_alias="ORDER_BOOK_DEPTH_LEVELS", ge=1, le=20
    )  # Number of price levels to capture (best 1-20 levels)
    order_book_snapshot_interval_seconds: int = Field(
        60, validation_alias="ORDER_BOOK_SNAPSHOT_INTERVAL_SECONDS", ge=1, le=3600
    )  # Snapshot interval in seconds (1s to 1h)
    order_book_retention_days: int = Field(
        7, validation_alias="ORDER_BOOK_RETENTION_DAYS", ge=1, le=90
    )  # Retention period in days
    order_book_retry_limit: int = Field(
        3, validation_alias="ORDER_BOOK_RETRY_LIMIT", ge=1, le=10
    )  # Maximum retry attempts for failed fetches
    order_book_retry_backoff_seconds: int = Field(
        2, validation_alias="ORDER_BOOK_RETRY_BACKOFF_SECONDS", ge=1, le=60
    )  # Base backoff delay for exponential retry (seconds)
    order_book_output_dir: str = Field(
        "data/order_book", validation_alias="ORDER_BOOK_OUTPUT_DIR"
    )  # Output directory for order book snapshots

    # Options Chain Configuration
    options_enabled: bool = Field(
        False, validation_alias="OPTIONS_ENABLED"
    )  # Enable options chain data ingestion (disabled by default)
    options_provider: str = Field(
        "stub", validation_alias="OPTIONS_PROVIDER"
    )  # Options data provider: "stub", "breeze", "nse"
    options_endpoint: str = Field(
        "", validation_alias="OPTIONS_ENDPOINT"
    )  # Provider API endpoint URL
    options_refresh_interval_hours: int = Field(
        24, validation_alias="OPTIONS_REFRESH_INTERVAL_HOURS", ge=1, le=168
    )  # Refresh interval in hours (1h to 7 days)
    options_retention_days: int = Field(
        30, validation_alias="OPTIONS_RETENTION_DAYS", ge=1, le=365
    )  # Retention period in days
    options_retry_limit: int = Field(
        3, validation_alias="OPTIONS_RETRY_LIMIT", ge=1, le=10
    )  # Maximum retry attempts
    options_retry_backoff_seconds: int = Field(
        2, validation_alias="OPTIONS_RETRY_BACKOFF_SECONDS", ge=1, le=60
    )  # Base backoff delay for exponential retry (seconds)
    options_output_dir: str = Field(
        "data/options", validation_alias="OPTIONS_OUTPUT_DIR"
    )  # Output directory for options chain data

    # Macro Economic Data Configuration
    macro_enabled: bool = Field(
        False, validation_alias="MACRO_ENABLED"
    )  # Enable macro economic data ingestion (disabled by default)
    macro_provider: str = Field(
        "stub", validation_alias="MACRO_PROVIDER"
    )  # Macro data provider: "stub", "yfinance", "rbi"
    macro_endpoint: str = Field("", validation_alias="MACRO_ENDPOINT")  # Provider API endpoint URL
    macro_indicators: list[str] = Field(
        default=["NIFTY50", "INDIAVIX", "USDINR", "IN10Y"],
        validation_alias="MACRO_INDICATORS",
    )  # List of macro indicators to fetch
    macro_refresh_interval_hours: int = Field(
        24, validation_alias="MACRO_REFRESH_INTERVAL_HOURS", ge=1, le=168
    )  # Refresh interval in hours (1h to 7 days)
    macro_retention_days: int = Field(
        90, validation_alias="MACRO_RETENTION_DAYS", ge=1, le=730
    )  # Retention period in days (up to 2 years)
    macro_retry_limit: int = Field(
        3, validation_alias="MACRO_RETRY_LIMIT", ge=1, le=10
    )  # Maximum retry attempts
    macro_retry_backoff_seconds: int = Field(
        2, validation_alias="MACRO_RETRY_BACKOFF_SECONDS", ge=1, le=60
    )  # Base backoff delay for exponential retry (seconds)
    macro_output_dir: str = Field(
        "data/macro", validation_alias="MACRO_OUTPUT_DIR"
    )  # Output directory for macro indicator data

    # =====================================================================
    # US-029 Phase 2: Feature Engineering Configuration
    # =====================================================================

    # Feature Engineering Toggles (disabled by default)
    enable_order_book_features: bool = Field(
        False, validation_alias="ENABLE_ORDER_BOOK_FEATURES"
    )  # Enable order book feature computation
    enable_options_features: bool = Field(
        False, validation_alias="ENABLE_OPTIONS_FEATURES"
    )  # Enable options feature computation
    enable_macro_features: bool = Field(
        False, validation_alias="ENABLE_MACRO_FEATURES"
    )  # Enable macro feature computation

    # Feature Computation Parameters
    order_book_feature_lookback_seconds: int = Field(
        60, validation_alias="ORDER_BOOK_FEATURE_LOOKBACK_SECONDS", ge=1, le=600
    )  # Lookback window for order book time-weighted features
    options_feature_iv_lookback_days: int = Field(
        30, validation_alias="OPTIONS_FEATURE_IV_LOOKBACK_DAYS", ge=1, le=365
    )  # Lookback days for IV percentile calculation
    macro_feature_correlation_window: int = Field(
        30, validation_alias="MACRO_FEATURE_CORRELATION_WINDOW", ge=1, le=365
    )  # Window for macro correlation features
    macro_feature_short_window: int = Field(
        10, validation_alias="MACRO_FEATURE_SHORT_WINDOW", ge=1, le=100
    )  # Short MA window for macro momentum
    macro_feature_long_window: int = Field(
        50, validation_alias="MACRO_FEATURE_LONG_WINDOW", ge=1, le=500
    )  # Long MA window for macro momentum

    # =====================================================================
    # US-029 Phase 3: Strategy Feature Integration (all disabled by default)
    # =====================================================================

    # Intraday Strategy Feature Gates
    intraday_spread_filter_enabled: bool = Field(
        False, validation_alias="INTRADAY_SPREAD_FILTER_ENABLED"
    )  # Enable order book spread filter
    intraday_max_spread_pct: float = Field(
        0.5, validation_alias="INTRADAY_MAX_SPREAD_PCT", ge=0.01, le=5.0
    )  # Max allowed spread percentage (block if exceeded)
    intraday_market_pressure_adjustment_enabled: bool = Field(
        False, validation_alias="INTRADAY_MARKET_PRESSURE_ADJUSTMENT_ENABLED"
    )  # Enable signal strength adjustment based on market pressure
    intraday_iv_adjustment_enabled: bool = Field(
        False, validation_alias="INTRADAY_IV_ADJUSTMENT_ENABLED"
    )  # Enable IV-based signal gating
    intraday_macro_regime_filter_enabled: bool = Field(
        False, validation_alias="INTRADAY_MACRO_REGIME_FILTER_ENABLED"
    )  # Enable macro regime filtering

    # Swing Strategy Feature Gates
    swing_iv_position_sizing_enabled: bool = Field(
        False, validation_alias="SWING_IV_POSITION_SIZING_ENABLED"
    )  # Enable IV-based position size adjustment
    swing_macro_correlation_filter_enabled: bool = Field(
        False, validation_alias="SWING_MACRO_CORRELATION_FILTER_ENABLED"
    )  # Enable macro correlation filtering

    # Optimizer Feature Testing
    optimizer_test_feature_combinations: bool = Field(
        False, validation_alias="OPTIMIZER_TEST_FEATURE_COMBINATIONS"
    )  # Include feature toggles in optimizer parameter grid

    # =====================================================================
    # US-029 Phase 5: Real-Time Streaming Configuration
    # =====================================================================

    # Streaming Master Control
    streaming_enabled: bool = Field(
        False, validation_alias="STREAMING_ENABLED"
    )  # Master switch for real-time streaming (disabled by default)

    # Streaming Buffer Configuration
    streaming_buffer_size: int = Field(
        100, validation_alias="STREAMING_BUFFER_SIZE", ge=10, le=1000
    )  # Max snapshots to buffer per symbol (circular buffer)

    streaming_update_interval_seconds: int = Field(
        1, validation_alias="STREAMING_UPDATE_INTERVAL_SECONDS", ge=1, le=60
    )  # Update frequency for streaming snapshots (seconds)

    # Streaming Monitoring & Health
    streaming_heartbeat_timeout_seconds: int = Field(
        30, validation_alias="STREAMING_HEARTBEAT_TIMEOUT_SECONDS", ge=5, le=300
    )  # Heartbeat timeout for streaming health alerts (seconds)

    streaming_max_consecutive_errors: int = Field(
        10, validation_alias="STREAMING_MAX_CONSECUTIVE_ERRORS", ge=1, le=100
    )  # Max consecutive errors before alerting

    # Background Ingestion (Daemon Mode)
    background_ingestion_enabled: bool = Field(
        False, validation_alias="BACKGROUND_INGESTION_ENABLED"
    )  # Enable background daemon ingestion (disabled by default)

    background_ingestion_interval_seconds: int = Field(
        300, validation_alias="BACKGROUND_INGESTION_INTERVAL_SECONDS", ge=60, le=3600
    )  # Interval for background fetch loops (5 min default)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    def get_symbols_for_mode(self, mode: str | None = None) -> list[str]:
        """Load symbols from metadata based on symbols_mode (US-028 Phase 7 Initiative 1).

        Args:
            mode: Symbol mode ("nifty100", "metals_etfs", "pilot", "all")
                 If None, uses self.historical_data_symbols_mode or falls back to self.historical_data_symbols

        Returns:
            List of symbols for the specified mode

        Raises:
            FileNotFoundError: If metadata file doesn't exist
            ValueError: If mode is invalid
        """
        from pathlib import Path

        mode = mode or self.historical_data_symbols_mode

        # If no mode specified, return default symbols
        if not mode:
            return self.historical_data_symbols

        # Load metadata file
        metadata_path = Path("data/historical/metadata/nifty100_constituents.json")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Symbol metadata not found: {metadata_path}. "
                "Run 'python scripts/setup_metadata.py' to create it."
            )

        with open(metadata_path) as f:
            metadata = json.load(f)

        symbols_data = metadata.get("symbols", [])
        categories_data = metadata.get("categories", {})

        # Filter by mode
        if mode == "all":
            # Return all symbols
            return symbols_data if isinstance(symbols_data, list) else []
        elif mode == "nifty100":
            # Return all NIFTY 100 symbols
            # Include all verified and placeholder categories (metals companies are valid NIFTY constituents)
            # Only exclude symbols explicitly listed in data_unavailable array
            nifty_symbols = []
            data_unavailable = metadata.get("data_unavailable", [])

            for _category, symbols in categories_data.items():
                # Include all categories (metals_etfs_verified, metals_placeholder, large_cap_verified, etc.)
                if isinstance(symbols, list):
                    # Filter out symbols with no data available
                    available_symbols = [s for s in symbols if s not in data_unavailable]
                    nifty_symbols.extend(available_symbols)
            return nifty_symbols
        elif mode == "metals_etfs":
            # Return only metals ETFs (GOLDBEES, SILVERBEES)
            metals_symbols = []
            for category, symbols in categories_data.items():
                if "metals_etfs" in category:
                    if isinstance(symbols, list):
                        metals_symbols.extend(symbols)
            return metals_symbols
        elif mode == "pilot":
            # Return pilot subset (first 3 large_cap symbols + both metals ETFs)
            # Look for verified large cap symbols
            large_cap = []
            metals = []
            for category, symbols in categories_data.items():
                if "large_cap_verified" in category and isinstance(symbols, list):
                    large_cap = symbols[:3]
                if "metals_etfs" in category and isinstance(symbols, list):
                    metals = symbols
            return large_cap + metals
        else:
            raise ValueError(
                f"Invalid symbols_mode: {mode}. "
                "Valid modes: 'all', 'nifty100', 'metals_etfs', 'pilot'"
            )

    def get_stress_periods(
        self, period_ids: list[str] | None = None, severity_filter: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Load stress period metadata for black-swan testing (US-028 Phase 7 Initiative 3).

        Args:
            period_ids: List of specific period IDs to load (e.g., ["covid_crash_2020"])
                       If None, returns all periods
            severity_filter: Filter by severity levels (e.g., ["extreme", "high"])
                           If None, no severity filtering

        Returns:
            List of stress period dicts with id, name, start_date, end_date, etc.

        Raises:
            FileNotFoundError: If stress_periods.json doesn't exist
            ValueError: If requested period_id is not found
        """
        from pathlib import Path

        # Load stress periods metadata
        metadata_path = Path("data/historical/metadata/stress_periods.json")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Stress periods metadata not found: {metadata_path}. "
                "Expected metadata file with historical crisis periods."
            )

        with open(metadata_path) as f:
            metadata = json.load(f)

        periods = metadata.get("periods", [])

        # Filter by period IDs if specified
        if period_ids:
            filtered = [p for p in periods if p["id"] in period_ids]
            # Verify all requested IDs were found
            found_ids = {p["id"] for p in filtered}
            missing = set(period_ids) - found_ids
            if missing:
                raise ValueError(
                    f"Stress period ID(s) not found: {missing}. "
                    f"Available IDs: {[p['id'] for p in periods]}"
                )
            periods = filtered

        # Filter by severity if specified
        if severity_filter:
            periods = [p for p in periods if p.get("severity") in severity_filter]

        return periods  # type: ignore[no-any-return]


settings = Settings()  # type: ignore[call-arg]
