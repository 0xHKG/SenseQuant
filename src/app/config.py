"""Application configuration using Pydantic v2."""

from __future__ import annotations

import json
from typing import Any, Literal

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from repo root robustly
load_dotenv(find_dotenv())


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

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )


settings = Settings()  # type: ignore[call-arg]
