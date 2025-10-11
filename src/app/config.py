"""Application configuration using Pydantic v2."""

from __future__ import annotations

from typing import Literal

from dotenv import find_dotenv, load_dotenv
from pydantic import Field
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
    symbols: list[str] = Field(default_factory=lambda: ["RELIANCE"], validation_alias="SYMBOLS")
    mode: Literal["dryrun", "live", "backtest"] = Field("dryrun", validation_alias="MODE")

    # Risk
    max_position_value: float = Field(50_000.0, validation_alias="MAX_POSITION_VALUE")
    per_trade_risk_pct: float = Field(0.01, validation_alias="PER_TRADE_RISK_PCT")

    # Intraday Strategy
    intraday_bar_interval: Literal["1minute"] = Field("1minute", validation_alias="INTRADAY_BAR_INTERVAL")
    intraday_feature_lookback_minutes: int = Field(60, validation_alias="INTRADAY_FEATURE_LOOKBACK_MINUTES")
    intraday_tick_seconds: int = Field(5, validation_alias="INTRADAY_TICK_SECONDS")
    intraday_sma_period: int = Field(20, validation_alias="INTRADAY_SMA_PERIOD")
    intraday_ema_period: int = Field(50, validation_alias="INTRADAY_EMA_PERIOD")
    intraday_rsi_period: int = Field(14, validation_alias="INTRADAY_RSI_PERIOD")
    intraday_atr_period: int = Field(14, validation_alias="INTRADAY_ATR_PERIOD")
    intraday_long_rsi_min: int = Field(55, validation_alias="INTRADAY_LONG_RSI_MIN")
    intraday_short_rsi_max: int = Field(45, validation_alias="INTRADAY_SHORT_RSI_MAX")

    # Sentiment Analysis
    sentiment_pos_limit: float = Field(0.15, validation_alias="SENTIMENT_POS_LIMIT")
    sentiment_neg_limit: float = Field(-0.15, validation_alias="SENTIMENT_NEG_LIMIT")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )


settings = Settings()
