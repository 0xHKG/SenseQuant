from pydantic import BaseSettings, Field
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    # Breeze creds
    breeze_api_key: str = Field(default="", env="BREEZE_API_KEY")
    breeze_api_secret: str = Field(default="", env="BREEZE_API_SECRET")
    breeze_session_token: str = Field(default="", env="BREEZE_SESSION_TOKEN")
    # Trading
    symbols: List[str] = Field(default_factory=lambda: ["RELIANCE"])
    mode: str = Field(default="dryrun")  # dryrun|live
    # risk
    max_position_value: float = 50000.0
    per_trade_risk_pct: float = 0.01  # 1%

    class Config:
        case_sensitive = False

settings = Settings()

