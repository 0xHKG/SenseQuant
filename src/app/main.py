"""Main CLI entry point."""

from __future__ import annotations

import sys
import time
from typing import Any

from loguru import logger

from src.app.config import settings
from src.services.engine import Engine, is_market_open


def setup_logging() -> None:
    """Configure structured JSON logging with redaction."""
    # Remove default handler
    logger.remove()

    # Console handler with text format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # JSON log file with daily rotation and retention
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        serialize=True,
        rotation="00:00",
        retention="30 days",
        level="INFO",
        # Redact sensitive keys
        filter=lambda record: redact_sensitive_fields(record),
    )


def redact_sensitive_fields(record: Any) -> bool:
    """Redact sensitive fields from log records."""
    sensitive_keys = {"api_key", "api_secret", "session_token", "password", "token"}

    # Check extra context
    if "extra" in record:
        for key in list(record["extra"].keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                record["extra"][key] = "***REDACTED***"

    # Check message for sensitive patterns (basic)
    message = record.get("message", "")
    for sensitive in sensitive_keys:
        if sensitive in message.lower():
            # Keep the log but warn
            pass

    return True


def main() -> None:
    """Main entry point."""
    setup_logging()

    logger.info("Starting SenseQuant v0.1.0")
    logger.info("Mode: {}", settings.mode)
    logger.info("Symbols: {}", settings.symbols)

    engine = Engine(settings.symbols)
    engine.start()

    # Simple event loop
    try:
        while True:
            if is_market_open():
                for sym in settings.symbols:
                    engine.tick_intraday(sym)
            else:
                logger.info("Market closed; sleeping longer")
                time.sleep(60)
            time.sleep(5)  # throttle
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        engine.stop()
    except Exception as e:
        logger.exception("Fatal error: {}", e)
        engine.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
