"""Domain types for SenseQuant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

# Type aliases
OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]
OrderStatus = Literal["NEW", "PLACED", "FILLED", "REJECTED", "CANCELLED"]
SignalDirection = Literal["LONG", "SHORT", "FLAT"]


@dataclass
class Bar:
    """OHLCV bar."""

    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Signal:
    """Trading signal."""

    symbol: str
    direction: SignalDirection
    strength: float
    meta: dict[str, Any] | None = None


@dataclass
class Order:
    """Order request."""

    symbol: str
    side: OrderSide
    qty: int
    order_type: OrderType = "MARKET"
    price: float | None = None


@dataclass
class OrderResponse:
    """Order response from broker."""

    order_id: str
    status: OrderStatus
    raw: dict[str, Any] | None = None


@dataclass
class Position:
    """Open position."""

    symbol: str
    qty: int
    avg_price: float
