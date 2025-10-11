"""Trade journal CSV writer."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz

from src.domain.types import Order, OrderResponse


class TradeJournal:
    """CSV-based trade journal with daily rotation."""

    SCHEMA = [
        "timestamp_ist",
        "symbol",
        "action",
        "qty",
        "price",
        "pnl",
        "reason",
        "mode",
        "order_id",
        "status",
        "strategy",
        "meta_json",
    ]

    def __init__(self, log_dir: Path | str = "logs/journal") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file_handle: Any = None
        self._writer: csv.DictWriter[str] | None = None
        self._current_date: str | None = None

    def _rotate_if_needed(self) -> None:
        """Create new CSV file daily."""
        ist = pytz.timezone("Asia/Kolkata")
        today = datetime.now(ist).date().isoformat()

        if self._current_date != today:
            if self._file_handle:
                self._file_handle.close()

            self._current_date = today
            csv_path = self.log_dir / f"{today}.csv"
            file_exists = csv_path.exists()

            self._file_handle = open(csv_path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file_handle, fieldnames=self.SCHEMA)

            if not file_exists:
                self._writer.writeheader()

    def log_trade(
        self,
        order: Order,
        response: OrderResponse,
        pnl: float | None = None,
        reason: str = "",
        mode: str = "dryrun",
        strategy: str = "",
    ) -> None:
        """
        Append trade to journal.

        Args:
            order: Order object
            response: OrderResponse object
            pnl: Profit/loss for the trade (optional)
            reason: Reason for the trade
            mode: Trading mode (dryrun/live)
            strategy: Strategy name
        """
        self._rotate_if_needed()
        assert self._writer is not None

        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)

        meta_dict: dict[str, Any] = {}
        if response.raw:
            meta_dict["response"] = response.raw

        row = {
            "timestamp_ist": now_ist.isoformat(),
            "symbol": order.symbol,
            "action": order.side,
            "qty": order.qty,
            "price": order.price or "",
            "pnl": f"{pnl:.2f}" if pnl is not None else "",
            "reason": reason,
            "mode": mode,
            "order_id": response.order_id,
            "status": response.status,
            "strategy": strategy,
            "meta_json": json.dumps(meta_dict) if meta_dict else "",
        }

        self._writer.writerow(row)
        if self._file_handle:
            self._file_handle.flush()

    def log(
        self,
        symbol: str,
        action: str,
        qty: int,
        price: float,
        pnl: float,
        reason: str,
        mode: str,
        order_id: str,
        status: str,
        strategy: str = "",
        meta_json: str = "",
    ) -> None:
        """
        Append generic log entry to journal.

        Args:
            symbol: Stock symbol
            action: Action (BUY/SELL/LONG/SHORT/FLAT)
            qty: Quantity
            price: Price
            pnl: Profit/loss
            reason: Reason for the action
            mode: Trading mode (dryrun/live)
            order_id: Order ID
            status: Order status
            strategy: Strategy name (optional)
            meta_json: JSON metadata (optional)
        """
        self._rotate_if_needed()
        assert self._writer is not None

        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)

        row = {
            "timestamp_ist": now_ist.isoformat(),
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "price": f"{price:.2f}" if price else "",
            "pnl": f"{pnl:.2f}" if pnl else "",
            "reason": reason,
            "mode": mode,
            "order_id": order_id,
            "status": status,
            "strategy": strategy,
            "meta_json": meta_json,
        }

        self._writer.writerow(row)
        if self._file_handle:
            self._file_handle.flush()

    def close(self) -> None:
        """Close journal file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
            self._writer = None
