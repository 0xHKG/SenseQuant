"""State management for incremental data fetching (US-024 Phase 4).

Tracks last fetch dates for historical OHLCV and sentiment data to enable
incremental daily updates without re-downloading existing data.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class StateManager:
    """Manages state files for incremental data fetching.

    State files track the last successful fetch date per symbol/data type,
    enabling incremental updates that fetch only new data since the last run.

    State files are stored as JSON in data/state/ directory:
    - historical_fetch.json: Last OHLCV fetch date per symbol
    - sentiment_fetch.json: Last sentiment fetch date per symbol
    """

    def __init__(self, state_file: Path):
        """Initialize state manager.

        Args:
            state_file: Path to state JSON file
        """
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing state
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        """Load state from file.

        Returns:
            State dictionary, or empty dict if file doesn't exist
        """
        if not self.state_file.exists():
            logger.debug(f"State file not found, creating new: {self.state_file}")
            return {}

        try:
            with open(self.state_file) as f:
                state: dict[str, Any] = json.load(f)
            logger.debug(
                f"Loaded state from {self.state_file}: {len(state.get('symbols', {}))} symbols"
            )
            return state
        except Exception as e:
            logger.warning(f"Failed to load state file {self.state_file}: {e}, using empty state")
            return {}

    def _save_state(self) -> None:
        """Save state to file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
            logger.debug(f"Saved state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state file {self.state_file}: {e}")

    def get_last_fetch_date(self, symbol: str) -> datetime | None:
        """Get last fetch date for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Last fetch date as datetime, or None if never fetched
        """
        symbols = self.state.get("symbols", {})
        if symbol not in symbols:
            return None

        date_str = symbols[symbol].get("last_fetch_date")
        if not date_str:
            return None

        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            logger.warning(f"Invalid date format in state for {symbol}: {date_str}")
            return None

    def set_last_fetch_date(self, symbol: str, date: datetime) -> None:
        """Set last fetch date for symbol.

        Args:
            symbol: Stock symbol
            date: Last fetch date
        """
        if "symbols" not in self.state:
            self.state["symbols"] = {}

        if symbol not in self.state["symbols"]:
            self.state["symbols"][symbol] = {}

        self.state["symbols"][symbol]["last_fetch_date"] = date.isoformat()
        self.state["symbols"][symbol]["last_updated"] = datetime.now().isoformat()

        self._save_state()

        logger.info(
            f"Updated last fetch date for {symbol}: {date.strftime('%Y-%m-%d')}",
            extra={"symbol": symbol, "last_fetch_date": date.isoformat()},
        )

    def get_last_run_info(self) -> dict[str, Any] | None:
        """Get information about the last run.

        Returns:
            Dict with last run metadata, or None if never run
        """
        return self.state.get("last_run")

    def set_last_run_info(
        self,
        run_type: str,
        success: bool,
        symbols_processed: list[str],
        files_created: int,
        errors: int = 0,
    ) -> None:
        """Set information about the last run.

        Args:
            run_type: Type of run ('full' or 'incremental')
            success: Whether run completed successfully
            symbols_processed: List of symbols processed
            files_created: Number of files created
            errors: Number of errors encountered
        """
        self.state["last_run"] = {
            "timestamp": datetime.now().isoformat(),
            "run_type": run_type,
            "success": success,
            "symbols_processed": symbols_processed,
            "files_created": files_created,
            "errors": errors,
        }

        self._save_state()

    def get_all_symbols(self) -> list[str]:
        """Get list of all symbols in state.

        Returns:
            List of symbol names
        """
        return list(self.state.get("symbols", {}).keys())

    def clear_symbol(self, symbol: str) -> None:
        """Clear state for a specific symbol.

        Args:
            symbol: Stock symbol to clear
        """
        if "symbols" in self.state and symbol in self.state["symbols"]:
            del self.state["symbols"][symbol]
            self._save_state()
            logger.info(f"Cleared state for symbol: {symbol}")

    def clear_all(self) -> None:
        """Clear all state."""
        self.state = {}
        self._save_state()
        logger.info("Cleared all state")

    # US-024 Phase 5: Batch execution status tracking

    def get_batch_status(self, batch_id: str) -> dict[str, Any] | None:
        """Get batch execution status.

        Args:
            batch_id: Unique batch identifier

        Returns:
            Dict with batch status, or None if not found
        """
        batches = self.state.get("batches", {})
        return batches.get(batch_id)

    def set_batch_status(
        self,
        batch_id: str,
        status: str,
        total_tasks: int,
        completed: int = 0,
        failed: int = 0,
        pending_retries: int = 0,
        failed_tasks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Set batch execution status.

        Args:
            batch_id: Unique batch identifier
            status: Batch status ('running', 'completed', 'failed', 'partial')
            total_tasks: Total number of tasks in batch
            completed: Number of completed tasks
            failed: Number of failed tasks (after all retries)
            pending_retries: Number of tasks pending retry
            failed_tasks: List of failed task details (symbol, window, reason, attempts)
        """
        if "batches" not in self.state:
            self.state["batches"] = {}

        self.state["batches"][batch_id] = {
            "status": status,
            "total_tasks": total_tasks,
            "completed": completed,
            "failed": failed,
            "pending_retries": pending_retries,
            "failed_tasks": failed_tasks or [],
            "last_updated": datetime.now().isoformat(),
        }

        self._save_state()

        logger.info(
            f"Updated batch status: {batch_id} - {status} "
            f"({completed}/{total_tasks} completed, {failed} failed, {pending_retries} retries)",
            extra={
                "batch_id": batch_id,
                "status": status,
                "completed": completed,
                "failed": failed,
                "pending_retries": pending_retries,
            },
        )

    def record_task_failure(
        self,
        batch_id: str,
        task_id: str,
        symbol: str,
        window_label: str,
        reason: str,
        attempts: int,
    ) -> None:
        """Record a failed task for manual review.

        Args:
            batch_id: Unique batch identifier
            task_id: Task identifier
            symbol: Stock symbol
            window_label: Training window label
            reason: Failure reason
            attempts: Number of attempts made
        """
        if "batches" not in self.state:
            self.state["batches"] = {}

        if batch_id not in self.state["batches"]:
            self.state["batches"][batch_id] = {
                "status": "partial",
                "failed_tasks": [],
                "last_updated": datetime.now().isoformat(),
            }

        batch = self.state["batches"][batch_id]
        if "failed_tasks" not in batch:
            batch["failed_tasks"] = []

        batch["failed_tasks"].append(
            {
                "task_id": task_id,
                "symbol": symbol,
                "window_label": window_label,
                "reason": reason,
                "attempts": attempts,
                "timestamp": datetime.now().isoformat(),
            }
        )

        batch["last_updated"] = datetime.now().isoformat()
        self._save_state()

        logger.warning(
            f"Recorded task failure: {task_id} ({symbol}/{window_label}) "
            f"after {attempts} attempts: {reason}",
            extra={
                "batch_id": batch_id,
                "task_id": task_id,
                "symbol": symbol,
                "window_label": window_label,
                "attempts": attempts,
                "reason": reason,
            },
        )

    def get_failed_tasks(self, batch_id: str) -> list[dict[str, Any]]:
        """Get list of failed tasks for a batch.

        Args:
            batch_id: Unique batch identifier

        Returns:
            List of failed task dicts
        """
        batch = self.get_batch_status(batch_id)
        if not batch:
            return []
        return batch.get("failed_tasks", [])

    def clear_batch_status(self, batch_id: str) -> None:
        """Clear status for a specific batch.

        Args:
            batch_id: Batch identifier to clear
        """
        if "batches" in self.state and batch_id in self.state["batches"]:
            del self.state["batches"][batch_id]
            self._save_state()
            logger.info(f"Cleared batch status: {batch_id}")

    # US-024 Phase 6: Data quality metrics tracking

    def record_quality_metrics(
        self,
        symbol: str,
        data_type: str,
        metrics: dict[str, Any],
    ) -> None:
        """Record data quality metrics for a symbol.

        Args:
            symbol: Stock symbol
            data_type: Type of data ('historical' or 'sentiment')
            metrics: Quality metrics dict
        """
        if "quality_metrics" not in self.state:
            self.state["quality_metrics"] = {}

        if symbol not in self.state["quality_metrics"]:
            self.state["quality_metrics"][symbol] = {}

        self.state["quality_metrics"][symbol][data_type] = {
            **metrics,
            "last_scanned": datetime.now().isoformat(),
        }

        self._save_state()

        logger.info(
            f"Recorded quality metrics for {symbol}/{data_type}",
            extra={
                "symbol": symbol,
                "data_type": data_type,
                "metrics": metrics,
            },
        )

    def get_quality_metrics(
        self, symbol: str | None = None, data_type: str | None = None
    ) -> dict[str, Any]:
        """Get data quality metrics.

        Args:
            symbol: Stock symbol (optional, returns all if None)
            data_type: Data type filter ('historical' or 'sentiment')

        Returns:
            Dict with quality metrics
        """
        quality_metrics = self.state.get("quality_metrics", {})

        if symbol is None:
            return quality_metrics

        if symbol not in quality_metrics:
            return {}

        if data_type is None:
            return quality_metrics[symbol]

        return quality_metrics[symbol].get(data_type, {})

    def record_quality_alert(
        self,
        symbol: str,
        data_type: str,
        severity: str,
        metric: str,
        value: float,
        threshold: float,
        message: str,
    ) -> None:
        """Record a data quality alert.

        Args:
            symbol: Stock symbol
            data_type: Type of data ('historical' or 'sentiment')
            severity: Alert severity ('warning' or 'error')
            metric: Metric name
            value: Metric value
            threshold: Threshold exceeded
            message: Alert message
        """
        if "quality_alerts" not in self.state:
            self.state["quality_alerts"] = []

        alert = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "data_type": data_type,
            "severity": severity,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "message": message,
        }

        self.state["quality_alerts"].append(alert)

        # Keep only last 100 alerts
        if len(self.state["quality_alerts"]) > 100:
            self.state["quality_alerts"] = self.state["quality_alerts"][-100:]

        self._save_state()

        logger.warning(
            f"Quality alert: {symbol}/{data_type} - {message}",
            extra=alert,
        )

    def get_quality_alerts(
        self, symbol: str | None = None, severity: str | None = None
    ) -> list[dict[str, Any]]:
        """Get data quality alerts.

        Args:
            symbol: Stock symbol filter (optional)
            severity: Severity filter ('warning' or 'error')

        Returns:
            List of alert dicts
        """
        alerts = self.state.get("quality_alerts", [])

        # Filter by symbol if provided
        if symbol:
            alerts = [a for a in alerts if a["symbol"] == symbol]

        # Filter by severity if provided
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]

        return alerts

    # US-025: Model Validation Run tracking

    def record_validation_run(
        self,
        run_id: str,
        timestamp: str,
        symbols: list[str],
        date_range: dict[str, str],
        status: str,
        dryrun: bool,
        results: dict[str, Any],
    ) -> None:
        """Record a model validation run.

        Args:
            run_id: Unique validation run identifier
            timestamp: Run timestamp (ISO 8601)
            symbols: List of symbols validated
            date_range: Dict with start/end dates
            status: Run status (running/completed/failed)
            dryrun: Whether run was in dryrun mode
            results: Validation results dict
        """
        if "validation_runs" not in self.state:
            self.state["validation_runs"] = {}

        self.state["validation_runs"][run_id] = {
            "run_id": run_id,
            "timestamp": timestamp,
            "symbols": symbols,
            "date_range": date_range,
            "status": status,
            "dryrun": dryrun,
            "results": results,
        }

        self._save_state()

        logger.info(
            f"Recorded validation run: {run_id} ({status})",
            extra={
                "run_id": run_id,
                "status": status,
                "symbols": symbols,
                "dryrun": dryrun,
            },
        )

    def get_validation_run(self, run_id: str) -> dict[str, Any] | None:
        """Get validation run by ID.

        Args:
            run_id: Validation run identifier

        Returns:
            Dict with validation run data, or None if not found
        """
        validation_runs = self.state.get("validation_runs", {})
        return validation_runs.get(run_id)

    def get_validation_runs(
        self, status: str | None = None, dryrun: bool | None = None
    ) -> list[dict[str, Any]]:
        """Get all validation runs with optional filtering.

        Args:
            status: Filter by status (running/completed/failed)
            dryrun: Filter by dryrun mode

        Returns:
            List of validation run dicts
        """
        validation_runs = self.state.get("validation_runs", {})
        runs = list(validation_runs.values())

        # Filter by status if provided
        if status:
            runs = [r for r in runs if r.get("status") == status]

        # Filter by dryrun if provided
        if dryrun is not None:
            runs = [r for r in runs if r.get("dryrun") == dryrun]

        # Sort by timestamp (most recent first)
        runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

        return runs
