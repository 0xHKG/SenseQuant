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

    def __init__(self, state_file: Path | None = None):
        """Initialize state manager.

        Args:
            state_file: Path to state JSON file (default: data/state/state.json)
        """
        if state_file is None:
            state_file = Path("data/state/state.json")

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

    # US-026: Statistical Validation tracking

    def record_statistical_validation(
        self,
        run_id: str,
        timestamp: str,
        status: str,
        walk_forward_results: dict[str, Any],
        bootstrap_results: dict[str, Any],
        benchmark_comparison: dict[str, Any],
    ) -> None:
        """Record statistical validation results (US-026).

        Args:
            run_id: Validation run ID
            timestamp: ISO 8601 timestamp
            status: Status (completed/failed/skipped)
            walk_forward_results: Walk-forward CV results
            bootstrap_results: Bootstrap test results
            benchmark_comparison: Benchmark comparison results
        """
        if "statistical_validations" not in self.state:
            self.state["statistical_validations"] = {}

        self.state["statistical_validations"][run_id] = {
            "run_id": run_id,
            "timestamp": timestamp,
            "status": status,
            "walk_forward_results": walk_forward_results,
            "bootstrap_results": bootstrap_results,
            "benchmark_comparison": benchmark_comparison,
        }

        # Update last benchmark comparison
        if benchmark_comparison:
            self.state["last_benchmark_comparison"] = {
                "run_id": run_id,
                "timestamp": timestamp,
                "benchmark": benchmark_comparison.get("benchmark"),
                "alpha": benchmark_comparison.get("alpha"),
                "beta": benchmark_comparison.get("beta"),
                "information_ratio": benchmark_comparison.get("information_ratio"),
            }

        self._save_state()
        logger.info(f"Recorded statistical validation: {run_id} ({status})")

    def get_statistical_validation(self, run_id: str) -> dict[str, Any] | None:
        """Get statistical validation results by run ID.

        Args:
            run_id: Validation run ID

        Returns:
            Statistical validation dict or None if not found
        """
        statistical_validations = self.state.get("statistical_validations", {})
        return statistical_validations.get(run_id)

    def get_last_benchmark_comparison(self) -> dict[str, Any] | None:
        """Get last benchmark comparison results.

        Returns:
            Last benchmark comparison dict or None if not found
        """
        return self.state.get("last_benchmark_comparison")

    def get_statistical_validations(self, status: str | None = None) -> list[dict[str, Any]]:
        """Get all statistical validations with optional filtering.

        Args:
            status: Optional status filter (completed/failed/skipped)

        Returns:
            List of statistical validation dicts
        """
        statistical_validations = self.state.get("statistical_validations", {})
        validations = list(statistical_validations.values())

        # Filter by status if provided
        if status:
            validations = [v for v in validations if v.get("status") == status]

        # Sort by timestamp (most recent first)
        validations.sort(key=lambda v: v.get("timestamp", ""), reverse=True)

        return validations

    # US-027: Deployment History Tracking

    def record_deployment(
        self,
        release_id: str,
        environment: str,
        timestamp: str,
        status: str,
        artifacts: list[str],
        rollback: bool,
        smoke_test_passed: bool,
        deployed_by: str,
    ) -> None:
        """Record deployment event (US-027).

        Args:
            release_id: Unique release identifier (e.g., "v1.2.3" or timestamp)
            environment: Target environment ("prod", "staging")
            timestamp: ISO timestamp of deployment
            status: Deployment status ("success", "failed", "rolled_back")
            artifacts: List of deployed artifact names
            rollback: Whether this was a rollback operation
            smoke_test_passed: Whether smoke tests passed
            deployed_by: User or system that initiated deployment
        """
        if "deployments" not in self.state:
            self.state["deployments"] = []

        deployment_record = {
            "release_id": release_id,
            "environment": environment,
            "timestamp": timestamp,
            "status": status,
            "artifacts": artifacts,
            "rollback": rollback,
            "smoke_test_passed": smoke_test_passed,
            "deployed_by": deployed_by,
        }

        self.state["deployments"].append(deployment_record)

        # Update last deployment pointer
        if status == "success" and not rollback:
            if "last_deployments" not in self.state:
                self.state["last_deployments"] = {}

            self.state["last_deployments"][environment] = {
                "release_id": release_id,
                "timestamp": timestamp,
                "artifacts": artifacts,
            }

        self._save_state()
        logger.info(
            f"Recorded deployment: {release_id} to {environment} ({status})",
            extra={
                "release_id": release_id,
                "environment": environment,
                "status": status,
                "rollback": rollback,
            },
        )

    def get_deployment_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent deployment history (US-027).

        Args:
            limit: Maximum number of deployments to return

        Returns:
            List of deployment records, most recent first
        """
        deployments = self.state.get("deployments", [])

        # Sort by timestamp descending
        sorted_deployments = sorted(
            deployments,
            key=lambda d: d.get("timestamp", ""),
            reverse=True,
        )

        return sorted_deployments[:limit]

    def get_last_deployment(self, environment: str = "prod") -> dict[str, Any] | None:
        """Get last successful deployment for environment (US-027).

        Args:
            environment: Target environment ("prod", "staging")

        Returns:
            Last deployment record or None if no deployments
        """
        last_deployments = self.state.get("last_deployments", {})
        return last_deployments.get(environment)

    def get_deployments_by_environment(self, environment: str) -> list[dict[str, Any]]:
        """Get deployment history for specific environment (US-027).

        Args:
            environment: Target environment ("prod", "staging")

        Returns:
            List of deployments for environment, most recent first
        """
        deployments = self.state.get("deployments", [])

        # Filter by environment
        env_deployments = [d for d in deployments if d.get("environment") == environment]

        # Sort by timestamp descending
        env_deployments.sort(key=lambda d: d.get("timestamp", ""), reverse=True)

        return env_deployments

    # US-028: Candidate Run Tracking

    def record_candidate_run(
        self,
        run_id: str,
        timestamp: str,
        status: str,
        training: dict[str, Any],
        validation: dict[str, Any],
        statistical_tests: dict[str, Any],
        artifacts: dict[str, str],
    ) -> None:
        """Record candidate run for promotion tracking (US-028).

        Args:
            run_id: Unique run identifier (e.g., "live_candidate_20251012_153000")
            timestamp: ISO timestamp of run
            status: Run status ("ready-for-review", "approved", "rejected", "failed")
            training: Training metrics dict (symbols, teacher/student metrics)
            validation: Validation results dict
            statistical_tests: Statistical test results dict
            artifacts: Artifact paths dict (model_dir, audit_dir, manifest)
        """
        if "candidate_runs" not in self.state:
            self.state["candidate_runs"] = []

        candidate_record = {
            "run_id": run_id,
            "timestamp": timestamp,
            "status": status,
            "training": training,
            "validation": validation,
            "statistical_tests": statistical_tests,
            "artifacts": artifacts,
        }

        self.state["candidate_runs"].append(candidate_record)

        # Update latest candidate pointer
        if status in ["ready-for-review", "approved"]:
            self.state["latest_candidate"] = {
                "run_id": run_id,
                "timestamp": timestamp,
                "status": status,
            }

        self._save_state()
        logger.info(
            f"Recorded candidate run: {run_id} ({status})",
            extra={"run_id": run_id, "status": status},
        )

    def get_latest_candidate_run(self) -> dict[str, Any] | None:
        """Get latest candidate run (US-028).

        Returns:
            Latest candidate record or None if no candidates
        """
        latest = self.state.get("latest_candidate")
        if not latest:
            return None

        # Return full record
        candidate_runs = self.state.get("candidate_runs", [])
        for candidate in candidate_runs:
            if candidate.get("run_id") == latest["run_id"]:
                return candidate

        return None

    def get_candidate_runs(self, status: str | None = None) -> list[dict[str, Any]]:
        """Get all candidate runs with optional status filter (US-028).

        Args:
            status: Optional status filter ("ready-for-review", "approved", "rejected", "failed")

        Returns:
            List of candidate records, most recent first
        """
        candidate_runs = self.state.get("candidate_runs", [])

        # Filter by status if provided
        if status:
            candidate_runs = [c for c in candidate_runs if c.get("status") == status]

        # Sort by timestamp descending
        candidate_runs.sort(key=lambda c: c.get("timestamp", ""), reverse=True)

        return candidate_runs

    def approve_candidate_run(self, run_id: str, approved_by: str) -> None:
        """Approve candidate run for deployment (US-028).

        Args:
            run_id: Candidate run identifier
            approved_by: User who approved
        """
        candidate_runs = self.state.get("candidate_runs", [])

        for candidate in candidate_runs:
            if candidate.get("run_id") == run_id:
                candidate["status"] = "approved"
                candidate["approved_by"] = approved_by
                candidate["approved_at"] = datetime.now().isoformat()

                # Update latest candidate
                self.state["latest_candidate"] = {
                    "run_id": run_id,
                    "timestamp": candidate["timestamp"],
                    "status": "approved",
                }

                self._save_state()
                logger.info(
                    f"Approved candidate run: {run_id}",
                    extra={"run_id": run_id, "approved_by": approved_by},
                )
                return

        logger.warning(f"Candidate run not found: {run_id}")

    # =====================================================================
    # US-029: Market Data Ingestion Tracking
    # =====================================================================

    def record_market_data_fetch(
        self,
        data_type: str,
        symbol: str,
        timestamp: str,
        stats: dict[str, Any],
    ) -> None:
        """Record market data fetch metadata (US-029).

        Args:
            data_type: Type of market data ("order_book", "options", "macro")
            symbol: Stock symbol or indicator name
            timestamp: Fetch timestamp (ISO format)
            stats: Fetch statistics (fetched, cached, failed, etc.)
        """
        if "market_data" not in self.state:
            self.state["market_data"] = {}

        if data_type not in self.state["market_data"]:
            self.state["market_data"][data_type] = {}

        if symbol not in self.state["market_data"][data_type]:
            self.state["market_data"][data_type][symbol] = {
                "first_fetch": timestamp,
                "last_fetch": timestamp,
                "total_fetches": 0,
                "total_errors": 0,
                "history": [],
            }

        symbol_state = self.state["market_data"][data_type][symbol]

        # Update summary
        symbol_state["last_fetch"] = timestamp
        symbol_state["total_fetches"] += 1
        symbol_state["total_errors"] += stats.get("failed", 0)

        # Append to history (keep last 100 entries)
        symbol_state["history"].append(
            {
                "timestamp": timestamp,
                "stats": stats,
            }
        )

        # Trim history to last 100 entries
        if len(symbol_state["history"]) > 100:
            symbol_state["history"] = symbol_state["history"][-100:]

        self._save_state()

        logger.debug(
            f"Recorded {data_type} fetch: {symbol}",
            extra={
                "data_type": data_type,
                "symbol": symbol,
                "timestamp": timestamp,
                "stats": stats,
            },
        )

    def get_last_market_data_fetch(
        self,
        data_type: str,
        symbol: str,
    ) -> dict[str, Any] | None:
        """Get last fetch metadata for market data (US-029).

        Args:
            data_type: Type of market data ("order_book", "options", "macro")
            symbol: Stock symbol or indicator name

        Returns:
            Dict with last fetch metadata, or None if never fetched
        """
        market_data = self.state.get("market_data", {})
        data_type_state = market_data.get(data_type, {})
        symbol_state = data_type_state.get(symbol)

        if not symbol_state or not symbol_state.get("history"):
            return None

        # Return last history entry
        last_entry = symbol_state["history"][-1]
        return {
            "timestamp": last_entry["timestamp"],
            "stats": last_entry["stats"],
            "total_fetches": symbol_state["total_fetches"],
            "total_errors": symbol_state["total_errors"],
        }

    def get_market_data_fetch_stats(
        self,
        data_type: str,
    ) -> dict[str, Any]:
        """Get aggregated fetch statistics for market data type (US-029).

        Args:
            data_type: Type of market data ("order_book", "options", "macro")

        Returns:
            Dict with aggregated statistics across all symbols
        """
        market_data = self.state.get("market_data", {})
        data_type_state = market_data.get(data_type, {})

        if not data_type_state:
            return {
                "total_symbols": 0,
                "total_fetches": 0,
                "total_errors": 0,
                "symbols": [],
            }

        total_fetches = 0
        total_errors = 0
        symbols = []

        for symbol, symbol_state in data_type_state.items():
            total_fetches += symbol_state.get("total_fetches", 0)
            total_errors += symbol_state.get("total_errors", 0)
            symbols.append(
                {
                    "symbol": symbol,
                    "last_fetch": symbol_state.get("last_fetch"),
                    "total_fetches": symbol_state.get("total_fetches", 0),
                    "total_errors": symbol_state.get("total_errors", 0),
                }
            )

        return {
            "total_symbols": len(symbols),
            "total_fetches": total_fetches,
            "total_errors": total_errors,
            "symbols": symbols,
        }

    # =====================================================================
    # US-029 Phase 2: Feature Coverage Tracking
    # =====================================================================

    def record_feature_coverage(
        self,
        symbol: str,
        date_range: tuple[str, str],
        feature_types: list[str],
    ) -> None:
        """Record feature coverage for symbol/date range (US-029 Phase 2).

        Args:
            symbol: Stock symbol
            date_range: Tuple of (start_date, end_date) in ISO format
            feature_types: List of feature types generated (e.g., ["order_book", "options", "macro"])
        """
        if "feature_coverage" not in self.state:
            self.state["feature_coverage"] = {}

        if symbol not in self.state["feature_coverage"]:
            self.state["feature_coverage"][symbol] = {
                "first_coverage": datetime.now().isoformat(),
                "last_coverage": datetime.now().isoformat(),
                "total_coverage_updates": 0,
                "date_ranges": [],
            }

        symbol_coverage = self.state["feature_coverage"][symbol]

        # Update summary
        symbol_coverage["last_coverage"] = datetime.now().isoformat()
        symbol_coverage["total_coverage_updates"] += 1

        # Append date range entry
        symbol_coverage["date_ranges"].append(
            {
                "start_date": date_range[0],
                "end_date": date_range[1],
                "feature_types": feature_types,
                "recorded_at": datetime.now().isoformat(),
            }
        )

        # Trim history to last 100 entries
        if len(symbol_coverage["date_ranges"]) > 100:
            symbol_coverage["date_ranges"] = symbol_coverage["date_ranges"][-100:]

        self._save_state()

        logger.debug(
            f"Recorded feature coverage for {symbol}: {feature_types}",
            extra={
                "symbol": symbol,
                "date_range": date_range,
                "feature_types": feature_types,
            },
        )

    def get_feature_coverage(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        """Get feature coverage stats for symbol (US-029 Phase 2).

        Args:
            symbol: Stock symbol

        Returns:
            Dict with feature coverage statistics
        """
        feature_coverage = self.state.get("feature_coverage", {})
        symbol_coverage = feature_coverage.get(symbol)

        if not symbol_coverage:
            return {
                "order_book_dates": 0,
                "options_dates": 0,
                "macro_dates": 0,
                "total_coverage_updates": 0,
                "last_coverage": None,
            }

        # Count unique dates by feature type
        order_book_dates = set()
        options_dates = set()
        macro_dates = set()

        for date_range_entry in symbol_coverage.get("date_ranges", []):
            feature_types = date_range_entry.get("feature_types", [])
            start_date = date_range_entry.get("start_date", "")
            end_date = date_range_entry.get("end_date", "")

            if "order_book" in feature_types:
                order_book_dates.add((start_date, end_date))
            if "options" in feature_types:
                options_dates.add((start_date, end_date))
            if "macro" in feature_types:
                macro_dates.add((start_date, end_date))

        return {
            "order_book_dates": len(order_book_dates),
            "options_dates": len(options_dates),
            "macro_dates": len(macro_dates),
            "total_coverage_updates": symbol_coverage.get("total_coverage_updates", 0),
            "last_coverage": symbol_coverage.get("last_coverage"),
        }

    # =====================================================================
    # US-029 Phase 4: Provider Metrics Tracking
    # =====================================================================

    def record_provider_metrics(
        self,
        provider_name: str,
        success: bool,
        retries: int = 0,
        latency_ms: float | None = None,
        error_message: str | None = None,
    ) -> None:
        """Record provider-level fetch metrics (US-029 Phase 4).

        Tracks success/error counts, retry attempts, and latency for each provider.

        Args:
            provider_name: Provider name ("order_book", "options", "macro")
            success: Whether the fetch succeeded
            retries: Number of retry attempts
            latency_ms: Fetch latency in milliseconds
            error_message: Error message if failed
        """
        if "provider_metrics" not in self.state:
            self.state["provider_metrics"] = {}

        if provider_name not in self.state["provider_metrics"]:
            self.state["provider_metrics"][provider_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_retries": 0,
                "last_success_timestamp": None,
                "last_error_timestamp": None,
                "last_error_message": None,
                "avg_latency_ms": 0.0,
                "max_latency_ms": 0.0,
            }

        provider_state = self.state["provider_metrics"][provider_name]

        # Update counts
        provider_state["total_requests"] += 1
        if success:
            provider_state["successful_requests"] += 1
            provider_state["last_success_timestamp"] = datetime.now().isoformat()
        else:
            provider_state["failed_requests"] += 1
            provider_state["last_error_timestamp"] = datetime.now().isoformat()
            if error_message:
                provider_state["last_error_message"] = error_message

        # Update retry count
        provider_state["total_retries"] += retries

        # Update latency stats (only for requests with latency data)
        if latency_ms is not None:
            # Track count of requests with latency separately for accurate averaging
            if "latency_count" not in provider_state:
                provider_state["latency_count"] = 0

            current_avg = provider_state["avg_latency_ms"]
            latency_count = provider_state["latency_count"]

            # Calculate new running average (only for requests with latency)
            new_avg = ((current_avg * latency_count) + latency_ms) / (latency_count + 1)
            provider_state["avg_latency_ms"] = round(new_avg, 2)
            provider_state["latency_count"] = latency_count + 1

            # Update max latency
            if latency_ms > provider_state["max_latency_ms"]:
                provider_state["max_latency_ms"] = round(latency_ms, 2)

        self._save_state()

        logger.debug(
            f"Recorded provider metrics: {provider_name}",
            extra={
                "provider": provider_name,
                "success": success,
                "retries": retries,
                "latency_ms": latency_ms,
            },
        )

    def get_provider_stats(self, provider_name: str) -> dict[str, Any]:
        """Get aggregated statistics for a provider (US-029 Phase 4).

        Args:
            provider_name: Provider name ("order_book", "options", "macro")

        Returns:
            Dict with provider statistics
        """
        provider_metrics = self.state.get("provider_metrics", {})
        provider_state = provider_metrics.get(provider_name)

        if not provider_state:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 0.0,
                "total_retries": 0,
                "avg_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "last_success_timestamp": None,
                "last_error_timestamp": None,
                "last_error_message": None,
            }

        # Calculate success rate
        total = provider_state["total_requests"]
        success_rate = (provider_state["successful_requests"] / total * 100) if total > 0 else 0.0

        return {
            "total_requests": provider_state["total_requests"],
            "successful_requests": provider_state["successful_requests"],
            "failed_requests": provider_state["failed_requests"],
            "success_rate": round(success_rate, 2),
            "total_retries": provider_state["total_retries"],
            "avg_latency_ms": provider_state["avg_latency_ms"],
            "max_latency_ms": provider_state["max_latency_ms"],
            "last_success_timestamp": provider_state["last_success_timestamp"],
            "last_error_timestamp": provider_state["last_error_timestamp"],
            "last_error_message": provider_state["last_error_message"],
        }

    def get_all_provider_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all providers (US-029 Phase 4).

        Returns:
            Dict mapping provider names to their statistics
        """
        provider_metrics = self.state.get("provider_metrics", {})

        stats = {}
        for provider_name in provider_metrics:
            stats[provider_name] = self.get_provider_stats(provider_name)

        return stats

    # =========================================================================
    # US-029 Phase 5: Streaming Heartbeat Tracking
    # =========================================================================

    def record_streaming_heartbeat(
        self,
        stream_type: str,
        symbols: list[str],
        stats: dict[str, Any],
        buffer_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record streaming heartbeat for health monitoring (US-029 Phase 5/5b).

        Args:
            stream_type: Type of stream (e.g., "order_book", "trades")
            symbols: List of symbols being streamed
            stats: Current streaming statistics (updates, errors, last_heartbeat)
            buffer_metadata: Optional buffer metadata dict with:
                - buffer_lengths: dict[str, int] - Current buffer size per symbol
                - last_snapshot_times: dict[str, str] - Last snapshot timestamp per symbol
                - total_capacity: int - Total buffer capacity
        """
        from datetime import datetime

        if "streaming" not in self.state:
            self.state["streaming"] = {}

        if stream_type not in self.state["streaming"]:
            self.state["streaming"][stream_type] = {}

        stream_state = self.state["streaming"][stream_type]

        # Update heartbeat metadata
        stream_state["last_heartbeat"] = datetime.now().isoformat()
        stream_state["symbols"] = symbols
        stream_state["update_count"] = stats.get("updates", 0)
        stream_state["error_count"] = stats.get("errors", 0)
        stream_state["is_healthy"] = True  # Will be checked by monitoring

        # Store buffer metadata (US-029 Phase 5b)
        if buffer_metadata:
            stream_state["buffer_lengths"] = buffer_metadata.get("buffer_lengths", {})
            stream_state["last_snapshot_times"] = buffer_metadata.get("last_snapshot_times", {})
            stream_state["total_capacity"] = buffer_metadata.get("total_capacity", 0)

            # Calculate buffer utilization percentage
            if stream_state["total_capacity"] > 0:
                total_used = sum(stream_state["buffer_lengths"].values())
                total_max = stream_state["total_capacity"] * len(symbols)
                stream_state["buffer_utilization_pct"] = round(
                    (total_used / total_max * 100) if total_max > 0 else 0, 2
                )

        self._save_state()

    def get_streaming_health(self, stream_type: str) -> dict[str, Any]:
        """Get streaming health status (US-029 Phase 5).

        Args:
            stream_type: Type of stream to check

        Returns:
            Dict with health status: last_heartbeat, is_healthy, time_since_heartbeat_seconds
        """
        from datetime import datetime

        streaming_state = self.state.get("streaming", {})

        if stream_type not in streaming_state:
            return {
                "exists": False,
                "is_healthy": False,
                "reason": "Stream not found",
            }

        stream = streaming_state[stream_type]
        last_heartbeat_str = stream.get("last_heartbeat")

        if not last_heartbeat_str:
            return {
                "exists": True,
                "is_healthy": False,
                "reason": "No heartbeat recorded",
            }

        # Calculate time since last heartbeat
        last_heartbeat = datetime.fromisoformat(last_heartbeat_str)
        now = datetime.now()
        time_since_seconds = (now - last_heartbeat).total_seconds()

        # Import config to get timeout threshold
        from src.app.config import settings

        timeout_seconds = settings.streaming_heartbeat_timeout_seconds
        is_healthy = time_since_seconds < timeout_seconds

        health_dict = {
            "exists": True,
            "is_healthy": is_healthy,
            "last_heartbeat": last_heartbeat_str,
            "time_since_heartbeat_seconds": round(time_since_seconds, 2),
            "timeout_threshold_seconds": timeout_seconds,
            "symbols": stream.get("symbols", []),
            "update_count": stream.get("update_count", 0),
            "error_count": stream.get("error_count", 0),
        }

        # Include buffer metadata if available (US-029 Phase 5b)
        if "buffer_lengths" in stream:
            health_dict["buffer_lengths"] = stream["buffer_lengths"]
        if "last_snapshot_times" in stream:
            health_dict["last_snapshot_times"] = stream["last_snapshot_times"]
        if "total_capacity" in stream:
            health_dict["total_capacity"] = stream["total_capacity"]
        if "buffer_utilization_pct" in stream:
            health_dict["buffer_utilization_pct"] = stream["buffer_utilization_pct"]

        return health_dict

    def get_all_streaming_health(self) -> dict[str, dict[str, Any]]:
        """Get health status for all active streams (US-029 Phase 5).

        Returns:
            Dict mapping stream types to their health status
        """
        streaming_state = self.state.get("streaming", {})

        health_stats = {}
        for stream_type in streaming_state:
            health_stats[stream_type] = self.get_streaming_health(stream_type)

        return health_stats
