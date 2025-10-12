"""Monitoring and alerting service for runtime health tracking (Enterprise v2)."""

from __future__ import annotations

import gzip
import json
import smtplib
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Literal

import requests
from loguru import logger

from src.app.config import Settings

AlertSeverity = Literal["INFO", "WARNING", "CRITICAL"]


@dataclass
class Alert:
    """Alert event."""

    timestamp: str
    severity: AlertSeverity
    rule: str
    message: str
    context: dict[str, Any]
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HealthCheckResult:
    """Result of health check."""

    check_name: str
    status: Literal["OK", "WARNING", "ERROR"]
    message: str
    details: dict[str, Any] | None = None


@dataclass
class RollupStats:
    """Aggregated statistics for a metric."""

    min: float
    max: float
    avg: float
    count: int
    sum: float


@dataclass
class MetricRollup:
    """Metric rollup for a time interval."""

    interval_start: str
    interval_end: str
    interval_seconds: int
    metrics: dict[str, RollupStats]


@dataclass
class AckRecord:
    """Alert acknowledgement record."""

    rule: str
    acknowledged_at: str
    acknowledged_by: str
    ttl_seconds: int
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def is_expired(self) -> bool:
        """Check if acknowledgement has expired."""
        acked_time = datetime.fromisoformat(self.acknowledged_at)
        return (datetime.now() - acked_time).total_seconds() > self.ttl_seconds


@dataclass
class PerformanceMetric:
    """Performance metric (latency tracking)."""

    name: str
    timestamp: str
    value_ms: float
    context: dict[str, Any] = field(default_factory=dict)


class AlertDeliveryPlugin(ABC):
    """Abstract base class for alert delivery plugins."""

    def __init__(self, settings: Settings) -> None:
        """Initialize plugin with settings."""
        self.settings = settings
        self.enabled = False

    @abstractmethod
    def deliver(self, alert: Alert) -> bool:
        """Deliver alert via this channel.

        Args:
            alert: Alert to deliver

        Returns:
            True if delivery succeeded, False otherwise
        """
        pass


class EmailPlugin(AlertDeliveryPlugin):
    """Email alert delivery via SMTP."""

    def __init__(self, settings: Settings) -> None:
        """Initialize email plugin."""
        super().__init__(settings)
        self.enabled = settings.monitoring_enable_email_alerts
        self.smtp_host = settings.monitoring_email_smtp_host
        self.smtp_port = settings.monitoring_email_smtp_port
        self.smtp_user = settings.monitoring_email_smtp_user
        self.smtp_password = settings.monitoring_email_smtp_password
        self.from_addr = settings.monitoring_email_from
        self.to_addrs = settings.monitoring_email_to

    def deliver(self, alert: Alert) -> bool:
        """Send alert via SMTP email."""
        if not self.enabled or not self.to_addrs:
            return False

        try:
            subject = f"[{alert.severity}] {alert.rule}"
            body = f"""
SenseQuant Alert

Severity: {alert.severity}
Rule: {alert.rule}
Time: {alert.timestamp}

Message: {alert.message}

Context:
{json.dumps(alert.context, indent=2)}
"""
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(
                f"Alert delivered via email: {alert.rule}",
                extra={"component": "monitoring", "delivery_channel": "email"},
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to deliver alert via email: {e}",
                extra={
                    "component": "monitoring",
                    "delivery_channel": "email",
                    "alert_rule": alert.rule,
                    "error": str(e),
                },
            )
            return False


class SlackPlugin(AlertDeliveryPlugin):
    """Slack alert delivery via webhook."""

    def __init__(self, settings: Settings) -> None:
        """Initialize Slack plugin."""
        super().__init__(settings)
        self.enabled = settings.monitoring_enable_slack_alerts
        self.webhook_url = settings.monitoring_slack_webhook_url

    def deliver(self, alert: Alert) -> bool:
        """Post alert to Slack webhook."""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            severity_emoji = {
                "INFO": ":information_source:",
                "WARNING": ":warning:",
                "CRITICAL": ":rotating_light:",
            }

            payload = {
                "text": f"{severity_emoji.get(alert.severity, ':bell:')} *SenseQuant Alert*",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": f"{alert.severity}: {alert.rule}"},
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Severity:*\n{alert.severity}"},
                            {"type": "mrkdwn", "text": f"*Time:*\n{alert.timestamp}"},
                        ],
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Message:*\n{alert.message}"},
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Context:*\n```{json.dumps(alert.context, indent=2)}```",
                        },
                    },
                ],
            }

            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(
                f"Alert delivered via Slack: {alert.rule}",
                extra={"component": "monitoring", "delivery_channel": "slack"},
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to deliver alert via Slack: {e}",
                extra={
                    "component": "monitoring",
                    "delivery_channel": "slack",
                    "alert_rule": alert.rule,
                    "error": str(e),
                },
            )
            return False


class WebhookPlugin(AlertDeliveryPlugin):
    """Generic webhook alert delivery."""

    def __init__(self, settings: Settings) -> None:
        """Initialize webhook plugin."""
        super().__init__(settings)
        self.enabled = settings.monitoring_enable_webhook_alerts
        self.webhook_url = settings.monitoring_webhook_url
        self.headers = settings.monitoring_webhook_headers

    def deliver(self, alert: Alert) -> bool:
        """POST alert JSON to webhook."""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            payload = alert.to_dict()
            response = requests.post(
                self.webhook_url, json=payload, headers=self.headers, timeout=10
            )
            response.raise_for_status()

            logger.info(
                f"Alert delivered via webhook: {alert.rule}",
                extra={"component": "monitoring", "delivery_channel": "webhook"},
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to deliver alert via webhook: {e}",
                extra={
                    "component": "monitoring",
                    "delivery_channel": "webhook",
                    "alert_rule": alert.rule,
                    "error": str(e),
                },
            )
            return False


class MonitoringService:
    """Monitoring and alerting service for trading engine (Enterprise v2).

    Collects runtime metrics, evaluates alert rules, persists alerts/metrics,
    computes aggregated rollups, manages retention, tracks performance, and
    delivers alerts via multiple channels.

    Attributes:
        settings: Application settings
        heartbeat_timestamp: Last engine tick timestamp
        sentiment_failures: List of sentiment failure timestamps
        metrics_history: Recent metrics snapshots
        rollup_history: Aggregated metric rollups
        performance_metrics: Performance tracking (latency measurements)
        acknowledgements: Alert acknowledgement records
        delivery_plugins: Alert delivery channel plugins
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize MonitoringService.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.heartbeat_timestamp: datetime | None = None
        self.sentiment_failures: list[datetime] = []
        self.metrics_history: list[dict[str, Any]] = []
        self.last_alert_times: dict[str, datetime] = {}

        # v2: Aggregation
        self.rollup_history: list[MetricRollup] = []
        self.last_rollup_time: datetime | None = None

        # v2: Performance tracking
        self.performance_metrics: dict[str, list[PerformanceMetric]] = {}

        # v2: Acknowledgements
        self.acknowledgements: dict[str, AckRecord] = {}
        self.ack_log_path = Path("logs/alerts/acknowledgements.jsonl")

        # Create directories
        self.alerts_dir = Path("logs/alerts")
        self.metrics_dir = Path("data/monitoring")
        self.archive_dir = Path("data/monitoring/archive")
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # v2: Initialize delivery plugins
        self.delivery_plugins: list[AlertDeliveryPlugin] = []
        try:
            self.delivery_plugins.append(EmailPlugin(settings))
            self.delivery_plugins.append(SlackPlugin(settings))
            self.delivery_plugins.append(WebhookPlugin(settings))
        except Exception as e:
            logger.error(
                f"Failed to initialize delivery plugins: {e}",
                extra={"component": "monitoring", "error": str(e)},
            )

        # Load existing acknowledgements
        self._load_acknowledgements()

        # US-021 Phase 3: Initialize student model monitoring
        self.__init_student_monitoring__()

        # US-023: Initialize release tracking
        self.__init_release_tracking__()

        logger.info(
            "MonitoringService initialized (Enterprise v2)",
            extra={
                "component": "monitoring",
                "enabled": settings.enable_monitoring,
                "heartbeat_interval": settings.monitoring_heartbeat_interval,
                "aggregation_enabled": settings.monitoring_enable_aggregation,
                "performance_tracking": settings.monitoring_enable_performance_tracking,
                "delivery_plugins": len([p for p in self.delivery_plugins if p.enabled]),
                "student_monitoring_enabled": settings.student_monitoring_enabled,
            },
        )

    def record_tick(self, metrics: dict[str, Any]) -> None:
        """Record engine tick with metrics.

        Args:
            metrics: Current engine metrics
        """
        if not self.settings.enable_monitoring:
            return

        try:
            # Update heartbeat
            self.heartbeat_timestamp = datetime.now()

            # Add timestamp to metrics
            metrics["timestamp"] = self.heartbeat_timestamp.isoformat()

            # Store in history (keep last N)
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.settings.monitoring_max_raw_metrics:
                self.metrics_history.pop(0)

            # v2: Compute rollups if enabled
            if self.settings.monitoring_enable_aggregation:
                self._compute_rollups()

            # Evaluate alerts
            alerts = self.evaluate_alerts(metrics)

            # Emit alerts (with delivery)
            for alert in alerts:
                self.emit_alert(alert)

            # Persist metrics to disk
            self._persist_metrics(metrics)

            # Log tick
            logger.debug(
                "Monitoring tick recorded",
                extra={
                    "component": "monitoring",
                    "metrics_count": len(self.metrics_history),
                    "alerts_triggered": len(alerts),
                },
            )

        except Exception as e:
            logger.error(
                f"Failed to record monitoring tick: {e}",
                extra={"component": "monitoring", "error": str(e)},
            )

    def evaluate_alerts(self, metrics: dict[str, Any]) -> list[Alert]:
        """Evaluate alert rules against current metrics.

        Args:
            metrics: Current engine metrics

        Returns:
            List of triggered alerts
        """
        alerts: list[Alert] = []
        now = datetime.now()

        # Rule 1: Circuit breaker triggered
        if metrics.get("risk", {}).get("circuit_breaker_active", False):
            if not self._recently_alerted("circuit_breaker_triggered", minutes=15):
                if not self._is_acknowledged("circuit_breaker_triggered"):
                    alert = Alert(
                        timestamp=now.isoformat(),
                        severity="CRITICAL",
                        rule="circuit_breaker_triggered",
                        message="Circuit breaker active - daily loss limit exceeded",
                        context={
                            "daily_loss_pct": metrics.get("pnl", {}).get("daily_loss_pct", 0),
                            "threshold": self.settings.max_daily_loss_pct,
                            "open_positions": metrics.get("positions", {}).get("count", 0),
                        },
                    )
                    alerts.append(alert)
                    self.last_alert_times[alert.rule] = now

        # Rule 2: Daily loss approaching threshold
        daily_loss_pct = abs(metrics.get("pnl", {}).get("daily_loss_pct", 0))
        if daily_loss_pct >= self.settings.monitoring_daily_loss_alert_pct:
            if not self._recently_alerted("daily_loss_high", minutes=30):
                if not self._is_acknowledged("daily_loss_high"):
                    alert = Alert(
                        timestamp=now.isoformat(),
                        severity="WARNING",
                        rule="daily_loss_high",
                        message=f"Daily loss {daily_loss_pct:.2f}% approaching threshold",
                        context={
                            "daily_loss_pct": daily_loss_pct,
                            "threshold": self.settings.monitoring_daily_loss_alert_pct,
                            "circuit_breaker_threshold": self.settings.max_daily_loss_pct,
                        },
                    )
                    alerts.append(alert)
                    self.last_alert_times[alert.rule] = now

        # Rule 3: Sentiment provider failures
        sentiment_failures_1h = self._count_sentiment_failures(window_seconds=3600)
        if sentiment_failures_1h >= self.settings.monitoring_max_sentiment_failures:
            if not self._recently_alerted("sentiment_failures_high", minutes=60):
                if not self._is_acknowledged("sentiment_failures_high"):
                    alert = Alert(
                        timestamp=now.isoformat(),
                        severity="WARNING",
                        rule="sentiment_failures_high",
                        message=f"Sentiment provider failures: {sentiment_failures_1h}/hour",
                        context={
                            "failures_1h": sentiment_failures_1h,
                            "threshold": self.settings.monitoring_max_sentiment_failures,
                        },
                    )
                    alerts.append(alert)
                    self.last_alert_times[alert.rule] = now

        # Rule 4: Stale artifacts
        artifact_checks = self.check_artifact_freshness()
        for check in artifact_checks:
            if check.status == "WARNING" and not self._recently_alerted(
                f"stale_{check.check_name}", minutes=360
            ):
                if not self._is_acknowledged(f"stale_{check.check_name}"):
                    alert = Alert(
                        timestamp=now.isoformat(),
                        severity="WARNING",
                        rule=f"stale_{check.check_name}",
                        message=check.message,
                        context=check.details or {},
                    )
                    alerts.append(alert)
                    self.last_alert_times[alert.rule] = now

        # Rule 5: Heartbeat lapsed
        if self.heartbeat_timestamp:
            lapse_seconds = (now - self.heartbeat_timestamp).total_seconds()
            if lapse_seconds > self.settings.monitoring_heartbeat_lapse_seconds:
                if not self._recently_alerted("heartbeat_lapsed", minutes=5):
                    if not self._is_acknowledged("heartbeat_lapsed"):
                        alert = Alert(
                            timestamp=now.isoformat(),
                            severity="CRITICAL",
                            rule="heartbeat_lapsed",
                            message=f"Heartbeat lapsed for {lapse_seconds:.0f} seconds",
                            context={
                                "lapse_seconds": lapse_seconds,
                                "threshold": self.settings.monitoring_heartbeat_lapse_seconds,
                                "last_heartbeat": self.heartbeat_timestamp.isoformat(),
                            },
                        )
                        alerts.append(alert)
                        self.last_alert_times[alert.rule] = now

        # Rule 6: Breeze connectivity lost
        if not metrics.get("connectivity", {}).get("breeze_authenticated", True):
            if not self._recently_alerted("breeze_connectivity_lost", minutes=10):
                if not self._is_acknowledged("breeze_connectivity_lost"):
                    alert = Alert(
                        timestamp=now.isoformat(),
                        severity="CRITICAL",
                        rule="breeze_connectivity_lost",
                        message="Breeze API authentication lost",
                        context={
                            "last_api_call": metrics.get("connectivity", {}).get(
                                "last_api_call", "unknown"
                            )
                        },
                    )
                    alerts.append(alert)
                    self.last_alert_times[alert.rule] = now

        # v2: Rule 7: Performance degradation
        if self.settings.monitoring_enable_performance_tracking:
            perf_stats = self._aggregate_performance_stats()
            for metric_name, stats in perf_stats.items():
                if stats["avg"] > self.settings.monitoring_performance_alert_threshold_ms:
                    rule_name = f"performance_degradation_{metric_name}"
                    if not self._recently_alerted(rule_name, minutes=30):
                        if not self._is_acknowledged(rule_name):
                            alert = Alert(
                                timestamp=now.isoformat(),
                                severity="WARNING",
                                rule=rule_name,
                                message=f"Performance degradation: {metric_name} avg latency {stats['avg']:.2f}ms",
                                context={
                                    "metric": metric_name,
                                    "avg_latency_ms": stats["avg"],
                                    "max_latency_ms": stats["max"],
                                    "threshold_ms": self.settings.monitoring_performance_alert_threshold_ms,
                                    "sample_count": stats["count"],
                                },
                            )
                            alerts.append(alert)
                            self.last_alert_times[alert.rule] = now

        return alerts

    def emit_alert(self, alert: Alert) -> None:
        """Emit alert to structured log file and delivery channels.

        Args:
            alert: Alert to emit
        """
        # Write to JSONL log
        alert_date = datetime.now().strftime("%Y-%m-%d")
        alert_file = self.alerts_dir / f"{alert_date}.jsonl"

        try:
            with open(alert_file, "a") as f:
                json.dump(alert.to_dict(), f)
                f.write("\n")

            # Log with component="alert"
            logger.warning(
                f"[{alert.severity}] {alert.message}",
                extra={
                    "component": "alert",
                    "severity": alert.severity,
                    "rule": alert.rule,
                    "context": alert.context,
                },
            )

            # v2: Deliver via plugins (async)
            self._deliver_alert_async(alert)

        except Exception as e:
            logger.error(
                f"Failed to emit alert: {e}",
                extra={"component": "monitoring", "alert_rule": alert.rule, "error": str(e)},
            )

    def _deliver_alert_async(self, alert: Alert) -> None:
        """Deliver alert via plugins asynchronously."""

        def _deliver() -> None:
            for plugin in self.delivery_plugins:
                if plugin.enabled:
                    try:
                        plugin.deliver(alert)
                    except Exception as e:
                        logger.error(
                            f"Delivery plugin {plugin.__class__.__name__} failed: {e}",
                            extra={"component": "monitoring", "error": str(e)},
                        )

        # Run delivery in background thread
        thread = threading.Thread(target=_deliver, daemon=True)
        thread.start()

    def record_sentiment_failure(self) -> None:
        """Record sentiment provider failure timestamp."""
        self.sentiment_failures.append(datetime.now())

    def _count_sentiment_failures(self, window_seconds: int) -> int:
        """Count sentiment failures in time window."""
        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        return sum(1 for ts in self.sentiment_failures if ts >= cutoff)

    def _recently_alerted(self, rule: str, minutes: int) -> bool:
        """Check if alert was recently triggered."""
        if rule not in self.last_alert_times:
            return False
        elapsed = datetime.now() - self.last_alert_times[rule]
        return elapsed.total_seconds() < minutes * 60

    def check_artifact_freshness(self) -> list[HealthCheckResult]:
        """Check freshness of Teacher/Student artifacts."""
        results: list[HealthCheckResult] = []
        threshold_hours = self.settings.monitoring_artifact_staleness_hours

        # Check Teacher models
        teacher_dir = Path("data/teacher_models")
        if teacher_dir.exists():
            metadata_files = list(teacher_dir.glob("*_metadata.json"))
            if not metadata_files:
                results.append(
                    HealthCheckResult(
                        check_name="teacher_artifacts",
                        status="WARNING",
                        message="No Teacher model metadata found",
                        details={"expected_dir": str(teacher_dir)},
                    )
                )
            else:
                latest_file = max(metadata_files, key=lambda p: p.stat().st_mtime)
                age_hours = (datetime.now().timestamp() - latest_file.stat().st_mtime) / 3600
                if age_hours > threshold_hours:
                    results.append(
                        HealthCheckResult(
                            check_name="teacher_artifacts",
                            status="WARNING",
                            message=f"Teacher artifacts stale ({age_hours:.1f} hours old)",
                            details={"latest_file": str(latest_file), "age_hours": age_hours},
                        )
                    )
                else:
                    results.append(
                        HealthCheckResult(
                            check_name="teacher_artifacts",
                            status="OK",
                            message=f"Teacher artifacts fresh ({age_hours:.1f} hours old)",
                        )
                    )

        # Check Student model
        if self.settings.enable_student_inference and self.settings.student_model_path:
            student_path = Path(self.settings.student_model_path)
            if not student_path.exists():
                results.append(
                    HealthCheckResult(
                        check_name="student_model",
                        status="ERROR",
                        message="Student model file not found",
                        details={"path": str(student_path)},
                    )
                )
            else:
                age_hours = (datetime.now().timestamp() - student_path.stat().st_mtime) / 3600
                if age_hours > threshold_hours:
                    results.append(
                        HealthCheckResult(
                            check_name="student_model",
                            status="WARNING",
                            message=f"Student model stale ({age_hours:.1f} hours old)",
                            details={"path": str(student_path), "age_hours": age_hours},
                        )
                    )
                else:
                    results.append(
                        HealthCheckResult(
                            check_name="student_model",
                            status="OK",
                            message=f"Student model fresh ({age_hours:.1f} hours old)",
                        )
                    )

        return results

    def run_health_checks(self) -> list[HealthCheckResult]:
        """Run all health checks."""
        results: list[HealthCheckResult] = []

        # Artifact freshness
        results.extend(self.check_artifact_freshness())

        # Heartbeat check
        if self.heartbeat_timestamp:
            lapse_seconds = (datetime.now() - self.heartbeat_timestamp).total_seconds()
            if lapse_seconds > self.settings.monitoring_heartbeat_lapse_seconds:
                results.append(
                    HealthCheckResult(
                        check_name="heartbeat",
                        status="ERROR",
                        message=f"Heartbeat lapsed for {lapse_seconds:.0f} seconds",
                        details={"lapse_seconds": lapse_seconds},
                    )
                )
            else:
                results.append(
                    HealthCheckResult(
                        check_name="heartbeat",
                        status="OK",
                        message=f"Heartbeat healthy ({lapse_seconds:.0f}s ago)",
                    )
                )
        else:
            results.append(
                HealthCheckResult(
                    check_name="heartbeat",
                    status="WARNING",
                    message="No heartbeat recorded yet",
                )
            )

        # Sentiment failures
        failures_1h = self._count_sentiment_failures(window_seconds=3600)
        if failures_1h >= self.settings.monitoring_max_sentiment_failures:
            results.append(
                HealthCheckResult(
                    check_name="sentiment_provider",
                    status="WARNING",
                    message=f"High sentiment failure rate: {failures_1h}/hour",
                    details={"failures_1h": failures_1h},
                )
            )
        else:
            results.append(
                HealthCheckResult(
                    check_name="sentiment_provider",
                    status="OK",
                    message=f"Sentiment provider healthy ({failures_1h} failures/hour)",
                )
            )

        return results

    def get_active_alerts(self, hours: int = 24) -> list[Alert]:
        """Get active alerts from recent log files.

        Args:
            hours: Time window in hours

        Returns:
            List of active alerts
        """
        alerts: list[Alert] = []
        cutoff = datetime.now() - timedelta(hours=hours)

        # Read recent alert files
        for date_offset in range(hours // 24 + 2):
            date = datetime.now() - timedelta(days=date_offset)
            date_str = date.strftime("%Y-%m-%d")
            alert_file = self.alerts_dir / f"{date_str}.jsonl"

            if alert_file.exists():
                try:
                    with open(alert_file) as f:
                        for line in f:
                            alert_dict = json.loads(line)
                            alert_time = datetime.fromisoformat(alert_dict["timestamp"])
                            if alert_time >= cutoff:
                                alerts.append(Alert(**alert_dict))
                except Exception as e:
                    logger.error(
                        f"Failed to read alert file {alert_file}: {e}",
                        extra={"component": "monitoring", "error": str(e)},
                    )

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def _persist_metrics(self, metrics: dict[str, Any]) -> None:
        """Persist metrics to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.metrics_dir / f"metrics_{timestamp}.json"

        try:
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.error(
                f"Failed to persist metrics: {e}",
                extra={"component": "monitoring", "error": str(e)},
            )

    # v2: Metric Aggregation

    def _compute_rollups(self) -> None:
        """Compute aggregated rollups for metrics."""
        if not self.settings.monitoring_enable_aggregation:
            return

        now = datetime.now()
        interval_seconds = self.settings.monitoring_aggregation_interval_seconds

        # Check if it's time for a new rollup
        if self.last_rollup_time:
            elapsed = (now - self.last_rollup_time).total_seconds()
            if elapsed < interval_seconds:
                return

        # Compute rollup from metrics_history
        if not self.metrics_history:
            return

        interval_start = now - timedelta(seconds=interval_seconds)
        interval_metrics = [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) >= interval_start
        ]

        if not interval_metrics:
            return

        # Aggregate key metrics
        rollup_stats: dict[str, RollupStats] = {}

        # PnL aggregation
        pnl_values = [m.get("pnl", {}).get("daily", 0.0) for m in interval_metrics]
        if pnl_values:
            rollup_stats["pnl_daily"] = RollupStats(
                min=min(pnl_values),
                max=max(pnl_values),
                avg=sum(pnl_values) / len(pnl_values),
                count=len(pnl_values),
                sum=sum(pnl_values),
            )

        # Position count aggregation
        position_counts = [m.get("positions", {}).get("count", 0) for m in interval_metrics]
        if position_counts:
            rollup_stats["position_count"] = RollupStats(
                min=float(min(position_counts)),
                max=float(max(position_counts)),
                avg=sum(position_counts) / len(position_counts),
                count=len(position_counts),
                sum=float(sum(position_counts)),
            )

        # Daily loss % aggregation
        loss_pcts = [m.get("pnl", {}).get("daily_loss_pct", 0.0) for m in interval_metrics]
        if loss_pcts:
            rollup_stats["daily_loss_pct"] = RollupStats(
                min=min(loss_pcts),
                max=max(loss_pcts),
                avg=sum(loss_pcts) / len(loss_pcts),
                count=len(loss_pcts),
                sum=sum(loss_pcts),
            )

        # Performance metrics aggregation
        for metric_name in self.performance_metrics.keys():
            perf_values = [
                pm.value_ms
                for pm in self.performance_metrics[metric_name]
                if datetime.fromisoformat(pm.timestamp) >= interval_start
            ]
            if perf_values:
                rollup_stats[f"perf_{metric_name}"] = RollupStats(
                    min=min(perf_values),
                    max=max(perf_values),
                    avg=sum(perf_values) / len(perf_values),
                    count=len(perf_values),
                    sum=sum(perf_values),
                )

        # Create rollup record
        rollup = MetricRollup(
            interval_start=interval_start.isoformat(),
            interval_end=now.isoformat(),
            interval_seconds=interval_seconds,
            metrics=rollup_stats,
        )

        self.rollup_history.append(rollup)
        self.last_rollup_time = now

        # Keep last 288 rollups (24 hours at 5-min intervals)
        if len(self.rollup_history) > 288:
            self.rollup_history.pop(0)

        logger.debug(
            f"Computed metric rollup with {len(rollup_stats)} metrics",
            extra={"component": "monitoring", "rollup_count": len(self.rollup_history)},
        )

    def get_aggregated_metrics(
        self, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[MetricRollup]:
        """Get aggregated metrics for time range.

        Args:
            start_time: Start of time range (default: 24h ago)
            end_time: End of time range (default: now)

        Returns:
            List of metric rollups
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now()

        return [
            r
            for r in self.rollup_history
            if start_time <= datetime.fromisoformat(r.interval_start) <= end_time
        ]

    # v2: Retention Management

    def archive_old_metrics(self) -> None:
        """Archive old raw metrics to compressed JSON."""
        if not self.metrics_history:
            return

        try:
            # Archive metrics older than 1 day
            cutoff = datetime.now() - timedelta(days=1)
            old_metrics = [
                m for m in self.metrics_history if datetime.fromisoformat(m["timestamp"]) < cutoff
            ]

            if not old_metrics:
                return

            # Write to compressed archive
            archive_date = datetime.now().strftime("%Y-%m-%d")
            archive_file = self.archive_dir / f"metrics_{archive_date}.json.gz"

            with gzip.open(archive_file, "wt") as f:
                json.dump(old_metrics, f, indent=2)

            # Remove from memory
            self.metrics_history = [
                m for m in self.metrics_history if datetime.fromisoformat(m["timestamp"]) >= cutoff
            ]

            logger.info(
                f"Archived {len(old_metrics)} old metrics to {archive_file}",
                extra={"component": "monitoring", "archived_count": len(old_metrics)},
            )

        except Exception as e:
            logger.error(
                f"Failed to archive metrics: {e}",
                extra={"component": "monitoring", "error": str(e)},
            )

    def cleanup_old_archives(self) -> None:
        """Remove archives older than retention period."""
        try:
            cutoff = datetime.now() - timedelta(days=self.settings.monitoring_max_archive_days)
            removed_count = 0

            for archive_file in self.archive_dir.glob("metrics_*.json.gz"):
                if archive_file.stat().st_mtime < cutoff.timestamp():
                    archive_file.unlink()
                    removed_count += 1

            if removed_count > 0:
                logger.info(
                    f"Cleaned up {removed_count} old archive files",
                    extra={"component": "monitoring", "removed_count": removed_count},
                )

        except Exception as e:
            logger.error(
                f"Failed to cleanup old archives: {e}",
                extra={"component": "monitoring", "error": str(e)},
            )

    # v2: Performance Tracking

    def record_performance_metric(
        self, name: str, value_ms: float, context: dict[str, Any] | None = None
    ) -> None:
        """Record a performance metric (latency).

        Args:
            name: Metric name (e.g., "tick_latency", "sentiment_latency")
            value_ms: Latency value in milliseconds
            context: Optional context (e.g., {"strategy": "intraday"})
        """
        if not self.settings.monitoring_enable_performance_tracking:
            return

        metric = PerformanceMetric(
            name=name,
            timestamp=datetime.now().isoformat(),
            value_ms=value_ms,
            context=context or {},
        )

        if name not in self.performance_metrics:
            self.performance_metrics[name] = []

        self.performance_metrics[name].append(metric)

        # Keep last 1000 measurements per metric
        if len(self.performance_metrics[name]) > 1000:
            self.performance_metrics[name].pop(0)

    def _aggregate_performance_stats(self) -> dict[str, dict[str, float]]:
        """Aggregate performance statistics (last hour)."""
        stats: dict[str, dict[str, float]] = {}
        cutoff = datetime.now() - timedelta(hours=1)

        for metric_name, measurements in self.performance_metrics.items():
            recent = [
                m.value_ms for m in measurements if datetime.fromisoformat(m.timestamp) >= cutoff
            ]
            if recent:
                stats[metric_name] = {
                    "min": min(recent),
                    "max": max(recent),
                    "avg": sum(recent) / len(recent),
                    "count": len(recent),
                }

        return stats

    # v2: Alert Acknowledgement

    def acknowledge_alert(
        self, rule: str, acknowledged_by: str = "operator", reason: str | None = None
    ) -> None:
        """Acknowledge an alert to prevent re-notification.

        Args:
            rule: Alert rule to acknowledge
            acknowledged_by: Operator name/ID
            reason: Optional reason for acknowledgement
        """
        ack = AckRecord(
            rule=rule,
            acknowledged_at=datetime.now().isoformat(),
            acknowledged_by=acknowledged_by,
            ttl_seconds=self.settings.monitoring_ack_ttl_seconds,
            reason=reason,
        )

        self.acknowledgements[rule] = ack
        self._save_acknowledgement(ack)

        logger.info(
            f"Alert acknowledged: {rule}",
            extra={
                "component": "monitoring",
                "rule": rule,
                "acknowledged_by": acknowledged_by,
            },
        )

    def clear_acknowledgement(self, rule: str) -> None:
        """Clear acknowledgement for an alert rule.

        Args:
            rule: Alert rule to clear
        """
        if rule in self.acknowledgements:
            del self.acknowledgements[rule]
            logger.info(
                f"Acknowledgement cleared: {rule}",
                extra={"component": "monitoring", "rule": rule},
            )

    def _is_acknowledged(self, rule: str) -> bool:
        """Check if alert rule is currently acknowledged."""
        if rule not in self.acknowledgements:
            return False

        ack = self.acknowledgements[rule]
        if ack.is_expired():
            # Auto-clear expired acknowledgements
            del self.acknowledgements[rule]
            return False

        return True

    def _load_acknowledgements(self) -> None:
        """Load acknowledgements from log file."""
        if not self.ack_log_path.exists():
            return

        try:
            with open(self.ack_log_path) as f:
                for line in f:
                    ack_dict = json.loads(line)
                    ack = AckRecord(**ack_dict)
                    if not ack.is_expired():
                        self.acknowledgements[ack.rule] = ack

            logger.debug(
                f"Loaded {len(self.acknowledgements)} active acknowledgements",
                extra={"component": "monitoring"},
            )

        except Exception as e:
            logger.error(
                f"Failed to load acknowledgements: {e}",
                extra={"component": "monitoring", "error": str(e)},
            )

    def _save_acknowledgement(self, ack: AckRecord) -> None:
        """Save acknowledgement to log file."""
        try:
            with open(self.ack_log_path, "a") as f:
                json.dump(ack.to_dict(), f)
                f.write("\n")
        except Exception as e:
            logger.error(
                f"Failed to save acknowledgement: {e}",
                extra={"component": "monitoring", "error": str(e)},
            )

    # US-021 Phase 3: Student Model Monitoring

    def __init_student_monitoring__(self) -> None:
        """Initialize student model monitoring structures (US-021 Phase 3)."""
        self.student_metrics_history: list[dict[str, Any]] = []
        self.student_baseline_metrics: dict[str, float] = {}
        self.student_alerts: list[Alert] = []
        self.student_monitoring_dir = Path("data/monitoring/student_model")
        self.student_monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.last_student_alert_time: datetime | None = None

        # Load baseline metrics if available
        baseline_path = self.student_monitoring_dir / "baseline_metrics.json"
        if baseline_path.exists():
            try:
                with open(baseline_path) as f:
                    self.student_baseline_metrics = json.load(f)
                logger.info(
                    "Loaded student model baseline metrics",
                    extra={"component": "monitoring", "baseline": self.student_baseline_metrics},
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load student baseline metrics: {e}",
                    extra={"component": "monitoring"},
                )

    def record_student_prediction(
        self,
        symbol: str,
        prediction: str,
        probability: float,
        confidence: float,
        actual_outcome: str | None = None,
        model_version: str | None = None,
    ) -> None:
        """Record student model prediction for monitoring (US-021 Phase 3).

        Args:
            symbol: Trading symbol
            prediction: Student prediction (BUY/SELL/HOLD)
            probability: Prediction probability
            confidence: Prediction confidence
            actual_outcome: Actual outcome (if known)
            model_version: Model version identifier
        """
        if not self.settings.student_monitoring_enabled:
            return

        metric = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
            "actual_outcome": actual_outcome,
            "model_version": model_version or "unknown",
        }

        self.student_metrics_history.append(metric)

        # Persist to disk
        metrics_file = self.student_monitoring_dir / "predictions.jsonl"
        try:
            with open(metrics_file, "a") as f:
                json.dump(metric, f)
                f.write("\n")
        except Exception as e:
            logger.error(
                f"Failed to persist student prediction metric: {e}",
                extra={"component": "monitoring"},
            )

        # Trim history to window size
        window_cutoff = datetime.now() - timedelta(
            hours=self.settings.student_monitoring_window_hours
        )
        self.student_metrics_history = [
            m
            for m in self.student_metrics_history
            if datetime.fromisoformat(m["timestamp"]) > window_cutoff
        ]

    def evaluate_student_model_performance(self) -> dict[str, Any]:
        """Evaluate student model performance over rolling window (US-021 Phase 3).

        Returns:
            Dictionary with performance metrics:
            {
                "total_predictions": int,
                "precision": float,
                "hit_ratio": float,
                "avg_confidence": float,
                "model_version": str,
                "window_start": str,
                "window_end": str
            }
        """
        if not self.settings.student_monitoring_enabled:
            return {}

        # Filter to predictions with known outcomes
        predictions_with_outcomes = [
            m for m in self.student_metrics_history if m.get("actual_outcome") is not None
        ]

        if len(predictions_with_outcomes) < self.settings.student_monitoring_min_samples:
            logger.debug(
                f"Insufficient samples for student model evaluation: "
                f"{len(predictions_with_outcomes)} < {self.settings.student_monitoring_min_samples}",
                extra={"component": "monitoring"},
            )
            return {
                "total_predictions": len(predictions_with_outcomes),
                "insufficient_samples": True,
                "min_required": self.settings.student_monitoring_min_samples,
            }

        # Calculate precision (correct predictions / total predictions)
        correct_predictions = sum(
            1 for m in predictions_with_outcomes if m["prediction"] == m["actual_outcome"]
        )
        precision = correct_predictions / len(predictions_with_outcomes)

        # Calculate hit ratio (predictions with confidence >= threshold that were correct)
        high_confidence_predictions = [
            m
            for m in predictions_with_outcomes
            if m["confidence"] >= self.settings.student_model_confidence_threshold
        ]
        if high_confidence_predictions:
            high_confidence_correct = sum(
                1 for m in high_confidence_predictions if m["prediction"] == m["actual_outcome"]
            )
            hit_ratio = high_confidence_correct / len(high_confidence_predictions)
        else:
            hit_ratio = 0.0

        # Average confidence
        avg_confidence = sum(m["confidence"] for m in predictions_with_outcomes) / len(
            predictions_with_outcomes
        )

        # Get model version (most recent)
        model_version = (
            predictions_with_outcomes[-1]["model_version"]
            if predictions_with_outcomes
            else "unknown"
        )

        # Window bounds
        timestamps = [datetime.fromisoformat(m["timestamp"]) for m in predictions_with_outcomes]
        window_start = min(timestamps).isoformat() if timestamps else ""
        window_end = max(timestamps).isoformat() if timestamps else ""

        metrics = {
            "total_predictions": len(predictions_with_outcomes),
            "correct_predictions": correct_predictions,
            "precision": precision,
            "hit_ratio": hit_ratio,
            "avg_confidence": avg_confidence,
            "model_version": model_version,
            "window_start": window_start,
            "window_end": window_end,
            "high_confidence_count": len(high_confidence_predictions),
            "evaluated_at": datetime.now().isoformat(),
        }

        # Persist metrics
        metrics_snapshot_file = (
            self.student_monitoring_dir
            / f"metrics_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            with open(metrics_snapshot_file, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.error(
                f"Failed to persist student metrics snapshot: {e}",
                extra={"component": "monitoring"},
            )

        return metrics

    def check_student_model_degradation(self) -> list[Alert]:
        """Check for student model performance degradation (US-021 Phase 3).

        Compares current rolling metrics against baseline and generates alerts
        if degradation exceeds thresholds.

        Returns:
            List of alerts triggered
        """
        if not self.settings.student_monitoring_enabled:
            return []

        alerts = []

        # Evaluate current performance
        current_metrics = self.evaluate_student_model_performance()

        if current_metrics.get("insufficient_samples"):
            return []

        # Check cooldown period
        if self.last_student_alert_time:
            hours_since_last_alert = (
                datetime.now() - self.last_student_alert_time
            ).total_seconds() / 3600
            if hours_since_last_alert < self.settings.student_monitoring_alert_cooldown_hours:
                logger.debug(
                    f"Student model alert in cooldown: {hours_since_last_alert:.1f}h "
                    f"< {self.settings.student_monitoring_alert_cooldown_hours}h",
                    extra={"component": "monitoring"},
                )
                return []

        # Compare with baseline
        if not self.student_baseline_metrics:
            logger.warning(
                "No student model baseline metrics available, cannot check degradation",
                extra={"component": "monitoring"},
            )
            return []

        baseline_precision = self.student_baseline_metrics.get("precision", 0.0)
        baseline_hit_ratio = self.student_baseline_metrics.get("hit_ratio", 0.0)

        current_precision = current_metrics["precision"]
        current_hit_ratio = current_metrics["hit_ratio"]

        # Calculate drops
        precision_drop = baseline_precision - current_precision
        hit_ratio_drop = baseline_hit_ratio - current_hit_ratio

        # Check precision degradation
        if precision_drop > self.settings.student_monitoring_precision_drop_threshold:
            alert = Alert(
                timestamp=datetime.now().isoformat(),
                severity="CRITICAL",
                rule="student_model_precision_degradation",
                message=(
                    f"Student model precision dropped {precision_drop:.2%} "
                    f"(baseline: {baseline_precision:.2%}, current: {current_precision:.2%})"
                ),
                context={
                    "baseline_precision": baseline_precision,
                    "current_precision": current_precision,
                    "precision_drop": precision_drop,
                    "threshold": self.settings.student_monitoring_precision_drop_threshold,
                    "model_version": current_metrics["model_version"],
                    "total_predictions": current_metrics["total_predictions"],
                    "window_hours": self.settings.student_monitoring_window_hours,
                },
            )
            alerts.append(alert)

        # Check hit ratio degradation
        if hit_ratio_drop > self.settings.student_monitoring_hit_ratio_drop_threshold:
            alert = Alert(
                timestamp=datetime.now().isoformat(),
                severity="CRITICAL",
                rule="student_model_hit_ratio_degradation",
                message=(
                    f"Student model hit ratio dropped {hit_ratio_drop:.2%} "
                    f"(baseline: {baseline_hit_ratio:.2%}, current: {current_hit_ratio:.2%})"
                ),
                context={
                    "baseline_hit_ratio": baseline_hit_ratio,
                    "current_hit_ratio": current_hit_ratio,
                    "hit_ratio_drop": hit_ratio_drop,
                    "threshold": self.settings.student_monitoring_hit_ratio_drop_threshold,
                    "model_version": current_metrics["model_version"],
                    "total_predictions": current_metrics["total_predictions"],
                    "window_hours": self.settings.student_monitoring_window_hours,
                },
            )
            alerts.append(alert)

        if alerts:
            self.last_student_alert_time = datetime.now()
            self.student_alerts.extend(alerts)

            # Persist alerts to JSONL (following _emit_alert pattern)
            alert_date = datetime.now().strftime("%Y-%m-%d")
            alert_file = self.alerts_dir / f"{alert_date}.jsonl"
            try:
                with open(alert_file, "a") as f:
                    for alert in alerts:
                        json.dump(alert.to_dict(), f)
                        f.write("\n")

                        # Log with component="alert"
                        logger.warning(
                            f"[{alert.severity}] {alert.message}",
                            extra={
                                "component": "alert",
                                "severity": alert.severity,
                                "rule": alert.rule,
                                "context": alert.context,
                            },
                        )

                        # Deliver via plugins (async)
                        self._deliver_alert_async(alert)
            except Exception as e:
                logger.error(
                    f"Failed to persist student monitoring alerts: {e}",
                    extra={"component": "monitoring", "error": str(e)},
                )

            logger.warning(
                f"Student model degradation detected: {len(alerts)} alert(s) triggered",
                extra={
                    "component": "monitoring",
                    "precision_drop": precision_drop,
                    "hit_ratio_drop": hit_ratio_drop,
                },
            )

        return alerts

    def set_student_baseline_metrics(self, metrics: dict[str, float]) -> None:
        """Set baseline metrics for student model monitoring (US-021 Phase 3).

        Args:
            metrics: Baseline metrics dictionary with precision, hit_ratio, etc.
        """
        self.student_baseline_metrics = metrics

        # Persist to disk
        baseline_path = self.student_monitoring_dir / "baseline_metrics.json"
        try:
            with open(baseline_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(
                "Saved student model baseline metrics",
                extra={"component": "monitoring", "baseline": metrics},
            )
        except Exception as e:
            logger.error(
                f"Failed to persist student baseline metrics: {e}",
                extra={"component": "monitoring"},
            )

    def get_student_monitoring_status(self) -> dict[str, Any]:
        """Get current student model monitoring status (US-021 Phase 3).

        Returns:
            Dictionary with monitoring status and latest metrics
        """
        if not self.settings.student_monitoring_enabled:
            return {"enabled": False}

        current_metrics = self.evaluate_student_model_performance()

        status = {
            "enabled": True,
            "baseline_metrics": self.student_baseline_metrics,
            "current_metrics": current_metrics,
            "recent_alerts": [alert.to_dict() for alert in self.student_alerts[-10:]],
            "monitoring_window_hours": self.settings.student_monitoring_window_hours,
            "alert_thresholds": {
                "precision_drop": self.settings.student_monitoring_precision_drop_threshold,
                "hit_ratio_drop": self.settings.student_monitoring_hit_ratio_drop_threshold,
            },
            "auto_rollback_enabled": self.settings.student_auto_rollback_enabled,
        }

        return status

    # ========================================================================
    # Release Tracking & Heightened Monitoring (US-023)
    # ========================================================================

    def __init_release_tracking__(self) -> None:
        """Initialize release tracking structures (US-023)."""
        self.releases_dir = Path("data/monitoring/releases")
        self.releases_dir.mkdir(parents=True, exist_ok=True)
        self.active_release_info: dict[str, Any] | None = None
        self._load_active_release()

    def _load_active_release(self) -> None:
        """Load active release from disk."""
        active_file = self.releases_dir / "active_release.yaml"
        if active_file.exists():
            import yaml

            with open(active_file) as f:
                self.active_release_info = yaml.safe_load(f)
            logger.debug(
                f"Loaded active release: {self.active_release_info.get('release_id', 'unknown')}"
            )

    def register_release(
        self,
        release_id: str,
        manifest_path: str | Path,
        heightened_hours: int = 48,
    ) -> None:
        """Register new production release and activate heightened monitoring.

        Args:
            release_id: Unique release identifier (e.g., release_20251012_190000)
            manifest_path: Path to release manifest YAML file
            heightened_hours: Duration of heightened monitoring in hours (default: 48)
        """
        from datetime import datetime, timedelta

        import yaml

        now = datetime.now()
        heightened_end = now + timedelta(hours=heightened_hours)

        self.active_release_info = {
            "release_id": release_id,
            "deployment_timestamp": now.isoformat(),
            "manifest_path": str(manifest_path),
            "heightened_monitoring_active": True,
            "heightened_monitoring_end": heightened_end.isoformat(),
            "heightened_hours": heightened_hours,
        }

        # Persist to disk
        active_file = self.releases_dir / "active_release.yaml"
        with open(active_file, "w") as f:
            yaml.dump(self.active_release_info, f, default_flow_style=False)

        # Also save to release history
        history_file = self.releases_dir / f"{release_id}.yaml"
        with open(history_file, "w") as f:
            yaml.dump(self.active_release_info, f, default_flow_style=False)

        logger.warning(
            f"🚀 Production release deployed: {release_id}",
            extra={
                "component": "monitoring",
                "release_id": release_id,
                "heightened_monitoring_hours": heightened_hours,
                "heightened_end": heightened_end.isoformat(),
            },
        )

    def get_active_release(self) -> dict[str, Any] | None:
        """Get currently active release info.

        Returns:
            Dict with release info or None if no active release
        """
        from datetime import datetime

        if not self.active_release_info:
            return None

        # Check if still in heightened period
        heightened_end = datetime.fromisoformat(
            self.active_release_info["heightened_monitoring_end"]
        )

        if datetime.now() >= heightened_end and self.active_release_info.get(
            "heightened_monitoring_active", False
        ):
            # Transition to normal monitoring
            self.active_release_info["heightened_monitoring_active"] = False
            self._save_active_release()
            logger.info(
                f"Heightened monitoring period ended for {self.active_release_info['release_id']}",
                extra={"component": "monitoring"},
            )

        return self.active_release_info

    def _save_active_release(self) -> None:
        """Save active release state to disk."""
        if self.active_release_info:
            import yaml

            active_file = self.releases_dir / "active_release.yaml"
            with open(active_file, "w") as f:
                yaml.dump(self.active_release_info, f, default_flow_style=False)

    def is_in_heightened_monitoring(self) -> bool:
        """Check if currently in heightened monitoring period.

        Returns:
            True if in heightened monitoring, False otherwise
        """
        release = self.get_active_release()
        if not release:
            return False
        return release.get("heightened_monitoring_active", False)

    def get_alert_thresholds(self) -> dict[str, float]:
        """Get alert thresholds based on current monitoring mode.

        Returns:
            Dict of alert thresholds (stricter during heightened monitoring)
        """
        if self.is_in_heightened_monitoring():
            # Heightened monitoring: 5% degradation triggers alert
            return {
                "intraday_hit_ratio_drop": 0.05,
                "swing_precision_drop": 0.05,
                "intraday_sharpe_drop": 0.15,
                "swing_max_drawdown": 0.05,
            }
        else:
            # Normal monitoring: 10% degradation triggers alert
            return {
                "intraday_hit_ratio_drop": 0.10,
                "swing_precision_drop": 0.10,
                "intraday_sharpe_drop": 0.25,
                "swing_max_drawdown": 0.10,
            }

    def get_monitoring_window_hours(self) -> dict[str, int]:
        """Get monitoring window sizes based on current monitoring mode.

        Returns:
            Dict of rolling window hours for different strategies
        """
        if self.is_in_heightened_monitoring():
            # Heightened monitoring: shorter windows
            return {
                "intraday": 6,  # 6 hours vs normal 24h
                "swing": 24,  # 24 hours vs normal 90 days
            }
        else:
            # Normal monitoring: standard windows
            return {
                "intraday": 24,
                "swing": 2160,  # 90 days in hours
            }
