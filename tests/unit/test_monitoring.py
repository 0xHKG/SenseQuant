"""Unit tests for Monitoring Service."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.app.config import Settings
from src.services.monitoring import Alert, HealthCheckResult, MonitoringService


@pytest.fixture
def monitoring_service(tmp_path: Path) -> MonitoringService:
    """Create monitoring service with temp directories."""
    settings = Settings()  # type: ignore[call-arg]
    service = MonitoringService(settings)

    # Override directories to use temp paths
    service.alerts_dir = tmp_path / "alerts"
    service.metrics_dir = tmp_path / "metrics"
    service.alerts_dir.mkdir(parents=True, exist_ok=True)
    service.metrics_dir.mkdir(parents=True, exist_ok=True)

    return service


def test_monitoring_initialization(monitoring_service: MonitoringService) -> None:
    """Test MonitoringService initialization."""
    assert monitoring_service.heartbeat_timestamp is None
    assert monitoring_service.sentiment_failures == []
    assert monitoring_service.metrics_history == []


def test_record_tick(monitoring_service: MonitoringService) -> None:
    """Test recording engine tick."""
    metrics = {
        "positions": {"count": 2},
        "pnl": {"daily": 1000.0},
        "risk": {"circuit_breaker_active": False},
    }

    monitoring_service.record_tick(metrics)

    assert monitoring_service.heartbeat_timestamp is not None
    assert len(monitoring_service.metrics_history) == 1


def test_circuit_breaker_alert(monitoring_service: MonitoringService) -> None:
    """Test circuit breaker triggers alert."""
    # Clear any loaded acknowledgements
    monitoring_service.acknowledgements.clear()

    metrics = {
        "positions": {"count": 3},
        "pnl": {"daily": -5000.0, "daily_loss_pct": -5.2},
        "risk": {"circuit_breaker_active": True},
        "connectivity": {"breeze_authenticated": True},
    }

    alerts = monitoring_service.evaluate_alerts(metrics)

    # Should generate circuit breaker alert
    assert len(alerts) > 0
    assert any(a.rule == "circuit_breaker_triggered" for a in alerts)


def test_daily_loss_alert(monitoring_service: MonitoringService) -> None:
    """Test daily loss approaching threshold triggers alert."""
    # Clear any loaded acknowledgements
    monitoring_service.acknowledgements.clear()

    metrics = {
        "positions": {"count": 1},
        "pnl": {"daily": -4500.0, "daily_loss_pct": -4.5},
        "risk": {"circuit_breaker_active": False},
        "connectivity": {"breeze_authenticated": True},
    }

    alerts = monitoring_service.evaluate_alerts(metrics)

    # Should generate daily loss warning
    assert any(a.rule == "daily_loss_high" for a in alerts)


def test_sentiment_failure_tracking(monitoring_service: MonitoringService) -> None:
    """Test sentiment failure tracking."""
    # Record multiple failures
    for _ in range(6):
        monitoring_service.record_sentiment_failure()

    failures = monitoring_service._count_sentiment_failures(3600)
    assert failures == 6


def test_sentiment_failure_alert(monitoring_service: MonitoringService) -> None:
    """Test sentiment failures trigger alert."""
    # Record failures exceeding threshold
    for _ in range(10):
        monitoring_service.record_sentiment_failure()

    metrics = {
        "positions": {"count": 0},
        "pnl": {"daily": 0.0, "daily_loss_pct": 0.0},
        "risk": {"circuit_breaker_active": False},
        "connectivity": {"breeze_authenticated": True},
    }

    alerts = monitoring_service.evaluate_alerts(metrics)

    assert any(a.rule == "sentiment_failures_high" for a in alerts)


def test_alert_deduplication(monitoring_service: MonitoringService) -> None:
    """Test alert deduplication prevents spam."""
    # Clear any loaded acknowledgements
    monitoring_service.acknowledgements.clear()

    metrics = {
        "positions": {"count": 0},
        "pnl": {"daily": -4500.0, "daily_loss_pct": -4.5},
        "risk": {"circuit_breaker_active": False},
        "connectivity": {"breeze_authenticated": True},
    }

    # First evaluation should trigger alert
    alerts1 = monitoring_service.evaluate_alerts(metrics)
    assert any(a.rule == "daily_loss_high" for a in alerts1)

    # Second evaluation within window should NOT trigger (deduplicated)
    alerts2 = monitoring_service.evaluate_alerts(metrics)
    assert not any(a.rule == "daily_loss_high" for a in alerts2)


def test_emit_alert(monitoring_service: MonitoringService, tmp_path: Path) -> None:
    """Test alert emission to JSONL file."""
    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="CRITICAL",
        rule="test_rule",
        message="Test alert",
        context={"test": "value"},
    )

    monitoring_service.emit_alert(alert)

    # Check alert file was created
    alert_files = list(monitoring_service.alerts_dir.glob("*.jsonl"))
    assert len(alert_files) == 1

    # Verify content
    with open(alert_files[0]) as f:
        content = f.read().strip()
        assert "test_rule" in content


def test_persist_metrics(monitoring_service: MonitoringService) -> None:
    """Test metrics persistence."""
    # Add some metrics and record tick to persist
    metrics = {
        "positions": {"count": 2},
    }

    monitoring_service.record_tick(metrics)

    # Check metrics file was created
    metric_files = list(monitoring_service.metrics_dir.glob("metrics_*.json"))
    assert len(metric_files) >= 1


def test_artifact_freshness_check(monitoring_service: MonitoringService) -> None:
    """Test artifact freshness checking."""
    results = monitoring_service.check_artifact_freshness()

    # Should return results (may be warnings if artifacts don't exist)
    # In test environment, artifacts may not exist, so just verify method works
    assert isinstance(results, list)
    # All results should be HealthCheckResult objects
    assert all(isinstance(r, HealthCheckResult) for r in results)


def test_health_checks(monitoring_service: MonitoringService) -> None:
    """Test running all health checks."""
    results = monitoring_service.run_health_checks()

    assert len(results) > 0
    assert all(isinstance(r, HealthCheckResult) for r in results)


def test_get_active_alerts(monitoring_service: MonitoringService) -> None:
    """Test retrieving active alerts."""
    # Emit some alerts
    alert1 = Alert(
        timestamp=datetime.now().isoformat(),
        severity="WARNING",
        rule="test1",
        message="Test 1",
        context={},
    )
    alert2 = Alert(
        timestamp=datetime.now().isoformat(),
        severity="CRITICAL",
        rule="test2",
        message="Test 2",
        context={},
    )

    monitoring_service.emit_alert(alert1)
    monitoring_service.emit_alert(alert2)

    # Retrieve active alerts
    active = monitoring_service.get_active_alerts(hours=1)

    assert len(active) == 2
    assert any(a.rule == "test1" for a in active)
    assert any(a.rule == "test2" for a in active)


def test_metrics_history_limit(monitoring_service: MonitoringService) -> None:
    """Test metrics history is limited to 100 entries."""
    # Add 150 metrics
    for i in range(150):
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "positions": {"count": i},
        }
        monitoring_service.record_tick(metrics)

    # Should keep only last 100
    assert len(monitoring_service.metrics_history) == 100


def test_heartbeat_lapse_detection(monitoring_service: MonitoringService) -> None:
    """Test heartbeat lapse detection."""
    # Set old heartbeat
    monitoring_service.heartbeat_timestamp = datetime.now() - timedelta(minutes=10)

    metrics = {
        "positions": {"count": 0},
        "pnl": {"daily": 0.0, "daily_loss_pct": 0.0},
        "risk": {"circuit_breaker_active": False},
        "connectivity": {"breeze_authenticated": True},
    }

    alerts = monitoring_service.evaluate_alerts(metrics)

    # Should detect heartbeat lapse
    assert any(a.rule == "heartbeat_lapsed" for a in alerts)


# ============================================================================
# US-013 ENTERPRISE FEATURES - METRIC AGGREGATION/ROLLUPS
# ============================================================================


def test_compute_rollups_creates_rollup(monitoring_service: MonitoringService) -> None:
    """Test _compute_rollups() creates rollup records."""
    monitoring_service.settings.monitoring_enable_aggregation = True
    monitoring_service.settings.monitoring_aggregation_interval_seconds = 1

    # Add metrics
    for i in range(5):
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "positions": {"count": i},
            "pnl": {"daily": i * 100.0, "daily_loss_pct": i * 0.1},
        }
        monitoring_service.metrics_history.append(metrics)

    # Compute rollups
    monitoring_service._compute_rollups()

    assert len(monitoring_service.rollup_history) == 1
    rollup = monitoring_service.rollup_history[0]
    assert rollup.interval_seconds == 1
    assert "pnl_daily" in rollup.metrics
    assert "position_count" in rollup.metrics


def test_compute_rollups_rollup_stats(monitoring_service: MonitoringService) -> None:
    """Test RollupStats computation (min/max/avg/count/sum)."""
    monitoring_service.settings.monitoring_enable_aggregation = True
    monitoring_service.settings.monitoring_aggregation_interval_seconds = 1

    # Add known values
    pnl_values = [100.0, 200.0, 300.0, 400.0, 500.0]
    for pnl in pnl_values:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "positions": {"count": 2},
            "pnl": {"daily": pnl, "daily_loss_pct": -1.0},
        }
        monitoring_service.metrics_history.append(metrics)

    monitoring_service._compute_rollups()

    rollup = monitoring_service.rollup_history[0]
    pnl_stats = rollup.metrics["pnl_daily"]

    assert pnl_stats.min == 100.0
    assert pnl_stats.max == 500.0
    assert pnl_stats.avg == 300.0
    assert pnl_stats.count == 5
    assert pnl_stats.sum == 1500.0


def test_compute_rollups_respects_interval(monitoring_service: MonitoringService) -> None:
    """Test rollup only computed after interval elapses."""
    monitoring_service.settings.monitoring_enable_aggregation = True
    monitoring_service.settings.monitoring_aggregation_interval_seconds = 3600  # 1 hour

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "positions": {"count": 1},
        "pnl": {"daily": 100.0},
    }
    monitoring_service.metrics_history.append(metrics)

    # First compute should work
    monitoring_service._compute_rollups()
    assert len(monitoring_service.rollup_history) == 1

    # Immediate second compute should not create new rollup
    monitoring_service._compute_rollups()
    assert len(monitoring_service.rollup_history) == 1


def test_rollup_history_limit(monitoring_service: MonitoringService) -> None:
    """Test rollup history limited to 288 max."""
    from src.services.monitoring import MetricRollup

    monitoring_service.settings.monitoring_enable_aggregation = True
    monitoring_service.settings.monitoring_aggregation_interval_seconds = 1

    # Add 288 rollups (at limit)
    for _ in range(288):
        monitoring_service.rollup_history.append(
            MetricRollup(
                interval_start=datetime.now().isoformat(),
                interval_end=datetime.now().isoformat(),
                interval_seconds=300,
                metrics={},
            )
        )

    # Verify we have 288
    assert len(monitoring_service.rollup_history) == 288

    # Trigger rollup computation which should add one and remove oldest
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "positions": {"count": 1},
        "pnl": {"daily": 100.0},
    }
    monitoring_service.metrics_history.append(metrics)

    # Set last_rollup_time to None to force a new rollup
    monitoring_service.last_rollup_time = None
    monitoring_service._compute_rollups()

    # Should still be at 288 (oldest one removed, new one added)
    assert len(monitoring_service.rollup_history) == 288


def test_get_aggregated_metrics_time_range(monitoring_service: MonitoringService) -> None:
    """Test get_aggregated_metrics() with time range filter."""
    from src.services.monitoring import MetricRollup, RollupStats

    # Add rollups at different times
    now = datetime.now()
    rollup1 = MetricRollup(
        interval_start=(now - timedelta(hours=2)).isoformat(),
        interval_end=(now - timedelta(hours=1, minutes=55)).isoformat(),
        interval_seconds=300,
        metrics={"pnl_daily": RollupStats(min=100, max=200, avg=150, count=5, sum=750)},
    )
    rollup2 = MetricRollup(
        interval_start=(now - timedelta(minutes=30)).isoformat(),
        interval_end=(now - timedelta(minutes=25)).isoformat(),
        interval_seconds=300,
        metrics={"pnl_daily": RollupStats(min=300, max=400, avg=350, count=3, sum=1050)},
    )

    monitoring_service.rollup_history = [rollup1, rollup2]

    # Query last hour
    results = monitoring_service.get_aggregated_metrics(
        start_time=now - timedelta(hours=1), end_time=now
    )

    # Should only get rollup2
    assert len(results) == 1
    assert results[0].metrics["pnl_daily"].avg == 350


def test_get_aggregated_metrics_default_range(monitoring_service: MonitoringService) -> None:
    """Test get_aggregated_metrics() with default 24h range."""
    from src.services.monitoring import MetricRollup, RollupStats

    now = datetime.now()
    rollup = MetricRollup(
        interval_start=(now - timedelta(hours=12)).isoformat(),
        interval_end=(now - timedelta(hours=11, minutes=55)).isoformat(),
        interval_seconds=300,
        metrics={"pnl_daily": RollupStats(min=100, max=200, avg=150, count=5, sum=750)},
    )
    monitoring_service.rollup_history = [rollup]

    # Query with default range (24h)
    results = monitoring_service.get_aggregated_metrics()

    assert len(results) == 1


# ============================================================================
# US-013 ENTERPRISE FEATURES - PERFORMANCE METRICS
# ============================================================================


def test_record_performance_metric(monitoring_service: MonitoringService) -> None:
    """Test record_performance_metric() stores latency."""
    monitoring_service.settings.monitoring_enable_performance_tracking = True

    monitoring_service.record_performance_metric("tick_latency", 15.5, {"strategy": "intraday"})

    assert "tick_latency" in monitoring_service.performance_metrics
    assert len(monitoring_service.performance_metrics["tick_latency"]) == 1

    metric = monitoring_service.performance_metrics["tick_latency"][0]
    assert metric.name == "tick_latency"
    assert metric.value_ms == 15.5
    assert metric.context == {"strategy": "intraday"}


def test_record_performance_metric_disabled(monitoring_service: MonitoringService) -> None:
    """Test performance tracking respects enable flag."""
    monitoring_service.settings.monitoring_enable_performance_tracking = False

    monitoring_service.record_performance_metric("tick_latency", 15.5)

    assert "tick_latency" not in monitoring_service.performance_metrics


def test_performance_metric_limit(monitoring_service: MonitoringService) -> None:
    """Test performance metrics limited to 1000 per metric."""
    monitoring_service.settings.monitoring_enable_performance_tracking = True

    # Add 1100 metrics
    for i in range(1100):
        monitoring_service.record_performance_metric("test_metric", float(i))

    # Should keep only last 1000
    assert len(monitoring_service.performance_metrics["test_metric"]) == 1000
    # Should have removed oldest (0-99)
    assert monitoring_service.performance_metrics["test_metric"][0].value_ms == 100.0


def test_aggregate_performance_stats(monitoring_service: MonitoringService) -> None:
    """Test _aggregate_performance_stats() computes stats."""
    monitoring_service.settings.monitoring_enable_performance_tracking = True

    # Add performance metrics
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    for latency in latencies:
        monitoring_service.record_performance_metric("api_call", latency)

    stats = monitoring_service._aggregate_performance_stats()

    assert "api_call" in stats
    assert stats["api_call"]["min"] == 10.0
    assert stats["api_call"]["max"] == 50.0
    assert stats["api_call"]["avg"] == 30.0
    assert stats["api_call"]["count"] == 5


def test_aggregate_performance_stats_one_hour_window(monitoring_service: MonitoringService) -> None:
    """Test performance stats only include last hour."""
    monitoring_service.settings.monitoring_enable_performance_tracking = True
    from src.services.monitoring import PerformanceMetric

    # Add old metric (2 hours ago)
    old_metric = PerformanceMetric(
        name="old_metric",
        timestamp=(datetime.now() - timedelta(hours=2)).isoformat(),
        value_ms=100.0,
        context={},
    )
    monitoring_service.performance_metrics["old_metric"] = [old_metric]

    # Add recent metric
    monitoring_service.record_performance_metric("recent_metric", 50.0)

    stats = monitoring_service._aggregate_performance_stats()

    # Old metric should not appear in stats
    assert "old_metric" not in stats
    # Recent metric should appear
    assert "recent_metric" in stats


def test_performance_degradation_alert(monitoring_service: MonitoringService) -> None:
    """Test performance degradation triggers alert."""
    monitoring_service.settings.monitoring_enable_performance_tracking = True
    monitoring_service.settings.monitoring_performance_alert_threshold_ms = 100.0

    # Record high latencies
    for _ in range(10):
        monitoring_service.record_performance_metric("slow_operation", 150.0)

    metrics = {
        "positions": {"count": 0},
        "pnl": {"daily": 0.0, "daily_loss_pct": 0.0},
        "risk": {"circuit_breaker_active": False},
        "connectivity": {"breeze_authenticated": True},
    }

    alerts = monitoring_service.evaluate_alerts(metrics)

    # Should trigger performance alert
    assert any("performance_degradation" in a.rule for a in alerts)
    perf_alert = next(a for a in alerts if "performance_degradation" in a.rule)
    assert perf_alert.severity == "WARNING"
    assert perf_alert.context["avg_latency_ms"] == 150.0


# ============================================================================
# US-013 ENTERPRISE FEATURES - ALERT DELIVERY PLUGINS
# ============================================================================


def test_email_plugin_deliver(monitoring_service: MonitoringService) -> None:
    """Test EmailPlugin.deliver() sends email."""
    from unittest.mock import MagicMock, patch

    from src.services.monitoring import Alert, EmailPlugin

    # Configure email plugin
    monitoring_service.settings.monitoring_enable_email_alerts = True
    monitoring_service.settings.monitoring_email_smtp_host = "smtp.example.com"
    monitoring_service.settings.monitoring_email_smtp_port = 587
    monitoring_service.settings.monitoring_email_from = "alerts@sensequant.com"
    monitoring_service.settings.monitoring_email_to = ["ops@sensequant.com"]

    plugin = EmailPlugin(monitoring_service.settings)
    assert plugin.enabled

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="CRITICAL",
        rule="test_rule",
        message="Test alert",
        context={"test": "value"},
    )

    # Mock SMTP
    with patch("smtplib.SMTP") as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        result = plugin.deliver(alert)

        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.send_message.assert_called_once()


def test_email_plugin_disabled(monitoring_service: MonitoringService) -> None:
    """Test EmailPlugin respects enable flag."""
    from src.services.monitoring import Alert, EmailPlugin

    monitoring_service.settings.monitoring_enable_email_alerts = False
    plugin = EmailPlugin(monitoring_service.settings)

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="WARNING",
        rule="test",
        message="Test",
        context={},
    )

    result = plugin.deliver(alert)
    assert result is False


def test_email_plugin_delivery_failure(monitoring_service: MonitoringService) -> None:
    """Test EmailPlugin handles delivery failures gracefully."""
    from unittest.mock import patch

    from src.services.monitoring import Alert, EmailPlugin

    monitoring_service.settings.monitoring_enable_email_alerts = True
    monitoring_service.settings.monitoring_email_smtp_host = "smtp.example.com"
    monitoring_service.settings.monitoring_email_smtp_port = 587
    monitoring_service.settings.monitoring_email_to = ["ops@sensequant.com"]

    plugin = EmailPlugin(monitoring_service.settings)

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="CRITICAL",
        rule="test_rule",
        message="Test alert",
        context={},
    )

    # Mock SMTP to raise exception
    with patch("smtplib.SMTP", side_effect=Exception("Connection failed")):
        result = plugin.deliver(alert)
        assert result is False


def test_slack_plugin_deliver(monitoring_service: MonitoringService) -> None:
    """Test SlackPlugin.deliver() posts to webhook."""
    from unittest.mock import MagicMock, patch

    from src.services.monitoring import Alert, SlackPlugin

    monitoring_service.settings.monitoring_enable_slack_alerts = True
    monitoring_service.settings.monitoring_slack_webhook_url = "https://hooks.slack.com/test"

    plugin = SlackPlugin(monitoring_service.settings)
    assert plugin.enabled

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="WARNING",
        rule="test_rule",
        message="Test alert",
        context={"key": "value"},
    )

    # Mock requests.post
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = plugin.deliver(alert)

        assert result is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["timeout"] == 10
        assert "text" in call_args[1]["json"]
        assert "blocks" in call_args[1]["json"]


def test_slack_plugin_disabled(monitoring_service: MonitoringService) -> None:
    """Test SlackPlugin respects enable flag."""
    from src.services.monitoring import Alert, SlackPlugin

    monitoring_service.settings.monitoring_enable_slack_alerts = False
    plugin = SlackPlugin(monitoring_service.settings)

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="WARNING",
        rule="test",
        message="Test",
        context={},
    )

    result = plugin.deliver(alert)
    assert result is False


def test_slack_plugin_delivery_failure(monitoring_service: MonitoringService) -> None:
    """Test SlackPlugin handles delivery failures gracefully."""
    from unittest.mock import patch

    from src.services.monitoring import Alert, SlackPlugin

    monitoring_service.settings.monitoring_enable_slack_alerts = True
    monitoring_service.settings.monitoring_slack_webhook_url = "https://hooks.slack.com/test"

    plugin = SlackPlugin(monitoring_service.settings)

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="WARNING",
        rule="test_rule",
        message="Test alert",
        context={},
    )

    # Mock requests.post to raise exception
    with patch("requests.post", side_effect=Exception("Network error")):
        result = plugin.deliver(alert)
        assert result is False


def test_webhook_plugin_deliver(monitoring_service: MonitoringService) -> None:
    """Test WebhookPlugin.deliver() posts JSON."""
    from unittest.mock import MagicMock, patch

    from src.services.monitoring import Alert, WebhookPlugin

    monitoring_service.settings.monitoring_enable_webhook_alerts = True
    monitoring_service.settings.monitoring_webhook_url = "https://example.com/webhook"
    monitoring_service.settings.monitoring_webhook_headers = {"Authorization": "Bearer token"}

    plugin = WebhookPlugin(monitoring_service.settings)
    assert plugin.enabled

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="INFO",
        rule="test_rule",
        message="Test alert",
        context={"data": 123},
    )

    # Mock requests.post
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = plugin.deliver(alert)

        assert result is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["timeout"] == 10
        assert call_args[1]["json"] == alert.to_dict()
        assert call_args[1]["headers"]["Authorization"] == "Bearer token"


def test_webhook_plugin_disabled(monitoring_service: MonitoringService) -> None:
    """Test WebhookPlugin respects enable flag."""
    from src.services.monitoring import Alert, WebhookPlugin

    monitoring_service.settings.monitoring_enable_webhook_alerts = False
    plugin = WebhookPlugin(monitoring_service.settings)

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="INFO",
        rule="test",
        message="Test",
        context={},
    )

    result = plugin.deliver(alert)
    assert result is False


def test_deliver_alert_async(monitoring_service: MonitoringService) -> None:
    """Test _deliver_alert_async() calls plugins in background."""
    import time
    from unittest.mock import MagicMock

    from src.services.monitoring import Alert

    monitoring_service.settings.monitoring_enable_email_alerts = True
    monitoring_service.settings.monitoring_email_to = ["test@example.com"]

    # Reinitialize plugins
    from src.services.monitoring import EmailPlugin

    mock_plugin = MagicMock(spec=EmailPlugin)
    mock_plugin.enabled = True
    mock_plugin.deliver = MagicMock(return_value=True)

    monitoring_service.delivery_plugins = [mock_plugin]

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="INFO",
        rule="test",
        message="Test",
        context={},
    )

    monitoring_service._deliver_alert_async(alert)

    # Wait a bit for thread to execute
    time.sleep(0.1)

    mock_plugin.deliver.assert_called_once_with(alert)


def test_emit_alert_triggers_delivery(monitoring_service: MonitoringService) -> None:
    """Test emit_alert() triggers async delivery."""
    from unittest.mock import patch

    from src.services.monitoring import Alert

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="WARNING",
        rule="test_delivery",
        message="Test message",
        context={"test": "data"},
    )

    # Mock async delivery
    with patch.object(monitoring_service, "_deliver_alert_async") as mock_deliver:
        monitoring_service.emit_alert(alert)

        mock_deliver.assert_called_once_with(alert)


def test_delivery_plugin_failure_does_not_crash(monitoring_service: MonitoringService) -> None:
    """Test delivery plugin failures don't crash monitoring."""
    from src.services.monitoring import Alert, AlertDeliveryPlugin

    # Create failing plugin
    class FailingPlugin(AlertDeliveryPlugin):
        def deliver(self, alert: Alert) -> bool:
            raise Exception("Plugin crashed!")

    failing_plugin = FailingPlugin(monitoring_service.settings)
    failing_plugin.enabled = True
    monitoring_service.delivery_plugins = [failing_plugin]

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="INFO",
        rule="test",
        message="Test",
        context={},
    )

    # Should not raise exception
    try:
        monitoring_service._deliver_alert_async(alert)
        import time

        time.sleep(0.1)
    except Exception:
        pytest.fail("Delivery failure should not crash monitoring")


# ============================================================================
# US-013 ENTERPRISE FEATURES - ALERT ACKNOWLEDGEMENT
# ============================================================================


def test_acknowledge_alert(monitoring_service: MonitoringService) -> None:
    """Test acknowledge_alert() creates acknowledgement."""
    monitoring_service.acknowledge_alert(
        "test_rule", acknowledged_by="operator1", reason="False alarm"
    )

    assert "test_rule" in monitoring_service.acknowledgements
    ack = monitoring_service.acknowledgements["test_rule"]
    assert ack.rule == "test_rule"
    assert ack.acknowledged_by == "operator1"
    assert ack.reason == "False alarm"
    assert ack.ttl_seconds == monitoring_service.settings.monitoring_ack_ttl_seconds


def test_acknowledge_alert_prevents_notification(monitoring_service: MonitoringService) -> None:
    """Test acknowledged alerts are not re-triggered."""
    # Acknowledge circuit breaker alert
    monitoring_service.acknowledge_alert("circuit_breaker_triggered")

    metrics = {
        "positions": {"count": 3},
        "pnl": {"daily": -5000.0, "daily_loss_pct": -5.2},
        "risk": {"circuit_breaker_active": True},
        "connectivity": {"breeze_authenticated": True},
    }

    alerts = monitoring_service.evaluate_alerts(metrics)

    # Should NOT generate circuit breaker alert (acknowledged)
    assert not any(a.rule == "circuit_breaker_triggered" for a in alerts)


def test_clear_acknowledgement(monitoring_service: MonitoringService) -> None:
    """Test clear_acknowledgement() removes acknowledgement."""
    monitoring_service.acknowledge_alert("test_rule")
    assert "test_rule" in monitoring_service.acknowledgements

    monitoring_service.clear_acknowledgement("test_rule")
    assert "test_rule" not in monitoring_service.acknowledgements


def test_is_acknowledged(monitoring_service: MonitoringService) -> None:
    """Test _is_acknowledged() checks acknowledgement status."""
    # Clear any loaded acknowledgements from fixture
    monitoring_service.acknowledgements.clear()

    # Not acknowledged initially
    assert not monitoring_service._is_acknowledged("test_rule")

    # Acknowledge
    monitoring_service.acknowledge_alert("test_rule")
    assert monitoring_service._is_acknowledged("test_rule")

    # Clear
    monitoring_service.clear_acknowledgement("test_rule")
    assert not monitoring_service._is_acknowledged("test_rule")


def test_acknowledgement_expiration(monitoring_service: MonitoringService) -> None:
    """Test acknowledgement expires after TTL."""
    from src.services.monitoring import AckRecord

    # Create expired acknowledgement
    expired_ack = AckRecord(
        rule="expired_rule",
        acknowledged_at=(datetime.now() - timedelta(hours=2)).isoformat(),
        acknowledged_by="operator",
        ttl_seconds=3600,  # 1 hour TTL
        reason=None,
    )

    monitoring_service.acknowledgements["expired_rule"] = expired_ack

    # Should return False (auto-clears)
    assert not monitoring_service._is_acknowledged("expired_rule")
    # Should be removed from acknowledgements
    assert "expired_rule" not in monitoring_service.acknowledgements


def test_ack_record_is_expired(monitoring_service: MonitoringService) -> None:
    """Test AckRecord.is_expired() method."""
    from src.services.monitoring import AckRecord

    # Create non-expired ack
    recent_ack = AckRecord(
        rule="recent",
        acknowledged_at=datetime.now().isoformat(),
        acknowledged_by="operator",
        ttl_seconds=3600,
    )
    assert not recent_ack.is_expired()

    # Create expired ack
    old_ack = AckRecord(
        rule="old",
        acknowledged_at=(datetime.now() - timedelta(hours=2)).isoformat(),
        acknowledged_by="operator",
        ttl_seconds=3600,
    )
    assert old_ack.is_expired()


def test_save_acknowledgement(monitoring_service: MonitoringService, tmp_path: Path) -> None:
    """Test _save_acknowledgement() persists to file."""
    monitoring_service.ack_log_path = tmp_path / "ack.jsonl"

    monitoring_service.acknowledge_alert("test_rule", acknowledged_by="operator1")

    # Check file exists and contains acknowledgement
    assert monitoring_service.ack_log_path.exists()
    with open(monitoring_service.ack_log_path) as f:
        content = f.read()
        assert "test_rule" in content
        assert "operator1" in content


def test_load_acknowledgements(monitoring_service: MonitoringService, tmp_path: Path) -> None:
    """Test _load_acknowledgements() loads from file."""
    monitoring_service.ack_log_path = tmp_path / "ack.jsonl"

    # Save acknowledgement
    monitoring_service.acknowledge_alert("rule1", acknowledged_by="op1")
    monitoring_service.acknowledge_alert("rule2", acknowledged_by="op2")

    # Create new service and load
    from src.app.config import Settings

    settings = Settings()  # type: ignore[call-arg]
    new_service = MonitoringService(settings)
    new_service.ack_log_path = tmp_path / "ack.jsonl"
    new_service._load_acknowledgements()

    # Should have loaded acknowledgements
    assert "rule1" in new_service.acknowledgements
    assert "rule2" in new_service.acknowledgements


def test_load_acknowledgements_filters_expired(
    monitoring_service: MonitoringService, tmp_path: Path
) -> None:
    """Test _load_acknowledgements() skips expired acks."""
    from src.services.monitoring import AckRecord

    monitoring_service.ack_log_path = tmp_path / "ack.jsonl"
    monitoring_service.ack_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Write expired acknowledgement directly
    expired_ack = AckRecord(
        rule="expired_rule",
        acknowledged_at=(datetime.now() - timedelta(hours=2)).isoformat(),
        acknowledged_by="operator",
        ttl_seconds=3600,
    )

    with open(monitoring_service.ack_log_path, "w") as f:
        json.dump(expired_ack.to_dict(), f)
        f.write("\n")

    # Load acknowledgements
    monitoring_service._load_acknowledgements()

    # Expired ack should not be loaded
    assert "expired_rule" not in monitoring_service.acknowledgements


# ============================================================================
# US-013 ENTERPRISE FEATURES - RETENTION MANAGEMENT
# ============================================================================


def test_archive_old_metrics(monitoring_service: MonitoringService, tmp_path: Path) -> None:
    """Test archive_old_metrics() archives old data."""
    monitoring_service.archive_dir = tmp_path / "archive"
    monitoring_service.archive_dir.mkdir(parents=True, exist_ok=True)

    # Add old metrics (> 1 day)
    old_timestamp = (datetime.now() - timedelta(days=2)).isoformat()
    old_metrics = {
        "timestamp": old_timestamp,
        "positions": {"count": 1},
        "pnl": {"daily": 100.0},
    }
    monitoring_service.metrics_history.append(old_metrics)

    # Add recent metrics
    recent_metrics = {
        "timestamp": datetime.now().isoformat(),
        "positions": {"count": 2},
        "pnl": {"daily": 200.0},
    }
    monitoring_service.metrics_history.append(recent_metrics)

    initial_count = len(monitoring_service.metrics_history)
    monitoring_service.archive_old_metrics()

    # Should remove old metrics from history
    assert len(monitoring_service.metrics_history) < initial_count

    # Should create archive file
    archive_files = list(monitoring_service.archive_dir.glob("metrics_*.json.gz"))
    assert len(archive_files) > 0


def test_archive_old_metrics_compression(
    monitoring_service: MonitoringService, tmp_path: Path
) -> None:
    """Test archived metrics are compressed."""
    import gzip

    monitoring_service.archive_dir = tmp_path / "archive"
    monitoring_service.archive_dir.mkdir(parents=True, exist_ok=True)

    # Add old metrics
    old_timestamp = (datetime.now() - timedelta(days=2)).isoformat()
    old_metrics = {
        "timestamp": old_timestamp,
        "positions": {"count": 5},
        "pnl": {"daily": 500.0},
    }
    monitoring_service.metrics_history.append(old_metrics)

    monitoring_service.archive_old_metrics()

    # Find archive file
    archive_files = list(monitoring_service.archive_dir.glob("metrics_*.json.gz"))
    assert len(archive_files) > 0

    # Verify it's compressed and readable
    with gzip.open(archive_files[0], "rt") as f:
        archived_data = json.load(f)
        assert len(archived_data) == 1
        assert archived_data[0]["pnl"]["daily"] == 500.0


def test_cleanup_old_archives(monitoring_service: MonitoringService, tmp_path: Path) -> None:
    """Test cleanup_old_archives() removes old archives."""
    monitoring_service.archive_dir = tmp_path / "archive"
    monitoring_service.archive_dir.mkdir(parents=True, exist_ok=True)
    monitoring_service.settings.monitoring_max_archive_days = 30

    # Create old archive file
    old_archive = monitoring_service.archive_dir / "metrics_old.json.gz"
    old_archive.write_text("test")
    # Set mtime to 40 days ago
    old_time = (datetime.now() - timedelta(days=40)).timestamp()
    old_archive.touch()
    import os

    os.utime(old_archive, (old_time, old_time))

    # Create recent archive
    recent_archive = monitoring_service.archive_dir / "metrics_recent.json.gz"
    recent_archive.write_text("test")

    monitoring_service.cleanup_old_archives()

    # Old archive should be deleted
    assert not old_archive.exists()
    # Recent archive should remain
    assert recent_archive.exists()


def test_max_raw_metrics_limit_enforcement(monitoring_service: MonitoringService) -> None:
    """Test max_raw_metrics limit is enforced."""
    monitoring_service.settings.monitoring_max_raw_metrics = 50

    # Add 100 metrics
    for i in range(100):
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "positions": {"count": i},
        }
        monitoring_service.record_tick(metrics)

    # Should keep only last 50
    assert len(monitoring_service.metrics_history) == 50


def test_archive_no_old_metrics(monitoring_service: MonitoringService, tmp_path: Path) -> None:
    """Test archive_old_metrics() does nothing when no old metrics."""
    monitoring_service.archive_dir = tmp_path / "archive"
    monitoring_service.archive_dir.mkdir(parents=True, exist_ok=True)

    # Add only recent metrics
    recent_metrics = {
        "timestamp": datetime.now().isoformat(),
        "positions": {"count": 1},
    }
    monitoring_service.metrics_history.append(recent_metrics)

    monitoring_service.archive_old_metrics()

    # No archive files should be created
    archive_files = list(monitoring_service.archive_dir.glob("metrics_*.json.gz"))
    assert len(archive_files) == 0

    # Metrics should remain
    assert len(monitoring_service.metrics_history) == 1
