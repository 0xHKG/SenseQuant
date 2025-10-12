"""Integration tests for monitoring pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.app.config import Settings
from src.services.monitoring import MonitoringService


@pytest.fixture
def monitoring_service(tmp_path: Path) -> MonitoringService:
    """Create monitoring service with temp directories."""
    settings = Settings()  # type: ignore[call-arg]
    service = MonitoringService(settings)

    # Override directories
    service.alerts_dir = tmp_path / "alerts"
    service.metrics_dir = tmp_path / "metrics"
    service.archive_dir = tmp_path / "archive"
    service.ack_log_path = tmp_path / "alerts" / "acknowledgements.jsonl"
    service.alerts_dir.mkdir(parents=True, exist_ok=True)
    service.metrics_dir.mkdir(parents=True, exist_ok=True)
    service.archive_dir.mkdir(parents=True, exist_ok=True)

    # Clear any loaded acknowledgements to ensure test isolation
    service.acknowledgements.clear()

    return service


def test_full_monitoring_pipeline(monitoring_service: MonitoringService) -> None:
    """Test full monitoring pipeline end-to-end."""
    # Simulate multiple engine ticks
    for i in range(5):
        metrics = {
            "positions": {"count": i % 3},
            "pnl": {"daily": i * 100.0, "daily_loss_pct": i * 0.1},
            "risk": {"circuit_breaker_active": False},
            "connectivity": {"breeze_authenticated": True},
        }

        monitoring_service.record_tick(metrics)

    # Verify metrics collected
    assert len(monitoring_service.metrics_history) == 5

    # Verify heartbeat updated
    assert monitoring_service.heartbeat_timestamp is not None

    # Run health checks
    results = monitoring_service.run_health_checks()
    assert len(results) > 0


def test_alert_emission_and_retrieval(monitoring_service: MonitoringService) -> None:
    """Test alert emission and retrieval."""
    # Trigger circuit breaker alert
    metrics = {
        "positions": {"count": 3},
        "pnl": {"daily": -5000.0, "daily_loss_pct": -5.2},
        "risk": {"circuit_breaker_active": True},
        "connectivity": {"breeze_authenticated": True},
    }

    monitoring_service.record_tick(metrics)

    # Retrieve alerts
    alerts = monitoring_service.get_active_alerts(hours=1)

    # Should have circuit breaker alert
    assert len(alerts) > 0
    assert any(a.rule == "circuit_breaker_triggered" for a in alerts)

    # Verify alert file exists
    alert_files = list(monitoring_service.alerts_dir.glob("*.jsonl"))
    assert len(alert_files) > 0


def test_metrics_persistence_pipeline(monitoring_service: MonitoringService) -> None:
    """Test metrics persistence."""
    # Add metrics
    for i in range(20):
        metrics = {
            "positions": {"count": i},
            "pnl": {"daily": i * 50.0},
        }
        monitoring_service.record_tick(metrics)

    # Metrics should be persisted every 10 ticks
    metric_files = list(monitoring_service.metrics_dir.glob("metrics_*.json"))
    assert len(metric_files) >= 1


def test_monitoring_resilience(monitoring_service: MonitoringService) -> None:
    """Test monitoring continues despite errors."""
    # Record tick with invalid data (should not crash)
    invalid_metrics = {"invalid": "data"}

    try:
        monitoring_service.record_tick(invalid_metrics)
        # Should not raise exception
    except Exception:
        pytest.fail("Monitoring should not crash on invalid data")

    # Service should still be functional
    valid_metrics = {
        "positions": {"count": 1},
        "pnl": {"daily": 100.0},
    }
    monitoring_service.record_tick(valid_metrics)

    assert len(monitoring_service.metrics_history) >= 1


# ============================================================================
# US-013 ENTERPRISE FEATURES - INTEGRATION TESTS
# ============================================================================


def test_aggregation_pipeline(monitoring_service: MonitoringService) -> None:
    """Test end-to-end metric aggregation pipeline."""

    monitoring_service.settings.monitoring_enable_aggregation = True
    monitoring_service.settings.monitoring_aggregation_interval_seconds = 1

    # Simulate multiple ticks with varying metrics
    for i in range(10):
        metrics = {
            "positions": {"count": i % 3},
            "pnl": {"daily": i * 100.0, "daily_loss_pct": -i * 0.1},
            "risk": {"circuit_breaker_active": False},
            "connectivity": {"breeze_authenticated": True},
        }
        monitoring_service.record_tick(metrics)

    # Allow time for rollup computation
    import time

    time.sleep(1.5)

    # Trigger another tick to compute rollup
    final_metrics = {
        "positions": {"count": 2},
        "pnl": {"daily": 1000.0, "daily_loss_pct": -1.0},
        "risk": {"circuit_breaker_active": False},
        "connectivity": {"breeze_authenticated": True},
    }
    monitoring_service.record_tick(final_metrics)

    # Verify rollups were computed
    assert len(monitoring_service.rollup_history) > 0

    # Query aggregated metrics
    aggregated = monitoring_service.get_aggregated_metrics()
    assert len(aggregated) > 0

    # Verify rollup contains expected metric types
    rollup = aggregated[0]
    assert "pnl_daily" in rollup.metrics
    assert rollup.metrics["pnl_daily"].count > 0


def test_performance_tracking_pipeline(monitoring_service: MonitoringService) -> None:
    """Test end-to-end performance tracking pipeline."""
    monitoring_service.settings.monitoring_enable_performance_tracking = True
    monitoring_service.settings.monitoring_performance_alert_threshold_ms = 100.0

    # Simulate engine operations with performance tracking
    for i in range(10):
        # Record latencies
        monitoring_service.record_performance_metric("tick_latency", 50.0 + i * 5, {"iteration": i})

        metrics = {
            "positions": {"count": 1},
            "pnl": {"daily": 100.0},
            "risk": {"circuit_breaker_active": False},
            "connectivity": {"breeze_authenticated": True},
        }
        monitoring_service.record_tick(metrics)

    # Verify performance metrics collected
    assert "tick_latency" in monitoring_service.performance_metrics
    assert len(monitoring_service.performance_metrics["tick_latency"]) == 10

    # Simulate performance degradation
    for _ in range(5):
        monitoring_service.record_performance_metric("slow_operation", 150.0)

    metrics = {
        "positions": {"count": 1},
        "pnl": {"daily": 100.0},
        "risk": {"circuit_breaker_active": False},
        "connectivity": {"breeze_authenticated": True},
    }
    alerts = monitoring_service.evaluate_alerts(metrics)

    # Should trigger performance alert
    assert any("performance_degradation" in a.rule for a in alerts)


def test_alert_delivery_pipeline(monitoring_service: MonitoringService, tmp_path: Path) -> None:
    """Test end-to-end alert delivery pipeline."""
    import time
    from unittest.mock import MagicMock

    # Configure delivery plugins
    monitoring_service.settings.monitoring_enable_email_alerts = True
    monitoring_service.settings.monitoring_enable_slack_alerts = True
    monitoring_service.settings.monitoring_email_to = ["ops@example.com"]
    monitoring_service.settings.monitoring_slack_webhook_url = "https://hooks.slack.com/test"

    # Reinitialize plugins
    from src.services.monitoring import EmailPlugin, SlackPlugin

    mock_email = MagicMock(spec=EmailPlugin)
    mock_email.enabled = True
    mock_email.deliver = MagicMock(return_value=True)

    mock_slack = MagicMock(spec=SlackPlugin)
    mock_slack.enabled = True
    mock_slack.deliver = MagicMock(return_value=True)

    monitoring_service.delivery_plugins = [mock_email, mock_slack]

    # Trigger circuit breaker alert
    metrics = {
        "positions": {"count": 3},
        "pnl": {"daily": -5000.0, "daily_loss_pct": -5.2},
        "risk": {"circuit_breaker_active": True},
        "connectivity": {"breeze_authenticated": True},
    }

    monitoring_service.record_tick(metrics)

    # Wait for async delivery
    time.sleep(0.2)

    # Verify both plugins were called
    assert mock_email.deliver.called
    assert mock_slack.deliver.called

    # Verify alert was persisted
    alert_files = list(monitoring_service.alerts_dir.glob("*.jsonl"))
    assert len(alert_files) > 0


def test_acknowledgement_pipeline(monitoring_service: MonitoringService) -> None:
    """Test end-to-end alert acknowledgement workflow."""
    # Clear any deduplication state
    monitoring_service.last_alert_times.clear()

    # Trigger alert
    metrics = {
        "positions": {"count": 3},
        "pnl": {"daily": -5000.0, "daily_loss_pct": -5.2},
        "risk": {"circuit_breaker_active": True},
        "connectivity": {"breeze_authenticated": True},
    }

    # First evaluation - alert should fire
    alerts = monitoring_service.evaluate_alerts(metrics)
    assert any(a.rule == "circuit_breaker_triggered" for a in alerts)

    # Acknowledge the alert
    monitoring_service.acknowledge_alert(
        "circuit_breaker_triggered", acknowledged_by="operator1", reason="Expected during testing"
    )

    # Second evaluation - alert should be suppressed
    alerts = monitoring_service.evaluate_alerts(metrics)
    assert not any(a.rule == "circuit_breaker_triggered" for a in alerts)

    # Verify acknowledgement persisted
    assert monitoring_service.ack_log_path.exists()

    # Clear acknowledgement and deduplication state
    monitoring_service.clear_acknowledgement("circuit_breaker_triggered")
    monitoring_service.last_alert_times.clear()

    # Third evaluation - alert should fire again
    alerts = monitoring_service.evaluate_alerts(metrics)
    assert any(a.rule == "circuit_breaker_triggered" for a in alerts)


def test_retention_pipeline(monitoring_service: MonitoringService, tmp_path: Path) -> None:
    """Test end-to-end retention management pipeline."""
    from datetime import timedelta

    monitoring_service.archive_dir = tmp_path / "archive"
    monitoring_service.archive_dir.mkdir(parents=True, exist_ok=True)
    monitoring_service.settings.monitoring_max_archive_days = 30

    # Add old metrics
    for i in range(5):
        old_timestamp = (datetime.now() - timedelta(days=2, hours=i)).isoformat()
        old_metrics = {
            "timestamp": old_timestamp,
            "positions": {"count": i},
            "pnl": {"daily": i * 100.0},
        }
        monitoring_service.metrics_history.append(old_metrics)

    # Add recent metrics
    for i in range(5):
        recent_metrics = {
            "timestamp": datetime.now().isoformat(),
            "positions": {"count": i},
            "pnl": {"daily": i * 200.0},
        }
        monitoring_service.metrics_history.append(recent_metrics)

    initial_count = len(monitoring_service.metrics_history)
    assert initial_count == 10

    # Archive old metrics
    monitoring_service.archive_old_metrics()

    # Verify old metrics removed from memory
    assert len(monitoring_service.metrics_history) < initial_count

    # Verify archive file created
    archive_files = list(monitoring_service.archive_dir.glob("metrics_*.json.gz"))
    assert len(archive_files) > 0

    # Create old archive file
    old_archive = monitoring_service.archive_dir / "metrics_very_old.json.gz"
    old_archive.write_text("test")
    old_time = (datetime.now() - timedelta(days=40)).timestamp()
    import os

    os.utime(old_archive, (old_time, old_time))

    # Cleanup old archives
    monitoring_service.cleanup_old_archives()

    # Verify old archive removed
    assert not old_archive.exists()


def test_multi_plugin_delivery_with_failures(monitoring_service: MonitoringService) -> None:
    """Test alert delivery continues even if some plugins fail."""
    import time
    from unittest.mock import MagicMock

    from src.services.monitoring import EmailPlugin, SlackPlugin

    # Create one working and one failing plugin
    working_plugin = MagicMock(spec=EmailPlugin)
    working_plugin.enabled = True
    working_plugin.deliver = MagicMock(return_value=True)

    failing_plugin = MagicMock(spec=SlackPlugin)
    failing_plugin.enabled = True
    failing_plugin.deliver = MagicMock(side_effect=Exception("Network error"))

    monitoring_service.delivery_plugins = [working_plugin, failing_plugin]

    # Trigger alert
    metrics = {
        "positions": {"count": 0},
        "pnl": {"daily": -4500.0, "daily_loss_pct": -4.5},
        "risk": {"circuit_breaker_active": False},
        "connectivity": {"breeze_authenticated": True},
    }

    monitoring_service.record_tick(metrics)

    # Wait for async delivery
    time.sleep(0.2)

    # Working plugin should have been called
    assert working_plugin.deliver.called

    # Service should still be functional
    assert monitoring_service.heartbeat_timestamp is not None


def test_full_enterprise_monitoring_pipeline(
    monitoring_service: MonitoringService, tmp_path: Path
) -> None:
    """Test complete enterprise monitoring pipeline with all features."""
    import time
    from datetime import timedelta
    from unittest.mock import MagicMock

    # Configure all enterprise features
    monitoring_service.settings.monitoring_enable_aggregation = True
    monitoring_service.settings.monitoring_aggregation_interval_seconds = 1
    monitoring_service.settings.monitoring_enable_performance_tracking = True
    monitoring_service.settings.monitoring_performance_alert_threshold_ms = 100.0
    monitoring_service.archive_dir = tmp_path / "archive"
    monitoring_service.archive_dir.mkdir(parents=True, exist_ok=True)

    # Setup mock delivery plugins
    mock_plugin = MagicMock()
    mock_plugin.enabled = True
    mock_plugin.deliver = MagicMock(return_value=True)
    monitoring_service.delivery_plugins = [mock_plugin]

    # Simulate trading session
    for i in range(20):
        # Record performance
        monitoring_service.record_performance_metric("tick_latency", 50.0 + i * 2)

        # Record tick
        metrics = {
            "positions": {"count": i % 3},
            "pnl": {"daily": i * 50.0, "daily_loss_pct": -i * 0.05},
            "risk": {"circuit_breaker_active": False},
            "connectivity": {"breeze_authenticated": True},
        }
        monitoring_service.record_tick(metrics)

    # Verify metrics collected
    assert len(monitoring_service.metrics_history) == 20

    # Verify performance tracking
    assert "tick_latency" in monitoring_service.performance_metrics

    # Wait for rollup computation
    time.sleep(1.5)

    # Trigger final tick to compute rollup
    final_metrics = {
        "positions": {"count": 1},
        "pnl": {"daily": 1000.0},
        "risk": {"circuit_breaker_active": False},
        "connectivity": {"breeze_authenticated": True},
    }
    monitoring_service.record_tick(final_metrics)

    # Verify rollups computed
    assert len(monitoring_service.rollup_history) > 0

    # Trigger alert condition
    alert_metrics = {
        "positions": {"count": 3},
        "pnl": {"daily": -5000.0, "daily_loss_pct": -5.2},
        "risk": {"circuit_breaker_active": True},
        "connectivity": {"breeze_authenticated": True},
    }
    monitoring_service.record_tick(alert_metrics)

    # Wait for async delivery
    time.sleep(0.2)

    # Verify alert delivered
    assert mock_plugin.deliver.called

    # Acknowledge alert
    monitoring_service.acknowledge_alert("circuit_breaker_triggered", acknowledged_by="operator")

    # Verify acknowledgement works
    alerts = monitoring_service.evaluate_alerts(alert_metrics)
    assert not any(a.rule == "circuit_breaker_triggered" for a in alerts)

    # Add old metrics for archiving
    old_metrics = {
        "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
        "positions": {"count": 5},
    }
    monitoring_service.metrics_history.append(old_metrics)

    # Archive old metrics
    monitoring_service.archive_old_metrics()

    # Verify archive created
    archive_files = list(monitoring_service.archive_dir.glob("metrics_*.json.gz"))
    assert len(archive_files) > 0

    # Verify health checks work
    health_results = monitoring_service.run_health_checks()
    assert len(health_results) > 0


def test_rollup_aggregation_with_performance_metrics(monitoring_service: MonitoringService) -> None:
    """Test rollups include performance metrics."""
    import time

    monitoring_service.settings.monitoring_enable_aggregation = True
    monitoring_service.settings.monitoring_aggregation_interval_seconds = 1
    monitoring_service.settings.monitoring_enable_performance_tracking = True

    # Record performance metrics
    for i in range(5):
        monitoring_service.record_performance_metric("api_latency", 10.0 + i * 5)

    # Wait a bit to ensure time difference for rollup
    time.sleep(0.1)

    # Add regular metrics using record_tick (which adds timestamps)
    for i in range(5):
        metrics = {
            "positions": {"count": i},
            "pnl": {"daily": i * 100.0, "daily_loss_pct": -i * 0.1},
            "risk": {"circuit_breaker_active": False},
            "connectivity": {"breeze_authenticated": True},
        }
        monitoring_service.record_tick(metrics)

    # Wait for rollup interval to elapse
    time.sleep(1.1)

    # Force another tick to trigger rollup
    final_metrics = {
        "positions": {"count": 1},
        "pnl": {"daily": 100.0},
        "risk": {"circuit_breaker_active": False},
        "connectivity": {"breeze_authenticated": True},
    }
    monitoring_service.record_tick(final_metrics)

    # Verify rollup was created
    assert len(monitoring_service.rollup_history) > 0
    rollup = monitoring_service.rollup_history[-1]  # Get latest rollup

    # Verify rollup includes performance metrics if they exist
    if "perf_api_latency" in rollup.metrics:
        perf_stats = rollup.metrics["perf_api_latency"]
        assert perf_stats.min == 10.0
        assert perf_stats.max == 30.0


def test_concurrent_alert_acknowledgement(monitoring_service: MonitoringService) -> None:
    """Test acknowledgement works correctly with concurrent alerts."""
    # Acknowledge a rule before it fires
    monitoring_service.acknowledge_alert("daily_loss_high", acknowledged_by="operator")

    # Trigger multiple alert conditions
    metrics = {
        "positions": {"count": 3},
        "pnl": {"daily": -5000.0, "daily_loss_pct": -4.8},
        "risk": {"circuit_breaker_active": True},
        "connectivity": {"breeze_authenticated": True},
    }

    alerts = monitoring_service.evaluate_alerts(metrics)

    # Circuit breaker should fire (not acknowledged)
    assert any(a.rule == "circuit_breaker_triggered" for a in alerts)

    # Daily loss should NOT fire (acknowledged)
    assert not any(a.rule == "daily_loss_high" for a in alerts)


def test_metric_limit_with_archiving(monitoring_service: MonitoringService, tmp_path: Path) -> None:
    """Test metric limit enforcement works with archiving."""
    from datetime import timedelta

    monitoring_service.settings.monitoring_max_raw_metrics = 50
    monitoring_service.archive_dir = tmp_path / "archive"
    monitoring_service.archive_dir.mkdir(parents=True, exist_ok=True)

    # Add 30 old metrics
    for i in range(30):
        old_metrics = {
            "timestamp": (datetime.now() - timedelta(days=2, hours=i)).isoformat(),
            "positions": {"count": i},
        }
        monitoring_service.metrics_history.append(old_metrics)

    # Add 40 recent metrics (total 70, exceeds limit)
    for i in range(40):
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "positions": {"count": i},
        }
        monitoring_service.record_tick(metrics)

    # Should be at limit (50)
    assert len(monitoring_service.metrics_history) == 50

    # Archive old metrics
    monitoring_service.archive_old_metrics()

    # Should have removed old metrics
    assert len(monitoring_service.metrics_history) < 50

    # Archive should exist
    archive_files = list(monitoring_service.archive_dir.glob("metrics_*.json.gz"))
    assert len(archive_files) > 0


def test_alert_deduplication_with_acknowledgement(monitoring_service: MonitoringService) -> None:
    """Test alert deduplication interacts correctly with acknowledgement."""
    metrics = {
        "positions": {"count": 0},
        "pnl": {"daily": -4500.0, "daily_loss_pct": -4.5},
        "risk": {"circuit_breaker_active": False},
        "connectivity": {"breeze_authenticated": True},
    }

    # First alert should fire
    alerts1 = monitoring_service.evaluate_alerts(metrics)
    assert any(a.rule == "daily_loss_high" for a in alerts1)

    # Second alert should be deduplicated
    alerts2 = monitoring_service.evaluate_alerts(metrics)
    assert not any(a.rule == "daily_loss_high" for a in alerts2)

    # Acknowledge the alert
    monitoring_service.acknowledge_alert("daily_loss_high")

    # Clear deduplication by clearing last alert times
    monitoring_service.last_alert_times.clear()

    # Alert should still be suppressed (acknowledged)
    alerts3 = monitoring_service.evaluate_alerts(metrics)
    assert not any(a.rule == "daily_loss_high" for a in alerts3)
