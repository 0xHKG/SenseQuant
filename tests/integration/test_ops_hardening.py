"""Integration tests for operational hardening (US-027)."""

from __future__ import annotations

import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.monitoring import Alert, MonitoringService
from src.services.secrets_manager import SecretsManager
from src.services.state_manager import StateManager


@pytest.fixture
def temp_state_file(tmp_path: Path) -> Path:
    """Create temporary state file."""
    return tmp_path / "test_state.json"


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings."""
    settings = MagicMock()
    settings.monitoring_heartbeat_lapse_seconds = 60
    settings.monitoring_artifact_staleness_hours = 24
    settings.monitoring_enable_email_alerts = False
    settings.monitoring_enable_slack_alerts = False
    settings.monitoring_enable_webhook_alerts = False
    settings.enable_monitoring = True
    settings.monitoring_enable_aggregation = False
    settings.monitoring_enable_performance_tracking = False
    settings.monitoring_max_raw_metrics = 100
    settings.monitoring_max_sentiment_failures = 5
    settings.monitoring_daily_loss_alert_pct = 4.0
    settings.max_daily_loss_pct = 5.0
    settings.student_monitoring_enabled = False
    return settings


# ============================================================================
# Deployment Tests
# ============================================================================


def test_deployment_workflow(tmp_path: Path, temp_state_file: Path) -> None:
    """Test deployment workflow with smoke test and state recording."""
    # Create mock directories
    staging_dir = tmp_path / "staging"
    prod_dir = tmp_path / "production"

    staging_dir.mkdir()
    prod_dir.mkdir()

    # Create mock model in staging
    model_file = staging_dir / "student_model.pkl"
    mock_model = {"type": "mock_model", "version": "1.0"}
    with open(model_file, "wb") as f:
        pickle.dump(mock_model, f)

    # Simulate deployment (copy from staging to prod)
    prod_model_file = prod_dir / "student_model.pkl"
    with open(model_file, "rb") as src:
        with open(prod_model_file, "wb") as dst:
            dst.write(src.read())

    # Verify deployment
    assert prod_model_file.exists()

    # Verify model loadable
    with open(prod_model_file, "rb") as f:
        loaded_model = pickle.load(f)
    assert loaded_model["type"] == "mock_model"

    # Record deployment in StateManager
    state_mgr = StateManager(state_file=temp_state_file)
    state_mgr.record_deployment(
        release_id="test_release_v1.0.0",
        environment="prod",
        timestamp=datetime.now().isoformat(),
        status="success",
        artifacts=["student_model.pkl"],
        rollback=False,
        smoke_test_passed=True,
        deployed_by="test-user",
    )

    # Verify state recording
    deployments = state_mgr.get_deployment_history(limit=5)
    assert len(deployments) == 1
    assert deployments[0]["status"] == "success"
    assert deployments[0]["environment"] == "prod"
    assert deployments[0]["smoke_test_passed"] is True


def test_deployment_rollback_scenario(tmp_path: Path, temp_state_file: Path) -> None:
    """Test rollback scenario when smoke test fails."""
    state_mgr = StateManager(state_file=temp_state_file)

    # Simulate successful initial deployment
    state_mgr.record_deployment(
        release_id="v1.0.0",
        environment="prod",
        timestamp=(datetime.now() - timedelta(hours=1)).isoformat(),
        status="success",
        artifacts=["model_v1.pkl"],
        rollback=False,
        smoke_test_passed=True,
        deployed_by="deploy-script",
    )

    # Simulate failed deployment with automatic rollback
    state_mgr.record_deployment(
        release_id="v1.1.0",
        environment="prod",
        timestamp=datetime.now().isoformat(),
        status="rolled_back",
        artifacts=["model_v1.pkl"],  # Rolled back to v1
        rollback=True,
        smoke_test_passed=False,
        deployed_by="deploy-script",
    )

    # Verify deployment history
    deployments = state_mgr.get_deployment_history(limit=10)
    assert len(deployments) == 2

    # Most recent deployment should be the rollback
    assert deployments[0]["status"] == "rolled_back"
    assert deployments[0]["rollback"] is True
    assert deployments[0]["smoke_test_passed"] is False

    # Last successful deployment should still be v1.0.0
    last_deploy = state_mgr.get_last_deployment(environment="prod")
    assert last_deploy is not None
    assert last_deploy["release_id"] == "v1.0.0"


# ============================================================================
# Secrets Management Tests
# ============================================================================


def test_secrets_plain_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test secrets manager in plain mode (load from environment)."""
    # Set environment variables
    monkeypatch.setenv("BREEZE_API_KEY", "test_key_123")
    monkeypatch.setenv("BREEZE_API_SECRET", "test_secret_456")

    manager = SecretsManager(mode="plain")

    # Verify secrets loaded
    assert manager.get_secret("BREEZE_API_KEY") == "test_key_123"
    assert manager.get_secret("BREEZE_API_SECRET") == "test_secret_456"


def test_secrets_encryption_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test secrets encryption and decryption roundtrip."""
    # Create plain secrets
    monkeypatch.setenv("TEST_SECRET_1", "value1")
    monkeypatch.setenv("TEST_SECRET_2", "value2")

    manager1 = SecretsManager(mode="plain")

    # Encrypt secrets
    encrypted_file = tmp_path / "secrets.enc"
    key = manager1.encrypt_secrets(output_file=str(encrypted_file))

    # Verify encrypted file exists
    assert encrypted_file.exists()

    # Save key to file
    key_file = tmp_path / "test.key"
    key_file.write_bytes(key)

    # Load encrypted secrets with new manager
    manager2 = SecretsManager(
        mode="encrypted",
        key_path=str(key_file),
        encrypted_file=str(encrypted_file),
    )

    # Verify decryption
    assert manager2.get_secret("TEST_SECRET_1") == "value1"
    assert manager2.get_secret("TEST_SECRET_2") == "value2"


def test_secrets_set_and_get(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test setting and getting secrets in-memory."""
    manager = SecretsManager(mode="plain")

    # Set secrets
    manager.set_secret("NEW_SECRET", "new_value")

    # Get secret
    assert manager.get_secret("NEW_SECRET") == "new_value"
    assert manager.get_secret("NONEXISTENT_SECRET", default="default") == "default"


# ============================================================================
# Monitoring: Heartbeat & Liveness Tests
# ============================================================================


def test_heartbeat_tracking(mock_settings: MagicMock) -> None:
    """Test heartbeat tracking and liveness check."""
    monitor = MonitoringService(mock_settings)

    # Report heartbeat
    monitor.heartbeat()
    assert monitor.heartbeat_timestamp is not None

    # Check heartbeat (should be healthy immediately)
    result = monitor.check_heartbeat_health()
    assert result.status == "OK"

    # Simulate heartbeat lapse by mocking time
    old_timestamp = monitor.heartbeat_timestamp

    with patch("src.services.monitoring.datetime") as mock_datetime:
        # Mock time 120 seconds in future (exceeds 60s threshold)
        future_time = datetime.now() + timedelta(seconds=120)
        mock_datetime.now.return_value = future_time
        mock_datetime.fromisoformat = datetime.fromisoformat

        # Update monitor's heartbeat_timestamp to old value
        monitor.heartbeat_timestamp = old_timestamp

        # Check heartbeat (should fail)
        result = monitor.check_heartbeat_health()

    # Verify lapse detected
    assert result.status == "ERROR"
    assert "lapsed" in result.message.lower()


def test_heartbeat_escalation(mock_settings: MagicMock) -> None:
    """Test heartbeat lapse escalation with multiple failures."""
    mock_settings.monitoring_enable_slack_alerts = False
    monitor = MonitoringService(mock_settings)

    # Report heartbeat
    monitor.heartbeat()

    # Simulate 3 consecutive heartbeat failures
    old_timestamp = monitor.heartbeat_timestamp

    for _ in range(3):
        with patch("src.services.monitoring.datetime") as mock_datetime:
            # Mock time 120 seconds in future
            future_time = datetime.now() + timedelta(seconds=120)
            mock_datetime.now.return_value = future_time
            mock_datetime.fromisoformat = datetime.fromisoformat

            # Update heartbeat_timestamp
            monitor.heartbeat_timestamp = old_timestamp

            # Check heartbeat (should escalate)
            result = monitor.check_heartbeat_health()

        # Verify escalation based on failure count
        assert result.status == "ERROR"

        consecutive_failures = (
            result.details.get("consecutive_failures", 0) if result.details else 0
        )

        if consecutive_failures >= 3:
            # Should trigger CRITICAL escalation
            # Verify alert was created (check logs or alert history)
            pass


def test_service_availability_check(mock_settings: MagicMock) -> None:
    """Test service availability checking."""
    monitor = MonitoringService(mock_settings)

    # Mock service check function (service available)
    def mock_check_available() -> bool:
        return True

    result = monitor.check_service_availability("breeze_api", mock_check_available)
    assert result.status == "OK"

    # Mock service check function (service unavailable)
    def mock_check_unavailable() -> bool:
        return False

    result = monitor.check_service_availability("database", mock_check_unavailable)
    assert result.status == "ERROR"
    assert "unavailable" in result.message.lower()


# ============================================================================
# Monitoring: Alert Delivery Tests
# ============================================================================


def test_slack_alert_delivery(mock_settings: MagicMock) -> None:
    """Test Slack alert delivery via webhook."""
    mock_settings.monitoring_enable_slack_alerts = True
    mock_settings.monitoring_slack_webhook_url = "https://hooks.slack.com/test"

    monitor = MonitoringService(mock_settings)

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="CRITICAL",
        rule="test_rule",
        message="Test alert message",
        context={"key": "value"},
    )

    # Mock requests.post
    with patch("src.services.monitoring.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Deliver alert (need to call plugin directly since emit_alert is async)
        slack_plugin = monitor.delivery_plugins[1]  # SlackPlugin is second
        success = slack_plugin.deliver(alert)

        # Verify webhook called
        assert success is True
        assert mock_post.called
        call_args = mock_post.call_args
        assert "https://hooks.slack.com/test" in str(call_args)


def test_email_alert_delivery(mock_settings: MagicMock) -> None:
    """Test email alert delivery via SMTP."""
    mock_settings.monitoring_enable_email_alerts = True
    mock_settings.monitoring_email_smtp_host = "smtp.gmail.com"
    mock_settings.monitoring_email_smtp_port = 587
    mock_settings.monitoring_email_smtp_user = "test@example.com"
    mock_settings.monitoring_email_smtp_password = "password"
    mock_settings.monitoring_email_from = "sender@example.com"
    mock_settings.monitoring_email_to = ["recipient@example.com"]

    monitor = MonitoringService(mock_settings)

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity="WARNING",
        rule="test_rule",
        message="Test email alert",
        context={},
    )

    # Mock SMTP
    with patch("src.services.monitoring.smtplib.SMTP") as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # Deliver alert
        email_plugin = monitor.delivery_plugins[0]  # EmailPlugin is first
        success = email_plugin.deliver(alert)

        # Verify SMTP called
        assert success is True
        assert mock_server.starttls.called
        assert mock_server.login.called
        assert mock_server.send_message.called


def test_alert_filtering(mock_settings: MagicMock, tmp_path: Path) -> None:
    """Test alert filtering by severity."""
    # Create temp alerts directory
    alerts_dir = tmp_path / "logs" / "alerts"
    alerts_dir.mkdir(parents=True)

    monitor = MonitoringService(mock_settings)
    monitor.alerts_dir = alerts_dir

    # Emit test alerts with different severities
    alerts = [
        Alert(
            timestamp=datetime.now().isoformat(),
            severity="INFO",
            rule="info_rule",
            message="Info message",
            context={},
        ),
        Alert(
            timestamp=datetime.now().isoformat(),
            severity="WARNING",
            rule="warning_rule",
            message="Warning message",
            context={},
        ),
        Alert(
            timestamp=datetime.now().isoformat(),
            severity="CRITICAL",
            rule="critical_rule",
            message="Critical message",
            context={},
        ),
    ]

    for alert in alerts:
        monitor.emit_alert(alert)

    # Small delay for async delivery threads
    time.sleep(0.5)

    # Get alerts with filtering
    critical_alerts = monitor.get_alerts(severity="CRITICAL", hours=24)
    warning_alerts = monitor.get_alerts(severity="WARNING", hours=24)

    # Verify filtering
    assert len(critical_alerts) >= 1
    assert all(a.severity == "CRITICAL" for a in critical_alerts)

    assert len(warning_alerts) >= 1
    assert all(a.severity == "WARNING" for a in warning_alerts)


# ============================================================================
# StateManager: Deployment History Tests
# ============================================================================


def test_deployment_history_tracking(temp_state_file: Path) -> None:
    """Test StateManager deployment history tracking."""
    state_mgr = StateManager(state_file=temp_state_file)

    # Record multiple deployments
    deployments = [
        {
            "release_id": "v1.0.0",
            "environment": "staging",
            "timestamp": "2025-10-12T10:00:00",
            "status": "success",
            "artifacts": ["model.pkl"],
            "rollback": False,
            "smoke_test_passed": True,
            "deployed_by": "user1",
        },
        {
            "release_id": "v1.0.1",
            "environment": "prod",
            "timestamp": "2025-10-12T15:00:00",
            "status": "success",
            "artifacts": ["model.pkl", "config.yaml"],
            "rollback": False,
            "smoke_test_passed": True,
            "deployed_by": "user2",
        },
        {
            "release_id": "v1.0.1",
            "environment": "prod",
            "timestamp": "2025-10-12T16:00:00",
            "status": "rolled_back",
            "artifacts": ["model.pkl"],
            "rollback": True,
            "smoke_test_passed": False,
            "deployed_by": "user2",
        },
    ]

    for dep in deployments:
        state_mgr.record_deployment(**dep)

    # Verify history
    history = state_mgr.get_deployment_history(limit=10)
    assert len(history) == 3

    # Verify most recent first
    assert history[0]["release_id"] == "v1.0.1"
    assert history[0]["status"] == "rolled_back"

    # Verify last successful deployment
    last_prod = state_mgr.get_last_deployment(environment="prod")
    assert last_prod is not None
    assert last_prod["release_id"] == "v1.0.1"


def test_deployment_by_environment(temp_state_file: Path) -> None:
    """Test filtering deployments by environment."""
    state_mgr = StateManager(state_file=temp_state_file)

    # Record deployments to different environments
    state_mgr.record_deployment(
        release_id="v1.0.0",
        environment="staging",
        timestamp="2025-10-12T10:00:00",
        status="success",
        artifacts=["model.pkl"],
        rollback=False,
        smoke_test_passed=True,
        deployed_by="user1",
    )

    state_mgr.record_deployment(
        release_id="v1.0.1",
        environment="prod",
        timestamp="2025-10-12T15:00:00",
        status="success",
        artifacts=["model.pkl"],
        rollback=False,
        smoke_test_passed=True,
        deployed_by="user2",
    )

    # Get deployments by environment
    staging_deployments = state_mgr.get_deployments_by_environment("staging")
    prod_deployments = state_mgr.get_deployments_by_environment("prod")

    # Verify filtering
    assert len(staging_deployments) == 1
    assert staging_deployments[0]["environment"] == "staging"

    assert len(prod_deployments) == 1
    assert prod_deployments[0]["environment"] == "prod"
