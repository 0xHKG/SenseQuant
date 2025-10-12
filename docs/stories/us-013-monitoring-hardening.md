# US-013 — Monitoring & Alerts v2 (Enterprise Hardening)

## Problem Statement

US-012 delivered core monitoring (metrics collection, alert rules, JSONL persistence, basic CLI). However, production-grade trading systems require:
- **Metric aggregation & analytics**: Min/max/avg rollups for time-series analysis
- **Retention management**: Configurable caps, archival to compressed storage
- **Performance tracking**: Tick latency, sentiment API response times, order latency
- **Alert delivery**: Email, Slack/webhook integrations (not just local logs)
- **Alert lifecycle**: Acknowledgement workflow to prevent notification spam
- **Enhanced observability**: Real-time monitoring, CSV exports, aggregated health dashboards

Without these capabilities, operators cannot:
- Diagnose performance regressions or bottlenecks
- Manage alert fatigue in production
- Export metrics for external analysis tools
- Receive timely notifications for critical issues
- Track system health trends over time

## Objectives

1. **Metric Rollups**: Compute min/max/avg statistics per configurable time interval
2. **Retention Policy**: Automatic archival of old metrics to compressed JSON
3. **Performance Metrics**: Track and expose tick latency, sentiment latency, order response times
4. **Alert Delivery**: Plugin architecture for email, Slack, webhook notifications
5. **Alert Acknowledgement**: Allow operators to ack alerts, prevent re-notification
6. **Enhanced CLI**: Subcommands for alert management, metrics export (CSV), real-time watching
7. **Architecture Documentation**: Update docs/architecture.md with monitoring capabilities

## Requirements

### Functional Requirements

#### FR-1: Metric Aggregation
- Compute rollup statistics (min/max/avg/count) per configurable interval (default: 5min)
- Store aggregated metrics separately from raw metrics
- Support queries for aggregated metrics by time range
- Include rollups for: PnL, position count, tick latency, sentiment latency

#### FR-2: Retention Policy
- Configurable max raw metrics to retain in memory (default: 100)
- Configurable max archived metrics files (default: 30 days)
- Archive old raw metrics to compressed JSON (`data/monitoring/archive/metrics_YYYY-MM-DD.json.gz`)
- Automatic cleanup of archives older than retention period

#### FR-3: Performance Metrics
- Track and record:
  - Tick processing latency (intraday & swing)
  - Sentiment provider API latency
  - Order placement/execution latency (if available from Breeze)
  - Strategy execution time
- Expose performance metrics in aggregated rollups
- Alert on performance degradation (configurable thresholds)

#### FR-4: Alert Delivery Plugins
- Plugin interface for alert delivery channels
- Built-in plugins (stubs for production configuration):
  - **EmailPlugin**: SMTP-based email alerts
  - **SlackPlugin**: Webhook-based Slack notifications
  - **WebhookPlugin**: Generic HTTP webhook POST
- Configuration-driven channel selection per alert severity
- Graceful failure handling (log errors, don't crash monitoring)
- Delivery retry logic with exponential backoff

#### FR-5: Alert Acknowledgement
- CLI command to acknowledge alerts by rule or timestamp
- Store acknowledgements in JSONL with operator info
- Prevent re-notification of acknowledged alerts
- Auto-clear acknowledgements after configurable TTL
- Track acknowledgement history for audit trail

#### FR-6: Enhanced CLI
- **`alerts` subcommand**:
  - `alerts list [--severity LEVEL] [--hours N]` - List active alerts
  - `alerts ack <rule> [--all]` - Acknowledge alerts
  - `alerts clear <rule>` - Clear acknowledgements
- **`metrics` subcommand**:
  - `metrics show [--interval 5m|1h|1d]` - Show aggregated metrics
  - `metrics export [--format csv|json] [--hours N]` - Export metrics
  - `metrics summary` - Statistical summary (min/max/avg)
- **`watch` subcommand**:
  - Real-time alert monitoring using filesystem events (watchdog)
  - Color-coded severity display
  - Filter by severity, rule pattern
- **`status` subcommand**:
  - Aggregated health dashboard
  - Alert summary by severity
  - Performance metrics summary
  - System health score

### Non-Functional Requirements

#### NFR-1: Performance
- Metric aggregation should not impact tick processing (<10ms overhead)
- Alert delivery should be asynchronous (non-blocking)
- Archive operations should run in background

#### NFR-2: Reliability
- Alert delivery failures must not crash monitoring service
- Retention cleanup must handle corrupted/missing files gracefully
- All external integrations must have timeout limits

#### NFR-3: Configurability
- All thresholds, intervals, retention periods configurable via Settings
- Alert delivery channels configurable per severity level
- Plugin enable/disable flags in config

#### NFR-4: Observability
- Log all monitoring operations with `component="monitoring"`
- Track delivery success/failure rates
- Expose monitoring service health via health checks

## Architecture

### Component Design

```
┌─────────────────────────────────────────────────────────────┐
│                    MonitoringService                         │
├─────────────────────────────────────────────────────────────┤
│  Core (Existing)                                             │
│  - record_tick()        - evaluate_alerts()                  │
│  - emit_alert()         - check_artifact_freshness()        │
│                                                              │
│  NEW: Metric Aggregation                                     │
│  - _compute_rollups()         [min/max/avg per interval]    │
│  - _get_aggregated_metrics()  [query by time range]         │
│  - rollup_history: list[MetricRollup]                       │
│                                                              │
│  NEW: Retention Management                                   │
│  - _archive_old_metrics()     [compress to .json.gz]        │
│  - _cleanup_old_archives()    [enforce retention policy]    │
│  - archive_dir: Path                                         │
│                                                              │
│  NEW: Performance Tracking                                   │
│  - record_performance_metric()                               │
│  - performance_metrics: dict[str, list[float]]              │
│  - _aggregate_performance_stats()                            │
│                                                              │
│  NEW: Alert Delivery                                         │
│  - delivery_plugins: list[AlertDeliveryPlugin]              │
│  - _deliver_alert()           [async plugin invocation]     │
│  - delivery_queue: queue.Queue                               │
│                                                              │
│  NEW: Alert Acknowledgement                                  │
│  - acknowledge_alert()                                       │
│  - clear_acknowledgement()                                   │
│  - acknowledgements: dict[str, AckRecord]                   │
│  - ack_log_path: Path                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Alert Delivery Plugin Interface                 │
├─────────────────────────────────────────────────────────────┤
│  AlertDeliveryPlugin (ABC)                                   │
│    - name: str                                               │
│    - enabled: bool                                           │
│    - deliver(alert: Alert) -> bool                          │
│                                                              │
│  EmailPlugin(AlertDeliveryPlugin)                           │
│    - smtp_host, smtp_port, smtp_user, smtp_password        │
│    - from_addr, to_addrs                                    │
│    - deliver() -> send via SMTP with retry                  │
│                                                              │
│  SlackPlugin(AlertDeliveryPlugin)                           │
│    - webhook_url                                             │
│    - deliver() -> POST to Slack webhook                     │
│                                                              │
│  WebhookPlugin(AlertDeliveryPlugin)                         │
│    - url, headers, auth                                      │
│    - deliver() -> POST alert JSON to webhook                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Enhanced CLI                             │
├─────────────────────────────────────────────────────────────┤
│  scripts/monitor.py                                          │
│    - alerts list/ack/clear                                   │
│    - metrics show/export/summary                             │
│    - watch [real-time with watchdog]                         │
│    - status [aggregated dashboard]                           │
│    - health [existing health checks]                         │
└─────────────────────────────────────────────────────────────┘
```

### Data Structures

#### MetricRollup
```python
@dataclass
class MetricRollup:
    interval_start: str      # ISO timestamp
    interval_end: str        # ISO timestamp
    interval_seconds: int    # e.g., 300 for 5min
    metrics: dict[str, RollupStats]

@dataclass
class RollupStats:
    min: float
    max: float
    avg: float
    count: int
    sum: float
```

#### AckRecord
```python
@dataclass
class AckRecord:
    rule: str
    acknowledged_at: str     # ISO timestamp
    acknowledged_by: str     # operator name/ID
    ttl_seconds: int         # auto-clear after TTL
    reason: str | None       # optional ack reason
```

#### PerformanceMetric
```python
@dataclass
class PerformanceMetric:
    name: str                # "tick_latency", "sentiment_latency", etc.
    timestamp: str
    value_ms: float
    context: dict[str, Any]  # e.g., {"strategy": "intraday", "symbol": "RELIANCE"}
```

### Configuration Schema

```python
# Settings extension
class Settings(BaseSettings):
    # ... existing monitoring settings ...

    # NEW: Aggregation
    monitoring_aggregation_interval_seconds: int = Field(300, ge=60, le=3600)
    monitoring_enable_aggregation: bool = Field(True)

    # NEW: Retention
    monitoring_max_raw_metrics: int = Field(100, ge=50, le=1000)
    monitoring_max_archive_days: int = Field(30, ge=1, le=365)
    monitoring_archive_interval_hours: int = Field(24, ge=1, le=168)

    # NEW: Performance
    monitoring_enable_performance_tracking: bool = Field(True)
    monitoring_performance_alert_threshold_ms: float = Field(1000.0, ge=100.0, le=10000.0)

    # NEW: Alert Delivery
    monitoring_enable_email_alerts: bool = Field(False)
    monitoring_email_smtp_host: str = Field("smtp.gmail.com")
    monitoring_email_smtp_port: int = Field(587)
    monitoring_email_from: str = Field("")
    monitoring_email_to: list[str] = Field(default_factory=list)

    monitoring_enable_slack_alerts: bool = Field(False)
    monitoring_slack_webhook_url: str = Field("")

    monitoring_enable_webhook_alerts: bool = Field(False)
    monitoring_webhook_url: str = Field("")
    monitoring_webhook_headers: dict[str, str] = Field(default_factory=dict)

    # NEW: Acknowledgement
    monitoring_ack_ttl_seconds: int = Field(86400, ge=3600, le=604800)  # 1 day default
```

## Implementation Plan

### Phase 1: Core Enhancements (MonitoringService)
1. Implement `MetricRollup` and rollup computation
2. Add retention policy with archival to `.json.gz`
3. Add performance metric tracking
4. Implement alert acknowledgement storage/retrieval

### Phase 2: Alert Delivery Plugins
1. Create `AlertDeliveryPlugin` ABC
2. Implement `EmailPlugin` (stub with SMTP)
3. Implement `SlackPlugin` (webhook POST)
4. Implement `WebhookPlugin` (generic HTTP POST)
5. Integrate delivery pipeline into `emit_alert()`

### Phase 3: Enhanced CLI
1. Refactor CLI with subparsers for commands
2. Implement `alerts` subcommand (list/ack/clear)
3. Implement `metrics` subcommand (show/export/summary)
4. Implement `watch` subcommand (real-time with watchdog)
5. Enhance `status` subcommand with aggregated metrics

### Phase 4: Engine Integration
1. Update Engine to track tick start/end times
2. Update Engine to track sentiment API latency
3. Record performance metrics via MonitoringService

### Phase 5: Testing & Documentation
1. Unit tests for rollups, retention, delivery plugins, acknowledgement
2. Integration tests for end-to-end workflows
3. Update architecture.md with monitoring section
4. Quality gates (ruff, mypy, pytest)

## Acceptance Criteria

### AC-1: Metric Aggregation
- [ ] MonitoringService computes rollups every N seconds (configurable)
- [ ] Rollups include min/max/avg/count for key metrics (PnL, positions, latency)
- [ ] CLI can query and display aggregated metrics
- [ ] CSV export includes aggregated metrics

### AC-2: Retention Policy
- [ ] Raw metrics limited to configurable cap (default 100)
- [ ] Old metrics archived to `data/monitoring/archive/metrics_YYYY-MM-DD.json.gz`
- [ ] Archives older than retention period automatically cleaned up
- [ ] Archival runs asynchronously without blocking tick processing

### AC-3: Performance Tracking
- [ ] Engine records tick processing latency
- [ ] Engine records sentiment provider API latency
- [ ] Performance metrics included in rollups
- [ ] Alert triggered when latency exceeds threshold

### AC-4: Alert Delivery
- [ ] EmailPlugin sends alerts via SMTP (stub tested with local SMTP server)
- [ ] SlackPlugin posts to webhook URL
- [ ] WebhookPlugin POSTs JSON to generic endpoint
- [ ] Delivery failures logged but don't crash monitoring
- [ ] Delivery plugins configurable via Settings

### AC-5: Alert Acknowledgement
- [ ] CLI command `alerts ack <rule>` acknowledges alerts
- [ ] Acknowledged alerts not re-notified until TTL expires
- [ ] Acknowledgements stored in `logs/alerts/acknowledgements.jsonl`
- [ ] CLI command `alerts clear <rule>` clears acknowledgements

### AC-6: Enhanced CLI
- [ ] `alerts list` shows active alerts with severity filtering
- [ ] `alerts ack <rule>` acknowledges specific rule
- [ ] `metrics show` displays aggregated statistics
- [ ] `metrics export --format csv` exports to CSV
- [ ] `watch` tails alerts in real-time with color coding
- [ ] `status` shows aggregated health dashboard

### AC-7: Tests & Quality
- [ ] Unit tests for rollup computation, retention, delivery, acknowledgement
- [ ] Integration tests for full monitoring pipeline
- [ ] All tests pass (pytest -q)
- [ ] Type checking passes (mypy src)
- [ ] Linting passes (ruff check/format)

### AC-8: Documentation
- [ ] architecture.md updated with Monitoring section
- [ ] US-013 story document complete
- [ ] CLI help text updated with new subcommands

## Technical Risks & Mitigations

### Risk 1: Performance Overhead
**Risk**: Metric aggregation and archival may slow down tick processing
**Mitigation**:
- Run aggregation asynchronously in background thread
- Use rolling window for aggregation (don't recompute all history)
- Archive operations run on separate schedule (e.g., nightly)

### Risk 2: Alert Delivery Failures
**Risk**: External services (SMTP, Slack) may be unavailable
**Mitigation**:
- All delivery wrapped in try-catch with timeout
- Retry logic with exponential backoff (max 3 retries)
- Fallback to local JSONL logging always enabled
- Delivery failures logged but don't crash monitoring

### Risk 3: Configuration Complexity
**Risk**: Many new config options may confuse users
**Mitigation**:
- Sensible defaults for all settings
- Clear documentation in .env.example
- Validation with helpful error messages
- CLI help text with examples

## Future Enhancements (Out of Scope for US-013)

- Grafana/Prometheus integration for dashboards
- Machine learning-based anomaly detection
- Multi-channel alert routing rules (different channels per severity)
- Alert escalation policies (page on-call after N minutes)
- Web-based monitoring UI (Flask/FastAPI dashboard)
- Distributed tracing integration (OpenTelemetry)
- Alert correlation and grouping (reduce noise)
- SLA/SLO tracking and reporting

## Dependencies

- **watchdog**: For filesystem event monitoring in `watch` command
- **requests**: For webhook HTTP POST delivery
- **smtplib**: Built-in Python library for email delivery

## Estimated Effort

- Phase 1 (Core): 4-6 hours
- Phase 2 (Delivery): 2-3 hours
- Phase 3 (CLI): 3-4 hours
- Phase 4 (Engine): 1-2 hours
- Phase 5 (Tests/Docs): 3-4 hours

**Total**: 13-19 hours (2-3 development days)

## Success Metrics

1. **Alert Response Time**: Operators notified within 30 seconds of critical alerts
2. **System Observability**: 100% visibility into tick latency, sentiment latency, PnL
3. **Alert Fatigue Reduction**: Acknowledgement workflow reduces duplicate notifications by >80%
4. **Operational Efficiency**: CLI exports enable faster incident investigation
5. **Test Coverage**: >90% code coverage for monitoring module
