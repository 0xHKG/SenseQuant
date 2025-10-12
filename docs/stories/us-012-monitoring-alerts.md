# US-012: Monitoring & Alerts v1

**Status**: 🚧 In Progress
**Priority**: High
**Estimated Effort**: Large

---

## Problem Statement

Production trading systems require continuous monitoring to detect:
1. System health issues (heartbeat failures, connectivity loss)
2. Risk events (circuit breakers triggered, excessive losses)
3. Data quality problems (stale models, missing sentiment data)
4. Operational anomalies (API rate limits, authentication failures)

Without proactive monitoring, critical issues go undetected until they cause significant losses or system failures.

---

## Objectives

Build a monitoring and alerting system that:
- Collects runtime metrics (positions, PnL, circuit breaker state, connectivity)
- Evaluates alert rules against thresholds
- Emits structured alerts to logs and future notification channels
- Persists metrics for analysis and debugging
- Provides CLI tools for health checks and alert monitoring
- Integrates seamlessly with existing engine without disrupting trading logic

---

## Requirements

### Functional

1. **MonitoringService** (`src/services/monitoring.py`)
   - Collect metrics at configurable intervals (heartbeat, positions, PnL, state)
   - Evaluate alert rules (circuit breakers, sentiment failures, stale artifacts, heartbeat lapse)
   - Write alerts to `logs/alerts/<date>.jsonl` (structured JSONL format)
   - Persist metrics to `data/monitoring/metrics_<timestamp>.json`
   - Support metric snapshots (current state) and time-series aggregation

2. **Alert Rules**
   - Circuit breaker triggered
   - Sentiment provider failures exceed threshold (N failures per time window)
   - Teacher/Student artifacts stale (not updated within X hours)
   - Heartbeat lapsed (no update within interval)
   - Breeze connectivity lost
   - Daily loss exceeds threshold
   - Position limits breached

3. **Metrics Collection**
   - Heartbeat timestamp (last engine tick)
   - Open positions count and value
   - Daily PnL
   - Circuit breaker state (active/inactive)
   - Last sentiment score and provider errors
   - Breeze API connectivity status
   - Teacher/Student model timestamps

4. **Settings Integration** (`src/app/config.py`)
   - Enable/disable monitoring
   - Heartbeat interval (seconds)
   - Alert recipients (stub list for future email/Telegram)
   - Max sentiment failures per window
   - Artifact staleness threshold (hours)
   - Alert rule thresholds (PnL, position limits)

5. **Engine Integration** (`src/services/engine.py`)
   - Publish metrics after each trading tick
   - Non-blocking integration (monitoring doesn't delay trading)
   - Graceful degradation if monitoring fails

6. **CLI Monitoring Tool** (`scripts/monitor.py`)
   - Tail alert logs in real-time
   - Print active alerts summary
   - Run ad-hoc health checks (model freshness, connectivity, circuit breaker state)
   - Query metrics history

### Non-Functional

- **Performance**: Monitoring overhead < 50ms per tick
- **Reliability**: Monitoring failures don't crash trading engine
- **Observability**: All monitoring uses `component="monitoring"`, alerts use `component="alert"`
- **Data Retention**: Keep alerts for 90 days, metrics for 30 days
- **Serialization**: JSONL for alerts (append-only), JSON for metrics snapshots

---

## Architecture Design

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                  src/services/engine.py                      │
│  - Publish metrics after each tick                           │
│  - monitoring_service.record_tick(metrics)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│              src/services/monitoring.py                      │
│  MonitoringService:                                          │
│   - collect_metrics() → current state snapshot              │
│   - evaluate_alerts(metrics) → alert events                 │
│   - persist_metrics(metrics, path)                           │
│   - emit_alert(alert, log_path)                              │
│   - check_artifact_freshness(model_path) → staleness        │
└─────────────────────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│              Persistent Storage                              │
│  - logs/alerts/<date>.jsonl (structured alerts)              │
│  - data/monitoring/metrics_<timestamp>.json (snapshots)      │
└─────────────────────────────────────────────────────────────┘
```

### Alert Rule Evaluation

```python
def evaluate_alerts(metrics: dict) -> list[Alert]:
    alerts = []

    # Rule 1: Circuit breaker triggered
    if metrics["circuit_breaker_active"]:
        alerts.append(Alert(
            severity="CRITICAL",
            rule="circuit_breaker_triggered",
            message="Circuit breaker active - daily loss limit exceeded"
        ))

    # Rule 2: Sentiment failures
    if metrics["sentiment_failures_1h"] > threshold:
        alerts.append(Alert(
            severity="WARNING",
            rule="sentiment_failures_high",
            message=f"Sentiment provider failures: {count}/hour"
        ))

    # Rule 3: Stale artifacts
    if artifact_age_hours > staleness_threshold:
        alerts.append(Alert(
            severity="WARNING",
            rule="stale_teacher_model",
            message=f"Teacher model not updated for {age} hours"
        ))

    return alerts
```

### Metrics Schema

```json
{
  "timestamp": "2025-10-12T12:34:56+05:30",
  "heartbeat": {
    "last_tick": "2025-10-12T12:34:56+05:30",
    "uptime_seconds": 3600
  },
  "positions": {
    "count": 3,
    "total_value": 150000.0,
    "symbols": ["RELIANCE", "TCS", "INFY"]
  },
  "pnl": {
    "daily": 2500.50,
    "unrealized": 1200.00,
    "realized": 1300.50
  },
  "risk": {
    "circuit_breaker_active": false,
    "daily_loss_pct": -0.25,
    "max_position_value": 50000.0
  },
  "sentiment": {
    "last_score": 0.65,
    "provider_errors_1h": 0,
    "cache_hit_rate": 0.85
  },
  "connectivity": {
    "breeze_authenticated": true,
    "last_api_call": "2025-10-12T12:34:50+05:30"
  },
  "artifacts": {
    "teacher_model_age_hours": 12,
    "student_model_age_hours": 6
  }
}
```

### Alert Schema

```json
{
  "timestamp": "2025-10-12T12:35:00+05:30",
  "severity": "CRITICAL",
  "rule": "circuit_breaker_triggered",
  "message": "Circuit breaker active - daily loss limit exceeded",
  "context": {
    "daily_loss_pct": -5.2,
    "threshold": -5.0,
    "open_positions": 3
  },
  "acknowledged": false
}
```

---

## Implementation Plan

### Tasks

1. **Extend Settings** (`src/app/config.py`)
   - Add monitoring configuration section
   - Enable flag, intervals, thresholds, recipients

2. **Implement MonitoringService** (`src/services/monitoring.py`)
   - `collect_metrics()` - gather current state
   - `evaluate_alerts()` - check rules
   - `emit_alert()` - write to JSONL log
   - `persist_metrics()` - save snapshot
   - `check_artifact_freshness()` - validate model ages
   - `get_active_alerts()` - query recent alerts

3. **Integrate with Engine** (`src/services/engine.py`)
   - Add optional monitoring_service parameter
   - Call `record_tick()` after trading logic
   - Handle monitoring failures gracefully

4. **Create CLI Monitor** (`scripts/monitor.py`)
   - `tail` - follow alert log in real-time
   - `status` - show active alerts
   - `health` - run health checks
   - `metrics` - query recent metrics

5. **Write Tests**
   - Unit: Alert rule evaluation, metric collection, staleness detection
   - Integration: Full engine + monitoring flow, alert emission, file persistence

6. **Update Documentation** (`docs/architecture.md`)
   - Add Monitoring & Alerts section
   - Diagram showing integration points

---

## Acceptance Criteria

### Must Have
- [ ] MonitoringService collects metrics (heartbeat, positions, PnL, circuit breaker)
- [ ] Alert rules evaluate correctly (circuit breaker, sentiment failures, stale artifacts)
- [ ] Alerts written to `logs/alerts/<date>.jsonl` in JSONL format
- [ ] Metrics persisted to `data/monitoring/metrics_<timestamp>.json`
- [ ] Settings extended with monitoring configuration
- [ ] Engine integration via `record_tick()` without blocking trading
- [ ] CLI tool with `tail`, `status`, `health` commands
- [ ] Logging uses `component="monitoring"` and `component="alert"`
- [ ] All quality gates pass (ruff, mypy, pytest)

### Should Have
- [ ] Artifact freshness checks for Teacher/Student models
- [ ] Heartbeat lapse detection
- [ ] Sentiment provider failure tracking
- [ ] Alert severity levels (INFO, WARNING, CRITICAL)
- [ ] Graceful degradation on monitoring failures

### Nice to Have
- [ ] Alert deduplication (don't repeat same alert)
- [ ] Alert acknowledgment tracking
- [ ] Metrics time-series aggregation
- [ ] Dashboard visualization (future story)

---

## Test Strategy

### Unit Tests
- Alert rule evaluation (each rule independently)
- Metric collection from mock engine state
- Artifact staleness calculation
- Alert serialization to JSONL
- Settings validation

### Integration Tests
- Engine + monitoring full flow
- Alert emission and file persistence
- Multiple ticks generating time-series metrics
- Health check CLI commands
- Alert log tailing

### Edge Cases
- Monitoring service initialization failure
- Alert log directory doesn't exist
- Stale artifact files missing
- Heartbeat never initialized
- Invalid metrics format

---

## Success Metrics

- Monitoring overhead < 50ms per engine tick
- All critical events (circuit breaker, connectivity loss) trigger alerts
- Zero monitoring-related engine crashes
- 100% test coverage on alert rules
- CLI health checks return correct status

---

## Future Enhancements (Post-US-012)

- **US-013**: Email/Telegram notification channels
- **US-014**: Prometheus metrics export for Grafana
- **US-015**: Alert escalation and on-call rotation
- **US-016**: Anomaly detection with ML (unusual PnL patterns)
- **US-017**: Web dashboard for real-time monitoring
- **US-018**: Historical metrics querying and charting

---

## References

- Engine: `src/services/engine.py`
- Risk Manager: `src/services/risk_manager.py`
- Settings: `src/app/config.py`
- Logging: Uses loguru with structured extra fields
