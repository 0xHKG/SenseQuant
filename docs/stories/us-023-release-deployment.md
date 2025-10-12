# US-023: Release Deployment Automation & Post-Deploy Monitoring

**Epic**: Production Deployment & Operations
**Status**: IN PROGRESS
**Priority**: HIGH
**Created**: 2025-10-12
**Updated**: 2025-10-12
**Depends On**: US-022 (Release Audit)

---

## Problem Statement

After completing release audits (US-022), we need automated deployment workflows to safely promote configurations and models to production. Manual deployment steps are error-prone and lack traceability. Additionally, we need heightened monitoring immediately after deployment to catch regressions early.

Without deployment automation and post-deploy monitoring:
- Manual errors during config/model promotion
- No audit trail of what was deployed and when
- Regressions may go undetected for days
- Rollback procedures are undocumented or inconsistent

## Acceptance Criteria

### 1. Release Automation (Makefile/Shell Script)
- [ ] Create `Makefile` with release pipeline targets:
  - `make release-audit` - Run audit bundle generation
  - `make release-validate` - Verify promotion checklist + run dryrun backtest
  - `make release-manifest` - Generate signed release manifest
  - `make release-deploy` - Execute deployment (with confirmation)
  - `make release-rollback` - Automated rollback to previous release
- [ ] Release manifest schema (`release/manifests/<timestamp>.yaml`):
  - Release ID (timestamp-based)
  - Audit bundle reference
  - Promoted artifacts with SHA256 hashes (configs, models, notebooks)
  - Deployment metadata (deployer, timestamp, approval sign-offs)
  - Rollback plan (previous release ID, restore commands)

### 2. Monitoring Integration
- [ ] Extend `MonitoringService` with release tracking:
  - `register_release(release_id, manifest)` - Register new deployment
  - `get_active_release()` - Retrieve current release metadata
  - Heightened monitoring for 48 hours post-deploy:
    - Shorter rolling windows (6h vs 24h for intraday, 24h vs 90d for swing)
    - Lower alert thresholds (trigger at 5% degradation vs 10%)
    - Increased alert frequency (every 2h vs 6h)
- [ ] Store release history in `data/monitoring/releases/`
- [ ] Automatic transition to normal monitoring after 48h

### 3. Dashboard Enhancement
- [ ] Add "Active Release" panel to telemetry dashboard:
  - Display current release ID and deployment timestamp
  - Show post-deploy monitoring status (heightened/normal)
  - Time remaining in heightened monitoring window
  - Quick links to release manifest and audit bundle
- [ ] Real-time alert feed for post-deploy period
- [ ] Rollback button (confirmation required)

### 4. Integration Testing
- [ ] Create `tests/integration/test_release_pipeline.py`:
  - `test_release_manifest_generation()` - Verify manifest structure and hashes
  - `test_monitoring_release_registration()` - Mock deployment and check monitoring state
  - `test_heightened_monitoring_alerts()` - Trigger alerts during post-deploy window
  - `test_rollback_plan_execution()` - Simulate rollback workflow

### 5. Documentation
- [ ] Create `docs/stories/us-023-release-deployment.md` (this document)
- [ ] Update `docs/architecture.md` with Section 10: "Release Deployment Workflow"
- [ ] Document approval gates and sign-off procedures
- [ ] Create runbook for common deployment scenarios

---

## Technical Design

### Release Manifest Schema

```yaml
# release/manifests/release_20251012_190000.yaml

release_id: release_20251012_190000
release_type: minor  # major | minor | hotfix
audit_bundle: release/audit_20251012_183000
deployment:
  timestamp: 2025-10-12T19:00:00+05:30
  deployer: engineering_lead
  environment: production

approvals:
  - role: Engineering Lead
    name: John Doe
    email: john@example.com
    timestamp: 2025-10-12T18:45:00+05:30
    signature: sha256:abc123...
  - role: Risk Manager
    name: Jane Smith
    email: jane@example.com
    timestamp: 2025-10-12T18:50:00+05:30
    signature: sha256:def456...

artifacts:
  configs:
    - path: src/app/config.py
      hash: sha256:1a2b3c4d...
      backup: release/backups/release_20251012_190000/config.py
    - path: search_space.yaml
      hash: sha256:5e6f7g8h...
      backup: release/backups/release_20251012_190000/search_space.yaml

  models:
    - path: data/models/student_model.pkl
      hash: sha256:9i0j1k2l...
      version: v1.0_20251010
      backup: release/backups/release_20251012_190000/student_model.pkl

  notebooks:
    - path: notebooks/accuracy_report.ipynb
      hash: sha256:3m4n5o6p...
      backup: release/backups/release_20251012_190000/accuracy_report.ipynb

rollback_plan:
  previous_release_id: release_20251005_180000
  previous_manifest: release/manifests/release_20251005_180000.yaml
  restore_commands:
    - "cp release/backups/release_20251005_180000/config.py src/app/config.py"
    - "cp release/backups/release_20251005_180000/student_model.pkl data/models/student_model.pkl"
  verification_steps:
    - "python scripts/release_audit.py --validate-only"
    - "make test"

monitoring:
  heightened_period_hours: 48
  heightened_start: 2025-10-12T19:00:00+05:30
  heightened_end: 2025-10-14T19:00:00+05:30
  alert_thresholds:
    intraday_hit_ratio_drop: 0.05  # 5% vs normal 10%
    swing_precision_drop: 0.05
  alert_frequency_hours: 2  # vs normal 6h

metadata:
  git_commit: a1b2c3d4e5f6...
  git_branch: release/v1.2.0
  git_tag: v1.2.0
  build_id: build_12345
  jira_tickets: [SQNT-123, SQNT-124, SQNT-125]
```

### Makefile Targets

```makefile
# Makefile for SenseQuant Release Automation

.PHONY: help release-audit release-validate release-manifest release-deploy release-rollback

help:
	@echo "SenseQuant Release Automation"
	@echo ""
	@echo "Available targets:"
	@echo "  release-audit      - Generate release audit bundle"
	@echo "  release-validate   - Run promotion checklist and dryrun backtest"
	@echo "  release-manifest   - Generate signed release manifest"
	@echo "  release-deploy     - Deploy to production (requires confirmation)"
	@echo "  release-rollback   - Rollback to previous release"
	@echo "  release-status     - Show current release status"

# Phase 1: Generate audit bundle
release-audit:
	@echo "==== Phase 1: Release Audit ===="
	python scripts/release_audit.py
	@echo "✅ Audit bundle generated"

# Phase 2: Validation
release-validate: release-audit
	@echo "==== Phase 2: Release Validation ===="
	@echo "Running promotion checklist verification..."
	python scripts/validate_promotion.py
	@echo "Running dryrun backtest..."
	python scripts/backtest.py --config config/dryrun_backtest.yaml --dry-run
	@echo "✅ Validation complete"

# Phase 3: Generate manifest
release-manifest: release-validate
	@echo "==== Phase 3: Release Manifest Generation ===="
	python scripts/generate_manifest.py
	@echo "✅ Release manifest generated"

# Phase 4: Deploy (with confirmation)
release-deploy: release-manifest
	@echo "==== Phase 4: Production Deployment ===="
	@echo "⚠️  WARNING: About to deploy to PRODUCTION"
	@echo ""
	@read -p "Are you sure? (type 'yes' to continue): " confirm && [ "$$confirm" = "yes" ]
	python scripts/deploy_release.py
	@echo "✅ Deployment complete"
	@echo ""
	@echo "Next steps:"
	@echo "1. Monitor dashboard for 48 hours (heightened monitoring active)"
	@echo "2. Review post-deploy metrics every 2 hours"
	@echo "3. Be ready to execute 'make release-rollback' if issues detected"

# Rollback
release-rollback:
	@echo "==== Release Rollback ===="
	@echo "⚠️  WARNING: About to ROLLBACK to previous release"
	@echo ""
	@read -p "Are you sure? (type 'yes' to continue): " confirm && [ "$$confirm" = "yes" ]
	python scripts/rollback_release.py
	@echo "✅ Rollback complete"

# Status
release-status:
	@echo "==== Current Release Status ===="
	python scripts/release_status.py
```

### MonitoringService Extensions

```python
# src/services/monitoring.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import yaml

@dataclass
class ReleaseInfo:
    """Active release information."""
    release_id: str
    deployment_timestamp: datetime
    manifest_path: Path
    heightened_monitoring_active: bool
    heightened_monitoring_end: datetime

    def is_in_heightened_period(self) -> bool:
        """Check if still in 48h heightened monitoring window."""
        return datetime.now() < self.heightened_monitoring_end

class MonitoringService:
    def __init__(self, settings: Settings):
        # ... existing initialization ...
        self.releases_dir = Path("data/monitoring/releases")
        self.releases_dir.mkdir(parents=True, exist_ok=True)
        self.active_release: ReleaseInfo | None = None
        self._load_active_release()

    def _load_active_release(self) -> None:
        """Load active release from disk."""
        active_file = self.releases_dir / "active_release.yaml"
        if active_file.exists():
            with open(active_file) as f:
                data = yaml.safe_load(f)
            self.active_release = ReleaseInfo(**data)

    def register_release(
        self, release_id: str, manifest_path: Path,
        heightened_hours: int = 48
    ) -> None:
        """Register new production release and activate heightened monitoring.

        Args:
            release_id: Unique release identifier
            manifest_path: Path to release manifest
            heightened_hours: Duration of heightened monitoring (default 48h)
        """
        now = datetime.now()

        self.active_release = ReleaseInfo(
            release_id=release_id,
            deployment_timestamp=now,
            manifest_path=manifest_path,
            heightened_monitoring_active=True,
            heightened_monitoring_end=now + timedelta(hours=heightened_hours)
        )

        # Persist to disk
        active_file = self.releases_dir / "active_release.yaml"
        with open(active_file, 'w') as f:
            yaml.dump(asdict(self.active_release), f)

        # Log deployment
        logger.warning(
            f"🚀 Production release deployed: {release_id}",
            extra={
                "component": "monitoring",
                "release_id": release_id,
                "heightened_monitoring_hours": heightened_hours
            }
        )

    def get_active_release(self) -> ReleaseInfo | None:
        """Get currently active release info."""
        if self.active_release and not self.active_release.is_in_heightened_period():
            # Transition to normal monitoring
            self.active_release.heightened_monitoring_active = False
            self._save_active_release()
        return self.active_release

    def _get_alert_thresholds(self) -> dict[str, float]:
        """Get alert thresholds based on monitoring mode."""
        if self.active_release and self.active_release.is_in_heightened_period():
            # Heightened monitoring: stricter thresholds
            return {
                "intraday_hit_ratio_drop": 0.05,  # 5%
                "swing_precision_drop": 0.05,
                "intraday_sharpe_drop": 0.15,
            }
        else:
            # Normal monitoring: standard thresholds
            return {
                "intraday_hit_ratio_drop": 0.10,  # 10%
                "swing_precision_drop": 0.10,
                "intraday_sharpe_drop": 0.25,
            }
```

### Dashboard Release Panel

```python
# dashboards/telemetry_dashboard.py

def render_release_panel(monitoring_service: MonitoringService) -> None:
    """Render active release monitoring panel."""
    st.header("🚀 Active Release")

    release = monitoring_service.get_active_release()

    if release:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Release ID", release.release_id)

        with col2:
            deployed_ago = datetime.now() - release.deployment_timestamp
            st.metric("Deployed", f"{deployed_ago.days}d {deployed_ago.seconds//3600}h ago")

        with col3:
            if release.heightened_monitoring_active:
                time_left = release.heightened_monitoring_end - datetime.now()
                hours_left = time_left.total_seconds() / 3600
                st.metric(
                    "Heightened Monitoring",
                    f"{hours_left:.1f}h remaining",
                    delta="Active",
                    delta_color="normal"
                )
            else:
                st.metric("Monitoring Mode", "Normal")

        # Release details
        st.subheader("Release Details")
        st.write(f"**Manifest:** `{release.manifest_path}`")

        # Load manifest
        if release.manifest_path.exists():
            with open(release.manifest_path) as f:
                manifest = yaml.safe_load(f)

            st.write(f"**Audit Bundle:** `{manifest.get('audit_bundle', 'N/A')}`")

            # Approvals
            if manifest.get('approvals'):
                st.write("**Approvals:**")
                for approval in manifest['approvals']:
                    st.write(f"  - {approval['role']}: {approval['name']} ({approval['timestamp']})")

        # Rollback button
        if st.button("🔄 Rollback Release", type="secondary"):
            st.warning("Rollback initiated! Execute: `make release-rollback`")
    else:
        st.info("No active release registered. Deploy using `make release-deploy`")
```

---

## Implementation Plan

### Phase 1: Manifest Generation (Day 1)
- [x] Design release manifest schema (YAML)
- [ ] Create `scripts/generate_manifest.py`:
  - Compute SHA256 hashes for all artifacts
  - Load approval data from audit bundle
  - Generate rollback plan
  - Write signed manifest
- [ ] Create backup mechanism (copy artifacts to `release/backups/`)

### Phase 2: Makefile & Scripts (Day 1-2)
- [ ] Create `Makefile` with all release targets
- [ ] Create `scripts/validate_promotion.py` - Check promotion checklist
- [ ] Create `scripts/deploy_release.py` - Execute deployment
- [ ] Create `scripts/rollback_release.py` - Automated rollback
- [ ] Create `scripts/release_status.py` - Display release info

### Phase 3: Monitoring Integration (Day 2)
- [ ] Extend `MonitoringService` with release tracking
- [ ] Implement `register_release()` and `get_active_release()`
- [ ] Add heightened monitoring logic (thresholds, windows)
- [ ] Persist release history to disk

### Phase 4: Dashboard Updates (Day 2)
- [ ] Add "Active Release" panel to telemetry dashboard
- [ ] Display release metadata and monitoring status
- [ ] Add rollback button with confirmation
- [ ] Real-time alert feed for post-deploy period

### Phase 5: Integration Testing (Day 3)
- [ ] Create `tests/integration/test_release_pipeline.py`
- [ ] Test manifest generation with hash verification
- [ ] Test monitoring registration and heightened alerts
- [ ] Test rollback plan execution
- [ ] Test 48h transition to normal monitoring

### Phase 6: Documentation (Day 3)
- [ ] Complete US-023 story document
- [ ] Update architecture.md Section 10
- [ ] Create deployment runbook
- [ ] Document approval workflow

---

## Testing Strategy

### Unit Tests
- Manifest generation with various artifact sets
- Hash computation and verification
- Heightened monitoring threshold calculation
- Release info serialization/deserialization

### Integration Tests
- End-to-end release pipeline (audit → manifest → deploy → monitor)
- Rollback workflow with verification
- Heightened monitoring alert triggers
- 48h transition to normal monitoring

### Manual Testing
- Execute `make release-deploy` on staging environment
- Verify dashboard displays release info correctly
- Trigger alerts during heightened period
- Test rollback procedure

---

## Dependencies

- US-022: Release Audit (audit bundles required)
- US-021: Model Promotion (promotion checklists)
- US-017: Telemetry & Monitoring (MonitoringService)

---

## Success Metrics

- Release deployment time < 10 minutes (automated)
- Zero manual errors in deployment process
- 100% of deployments have signed manifests with hashes
- Regressions detected within 2 hours (heightened monitoring)
- Rollback execution time < 5 minutes (automated)

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Deployment script fails mid-execution | HIGH | Atomic operations with transaction log; auto-rollback on failure |
| Heightened monitoring generates false positives | MEDIUM | Tune thresholds based on historical data; require 3 consecutive alerts |
| Manifest tampering | MEDIUM | SHA256 hashes + GPG signatures; verify on every read |
| Rollback corrupts state | HIGH | Backup verification before restore; dryrun mode for testing |

---

## Future Enhancements

- **Blue-Green Deployment**: Run old and new configs in parallel for A/B testing
- **Canary Releases**: Deploy to 10% of capital first, full rollout if metrics hold
- **Auto-Rollback**: Automatic rollback if 3 critical alerts within 1 hour
- **Release Analytics**: Track deployment frequency, success rate, MTTR trends
- **CI/CD Integration**: Trigger releases from GitHub Actions on tag push

---

## References

- [US-022: Release Audit](us-022-release-audit.md)
- [US-021: Model Promotion](us-021-model-promotion.md)
- [Architecture Documentation](../architecture.md)
- [Deployment Runbook](../runbooks/deployment.md) (TBD)
