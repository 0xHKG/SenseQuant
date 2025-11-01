# SenseQuant Project Analysis & Improvement Recommendations

**Date:** 2025-10-28  
**Status:** Post US-028 Phase 7 Batch 5  
**Coverage:** 96/96 NIFTY100 tradeable symbols ingested (3-year horizon), Batch 4 teacher/student run `live_candidate_20251028_154400` completed with 216/252 windows (85.7%) succeeding and zero failures.

---

## 1. Executive Summary

### Current State ✅
- **Historical Data Platform:** Batch 4 and Batch 5 ingestion complete. Ninety-six tradeable NIFTY 100 constituents verified with 3-year daily data; `data_unavailable` tracks four official tickers lacking Breeze OHLCV.
- **Teacher–Student Pipeline:** Phase 7 run executed end-to-end with promotion briefing, validated artifacts, and telemetry logs. Student models generated for each successful window.
- **Risk & Execution Stack:** Risk manager, position sizing, sentiment gating, and execution adapters are production-grade with structured logging and audit trails.
- **Governance:** Commandments, PRD, and architecture overview refreshed. Promotion bundle (`release/audit_live_candidate_20251028_154400/`) ready for approval.

### Critical Gaps 🔴
1. **Data Gaps:** ADANIGREEN, IDEA, APLAPOLLO, and DIXON lack Breeze OHLCV coverage; portfolio completeness and benchmarking require alternate sourcing or formal exclusion.
2. **GPU Utilisation:** Training still pins to GPU 0 (hardcoded `gpu_device_id`); multi-GPU execution pending despite hardware availability.
3. **Reward Loop Automation:** Reward metrics are recorded but not yet feeding back into weighting, sample selection, or deployment governance.
4. **Stress & Soak Testing:** No automated black-swan scenarios or extended reliability runs have been executed post Phase 7 upgrades.
5. **Operational Dashboards:** Training telemetry is not surfaced in dashboard/alerting flows, limiting observability during long runs.

---

## 2. Recent Deliverables (2025-10-27 → 2025-11-01)

| Deliverable | Outcome | Artifacts |
|-------------|---------|-----------|
| Batch 4 ingestion re-run (OBEROI mapping fix) | 36/36 symbols ingested, coverage 100% | `docs/batch4-ingestion-report.md`, `data/historical/metadata/symbol_mappings_batch4.json` |
| Batch 5 ingestion | 30 outstanding symbols ingested; universe now 96 verified | `docs/batch5-ingestion-report.md`, coverage summaries under `data/historical/metadata/` |
| Teacher/student run `live_candidate_20251028_154400` | Phases 1–7 success, audited bundle produced | `docs/batch4-training-results.md`, `release/audit_live_candidate_20251028_154400/` |
| Training telemetry spike | Streaming `TrainingEvent` schema implemented | `src/services/training_telemetry.py`, `data/analytics/training/` |
| Governance refresh | Commandments, PRD, overview updated | `docs/commandments.md`, `docs/prd.md`, `docs/SenseQuant_Project_Overview.md` |
| Telemetry flush fixes (2025-11-01) | Explicit flush + line buffering + unbuffered output | `src/services/training_telemetry.py`, `scripts/run_historical_training.py`, commit 14614b8 |
| Live order-book provider (2025-11-01) | Breeze API integration with graceful fallback | `src/adapters/market_data_providers.py`, commit 7158b93 |
| Support/resistance analytics (2025-11-01) | 4 long-horizon analytics integrated into swing strategy | `src/domain/support_resistance.py`, `src/domain/strategies/swing.py`, commit a6cf80d |

---

## 3. Performance & Quality Metrics

- **Teacher Windows:** 252 processed across 36 symbols; 216 success, 36 expected skips (insufficient forward data for 2024-12-31 window), 0 failures.
- **Student Training:** 216 student models generated; reward mean 0.0161, zero training failures.
- **Ingestion Runtime:** Batch 4 live ingestion completed in 15m09s; Batch 5 ingestion finished in ~6 minutes leveraging cached chunks (90-day windowing, 2.0s delay).
- **Coverage Audits:** `coverage_summary_20251028_184522.json` confirms 100% coverage for Batch 5 symbol set; Batch 4 audit passes following OBEROI fix.
- **Telemetry Volume:** Multiple `training_run_live_candidate_*.jsonl` files capture phase transitions, window outcomes, and GPU utilisation for dashboards.

---

## 4. Improvement Recommendations

### 4.1 Data Platform
- **Action:** Engage NSE/Breeze support or alternate vendors to source OHLCV for ADANIGREEN, IDEA, APLAPOLLO, DIXON; alternatively escalate exclusion to Risk Committee.
- **Owner:** Data Platform (Gargi)
- **Timeline:** 1 sprint
- **Dependencies:** Vendor access approvals, legal review.

### 4.2 Multi-GPU Execution
- **Action:** Parameterise `gpu_device_id` in `TeacherLabeler` and batch executor, schedule workers across both RTX A6000 GPUs, and add integration tests verifying per-worker GPU allocation.
- **Owner:** Training Engineering (Arjun)
- **Timeline:** Implement before Batch 5 teacher training run.
- **Dependencies:** Update CLI to expose GPU map, ensure telemetry captures per-GPU load.

### 4.3 Reward Loop Enablement
- **Action:** Translate recorded reward metrics into adaptive sample weighting, promotion gating, and rollback criteria. Document QA acceptance thresholds (Sharpe > 0.3, win-rate > 45%, drawdown limits).
- **Owner:** Quant Research (Nisha) with QA (Rahul)
- **Timeline:** 2 sprints
- **Dependencies:** Historical realised P&L dataset, governance approval.

### 4.4 Stress & Reliability Testing
- **Action:** Build Phase 8 orchestrator for black-swan scenarios (2008, 2013, 2020) and schedule 24h soak runs; integrate outputs into promotion briefing and risk dashboards.
- **Owner:** Reliability Engineering (Sana)
- **Timeline:** 2–3 sprints
- **Dependencies:** Scenario datasets, compute allocation, alerting rules.

### 4.5 Telemetry & Ops
- **Action:** Extend Streamlit/ops dashboard with training tab (event timeline, GPU utilisation, skip/fail counts) and configure alerts for ingestion/training anomalies.
- **Owner:** Ops Automation (Vinay)
- **Timeline:** 1 sprint
- **Dependencies:** Finalise `TrainingEvent` schema, integrate with monitoring stack.

### 4.6 Order Book Enablement
- **Action:** Establish a live depth feed (Breeze or alternate), harden ingestion/storage, and enable order book features across training and strategies.
- **Owner:** Market Data Integrations (Priya)
- **Timeline:** 1 sprint once vendor API confirmed
- **Dependencies:** Credential provisioning, latency/quality validation, feature toggle rollout.

### 4.7 Structural Support/Resistance Analytics ✅ COMPLETED
- **Status:** Implemented (2025-11-01)
- **Implementation:** Created `src/domain/support_resistance.py` module with 4 analytics functions:
  - `calculate_52week_levels()`: 52-week high/low with range position
  - `calculate_anchored_vwap()`: Anchored VWAP with 1σ/2σ bands
  - `calculate_volume_profile_levels()`: Volume profile with POC and top-5 levels
  - `calculate_swing_highs_lows()`: Pivot point identification
- **Integration:** Wired into swing strategy `compute_features()` with 252-day lookback
- **Quality:** ruff check PASS, mypy PASS (1 pandas-stubs warning, project-wide), pytest 642/643 PASS
- **Next Steps:** Backtest on representative symbol subset, document parameter tuning, evaluate performance impact
- **Files:** `src/domain/support_resistance.py` (435 lines), `src/domain/strategies/swing.py` (lines 20-24, 107-124)

---

## 5. Risk Register (2025-10-28)

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| Missing data for four NIFTY tickers biases portfolio | Medium | High | Source alternate data or gain formal exclusion approval | Data Platform |
| GPU assignment change introduces instability | Medium | Medium | Canary run, telemetry validation, rollback plan | Training Engineering |
| Reward loop without governance could overfit | High | Medium | Define QA criteria, implement rollback guardrails | Quant Research + QA |
| Lack of stress testing hides tail risks | High | Medium | Prioritise Phase 8 orchestrator and recurring runs | Reliability Engineering |
| Telemetry gaps delay incident detection | Medium | ~~Medium~~ **LOW** | ✅ Flush fixes implemented (commit 14614b8); dashboard integration pending | Ops Automation |
| ~~Stubbed order book feed limits microstructure fidelity~~ | ~~Medium~~ | ~~High~~ | ✅ Live Breeze provider implemented (commit 7158b93); testing with live credentials pending | Market Data Integrations |
| ~~Absence of structural support/resistance features reduces swing robustness~~ | ~~Medium~~ | ~~Medium~~ | ✅ Support/resistance module implemented (commit a6cf80d); backtesting pending | Quant Research |

---

## 6. Compliance Checklist

- ✅ Commandments updated and acknowledged (2025-10-28).
- ✅ PRD, architecture overview, and batch documentation refreshed.
- ✅ Coverage reports stored in `data/historical/metadata/` for audit trail.
- ✅ Promotion briefing reviewed for `live_candidate_20251028_154400`.
- ⏳ Awaiting implementation and QA sign-off for multi-GPU execution plan.
- ⏳ Reward loop specification and acceptance criteria pending stakeholder approval.
