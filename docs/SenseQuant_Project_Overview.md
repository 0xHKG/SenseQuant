# SenseQuant Project Overview

> **Audience:** Engineering, Data Science, Product, Operations  
> **Document Type:** Development & Operations Reference  
> **Last Updated:** 2025-10-28  
> **Maintainer:** SenseQuant Core Team (`claude.md#Contacts`)

---

## 1. Executive Summary

SenseQuant delivers an end-to-end algorithmic trading platform for Indian equities built on a teacher–student pipeline with auditable BMAD governance. As of 28 October 2025 we have completed a full historical training run across the NIFTY 100 universe (96 tradeable constituents plus metals ETFs), validated release artifacts, and instrumented telemetry for Phase 7 training.

**October 2025 Highlights**

- **NIFTY 100 Coverage:** 96 verified constituents with 3-year OHLCV history (2022-01-01 through 2025-10-28) across `1day` and `5minute` intervals. Four official tickers (ADANIGREEN, IDEA, APLAPOLLO, DIXON) remain data-unavailable via Breeze and are tracked in metadata for follow-up.
- **Teacher/Student Run:** `live_candidate_20251028_154400` completed phases 1–7 with 216/252 windows succeeding (36 expected skips, 0 failures) and produced audited artifacts ready for promotion.
- **Telemetry:** New `TrainingEvent` stream records per-window progress and GPU metrics under `data/analytics/training/`, enabling dashboards to visualise training health.
- **Governance Refresh:** Commandments, PRD, and architecture overview updated to reflect current operating procedures and system state.
- **Next Focus:** Enable multi-GPU execution, close the four-symbol data gap, automate reward loop weighting, and build Phase 8 stress testing.

---

## 2. Solution Architecture Overview

```mermaid
flowchart TD
    subgraph Data["Market & Metadata Sources"]
        A[Breeze OHLCV \n (Equities, ETFs)]
        B[Breeze Sentiment API]
        C[Index Metadata \n (nifty100_constituents.json)]
    end

    subgraph Pipeline["Historical Training Orchestrator"]
        P1[Phase 1:\nData Ingestion]
        P2[Phase 2:\nTeacher Training]
        P3[Phase 3:\nStudent Training]
        P4[Phase 4:\nModel Validation]
        P5[Phase 5:\nStatistical Tests]
        P6[Phase 6:\nRelease Audit]
        P7[Phase 7:\nPromotion Briefing]
        P8[Telemetry:\nTraining Events]
    end

    subgraph Storage["Persistent Artifacts"]
        D1[data/historical/<symbol>/<interval>/]
        D2[data/historical/metadata/*.json]
        D3[data/models/<batch_id>/<symbol_window>/]
        D4[release/audit_live_candidate_<run_id>/]
        D5[data/analytics/training/training_run_*.jsonl]
        D6[data/state/*]
    end

    Data --> P1
    P1 -->|Chunked OHLCV| D1
    P1 -->|Mappings & coverage| D2
    D1 --> P2
    P2 -->|Teacher Artifacts| D3
    D3 --> P3
    P2 & P3 --> P4
    P4 -->|validation_run_id| P5
    P5 -->|stat_tests.json| P6
    P6 -->|Audit Bundle| P7
    P2 & P3 --> P8
    P8 -->|TrainingEvent logs| D5
    P7 -->|Promotion Briefing| D4
    P1 & P2 & P3 -->|State Snapshots| D6
```

---

## 3. Filesystem & Artifact Layout (2025-10-28)

| Path | Description |
|------|-------------|
| `data/historical/<SYMBOL>/<INTERVAL>/<DATE>.csv` | Canonical OHLCV data (deduplicated chunk ingestion) |
| `data/historical/metadata/nifty100_constituents.json` | Index metadata (96 verified, 4 data-unavailable tickers) |
| `data/historical/metadata/symbol_mappings*.json` | NSE → ISEC symbol mappings by batch |
| `data/historical/metadata/coverage_*.json[l]` | Coverage audit outputs per ingestion batch |
| `data/models/<BATCH_ID>/<SYMBOL_WINDOW>/` | Teacher & student artifacts (`model.pkl`, `labels.csv.gz`, `metadata.json`, `feature_importance.csv`) |
| `release/audit_live_candidate_<RUN_ID>/` | Validation bundle, statistical tests, promotion briefing |
| `data/analytics/training/training_run_*.jsonl` | Streaming `TrainingEvent` telemetry for dashboards |
| `data/state/` | StateManager checkpoints, session notes, progress snapshots |
| `docs/batch*-*.md` | Batch-specific ingestion and training reports |
| `claude.md` | Operational guide, session history, contacts |

---

## 4. Historical Training Pipeline

| Phase | Description | Key Scripts / Modules | Primary Outputs |
|-------|-------------|-----------------------|-----------------|
| **1. Data Ingestion** | Chunked Breeze ingestion with caching, retries, and coverage audits | `scripts/fetch_historical_data.py`, `scripts/check_symbol_coverage.py` | `data/historical/*`, coverage reports, ingestion logs |
| **2. Teacher Training** | LightGBM teacher models per rolling window (GPU) | `scripts/train_teacher_batch.py`, `src/services/teacher_student.py` | Teacher artifacts per window, telemetry events |
| **3. Student Training** | Student retraining on teacher labels, reward metrics | `scripts/train_student_batch.py`, `src/services/teacher_student.py` | Student model artifacts, reward summaries |
| **4. Model Validation** | Aggregate metrics, produce validation run id | `scripts/run_model_validation.py`, `src/services/monitoring.py` | `validation_<timestamp>/metrics.json` |
| **5. Statistical Tests** | Walk-forward CV, bootstrap, hypothesis tests | `scripts/run_statistical_tests.py` | `release/audit_validation_<run_id>/stat_tests.json` |
| **6. Release Audit** | Compile audit bundle with warnings tolerance | `scripts/release_audit.py`, `src/services/state_manager.py` | `release/audit_live_candidate_<run_id>/manifest.yaml` |
| **7. Promotion Briefing** | Generate promotion briefing and artifact validation | `scripts/promote_candidate.py` | `promotion_briefing.md`, manifest |
| **Telemetry** | Streaming progress, GPU load, outcomes | `src/services/training_telemetry.py` | `TrainingEvent` JSONL series under `data/analytics/training/` |

---

## 5. Teacher & Student Training – Batch 4 (2025-10-28)

- **Batch Directory:** `data/models/20251028_154400/` (`live_candidate_20251028_154400`)
- **Symbol Coverage:** 36/36 Batch 4 symbols (100%), seven windows per symbol.
- **Windows:** 252 total (216 success, 36 expected skips because forward labels exceeded available data), 0 failures.
- **Artifacts:** Every successful window stores `model.pkl`, `labels.csv.gz`, `metadata.json`, `feature_importance.csv`, plus student counterparts and reward logs.
- **Fixes Applied:** Timezone alignment for DRYRUN CSV timestamps and removal of invalid `symbol` parameter in `Bar` dataclass.
- **Telemetry:** Training runs emit structured events under `data/analytics/training/training_run_live_candidate_*.jsonl`.

_Action:_ Extend teacher training to Batch 5 symbols (30 remaining) after implementing GPU assignment for multi-worker runs.

---

## 6. Statistical Validation & Release Audit

- **Validation Run:** `validation_20251028_155300` completed successfully (Phase 4).
- **Statistical Tests:** Stored under `release/audit_validation_20251028_155300/` with real metrics and pass verdict.
- **Release Audit:** `release/audit_live_candidate_20251028_154400/` produced manifest, promotion briefing, and QA checklist with exit code 0.
- **Promotion Briefing:** Summarises per-symbol performance, reward metrics, and outstanding risks (none blocking).

---

## 7. Phase 7 Roadmap (Status: 2025-10-28)

| Initiative | Objective | Status | Notes / Next Actions |
|------------|-----------|--------|----------------------|
| **1. Broaden Training Data** | NIFTY100 + Metals coverage with 3-year history | ✅ Complete | 96 verified symbols ingested; four official tickers tracked as data-unavailable. |
| **2. Reward Loop** | Adaptive weighting from reward signals | 🟡 In Progress | Reward metrics captured; weighting & governance rules to be implemented. |
| **3. Black-Swan Stress Tests** | Scenario testing for tail events | 🔴 Not Started | Requires Phase 8 orchestrator and scenario datasets. |
| **4. Training Progress Monitoring** | Real-time dashboards & alerts | 🟡 In Progress | `TrainingEvent` telemetry available; dashboard integration pending. |
| **5. Multi-GPU Acceleration** | Utilise dual RTX A6000 for batch runs | 🟡 In Progress | Spike complete; update worker GPU assignment before Batch 5 training. |

---

## 8. Monitoring & Telemetry

- **Structured Logs:** `logs/` (loguru JSON) for ingestion, training, validation, and audits.
- **Training Telemetry:** `data/analytics/training/training_run_live_candidate_*.jsonl` with phase transitions, window status, GPU utilisation.
- **Backtest Telemetry:** `dashboards/telemetry_dashboard.py` (extend with training tab).
- **State Tracking:** `data/state/training_progress.json` (planned), `data/state/session_notes.json`, batch coverage summaries.
- **Command Logs:** `docs/logs/session_YYYYMMDD_commands.txt` (per session).

---

## 9. Known Limitations & Upcoming Work

1. **Data gaps** for ADANIGREEN, IDEA, APLAPOLLO, DIXON — pursue alternate sources or risk sign-off for exclusion.
2. **GPU assignment** currently hardcoded to device 0 — implement worker-level `gpu_device_id` selection to leverage both RTX A6000 GPUs.
3. **Reward loop automation** — convert recorded reward metrics into adaptive sampling and deployment governance.
4. **Stress testing** — add Phase 8 orchestrator with 2008/2013/2020 scenarios and automate reporting.
5. **Telemetry dashboards** — surface training telemetry, GPU load, and anomaly alerts in Streamlit / Ops tooling.
6. **Order book ingestion** — flip `ORDER_BOOK_ENABLED` and `ENABLE_ORDER_BOOK_FEATURES` only after replacing the stub provider with validated live depth snapshots.
7. **Structural support/resistance analytics** — design long-horizon support/resistance feature module and integrate into swing/teacher signals prior to release.

---

## Appendix A – CLI Cheat Sheet (2025-10-28)

| Purpose | Command |
|---------|---------|
| Full pipeline run | `conda run -n sensequant python scripts/run_historical_training.py --symbols $(paste -sd, data/historical/metadata/batch4_training_symbols.txt) --start-date 2022-01-01 --end-date 2024-12-31 --skip-fetch` |
| Teacher batch (multi-GPU ready) | `conda run -n sensequant python scripts/train_teacher_batch.py --symbols $(paste -sd, data/historical/metadata/batch4_training_symbols.txt) --start-date 2022-01-01 --end-date 2024-12-31 --workers 2` |
| Student batch | `conda run -n sensequant python scripts/train_student_batch.py --batch-id 20251028_154400` |
| Statistical tests | `conda run -n sensequant python scripts/run_statistical_tests.py --run-id validation_20251028_155300` |
| Release audit | `conda run -n sensequant python scripts/release_audit.py --candidate live_candidate_20251028_154400` |
| Coverage audit | `conda run -n sensequant python scripts/check_symbol_coverage.py --symbols-file data/historical/metadata/nifty100_batch4.txt` |

---

## Appendix B – Change Log (Recent)

| Milestone | Focus | Date |
|-----------|-------|------|
| 7.1 | Batch 4 ingestion (OBEROI mapping fix) | 2025-10-28 |
| 7.2 | Training telemetry instrumentation (`TrainingEvent` stream) | 2025-10-28 |
| 7.3 | Batch 4 teacher/student run (`live_candidate_20251028_154400`) | 2025-10-28 |
| 7.4 | Batch 5 ingestion (30 symbols, coverage 100%) | 2025-10-28 |
| 6x | Telemetry resilience | 2025-10-14 |
| 6w | Real metrics integration | 2025-10-14 |
| 6r | Release audit tolerance | 2025-10-14 |

**Contacts & Support:** See `claude.md#Contacts`. Production incidents follow Ops escalation procedures outlined there.
