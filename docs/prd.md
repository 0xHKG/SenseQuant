# SenseQuant — Product Requirements (v2)

**Last Reviewed:** 2025-10-28  
**Maintainer:** Product & Program (see `claude.md#Contacts`)  
**Scope:** NIFTY 100 + metals ETFs, teacher–student pipeline governed by BMAD.

## Release 2025-10 Snapshot

- Historical data ingestion complete for 96 tradeable NIFTY 100 constituents and metals ETFs (2022-01-01 to 2025-10-28). Four official tickers lack Breeze coverage and are tracked separately for remediation.
- Teacher–student pipeline (Phases 1–7) executed successfully (`live_candidate_20251028_154400`) with audited artifacts and promotion briefing.
- Training telemetry streaming is live; reward metrics captured for future adaptive weighting.
- Upcoming focus: multi-GPU training execution, data remediation, reward loop automation, and Phase 8 stress testing.

## Functional Requirements

| Area | Requirement | Status (2025-10-28) | Evidence / Notes |
|------|-------------|----------------------|------------------|
| **Trading Strategies** | Intraday strategy opens 09:15 IST and exits by 15:29 IST | ✅ Complete | `src/domain/strategies/intraday.py` ensures flat end-of-day positions. |
|  | Swing strategy supports 2–10 day holds with SL/TP | ✅ Complete | `src/domain/strategies/swing.py` manages stop-loss / take-profit. |
|  | Position sizing respects exposure limits | ✅ Complete | `src/services/risk_manager.py::can_open_position` enforces per-symbol caps. |
| **Teacher–Student Loop** | Teacher labels rolling historical windows | ✅ Complete | `scripts/train_teacher_batch.py`, Batch 4 run 216/252 windows success. |
|  | Student retrains on teacher labels | ✅ Complete | `scripts/train_student_batch.py` produces student artifacts for Batch 4 directory. |
|  | Walk-forward cadence (expanding windows) | ✅ Complete | Rolling window schedule (7 windows per symbol, 2022–2024). |
|  | Reward/punishment loop influences training | 🟡 In Progress | Reward metrics recorded; weighting logic pending. |
| **Sentiment Integration** | News/social sentiment ingestion with gating | ✅ Complete | `src/services/sentiment/` providers + gating in both strategies. |
| **Breeze API Integration** | OHLCV ingestion with retry & rate limit handling | ✅ Complete | `scripts/fetch_historical_data.py` (90-day chunks, 2.0s delay, cache). |
|  | Live market data subscription resilience | 🟡 In Progress | Reconnect logic exists; observability enhancements pending. |
|  | Order placement with retry on transient errors | ✅ Complete | `src/adapters/breeze_client.py::place_order` handles retry/logging. |
| **Risk Management** | Global exposure cap & circuit breaker | ✅ Complete | RiskManager tracks capital, daily loss, circuit breaker state. |
|  | Per-trade SL/TP enforcement | ✅ Complete | SL/TP attached via strategies + risk manager. |
| **Data Platform** | NIFTY100 metadata with mapping coverage | ✅ Complete | `data/historical/metadata/nifty100_constituents.json`, symbol mapping files. |
|  | Automated coverage audits | ✅ Complete | `scripts/check_symbol_coverage.py` JSONL summaries (20251028). |
|  | Data gap tracking & remediation | 🟡 In Progress | `data_unavailable` array tracks four missing symbols; remediation pending. |
| **Telemetry & Monitoring** | Training telemetry stream | ✅ Complete | `data/analytics/training/training_run_live_candidate_*.jsonl`. |
|  | Backtest/live dashboards | 🟡 In Progress | Streamlit dashboard exists; training tab/alerts outstanding. |
| **Operations & Governance** | Configuration via `.env` and secrets manager | ✅ Complete | `src/services/secrets_manager.py`, `claude.md` instructions. |
|  | Dry-run & backtest execution modes | ✅ Complete | `src/app/engine.py` modes, CLI `--mode` handling. |
|  | BMAD documentation & promotion artifacts | ✅ Complete | Commandments, PRD, architecture refreshed; promotion briefing produced per run. |

## Non-Functional Requirements

| Category | Requirement | Status | Notes |
|----------|-------------|--------|-------|
| **Performance** | Historical run completes within targeted window (≤20 min for Batch 4) | ✅ Achieved | Batch 4 teacher/student run completed in 18 minutes on single GPU. |
|  | Multi-GPU utilisation | 🟡 Planned | GPU assignment spike complete; implementation pending. |
| **Reliability** | Graceful shutdown & restart-safe state | ✅ Complete | `src/app/engine.py` signal handling; StateManager checkpoints. |
|  | 24h soak without unhandled exceptions | 🟡 Monitor | Requires execution of extended dry-run/live soak post Batch 5. |
| **Maintainability** | Structured logging & traceability | ✅ Complete | Loguru JSON logs + telemetry provide auditable trail. |
|  | ≥70% automated test coverage on core adapters | 🟡 Monitor | Last measurement 87% strategy coverage (2025-10-11); rerun planned. |
| **Security & Compliance** | Secrets isolated from code, audit trail maintained | ✅ Complete | `.env` guidelines, promotion bundle manifests, coverage reports. |
| **Documentation** | Living documentation kept current | ✅ Updated | Commandments, PRD, overview, architecture refreshed 2025-10-28. |

## Open Scope & Next Actions

1. **Data Remediation:** Source alternate OHLCV data (or secure formal exclusion) for ADANIGREEN, IDEA, APLAPOLLO, DIXON; document decision in risk log.
2. **Multi-GPU Training:** Parameterise `gpu_device_id` and schedule workers across both RTX A6000 GPUs before Batch 5 teacher training run.
3. **Reward Loop Automation:** Convert reward metrics into sample weighting and governance checks; define QA sign-off criteria (Sharpe, win-rate, drawdown thresholds).
4. **Stress Testing:** Build Phase 8 scenario orchestrator (2008, 2013, 2020 etc.) and integrate results into promotion briefing.
5. **Telemetry & Alerting:** Extend dashboards/alerts for ingestion and training anomalies leveraging `TrainingEvent` stream.
6. **Reliability Validation:** Execute multi-day dry-run/soak tests once Batch 5 training is complete and log outcomes in `docs/logs/`.
7. **Order Book Enablement:** Replace stub provider with validated Breeze (or alternate) depth feed and enable order book ingestion/feature toggles prior to release.
8. **Long-Term Support/Resistance Analytics:** Implement and validate structural support/resistance features feeding strategies and teacher labeling.
