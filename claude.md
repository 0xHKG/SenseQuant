# Claude Code Onboarding Guide

**Role**: Claude Code agent in the SenseQuant BMAD workflow
**Mission**: Implement production-grade algorithmic trading features with strict quality gates and comprehensive testing

---

## 1. Overview

**SenseQuant** is a multi-strategy algorithmic trading system for Indian equities (NSE) via ICICI Breeze API.

**Core Architecture**:
- **Engine**: `src/services/engine.py` - Main trading loop (intraday + swing)
- **Strategies**: `src/domain/strategies/{intraday,swing}.py` - Signal generation with SMA/EMA crossovers
- **Risk Management**: `src/services/risk_manager.py` - Position sizing, circuit breakers, fees/slippage
- **ML Pipeline**: Teacher-Student learning for swing predictions
- **Backtesting**: Historical strategy validation with performance metrics

**Key Documentation**:
- Product Requirements: `docs/prd.md`
- System Architecture: `docs/architecture.md`
- User Stories: `docs/stories/` (US-000 through US-010)

---

## 2. Roles & Workflow

**BMAD Workflow** (Planner → Claude → QA loop):
1. **Planner** (Business Analyst + Mad Architect) defines story, acceptance criteria
2. **Claude Code** implements feature with tests
3. **QA** validates quality gates, integration testing

**Configuration**: `bmad.project.yaml` defines active workflows and agent roles

**Active Process**:
- Implement story per spec in `docs/stories/us-*.md`
- Run all quality gates before marking complete
- Update story document with implementation notes
- Report PASS/FAIL status for each gate

---

## 3. Quality Gates

**Required Commands** (all must PASS before story completion):

```bash
# Linting
python -m ruff check .

# Formatting
python -m ruff format --check .

# Type checking
python -m mypy src/

# Tests
python -m pytest -q
```

**Expectations**:
- **ruff check**: 0 errors in `src/`, `tests/`, `scripts/` (external files ignored)
- **ruff format**: All files formatted correctly
- **mypy**: No type errors in `src/` and `scripts/`
- **pytest**: 100% pass rate on implemented features (skipped tests allowed for future work)

**Current Status** (as of US-010):
- ruff check: PASS (0 project errors)
- ruff format: PASS (all files formatted)
- mypy: PASS (no type errors)
- pytest: PASS (210/211 passing, 1 skipped)

---

## 4. Current Capabilities

**Completed Stories**: US-000 through US-010

**Key Modules**:

| Module | Path | Purpose |
|--------|------|---------|
| Engine | `src/services/engine.py` | Main trading orchestrator (intraday + swing) |
| Breeze Client | `src/adapters/breeze_client.py` | ICICI Breeze API wrapper with retry/rate limiting |
| Risk Manager | `src/services/risk_manager.py` | Position sizing, circuit breakers, capital tracking |
| Sentiment Cache | `src/services/sentiment_cache.py` | Rate-limited sentiment provider with caching |
| Feature Library | `src/domain/features.py` | Technical indicators (SMA, EMA, RSI, MACD, ATR, etc.) |
| Teacher Labeler | `src/services/teacher_student.py` | Generate labeled training data from historical bars |
| Student Model | `src/services/teacher_student.py` | Lightweight inference for swing predictions |
| Backtester | `src/services/backtester.py` | Historical strategy validation with metrics |
| Intraday Strategy | `src/domain/strategies/intraday.py` | Momentum-based intraday entries/exits |
| Swing Strategy | `src/domain/strategies/swing.py` | SMA crossover with TP/SL/max-hold logic |

**Test Coverage**:
- Unit tests: `tests/unit/` (140+ tests)
- Integration tests: `tests/integration/` (70+ tests)
- All critical paths covered with edge case testing

---

## 5. Key Commands & Scripts

**Make Targets**:
```bash
make test          # Run pytest with coverage
make lint          # Run ruff check
make format        # Auto-format with ruff
make typecheck     # Run mypy
make test-cov      # Generate HTML coverage report
```

**Training Teacher Model**:
```bash
# Train Teacher for swing prediction
python scripts/train_teacher.py \
  --symbols RELIANCE TCS INFY \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir data/models/teacher_v1

# Output: model.pkl, labels.csv, metadata.json in output-dir
```

**Running Backtests**:
```bash
# Backtest swing strategy
python scripts/backtest.py \
  --symbols RELIANCE TCS \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --strategy swing \
  --initial-capital 1000000

# Outputs: summary.json, equity.csv, trades.csv in data/backtests/

# Multi-strategy backtest
python scripts/backtest.py \
  --symbols RELIANCE \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --strategy both \
  --verbose
```

**Running Live Engine**:
```bash
# Dryrun mode (no real orders, no network calls except auth)
MODE=dryrun python -m src.services.engine

# Live mode (requires valid Breeze credentials)
MODE=live python -m src.services.engine
```

**Notes**:
- **Dryrun mode**: Generates deterministic order IDs, simulates fills, no Breeze SDK calls
- **Backtest mode**: Historical simulation with Breeze API data, no real trading
- **Live mode**: Real orders placed via Breeze API (use with caution)

---

## 6. Coding Guidelines

**Type Safety**:
- All functions must have type hints (params + return types)
- Use `from __future__ import annotations` for forward references
- Run `mypy src/` before committing - must pass with 0 errors

**Docstrings**:
- All public functions/classes require docstrings
- Format: Google-style with Args, Returns, Raises sections
- Include usage examples for complex functions

**Logging**:
- Use `loguru` logger with structured `extra={"component": "..."}` tags
- Component tags: `"engine"`, `"risk"`, `"sentiment"`, `"teacher"`, `"student"`, `"backtest"`, etc.
- Example:
  ```python
  logger.info("Signal generated", extra={"component": "swing", "symbol": symbol, "action": action})
  ```

**Dryrun Network Restrictions**:
- **NEVER** make real network calls in dryrun mode (except authentication)
- BreezeClient automatically stubs API calls when `dry_run=True`
- Sentiment provider returns neutral score in dryrun
- Tests should mock all external dependencies

**Artifact Locations**:
- **Teacher models**: `data/models/teacher_*/` (model.pkl, labels.csv, metadata.json)
- **Student models**: `data/models/student_*/` (model.pkl, metadata.json)
- **Backtests**: `data/backtests/` (summary JSON, equity CSV, trades CSV)
- **Trade journals**: `data/journals/` (daily trade logs with timestamps)
- **Logs**: `logs/` (component-specific log files with rotation)

**Data Schemas**:
- Use `@dataclass` for domain types in `src/domain/types.py`
- Bar schema: `ts`, `open`, `high`, `low`, `close`, `volume` (all required)
- Timestamps must be timezone-aware (`Asia/Kolkata`)
- Prices in float, quantities in int

**Testing Requirements**:
- Every new feature requires unit tests
- Integration tests for multi-component workflows
- Use `pytest.mark.skip` for unimplemented features with TODO comment
- Mock external dependencies (Breeze API, sentiment providers)
- Fixtures in `conftest.py` for reusable test data

**Error Handling**:
- Raise `ValueError` for invalid inputs with clear messages
- Raise `RuntimeError` for operational failures
- Log exceptions with `logger.exception()` before re-raising
- Never silence errors - propagate or handle explicitly

---

## 7. Roadmap / Pending Work

**Completed** (US-000 through US-010):
- Core engine with intraday + swing strategies
- Risk management with position sizing and circuit breakers
- Sentiment integration with caching
- Feature library (10+ technical indicators)
- Teacher-Student ML pipeline
- Backtest engine with comprehensive metrics

**Upcoming Focus** (US-011+):
- **Real-time monitoring dashboard** - Web UI for live positions, PnL, metrics
- **Advanced order types** - Bracket orders, trailing stops, iceberg orders
- **Multi-timeframe analysis** - Combine 1min, 5min, daily signals
- **Walk-forward optimization** - Automated parameter tuning with cross-validation
- **Alert system** - Telegram/Email notifications for signals, fills, risk events
- **Portfolio optimization** - Multi-asset allocation with correlation analysis

**Outstanding Enhancements**:
- **CSV data source**: Backtest with custom historical data (currently only Breeze API)
- **Teacher labels data source**: Backtest using pre-labeled Teacher data
- **Real sentiment providers**: Integrate NewsAPI, Twitter, FinBERT (currently mock/cache only)
- **Intraday backtest**: Full minute-level simulation (currently simplified/stub)
- **Performance optimization**: Vectorized feature calculations, parallel backtests
- **Monitoring & alerting**: Prometheus metrics, Grafana dashboards

**Known Limitations**:
- Intraday strategy uses simplified logic (needs minute-level OHLCV)
- Sentiment analysis is mocked (requires real API integrations)
- Backtester only supports swing strategy fully (intraday stub)
- No position rebalancing or portfolio-level risk management yet

**Next Steps**:
1. Review active story in `docs/stories/` for current sprint
2. Implement feature per acceptance criteria
3. Write comprehensive tests (unit + integration)
4. Run all quality gates and report status
5. Update story document with implementation notes
6. Commit with descriptive message referencing story ID

---

**Questions?** Refer to:
- PRD: `docs/prd.md` - Product requirements and business context
- Architecture: `docs/architecture.md` - System design and component interactions
- Stories: `docs/stories/` - Detailed feature specifications with acceptance criteria
- Code: Start with `src/services/engine.py` and follow imports

**Quality Mantra**: No code ships without passing all quality gates. Period.
