# US-011: Parameter Optimization Engine

**Status**: 🚧 In Progress
**Priority**: High
**Estimated Effort**: Large

---

## Problem Statement

Trading strategies have numerous parameters (SMA periods, stop-loss %, risk per trade, etc.) that significantly impact performance. Manual parameter selection is:
1. Time-consuming and subjective
2. Prone to overfitting on recent data
3. Difficult to validate systematically
4. Not reproducible across team members

Without systematic optimization, strategies may underperform or carry excessive risk due to suboptimal parameters.

---

## Objectives

Build a parameter optimization engine that:
- Supports **grid search** (exhaustive) and **random search** (sampling) over strategy/risk parameter spaces
- Invokes the Backtester deterministically with controlled random seeds
- Evaluates candidates using configurable objective metrics (Sharpe ratio, CAGR, risk-adjusted return)
- Persists full optimization results with reproducibility metadata (git hash, timestamp, search space)
- Provides CLI interface for loading search configurations and executing optimizations
- Handles failures gracefully (log errors, continue with remaining candidates)

---

## Requirements

### Functional

1. **Parameter Optimizer** (`src/services/optimizer.py`)
   - `ParameterOptimizer` class with configurable search strategy (grid/random)
   - Generate parameter combinations from search space definition
   - Execute backtest for each candidate with deterministic seeding
   - Rank results by objective function (e.g., Sharpe ratio, CAGR, custom score)
   - Track failed candidates with error messages
   - Log progress with `component="optimizer"`

2. **Search Space Configuration**
   - Support nested parameter spaces (strategy params, risk params, data params)
   - Grid search: Cartesian product of all parameter values
   - Random search: Sample N random combinations from continuous/discrete distributions
   - Example search space:
     ```yaml
     strategy:
       swing:
         sma_fast: [5, 10, 15, 20]
         sma_slow: [30, 40, 50, 60]
         stop_loss_pct: [1.0, 1.5, 2.0]
         take_profit_pct: [2.0, 3.0, 4.0]
     risk:
       risk_per_trade_pct: [0.5, 1.0, 1.5, 2.0]
       max_positions: [3, 5, 8]
     ```

3. **Optimization Results**
   - Persist to `data/optimization/<timestamp>/`
   - `summary.json`: Metadata (symbols, date range, strategy, search type, total candidates, git hash)
   - `ranked_results.csv`: All candidates sorted by score (rank, params, metrics, runtime)
   - `best_config.json`: Top-ranked configuration ready for deployment
   - Track execution time per candidate and total optimization time

4. **CLI Interface** (`scripts/optimize.py`)
   - Load search space from YAML/JSON file
   - Execute optimization with progress reporting
   - Print top-N results with formatted metrics
   - Validate inputs (date ranges, symbol lists, search space structure)
   - Exit with non-zero status on errors

5. **Domain Types** (`src/domain/types.py`)
   - `OptimizationConfig`: Search space, strategy, symbols, dates, objective metric
   - `OptimizationCandidate`: Parameter combination, backtest result, score, error (if failed)
   - `OptimizationResult`: Summary metadata, ranked candidates, best config, execution stats

### Non-Functional

- **Determinism**: Same search space + seed → same candidate order and results
- **Fault Tolerance**: Individual candidate failures don't abort optimization
- **Performance**: Parallel execution (future enhancement, single-threaded for v1)
- **Reproducibility**: Git hash, timestamp, full config captured
- **Logging**: Structured logs with candidate ID, params, score, elapsed time

---

## Architecture Design

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    scripts/optimize.py                       │
│  (CLI: Load YAML config, invoke optimizer, print results)   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│              src/services/optimizer.py                       │
│  ParameterOptimizer:                                         │
│   - generate_candidates(search_space, search_type, N)        │
│   - evaluate_candidate(params, backtest_config)              │
│   - rank_candidates(results, objective)                      │
│   - save_results(output_dir)                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│              src/services/backtester.py                      │
│  Backtester.run() → BacktestResult                           │
│  (Deterministic simulation with metrics)                     │
└─────────────────────────────────────────────────────────────┘
```

### Workflow

1. **Load Configuration**: Parse YAML/JSON search space + optimization settings
2. **Generate Candidates**: Create parameter combinations (grid or random sampling)
3. **Evaluate Candidates**: For each combination:
   - Create BacktestConfig with candidate parameters
   - Run Backtester with fixed random seed
   - Extract objective metric (e.g., Sharpe ratio)
   - Catch and log errors without aborting
4. **Rank Results**: Sort candidates by objective score (descending)
5. **Persist Results**: Save summary, ranked list, best config to `data/optimization/`
6. **Report**: Print top-N candidates with metrics

---

## Implementation Plan

### Tasks

1. **Update domain types** (`src/domain/types.py`)
   - Add `OptimizationConfig` dataclass
   - Add `OptimizationCandidate` dataclass
   - Add `OptimizationResult` dataclass

2. **Implement ParameterOptimizer** (`src/services/optimizer.py`)
   - `__init__(config: OptimizationConfig, client, settings)`
   - `generate_candidates() -> list[dict]` (grid/random)
   - `evaluate_candidate(params: dict) -> OptimizationCandidate`
   - `run() -> OptimizationResult`
   - `_save_results(result: OptimizationResult, output_dir: Path)`
   - Use `component="optimizer"` for all logging

3. **Create CLI** (`scripts/optimize.py`)
   - Argparse: `--config`, `--symbols`, `--start-date`, `--end-date`, `--top-n`
   - Load YAML/JSON search space
   - Invoke ParameterOptimizer
   - Print formatted results table
   - Handle errors with sys.exit(1)

4. **Write unit tests** (`tests/unit/test_optimizer.py`)
   - Test grid search candidate generation
   - Test random search sampling
   - Test candidate ranking by different objectives
   - Test error handling for failed candidates
   - Test result persistence

5. **Write integration tests** (`tests/integration/test_optimization_pipeline.py`)
   - Test full optimization workflow (grid search on small space)
   - Test random search with sampling
   - Test fault tolerance (inject backtest failures)
   - Test result artifacts completeness
   - Test determinism (same seed → same results)

6. **Run quality gates**
   - `ruff check .`
   - `ruff format --check .`
   - `mypy src/`
   - `pytest -q`

---

## Acceptance Criteria

### Must Have
- [ ] Grid search generates all combinations correctly (Cartesian product)
- [ ] Random search samples N candidates with seed determinism
- [ ] Backtester invoked with correct parameters per candidate
- [ ] Failed candidates logged without aborting optimization
- [ ] Results ranked by objective metric (Sharpe, CAGR, custom)
- [ ] Artifacts persisted: `summary.json`, `ranked_results.csv`, `best_config.json`
- [ ] Git hash captured in metadata
- [ ] CLI validates inputs and exits non-zero on errors
- [ ] All quality gates pass (ruff, mypy, pytest)

### Should Have
- [ ] Progress logging with candidate ID, params, score, elapsed time
- [ ] YAML/JSON config file loading for search spaces
- [ ] Top-N results printed in formatted table
- [ ] Execution time tracked per candidate and total
- [ ] Documentation in docstrings with examples

### Nice to Have
- [ ] Parallel execution (multi-processing pool)
- [ ] Resume from checkpoint (partial results)
- [ ] Bayesian optimization (future US)
- [ ] Cross-validation (walk-forward splits)

---

## Test Strategy

### Unit Tests
- Parameter space parsing and validation
- Grid search candidate generation (small spaces)
- Random search sampling with seed reproducibility
- Objective function calculations
- Error handling for invalid configs
- Result serialization/deserialization

### Integration Tests
- Full optimization on 2x2 grid (4 candidates)
- Random search with 10 samples
- Multi-symbol optimization
- Failed candidate recovery
- Deterministic results verification

### Edge Cases
- Empty search space
- Single-parameter optimization
- All candidates fail
- Invalid objective metric

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large search spaces → long runtimes | High | Start with small grids, add parallel execution later |
| Overfitting to optimization period | High | Use walk-forward validation (future US) |
| Parameter interactions not captured | Medium | Grid search explores full space systematically |
| Failed backtests abort optimization | Medium | Try-catch each candidate, log errors, continue |

---

## Success Metrics

- Optimization completes successfully on 4x4 grid (16 candidates) in < 5 minutes
- Failed candidates logged without process exit
- Best config JSON deployable directly to engine
- 100% test coverage on optimizer module
- All quality gates passing

---

## Future Enhancements (Post-US-011)

- **US-012**: Walk-forward optimization with out-of-sample validation
- **US-013**: Bayesian optimization for efficient search
- **US-014**: Multi-objective optimization (Pareto frontiers)
- **US-015**: Parallel execution with multiprocessing
- **US-016**: Hyperparameter importance analysis (sensitivity)

---

## References

- Backtester: `src/services/backtester.py`
- Domain types: `src/domain/types.py`
- Strategy params: `src/domain/strategies/{intraday,swing}.py`
- Risk params: `src/services/risk_manager.py`
