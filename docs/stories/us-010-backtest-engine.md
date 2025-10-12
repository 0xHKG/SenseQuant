# US-010: Backtest Engine v1

**Status**: 🚧 In Progress
**Priority**: High
**Estimated Effort**: Large

---

## Problem Statement

The trading system needs a robust backtesting framework to:
1. Validate strategy performance on historical data before live deployment
2. Compare intraday vs swing strategies systematically
3. Evaluate Teacher-Student model predictions against actual outcomes
4. Generate reproducible performance metrics (returns, drawdown, Sharpe, win rate)
5. Support iterative strategy development with fast feedback loops

Without backtesting, strategies are deployed blind, risking capital on untested logic.

---

## Objectives

Build a deterministic backtesting engine that:
- Supports both **intraday** and **swing** strategies with identical logic to live engine
- Consumes historical OHLCV data from flexible sources (CSV, Breeze API, Teacher artifacts)
- Calculates comprehensive performance metrics (CAGR, max drawdown, Sharpe ratio, win rate, exposure)
- Optionally integrates Teacher labels and Student predictions for validation
- Persists results to `data/backtests/` with full reproducibility metadata
- Runs in pure dry-run mode (no network calls during simulation)

---

## Requirements

### Functional

1. **Backtest Runner** (`src/services/backtester.py`)
   - Configure via dataclass: symbol list, date range, strategy type, data source
   - Execute time-ordered simulation respecting market hours (9:15-15:29 IST intraday, EOD swing)
   - Track positions, equity curve, and trade log with identical fee/slippage calculations as live engine

2. **Performance Metrics**
   - Total Return %
   - CAGR (Compound Annual Growth Rate)
   - Max Drawdown %
   - Sharpe Ratio (risk-adjusted return)
   - Win Rate % (profitable trades / total trades)
   - Average Win/Loss
   - Exposure % (time in market)
   - Total Fees

3. **Data Sources**
   - Primary: Historical bars from Breeze API or CSV
   - Optional: Teacher labels CSV (for label-based simulation)
   - Optional: Student predictions (for model validation)

4. **Artifact Persistence**
   - JSON summary: `backtest_YYYYMMDD_HHMMSS_summary.json` (config, metrics, metadata)
   - CSV equity curve: `backtest_YYYYMMDD_HHMMSS_equity.csv` (timestamp, equity, positions)
   - CSV trades log: `backtest_YYYYMMDD_HHMMSS_trades.csv` (all entry/exit records)

5. **Determinism**
   - Fixed random seeds for reproducibility
   - Timestamp-based execution order
   - Immutable configuration capture

### Non-Functional

- **Performance**: Run 1-year backtest in <30 seconds
- **Accuracy**: Match live engine logic exactly (same signal generation, fees, slippage)
- **Extensibility**: Support pluggable data sources and custom metrics
- **Logging**: Use loguru with `component="backtest"` for consistency

---

## Design

### Architecture

```
BacktestConfig
    ├─ symbols: list[str]
    ├─ start_date: str
    ├─ end_date: str
    ├─ strategy: "intraday" | "swing" | "both"
    ├─ initial_capital: float
    ├─ data_source: "breeze" | "csv" | "teacher"
    └─ random_seed: int

Backtester
    ├─ __init__(config, client, settings)
    ├─ run() -> BacktestResult
    ├─ _simulate_intraday(symbol, bars)
    ├─ _simulate_swing(symbol, bars)
    ├─ _calculate_metrics() -> dict
    └─ _save_artifacts()

BacktestResult
    ├─ config: BacktestConfig
    ├─ metrics: dict[str, float]
    ├─ equity_curve: pd.DataFrame
    ├─ trades: pd.DataFrame
    └─ metadata: dict[str, Any]
```

### Metrics Calculation

```python
# Total Return
total_return = (final_equity - initial_capital) / initial_capital

# CAGR
years = (end_date - start_date).days / 365.25
cagr = (final_equity / initial_capital) ** (1 / years) - 1

# Max Drawdown
running_max = equity_curve["equity"].cummax()
drawdown = (equity_curve["equity"] - running_max) / running_max
max_drawdown = drawdown.min()

# Sharpe Ratio (annualized)
returns = equity_curve["equity"].pct_change().dropna()
sharpe = (returns.mean() / returns.std()) * sqrt(252) if returns.std() > 0 else 0

# Win Rate
win_rate = len(trades[trades["pnl"] > 0]) / len(trades)

# Exposure
exposure = days_with_positions / total_trading_days
```

---

## Implementation Plan

### Phase 1: Core Engine (This Story)

1. **Create `BacktestConfig` and `BacktestResult` types** ([src/domain/types.py](../../src/domain/types.py))
   - BacktestConfig with all configuration fields
   - BacktestResult with metrics, curves, and trades

2. **Implement `Backtester` class** ([src/services/backtester.py](../../src/services/backtester.py))
   - Initialization with config and dependencies
   - `run()` orchestrating full backtest
   - `_simulate_intraday()` for intraday strategy simulation
   - `_simulate_swing()` for swing strategy simulation
   - `_calculate_metrics()` computing all performance metrics
   - `_save_artifacts()` persisting results to `data/backtests/`

3. **Add CLI entry point** ([scripts/backtest.py](../../scripts/backtest.py))
   - Argparse for symbol list, dates, strategy, data source
   - Integration with existing Settings and BreezeClient
   - Progress logging and result summary

4. **Unit Tests** ([tests/unit/test_backtester.py](../../tests/unit/test_backtester.py))
   - Metric calculations (CAGR, drawdown, Sharpe, win rate)
   - Data slicing and validation
   - Result serialization

5. **Integration Tests** ([tests/integration/test_backtest_pipeline.py](../../tests/integration/test_backtest_pipeline.py))
   - End-to-end backtest with mock data
   - Both intraday and swing strategies
   - Artifact validation

### Phase 2: Advanced Features (Future)

- Multi-symbol portfolio backtesting
- Walk-forward optimization
- Monte Carlo simulation
- Strategy parameter grid search
- Benchmark comparison (buy-and-hold)
- Interactive visualization dashboard

---

## Acceptance Criteria

- [ ] `Backtester` class implements all required methods
- [ ] Metrics calculations produce expected values on sample data
- [ ] CLI runs backtest from command line with args
- [ ] Artifacts saved to `data/backtests/` with complete metadata
- [ ] Unit tests cover metric calculations and edge cases
- [ ] Integration test runs sample backtest (5+ trades, metrics validated)
- [ ] All quality gates pass: ruff, mypy, pytest
- [ ] README documents backtest usage with examples

---

## Test Strategy

### Unit Tests (test_backtester.py)

1. **test_calculate_cagr** - CAGR calculation accuracy
2. **test_calculate_max_drawdown** - Drawdown computation
3. **test_calculate_sharpe_ratio** - Sharpe with various return distributions
4. **test_calculate_win_rate** - Win rate with mixed trades
5. **test_equity_curve_generation** - Equity curve construction
6. **test_trades_log_format** - Trade log schema validation
7. **test_metrics_edge_cases** - Zero trades, single trade, all losses
8. **test_config_validation** - Invalid config detection
9. **test_date_range_handling** - Edge cases (weekends, holidays)
10. **test_artifact_serialization** - JSON/CSV output format

### Integration Tests (test_backtest_pipeline.py)

1. **test_intraday_backtest_pipeline** - Full intraday backtest
2. **test_swing_backtest_pipeline** - Full swing backtest
3. **test_both_strategies** - Combined intraday + swing
4. **test_artifact_completeness** - All files created with correct schema

---

## Dependencies

**Internal:**
- [src/services/engine.py](../../src/services/engine.py) - Reuse strategy logic
- [src/domain/strategies/intraday.py](../../src/domain/strategies/intraday.py) - Intraday signals
- [src/domain/strategies/swing.py](../../src/domain/strategies/swing.py) - Swing signals
- [src/domain/features.py](../../src/domain/features.py) - Feature calculations
- [src/services/risk_manager.py](../../src/services/risk_manager.py) - Fee/slippage calculations

**External:**
- pandas - Data manipulation and metrics
- numpy - Numerical computations
- loguru - Logging

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Logic divergence from live engine | High | Reuse exact strategy code, comprehensive tests |
| Look-ahead bias in simulation | High | Strict time-ordering, point-in-time data access |
| Slow performance on large datasets | Medium | Vectorized operations, progress bars |
| Overfitting to historical data | Medium | Walk-forward validation, out-of-sample testing |

---

## Future Enhancements (Out of Scope)

- **Parameter Optimization**: Grid search for strategy parameters
- **Walk-Forward Analysis**: Rolling window backtests
- **Transaction Cost Analysis**: Detailed fee breakdown
- **Slippage Modeling**: Dynamic slippage based on volume
- **Benchmark Comparison**: Compare to buy-and-hold, index
- **Interactive Dashboard**: Plotly/Dash visualization
- **Multi-Strategy Portfolio**: Combined strategy allocation
- **Monte Carlo Simulation**: Confidence intervals for metrics

---

## Story Completion Checklist

- [ ] Story document created
- [ ] Types added to domain/types.py
- [ ] Backtester class implemented
- [ ] CLI entry point created
- [ ] Unit tests written (10+ tests)
- [ ] Integration tests written (4+ tests)
- [ ] Quality gates pass
- [ ] Documentation updated
- [ ] Code review complete

---

## References

- [Quantitative Trading](https://www.quantstart.com/) - Backtesting best practices
- [Backtrader](https://www.backtrader.com/) - Reference backtesting framework
- [Zipline](https://github.com/quantopian/zipline) - Algorithmic trading library
