# US-026: Advanced Statistical Validation & Risk Benchmarks

## Problem Statement

Model validation runs (US-025) currently report accuracy metrics but lack statistical rigor to determine if improvements are significant or due to random chance. We need:

1. **Statistical Significance Testing**: Walk-forward cross-validation, bootstrap confidence intervals, and hypothesis tests to validate that model improvements are statistically significant
2. **Risk Benchmarks**: Comparison against market benchmarks (NIFTY 50) to assess relative performance and risk-adjusted returns
3. **Comprehensive Risk Metrics**: Sharpe/Sortino ratios, drawdown analysis, and z-scores vs benchmark
4. **Reproducible Analysis**: Persistent statistical test results for audit trail

This ensures promotion decisions are based on statistically robust evidence rather than point estimates.

## Acceptance Criteria

### AC-1: Walk-Forward Cross-Validation
- [ ] Implement rolling window validation with configurable window size
- [ ] Test teacher/student models on out-of-sample data
- [ ] Compute per-fold accuracy metrics (precision, recall, F1)
- [ ] Calculate mean and standard deviation across folds
- [ ] Store fold-level results in `stat_tests.json`

### AC-2: Bootstrap Significance Testing
- [ ] Implement bootstrap resampling (n=1000) for accuracy metrics
- [ ] Compute 95% confidence intervals for precision, recall, accuracy
- [ ] Run paired t-tests for teacher vs student comparisons
- [ ] Calculate p-values for metric improvements vs baseline
- [ ] Include null hypothesis rejection decisions (p < 0.05)

### AC-3: Sharpe/Sortino Comparisons
- [ ] Calculate Sharpe ratio for strategy returns
- [ ] Calculate Sortino ratio (downside deviation only)
- [ ] Compare against baseline configuration
- [ ] Compute delta and percentage improvement
- [ ] Test statistical significance of Sharpe improvement

### AC-4: Benchmark Integration
- [ ] Pull benchmark data (NIFTY 50) via DataFeed during validation
- [ ] Align benchmark returns with strategy date range
- [ ] Calculate alpha (excess returns vs benchmark)
- [ ] Calculate beta (correlation with benchmark)
- [ ] Compute information ratio and tracking error
- [ ] Generate z-scores for relative performance

### AC-5: Statistical Tests Script
- [ ] Create `scripts/run_statistical_tests.py`
- [ ] Load validation results from `validation_summary.json`
- [ ] Replay teacher/student metrics for statistical analysis
- [ ] Support multiple test types (bootstrap, t-test, walk-forward)
- [ ] Output `stat_tests.json` with all results
- [ ] Integrate into validation workflow as optional step

### AC-6: Enhanced Validation Summary
- [ ] Add statistical significance section to JSON summary
- [ ] Include p-values, confidence intervals, and z-scores
- [ ] Highlight statistically significant improvements
- [ ] Add benchmark comparison section
- [ ] Generate Markdown summary with interpretation guidance
- [ ] Update promotion recommendation logic to consider significance

### AC-7: StateManager Extensions
- [ ] Add `record_statistical_validation()` method
- [ ] Track last benchmark comparison timestamp
- [ ] Store statistical test status (passed/failed/skipped)
- [ ] Enable querying of statistical validation history

### AC-8: Integration Testing
- [ ] Create `test_statistical_validation.py`
- [ ] Mock validation outputs with sample data
- [ ] Verify walk-forward cross-validation executes
- [ ] Confirm bootstrap confidence intervals computed
- [ ] Validate benchmark comparisons run correctly
- [ ] Check stat_tests.json generated with correct schema
- [ ] Ensure validation summary updated with significance

### AC-9: Documentation
- [ ] Create US-026 story document (this file)
- [ ] Document statistical methods and interpretation
- [ ] Add architecture section explaining workflow
- [ ] Provide guidance on reading p-values and confidence intervals
- [ ] Document benchmark comparison methodology
- [ ] Include examples of statistical validation output

## Technical Design

### Statistical Validation Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Model Validation Run (US-025)                                │
│    ├─> Teacher/student batch training                           │
│    ├─> Optimizer evaluation                                     │
│    └─> Generate validation_summary.json                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Statistical Testing (US-026)                                 │
│    ├─> Load validation results                                  │
│    ├─> Walk-forward cross-validation                            │
│    ├─> Bootstrap significance tests                             │
│    ├─> Sharpe/Sortino comparisons                               │
│    └─> Benchmark integration (NIFTY 50)                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Results Aggregation                                          │
│    ├─> Generate stat_tests.json                                 │
│    ├─> Update validation_summary.json                           │
│    ├─> Update validation_summary.md                             │
│    └─> Record in StateManager                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Walk-Forward Cross-Validation

**Method**: Rolling window with expanding or sliding window

```python
# Example: 12-month window, 3-month step
windows = [
    ("2024-01-01", "2024-12-31"),  # Train
    ("2024-04-01", "2025-03-31"),  # Train
    ("2024-07-01", "2025-06-30"),  # Train
]

for train_start, train_end in windows:
    # Train on window
    # Test on next 3 months (out-of-sample)
    # Record metrics
```

**Output**: Per-fold metrics with mean/std across folds

### Bootstrap Significance Testing

**Method**: Stratified bootstrap resampling (n=1000 iterations)

```python
def bootstrap_confidence_interval(metric_values, n_iterations=1000, confidence=0.95):
    """
    Compute bootstrap confidence interval for metric.

    Args:
        metric_values: Array of metric values (e.g., accuracies)
        n_iterations: Number of bootstrap samples
        confidence: Confidence level (default 0.95)

    Returns:
        (lower_bound, upper_bound, mean, std)
    """
    bootstrap_means = []
    n = len(metric_values)

    for _ in range(n_iterations):
        # Resample with replacement
        sample = np.random.choice(metric_values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    # Calculate percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return (lower, upper, np.mean(bootstrap_means), np.std(bootstrap_means))
```

**Output**: Confidence intervals for precision, recall, accuracy, F1

### Sharpe/Sortino Comparison

**Sharpe Ratio**: Risk-adjusted return using total volatility
```
Sharpe = (Return - Risk_Free_Rate) / Volatility
```

**Sortino Ratio**: Risk-adjusted return using downside volatility only
```
Sortino = (Return - Target_Return) / Downside_Deviation
```

**Significance Test**: Paired t-test comparing strategy vs baseline Sharpe ratios

### Benchmark Integration

**Benchmark**: NIFTY 50 Index (symbol: ^NSEI or NIFTY_50)

**Metrics**:
- **Alpha**: Excess return vs benchmark
  ```
  Alpha = Strategy_Return - (Risk_Free + Beta * (Benchmark_Return - Risk_Free))
  ```
- **Beta**: Sensitivity to benchmark movements
  ```
  Beta = Covariance(Strategy, Benchmark) / Variance(Benchmark)
  ```
- **Information Ratio**: Risk-adjusted alpha
  ```
  IR = Alpha / Tracking_Error
  ```
- **Z-Score**: Standardized performance vs benchmark
  ```
  Z = (Strategy_Return - Benchmark_Return) / Std(Strategy_Return)
  ```

### Statistical Tests Output Schema

**File**: `release/audit_<run_id>/stat_tests.json`

```json
{
  "run_id": "validation_20251012_180000",
  "timestamp": "2025-10-12T18:30:00+05:30",
  "status": "completed",
  "walk_forward_cv": {
    "method": "rolling_window",
    "window_size_months": 12,
    "step_size_months": 3,
    "num_folds": 4,
    "results": [
      {
        "fold": 1,
        "train_period": {"start": "2024-01-01", "end": "2024-12-31"},
        "test_period": {"start": "2025-01-01", "end": "2025-03-31"},
        "teacher_metrics": {"precision": 0.82, "recall": 0.78, "f1": 0.80},
        "student_metrics": {"accuracy": 0.84, "precision": 0.81, "recall": 0.78}
      }
    ],
    "aggregate": {
      "teacher": {
        "precision": {"mean": 0.82, "std": 0.03, "cv": 0.037},
        "recall": {"mean": 0.78, "std": 0.04, "cv": 0.051}
      },
      "student": {
        "accuracy": {"mean": 0.84, "std": 0.02, "cv": 0.024},
        "precision": {"mean": 0.81, "std": 0.03, "cv": 0.037}
      }
    }
  },
  "bootstrap_tests": {
    "method": "stratified_bootstrap",
    "n_iterations": 1000,
    "confidence_level": 0.95,
    "results": {
      "student_accuracy": {
        "mean": 0.84,
        "std": 0.015,
        "ci_lower": 0.81,
        "ci_upper": 0.87,
        "significant": true
      },
      "student_precision": {
        "mean": 0.81,
        "std": 0.018,
        "ci_lower": 0.77,
        "ci_upper": 0.85,
        "significant": true
      }
    }
  },
  "hypothesis_tests": {
    "student_vs_baseline": {
      "test": "paired_t_test",
      "metric": "accuracy",
      "baseline_mean": 0.75,
      "strategy_mean": 0.84,
      "delta": 0.09,
      "delta_pct": 12.0,
      "t_statistic": 3.45,
      "p_value": 0.002,
      "reject_null": true,
      "conclusion": "Strategy significantly outperforms baseline (p=0.002)"
    }
  },
  "sharpe_comparison": {
    "baseline": {
      "sharpe_ratio": 1.25,
      "sortino_ratio": 1.45,
      "annual_return": 0.085,
      "annual_volatility": 0.068
    },
    "strategy": {
      "sharpe_ratio": 1.62,
      "sortino_ratio": 1.89,
      "annual_return": 0.110,
      "annual_volatility": 0.068
    },
    "delta": {
      "sharpe_delta": 0.37,
      "sharpe_delta_pct": 29.6,
      "sortino_delta": 0.44,
      "sortino_delta_pct": 30.3
    },
    "significance": {
      "test": "bootstrap_sharpe_test",
      "p_value": 0.015,
      "reject_null": true,
      "conclusion": "Sharpe improvement is statistically significant (p=0.015)"
    }
  },
  "benchmark_comparison": {
    "benchmark": "NIFTY_50",
    "period": {"start": "2024-01-01", "end": "2024-12-31"},
    "benchmark_return": 0.095,
    "strategy_return": 0.110,
    "alpha": 0.015,
    "beta": 0.82,
    "information_ratio": 0.45,
    "tracking_error": 0.033,
    "z_score": 1.85,
    "correlation": 0.72,
    "relative_performance": {
      "excess_return": 0.015,
      "excess_return_pct": 15.8,
      "significant": true,
      "p_value": 0.035
    }
  }
}
```

### Enhanced Validation Summary

**Updates to `validation_summary.json`**:

```json
{
  "run_id": "validation_20251012_180000",
  "statistical_validation": {
    "status": "completed",
    "timestamp": "2025-10-12T18:30:00+05:30",
    "walk_forward_cv": {
      "student_accuracy_mean": 0.84,
      "student_accuracy_std": 0.02,
      "student_accuracy_cv": 0.024,
      "num_folds": 4
    },
    "confidence_intervals": {
      "student_accuracy": [0.81, 0.87],
      "student_precision": [0.77, 0.85]
    },
    "significance_tests": {
      "student_vs_baseline_accuracy": {
        "p_value": 0.002,
        "significant": true,
        "delta": 0.09
      }
    },
    "risk_adjusted_performance": {
      "sharpe_ratio": 1.62,
      "sharpe_delta_vs_baseline": 0.37,
      "sharpe_significant": true,
      "sortino_ratio": 1.89
    },
    "benchmark_comparison": {
      "benchmark": "NIFTY_50",
      "alpha": 0.015,
      "beta": 0.82,
      "information_ratio": 0.45,
      "z_score": 1.85,
      "outperformance_significant": true
    }
  },
  "promotion_recommendation": {
    "approved": true,
    "reason": "Accuracy thresholds met AND statistically significant improvement (p=0.002)",
    "statistical_confidence": "high",
    "risk_assessment": "favorable",
    "next_steps": [...]
  }
}
```

### StateManager Extensions

```python
def record_statistical_validation(
    self,
    run_id: str,
    timestamp: str,
    status: str,
    walk_forward_results: dict[str, Any],
    bootstrap_results: dict[str, Any],
    benchmark_comparison: dict[str, Any],
) -> None:
    """Record statistical validation results."""
    if "statistical_validations" not in self.state:
        self.state["statistical_validations"] = {}

    self.state["statistical_validations"][run_id] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "status": status,
        "walk_forward_results": walk_forward_results,
        "bootstrap_results": bootstrap_results,
        "benchmark_comparison": benchmark_comparison,
    }

    # Update last benchmark comparison
    self.state["last_benchmark_comparison"] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "benchmark": benchmark_comparison.get("benchmark"),
        "alpha": benchmark_comparison.get("alpha"),
    }

    self._save_state()

def get_statistical_validation(self, run_id: str) -> dict[str, Any] | None:
    """Get statistical validation by run ID."""
    return self.state.get("statistical_validations", {}).get(run_id)

def get_last_benchmark_comparison(self) -> dict[str, Any] | None:
    """Get last benchmark comparison results."""
    return self.state.get("last_benchmark_comparison")
```

## Configuration

**Default Settings**:
```python
# Walk-forward cross-validation
WALK_FORWARD_WINDOW_MONTHS = 12
WALK_FORWARD_STEP_MONTHS = 3
WALK_FORWARD_MIN_FOLDS = 3

# Bootstrap testing
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

# Significance testing
SIGNIFICANCE_ALPHA = 0.05  # p < 0.05 for rejection

# Benchmark
DEFAULT_BENCHMARK = "NIFTY_50"
RISK_FREE_RATE = 0.065  # 6.5% (India 10-year G-Sec)
```

## Usage Examples

### Run Full Statistical Validation

```bash
# Step 1: Run model validation (US-025)
python scripts/run_model_validation.py \
    --symbols RELIANCE TCS \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --no-dryrun

# Step 2: Run statistical tests (US-026)
python scripts/run_statistical_tests.py \
    --run-id validation_20251012_180000 \
    --benchmark NIFTY_50 \
    --bootstrap-iterations 1000
```

### Run with Integrated Workflow

```bash
# Run validation with statistical tests in one step
python scripts/run_model_validation.py \
    --symbols RELIANCE TCS \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --no-dryrun \
    --run-statistical-tests
```

### Skip Statistical Tests (Default Dryrun)

```bash
# Dryrun mode skips statistical tests
python scripts/run_model_validation.py
# No stat_tests.json generated
```

## Interpretation Guidelines

### P-Values
- **p < 0.01**: Strong evidence against null hypothesis (highly significant)
- **p < 0.05**: Moderate evidence (significant)
- **p >= 0.05**: Insufficient evidence (not significant)

### Confidence Intervals
- **95% CI [0.81, 0.87]**: We are 95% confident true accuracy is between 81% and 87%
- **Narrow CI**: More precise estimate, stable metric
- **Wide CI**: Less precise, high variability

### Coefficient of Variation (CV)
- **CV < 0.10**: Low variability (stable metric)
- **CV 0.10-0.20**: Moderate variability
- **CV > 0.20**: High variability (unstable metric)

### Sharpe Ratio
- **Sharpe > 2.0**: Excellent risk-adjusted returns
- **Sharpe 1.0-2.0**: Good risk-adjusted returns
- **Sharpe < 1.0**: Poor risk-adjusted returns

### Alpha & Beta
- **Alpha > 0**: Outperforming benchmark (excess returns)
- **Beta < 1**: Less volatile than benchmark (defensive)
- **Beta > 1**: More volatile than benchmark (aggressive)

### Information Ratio
- **IR > 0.5**: Strong outperformance vs tracking error
- **IR 0.2-0.5**: Moderate outperformance
- **IR < 0.2**: Weak outperformance

## Safety Controls

### Dryrun Mode
- Statistical tests **skipped** when validation in dryrun mode
- Placeholder results returned with status = "skipped"
- No benchmark data fetched

### Missing Data Handling
- Validation results missing → Skip statistical tests, log warning
- Benchmark data unavailable → Skip benchmark comparison
- Insufficient folds → Skip walk-forward CV, log error

### Error Degradation
- Bootstrap failure → Continue with point estimates, log warning
- Benchmark fetch failure → Continue without benchmark comparison
- Individual test failure → Continue with other tests

## Related User Stories

- **US-025**: Model Validation Run (prerequisite)
- **US-019**: Strategy Optimization (provides baseline configs)
- **US-016**: Accuracy Audit System (telemetry foundation)
- **US-022**: Release Audit Workflow (consumes stat test results)

## Next Steps

1. **Bayesian Methods**: Add Bayesian hypothesis testing
2. **Out-of-Sample Testing**: Strict temporal holdout sets
3. **Monte Carlo Simulation**: Simulate strategy performance distribution
4. **Multi-Benchmark Comparison**: Compare vs multiple indices (NIFTY, BANK NIFTY, etc.)
5. **Regime Analysis**: Test performance across market regimes (bull/bear/sideways)

---

**Status**: Ready for Implementation
**Dependencies**: US-025 (Model Validation)
**Estimated Effort**: 3-4 days
**Risk**: Medium (statistical methods require careful validation)
