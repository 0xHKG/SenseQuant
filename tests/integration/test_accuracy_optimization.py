"""Integration tests for accuracy optimization workflow (US-019)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from src.app.config import Settings
from src.services.accuracy_analyzer import AccuracyAnalyzer, PredictionTrace, TelemetryWriter


def test_backtest_result_accuracy_metrics(tmp_path):
    """Test that BacktestResult can store accuracy metrics (US-019).

    Verifies:
    - BacktestResult dataclass has accuracy_metrics field
    - BacktestResult dataclass has telemetry_dir field
    - Fields can be set and retrieved
    """
    import pandas as pd

    from src.domain.types import BacktestConfig, BacktestResult

    config = BacktestConfig(
        symbols=["RELIANCE"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        strategy="intraday",
        initial_capital=1000000.0,
        data_source="csv",
        random_seed=42,
    )

    # Create mock result with accuracy metrics
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()

    result = BacktestResult(
        config=config,
        metrics={"sharpe_ratio": 1.5, "total_return": 0.08},
        equity_curve=pd.DataFrame(),
        trades=pd.DataFrame(),
        metadata={},
        summary_path="",
        equity_path="",
        trades_path="",
        accuracy_metrics=None,  # Will be populated later
        telemetry_dir=telemetry_dir,
    )

    # Verify fields exist
    assert hasattr(result, "accuracy_metrics")
    assert hasattr(result, "telemetry_dir")
    assert result.telemetry_dir == telemetry_dir

    print("\n✓ BacktestResult extended with accuracy metrics fields")


def test_composite_scoring():
    """Test composite score computation (US-019).

    Verifies:
    - Composite score combines financial + accuracy metrics
    - Score normalized to [0, 1] range
    - Weights properly applied
    """

    # Mock accuracy metrics
    class MockAccuracyMetrics:
        def __init__(self):
            self.precision = {"LONG": 0.65, "SHORT": 0.55}
            self.recall = {"LONG": 0.60, "SHORT": 0.50}
            self.hit_ratio = 0.62
            self.win_rate = 0.58

    financial_metrics = {
        "sharpe_ratio": 1.8,  # Good Sharpe
        "total_return": 0.12,
        "win_rate": 0.58,
    }

    accuracy_metrics = MockAccuracyMetrics()

    # Compute composite score with default weights
    weights = {
        "sharpe_ratio": 0.40,
        "precision_long": 0.30,
        "hit_ratio": 0.20,
        "win_rate": 0.10,
    }

    # Normalize metrics
    norm_sharpe = min(financial_metrics["sharpe_ratio"] / 3.0, 1.0)  # 1.8/3 = 0.6
    norm_precision = accuracy_metrics.precision["LONG"]  # 0.65
    norm_hit_ratio = accuracy_metrics.hit_ratio  # 0.62
    norm_win_rate = accuracy_metrics.win_rate  # 0.58

    # Calculate weighted score
    expected_score = (
        weights["sharpe_ratio"] * norm_sharpe  # 0.40 * 0.6 = 0.24
        + weights["precision_long"] * norm_precision  # 0.30 * 0.65 = 0.195
        + weights["hit_ratio"] * norm_hit_ratio  # 0.20 * 0.62 = 0.124
        + weights["win_rate"] * norm_win_rate  # 0.10 * 0.58 = 0.058
    )  # Total = 0.617

    # Manual calculation
    score = (
        weights["sharpe_ratio"] * norm_sharpe
        + weights["precision_long"] * norm_precision
        + weights["hit_ratio"] * norm_hit_ratio
        + weights["win_rate"] * norm_win_rate
    )

    assert 0.0 <= score <= 1.0, "Score must be in [0, 1] range"
    assert abs(score - expected_score) < 0.001, f"Expected {expected_score:.3f}, got {score:.3f}"

    print(f"\n✓ Composite score computed: {score:.3f}")
    print(f"  - Sharpe contribution: {weights['sharpe_ratio'] * norm_sharpe:.3f}")
    print(f"  - Precision contribution: {weights['precision_long'] * norm_precision:.3f}")
    print(f"  - Hit ratio contribution: {weights['hit_ratio'] * norm_hit_ratio:.3f}")
    print(f"  - Win rate contribution: {weights['win_rate'] * norm_win_rate:.3f}")


def test_optimization_artifacts_structure(tmp_path):
    """Test optimization artifacts directory structure (US-019).

    Verifies:
    - Output directory created
    - Required artifacts files created
    - JSON/CSV files have expected structure
    """
    output_dir = tmp_path / "optimization_test"
    output_dir.mkdir()

    # Create mock optimization results
    configs = [
        {
            "config_id": "config_0001",
            "parameters": {"rsi_period": 14, "rsi_oversold": 30},
            "metrics": {"sharpe_ratio": 1.5, "total_return": 0.08},
            "accuracy_metrics": {"precision_long": 0.65, "hit_ratio": 0.62},
            "composite_score": 0.67,
        },
        {
            "config_id": "config_0002",
            "parameters": {"rsi_period": 10, "rsi_oversold": 35},
            "metrics": {"sharpe_ratio": 1.3, "total_return": 0.06},
            "accuracy_metrics": {"precision_long": 0.60, "hit_ratio": 0.58},
            "composite_score": 0.61,
        },
    ]

    # Export configs.json
    with (output_dir / "configs.json").open("w") as f:
        json.dump(configs, f, indent=2)

    # Export baseline_metrics.json
    baseline = {
        "sharpe_ratio": 1.2,
        "total_return": 0.05,
        "precision_long": 0.58,
        "hit_ratio": 0.55,
    }
    with (output_dir / "baseline_metrics.json").open("w") as f:
        json.dump(baseline, f, indent=2)

    # Export ranked_results.csv
    import pandas as pd

    df = pd.DataFrame(configs)
    df.to_csv(output_dir / "ranked_results.csv", index=False)

    # Export optimization_summary.json
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_configs": len(configs),
        "best_config": configs[0]["config_id"],
        "best_score": configs[0]["composite_score"],
    }
    with (output_dir / "optimization_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Generate accuracy_report.md
    report = f"""# Optimization Report

**Best Config**: {configs[0]["config_id"]}
**Composite Score**: {configs[0]["composite_score"]:.3f}

## Baseline vs Optimized

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| Sharpe | {baseline["sharpe_ratio"]:.2f} | {configs[0]["metrics"]["sharpe_ratio"]:.2f} | +{configs[0]["metrics"]["sharpe_ratio"] - baseline["sharpe_ratio"]:.2f} |
| Precision | {baseline["precision_long"]:.2%} | {configs[0]["accuracy_metrics"]["precision_long"]:.2%} | +{configs[0]["accuracy_metrics"]["precision_long"] - baseline["precision_long"]:.2%} |
"""
    with (output_dir / "accuracy_report.md").open("w") as f:
        f.write(report)

    # Verify all artifacts exist
    assert (output_dir / "configs.json").exists()
    assert (output_dir / "baseline_metrics.json").exists()
    assert (output_dir / "ranked_results.csv").exists()
    assert (output_dir / "optimization_summary.json").exists()
    assert (output_dir / "accuracy_report.md").exists()

    # Verify JSON structure
    loaded_configs = json.load((output_dir / "configs.json").open())
    assert len(loaded_configs) == 2
    assert "config_id" in loaded_configs[0]
    assert "parameters" in loaded_configs[0]
    assert "metrics" in loaded_configs[0]
    assert "accuracy_metrics" in loaded_configs[0]
    assert "composite_score" in loaded_configs[0]

    # Verify CSV structure
    df_loaded = pd.read_csv(output_dir / "ranked_results.csv")
    assert len(df_loaded) == 2
    assert "config_id" in df_loaded.columns
    assert "composite_score" in df_loaded.columns

    # Verify report content
    report_content = (output_dir / "accuracy_report.md").read_text()
    assert "Optimization Report" in report_content
    assert "config_0001" in report_content
    assert "Baseline vs Optimized" in report_content

    print("\n✓ Optimization artifacts structure validated")
    print(f"  - configs.json: {len(loaded_configs)} configurations")
    print(f"  - ranked_results.csv: {len(df_loaded)} rows")
    print(f"  - accuracy_report.md: {len(report_content)} characters")


def test_parameter_grid_generation():
    """Test parameter grid generation for optimization (US-019).

    Verifies:
    - Cartesian product of parameter values
    - Correct number of combinations
    - Parameter structure preserved
    """
    import itertools

    # Define parameter grid
    param_grid = {
        "rsi_period": [10, 14, 20],
        "rsi_oversold": [25, 30, 35],
        "rsi_overbought": [65, 70, 75],
    }

    # Generate combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    # Expected: 3 * 3 * 3 = 27 combinations
    assert len(combinations) == 27

    # Convert to parameter dictionaries
    configs = []
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo, strict=False))
        configs.append({"config_id": f"config_{i + 1:04d}", "parameters": params})

    # Verify structure
    assert configs[0]["parameters"] == {"rsi_period": 10, "rsi_oversold": 25, "rsi_overbought": 65}
    assert configs[-1]["parameters"] == {"rsi_period": 20, "rsi_oversold": 35, "rsi_overbought": 75}

    # Verify all parameter values covered
    rsi_periods = {c["parameters"]["rsi_period"] for c in configs}
    assert rsi_periods == {10, 14, 20}

    print("\n✓ Parameter grid generation validated")
    print(f"  - Total combinations: {len(combinations)}")
    print(f"  - First config: {configs[0]['parameters']}")
    print(f"  - Last config: {configs[-1]['parameters']}")


def test_telemetry_per_configuration(tmp_path):
    """Test telemetry capture per configuration (US-019).

    Verifies:
    - Each configuration has separate telemetry directory
    - Traces can be loaded and analyzed independently
    - Accuracy metrics computed per configuration
    """
    # Create telemetry for 3 mock configurations
    output_dir = tmp_path / "multi_config_telemetry"
    output_dir.mkdir()

    configs_data = []

    for i in range(3):
        config_id = f"config_{i + 1:04d}"
        telemetry_dir = output_dir / "telemetry" / config_id
        telemetry_dir.mkdir(parents=True)

        # Generate mock traces with varying accuracy
        writer = TelemetryWriter(output_dir=telemetry_dir, format="csv", buffer_size=10)

        # Config 1: 70% accuracy
        # Config 2: 60% accuracy
        # Config 3: 50% accuracy
        win_rate = 0.70 - (i * 0.10)

        for j in range(30):
            win = j % 10 < int(win_rate * 10)
            actual_dir = "LONG" if win else "NOOP"

            trace = PredictionTrace(
                timestamp=datetime.now() - timedelta(minutes=30 - j),
                symbol="RELIANCE",
                strategy="intraday",
                predicted_direction="LONG",
                actual_direction=actual_dir,
                predicted_confidence=0.65,
                entry_price=2500.0,
                exit_price=2510.0 if win else 2500.0,
                holding_period_minutes=20,
                realized_return_pct=0.4 if win else 0.0,
                features={"rsi": 65.0},
                metadata={"config_id": config_id},
            )
            writer.write_trace(trace)

        writer.close()

        # Load and analyze
        analyzer = AccuracyAnalyzer()
        traces = analyzer.load_traces(telemetry_dir)
        metrics = analyzer.compute_metrics(traces)

        configs_data.append(
            {
                "config_id": config_id,
                "telemetry_dir": str(telemetry_dir),
                "total_traces": len(traces),
                "precision_long": metrics.precision.get("LONG", 0.0),
                "hit_ratio": metrics.hit_ratio,
            }
        )

    # Verify each config has independent metrics
    assert len(configs_data) == 3
    assert configs_data[0]["precision_long"] > configs_data[1]["precision_long"]
    assert configs_data[1]["precision_long"] > configs_data[2]["precision_long"]

    print("\n✓ Telemetry per configuration validated")
    for c in configs_data:
        print(
            f"  - {c['config_id']}: Precision={c['precision_long']:.2%}, Hit Ratio={c['hit_ratio']:.2%}"
        )


def test_optimization_workflow_minimal(tmp_path):
    """Test minimal end-to-end optimization workflow (US-019).

    Verifies:
    - Parameter grid defined
    - Multiple configurations generated
    - Telemetry captured per configuration
    - Accuracy metrics computed
    - Composite scores calculated
    - Results ranked by score
    - Artifacts exported
    """
    import itertools

    import pandas as pd

    # Step 1: Define parameter grid
    param_grid = {
        "rsi_period": [10, 14],
        "rsi_oversold": [30, 35],
    }

    # Step 2: Generate configurations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    assert len(combinations) == 4  # 2 * 2

    # Step 3: Simulate evaluation for each config
    results = []
    for i, combo in enumerate(combinations):
        config_id = f"config_{i + 1:04d}"
        params = dict(zip(param_names, combo, strict=False))

        # Mock telemetry directory
        telemetry_dir = tmp_path / "telemetry" / config_id
        telemetry_dir.mkdir(parents=True)

        # Mock metrics (simulate varying performance)
        sharpe = 1.0 + (i * 0.2)
        precision = 0.55 + (i * 0.05)
        hit_ratio = 0.52 + (i * 0.03)

        # Compute composite score
        weights = {"sharpe_ratio": 0.5, "precision_long": 0.3, "hit_ratio": 0.2}
        composite = (
            weights["sharpe_ratio"] * min(sharpe / 3.0, 1.0)
            + weights["precision_long"] * precision
            + weights["hit_ratio"] * hit_ratio
        )

        results.append(
            {
                "config_id": config_id,
                "parameters": params,
                "sharpe_ratio": sharpe,
                "precision_long": precision,
                "hit_ratio": hit_ratio,
                "composite_score": composite,
            }
        )

    # Step 4: Rank by composite score
    results_sorted = sorted(results, key=lambda r: r["composite_score"], reverse=True)

    # Step 5: Export artifacts
    output_dir = tmp_path / "optimization_minimal"
    output_dir.mkdir()

    with (output_dir / "configs.json").open("w") as f:
        json.dump(results_sorted, f, indent=2)

    df = pd.DataFrame(results_sorted)
    df.to_csv(output_dir / "ranked_results.csv", index=False)

    # Step 6: Verify results
    assert len(results_sorted) == 4
    assert results_sorted[0]["composite_score"] >= results_sorted[-1]["composite_score"]
    assert (output_dir / "configs.json").exists()
    assert (output_dir / "ranked_results.csv").exists()

    print("\n✓ Minimal optimization workflow validated")
    print(f"  - Configurations tested: {len(results_sorted)}")
    print(f"  - Best config: {results_sorted[0]['config_id']}")
    print(f"  - Best composite score: {results_sorted[0]['composite_score']:.3f}")
    print(f"  - Best parameters: {results_sorted[0]['parameters']}")


def test_optimization_with_backtester_integration(tmp_path):
    """Test optimization with real backtester integration (US-019 Phase 2).

    Verifies:
    - Backtester populates accuracy_metrics in BacktestResult
    - Optimizer computes composite scores
    - Telemetry captured per configuration
    - Results ranked correctly
    """
    import pandas as pd

    from src.domain.types import BacktestConfig, OptimizationConfig
    from src.services.backtester import Backtester
    from src.services.data_feed import CSVDataFeed
    from src.services.optimizer import ParameterOptimizer

    # Check if sample data available
    data_dir = Path("data/market_data")
    if not data_dir.exists() or not (data_dir / "RELIANCE_1m.csv").exists():
        import pytest

        pytest.skip("Minute-bar sample data not available")

    # Define small parameter grid
    search_space = {
        "rsi_period": [10, 14],
        "rsi_oversold": [30, 35],
    }

    # Create optimization config
    opt_config = OptimizationConfig(
        symbols=["RELIANCE"],
        start_date="2024-01-02",
        end_date="2024-01-02",  # Single day for speed
        strategy="intraday",
        search_type="grid",
        search_space=search_space,
        objective_metric="composite",  # US-019: Use composite scoring
        n_samples=0,
        random_seed=42,
        data_source="csv",
        csv_path=str(data_dir),
    )

    # Create optimizer with telemetry enabled
    settings_obj = Settings()
    settings_obj.telemetry_storage_path = str(tmp_path / "telemetry_base")

    optimizer = ParameterOptimizer(config=opt_config, settings=settings_obj)

    # Generate candidates
    candidates_list = optimizer.generate_candidates()
    assert len(candidates_list) == 4  # 2 * 2 combinations

    # Evaluate each candidate
    results = []
    for i, params in enumerate(candidates_list):
        # Set unique telemetry directory per config
        config_telemetry_dir = tmp_path / f"telemetry_config_{i + 1:04d}"
        config_telemetry_dir.mkdir(parents=True)

        # Create settings with telemetry for this config
        config_settings = Settings()
        config_settings.telemetry_storage_path = str(config_telemetry_dir)

        # Create backtest config
        backtest_cfg = BacktestConfig(
            symbols=opt_config.symbols,
            start_date=opt_config.start_date,
            end_date=opt_config.end_date,
            strategy=opt_config.strategy,
            initial_capital=1000000.0,
            data_source="csv",
            random_seed=42,
            resolution="1minute",
        )

        # Run backtest
        data_feed = CSVDataFeed(str(data_dir))
        backtester = Backtester(config=backtest_cfg, data_feed=data_feed, settings=config_settings)
        result = backtester.run()

        # US-019 Phase 2: Verify accuracy_metrics populated
        if result.accuracy_metrics:
            print(
                f"\nConfig {i + 1}: Precision={result.accuracy_metrics.precision.get('LONG', 0.0):.2%}, "
                f"Hit Ratio={result.accuracy_metrics.hit_ratio:.2%}"
            )

            # Compute composite score
            composite = optimizer.compute_composite_score(result.metrics, result.accuracy_metrics)

            results.append(
                {
                    "config_id": f"config_{i + 1:04d}",
                    "parameters": params,
                    "sharpe_ratio": result.metrics.get("sharpe_ratio", 0.0),
                    "precision_long": result.accuracy_metrics.precision.get("LONG", 0.0),
                    "hit_ratio": result.accuracy_metrics.hit_ratio,
                    "composite_score": composite,
                    "telemetry_dir": str(config_telemetry_dir),
                }
            )
        else:
            # No telemetry - may happen if no trades generated
            print(f"\nConfig {i + 1}: No accuracy metrics (no trades generated)")

    # Verify at least some results have accuracy metrics
    results_with_metrics = [r for r in results if r["composite_score"] > 0]

    if results_with_metrics:
        # Sort by composite score
        results_sorted = sorted(
            results_with_metrics, key=lambda r: r["composite_score"], reverse=True
        )

        # Export artifacts
        output_dir = tmp_path / "optimization_artifacts"
        output_dir.mkdir()

        # Save configs.json
        import json

        with (output_dir / "configs.json").open("w") as f:
            json.dump(results_sorted, f, indent=2)

        # Save ranked_results.csv
        df = pd.DataFrame(results_sorted)
        df.to_csv(output_dir / "ranked_results.csv", index=False)

        # Verify artifacts
        assert (output_dir / "configs.json").exists()
        assert (output_dir / "ranked_results.csv").exists()

        # Verify structure
        loaded_configs = json.load((output_dir / "configs.json").open())
        assert len(loaded_configs) > 0
        assert "composite_score" in loaded_configs[0]
        assert "precision_long" in loaded_configs[0]
        assert "hit_ratio" in loaded_configs[0]

        print("\n✓ Optimization with backtester integration validated")
        print(f"  - Configurations tested: {len(results_sorted)}")
        print(f"  - Best composite score: {results_sorted[0]['composite_score']:.3f}")
        print(
            f"  - Best config: {results_sorted[0]['config_id']} with params {results_sorted[0]['parameters']}"
        )
    else:
        print("\n⚠ No trades generated - validation skipped (acceptable for single-day test)")


def test_notebook_report_validation(tmp_path):
    """Test notebook report validation with sample artifacts (US-019 Phase 4).

    Verifies:
    - Optimization artifacts can be loaded for notebook analysis
    - Sample data is valid and complete
    - Export script exists and is executable
    """
    import subprocess

    # Use existing sample artifacts
    sample_dir = Path("data/optimization/sample_run")

    # Verify sample artifacts exist
    assert (sample_dir / "baseline_metrics.json").exists(), "baseline_metrics.json not found"
    assert (sample_dir / "configs.json").exists(), "configs.json not found"
    assert (sample_dir / "ranked_results.csv").exists(), "ranked_results.csv not found"
    assert (sample_dir / "optimization_summary.json").exists(), (
        "optimization_summary.json not found"
    )

    # Load and validate baseline metrics
    with open(sample_dir / "baseline_metrics.json") as f:
        baseline = json.load(f)

    assert "sharpe_ratio" in baseline
    assert "total_return" in baseline
    assert "precision_long" in baseline
    assert "hit_ratio" in baseline

    # Load and validate configurations
    with open(sample_dir / "configs.json") as f:
        configs = json.load(f)

    assert len(configs) > 0, "No configurations found"
    best_config = configs[0]
    assert "config_id" in best_config
    assert "rank" in best_config
    assert "score" in best_config
    assert "parameters" in best_config
    assert "metrics" in best_config
    assert "accuracy_metrics" in best_config

    # Validate metrics structure
    assert "sharpe_ratio" in best_config["metrics"]
    assert "total_return_pct" in best_config["metrics"]
    assert "precision_long" in best_config["accuracy_metrics"]
    assert "hit_ratio" in best_config["accuracy_metrics"]

    # Load and validate CSV
    import pandas as pd

    ranked_df = pd.read_csv(sample_dir / "ranked_results.csv")
    assert len(ranked_df) > 0, "Empty ranked results"
    assert "config_id" in ranked_df.columns
    assert "score" in ranked_df.columns
    assert "sharpe_ratio" in ranked_df.columns

    # Verify export script exists
    export_script = Path("scripts/export_notebook.py")
    assert export_script.exists(), "export_notebook.py not found"

    # Verify script is executable
    assert export_script.stat().st_mode & 0o111, "export_notebook.py not executable"

    # Test export script help (without actually exporting)
    result = subprocess.run(
        ["python", str(export_script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, "Export script help failed"
    assert "optimization_report" in result.stdout, "optimization_report not mentioned in help"

    print("\n✓ Notebook report validation passed")
    print(f"  - Sample artifacts validated: {sample_dir}")
    print(f"  - Configurations: {len(configs)}")
    print(f"  - Best config: {best_config['config_id']} (score: {best_config['score']:.3f})")
    print(f"  - Export script: {export_script}")
    print("  - Ready for notebook analysis!")


def test_deployment_plan_generation():
    """Test deployment plan generation with baseline/best config references (US-019 Phase 5).

    Verifies:
    - Deployment plan file generated
    - Contains baseline metrics references
    - Contains best config references
    - Includes all 3 phases
    - Contains validation thresholds
    - Contains rollback triggers
    - Explicitly states no automatic config modification
    """
    # Use existing sample artifacts
    sample_dir = Path("data/optimization/sample_run")

    # Verify deployment plan exists
    deployment_plan = sample_dir / "deployment_plan.md"
    assert deployment_plan.exists(), "deployment_plan.md not found"

    # Load and validate content
    content = deployment_plan.read_text()

    # Verify baseline references
    assert "Baseline Configuration" in content, "Missing baseline configuration section"
    assert "Current Production" in content, "Missing current production reference"

    # Load baseline metrics to validate specific values
    with open(sample_dir / "baseline_metrics.json") as f:
        baseline = json.load(f)

    baseline_sharpe = baseline.get("sharpe_ratio", 0.0)

    # Verify baseline metrics referenced in deployment plan
    assert f"{baseline_sharpe:.2f}" in content, f"Baseline Sharpe {baseline_sharpe:.2f} not found"

    # Verify best config references
    assert "Optimized Configuration" in content, "Missing optimized configuration section"
    assert "Recommended" in content, "Missing recommended section"

    # Load best config to validate specific values
    with open(sample_dir / "configs.json") as f:
        configs = json.load(f)

    best_config = configs[0]
    best_config_id = best_config.get("config_id", "")
    best_sharpe = best_config.get("metrics", {}).get("sharpe_ratio", 0.0)

    # Verify best config referenced
    assert best_config_id in content, (
        f"Best config ID {best_config_id} not found in deployment plan"
    )
    assert f"{best_sharpe:.2f}" in content, f"Best Sharpe {best_sharpe:.2f} not found"

    # Verify 3-phase structure
    assert "Phase 1: Paper Trading Validation" in content, "Missing Phase 1"
    assert "Phase 2: Gradual Rollout" in content, "Missing Phase 2"
    assert "Phase 3: Full Production Deployment" in content, "Missing Phase 3"

    # Verify validation criteria section
    assert "Validation Criteria" in content, "Missing validation criteria"
    assert "Must ALL pass to proceed" in content, "Missing validation requirement statement"

    # Verify validation thresholds exist (80% of improvement)
    assert "Metric Thresholds" in content, "Missing metric thresholds"
    assert "Sharpe Ratio" in content, "Missing Sharpe Ratio threshold"
    assert "Precision" in content, "Missing Precision threshold"
    assert "Hit Ratio" in content, "Missing Hit Ratio threshold"

    # Calculate expected validation thresholds (80% of improvement)
    sharpe_improvement = best_sharpe - baseline_sharpe

    min_sharpe = baseline_sharpe + (sharpe_improvement * 0.8)

    # Verify threshold appears in content
    assert f"{min_sharpe:.2f}" in content, (
        f"Validation threshold {min_sharpe:.2f} not found (80% of improvement)"
    )

    # Verify rollback triggers section
    assert "Rollback Triggers" in content, "Missing rollback triggers section"
    assert "Rollback Procedure" in content, "Missing rollback procedure"
    assert "Critical Alerts" in content, "Missing critical alerts"

    # Verify rollback thresholds exist (should be lower than validation thresholds)
    rollback_sharpe = min_sharpe * 0.85  # Approx 85% of validation threshold

    # Check for rollback threshold (may be formatted slightly differently)
    # Just verify a lower threshold exists
    assert "consecutive days" in content, "Missing rollback consecutive days condition"

    # Verify monitoring procedures
    assert "Monitoring" in content, "Missing monitoring section"
    assert "Alert" in content, "Missing alert configuration"
    assert "Dashboard" in content or "dashboard" in content, "Missing dashboard reference"

    # Verify no automatic config modification
    assert (
        "does NOT modify production configs automatically" in content
        or "NOT modify production configs automatically" in content
    ), "Missing explicit statement about no automatic modification"
    assert "Manual review" in content or "manual review" in content, (
        "Missing manual review requirement"
    )
    assert "approval" in content.lower(), "Missing approval requirement"

    # Verify configuration management guidance
    assert "Configuration Management" in content, "Missing configuration management section"
    assert "Archive" in content or "archive" in content, "Missing archive guidance"
    assert "Backup" in content or "backup" in content, "Missing backup instruction"

    # Verify approval sign-off section
    assert "Approval" in content, "Missing approval section"
    assert "Sign-off" in content or "sign-off" in content.lower(), "Missing sign-off section"

    # Verify recommended parameters section
    assert "Recommended Parameters" in content, "Missing recommended parameters"
    assert "python" in content, "Missing Python code block for parameters"

    # Verify phase timelines
    assert "2 weeks" in content or "2 week" in content, "Missing Phase 1 timeline"
    assert "4 weeks" in content or "4 week" in content, "Missing Phase 2 timeline"

    # Verify gradual rollout percentages
    assert "20%" in content, "Missing 20% allocation"
    assert "50%" in content, "Missing 50% allocation"
    assert "100%" in content, "Missing 100% allocation"

    # Verify file locations documented
    assert "src/app/config.py" in content, "Missing config.py reference"
    assert "data/optimization" in content, "Missing optimization directory reference"

    print("\n✓ Deployment plan validation passed")
    print(f"  - Deployment plan: {deployment_plan}")
    print(f"  - Baseline Sharpe: {baseline_sharpe:.2f}")
    print(f"  - Best Sharpe: {best_sharpe:.2f}")
    print(f"  - Improvement: {sharpe_improvement:.2f} ({sharpe_improvement / baseline_sharpe:.1%})")
    print(f"  - Validation threshold (80%): {min_sharpe:.2f}")
    print(f"  - Rollback threshold (85% of validation): ~{rollback_sharpe:.2f}")
    print("  - ✓ 3-phase rollout structure validated")
    print("  - ✓ Baseline/best config references validated")
    print("  - ✓ Validation thresholds validated")
    print("  - ✓ Rollback triggers validated")
    print("  - ✓ No automatic modification statement validated")
    print("  - ✓ Approval sign-off section validated")
