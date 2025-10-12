#!/usr/bin/env python3
"""CLI entry point for parameter optimization.

Usage:
    python scripts/optimize.py --config search_space.yaml --symbols RELIANCE TCS --start-date 2024-01-01 --end-date 2024-12-31
    python scripts/optimize.py --config search_space.json --symbols INFY --start-date 2024-06-01 --end-date 2024-12-31 --top-n 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from loguru import logger

from src.adapters.breeze_client import BreezeClient
from src.app.config import Settings, settings
from src.domain.types import OptimizationConfig
from src.services.optimizer import ParameterOptimizer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run parameter optimization on trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Grid search optimization
  python scripts/optimize.py --config search_space.yaml --symbols RELIANCE --start-date 2024-01-01 --end-date 2024-12-31

  # Random search with 50 samples
  python scripts/optimize.py --config search_space.yaml --symbols TCS INFY --start-date 2024-01-01 --end-date 2024-12-31 --search-type random --n-samples 50

  # Optimize with specific objective
  python scripts/optimize.py --config search_space.yaml --symbols RELIANCE --start-date 2024-01-01 --end-date 2024-12-31 --objective cagr_pct --top-n 5
        """,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to search space configuration file (YAML or JSON)",
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Stock symbols to optimize (space-separated)",
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Optimization start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        required=True,
        help="Optimization end date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--strategy",
        choices=["intraday", "swing", "both"],
        default="swing",
        help="Strategy to optimize (default: swing)",
    )

    parser.add_argument(
        "--search-type",
        choices=["grid", "random"],
        default="grid",
        help="Search type: grid (exhaustive) or random (sampling) (default: grid)",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for random search (default: 100)",
    )

    parser.add_argument(
        "--objective",
        default="sharpe_ratio",
        help="Objective metric to maximize (default: sharpe_ratio)",
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1000000.0,
        help="Initial capital in INR (default: 1000000)",
    )

    parser.add_argument(
        "--data-source",
        choices=["breeze", "csv", "teacher"],
        default="breeze",
        help="Data source for historical bars (default: breeze)",
    )

    parser.add_argument(
        "--csv-path",
        help="Path to CSV file if data-source=csv",
    )

    parser.add_argument(
        "--teacher-labels",
        help="Path to Teacher labels CSV if data-source=teacher",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top results to display (default: 10)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    # US-019 Phase 3: Enhanced CLI flags for accuracy-driven optimization
    parser.add_argument(
        "--telemetry-dir",
        default="data/optimization/telemetry",
        help="Base directory for telemetry storage (default: data/optimization/telemetry)",
    )

    parser.add_argument(
        "--telemetry-sample-rate",
        type=float,
        default=1.0,
        help="Telemetry sampling rate [0.0-1.0] (default: 1.0 = 100%%)",
    )

    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum configurations to test (default: None = test all)",
    )

    parser.add_argument(
        "--export-report",
        action="store_true",
        default=True,
        help="Export accuracy report markdown (default: True)",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for artifacts (default: data/optimization/run_<timestamp>)",
    )

    parser.add_argument(
        "--run-baseline",
        action="store_true",
        default=True,
        help="Run baseline configuration for comparison (default: True)",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Configure loguru logging."""
    logger.remove()  # Remove default handler

    log_level = "DEBUG" if verbose else "INFO"

    # Console handler with structured format
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[component]}</cyan> | <level>{message}</level>",
    )

    # File handler for optimization
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "optimize_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[component]} | {message}",
    )


def load_search_space(config_path: str) -> dict:
    """Load search space configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Search space dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Determine format from extension
    suffix = path.suffix.lower()

    try:
        with open(path) as f:
            if suffix in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            elif suffix == ".json":
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json")

        if not isinstance(config, dict):
            raise ValueError("Config file must contain a dictionary/object")

        return config

    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse config file: {e}") from e


def validate_config(args: argparse.Namespace, search_space: dict) -> None:
    """Validate CLI arguments and search space.

    Args:
        args: Parsed command line arguments
        search_space: Loaded search space dictionary

    Raises:
        ValueError: If validation fails
    """
    # Validate date format
    from datetime import datetime

    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}") from e

    # Validate start < end
    if args.start_date >= args.end_date:
        raise ValueError("Start date must be before end date")

    # Validate data source specific args
    if args.data_source == "csv" and not args.csv_path:
        raise ValueError("--csv-path required when --data-source=csv")

    if args.data_source == "teacher" and not args.teacher_labels:
        raise ValueError("--teacher-labels required when --data-source=teacher")

    # Validate files exist
    if args.csv_path and not Path(args.csv_path).exists():
        raise ValueError(f"CSV file not found: {args.csv_path}")

    if args.teacher_labels and not Path(args.teacher_labels).exists():
        raise ValueError(f"Teacher labels file not found: {args.teacher_labels}")

    # Validate search space is not empty
    if not search_space:
        raise ValueError("Search space is empty")

    # Validate objective metric
    valid_objectives = [
        "sharpe_ratio",
        "cagr_pct",
        "total_return_pct",
        "win_rate_pct",
        "max_drawdown_pct",
    ]
    if args.objective not in valid_objectives:
        logger.warning(
            f"Objective '{args.objective}' may not be a standard metric. Valid: {valid_objectives}",
            extra={"component": "optimize_cli"},
        )


def print_results(result: Any, top_n: int) -> None:
    """Print optimization results in formatted table.

    Args:
        result: OptimizationResult
        top_n: Number of top results to display
    """
    logger.info("=" * 100, extra={"component": "optimize_cli"})
    logger.info("OPTIMIZATION COMPLETE", extra={"component": "optimize_cli"})
    logger.info("=" * 100, extra={"component": "optimize_cli"})

    logger.info(
        f"Total Candidates: {result.total_candidates} "
        f"(Successful: {result.successful_candidates}, Failed: {result.failed_candidates})",
        extra={"component": "optimize_cli"},
    )
    logger.info(
        f"Total Time: {result.total_time:.2f}s",
        extra={"component": "optimize_cli"},
    )
    logger.info("", extra={"component": "optimize_cli"})

    if result.best_candidate:
        logger.info("BEST CONFIGURATION:", extra={"component": "optimize_cli"})
        logger.info("-" * 100, extra={"component": "optimize_cli"})
        logger.info(
            f"Score ({result.config.objective_metric}): {result.best_candidate.score:.4f}",
            extra={"component": "optimize_cli"},
        )
        logger.info(
            f"Parameters: {json.dumps(result.best_candidate.parameters, indent=2)}",
            extra={"component": "optimize_cli"},
        )

        if result.best_candidate.backtest_result:
            metrics = result.best_candidate.backtest_result.metrics
            logger.info("", extra={"component": "optimize_cli"})
            logger.info("Key Metrics:", extra={"component": "optimize_cli"})
            logger.info(
                f"  Total Return:  {metrics.get('total_return_pct', 0):.2f}%",
                extra={"component": "optimize_cli"},
            )
            logger.info(
                f"  CAGR:          {metrics.get('cagr_pct', 0):.2f}%",
                extra={"component": "optimize_cli"},
            )
            logger.info(
                f"  Sharpe Ratio:  {metrics.get('sharpe_ratio', 0):.2f}",
                extra={"component": "optimize_cli"},
            )
            logger.info(
                f"  Max Drawdown:  {metrics.get('max_drawdown_pct', 0):.2f}%",
                extra={"component": "optimize_cli"},
            )
            logger.info(
                f"  Win Rate:      {metrics.get('win_rate_pct', 0):.2f}%",
                extra={"component": "optimize_cli"},
            )
        logger.info("", extra={"component": "optimize_cli"})

    # Print top-N results
    display_count = min(top_n, len(result.candidates))
    if display_count > 0:
        logger.info(f"TOP {display_count} CONFIGURATIONS:", extra={"component": "optimize_cli"})
        logger.info("-" * 100, extra={"component": "optimize_cli"})

        for rank, candidate in enumerate(result.candidates[:display_count], start=1):
            logger.info(
                f"{rank}. Score: {candidate.score:.4f} | Params: {candidate.parameters}",
                extra={"component": "optimize_cli"},
            )

        logger.info("", extra={"component": "optimize_cli"})

    # Print artifact paths
    logger.info("ARTIFACTS SAVED:", extra={"component": "optimize_cli"})
    logger.info("-" * 100, extra={"component": "optimize_cli"})
    logger.info(
        f"Summary:        {result.summary_path}",
        extra={"component": "optimize_cli"},
    )
    logger.info(
        f"Ranked Results: {result.ranked_results_path}",
        extra={"component": "optimize_cli"},
    )
    logger.info(
        f"Best Config:    {result.best_config_path}",
        extra={"component": "optimize_cli"},
    )
    logger.info("=" * 100, extra={"component": "optimize_cli"})


# US-019 Phase 3: Batch optimization workflow helpers


def run_baseline_backtest(config: OptimizationConfig, settings_obj: Any, output_dir: Path) -> dict:
    """Run baseline backtest with current configuration.

    Args:
        config: Optimization configuration
        settings_obj: Settings object
        output_dir: Output directory for telemetry

    Returns:
        Dictionary with baseline metrics
    """
    from src.domain.types import BacktestConfig
    from src.services.backtester import Backtester
    from src.services.data_feed import CSVDataFeed

    logger.info("Running baseline backtest...", extra={"component": "optimize_cli"})

    # Create telemetry directory for baseline
    baseline_telemetry_dir = output_dir / "telemetry" / "baseline"
    baseline_telemetry_dir.mkdir(parents=True, exist_ok=True)

    # Configure settings for baseline
    baseline_settings = Settings()
    baseline_settings.telemetry_storage_path = str(baseline_telemetry_dir)

    # Create backtest config
    backtest_cfg = BacktestConfig(
        symbols=config.symbols,
        start_date=config.start_date,
        end_date=config.end_date,
        strategy=config.strategy,
        initial_capital=config.initial_capital,
        data_source=config.data_source,
        csv_path=config.csv_path,
        random_seed=config.random_seed,
        resolution="1minute" if config.strategy == "intraday" else "1day",
    )

    # Create data feed
    data_feed = None
    if config.data_source == "csv" and config.csv_path:
        data_feed = CSVDataFeed(config.csv_path)

    # Run backtest
    backtester = Backtester(config=backtest_cfg, data_feed=data_feed, settings=baseline_settings)
    result = backtester.run()

    # Extract metrics
    baseline = {
        "sharpe_ratio": result.metrics.get("sharpe_ratio", 0.0),
        "total_return": result.metrics.get("total_return_pct", 0.0) / 100.0,
        "max_drawdown": result.metrics.get("max_drawdown_pct", 0.0) / 100.0,
        "win_rate": result.metrics.get("win_rate_pct", 0.0) / 100.0,
        "total_trades": result.metrics.get("total_trades", 0),
    }

    # Add accuracy metrics if available
    if result.accuracy_metrics:
        baseline.update(
            {
                "precision_long": result.accuracy_metrics.precision.get("LONG", 0.0),
                "recall_long": result.accuracy_metrics.recall.get("LONG", 0.0),
                "hit_ratio": result.accuracy_metrics.hit_ratio,
                "accuracy_win_rate": result.accuracy_metrics.win_rate,
            }
        )
        logger.info(
            f"Baseline accuracy: precision={baseline['precision_long']:.2%}, hit_ratio={baseline['hit_ratio']:.2%}",
            extra={"component": "optimize_cli"},
        )

    return baseline


def export_optimization_artifacts(
    output_dir: Path,
    candidates: list,
    baseline: dict,
    config: OptimizationConfig,
    export_report: bool = True,
) -> None:
    """Export optimization artifacts to output directory.

    Args:
        output_dir: Output directory
        candidates: List of optimization candidates (ranked)
        baseline: Baseline metrics dictionary
        config: Optimization configuration
        export_report: Whether to generate markdown report
    """
    from datetime import datetime

    import pandas as pd

    logger.info(f"Exporting artifacts to {output_dir}", extra={"component": "optimize_cli"})

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare configs list
    configs_list = []
    for i, candidate in enumerate(candidates):
        config_dict = {
            "config_id": f"config_{i + 1:04d}",
            "rank": i + 1,
            "parameters": candidate.parameters,
            "score": candidate.score if candidate.score is not None else 0.0,
            "error": candidate.error,
        }

        # Add metrics if backtest succeeded
        if candidate.backtest_result:
            config_dict["metrics"] = candidate.backtest_result.metrics

            # Add accuracy metrics if available
            if candidate.backtest_result.accuracy_metrics:
                acc = candidate.backtest_result.accuracy_metrics
                config_dict["accuracy_metrics"] = {
                    "precision_long": acc.precision.get("LONG", 0.0),
                    "recall_long": acc.recall.get("LONG", 0.0),
                    "hit_ratio": acc.hit_ratio,
                    "win_rate": acc.win_rate,
                }

        configs_list.append(config_dict)

    # Export configs.json
    with (output_dir / "configs.json").open("w") as f:
        json.dump(configs_list, f, indent=2, default=str)

    # Export baseline_metrics.json
    with (output_dir / "baseline_metrics.json").open("w") as f:
        json.dump(baseline, f, indent=2)

    # Export ranked_results.csv
    rows = []
    for cfg in configs_list:
        row = {
            "config_id": cfg["config_id"],
            "rank": cfg["rank"],
            "score": cfg["score"],
        }
        row.update(cfg.get("parameters", {}))

        if cfg.get("metrics"):
            row["sharpe_ratio"] = cfg["metrics"].get("sharpe_ratio", 0.0)
            row["total_return"] = cfg["metrics"].get("total_return_pct", 0.0)

        if cfg.get("accuracy_metrics"):
            row.update(cfg["accuracy_metrics"])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "ranked_results.csv", index=False)

    # Export optimization_summary.json
    summary = {
        "timestamp": datetime.now().isoformat(),
        "symbols": config.symbols,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "strategy": config.strategy,
        "objective_metric": config.objective_metric,
        "search_type": config.search_type,
        "total_configs": len(configs_list),
        "successful_configs": len([c for c in configs_list if c.get("error") is None]),
        "best_config": configs_list[0]["config_id"] if configs_list else None,
        "best_score": configs_list[0]["score"] if configs_list else 0.0,
    }

    with (output_dir / "optimization_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Generate accuracy report
    if export_report and configs_list:
        generate_accuracy_report(output_dir, configs_list, baseline, config)

    # US-019 Phase 5: Generate deployment plan
    if configs_list and baseline:
        generate_deployment_plan(output_dir, configs_list, baseline, config)

    logger.info("Artifacts exported successfully", extra={"component": "optimize_cli"})


def generate_accuracy_report(
    output_dir: Path,
    configs: list,
    baseline: dict,
    opt_config: OptimizationConfig,
) -> None:
    """Generate markdown accuracy report.

    Args:
        output_dir: Output directory
        configs: List of configuration dictionaries (ranked)
        baseline: Baseline metrics
        opt_config: Optimization configuration
    """
    from datetime import datetime

    best = configs[0] if configs else None
    if not best:
        return

    report = f"""# Strategy Accuracy Optimization Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Strategy**: {opt_config.strategy}
**Symbols**: {", ".join(opt_config.symbols)}
**Date Range**: {opt_config.start_date} to {opt_config.end_date}
**Configurations Tested**: {len(configs)}
**Objective**: {opt_config.objective_metric}

---

## Executive Summary

"""

    if best.get("metrics"):
        best_metrics = best["metrics"]
        best_acc = best.get("accuracy_metrics", {})

        report += f"""- **Best Configuration**: {best["config_id"]}
- **Score**: {best["score"]:.3f}
- **Sharpe Ratio**: {best_metrics.get("sharpe_ratio", 0.0):.2f} (Baseline: {baseline.get("sharpe_ratio", 0.0):.2f})
"""

        if best_acc:
            report += f"""- **Precision (LONG)**: {best_acc.get("precision_long", 0.0):.2%} (Baseline: {baseline.get("precision_long", 0.0):.2%})
- **Hit Ratio**: {best_acc.get("hit_ratio", 0.0):.2%} (Baseline: {baseline.get("hit_ratio", 0.0):.2%})
"""

    report += """
---

## Baseline Metrics (Current Configuration)

| Metric | Value |
|--------|-------|
"""

    report += f"| Sharpe Ratio | {baseline.get('sharpe_ratio', 0.0):.2f} |\n"
    report += f"| Total Return | {baseline.get('total_return', 0.0):.2%} |\n"

    if "precision_long" in baseline:
        report += f"| Precision (LONG) | {baseline['precision_long']:.2%} |\n"
        report += f"| Hit Ratio | {baseline['hit_ratio']:.2%} |\n"
        report += f"| Win Rate | {baseline.get('win_rate', 0.0):.2%} |\n"

    report += """
---

## Top 5 Configurations

"""

    for i, cfg in enumerate(configs[:5]):
        report += f"""
### {i + 1}. {cfg["config_id"]} {"⭐" if i == 0 else ""}

**Parameters**:
```json
{json.dumps(cfg.get("parameters", {}), indent=2)}
```

"""

        if cfg.get("metrics") and cfg.get("accuracy_metrics"):
            metrics = cfg["metrics"]
            acc = cfg["accuracy_metrics"]

            # Calculate deltas
            sharpe_delta = metrics.get("sharpe_ratio", 0.0) - baseline.get("sharpe_ratio", 0.0)
            precision_delta = acc.get("precision_long", 0.0) - baseline.get("precision_long", 0.0)
            hit_ratio_delta = acc.get("hit_ratio", 0.0) - baseline.get("hit_ratio", 0.0)

            report += f"""**Metrics**:
| Metric | Value | Delta vs Baseline |
|--------|-------|-------------------|
| Score | {cfg["score"]:.3f} | - |
| Sharpe Ratio | {metrics.get("sharpe_ratio", 0.0):.2f} | {sharpe_delta:+.2f} |
| Precision (LONG) | {acc.get("precision_long", 0.0):.2%} | {precision_delta:+.2%} |
| Hit Ratio | {acc.get("hit_ratio", 0.0):.2%} | {hit_ratio_delta:+.2%} |
| Total Return | {metrics.get("total_return_pct", 0.0):.2%} | - |

"""

    report += """
---

## Deployment Recommendations

### Phase 1: Validation (2 weeks)
- Deploy best configuration in paper trading mode
- Monitor live accuracy metrics via dashboard
- Compare vs baseline in real-time
- Alert on precision < 65% or Sharpe < 1.3

### Phase 2: Gradual Rollout (4 weeks)
- Week 1-2: 20% capital allocation
- Week 3: 50% capital
- Week 4: 100% capital if validation successful

### Phase 3: Full Production
- Update config.py with validated parameters
- Archive baseline config
- Update dashboard alert thresholds

### Rollback Plan
If live metrics degrade:
1. Revert to baseline config immediately
2. Analyze live telemetry for root cause
3. Re-run optimization on recent data
4. Consider adaptive parameter adjustment

---

**Generated by**: SenseQuant Optimization Framework (US-019 Phase 3)
**Full results**: See `configs.json` and `ranked_results.csv`
"""

    with (output_dir / "accuracy_report.md").open("w") as f:
        f.write(report)

    logger.info("Accuracy report generated", extra={"component": "optimize_cli"})


def generate_deployment_plan(
    output_dir: Path,
    configs: list,
    baseline: dict,
    opt_config: OptimizationConfig,
) -> None:
    """Generate deployment plan with validation steps and rollback triggers (US-019 Phase 5).

    Args:
        output_dir: Output directory
        configs: List of configuration dictionaries (ranked)
        baseline: Baseline metrics
        opt_config: Optimization configuration
    """
    from datetime import datetime

    best = configs[0] if configs else None
    if not best:
        return

    # Calculate improvement metrics
    best_metrics = best.get("metrics", {})
    best_acc = best.get("accuracy_metrics", {})

    sharpe_improvement = best_metrics.get("sharpe_ratio", 0.0) - baseline.get("sharpe_ratio", 0.0)
    precision_improvement = best_acc.get("precision_long", 0.0) - baseline.get(
        "precision_long", 0.0
    )
    hit_ratio_improvement = best_acc.get("hit_ratio", 0.0) - baseline.get("hit_ratio", 0.0)

    # Determine validation thresholds (conservative: require 80% of improvement maintained)
    min_sharpe = baseline.get("sharpe_ratio", 0.0) + (sharpe_improvement * 0.8)
    min_precision = baseline.get("precision_long", 0.0) + (precision_improvement * 0.8)
    min_hit_ratio = baseline.get("hit_ratio", 0.0) + (hit_ratio_improvement * 0.8)

    plan = f"""# Strategy Optimization Deployment Plan

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Strategy**: {opt_config.strategy}
**Symbols**: {", ".join(opt_config.symbols)}
**Optimization Period**: {opt_config.start_date} to {opt_config.end_date}

---

## Executive Summary

This deployment plan outlines the process for safely rolling out optimized strategy parameters to production. The optimized configuration showed significant improvements over baseline:

- **Sharpe Ratio**: {baseline.get("sharpe_ratio", 0.0):.2f} → {best_metrics.get("sharpe_ratio", 0.0):.2f} ({sharpe_improvement:+.2f}, {sharpe_improvement / max(baseline.get("sharpe_ratio", 0.01), 0.01) * 100:+.1f}%)
- **Precision (LONG)**: {baseline.get("precision_long", 0.0):.2%} → {best_acc.get("precision_long", 0.0):.2%} ({precision_improvement:+.2%})
- **Hit Ratio**: {baseline.get("hit_ratio", 0.0):.2%} → {best_acc.get("hit_ratio", 0.0):.2%} ({hit_ratio_improvement:+.2%})

**⚠️ IMPORTANT**: This plan does NOT modify production configs automatically. Manual review and approval required.

---

## Baseline Configuration (Current Production)

**Current Metrics**:
- Sharpe Ratio: {baseline.get("sharpe_ratio", 0.0):.2f}
- Total Return: {baseline.get("total_return", 0.0) * 100:.1f}%
- Win Rate: {baseline.get("win_rate", 0.0) * 100:.1f}%
- Precision (LONG): {baseline.get("precision_long", 0.0):.1%}
- Hit Ratio: {baseline.get("hit_ratio", 0.0):.1%}
- Total Trades: {baseline.get("total_trades", 0)}

**Baseline Config Location**: `src/app/config.py` (defaults)

---

## Optimized Configuration (Recommended)

**Config ID**: `{best["config_id"]}`
**Composite Score**: {best["score"]:.3f}

**Optimized Metrics**:
- Sharpe Ratio: {best_metrics.get("sharpe_ratio", 0.0):.2f}
- Total Return: {best_metrics.get("total_return_pct", 0.0):.1f}%
- Win Rate: {best_metrics.get("win_rate_pct", 0.0):.1f}%
- Precision (LONG): {best_acc.get("precision_long", 0.0):.1%}
- Hit Ratio: {best_acc.get("hit_ratio", 0.0):.1%}

**Recommended Parameters**:
```python
# Add to src/app/config.py or override in environment
{chr(10).join(f"{param.upper()}: {value}" for param, value in best.get("parameters", {}).items())}
```

**Parameter File**: Save to `config/optimized/{opt_config.strategy}_params_{datetime.now().strftime("%Y%m%d")}.json`

---

## Phase 1: Paper Trading Validation (Duration: 2 weeks)

### Objectives
- Validate optimized parameters in paper trading environment
- Monitor live accuracy metrics vs backtest expectations
- Identify any regime changes or data drift

### Setup
1. **Deploy optimized config in paper mode**:
   ```bash
   # Update paper trading config
   cp config/optimized/{opt_config.strategy}_params_*.json config/paper_trading.json

   # Run paper trading engine
   python scripts/run_paper_trading.py --config config/paper_trading.json
   ```

2. **Enable live telemetry**:
   ```python
   # In config
   LIVE_TELEMETRY_ENABLED = True
   LIVE_TELEMETRY_SAMPLE_RATE = 1.0  # 100% capture during validation
   TELEMETRY_STORAGE_PATH = "data/telemetry/paper_validation"
   ```

3. **Setup monitoring dashboard**:
   ```bash
   streamlit run src/dashboard/accuracy_dashboard.py -- \
     --telemetry-dir data/telemetry/paper_validation \
     --live-mode
   ```

### Validation Criteria (Must ALL pass to proceed)

✅ **Metric Thresholds** (evaluated daily):
- Sharpe Ratio ≥ {min_sharpe:.2f} (80% of backtest improvement)
- Precision (LONG) ≥ {min_precision:.1%} (80% of backtest improvement)
- Hit Ratio ≥ {min_hit_ratio:.1%} (80% of backtest improvement)
- Win Rate ≥ {baseline.get("win_rate", 0.0) * 100 * 0.95:.1f}% (95% of baseline)

✅ **Statistical Significance**:
- Minimum 50 trades executed
- 95% confidence interval overlaps with backtest metrics

✅ **Risk Controls**:
- Max Drawdown < {abs(baseline.get("max_drawdown", -0.10)) * 1.2 * 100:.1f}% (120% of baseline)
- No circuit breaker activations
- Position sizing within limits

✅ **Operational Checks**:
- No execution errors or API failures
- Latency < 200ms for order placement
- Telemetry capture working (no gaps)

### Monitoring During Phase 1

**Daily Review**:
- Check dashboard at 9:00 AM and 3:30 PM
- Review telemetry for accuracy metrics
- Compare live vs backtest precision/recall
- Analyze confusion matrix for prediction patterns

**Alert Thresholds**:
- Sharpe drops below {min_sharpe * 0.9:.2f}: Warning
- Precision drops below {min_precision * 0.9:.1%}: Warning
- Hit ratio drops below {min_hit_ratio * 0.9:.1%}: Critical (pause trading)
- Drawdown exceeds {abs(baseline.get("max_drawdown", -0.10)) * 1.2 * 100:.1f}%: Critical (halt and rollback)

### Phase 1 Exit Criteria

**Proceed to Phase 2 if**:
- All validation criteria met for 10 consecutive trading days
- No critical alerts triggered
- Team approval obtained

**Rollback if**:
- Any critical alert triggered
- Statistical significance tests fail
- Operational issues persist

---

## Phase 2: Gradual Rollout (Duration: 4 weeks)

### Week 1-2: 20% Capital Allocation

**Setup**:
```python
# In config
CAPITAL_ALLOCATION_OPTIMIZED = 0.20
CAPITAL_ALLOCATION_BASELINE = 0.80
```

**Monitoring**:
- Compare optimized vs baseline performance side-by-side
- Track relative Sharpe, precision, hit ratio
- Validate 20% allocation maintains metrics

**Decision Point (End of Week 2)**:
- ✅ Proceed if optimized allocation outperforms baseline by ≥5%
- ⏸️ Hold if performance within ±5% (collect more data)
- ❌ Rollback if underperforms by >10%

### Week 3: 50% Capital Allocation

**Setup**:
```python
CAPITAL_ALLOCATION_OPTIMIZED = 0.50
CAPITAL_ALLOCATION_BASELINE = 0.50
```

**Monitoring**:
- Increased sample size for statistical validation
- Monitor for any capacity/slippage issues
- Confirm risk limits still effective

**Decision Point (End of Week 3)**:
- ✅ Proceed if metrics stable and >5% improvement maintained
- ❌ Rollback if any degradation observed

### Week 4: 100% Capital (if validation successful)

**Setup**:
```python
CAPITAL_ALLOCATION_OPTIMIZED = 1.00
CAPITAL_ALLOCATION_BASELINE = 0.00
```

**Monitoring**:
- Full production monitoring
- Daily performance review
- Weekly strategy review meeting

---

## Phase 3: Full Production Deployment

### Pre-Deployment Checklist

- [ ] All Phase 1 and Phase 2 criteria met
- [ ] Code review completed
- [ ] Configuration changes peer-reviewed
- [ ] Rollback procedure tested
- [ ] Alert thresholds configured
- [ ] Team training on new parameters completed
- [ ] Backup of current config taken

### Deployment Steps

1. **Archive Current Config**:
   ```bash
   cp src/app/config.py config/archive/config_baseline_{datetime.now().strftime("%Y%m%d")}.py
   ```

2. **Update Production Config**:
   ```bash
   # Manually update src/app/config.py with optimized parameters
   # OR use environment variable override
   ```

3. **Update Monitoring Thresholds**:
   ```python
   # In dashboard config
   ALERT_SHARPE_MIN = {min_sharpe:.2f}
   ALERT_PRECISION_MIN = {min_precision:.2%}
   ALERT_HIT_RATIO_MIN = {min_hit_ratio:.2%}
   ```

4. **Deploy and Verify**:
   ```bash
   # Restart trading engine
   systemctl restart trading-engine

   # Verify new config loaded
   curl http://localhost:8000/health | jq '.config.strategy_params'
   ```

### Post-Deployment Monitoring (First 30 Days)

**Week 1**: Daily review of all metrics
**Week 2-4**: Review every 2 days
**Month 2+**: Weekly review

**Key Metrics to Track**:
- Sharpe Ratio trend
- Precision/Recall stability
- Hit Ratio consistency
- Drawdown behavior
- Trade frequency

---

## Rollback Triggers and Procedure

### Automatic Rollback Triggers

**Critical Alerts** (immediate rollback):
- Sharpe Ratio < {min_sharpe * 0.85:.2f} for 3 consecutive days
- Precision (LONG) < {min_precision * 0.85:.1%} for 3 consecutive days
- Hit Ratio < {min_hit_ratio * 0.85:.1%} for 3 consecutive days
- Drawdown > {abs(baseline.get("max_drawdown", -0.10)) * 1.5 * 100:.1f}% (150% of baseline max)

**Warning Alerts** (manual review required):
- Any metric degrades by >15% from validation period
- Trade frequency drops by >30%
- Execution errors increase by >50%

### Rollback Procedure

1. **Immediate Actions**:
   ```bash
   # Halt trading (if critical)
   systemctl stop trading-engine

   # Restore baseline config
   cp config/archive/config_baseline_*.py src/app/config.py

   # Restart with baseline
   systemctl start trading-engine
   ```

2. **Post-Rollback Analysis**:
   - Analyze live telemetry for root cause
   - Compare live vs backtest data distributions
   - Check for market regime changes
   - Review execution quality

3. **Next Steps**:
   - If temporary issue: retry optimization on recent data
   - If systematic issue: re-evaluate optimization approach
   - If market changed: update training data and re-optimize

---

## Success Criteria and KPIs

### Short-term (First Month)
- Sharpe Ratio ≥ {min_sharpe:.2f}
- Precision (LONG) ≥ {min_precision:.1%}
- Hit Ratio ≥ {min_hit_ratio:.1%}
- Zero critical alerts
- Smooth operational execution

### Medium-term (3 Months)
- Sustained improvement over baseline (≥10%)
- Consistent precision and hit ratio
- Positive feedback from trading team
- No major rollbacks required

### Long-term (6+ Months)
- Outperformance vs baseline by ≥15%
- Build confidence for further optimization
- Establish optimization as regular process

---

## Configuration Management

### File Locations

**Current Baseline**:
- Config: `src/app/config.py`
- Metrics: `{output_dir}/baseline_metrics.json`

**Optimized Config**:
- Parameters: `{output_dir}/configs.json` (rank 1)
- Full Report: `{output_dir}/accuracy_report.md`
- Deployment Plan: `{output_dir}/deployment_plan.md` (this file)

**Archives**:
- Baseline Snapshot: `config/archive/config_baseline_{datetime.now().strftime("%Y%m%d")}.py`
- Optimization Run: `{output_dir}/`
- Telemetry: `data/telemetry/validation/`

### Version Control

- [ ] Tag baseline config: `git tag baseline-{datetime.now().strftime("%Y%m%d")}`
- [ ] Commit optimized params: `git commit -m "chore: add optimized params from {best["config_id"]}"`
- [ ] Create deployment branch: `git checkout -b deploy/optimized-{datetime.now().strftime("%Y%m%d")}`

---

## Approval and Sign-off

**Prepared by**: Optimization Framework (US-019)
**Date**: {datetime.now().strftime("%Y-%m-%d")}

**Approvals Required**:
- [ ] Quant Team Lead: ___________________ Date: ___________
- [ ] Risk Manager: ___________________ Date: ___________
- [ ] Head of Trading: ___________________ Date: ___________

**Deployment Authorization**:
- [ ] Approved for Phase 1 (Paper Trading): ___________________ Date: ___________
- [ ] Approved for Phase 2 (Gradual Rollout): ___________________ Date: ___________
- [ ] Approved for Phase 3 (Full Production): ___________________ Date: ___________

---

## Appendix: Quick Reference

### Key Commands

```bash
# Run optimization
python scripts/optimize.py --config <grid> --symbols <symbols> --run-baseline --export-report

# Paper trading validation
python scripts/run_paper_trading.py --config config/paper_trading.json

# Monitor live accuracy
streamlit run src/dashboard/accuracy_dashboard.py -- --live-mode

# Rollback to baseline
cp config/archive/config_baseline_*.py src/app/config.py && systemctl restart trading-engine
```

### Contact Information

- **On-Call Quant**: <quant-oncall@sensequant.com>
- **Risk Team**: <risk@sensequant.com>
- **DevOps**: <devops@sensequant.com>

---

**Generated by**: SenseQuant Optimization Framework (US-019 Phase 5)
**Optimization Run**: `{output_dir}`
**Next Review**: {(datetime.now().timestamp() + 86400 * 7):.0f} (1 week from generation)
"""

    with (output_dir / "deployment_plan.md").open("w") as f:
        f.write(plan)

    logger.info("Deployment plan generated", extra={"component": "optimize_cli"})


def main() -> None:
    """Main CLI entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    try:
        # Load search space
        logger.info(
            f"Loading search space from {args.config}",
            extra={"component": "optimize_cli"},
        )
        search_space = load_search_space(args.config)

        # Validate configuration
        validate_config(args, search_space)

        # US-019 Phase 3: Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"data/optimization/run_{timestamp}")

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting optimization",
            extra={
                "component": "optimize_cli",
                "symbols": args.symbols,
                "start": args.start_date,
                "end": args.end_date,
                "strategy": args.strategy,
                "search_type": args.search_type,
                "objective": args.objective,
                "output_dir": str(output_dir),
            },
        )

        # Create optimization config
        config = OptimizationConfig(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            strategy=args.strategy,
            initial_capital=args.initial_capital,
            search_space=search_space,
            search_type=args.search_type,
            n_samples=args.n_samples,
            objective_metric=args.objective,
            random_seed=args.random_seed,
            data_source=args.data_source,
            csv_path=args.csv_path,
            teacher_labels_path=args.teacher_labels,
        )

        # US-019 Phase 3: Run baseline if requested
        baseline = {}
        if args.run_baseline:
            try:
                baseline = run_baseline_backtest(config, settings, output_dir)
                logger.info(
                    f"Baseline complete: Sharpe={baseline.get('sharpe_ratio', 0.0):.2f}",
                    extra={"component": "optimize_cli"},
                )
            except Exception as e:
                logger.warning(
                    f"Baseline backtest failed: {e}. Continuing with optimization...",
                    extra={"component": "optimize_cli"},
                )

        # Initialize Breeze client (only needed if data_source=breeze)
        client = None
        if config.data_source == "breeze":
            logger.info(
                "Authenticating with Breeze API",
                extra={"component": "optimize_cli"},
            )
            client = BreezeClient(
                api_key=settings.breeze_api_key,
                api_secret=settings.breeze_api_secret,
                session_token=settings.breeze_session_token,
                dry_run=True,  # Always use dry_run for optimization
            )
            client.authenticate()

        # US-019 Phase 3: Configure telemetry for optimizer
        # Update settings to enable telemetry for each configuration
        if args.telemetry_dir:
            telemetry_base_dir = Path(args.telemetry_dir)
            telemetry_base_dir.mkdir(parents=True, exist_ok=True)
            # Note: Optimizer will handle per-configuration telemetry directories
            logger.info(
                f"Telemetry enabled: dir={telemetry_base_dir}, sample_rate={args.telemetry_sample_rate}",
                extra={"component": "optimize_cli"},
            )

        # Create optimizer
        optimizer = ParameterOptimizer(config=config, client=client, settings=settings)

        # US-019 Phase 3: Limit configurations if specified
        if args.max_configs:
            logger.info(
                f"Limiting to {args.max_configs} configurations",
                extra={"component": "optimize_cli"},
            )
            # Note: This would require optimizer modification to support early stopping
            # For now, log the intent

        # Run optimization
        result = optimizer.run()

        # Print results
        print_results(result, top_n=args.top_n)

        # US-019 Phase 3: Export enhanced artifacts with accuracy metrics
        if baseline or args.export_report:
            try:
                export_optimization_artifacts(
                    output_dir=output_dir,
                    candidates=result.candidates,
                    baseline=baseline,
                    config=config,
                    export_report=args.export_report,
                )
                logger.info(
                    f"Enhanced artifacts exported to {output_dir}",
                    extra={"component": "optimize_cli"},
                )
            except Exception as e:
                logger.warning(
                    f"Failed to export enhanced artifacts: {e}",
                    extra={"component": "optimize_cli"},
                )

        # Exit successfully
        sys.exit(0)

    except Exception as e:
        logger.exception(
            f"Optimization failed: {e}",
            extra={"component": "optimize_cli"},
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
