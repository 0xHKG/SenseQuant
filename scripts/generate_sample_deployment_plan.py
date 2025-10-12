#!/usr/bin/env python3
"""Generate deployment plan for sample optimization run."""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from domain.types import OptimizationConfig


def main():
    """Generate deployment plan for sample run."""
    # Sample directory
    sample_dir = Path("data/optimization/sample_run")

    # Load artifacts
    with open(sample_dir / "baseline_metrics.json") as f:
        baseline = json.load(f)

    with open(sample_dir / "configs.json") as f:
        configs = json.load(f)

    # Create mock optimization config
    opt_config = OptimizationConfig(
        symbols=["RELIANCE", "TCS"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        strategy="intraday",
        initial_capital=1000000.0,
        search_space={},
        search_type="grid",
        n_samples=5,
        objective_metric="composite",
        random_seed=42,
        data_source="csv",
        csv_path=None,
        teacher_labels_path=None,
    )

    # Import generate_deployment_plan from optimize.py
    spec = __import__("importlib.util").util.spec_from_file_location(
        "optimize", Path(__file__).parent / "optimize.py"
    )
    optimize_module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(optimize_module)

    # Generate deployment plan
    optimize_module.generate_deployment_plan(sample_dir, configs, baseline, opt_config)

    print(f"✅ Deployment plan generated: {sample_dir / 'deployment_plan.md'}")


if __name__ == "__main__":
    main()
