"""Parameter optimization engine for systematic strategy tuning."""

from __future__ import annotations

import itertools
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.adapters.breeze_client import BreezeClient
from src.app.config import Settings
from src.app.config import settings as default_settings
from src.domain.types import (
    BacktestConfig,
    OptimizationCandidate,
    OptimizationConfig,
    OptimizationResult,
)
from src.services.backtester import Backtester


class ParameterOptimizer:
    """Parameter optimization engine using grid or random search.

    Systematically evaluates parameter combinations by running backtests
    and ranking results by an objective metric (Sharpe ratio, CAGR, etc.).

    Attributes:
        config: Optimization configuration
        client: Breeze API client (optional, for data fetching)
        settings: Application settings
    """

    def __init__(
        self,
        config: OptimizationConfig,
        client: BreezeClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize ParameterOptimizer.

        Args:
            config: Optimization configuration
            client: Breeze API client (optional)
            settings: Application settings (optional, uses default if None)
        """
        self.config = config
        self.client = client
        self.settings = settings if settings is not None else default_settings

        # Set random seed for reproducibility
        np.random.seed(config.random_seed)

        logger.info(
            "Initialized ParameterOptimizer",
            extra={
                "component": "optimizer",
                "symbols": config.symbols,
                "strategy": config.strategy,
                "search_type": config.search_type,
                "objective": config.objective_metric,
            },
        )

    def generate_candidates(self) -> list[dict[str, Any]]:
        """Generate parameter combinations from search space.

        Returns:
            List of parameter dictionaries

        Raises:
            ValueError: If search space is invalid or empty
        """
        if not self.config.search_space:
            raise ValueError("Search space cannot be empty")

        if self.config.search_type == "grid":
            return self._generate_grid_candidates()
        elif self.config.search_type == "random":
            return self._generate_random_candidates()
        else:
            raise ValueError(f"Unknown search type: {self.config.search_type}")

    def _generate_grid_candidates(self) -> list[dict[str, Any]]:
        """Generate all combinations for grid search (Cartesian product).

        Returns:
            List of parameter dictionaries
        """
        if self.config.search_space is None:
            return []

        # Flatten nested search space into list of (key, values) pairs
        param_names = []
        param_values = []

        for key, values in self.config.search_space.items():
            if isinstance(values, dict):
                # Nested parameters (e.g., strategy.swing.sma_fast)
                for subkey, subvalues in values.items():
                    param_names.append(f"{key}.{subkey}")
                    param_values.append(subvalues if isinstance(subvalues, list) else [subvalues])
            else:
                # Top-level parameters
                param_names.append(key)
                param_values.append(values if isinstance(values, list) else [values])

        # Generate Cartesian product
        combinations = list(itertools.product(*param_values))

        candidates = []
        for combo in combinations:
            params: dict[str, Any] = {}
            for name, value in zip(param_names, combo, strict=False):
                # Reconstruct nested structure
                if "." in name:
                    parts = name.split(".")
                    if parts[0] not in params:
                        params[parts[0]] = {}
                    params[parts[0]][parts[1]] = value
                else:
                    params[name] = value
            candidates.append(params)

        logger.info(
            f"Generated {len(candidates)} grid search candidates",
            extra={"component": "optimizer", "count": len(candidates)},
        )
        return candidates

    def _generate_random_candidates(self) -> list[dict[str, Any]]:
        """Generate random samples from search space.

        Returns:
            List of parameter dictionaries
        """
        if self.config.search_space is None:
            return []

        candidates = []

        for _ in range(self.config.n_samples):
            params: dict[str, Any] = {}
            for key, values in self.config.search_space.items():
                if isinstance(values, dict):
                    # Nested parameters
                    params[key] = {}
                    for subkey, subvalues in values.items():
                        if isinstance(subvalues, list):
                            params[key][subkey] = np.random.choice(subvalues)
                        else:
                            params[key][subkey] = subvalues
                else:
                    # Top-level parameters
                    if isinstance(values, list):
                        params[key] = np.random.choice(values)
                    else:
                        params[key] = values
            candidates.append(params)

        logger.info(
            f"Generated {len(candidates)} random search candidates",
            extra={"component": "optimizer", "count": len(candidates)},
        )
        return candidates

    def evaluate_candidate(
        self, candidate_id: int, parameters: dict[str, Any]
    ) -> OptimizationCandidate:
        """Evaluate single parameter combination via backtest.

        Args:
            candidate_id: Unique candidate identifier
            parameters: Parameter dictionary

        Returns:
            OptimizationCandidate with backtest results and score
        """
        start_time = time.time()

        try:
            # Create backtest config with candidate parameters
            backtest_config = self._create_backtest_config(parameters)

            # Run backtest
            logger.debug(
                f"Evaluating candidate {candidate_id}",
                extra={
                    "component": "optimizer",
                    "candidate_id": candidate_id,
                    "params": parameters,
                },
            )

            backtester = Backtester(
                config=backtest_config, client=self.client, settings=self.settings
            )
            result = backtester.run()

            # US-019 Phase 2: Use composite score if objective is "composite" and accuracy metrics available
            if self.config.objective_metric == "composite" and result.accuracy_metrics:
                score = self.compute_composite_score(result.metrics, result.accuracy_metrics)
                logger.info(
                    f"Candidate {candidate_id} composite score: {score:.3f}",
                    extra={
                        "component": "optimizer",
                        "sharpe": result.metrics.get("sharpe_ratio", 0.0),
                        "precision": result.accuracy_metrics.precision.get("LONG", 0.0),
                        "hit_ratio": result.accuracy_metrics.hit_ratio,
                    },
                )
            else:
                # Extract objective metric (traditional approach)
                score = self._extract_score(result.metrics)

            elapsed = time.time() - start_time

            logger.info(
                f"Candidate {candidate_id} complete",
                extra={
                    "component": "optimizer",
                    "candidate_id": candidate_id,
                    "score": score,
                    "elapsed": f"{elapsed:.2f}s",
                },
            )

            return OptimizationCandidate(
                candidate_id=candidate_id,
                parameters=parameters,
                backtest_result=result,
                score=score,
                error=None,
                elapsed_time=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)

            logger.error(
                f"Candidate {candidate_id} failed: {error_msg}",
                extra={
                    "component": "optimizer",
                    "candidate_id": candidate_id,
                    "error": error_msg,
                },
            )

            return OptimizationCandidate(
                candidate_id=candidate_id,
                parameters=parameters,
                backtest_result=None,
                score=None,
                error=error_msg,
                elapsed_time=elapsed,
            )

    def _create_backtest_config(self, parameters: dict[str, Any]) -> BacktestConfig:
        """Create BacktestConfig from optimization parameters.

        Args:
            parameters: Parameter dictionary

        Returns:
            BacktestConfig with parameters applied

        Note:
            Currently parameters are stored in metadata; future enhancement
            will apply them directly to strategy/risk settings.
        """
        # TODO: Apply strategy/risk parameters to settings
        # For now, create base config and store params in metadata
        return BacktestConfig(
            symbols=self.config.symbols,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            strategy=self.config.strategy,
            initial_capital=self.config.initial_capital,
            data_source=self.config.data_source,
            random_seed=self.config.random_seed,
            csv_path=self.config.csv_path,
            teacher_labels_path=self.config.teacher_labels_path,
        )

    def _extract_score(self, metrics: dict[str, float]) -> float:
        """Extract objective metric from backtest results.

        Args:
            metrics: Backtest metrics dictionary

        Returns:
            Objective metric value

        Raises:
            ValueError: If objective metric not found in results
        """
        if self.config.objective_metric not in metrics:
            raise ValueError(
                f"Objective metric '{self.config.objective_metric}' not found in backtest metrics"
            )

        return float(metrics[self.config.objective_metric])

    def compute_composite_score(
        self,
        financial_metrics: dict[str, float],
        accuracy_metrics: Any | None,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Compute composite score combining financial and accuracy metrics (US-019 Phase 2).

        Args:
            financial_metrics: Financial metrics (sharpe_ratio, total_return, etc.)
            accuracy_metrics: Accuracy metrics from AccuracyAnalyzer (or None)
            weights: Scoring weights (default: balanced across metrics)

        Returns:
            Composite score in [0, 1] range

        Example:
            >>> optimizer.compute_composite_score(
            ...     {"sharpe_ratio": 1.8, "total_return": 0.12},
            ...     accuracy_metrics,
            ...     {"sharpe_ratio": 0.4, "precision_long": 0.3, "hit_ratio": 0.2, "win_rate": 0.1}
            ... )
            0.617
        """
        if weights is None:
            weights = {
                "sharpe_ratio": 0.40,  # Risk-adjusted return
                "precision_long": 0.30,  # Prediction quality
                "hit_ratio": 0.20,  # Overall accuracy
                "win_rate": 0.10,  # Profitability
            }

        # Normalize Sharpe to [0, 1] (cap at 3.0 for exceptional performance)
        sharpe = financial_metrics.get("sharpe_ratio", 0.0)
        norm_sharpe = min(max(sharpe / 3.0, 0.0), 1.0)

        # Accuracy metrics (already in [0, 1] range)
        norm_precision = 0.0
        norm_hit_ratio = 0.0
        norm_win_rate = 0.0

        if accuracy_metrics:
            norm_precision = accuracy_metrics.precision.get("LONG", 0.0)
            norm_hit_ratio = accuracy_metrics.hit_ratio
            norm_win_rate = accuracy_metrics.win_rate

        # Weighted composite
        score = (
            weights.get("sharpe_ratio", 0.0) * norm_sharpe
            + weights.get("precision_long", 0.0) * norm_precision
            + weights.get("hit_ratio", 0.0) * norm_hit_ratio
            + weights.get("win_rate", 0.0) * norm_win_rate
        )

        return score

    def rank_candidates(
        self, candidates: list[OptimizationCandidate]
    ) -> list[OptimizationCandidate]:
        """Sort candidates by score (descending).

        Args:
            candidates: List of evaluated candidates

        Returns:
            Sorted list (best to worst)
        """
        # Filter out failed candidates
        successful = [c for c in candidates if c.score is not None]

        # Sort by score descending (cast to float for type safety)
        ranked = sorted(
            successful, key=lambda c: float(c.score) if c.score is not None else 0.0, reverse=True
        )

        logger.info(
            f"Ranked {len(ranked)} successful candidates",
            extra={
                "component": "optimizer",
                "successful": len(ranked),
                "failed": len(candidates) - len(ranked),
            },
        )

        return ranked

    def run(self) -> OptimizationResult:
        """Execute full optimization workflow.

        Returns:
            OptimizationResult with ranked candidates and metadata

        Raises:
            ValueError: If optimization fails completely
        """
        start_time = time.time()

        logger.info(
            "Starting optimization",
            extra={
                "component": "optimizer",
                "symbols": self.config.symbols,
                "dates": f"{self.config.start_date} to {self.config.end_date}",
                "strategy": self.config.strategy,
            },
        )

        # Generate candidates
        param_combinations = self.generate_candidates()

        # Evaluate each candidate
        candidates = []
        for idx, params in enumerate(param_combinations):
            candidate = self.evaluate_candidate(candidate_id=idx, parameters=params)
            candidates.append(candidate)

        # Rank results
        ranked_candidates = self.rank_candidates(candidates)

        # Calculate statistics
        total_time = time.time() - start_time
        successful_count = len([c for c in candidates if c.error is None])
        failed_count = len([c for c in candidates if c.error is not None])

        # Get best candidate
        best_candidate = ranked_candidates[0] if ranked_candidates else None

        # Get git hash for reproducibility
        git_hash = self._get_git_hash()

        # Create metadata
        metadata = {
            "run_date": datetime.now().isoformat(),
            "git_hash": git_hash,
            "symbols": self.config.symbols,
            "strategy": self.config.strategy,
            "search_type": self.config.search_type,
            "objective_metric": self.config.objective_metric,
            "date_range": f"{self.config.start_date} to {self.config.end_date}",
            "random_seed": self.config.random_seed,
        }

        # Save results
        output_dir = self._create_output_dir()
        summary_path, ranked_path, best_path = self._save_results(
            candidates=ranked_candidates,
            failed_candidates=[c for c in candidates if c.error is not None],
            best_candidate=best_candidate,
            metadata=metadata,
            output_dir=output_dir,
        )

        logger.info(
            f"Optimization complete: {successful_count}/{len(candidates)} successful",
            extra={
                "component": "optimizer",
                "total_time": f"{total_time:.2f}s",
                "best_score": best_candidate.score if best_candidate else None,
            },
        )

        return OptimizationResult(
            config=self.config,
            candidates=ranked_candidates,
            best_candidate=best_candidate,
            total_candidates=len(candidates),
            successful_candidates=successful_count,
            failed_candidates=failed_count,
            total_time=total_time,
            metadata=metadata,
            summary_path=summary_path,
            ranked_results_path=ranked_path,
            best_config_path=best_path,
        )

    def _create_output_dir(self) -> Path:
        """Create timestamped output directory.

        Returns:
            Path to output directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data") / "optimization" / f"opt_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"Created output directory: {output_dir}",
            extra={"component": "optimizer", "path": str(output_dir)},
        )

        return output_dir

    def _save_results(
        self,
        candidates: list[OptimizationCandidate],
        failed_candidates: list[OptimizationCandidate],
        best_candidate: OptimizationCandidate | None,
        metadata: dict[str, Any],
        output_dir: Path,
    ) -> tuple[str, str, str]:
        """Persist optimization results to disk.

        Args:
            candidates: Ranked successful candidates
            failed_candidates: Failed candidates
            best_candidate: Top-ranked candidate
            metadata: Optimization metadata
            output_dir: Output directory

        Returns:
            Tuple of (summary_path, ranked_results_path, best_config_path)
        """
        # Save summary JSON
        summary_path = output_dir / "summary.json"
        summary = {
            "metadata": metadata,
            "total_candidates": len(candidates) + len(failed_candidates),
            "successful_candidates": len(candidates),
            "failed_candidates": len(failed_candidates),
            "search_space": self.config.search_space,
            "objective_metric": self.config.objective_metric,
            "best_score": best_candidate.score if best_candidate else None,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save ranked results CSV
        ranked_path = output_dir / "ranked_results.csv"
        if candidates:
            rows = []
            for rank, candidate in enumerate(candidates, start=1):
                row = {
                    "rank": rank,
                    "candidate_id": candidate.candidate_id,
                    "score": candidate.score,
                    "elapsed_time": candidate.elapsed_time,
                    **self._flatten_params(candidate.parameters),
                }
                # Add key metrics
                if candidate.backtest_result:
                    for key in [
                        "total_return_pct",
                        "cagr_pct",
                        "sharpe_ratio",
                        "max_drawdown_pct",
                        "win_rate_pct",
                    ]:
                        row[key] = candidate.backtest_result.metrics.get(key, None)
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(ranked_path, index=False)

        # Save failed candidates CSV
        if failed_candidates:
            failed_path = output_dir / "failed_candidates.csv"
            failed_rows = []
            for candidate in failed_candidates:
                row = {
                    "candidate_id": candidate.candidate_id,
                    "error": candidate.error,
                    "elapsed_time": candidate.elapsed_time,
                    **self._flatten_params(candidate.parameters),
                }
                failed_rows.append(row)
            df_failed = pd.DataFrame(failed_rows)
            df_failed.to_csv(failed_path, index=False)

        # Save best config JSON
        best_path = output_dir / "best_config.json"
        if best_candidate:
            # Convert numpy types to Python native types for JSON serialization
            def convert_types(obj: Any) -> Any:
                """Convert numpy types to Python native types."""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj

            best_config = {
                "parameters": convert_types(best_candidate.parameters),
                "score": float(best_candidate.score) if best_candidate.score is not None else None,
                "metrics": convert_types(
                    best_candidate.backtest_result.metrics if best_candidate.backtest_result else {}
                ),
                "metadata": metadata,
            }
            with open(best_path, "w") as f:
                json.dump(best_config, f, indent=2)

        logger.info(
            "Results saved",
            extra={
                "component": "optimizer",
                "summary": str(summary_path),
                "ranked": str(ranked_path),
                "best": str(best_path),
            },
        )

        return str(summary_path), str(ranked_path), str(best_path)

    def _flatten_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested parameter dictionary for CSV export.

        Args:
            params: Nested parameter dictionary

        Returns:
            Flattened dictionary with dot-notation keys
        """
        flat = {}
        for key, value in params.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat[f"{key}.{subkey}"] = subvalue
            else:
                flat[key] = value
        return flat

    def _get_git_hash(self) -> str:
        """Get current git commit hash for reproducibility.

        Returns:
            Git commit hash or 'unknown' if not in git repo
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return "unknown"
