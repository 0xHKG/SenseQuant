"""Unit tests for ParameterOptimizer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.types import (
    BacktestResult,
    OptimizationCandidate,
    OptimizationConfig,
)
from src.services.backtester import Backtester
from src.services.optimizer import ParameterOptimizer


@pytest.fixture
def sample_search_space() -> dict:
    """Sample search space for testing."""
    return {
        "strategy": {
            "sma_fast": [5, 10, 15],
            "sma_slow": [30, 40],
        },
        "risk": {
            "risk_per_trade_pct": [1.0, 1.5],
        },
    }


@pytest.fixture
def optimization_config(sample_search_space: dict) -> OptimizationConfig:
    """Create sample optimization config."""
    return OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        strategy="swing",
        search_space=sample_search_space,
        search_type="grid",
        objective_metric="sharpe_ratio",
        random_seed=42,
    )


def test_optimizer_initialization(optimization_config: OptimizationConfig) -> None:
    """Test ParameterOptimizer initialization."""
    optimizer = ParameterOptimizer(config=optimization_config)

    assert optimizer.config == optimization_config
    assert optimizer.client is None
    assert optimizer.settings is not None


def test_grid_search_candidate_generation(optimization_config: OptimizationConfig) -> None:
    """Test grid search generates all combinations."""
    optimizer = ParameterOptimizer(config=optimization_config)
    candidates = optimizer.generate_candidates()

    # Should generate 3 * 2 * 2 = 12 combinations
    assert len(candidates) == 12

    # Verify all combinations are unique
    unique_combos = {str(c) for c in candidates}
    assert len(unique_combos) == 12

    # Verify structure
    for candidate in candidates:
        assert "strategy" in candidate
        assert "risk" in candidate
        assert "sma_fast" in candidate["strategy"]
        assert "sma_slow" in candidate["strategy"]
        assert "risk_per_trade_pct" in candidate["risk"]


def test_grid_search_empty_space() -> None:
    """Test grid search with empty search space."""
    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        search_space={},
    )

    optimizer = ParameterOptimizer(config=config)

    with pytest.raises(ValueError, match="Search space cannot be empty"):
        optimizer.generate_candidates()


def test_random_search_candidate_generation() -> None:
    """Test random search samples correct number of candidates."""
    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        search_space={
            "strategy": {
                "sma_fast": [5, 10, 15, 20],
                "sma_slow": [30, 40, 50],
            }
        },
        search_type="random",
        n_samples=10,
        random_seed=42,
    )

    optimizer = ParameterOptimizer(config=config)
    candidates = optimizer.generate_candidates()

    assert len(candidates) == 10

    # Verify structure
    for candidate in candidates:
        assert "strategy" in candidate
        assert "sma_fast" in candidate["strategy"]
        assert "sma_slow" in candidate["strategy"]
        assert candidate["strategy"]["sma_fast"] in [5, 10, 15, 20]
        assert candidate["strategy"]["sma_slow"] in [30, 40, 50]


def test_random_search_determinism() -> None:
    """Test random search is deterministic with same seed."""
    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        search_space={
            "strategy": {
                "sma_fast": [5, 10, 15, 20],
            }
        },
        search_type="random",
        n_samples=5,
        random_seed=42,
    )

    optimizer1 = ParameterOptimizer(config=config)
    candidates1 = optimizer1.generate_candidates()

    # Reset seed and generate again
    np.random.seed(42)
    optimizer2 = ParameterOptimizer(config=config)
    candidates2 = optimizer2.generate_candidates()

    # Should generate identical candidates
    assert candidates1 == candidates2


def test_extract_score_valid_metric(optimization_config: OptimizationConfig) -> None:
    """Test extracting objective metric from backtest results."""
    optimizer = ParameterOptimizer(config=optimization_config)

    metrics = {
        "sharpe_ratio": 1.85,
        "cagr_pct": 25.3,
        "total_return_pct": 42.5,
    }

    score = optimizer._extract_score(metrics)
    assert score == 1.85


def test_extract_score_missing_metric(optimization_config: OptimizationConfig) -> None:
    """Test error when objective metric not in results."""
    optimizer = ParameterOptimizer(config=optimization_config)

    metrics = {
        "cagr_pct": 25.3,
        "total_return_pct": 42.5,
    }

    with pytest.raises(ValueError, match="Objective metric 'sharpe_ratio' not found"):
        optimizer._extract_score(metrics)


def test_rank_candidates() -> None:
    """Test candidate ranking by score."""
    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        objective_metric="sharpe_ratio",
    )

    optimizer = ParameterOptimizer(config=config)

    candidates = [
        OptimizationCandidate(candidate_id=0, parameters={"a": 1}, score=1.5, elapsed_time=10.0),
        OptimizationCandidate(candidate_id=1, parameters={"a": 2}, score=2.3, elapsed_time=12.0),
        OptimizationCandidate(candidate_id=2, parameters={"a": 3}, score=0.8, elapsed_time=11.0),
        OptimizationCandidate(
            candidate_id=3, parameters={"a": 4}, score=None, error="Failed", elapsed_time=5.0
        ),
    ]

    ranked = optimizer.rank_candidates(candidates)

    # Should have 3 successful candidates (id=3 failed)
    assert len(ranked) == 3

    # Should be sorted by score descending
    assert ranked[0].score == 2.3
    assert ranked[1].score == 1.5
    assert ranked[2].score == 0.8


def test_rank_candidates_all_failed() -> None:
    """Test ranking when all candidates failed."""
    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    optimizer = ParameterOptimizer(config=config)

    candidates = [
        OptimizationCandidate(
            candidate_id=0, parameters={"a": 1}, score=None, error="Failed", elapsed_time=5.0
        ),
        OptimizationCandidate(
            candidate_id=1, parameters={"a": 2}, score=None, error="Failed", elapsed_time=6.0
        ),
    ]

    ranked = optimizer.rank_candidates(candidates)

    assert len(ranked) == 0


def test_flatten_params() -> None:
    """Test parameter flattening for CSV export."""
    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    optimizer = ParameterOptimizer(config=config)

    params = {
        "strategy": {
            "sma_fast": 10,
            "sma_slow": 40,
        },
        "risk": {
            "risk_per_trade_pct": 1.5,
        },
        "simple_param": 100,
    }

    flat = optimizer._flatten_params(params)

    assert flat == {
        "strategy.sma_fast": 10,
        "strategy.sma_slow": 40,
        "risk.risk_per_trade_pct": 1.5,
        "simple_param": 100,
    }


def test_evaluate_candidate_success(optimization_config: OptimizationConfig) -> None:
    """Test successful candidate evaluation."""
    optimizer = ParameterOptimizer(config=optimization_config)

    # Mock backtest result
    mock_result = BacktestResult(
        config=MagicMock(),
        metrics={"sharpe_ratio": 1.85, "cagr_pct": 25.0},
        equity_curve=MagicMock(),
        trades=MagicMock(),
        metadata={},
        summary_path="",
        equity_path="",
        trades_path="",
    )

    with patch.object(Backtester, "run", return_value=mock_result):
        candidate = optimizer.evaluate_candidate(
            candidate_id=0, parameters={"strategy": {"sma_fast": 10}}
        )

        assert candidate.candidate_id == 0
        assert candidate.score == 1.85
        assert candidate.error is None
        assert candidate.backtest_result is not None


def test_evaluate_candidate_failure(optimization_config: OptimizationConfig) -> None:
    """Test candidate evaluation with backtest failure."""
    optimizer = ParameterOptimizer(config=optimization_config)

    with patch.object(Backtester, "run", side_effect=RuntimeError("Backtest failed")):
        candidate = optimizer.evaluate_candidate(
            candidate_id=0, parameters={"strategy": {"sma_fast": 10}}
        )

        assert candidate.candidate_id == 0
        assert candidate.score is None
        assert candidate.error == "Backtest failed"
        assert candidate.backtest_result is None


def test_create_output_dir(optimization_config: OptimizationConfig, tmp_path) -> None:
    """Test output directory creation."""
    optimizer = ParameterOptimizer(config=optimization_config)

    # The method creates a timestamped subdirectory
    output_dir = optimizer._create_output_dir()

    # Verify it exists and is in data/optimization
    assert output_dir.exists()
    assert "optimization" in str(output_dir)
    assert output_dir.name.startswith("opt_")


def test_get_git_hash(optimization_config: OptimizationConfig) -> None:
    """Test git hash retrieval."""
    optimizer = ParameterOptimizer(config=optimization_config)

    git_hash = optimizer._get_git_hash()

    # Should return a hash or 'unknown'
    assert isinstance(git_hash, str)
    assert len(git_hash) > 0


def test_save_results(optimization_config: OptimizationConfig, tmp_path) -> None:
    """Test results persistence."""
    optimizer = ParameterOptimizer(config=optimization_config)

    mock_backtest_result = BacktestResult(
        config=MagicMock(),
        metrics={
            "sharpe_ratio": 1.85,
            "cagr_pct": 25.0,
            "total_return_pct": 42.0,
            "max_drawdown_pct": -15.0,
            "win_rate_pct": 65.0,
        },
        equity_curve=MagicMock(),
        trades=MagicMock(),
        metadata={},
        summary_path="",
        equity_path="",
        trades_path="",
    )

    candidates = [
        OptimizationCandidate(
            candidate_id=0,
            parameters={"strategy": {"sma_fast": 10}},
            backtest_result=mock_backtest_result,
            score=1.85,
            elapsed_time=10.0,
        ),
    ]

    failed = [
        OptimizationCandidate(
            candidate_id=1,
            parameters={"strategy": {"sma_fast": 15}},
            error="Failed",
            elapsed_time=5.0,
        ),
    ]

    metadata = {
        "run_date": "2024-01-01",
        "git_hash": "abc123",
    }

    output_dir = tmp_path / "test_output"
    output_dir.mkdir()

    summary_path, ranked_path, best_path = optimizer._save_results(
        candidates=candidates,
        failed_candidates=failed,
        best_candidate=candidates[0],
        metadata=metadata,
        output_dir=output_dir,
    )

    # Verify files created
    assert Path(summary_path).exists()
    assert Path(ranked_path).exists()
    assert Path(best_path).exists()
    assert (output_dir / "failed_candidates.csv").exists()


def test_single_parameter_optimization() -> None:
    """Test optimization with single parameter."""
    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        search_space={"simple_param": [1, 2, 3]},
    )

    optimizer = ParameterOptimizer(config=config)
    candidates = optimizer.generate_candidates()

    assert len(candidates) == 3
    assert candidates[0] == {"simple_param": 1}
    assert candidates[1] == {"simple_param": 2}
    assert candidates[2] == {"simple_param": 3}


def test_mixed_nested_and_flat_parameters() -> None:
    """Test search space with both nested and flat parameters."""
    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        search_space={
            "strategy": {
                "sma_fast": [5, 10],
            },
            "simple": [100, 200],
        },
    )

    optimizer = ParameterOptimizer(config=config)
    candidates = optimizer.generate_candidates()

    # 2 * 2 = 4 combinations
    assert len(candidates) == 4

    for candidate in candidates:
        assert "strategy" in candidate
        assert "simple" in candidate
        assert "sma_fast" in candidate["strategy"]
