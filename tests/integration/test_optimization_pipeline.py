"""Integration tests for full optimization pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

from src.domain.types import Bar, OptimizationConfig
from src.services.optimizer import ParameterOptimizer


@pytest.fixture
def mock_breeze_client() -> MagicMock:
    """Create mock Breeze client."""
    client = MagicMock()
    client.authenticate = MagicMock()
    client.historical_bars = MagicMock()
    return client


@pytest.fixture
def sample_bars() -> list[Bar]:
    """Generate sample bars for backtesting."""
    ist = pytz.timezone("Asia/Kolkata")
    base_date = pd.Timestamp("2024-01-01", tz=ist)
    bars = []

    for i in range(120):
        ts = base_date + pd.Timedelta(days=i)
        if i < 80:
            close = 150.0 - (i * 0.5)
        elif i < 100:
            close = 110.0 + ((i - 79) * 3.0)
        else:
            close = 170.0 + ((i - 99) * 0.5)

        bars.append(
            Bar(
                ts=ts,
                open=close - 0.5,
                high=close + 2.0,
                low=close - 2.0,
                close=close,
                volume=100000,
            )
        )

    return bars


def test_grid_search_optimization_pipeline(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test full grid search optimization pipeline."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    # Small search space for fast testing (2x2 = 4 candidates)
    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        search_space={
            "strategy": {
                "sma_fast": [5, 10],
                "sma_slow": [30, 40],
            }
        },
        search_type="grid",
        objective_metric="sharpe_ratio",
        random_seed=42,
    )

    # Patch data directory to use tmp_path
    with patch("src.services.optimizer.Path") as mock_path:
        opt_dir = tmp_path / "optimization"
        opt_dir.mkdir(parents=True, exist_ok=True)
        mock_path.return_value = opt_dir

        from src.app.config import settings

        optimizer = ParameterOptimizer(config=config, client=mock_breeze_client, settings=settings)

        # Generate candidates
        candidates = optimizer.generate_candidates()
        assert len(candidates) == 4

        # Run optimization
        result = optimizer.run()

        # Verify results structure
        assert result.total_candidates == 4
        assert result.successful_candidates >= 0
        assert result.failed_candidates >= 0
        assert result.successful_candidates + result.failed_candidates == 4

        # Verify metadata
        assert "run_date" in result.metadata
        assert "git_hash" in result.metadata
        assert result.metadata["search_type"] == "grid"
        assert result.metadata["objective_metric"] == "sharpe_ratio"

        # Verify best candidate if any succeeded
        if result.successful_candidates > 0:
            assert result.best_candidate is not None
            assert result.best_candidate.score is not None
            assert "strategy" in result.best_candidate.parameters


def test_random_search_optimization_pipeline(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test random search optimization pipeline."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        search_space={
            "strategy": {
                "sma_fast": [5, 10, 15],
                "sma_slow": [30, 40, 50],
            }
        },
        search_type="random",
        n_samples=5,
        objective_metric="sharpe_ratio",
        random_seed=42,
    )

    with patch("src.services.optimizer.Path") as mock_path:
        opt_dir = tmp_path / "optimization"
        opt_dir.mkdir(parents=True, exist_ok=True)
        mock_path.return_value = opt_dir

        from src.app.config import settings

        optimizer = ParameterOptimizer(config=config, client=mock_breeze_client, settings=settings)

        # Generate candidates
        candidates = optimizer.generate_candidates()
        assert len(candidates) == 5

        # Run optimization
        result = optimizer.run()

        assert result.total_candidates == 5
        assert result.metadata["search_type"] == "random"


def test_optimization_determinism(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test that optimization is deterministic with same seed."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        search_space={
            "strategy": {
                "sma_fast": [5, 10],
                "sma_slow": [30, 40],
            }
        },
        search_type="random",
        n_samples=3,
        objective_metric="sharpe_ratio",
        random_seed=42,
    )

    from src.app.config import settings

    # Run 1
    with patch("src.services.optimizer.Path") as mock_path:
        opt_dir1 = tmp_path / "opt1"
        opt_dir1.mkdir(parents=True, exist_ok=True)
        mock_path.return_value = opt_dir1

        optimizer1 = ParameterOptimizer(config=config, client=mock_breeze_client, settings=settings)
        result1 = optimizer1.run()

    # Run 2
    with patch("src.services.optimizer.Path") as mock_path:
        opt_dir2 = tmp_path / "opt2"
        opt_dir2.mkdir(parents=True, exist_ok=True)
        mock_path.return_value = opt_dir2

        optimizer2 = ParameterOptimizer(config=config, client=mock_breeze_client, settings=settings)
        result2 = optimizer2.run()

    # Should generate same candidates (random search with same seed)
    assert len(result1.candidates) == len(result2.candidates)

    # Scores should be identical (deterministic backtest)
    if result1.successful_candidates > 0 and result2.successful_candidates > 0:
        for c1, c2 in zip(result1.candidates, result2.candidates, strict=False):
            assert c1.parameters == c2.parameters
            assert c1.score == c2.score


def test_optimization_fault_tolerance(
    mock_breeze_client: MagicMock,
    tmp_path: Path,
) -> None:
    """Test that optimization continues when some candidates fail."""
    # Create bars that will cause failures for some parameters
    ist = pytz.timezone("Asia/Kolkata")
    bars = [
        Bar(
            ts=pd.Timestamp("2024-01-01", tz=ist) + pd.Timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10000,
        )
        for i in range(60)
    ]

    mock_breeze_client.historical_bars.return_value = bars

    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-03-01",
        strategy="swing",
        search_space={
            "strategy": {
                "sma_fast": [5, 10],
            }
        },
        search_type="grid",
        objective_metric="sharpe_ratio",
        random_seed=42,
    )

    with patch("src.services.optimizer.Path") as mock_path:
        opt_dir = tmp_path / "optimization"
        opt_dir.mkdir(parents=True, exist_ok=True)
        mock_path.return_value = opt_dir

        from src.app.config import settings

        optimizer = ParameterOptimizer(config=config, client=mock_breeze_client, settings=settings)

        # Run optimization (should not crash even if some fail)
        result = optimizer.run()

        # Should complete
        assert result.total_candidates == 2
        assert result.successful_candidates + result.failed_candidates == 2


def test_multi_symbol_optimization(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test optimization with multiple symbols."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    config = OptimizationConfig(
        symbols=["TEST1", "TEST2"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        search_space={
            "strategy": {
                "sma_fast": [5, 10],
            }
        },
        search_type="grid",
        objective_metric="sharpe_ratio",
        random_seed=42,
    )

    with patch("src.services.optimizer.Path") as mock_path:
        opt_dir = tmp_path / "optimization"
        opt_dir.mkdir(parents=True, exist_ok=True)
        mock_path.return_value = opt_dir

        from src.app.config import settings

        optimizer = ParameterOptimizer(config=config, client=mock_breeze_client, settings=settings)

        result = optimizer.run()

        assert result.config.symbols == ["TEST1", "TEST2"]
        assert result.total_candidates == 2


def test_optimization_artifact_completeness(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test that all optimization artifacts are created."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        search_space={
            "strategy": {
                "sma_fast": [5, 10],
            }
        },
        search_type="grid",
        objective_metric="sharpe_ratio",
        random_seed=42,
    )

    opt_dir = tmp_path / "optimization" / "test_run"
    opt_dir.mkdir(parents=True, exist_ok=True)

    with patch("src.services.optimizer.Path") as mock_path:
        mock_path.return_value = opt_dir

        from src.app.config import settings

        optimizer = ParameterOptimizer(config=config, client=mock_breeze_client, settings=settings)

        result = optimizer.run()

        # Verify paths are set
        assert result.summary_path is not None
        assert result.ranked_results_path is not None
        assert result.best_config_path is not None


def test_different_objective_metrics(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test optimization with different objective metrics."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    objectives = ["sharpe_ratio", "cagr_pct", "total_return_pct"]

    for objective in objectives:
        config = OptimizationConfig(
            symbols=["TEST"],
            start_date="2024-01-01",
            end_date="2024-04-30",
            strategy="swing",
            search_space={
                "strategy": {
                    "sma_fast": [5, 10],
                }
            },
            search_type="grid",
            objective_metric=objective,
            random_seed=42,
        )

        with patch("src.services.optimizer.Path") as mock_path:
            opt_dir = tmp_path / "optimization" / objective
            opt_dir.mkdir(parents=True, exist_ok=True)
            mock_path.return_value = opt_dir

            from src.app.config import settings

            optimizer = ParameterOptimizer(
                config=config, client=mock_breeze_client, settings=settings
            )

            result = optimizer.run()

            assert result.metadata["objective_metric"] == objective


def test_empty_results_handling(
    mock_breeze_client: MagicMock,
    tmp_path: Path,
) -> None:
    """Test optimization when all candidates fail."""
    # Return empty bars to cause failures
    mock_breeze_client.historical_bars.return_value = []

    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        search_space={
            "strategy": {
                "sma_fast": [5],
            }
        },
        search_type="grid",
        objective_metric="sharpe_ratio",
        random_seed=42,
    )

    with patch("src.services.optimizer.Path") as mock_path:
        opt_dir = tmp_path / "optimization"
        opt_dir.mkdir(parents=True, exist_ok=True)
        mock_path.return_value = opt_dir

        from src.app.config import settings

        optimizer = ParameterOptimizer(config=config, client=mock_breeze_client, settings=settings)

        result = optimizer.run()

        # Should complete even with all failures
        assert result.successful_candidates == 0
        assert result.failed_candidates == 1
        assert result.best_candidate is None


def test_optimization_execution_time_tracking(
    mock_breeze_client: MagicMock,
    sample_bars: list[Bar],
    tmp_path: Path,
) -> None:
    """Test that execution times are tracked correctly."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    config = OptimizationConfig(
        symbols=["TEST"],
        start_date="2024-01-01",
        end_date="2024-04-30",
        strategy="swing",
        search_space={
            "strategy": {
                "sma_fast": [5, 10],
            }
        },
        search_type="grid",
        objective_metric="sharpe_ratio",
        random_seed=42,
    )

    with patch("src.services.optimizer.Path") as mock_path:
        opt_dir = tmp_path / "optimization"
        opt_dir.mkdir(parents=True, exist_ok=True)
        mock_path.return_value = opt_dir

        from src.app.config import settings

        optimizer = ParameterOptimizer(config=config, client=mock_breeze_client, settings=settings)

        result = optimizer.run()

        # Total time should be positive
        assert result.total_time > 0

        # Each candidate should have elapsed time
        for candidate in result.candidates:
            assert candidate.elapsed_time >= 0
