"""Unit tests for batch trainer failure threshold logic (US-028 Phase 7)."""

from __future__ import annotations

import pytest

from src.app.config import Settings


def calculate_exit_code(total_windows: int, failed_windows: int, max_failure_rate: float) -> int:
    """Calculate batch trainer exit code based on failure rate threshold.

    This function replicates the logic from scripts/train_teacher_batch.py main() function.

    Args:
        total_windows: Total number of training windows
        failed_windows: Number of failed windows
        max_failure_rate: Maximum acceptable failure rate (0.0-1.0)

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    if total_windows == 0:
        return 0

    failure_rate = failed_windows / total_windows
    return 1 if failure_rate > max_failure_rate else 0


class TestBatchTrainerFailureThreshold:
    """Tests for batch trainer failure threshold logic."""

    def test_zero_failures(self) -> None:
        """Test that zero failures always returns exit code 0."""
        assert calculate_exit_code(total_windows=100, failed_windows=0, max_failure_rate=0.15) == 0
        assert calculate_exit_code(total_windows=768, failed_windows=0, max_failure_rate=0.15) == 0
        assert calculate_exit_code(total_windows=10, failed_windows=0, max_failure_rate=0.01) == 0

    def test_failures_below_threshold(self) -> None:
        """Test that failures below threshold return exit code 0."""
        # 10/100 = 10% failure rate, below 15% threshold
        assert calculate_exit_code(total_windows=100, failed_windows=10, max_failure_rate=0.15) == 0

        # 17/768 = 2.21% failure rate, below 15% threshold (real batch example)
        assert calculate_exit_code(total_windows=768, failed_windows=17, max_failure_rate=0.15) == 0

        # 5/100 = 5% failure rate, below 10% threshold
        assert calculate_exit_code(total_windows=100, failed_windows=5, max_failure_rate=0.10) == 0

    def test_failures_at_threshold(self) -> None:
        """Test that failures exactly at threshold return exit code 0."""
        # 15/100 = 15% failure rate, exactly at 15% threshold
        assert calculate_exit_code(total_windows=100, failed_windows=15, max_failure_rate=0.15) == 0

        # 10/100 = 10% failure rate, exactly at 10% threshold
        assert calculate_exit_code(total_windows=100, failed_windows=10, max_failure_rate=0.10) == 0

    def test_failures_above_threshold(self) -> None:
        """Test that failures above threshold return exit code 1."""
        # 20/100 = 20% failure rate, above 15% threshold
        assert calculate_exit_code(total_windows=100, failed_windows=20, max_failure_rate=0.15) == 1

        # 150/768 = 19.53% failure rate, above 15% threshold
        assert calculate_exit_code(total_windows=768, failed_windows=150, max_failure_rate=0.15) == 1

        # 11/100 = 11% failure rate, above 10% threshold
        assert calculate_exit_code(total_windows=100, failed_windows=11, max_failure_rate=0.10) == 1

    def test_all_failures(self) -> None:
        """Test that 100% failure rate always returns exit code 1."""
        assert calculate_exit_code(total_windows=100, failed_windows=100, max_failure_rate=0.15) == 1
        assert calculate_exit_code(total_windows=768, failed_windows=768, max_failure_rate=0.50) == 1
        assert calculate_exit_code(total_windows=10, failed_windows=10, max_failure_rate=0.99) == 1

    def test_zero_total_windows(self) -> None:
        """Test edge case of zero total windows."""
        assert calculate_exit_code(total_windows=0, failed_windows=0, max_failure_rate=0.15) == 0

    def test_settings_default_threshold(self) -> None:
        """Test that Settings has correct default max_failure_rate."""
        settings = Settings()  # type: ignore[call-arg]
        assert hasattr(settings, "batch_training_max_failure_rate")
        assert 0.0 <= settings.batch_training_max_failure_rate <= 1.0
        # Default should be 15%
        assert settings.batch_training_max_failure_rate == 0.15

    def test_real_batch_20251101_141635(self) -> None:
        """Test real batch_20251101_141635: 624/768 succeeded, 17 failed, 127 skipped.

        This batch had:
        - Total windows: 768
        - Completed: 624 (81.25%)
        - Failed: 17 (2.21%)
        - Skipped: 127 (16.54%)

        With default threshold of 15%, this should return exit code 0.
        Previously returned exit code 1 due to ANY failure causing exit.
        """
        total_windows = 768
        failed_windows = 17
        failure_rate = failed_windows / total_windows  # 2.21%

        # Should pass with default 15% threshold
        assert calculate_exit_code(total_windows, failed_windows, max_failure_rate=0.15) == 0

        # Verify actual failure rate (17/768 = 0.022135...)
        assert failure_rate == pytest.approx(0.02214, rel=1e-3)
        assert failure_rate < 0.15  # Well below threshold

    def test_different_thresholds(self) -> None:
        """Test same failure rate with different thresholds."""
        # 10/100 = 10% failure rate
        total_windows = 100
        failed_windows = 10

        # Should pass with 15% threshold
        assert calculate_exit_code(total_windows, failed_windows, max_failure_rate=0.15) == 0

        # Should pass with 10% threshold (at threshold)
        assert calculate_exit_code(total_windows, failed_windows, max_failure_rate=0.10) == 0

        # Should fail with 5% threshold
        assert calculate_exit_code(total_windows, failed_windows, max_failure_rate=0.05) == 1

        # Should fail with 0% threshold (zero tolerance)
        assert calculate_exit_code(total_windows, failed_windows, max_failure_rate=0.00) == 1
