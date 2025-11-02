"""Unit tests for student batch trainer failure threshold logic (US-028 Phase 7)."""

from __future__ import annotations

from src.app.config import Settings


def calculate_student_exit_code(total_students: int, failed_students: int, max_failure_rate: float) -> int:
    """Calculate student batch trainer exit code based on failure rate threshold.

    This function replicates the logic from scripts/train_student_batch.py main() function.

    Args:
        total_students: Total number of student training windows
        failed_students: Number of failed student trainings
        max_failure_rate: Maximum acceptable failure rate (0.0-1.0)

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    if total_students == 0:
        return 0

    failure_rate = failed_students / total_students
    return 1 if failure_rate > max_failure_rate else 0


class TestStudentBatchTrainerFailureThreshold:
    """Tests for student batch trainer failure threshold logic."""

    def test_zero_failures(self) -> None:
        """Test that zero failures always returns exit code 0."""
        assert calculate_student_exit_code(total_students=100, failed_students=0, max_failure_rate=0.15) == 0
        assert calculate_student_exit_code(total_students=624, failed_students=0, max_failure_rate=0.15) == 0
        assert calculate_student_exit_code(total_students=10, failed_students=0, max_failure_rate=0.01) == 0

    def test_failures_below_threshold(self) -> None:
        """Test that failures below threshold return exit code 0."""
        # 10/100 = 10% failure rate, below 15% threshold
        assert calculate_student_exit_code(total_students=100, failed_students=10, max_failure_rate=0.15) == 0

        # 5/624 = 0.8% failure rate, below 15% threshold
        assert calculate_student_exit_code(total_students=624, failed_students=5, max_failure_rate=0.15) == 0

        # 5/100 = 5% failure rate, below 10% threshold
        assert calculate_student_exit_code(total_students=100, failed_students=5, max_failure_rate=0.10) == 0

    def test_failures_at_threshold(self) -> None:
        """Test that failures exactly at threshold return exit code 0."""
        # 15/100 = 15% failure rate, exactly at 15% threshold
        assert calculate_student_exit_code(total_students=100, failed_students=15, max_failure_rate=0.15) == 0

        # 10/100 = 10% failure rate, exactly at 10% threshold
        assert calculate_student_exit_code(total_students=100, failed_students=10, max_failure_rate=0.10) == 0

    def test_failures_above_threshold(self) -> None:
        """Test that failures above threshold return exit code 1."""
        # 20/100 = 20% failure rate, above 15% threshold
        assert calculate_student_exit_code(total_students=100, failed_students=20, max_failure_rate=0.15) == 1

        # 120/624 = 19.23% failure rate, above 15% threshold
        assert calculate_student_exit_code(total_students=624, failed_students=120, max_failure_rate=0.15) == 1

        # 11/100 = 11% failure rate, above 10% threshold
        assert calculate_student_exit_code(total_students=100, failed_students=11, max_failure_rate=0.10) == 1

    def test_all_failures(self) -> None:
        """Test that 100% failure rate always returns exit code 1."""
        assert calculate_student_exit_code(total_students=100, failed_students=100, max_failure_rate=0.15) == 1
        assert calculate_student_exit_code(total_students=624, failed_students=624, max_failure_rate=0.50) == 1
        assert calculate_student_exit_code(total_students=10, failed_students=10, max_failure_rate=0.99) == 1

    def test_zero_total_students(self) -> None:
        """Test edge case of zero total students."""
        assert calculate_student_exit_code(total_students=0, failed_students=0, max_failure_rate=0.15) == 0

    def test_settings_default_threshold(self) -> None:
        """Test that Settings has correct default max_failure_rate for student batch."""
        settings = Settings()  # type: ignore[call-arg]
        assert hasattr(settings, "batch_training_max_failure_rate")
        assert 0.0 <= settings.batch_training_max_failure_rate <= 1.0
        # Default should be 15%
        assert settings.batch_training_max_failure_rate == 0.15

    def test_different_thresholds(self) -> None:
        """Test same failure rate with different thresholds."""
        # 10/100 = 10% failure rate
        total_students = 100
        failed_students = 10

        # Should pass with 15% threshold
        assert calculate_student_exit_code(total_students, failed_students, max_failure_rate=0.15) == 0

        # Should pass with 10% threshold (at threshold)
        assert calculate_student_exit_code(total_students, failed_students, max_failure_rate=0.10) == 0

        # Should fail with 5% threshold
        assert calculate_student_exit_code(total_students, failed_students, max_failure_rate=0.05) == 1

        # Should fail with 0% threshold (zero tolerance)
        assert calculate_student_exit_code(total_students, failed_students, max_failure_rate=0.00) == 1

    def test_old_logic_any_failure_exits(self) -> None:
        """Test that old logic (any failure = exit 1) is no longer the case.

        Previously, student batch trainer would exit with code 1 if ANY student failed.
        The new threshold logic should allow up to 15% failures by default.
        """
        # With 1 failure out of 624, old logic would exit 1
        # New logic should exit 0 (0.16% << 15%)
        assert calculate_student_exit_code(total_students=624, failed_students=1, max_failure_rate=0.15) == 0

        # With 5 failures out of 624, old logic would exit 1
        # New logic should exit 0 (0.8% << 15%)
        assert calculate_student_exit_code(total_students=624, failed_students=5, max_failure_rate=0.15) == 0

        # With 90 failures out of 624, old logic would exit 1
        # New logic should also exit 0 (14.42% < 15%)
        assert calculate_student_exit_code(total_students=624, failed_students=90, max_failure_rate=0.15) == 0

        # With 95 failures out of 624, failure rate is 15.22% > 15%
        # Should exit 1
        assert calculate_student_exit_code(total_students=624, failed_students=95, max_failure_rate=0.15) == 1
