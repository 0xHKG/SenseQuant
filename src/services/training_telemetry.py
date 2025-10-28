"""Training Telemetry Logger for Historical Training Pipeline (US-028 Phase 7 Batch 4).

This module provides telemetry capture for training events, enabling real-time
monitoring of historical training runs via the Streamlit dashboard.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger

EventType = Literal[
    "run_start",
    "run_end",
    "phase_start",
    "phase_end",
    "teacher_window_start",
    "teacher_window_success",
    "teacher_window_skip",
    "teacher_window_fail",
    "student_batch_start",
    "student_batch_end",
    "validation_complete",
    "stat_test_complete",
]

PhaseType = Literal[
    "data_ingestion",
    "teacher_training",
    "student_training",
    "model_validation",
    "statistical_tests",
    "release_audit",
    "promotion_briefing",
    "stress_tests",
]


@dataclass
class TrainingEvent:
    """Training event for telemetry capture.

    Captures key training milestones, window-level progress, and aggregate metrics
    to enable real-time monitoring via dashboard.
    """

    timestamp: str  # ISO format datetime
    run_id: str  # Training run identifier
    event_type: EventType  # Type of event
    phase: PhaseType | None = None  # Training phase (if applicable)
    symbol: str | None = None  # Symbol (for per-symbol events)
    window_label: str | None = None  # Window identifier (for teacher windows)
    status: str | None = None  # success, failed, skipped, in_progress
    metrics: dict[str, Any] | None = None  # Event-specific metrics
    message: str | None = None  # Human-readable message
    duration_seconds: float | None = None  # Duration (for end events)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        return asdict(self)


class TrainingTelemetryLogger:
    """Logger for training telemetry events.

    Writes JSONL-formatted training events to disk for dashboard consumption.
    Uses buffered writing with automatic flushing.
    """

    def __init__(
        self,
        output_dir: Path,
        run_id: str,
        buffer_size: int = 50,
        enabled: bool = True,
    ):
        """Initialize training telemetry logger.

        Args:
            output_dir: Directory for telemetry output (e.g., data/analytics/training/)
            run_id: Training run identifier
            buffer_size: Number of events to buffer before flushing (default: 50)
            enabled: Enable/disable telemetry (default: True)
        """
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.buffer_size = buffer_size
        self.enabled = enabled

        # Event buffer
        self._buffer: list[TrainingEvent] = []

        # Output file path
        self.output_file = self.output_dir / f"training_run_{run_id}.jsonl"

        if self.enabled:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize output file (create if doesn't exist)
            if not self.output_file.exists():
                self.output_file.touch()

            logger.info(
                f"TrainingTelemetryLogger initialized: {self.output_file}",
                extra={"component": "training_telemetry", "run_id": run_id},
            )
        else:
            logger.info(
                "TrainingTelemetryLogger disabled",
                extra={"component": "training_telemetry"},
            )

    def log_event(self, event: TrainingEvent) -> None:
        """Log a training event.

        Args:
            event: TrainingEvent to log
        """
        if not self.enabled:
            return

        # Add to buffer
        self._buffer.append(event)

        # Flush if buffer is full
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def log_run_event(
        self,
        event_type: Literal["run_start", "run_end"],
        status: str | None = None,
        message: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """Log a run-level event (start/end).

        Args:
            event_type: run_start or run_end
            status: Status (e.g., "in_progress", "success", "failed")
            message: Human-readable message
            duration_seconds: Duration (for run_end)
        """
        event = TrainingEvent(
            timestamp=datetime.now().isoformat(),
            run_id=self.run_id,
            event_type=event_type,
            status=status,
            message=message,
            duration_seconds=duration_seconds,
        )
        self.log_event(event)

    def log_phase_event(
        self,
        phase: PhaseType,
        event_type: Literal["phase_start", "phase_end"],
        status: str | None = None,
        metrics: dict[str, Any] | None = None,
        message: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """Log a phase-level event (start/end).

        Args:
            phase: Training phase (e.g., "teacher_training")
            event_type: phase_start or phase_end
            status: Status (e.g., "in_progress", "success", "failed")
            metrics: Phase-specific metrics
            message: Human-readable message
            duration_seconds: Duration (for phase_end)
        """
        event = TrainingEvent(
            timestamp=datetime.now().isoformat(),
            run_id=self.run_id,
            event_type=event_type,
            phase=phase,
            status=status,
            metrics=metrics,
            message=message,
            duration_seconds=duration_seconds,
        )
        self.log_event(event)

    def log_teacher_window(
        self,
        symbol: str,
        window_label: str,
        event_type: Literal[
            "teacher_window_start",
            "teacher_window_success",
            "teacher_window_skip",
            "teacher_window_fail",
        ],
        status: str,
        metrics: dict[str, Any] | None = None,
        message: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """Log a teacher window event.

        Args:
            symbol: Symbol being trained
            window_label: Window identifier (e.g., "RELIANCE_2022Q1_intraday")
            event_type: Window event type
            status: Status (success, skipped, failed)
            metrics: Window metrics (precision, recall, F1, sample counts, etc.)
            message: Human-readable message (e.g., skip reason, error message)
            duration_seconds: Training duration (for success/fail)
        """
        event = TrainingEvent(
            timestamp=datetime.now().isoformat(),
            run_id=self.run_id,
            event_type=event_type,
            phase="teacher_training",
            symbol=symbol,
            window_label=window_label,
            status=status,
            metrics=metrics,
            message=message,
            duration_seconds=duration_seconds,
        )
        self.log_event(event)

    def log_student_metrics(
        self,
        event_type: Literal["student_batch_start", "student_batch_end"],
        status: str,
        metrics: dict[str, Any] | None = None,
        message: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """Log student training metrics.

        Args:
            event_type: student_batch_start or student_batch_end
            status: Status (in_progress, success, failed)
            metrics: Student metrics (accuracy, precision, recall, sample counts, etc.)
            message: Human-readable message
            duration_seconds: Training duration (for batch_end)
        """
        event = TrainingEvent(
            timestamp=datetime.now().isoformat(),
            run_id=self.run_id,
            event_type=event_type,
            phase="student_training",
            status=status,
            metrics=metrics,
            message=message,
            duration_seconds=duration_seconds,
        )
        self.log_event(event)

    def log_validation_complete(
        self,
        status: str,
        metrics: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        """Log validation completion.

        Args:
            status: Status (success, failed)
            metrics: Validation metrics
            message: Human-readable message
        """
        event = TrainingEvent(
            timestamp=datetime.now().isoformat(),
            run_id=self.run_id,
            event_type="validation_complete",
            phase="model_validation",
            status=status,
            metrics=metrics,
            message=message,
        )
        self.log_event(event)

    def log_stat_test_complete(
        self,
        status: str,
        metrics: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        """Log statistical test completion.

        Args:
            status: Status (success, failed)
            metrics: Statistical test metrics
            message: Human-readable message
        """
        event = TrainingEvent(
            timestamp=datetime.now().isoformat(),
            run_id=self.run_id,
            event_type="stat_test_complete",
            phase="statistical_tests",
            status=status,
            metrics=metrics,
            message=message,
        )
        self.log_event(event)

    def flush(self) -> None:
        """Flush buffered events to disk."""
        if not self.enabled or not self._buffer:
            return

        try:
            # Append all buffered events to file as JSONL
            with open(self.output_file, "a") as f:
                for event in self._buffer:
                    json_line = json.dumps(event.to_dict())
                    f.write(json_line + "\n")

            logger.debug(
                f"Flushed {len(self._buffer)} telemetry events to {self.output_file}",
                extra={"component": "training_telemetry"},
            )

            # Clear buffer
            self._buffer.clear()

        except Exception as e:
            logger.error(
                f"Failed to flush telemetry events: {e}",
                extra={"component": "training_telemetry"},
            )

    def close(self) -> None:
        """Close logger and flush remaining events."""
        if not self.enabled:
            return

        # Flush any remaining buffered events
        self.flush()

        logger.info(
            f"TrainingTelemetryLogger closed: {self.output_file}",
            extra={"component": "training_telemetry", "run_id": self.run_id},
        )

    def __enter__(self) -> TrainingTelemetryLogger:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic flush."""
        self.close()


# Validation helper for manual testing
def validate_telemetry_output(output_file: Path) -> bool:
    """Validate JSONL telemetry output format.

    Args:
        output_file: Path to telemetry JSONL file

    Returns:
        True if valid, False otherwise
    """
    if not output_file.exists():
        logger.error(f"Telemetry file not found: {output_file}")
        return False

    try:
        with open(output_file) as f:
            lines = f.readlines()

        if not lines:
            logger.warning(f"Telemetry file is empty: {output_file}")
            return True  # Empty is valid

        # Parse each line as JSON
        for i, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            try:
                event_dict = json.loads(line)

                # Validate required fields
                required_fields = ["timestamp", "run_id", "event_type"]
                missing_fields = [f for f in required_fields if f not in event_dict]
                if missing_fields:
                    logger.error(
                        f"Line {i}: Missing required fields: {missing_fields}"
                    )
                    return False

            except json.JSONDecodeError as e:
                logger.error(f"Line {i}: Invalid JSON: {e}")
                return False

        logger.info(f"Validated {len(lines)} telemetry events in {output_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to validate telemetry file: {e}")
        return False


# Manual test snippet
if __name__ == "__main__":
    """Manual test of TrainingTelemetryLogger."""
    import tempfile

    # Create temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        run_id = "test_run_20251028_163500"

        # Initialize logger
        with TrainingTelemetryLogger(
            output_dir=test_dir,
            run_id=run_id,
            buffer_size=5,
            enabled=True,
        ) as telemetry:
            # Log run start
            telemetry.log_run_event(
                event_type="run_start",
                status="in_progress",
                message="Starting test training run",
            )

            # Log phase start
            telemetry.log_phase_event(
                phase="teacher_training",
                event_type="phase_start",
                status="in_progress",
                message="Starting teacher training",
            )

            # Log teacher window events
            for i in range(3):
                telemetry.log_teacher_window(
                    symbol="RELIANCE",
                    window_label=f"RELIANCE_2022Q{i+1}_intraday",
                    event_type="teacher_window_success",
                    status="success",
                    metrics={
                        "precision": 0.82 + (i * 0.01),
                        "recall": 0.78,
                        "f1": 0.80,
                        "train_samples": 5000,
                        "val_samples": 1250,
                    },
                    duration_seconds=45.2,
                )

            # Log skipped window
            telemetry.log_teacher_window(
                symbol="TCS",
                window_label="TCS_2022Q1_swing",
                event_type="teacher_window_skip",
                status="skipped",
                message="Insufficient samples (<100)",
            )

            # Log phase end
            telemetry.log_phase_event(
                phase="teacher_training",
                event_type="phase_end",
                status="success",
                metrics={"total_windows": 4, "success": 3, "skipped": 1, "failed": 0},
                duration_seconds=180.5,
            )

            # Log run end
            telemetry.log_run_event(
                event_type="run_end",
                status="success",
                message="Test training run complete",
                duration_seconds=200.0,
            )

        # Validate output
        output_file = test_dir / f"training_run_{run_id}.jsonl"
        if validate_telemetry_output(output_file):
            print(f"✅ Telemetry validation passed: {output_file}")
            print("\nSample events:")
            with open(output_file) as f:
                for i, line in enumerate(f, start=1):
                    if i <= 3:  # Show first 3 events
                        event = json.loads(line)
                        print(f"  {i}. {event['event_type']}: {event.get('message', 'N/A')}")
        else:
            print(f"❌ Telemetry validation failed")
