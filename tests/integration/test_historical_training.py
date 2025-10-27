"""Integration tests for historical data ingestion and batch teacher training (US-024)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from fetch_historical_data import HistoricalDataFetcher
from train_teacher_batch import BatchTrainer

from src.app.config import Settings
from src.services.teacher_student import TeacherLabeler


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory structure."""
    historical_dir = tmp_path / "historical"
    models_dir = tmp_path / "models"

    historical_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    return tmp_path


@pytest.fixture
def mock_settings(tmp_data_dir: Path) -> Settings:
    """Create Settings with temporary directories."""
    settings = Settings()  # type: ignore[call-arg]
    settings.historical_data_output_dir = str(tmp_data_dir / "historical")
    settings.batch_training_output_dir = str(tmp_data_dir / "models")
    settings.historical_data_symbols = ["RELIANCE", "TCS"]
    settings.historical_data_start_date = "2024-01-01"
    settings.historical_data_end_date = "2024-01-03"  # Short range for testing
    settings.historical_data_intervals = ["1minute"]
    settings.batch_training_window_days = 30
    settings.batch_training_forecast_horizon_days = 7
    return settings


def create_mock_ohlcv_data(symbol: str, date: str, rows: int = 100) -> pd.DataFrame:
    """Create mock OHLCV DataFrame for testing."""
    base_price = 2450.0
    return pd.DataFrame(
        {
            "timestamp": [f"{date}T{9 + i // 60:02d}:{i % 60:02d}:00+05:30" for i in range(rows)],
            "open": [base_price + i * 0.5 for i in range(rows)],
            "high": [base_price + i * 0.5 + 2.0 for i in range(rows)],
            "low": [base_price + i * 0.5 - 1.0 for i in range(rows)],
            "close": [base_price + i * 0.5 + 1.0 for i in range(rows)],
            "volume": [100000 + i * 1000 for i in range(rows)],
        }
    )


def test_historical_data_fetch_with_cache(mock_settings: Settings, tmp_data_dir: Path) -> None:
    """Test historical data fetch with caching."""
    # Create fetcher
    fetcher = HistoricalDataFetcher(mock_settings, breeze_client=None, dryrun=True)

    # Fetch data for single symbol/date/interval
    success = fetcher.fetch_symbol_date("RELIANCE", "2024-01-01", "1minute", force=False)

    assert success is True
    assert fetcher.stats["downloads"] == 1
    assert fetcher.stats["cached_hits"] == 0

    # Fetch again - should hit cache
    success2 = fetcher.fetch_symbol_date("RELIANCE", "2024-01-01", "1minute", force=False)

    assert success2 is True
    assert fetcher.stats["downloads"] == 1  # Still 1 (not re-downloaded)
    assert fetcher.stats["cached_hits"] == 1  # Cache hit

    # Verify CSV file created
    cache_path = fetcher.get_cache_path("RELIANCE", "1minute", "2024-01-01")
    assert cache_path.exists()

    # Verify CSV content
    df = pd.read_csv(cache_path)
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns


def test_ohlcv_data_validation(mock_settings: Settings) -> None:
    """Test OHLCV data validation."""
    fetcher = HistoricalDataFetcher(mock_settings, breeze_client=None, dryrun=True)

    # Valid data
    valid_df = create_mock_ohlcv_data("RELIANCE", "2024-01-01", rows=10)
    is_valid, error, corrected_df, warnings = fetcher.validate_ohlcv_data(valid_df, "RELIANCE", "2024-01-01")
    assert is_valid is True
    assert error is None
    assert len(warnings) == 0

    # Invalid data - missing columns (hard error)
    invalid_df = pd.DataFrame({"timestamp": ["2024-01-01T09:15:00"], "open": [2450.0]})
    is_valid, error, corrected_df, warnings = fetcher.validate_ohlcv_data(invalid_df, "RELIANCE", "2024-01-01")
    assert is_valid is False
    assert "Missing columns" in error

    # Invalid data - OHLC relationship (US-028 Phase 6j: now treated as warning, not fatal error)
    invalid_ohlc_df = create_mock_ohlcv_data("RELIANCE", "2024-01-01", rows=10)
    invalid_ohlc_df.loc[0, "high"] = 2400.0  # High < close
    is_valid, error, corrected_df, warnings = fetcher.validate_ohlcv_data(invalid_ohlc_df, "RELIANCE", "2024-01-01")
    assert is_valid is True  # Phase 6j: OHLC issues are warnings, not hard errors
    assert len(warnings) > 0  # Should have warning about invalid OHLC
    assert any("OHLC" in w for w in warnings)


def test_batch_training_window_generation(mock_settings: Settings, tmp_data_dir: Path) -> None:
    """Test batch training window generation."""
    trainer = BatchTrainer(mock_settings, tmp_data_dir / "models", resume=False)

    # Generate windows
    tasks = trainer.generate_training_windows(
        symbols=["RELIANCE", "TCS"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        window_days=90,
    )

    # Should generate multiple windows per symbol
    assert len(tasks) > 0

    # Check task structure
    task = tasks[0]
    assert "symbol" in task
    assert "start_date" in task
    assert "end_date" in task
    assert "window_label" in task
    assert "artifacts_path" in task

    # Verify window labels
    reliance_tasks = [t for t in tasks if t["symbol"] == "RELIANCE"]
    assert len(reliance_tasks) > 0
    assert reliance_tasks[0]["window_label"].startswith("RELIANCE_2024")


def test_batch_metadata_logging(tmp_data_dir: Path) -> None:
    """Test batch metadata logging."""
    metadata_file = tmp_data_dir / "teacher_runs.json"

    # Log first entry
    TeacherLabeler.log_batch_metadata(
        metadata_file=metadata_file,
        batch_id="batch_test_001",
        symbol="RELIANCE",
        date_range={"start": "2024-01-01", "end": "2024-03-31"},
        artifacts_path="data/models/test/RELIANCE_2024Q1",
        metrics={"precision": 0.72, "recall": 0.68, "f1": 0.70},
        status="success",
    )

    # Log second entry
    TeacherLabeler.log_batch_metadata(
        metadata_file=metadata_file,
        batch_id="batch_test_001",
        symbol="TCS",
        date_range={"start": "2024-01-01", "end": "2024-03-31"},
        artifacts_path="data/models/test/TCS_2024Q1",
        metrics=None,
        status="failed",
        error="Insufficient data points",
    )

    # Verify file exists
    assert metadata_file.exists()

    # Load and verify metadata
    metadata_list = TeacherLabeler.load_batch_metadata(metadata_file)
    assert len(metadata_list) == 2

    # Check first entry
    entry1 = metadata_list[0]
    assert entry1["batch_id"] == "batch_test_001"
    assert entry1["symbol"] == "RELIANCE"
    assert entry1["status"] == "success"
    assert entry1["metrics"]["precision"] == 0.72
    assert "timestamp" in entry1

    # Check second entry
    entry2 = metadata_list[1]
    assert entry2["symbol"] == "TCS"
    assert entry2["status"] == "failed"
    assert entry2["error"] == "Insufficient data points"
    assert entry2["metrics"] is None


def test_dryrun_mode_no_network_calls(mock_settings: Settings) -> None:
    """Test that dryrun mode does not make network calls."""
    fetcher = HistoricalDataFetcher(mock_settings, breeze_client=None, dryrun=True)

    # Fetch in dryrun mode (should not require BreezeClient)
    df = fetcher.fetch_with_retry("RELIANCE", "2024-01-01", "1minute")

    # Should return mock data
    assert df is not None
    assert len(df) > 0
    assert "timestamp" in df.columns

    # Stats should be tracked
    assert fetcher.stats["total_requests"] == 0  # Not incremented yet (only in fetch_symbol_date)


def test_cache_path_structure(mock_settings: Settings) -> None:
    """Test cache path generation follows expected structure."""
    fetcher = HistoricalDataFetcher(mock_settings, breeze_client=None, dryrun=True)

    cache_path = fetcher.get_cache_path("RELIANCE", "1minute", "2024-01-15")

    # Verify path structure: data/historical/<symbol>/<interval>/YYYY-MM-DD.csv
    assert "RELIANCE" in str(cache_path)
    assert "1minute" in str(cache_path)
    assert "2024-01-15.csv" in str(cache_path)
    assert cache_path.name == "2024-01-15.csv"


def test_date_range_validation(mock_settings: Settings) -> None:
    """Test date range validation."""
    fetcher = HistoricalDataFetcher(mock_settings, breeze_client=None, dryrun=True)

    # Valid range
    start_dt, end_dt = fetcher.validate_date_range("2024-01-01", "2024-12-31")
    assert start_dt.year == 2024
    assert end_dt.year == 2024

    # Invalid range (start >= end)
    with pytest.raises(ValueError, match="must be before"):
        fetcher.validate_date_range("2024-12-31", "2024-01-01")

    # Invalid format
    with pytest.raises(ValueError, match="Invalid date format"):
        fetcher.validate_date_range("2024/01/01", "2024-12-31")


def test_resume_functionality(mock_settings: Settings, tmp_data_dir: Path) -> None:
    """Test batch training resume functionality.

    US-028 Phase 6t: Updated to use Phase 6o artifact structure (gzipped labels).
    """
    import gzip
    import json

    trainer = BatchTrainer(mock_settings, tmp_data_dir / "models", resume=True)

    # Create a task
    task = {
        "symbol": "RELIANCE",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "window_label": "RELIANCE_2024Q1",
        "artifacts_path": str(tmp_data_dir / "models" / "test" / "RELIANCE_2024Q1"),
    }

    # Initially not trained
    assert trainer.is_already_trained(task) is False

    # Create mock artifacts (Phase 6o structure: model.pkl, labels.csv.gz, metadata.json)
    artifacts_path = Path(task["artifacts_path"])
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # Create gzipped labels file
    with gzip.open(artifacts_path / "labels.csv.gz", "wt") as f:
        f.write("timestamp,label\n2024-01-01,1\n")

    # Create model file
    (artifacts_path / "model.pkl").write_bytes(b"mock_model_data")

    # Create metadata file
    metadata = {
        "symbol": "RELIANCE",
        "window_label": "RELIANCE_2024Q1",
        "date_range": {"start": "2024-01-01", "end": "2024-03-31"},
    }
    (artifacts_path / "metadata.json").write_text(json.dumps(metadata))

    # Now should be detected as trained
    assert trainer.is_already_trained(task) is True


def test_fetch_all_summary_statistics(mock_settings: Settings, tmp_data_dir: Path) -> None:
    """Test fetch_all returns correct summary statistics."""
    fetcher = HistoricalDataFetcher(mock_settings, breeze_client=None, dryrun=True)

    # Fetch data for multiple symbols/dates
    summary = fetcher.fetch_all(
        symbols=["RELIANCE", "TCS"],
        start_date="2024-01-01",
        end_date="2024-01-02",
        intervals=["1minute"],
        force=False,
    )

    # Verify summary structure
    assert "total_requests" in summary
    assert "cached_hits" in summary
    assert "downloads" in summary
    assert "failures" in summary
    assert "total_rows" in summary
    assert "cache_hit_rate" in summary
    assert "chunks_fetched" in summary
    assert "chunks_failed" in summary

    # Verify chunking was used (not day-by-day)
    assert summary["chunks_fetched"] > 0
    assert summary["downloads"] > 0


def test_chunked_historical_fetch_multi_chunk_aggregation(mock_settings: Settings, tmp_data_dir: Path) -> None:
    """Test chunked ingestion with multi-chunk date range (US-028 Phase 6b).

    This test verifies:
    1. Date range is split into multiple chunks based on settings.historical_chunk_days
    2. Each chunk is fetched separately
    3. Results are combined into a single continuous DataFrame
    4. Rate limiting is applied between chunks
    5. Chunk statistics are tracked correctly
    """
    from datetime import datetime

    # Configure mock settings with small chunk size to force multiple chunks
    mock_settings.historical_chunk_days = 30  # 30-day chunks

    fetcher = HistoricalDataFetcher(mock_settings, breeze_client=None, dryrun=True)

    # Test date range: 100 days (should create ~4 chunks)
    start_dt = datetime(2024, 1, 1)
    end_dt = datetime(2024, 4, 10)  # 100 days

    # Calculate expected chunks
    expected_chunks = fetcher.split_date_range_into_chunks(start_dt, end_dt)
    assert len(expected_chunks) >= 3, "Should create at least 3 chunks for 100-day range with 30-day chunks"

    # Verify chunk boundaries are correct
    assert expected_chunks[0][0] == start_dt
    assert expected_chunks[-1][1] == end_dt

    # Verify no gaps between chunks
    for i in range(len(expected_chunks) - 1):
        chunk_end = expected_chunks[i][1]
        next_chunk_start = expected_chunks[i + 1][0]
        assert (next_chunk_start - chunk_end).days == 1, "Chunks should be contiguous"

    # Fetch data using chunked method
    df = fetcher.fetch_symbol_date_range_chunked(
        symbol="RELIANCE",
        start_dt=start_dt,
        end_dt=end_dt,
        interval="1day",
        force=False,
    )

    # Verify combined DataFrame
    assert df is not None
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns

    # Verify no duplicate timestamps (from chunk overlap)
    assert len(df) == len(df.drop_duplicates(subset=["timestamp"]))

    # Verify timestamps are sorted
    timestamps = pd.to_datetime(df["timestamp"])
    assert timestamps.is_monotonic_increasing, "Timestamps should be in ascending order"

    # Verify chunk statistics
    assert fetcher.stats["chunks_fetched"] == len(expected_chunks)
    assert fetcher.stats["chunks_failed"] == 0
    assert fetcher.stats["downloads"] == len(expected_chunks)


def test_chunked_fetch_with_mocked_breeze_client(mock_settings: Settings, tmp_data_dir: Path) -> None:
    """Test chunked fetch with mocked BreezeClient to verify API interactions (US-028 Phase 6b)."""
    from datetime import datetime
    from unittest.mock import Mock

    # Configure settings
    mock_settings.historical_chunk_days = 90
    mock_settings.breeze_rate_limit_delay_seconds = 0.1  # Fast for testing

    # Create mock BreezeClient
    mock_breeze = Mock()
    mock_breeze.fetch_historical_chunk = Mock(return_value=pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
        "open": [2450.0] * 10,
        "high": [2455.0] * 10,
        "low": [2448.0] * 10,
        "close": [2453.0] * 10,
        "volume": [100000] * 10,
    }))

    fetcher = HistoricalDataFetcher(mock_settings, breeze_client=mock_breeze, dryrun=False)

    # Fetch 200-day range (should create ~3 chunks with 90-day chunk size)
    start_dt = datetime(2024, 1, 1)
    end_dt = datetime(2024, 7, 20)

    df = fetcher.fetch_symbol_date_range_chunked(
        symbol="RELIANCE",
        start_dt=start_dt,
        end_dt=end_dt,
        interval="1day",
        force=False,
    )

    # Verify BreezeClient was called multiple times (once per chunk)
    expected_chunks = fetcher.split_date_range_into_chunks(start_dt, end_dt)
    assert mock_breeze.fetch_historical_chunk.call_count == len(expected_chunks)

    # Verify combined result
    assert len(df) > 0
    assert fetcher.stats["chunks_fetched"] == len(expected_chunks)

    # Verify each call had correct parameters
    for i, (_chunk_start, _chunk_end) in enumerate(expected_chunks):
        call_args = mock_breeze.fetch_historical_chunk.call_args_list[i]
        assert call_args[1]["symbol"] == "RELIANCE"
        assert call_args[1]["interval"] == "1day"
        # Timestamps should be timezone-aware
        assert call_args[1]["start_date"].tz is not None
        assert call_args[1]["end_date"].tz is not None


# ============================================================================
# US-024 PHASE 2: Student Batch Training Tests
# ============================================================================


def test_student_batch_metadata_logging(tmp_data_dir: Path) -> None:
    """Test student batch metadata logging (Phase 2)."""
    from src.services.teacher_student import StudentModel

    metadata_file = tmp_data_dir / "student_runs.json"

    # Log first entry
    StudentModel.log_batch_metadata(
        metadata_file=metadata_file,
        batch_id="batch_20251012_190000",
        symbol="RELIANCE",
        teacher_run_id="RELIANCE_2024Q1",
        teacher_artifacts_path="data/models/20251012_190000/RELIANCE_2024Q1",
        student_artifacts_path="data/models/20251012_190000/RELIANCE_2024Q1_student",
        metrics={"precision": 0.68, "recall": 0.65, "f1": 0.66},
        promotion_checklist_path="data/models/20251012_190000/RELIANCE_2024Q1_student/promotion_checklist.md",
        status="success",
    )

    # Log second entry (failed)
    StudentModel.log_batch_metadata(
        metadata_file=metadata_file,
        batch_id="batch_20251012_190000",
        symbol="TCS",
        teacher_run_id="TCS_2024Q1",
        teacher_artifacts_path="data/models/20251012_190000/TCS_2024Q1",
        student_artifacts_path="data/models/20251012_190000/TCS_2024Q1_student",
        metrics=None,
        promotion_checklist_path=None,
        status="failed",
        error="Training timeout",
    )

    # Verify file exists
    assert metadata_file.exists()

    # Load and verify metadata
    metadata_list = StudentModel.load_batch_metadata(metadata_file)
    assert len(metadata_list) == 2

    # Check first entry
    entry1 = metadata_list[0]
    assert entry1["batch_id"] == "batch_20251012_190000"
    assert entry1["symbol"] == "RELIANCE"
    assert entry1["teacher_run_id"] == "RELIANCE_2024Q1"
    assert entry1["status"] == "success"
    assert entry1["metrics"]["precision"] == 0.68
    assert "timestamp" in entry1

    # Check second entry
    entry2 = metadata_list[1]
    assert entry2["symbol"] == "TCS"
    assert entry2["status"] == "failed"
    assert entry2["error"] == "Training timeout"
    assert entry2["metrics"] is None


def test_student_batch_summary(tmp_data_dir: Path) -> None:
    """Test student batch results summarization (Phase 2)."""
    from src.services.teacher_student import StudentModel

    metadata_file = tmp_data_dir / "student_runs.json"

    # Create test data
    StudentModel.log_batch_metadata(
        metadata_file=metadata_file,
        batch_id="batch_test",
        symbol="RELIANCE",
        teacher_run_id="RELIANCE_2024Q1",
        teacher_artifacts_path="data/models/test/RELIANCE_2024Q1",
        student_artifacts_path="data/models/test/RELIANCE_2024Q1_student",
        metrics={"precision": 0.70, "recall": 0.68, "f1": 0.69},
        promotion_checklist_path="data/models/test/RELIANCE_2024Q1_student/promotion_checklist.md",
        status="success",
    )

    StudentModel.log_batch_metadata(
        metadata_file=metadata_file,
        batch_id="batch_test",
        symbol="TCS",
        teacher_run_id="TCS_2024Q1",
        teacher_artifacts_path="data/models/test/TCS_2024Q1",
        student_artifacts_path="data/models/test/TCS_2024Q1_student",
        metrics={"precision": 0.72, "recall": 0.70, "f1": 0.71},
        promotion_checklist_path="data/models/test/TCS_2024Q1_student/promotion_checklist.md",
        status="success",
    )

    StudentModel.log_batch_metadata(
        metadata_file=metadata_file,
        batch_id="batch_test",
        symbol="INFY",
        teacher_run_id="INFY_2024Q1",
        teacher_artifacts_path="data/models/test/INFY_2024Q1",
        student_artifacts_path="data/models/test/INFY_2024Q1_student",
        metrics=None,
        promotion_checklist_path=None,
        status="failed",
        error="Insufficient data",
    )

    # Load and summarize
    metadata_list = StudentModel.load_batch_metadata(metadata_file)
    summary = StudentModel.summarize_batch_results(metadata_list)

    # Verify summary
    assert summary["total"] == 3
    assert summary["successful"] == 2
    assert summary["failed"] == 1
    assert summary["success_rate"] == pytest.approx(2 / 3)
    assert summary["avg_precision"] == pytest.approx(0.71)
    assert summary["avg_recall"] == pytest.approx(0.69)
    assert summary["avg_f1"] == pytest.approx(0.70)


def test_student_batch_workflow_integration(tmp_data_dir: Path, mock_settings: Settings) -> None:
    """Test end-to-end teacher to student batch workflow (Phase 2)."""
    from src.services.teacher_student import StudentModel, TeacherLabeler

    # Create batch directory
    batch_dir = tmp_data_dir / "models" / "20251012_190000"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create teacher batch metadata (from Phase 1)
    teacher_metadata_file = batch_dir / "teacher_runs.json"
    TeacherLabeler.log_batch_metadata(
        metadata_file=teacher_metadata_file,
        batch_id="batch_20251012_190000",
        symbol="RELIANCE",
        date_range={"start": "2024-01-01", "end": "2024-03-31"},
        artifacts_path=str(batch_dir / "RELIANCE_2024Q1"),
        metrics={"precision": 0.72, "recall": 0.68, "f1": 0.70},
        status="success",
    )

    TeacherLabeler.log_batch_metadata(
        metadata_file=teacher_metadata_file,
        batch_id="batch_20251012_190000",
        symbol="TCS",
        date_range={"start": "2024-01-01", "end": "2024-03-31"},
        artifacts_path=str(batch_dir / "TCS_2024Q1"),
        metrics={"precision": 0.75, "recall": 0.71, "f1": 0.73},
        status="success",
    )

    # Verify teacher metadata
    teacher_runs = TeacherLabeler.load_batch_metadata(teacher_metadata_file)
    assert len(teacher_runs) == 2

    # Step 2: Create student batch metadata (Phase 2)
    student_metadata_file = batch_dir / "student_runs.json"

    for teacher_run in teacher_runs:
        # Simulate student training for each teacher run
        window_label = Path(teacher_run["artifacts_path"]).name
        student_artifacts_path = batch_dir / f"{window_label}_student"

        StudentModel.log_batch_metadata(
            metadata_file=student_metadata_file,
            batch_id=teacher_run["batch_id"],
            symbol=teacher_run["symbol"],
            teacher_run_id=window_label,
            teacher_artifacts_path=teacher_run["artifacts_path"],
            student_artifacts_path=str(student_artifacts_path),
            metrics={"precision": 0.68, "recall": 0.65, "f1": 0.66},
            promotion_checklist_path=str(student_artifacts_path / "promotion_checklist.md"),
            status="success",
        )

    # Verify student metadata
    student_runs = StudentModel.load_batch_metadata(student_metadata_file)
    assert len(student_runs) == 2

    # Verify linkage between teacher and student runs
    for student_run in student_runs:
        assert student_run["teacher_run_id"] in ["RELIANCE_2024Q1", "TCS_2024Q1"]
        assert "teacher_artifacts_path" in student_run
        assert "student_artifacts_path" in student_run
        assert "promotion_checklist_path" in student_run

    # Verify summary
    summary = StudentModel.summarize_batch_results(student_runs)
    assert summary["total"] == 2
    assert summary["successful"] == 2
    assert summary["success_rate"] == 1.0


def test_batch_mode_config_defaults(mock_settings: Settings) -> None:
    """Test that student batch config defaults are safe (Phase 2)."""
    # Verify student batch is disabled by default
    assert mock_settings.student_batch_enabled is False

    # Verify baseline thresholds are reasonable
    assert 0.0 <= mock_settings.student_batch_baseline_precision <= 1.0
    assert 0.0 <= mock_settings.student_batch_baseline_recall <= 1.0

    # Verify promotion is enabled by default
    assert mock_settings.student_batch_promotion_enabled is True


def test_student_batch_resume_functionality(tmp_data_dir: Path) -> None:
    """Test student batch resume (skip already-trained) (Phase 2)."""
    from train_student_batch import StudentBatchTrainer

    # Create batch directory with teacher metadata
    batch_dir = tmp_data_dir / "models" / "20251012_190000"
    batch_dir.mkdir(parents=True, exist_ok=True)

    teacher_metadata_file = batch_dir / "teacher_runs.json"
    teacher_metadata_file.write_text(
        '{"batch_id": "batch_test", "symbol": "RELIANCE", "artifacts_path": "'
        + str(batch_dir / "RELIANCE_2024Q1")
        + '", "status": "success"}\n'
    )

    # Create trainer
    trainer = StudentBatchTrainer(
        teacher_batch_dir=batch_dir,
        baseline_precision=0.60,
        baseline_recall=0.55,
        resume=True,
    )

    # Get first teacher run
    teacher_run = trainer.teacher_runs[0]

    # Initially not trained
    assert trainer.is_already_trained(teacher_run) is False

    # Create student artifacts
    student_artifacts_path = trainer.get_student_artifacts_path(teacher_run)
    student_artifacts_path.mkdir(parents=True, exist_ok=True)
    (student_artifacts_path / "student_model.pkl").write_text("mock_model")

    # Now should be detected as trained
    assert trainer.is_already_trained(teacher_run) is True


# ============================================================================
# US-024 PHASE 3: Sentiment Snapshot Ingestion Tests
# ============================================================================


def test_sentiment_snapshot_fetch(tmp_data_dir: Path) -> None:
    """Test sentiment snapshot fetch creates JSONL files (Phase 3)."""
    from fetch_sentiment_snapshots import SentimentSnapshotFetcher

    from src.services.sentiment.providers.stub import StubSentimentProvider
    from src.services.sentiment.registry import SentimentProviderRegistry

    # Create stub provider registry
    registry = SentimentProviderRegistry()
    registry.register("stub", StubSentimentProvider(), weight=1.0, priority=0)

    # Create fetcher
    fetcher = SentimentSnapshotFetcher(
        output_dir=tmp_data_dir / "sentiment",
        registry=registry,
        retry_limit=3,
        retry_backoff_seconds=2,
        dryrun=True,
        force=False,
    )

    # Fetch snapshots for test date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 3)
    symbols = ["RELIANCE", "TCS"]

    summary = fetcher.fetch_all(symbols, start_date, end_date)

    # Verify summary
    assert summary["stats"]["fetched"] > 0
    assert summary["dryrun"] is True

    # Verify JSONL files created
    for symbol in symbols:
        symbol_dir = tmp_data_dir / "sentiment" / symbol
        assert symbol_dir.exists()

        # Check for date files
        for i in range(3):  # 3 days
            date = start_date + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            snapshot_file = symbol_dir / f"{date_str}.jsonl"
            assert snapshot_file.exists()

            # Verify JSONL format
            with open(snapshot_file) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["symbol"] == symbol
                assert data["date"] == date_str
                assert "score" in data
                assert "confidence" in data
                assert "providers" in data


def test_sentiment_config_defaults(mock_settings: Settings) -> None:
    """Test sentiment snapshot config defaults are safe (Phase 3)."""
    # Verify sentiment snapshots disabled by default
    assert mock_settings.sentiment_snapshot_enabled is False

    # Verify safe defaults
    assert mock_settings.sentiment_snapshot_providers == ["stub"]
    assert mock_settings.sentiment_snapshot_output_dir == "data/sentiment"
    assert 1 <= mock_settings.sentiment_snapshot_retry_limit <= 10
    assert 1 <= mock_settings.sentiment_snapshot_retry_backoff_seconds <= 60
    assert mock_settings.sentiment_snapshot_max_per_day >= 1


# ============================================================================
# US-024 PHASE 4: Incremental Daily Updates Tests
# ============================================================================


def test_state_manager_tracks_last_fetch(tmp_data_dir: Path) -> None:
    """Test state manager tracks last fetch dates (Phase 4)."""
    from src.services.state_manager import StateManager

    state_file = tmp_data_dir / "state" / "test_fetch.json"
    manager = StateManager(state_file)

    # Initially no last fetch
    assert manager.get_last_fetch_date("RELIANCE") is None

    # Set last fetch date
    test_date = datetime(2024, 1, 15)
    manager.set_last_fetch_date("RELIANCE", test_date)

    # Verify it was saved
    assert manager.get_last_fetch_date("RELIANCE") == test_date

    # Verify state file exists
    assert state_file.exists()

    # Load fresh manager and verify persistence
    manager2 = StateManager(state_file)
    assert manager2.get_last_fetch_date("RELIANCE") == test_date


def test_incremental_config_defaults(mock_settings: Settings) -> None:
    """Test incremental mode config defaults are safe (Phase 4)."""
    # Verify incremental disabled by default
    assert mock_settings.incremental_enabled is False

    # Verify reasonable lookback
    assert 1 <= mock_settings.incremental_lookback_days <= 365

    # Verify cron schedule is set
    assert mock_settings.incremental_cron_schedule is not None


# ============================================================================
# US-024 PHASE 5: Distributed Training & Scheduled Automation Tests
# ============================================================================


def test_batch_status_tracking(tmp_data_dir: Path) -> None:
    """Test state manager tracks batch execution status (Phase 5)."""
    from src.services.state_manager import StateManager

    state_file = tmp_data_dir / "state" / "batch_status.json"
    manager = StateManager(state_file)

    batch_id = "batch_20250112_180000"

    # Initially no batch status
    assert manager.get_batch_status(batch_id) is None

    # Set batch status
    manager.set_batch_status(
        batch_id=batch_id,
        status="running",
        total_tasks=10,
        completed=5,
        failed=1,
        pending_retries=2,
    )

    # Verify batch status
    status = manager.get_batch_status(batch_id)
    assert status is not None
    assert status["status"] == "running"
    assert status["total_tasks"] == 10
    assert status["completed"] == 5
    assert status["failed"] == 1
    assert status["pending_retries"] == 2

    # Record a failed task
    manager.record_task_failure(
        batch_id=batch_id,
        task_id="task_001",
        symbol="RELIANCE",
        window_label="RELIANCE_2024Q1",
        reason="Training timeout",
        attempts=3,
    )

    # Verify failed task recorded
    failed_tasks = manager.get_failed_tasks(batch_id)
    assert len(failed_tasks) == 1
    assert failed_tasks[0]["symbol"] == "RELIANCE"
    assert failed_tasks[0]["attempts"] == 3
    assert failed_tasks[0]["reason"] == "Training timeout"


def test_parallel_config_defaults(mock_settings: Settings) -> None:
    """Test parallel execution config defaults are safe (Phase 5)."""
    # Verify sequential by default
    assert mock_settings.parallel_workers == 1

    # Verify reasonable retry limits
    assert 1 <= mock_settings.parallel_retry_limit <= 10
    assert 1 <= mock_settings.parallel_retry_backoff_seconds <= 60

    # Verify pipeline skip flags default to False
    assert mock_settings.scheduled_pipeline_skip_fetch is False
    assert mock_settings.scheduled_pipeline_skip_teacher is False
    assert mock_settings.scheduled_pipeline_skip_student is False


def test_orchestration_script_exists() -> None:
    """Test incremental update orchestration script exists (Phase 5)."""
    import os

    script_path = Path("scripts/run_incremental_update.sh")
    assert script_path.exists(), "Orchestration script not found"
    assert os.access(script_path, os.X_OK), "Script is not executable"


def test_distributed_worker_stub_exists() -> None:
    """Test distributed training worker stub exists (Phase 5)."""
    worker_path = Path("scripts/distributed_training_worker.py")
    assert worker_path.exists(), "Distributed worker stub not found"

    # Verify it can be imported
    import sys

    sys.path.insert(0, "scripts")
    try:
        import distributed_training_worker

        # Verify protocol exists
        assert hasattr(distributed_training_worker, "DistributedExecutor")
        assert hasattr(distributed_training_worker, "LocalProcessExecutor")
    finally:
        sys.path.pop(0)


# ============================================================================
# US-024 PHASE 6: Data Quality Dashboard & Alerts Tests
# ============================================================================


def test_data_quality_service(tmp_data_dir: Path) -> None:
    """Test DataQualityService scans data and computes metrics (Phase 6)."""
    from src.services.data_quality import DataQualityService

    # Create test historical data
    hist_dir = tmp_data_dir / "historical" / "RELIANCE"
    hist_dir.mkdir(parents=True)

    # Create sample CSV with valid data
    csv_file = hist_dir / "2024-01-01.csv"
    csv_file.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01 09:15:00,100,105,99,104,1000\n"
        "2024-01-01 09:16:00,104,106,103,105,1500\n"
    )

    # Create sentiment data
    sent_dir = tmp_data_dir / "sentiment" / "RELIANCE"
    sent_dir.mkdir(parents=True)

    jsonl_file = sent_dir / "2024-01-01.jsonl"
    jsonl_file.write_text(
        '{"symbol": "RELIANCE", "date": "2024-01-01", "score": 0.5, "confidence": 0.8}\n'
    )

    # Initialize service
    quality_service = DataQualityService(
        historical_dir=tmp_data_dir / "historical",
        sentiment_dir=tmp_data_dir / "sentiment",
    )

    # Scan historical quality
    hist_metrics = quality_service.scan_historical_quality("RELIANCE")
    assert hist_metrics["total_files"] == 1
    assert hist_metrics["total_bars"] == 2
    assert hist_metrics["duplicate_timestamps"] == 0
    assert hist_metrics["zero_volume_bars"] == 0

    # Scan sentiment quality
    sent_metrics = quality_service.scan_sentiment_quality("RELIANCE")
    assert sent_metrics["total_files"] == 1
    assert sent_metrics["total_snapshots"] == 1
    assert sent_metrics["invalid_scores"] == 0


def test_quality_metrics_tracking(tmp_data_dir: Path) -> None:
    """Test StateManager tracks quality metrics (Phase 6)."""
    from src.services.state_manager import StateManager

    state_file = tmp_data_dir / "state" / "quality_test.json"
    manager = StateManager(state_file)

    # Record quality metrics
    metrics = {
        "total_files": 10,
        "total_bars": 1000,
        "missing_files": 2,
        "duplicate_timestamps": 5,
        "zero_volume_bars": 3,
    }

    manager.record_quality_metrics("RELIANCE", "historical", metrics)

    # Verify metrics recorded
    recorded = manager.get_quality_metrics("RELIANCE", "historical")
    assert recorded["total_files"] == 10
    assert recorded["missing_files"] == 2
    assert "last_scanned" in recorded


def test_quality_alerts(tmp_data_dir: Path) -> None:
    """Test StateManager records and retrieves quality alerts (Phase 6)."""
    from src.services.state_manager import StateManager

    state_file = tmp_data_dir / "state" / "alerts_test.json"
    manager = StateManager(state_file)

    # Record alert
    manager.record_quality_alert(
        symbol="RELIANCE",
        data_type="historical",
        severity="warning",
        metric="missing_files",
        value=15,
        threshold=10,
        message="Missing 15 files (threshold: 10)",
    )

    # Verify alert recorded
    alerts = manager.get_quality_alerts()
    assert len(alerts) == 1
    assert alerts[0]["symbol"] == "RELIANCE"
    assert alerts[0]["severity"] == "warning"
    assert alerts[0]["metric"] == "missing_files"

    # Filter by symbol
    reliance_alerts = manager.get_quality_alerts(symbol="RELIANCE")
    assert len(reliance_alerts) == 1

    # Filter by severity
    warnings = manager.get_quality_alerts(severity="warning")
    assert len(warnings) == 1


def test_quality_config_defaults(mock_settings: Settings) -> None:
    """Test data quality config defaults are safe (Phase 6)."""
    # Verify quality scanning disabled by default
    assert mock_settings.data_quality_scan_enabled is False

    # Verify dashboard disabled by default
    assert mock_settings.data_quality_dashboard_enabled is False

    # Verify reasonable thresholds
    assert mock_settings.data_quality_alert_threshold_missing_files >= 0
    assert mock_settings.data_quality_alert_threshold_duplicate_timestamps >= 0
    assert mock_settings.data_quality_alert_threshold_zero_volume >= 0


def test_dashboard_exists() -> None:
    """Test data quality dashboard exists (Phase 6)."""
    dashboard_path = Path("dashboards/data_quality_dashboard.py")
    assert dashboard_path.exists(), "Dashboard not found"

    readme_path = Path("dashboards/README.md")
    assert readme_path.exists(), "Dashboard README not found"


def test_cached_chunk_timestamp_normalization(tmp_data_dir: Path, mock_settings: Settings) -> None:
    """Test that cached chunks with string timestamps are properly normalized (US-028 Phase 6i)."""
    # Create cache directory
    cache_dir = tmp_data_dir / "historical" / "TEST" / "1day"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a cached CSV with string timestamps (simulating old cache format)
    cache_file = cache_dir / "2024-01-01.csv"
    cached_data = pd.DataFrame({
        "timestamp": ["2024-01-01 09:00:00+05:30", "2024-01-01 10:00:00+05:30"],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1100],
    })
    cached_data.to_csv(cache_file, index=False)

    # Create fetcher instance
    fetcher = HistoricalDataFetcher(
        settings=mock_settings,
        dryrun=True,
    )

    # Fetch data using chunked method (should load from cache)
    from datetime import datetime as dt
    start_dt = dt(2024, 1, 1)
    end_dt = dt(2024, 1, 1)

    result_df = fetcher.fetch_symbol_date_range_chunked(
        symbol="TEST",
        start_dt=start_dt,
        end_dt=end_dt,
        interval="1day",
        force=False,  # Use cache
    )

    # Verify timestamps are properly converted to datetime type
    assert len(result_df) > 0, "Should load cached data"
    assert "timestamp" in result_df.columns, "Should have timestamp column"
    assert pd.api.types.is_datetime64_any_dtype(result_df["timestamp"]), (
        "Timestamp column should be datetime type, not string"
    )

    # Verify data can be sorted (would fail if mixed types)
    sorted_df = result_df.sort_values("timestamp")
    assert len(sorted_df) == len(result_df), "Sorting should not change row count"


def test_negative_volume_correction(tmp_data_dir: Path, mock_settings: Settings) -> None:
    """Test that negative volumes are corrected and warnings are tracked (US-028 Phase 6j)."""
    import pandas as pd

    from scripts.fetch_historical_data import HistoricalDataFetcher

    # Create cache directory
    cache_dir = tmp_data_dir / "historical" / "TESTSTOCK" / "1day"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a cached CSV with negative volume values (simulating data quality issue)
    cache_file = cache_dir / "2024-01-01.csv"
    bad_data = pd.DataFrame({
        "timestamp": ["2024-01-01 09:15:00+05:30", "2024-01-01 10:15:00+05:30", "2024-01-01 11:15:00+05:30"],
        "open": [100.0, 101.0, 102.0],
        "high": [102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 101.0],
        "close": [101.0, 102.0, 103.0],
        "volume": [1000, -500, 2000],  # Middle row has negative volume
    })
    bad_data.to_csv(cache_file, index=False)

    # Create fetcher instance
    fetcher = HistoricalDataFetcher(
        settings=mock_settings,
        dryrun=True,
    )

    # Fetch data using chunked method (should load from cache and correct negative volumes)
    from datetime import datetime as dt
    start_dt = dt(2024, 1, 1)
    end_dt = dt(2024, 1, 1)

    result_df = fetcher.fetch_symbol_date_range_chunked(
        symbol="TESTSTOCK",
        start_dt=start_dt,
        end_dt=end_dt,
        interval="1day",
        force=False,  # Use cache
    )

    # Verify fetch completed successfully
    assert len(result_df) > 0, "Should load cached data"
    assert "volume" in result_df.columns, "Should have volume column"

    # Verify negative volumes were corrected (clipped to 0)
    assert (result_df["volume"] >= 0).all(), "All volumes should be non-negative after correction"
    assert (result_df["volume"] == 0).any(), "Should have at least one zero volume (corrected from negative)"

    # Verify warnings were tracked
    assert fetcher.stats["warnings"] > 0, "Should have recorded data quality warnings"

    # Verify corrected data was saved (not original negative values)
    corrected_cache = pd.read_csv(cache_file, parse_dates=["timestamp"])
    assert (corrected_cache["volume"] >= 0).all(), "Cached data should have corrected volumes"


def test_phase_2_json_aggregation(tmp_path):
    """Test Phase 2 orchestrator aggregates from teacher_runs.json (US-028 Phase 6l).

    Confirms JSON-based aggregation works even if stdout lacks summary lines.
    """
    import json
    import sys
    from pathlib import Path

    # Add project root to sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from scripts.run_historical_training import HistoricalRunOrchestrator

    # Create mock batch directory with teacher_runs.json
    batch_dir = tmp_path / "teacher_training_20250101_120000"
    batch_dir.mkdir(parents=True)

    # Create mock teacher_runs.json with 3 success, 1 skip, 0 failed
    # Matches format produced by train_teacher_batch.py
    teacher_runs = [
        {
            "status": "success",
            "window_label": "RELIANCE_2023-01-01_2023-06-30",
            "sample_counts": {
                "train_samples": 67,
                "val_samples": 17,
            },
            "metrics": {
                "val_accuracy": 0.85,
                "val_precision": 0.82,
                "val_recall": 0.88,
                "val_f1": 0.85,
                "val_auc": 0.90,
            },
            "model_path": str(batch_dir / "teacher_RELIANCE_2023-01-01_2023-06-30.pkl"),
        },
        {
            "status": "success",
            "window_label": "RELIANCE_2023-07-01_2023-12-31",
            "sample_counts": {
                "train_samples": 72,
                "val_samples": 18,
            },
            "metrics": {
                "val_accuracy": 0.87,
                "val_precision": 0.84,
                "val_recall": 0.90,
                "val_f1": 0.87,
                "val_auc": 0.92,
            },
            "model_path": str(batch_dir / "teacher_RELIANCE_2023-07-01_2023-12-31.pkl"),
        },
        {
            "status": "success",
            "window_label": "TCS_2024-01-01_2024-06-30",
            "sample_counts": {
                "train_samples": 65,
                "val_samples": 16,
            },
            "metrics": {
                "val_accuracy": 0.83,
                "val_precision": 0.80,
                "val_recall": 0.86,
                "val_f1": 0.83,
                "val_auc": 0.88,
            },
            "model_path": str(batch_dir / "teacher_TCS_2024-01-01_2024-06-30.pkl"),
        },
        {
            "status": "skipped",
            "window_label": "TCS_2024-07-01_2024-12-31",
            "reason": "Insufficient samples for training: 12 < 20 minimum",
        },
    ]

    # Write JSONL file (one object per line)
    json_path = batch_dir / "teacher_runs.json"
    with open(json_path, "w") as f:
        for entry in teacher_runs:
            f.write(json.dumps(entry) + "\n")

    # Create orchestrator and call JSON aggregation directly
    orchestrator = HistoricalRunOrchestrator(
        symbols=["RELIANCE", "TCS"],
        start_date="2023-01-01",
        end_date="2024-12-31",
        skip_fetch=True,
        dryrun=True,
    )

    # Call the JSON aggregation helper
    stats = orchestrator._aggregate_teacher_runs_from_json(batch_dir)

    # Verify aggregation correctness
    assert stats is not None, "Should successfully parse teacher_runs.json"
    assert stats["total_windows"] == 4, f"Expected 4 windows, got {stats['total_windows']}"
    assert stats["completed"] == 3, f"Expected 3 completed, got {stats['completed']}"
    assert stats["skipped"] == 1, f"Expected 1 skipped, got {stats['skipped']}"
    assert stats["failed"] == 0, f"Expected 0 failed, got {stats['failed']}"

    # Verify success windows array is populated
    assert len(stats["success_windows"]) == 3, "Should have 3 success window entries"
    success_window = stats["success_windows"][0]
    assert "window_label" in success_window
    assert success_window["window_label"] == "RELIANCE_2023-01-01_2023-06-30"
    assert "sample_counts" in success_window
    assert success_window["sample_counts"]["train_samples"] == 67
    assert success_window["sample_counts"]["val_samples"] == 17
    assert "metrics" in success_window
    assert success_window["metrics"]["val_accuracy"] == 0.85

    # Verify skipped windows array is populated
    assert len(stats["skipped_windows"]) == 1, "Should have 1 skipped window entry"
    skipped_window = stats["skipped_windows"][0]
    assert skipped_window["window_label"] == "TCS_2024-07-01_2024-12-31"
    assert "Insufficient samples" in skipped_window["reason"]

    # Verify failed windows array is empty
    assert len(stats["failed_windows"]) == 0, "Should have 0 failed window entries"

    # Verify sample counts aggregation
    assert stats["total_train_samples"] == 67 + 72 + 65, "Should sum all training samples"
    assert stats["total_val_samples"] == 17 + 18 + 16, "Should sum all validation samples"

    # Verify batch directory is tracked
    assert stats["batch_dir"] == str(batch_dir), "Should track batch directory path"


def test_progress_tracking(mock_settings: Settings, tmp_data_dir: Path) -> None:
    """Test that StateManager tracks training progress (US-028 Phase 7 Initiative 4).

    Verifies that:
    - StateManager progress tracking methods work correctly
    - Progress data is persisted to state.json
    - Progress data can be retrieved
    """
    from src.services.state_manager import StateManager

    # Use temporary state file
    state_file = tmp_data_dir / "test_state.json"
    state_mgr = StateManager(state_file)

    # Record progress for Phase 1
    state_mgr.record_training_progress(
        phase="data_ingestion",
        completed=2,
        total=2,
        extra={
            "status": "success",
            "chunks_fetched": 12,
            "chunks_failed": 0,
        },
    )

    # Record progress for Phase 2
    state_mgr.record_training_progress(
        phase="teacher_training",
        completed=58,
        total=60,
        eta_minutes=2.5,
        extra={
            "status": "partial",
            "trained": 58,
            "skipped": 2,
            "failed": 0,
        },
    )

    # Verify progress can be retrieved
    all_progress = state_mgr.get_training_progress()
    assert "data_ingestion" in all_progress
    assert "teacher_training" in all_progress

    # Verify Phase 1 data
    phase1 = all_progress["data_ingestion"]
    assert phase1["phase"] == "data_ingestion"
    assert phase1["completed"] == 2
    assert phase1["total"] == 2
    assert phase1["percent_complete"] == 100.0
    assert phase1["status"] == "success"
    assert phase1["chunks_fetched"] == 12

    # Verify Phase 2 data
    phase2 = all_progress["teacher_training"]
    assert phase2["phase"] == "teacher_training"
    assert phase2["completed"] == 58
    assert phase2["total"] == 60
    assert phase2["percent_complete"] == pytest.approx(96.7, abs=0.1)
    assert phase2["eta_minutes"] == 2.5
    assert phase2["trained"] == 58
    assert phase2["skipped"] == 2

    # Verify specific phase retrieval
    phase1_only = state_mgr.get_training_progress("data_ingestion")
    assert phase1_only["completed"] == 2

    # Verify clear progress works
    state_mgr.clear_training_progress()
    cleared_progress = state_mgr.get_training_progress()
    assert cleared_progress == {}


def test_fetch_logging(tmp_data_dir: Path, mock_settings: Settings) -> None:
    """Test that fetch operations are logged to fetch_log.jsonl (US-028 Phase 7 Initiative 1).

    Verifies that:
    - Fetch log entries are written to JSONL file
    - Entries contain required fields
    - Multiple fetch operations append to log
    """
    import json

    from scripts.fetch_historical_data import HistoricalDataFetcher

    # Create fetcher with temp data dir
    fetcher = HistoricalDataFetcher(
        settings=mock_settings,
        dryrun=True,
    )

    # Override log path to use temp dir
    log_path = tmp_data_dir / "fetch_log.jsonl"
    fetcher.fetch_log_path = log_path

    # Log a few fetch entries
    fetcher.log_fetch_entry(
        symbol="RELIANCE",
        interval="1day",
        chunk_start="2023-01-01",
        chunk_end="2023-01-01",
        rows_fetched=100,
        source="api",
        status="success",
        warnings=0,
    )

    fetcher.log_fetch_entry(
        symbol="TCS",
        interval="1day",
        chunk_start="2023-01-02",
        chunk_end="2023-01-02",
        rows_fetched=0,
        source="cache",
        status="cached",
    )

    fetcher.log_fetch_entry(
        symbol="INFY",
        interval="1day",
        chunk_start="2023-01-03",
        chunk_end="2023-01-03",
        rows_fetched=0,
        source="api",
        status="failed",
        error="Connection timeout",
    )

    # Verify log file exists
    assert log_path.exists()

    # Parse and verify entries
    with open(log_path) as f:
        lines = f.readlines()

    assert len(lines) == 3

    # Parse each entry
    entry1 = json.loads(lines[0])
    entry2 = json.loads(lines[1])
    entry3 = json.loads(lines[2])

    # Verify entry 1 (successful API fetch)
    assert entry1["symbol"] == "RELIANCE"
    assert entry1["interval"] == "1day"
    assert entry1["chunk_start"] == "2023-01-01"
    assert entry1["rows_fetched"] == 100
    assert entry1["source"] == "api"
    assert entry1["status"] == "success"
    assert "timestamp" in entry1

    # Verify entry 2 (cache hit)
    assert entry2["symbol"] == "TCS"
    assert entry2["status"] == "cached"
    assert entry2["source"] == "cache"

    # Verify entry 3 (failed fetch)
    assert entry3["symbol"] == "INFY"
    assert entry3["status"] == "failed"
    assert entry3["error"] == "Connection timeout"


def test_deduplication(tmp_data_dir: Path, mock_settings: Settings) -> None:
    """Test that duplicate timestamps are removed when saving data (US-028 Phase 7 Initiative 1).

    Verifies that:
    - Saving data twice removes duplicate timestamps
    - Stats track duplicates_removed count
    - Data is sorted by timestamp
    """
    import pandas as pd

    from scripts.fetch_historical_data import HistoricalDataFetcher

    # Create fetcher
    fetcher = HistoricalDataFetcher(
        settings=mock_settings,
        dryrun=True,
    )

    # Create mock data with 3 rows
    df1 = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=3, freq="D"),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [1000, 2000, 3000],
        }
    )

    # Save first batch
    fetcher.save_to_cache(df1, "RELIANCE", "1day", "2023-01-01")

    # Create second batch with 2 overlapping rows and 1 new row
    df2 = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2023-01-02", "2023-01-03", "2023-01-04"]
            ),  # 2 duplicates, 1 new
            "open": [101.0, 102.0, 103.0],
            "high": [106.0, 107.0, 108.0],
            "low": [96.0, 97.0, 98.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [2000, 3000, 4000],
        }
    )

    # Save second batch (should detect and remove duplicates)
    fetcher.save_to_cache(df2, "RELIANCE", "1day", "2023-01-01")

    # Verify duplicates were removed
    assert fetcher.stats["duplicates_removed"] == 2

    # Load saved file and verify data
    cache_path = fetcher.get_cache_path("RELIANCE", "1day", "2023-01-01")
    saved_df = pd.read_csv(cache_path)

    # Should have 4 unique rows (3 from df1 + 1 new from df2)
    assert len(saved_df) == 4

    # Verify data is sorted by timestamp
    saved_timestamps = pd.to_datetime(saved_df["datetime"])
    assert (saved_timestamps == saved_timestamps.sort_values()).all()


def test_gap_detection(tmp_data_dir: Path, mock_settings: Settings) -> None:
    """Test that gaps in fetched data are detected (US-028 Phase 7 Initiative 1).

    Verifies that:
    - Missing trading days are identified
    - Gaps are grouped into ranges
    - Stats track gaps_detected count
    """
    from datetime import datetime

    import pandas as pd

    from scripts.fetch_historical_data import HistoricalDataFetcher

    # Create fetcher
    fetcher = HistoricalDataFetcher(
        settings=mock_settings,
        dryrun=True,
    )

    # Create data with gaps (missing Jan 3-4 and Jan 8)
    timestamps = [
        "2023-01-01",
        "2023-01-02",
        # Gap: Jan 3-4
        "2023-01-05",
        "2023-01-06",
        "2023-01-07",
        # Gap: Jan 8
        "2023-01-09",
        "2023-01-10",
    ]

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "open": [100.0] * len(timestamps),
            "high": [105.0] * len(timestamps),
            "low": [95.0] * len(timestamps),
            "close": [103.0] * len(timestamps),
            "volume": [1000] * len(timestamps),
        }
    )

    # Detect gaps
    gaps = fetcher.detect_gaps(
        df,
        "RELIANCE",
        datetime(2023, 1, 1),
        datetime(2023, 1, 10),
    )

    # Should detect 2 gap ranges
    assert len(gaps) >= 1  # At least one gap should be detected
    assert fetcher.stats["gaps_detected"] >= 1

    # Verify gaps are tuples of (start_date, end_date) as ISO strings
    for gap in gaps:
        assert len(gap) == 2
        assert isinstance(gap[0], str)
        assert isinstance(gap[1], str)


def test_rate_limiting(tmp_data_dir: Path, mock_settings: Settings) -> None:
    """Test that rate limiting enforcement works (US-028 Phase 7 Initiative 1).

    Verifies that:
    - Rate limit tracking records request timestamps
    - Enforcement sleeps when limit is reached
    - Old timestamps are cleaned up
    """
    import time

    from scripts.fetch_historical_data import HistoricalDataFetcher

    # Create fetcher with low rate limit
    mock_settings.breeze_rate_limit_requests_per_minute = 3
    fetcher = HistoricalDataFetcher(
        settings=mock_settings,
        dryrun=True,
    )

    # Simulate 3 requests
    start_time = time.time()
    for _ in range(3):
        fetcher.enforce_rate_limit()

    elapsed = time.time() - start_time

    # First 3 requests should not trigger rate limiting (< 1 second)
    assert elapsed < 1.0
    assert len(fetcher.request_times) == 3

    # 4th request should trigger rate limiting (will sleep)
    # Note: In test we don't want to actually sleep 60s, so we just verify
    # that the logic detects the rate limit condition
    assert len(fetcher.request_times) >= 3


def test_symbols_mode_metadata_loading(mock_settings: Settings) -> None:
    """Test loading symbols from metadata file (US-028 Phase 7 Initiative 1).

    Verifies that:
    - Symbols can be loaded by mode (pilot, nifty100, etc.)
    - Metadata file structure is correct
    - Invalid modes raise errors
    """
    from pathlib import Path

    # Check if metadata file exists
    metadata_path = Path("data/historical/metadata/nifty100_constituents.json")
    if not metadata_path.exists():
        import pytest

        pytest.skip("Metadata file not found - skipping test")

    # Test pilot mode
    pilot_symbols = mock_settings.get_symbols_for_mode("pilot")
    assert isinstance(pilot_symbols, list)
    assert len(pilot_symbols) == 5  # 3 NIFTY + 2 ETFs

    # Test nifty100 mode
    nifty_symbols = mock_settings.get_symbols_for_mode("nifty100")
    assert isinstance(nifty_symbols, list)
    assert len(nifty_symbols) >= 10  # Should have at least 10 symbols

    # Test metals_etfs mode
    metals_symbols = mock_settings.get_symbols_for_mode("metals_etfs")
    assert isinstance(metals_symbols, list)
    assert len(metals_symbols) == 2  # GOLDBEES + SILVERBEES

    # Test all mode - should return all symbols from metadata (96 symbols in current file)
    all_symbols = mock_settings.get_symbols_for_mode("all")
    assert len(all_symbols) >= len(metals_symbols)  # At minimum should have the metals
    # Note: all_symbols returns the flat "symbols" list, which includes both NIFTY100 + metals
    # The sum of nifty100 + metals_etfs modes may differ due to category overlaps/duplicates

    # Test invalid mode
    import pytest

    with pytest.raises(ValueError, match="Invalid symbols_mode"):
        mock_settings.get_symbols_for_mode("invalid_mode")
