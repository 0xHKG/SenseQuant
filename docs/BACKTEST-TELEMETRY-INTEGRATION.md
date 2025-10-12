# Backtester Telemetry Integration

## Overview

This document describes the telemetry capture integration in the Backtester service at `/home/gogi/Desktop/SenseQuant/src/services/backtester.py`.

Telemetry capture allows the backtester to record detailed prediction traces for each closed trade, enabling comprehensive accuracy analysis and model performance evaluation.

## Changes Made

### 1. Imports and Dependencies

Added imports for telemetry support:
```python
import random
from src.services.accuracy_analyzer import PredictionTrace, TelemetryWriter
```

### 2. Constructor (`__init__`) Updates

Added two new optional parameters:

- **`enable_telemetry: bool = False`**: Flag to enable/disable telemetry capture
- **`telemetry_dir: Path | None = None`**: Custom directory for telemetry files (defaults to settings path)

The constructor now:
1. Initializes a `TelemetryWriter` instance if telemetry is enabled
2. Uses settings configuration for compression, buffer size, and file size limits
3. Logs telemetry status (enabled/disabled)
4. Seeds the random module for deterministic sampling

```python
self.telemetry_writer: TelemetryWriter | None = None

if self.enable_telemetry:
    output_dir = telemetry_dir or Path(self.settings.telemetry_storage_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    self.telemetry_writer = TelemetryWriter(
        output_dir=output_dir,
        format="csv",
        compression=self.settings.telemetry_compression,
        buffer_size=self.settings.telemetry_buffer_size,
        max_file_size_mb=self.settings.telemetry_max_file_size_mb,
    )
```

### 3. Trade Closing (`_close_swing_position`) Updates

The `_close_swing_position` method now captures telemetry traces when trades are closed.

**Key features:**
- Respects the `telemetry_sample_rate` setting (0.0 = 0%, 1.0 = 100%)
- Calculates actual direction based on realized return percentage:
  - Return > 0.5%: `actual_direction = "LONG"`
  - Return < -0.5%: `actual_direction = "SHORT"`
  - Otherwise: `actual_direction = "NOOP"`
- Captures comprehensive metadata including:
  - Exit reason (stop-loss, take-profit, max hold, etc.)
  - Fee information (entry, exit, total)
  - Position value and gross PnL
  - Stop-loss and take-profit hit flags

**Sample rate logic:**
```python
should_capture = (
    self.settings.telemetry_sample_rate >= 1.0
    or (
        self.settings.telemetry_sample_rate > 0.0
        and random.random() < self.settings.telemetry_sample_rate
    )
)
```

**Trace structure:**
```python
trace = PredictionTrace(
    timestamp=position.entry_date.to_pydatetime(),
    symbol=symbol,
    strategy=self.config.strategy,
    predicted_direction=position.direction,
    actual_direction=actual_direction,
    predicted_confidence=0.5,  # Default for backtests
    entry_price=position.entry_price,
    exit_price=exit_price,
    holding_period_minutes=holding_period_minutes,
    realized_return_pct=realized_return_pct,
    features={},  # Empty for now, can be enhanced
    metadata=trace_metadata,
)
```

### 4. Backtest Completion (`run`) Updates

The `run()` method now:
1. Flushes the telemetry writer buffer
2. Closes the telemetry writer properly
3. Handles errors gracefully with try-except

```python
if self.enable_telemetry and self.telemetry_writer is not None:
    try:
        self.telemetry_writer.flush()
        self.telemetry_writer.close()
        logger.info("Telemetry flushed and closed")
    except Exception as e:
        logger.error(f"Failed to close telemetry writer: {e}")
```

## Configuration

Telemetry behavior is controlled via the `Settings` class in `/home/gogi/Desktop/SenseQuant/src/app/config.py`:

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `telemetry_storage_path` | `str` | `"data/analytics"` | Base directory for telemetry files |
| `telemetry_sample_rate` | `float` | `1.0` | Sampling rate (0.0-1.0) |
| `telemetry_compression` | `bool` | `False` | Enable gzip compression |
| `telemetry_buffer_size` | `int` | `100` | Buffer size before flushing |
| `telemetry_max_file_size_mb` | `int` | `100` | Max file size before rotation |

## Usage Examples

### Basic Usage (Telemetry Disabled - Default)

```python
from src.domain.types import BacktestConfig
from src.services.backtester import Backtester

config = BacktestConfig(
    symbols=["RELIANCE"],
    start_date="2024-01-01",
    end_date="2024-03-31",
    strategy="swing",
    initial_capital=1_000_000.0,
    data_source="breeze",
    random_seed=42,
)

# Telemetry is disabled by default (backward compatible)
backtester = Backtester(config=config, client=breeze_client)
result = backtester.run()
```

### With Telemetry Enabled

```python
from pathlib import Path
from src.domain.types import BacktestConfig
from src.services.backtester import Backtester

config = BacktestConfig(
    symbols=["RELIANCE", "TCS"],
    start_date="2024-01-01",
    end_date="2024-03-31",
    strategy="swing",
    initial_capital=1_000_000.0,
    data_source="breeze",
    random_seed=42,
)

# Enable telemetry with custom directory
telemetry_dir = Path("data/telemetry/backtest_2024")
backtester = Backtester(
    config=config,
    client=breeze_client,
    enable_telemetry=True,
    telemetry_dir=telemetry_dir,
)

result = backtester.run()

# Telemetry files will be created in data/telemetry/backtest_2024/
# Files follow format: predictions_YYYYMMDD_HHMMSS_####.csv[.gz]
```

### Analyzing Telemetry Data

```python
from pathlib import Path
from src.services.accuracy_analyzer import AccuracyAnalyzer

# Load telemetry traces
analyzer = AccuracyAnalyzer()
telemetry_dir = Path("data/telemetry/backtest_2024")

# Load all trace files
traces = []
for telemetry_file in telemetry_dir.glob("predictions_*.csv*"):
    traces.extend(analyzer.load_traces(telemetry_file))

print(f"Loaded {len(traces)} prediction traces")

# Compute accuracy metrics
metrics = analyzer.compute_metrics(traces)

print(f"Hit ratio: {metrics.hit_ratio:.2%}")
print(f"Win rate: {metrics.win_rate:.2%}")
print(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")

# Export reports
analyzer.export_report(metrics, Path("reports/accuracy.json"))
analyzer.plot_confusion_matrix(metrics, Path("reports/confusion_matrix.png"))
analyzer.plot_return_distribution(traces, Path("reports/returns.png"))
```

## Testing

Comprehensive integration tests have been added in `/home/gogi/Desktop/SenseQuant/tests/integration/test_backtest_telemetry.py`:

### Test Coverage

1. **`test_backtest_telemetry_capture`**: Verifies telemetry files are created and traces are captured
2. **`test_backtest_telemetry_disabled`**: Ensures no telemetry when disabled
3. **`test_backtest_telemetry_sample_rate`**: Validates sample rate behavior (0% = no traces)
4. **`test_backtest_telemetry_backward_compatibility`**: Confirms optional parameters don't break existing code
5. **`test_backtest_telemetry_trace_contents`**: Validates trace structure and metadata

### Running Tests

```bash
# Run all telemetry tests
pytest tests/integration/test_backtest_telemetry.py -v

# Run all backtest tests (including legacy tests)
pytest tests/integration/test_backtest_pipeline.py -v
pytest tests/unit/test_backtester.py -v
```

All existing tests pass without modification, ensuring backward compatibility.

## Trace Schema

Each `PredictionTrace` contains:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `datetime` | Entry timestamp |
| `symbol` | `str` | Stock symbol |
| `strategy` | `str` | Strategy name (e.g., "swing") |
| `predicted_direction` | `str` | "LONG", "SHORT" |
| `actual_direction` | `str` | "LONG", "SHORT", "NOOP" |
| `predicted_confidence` | `float` | Confidence score (0.0-1.0) |
| `entry_price` | `float` | Entry price |
| `exit_price` | `float` | Exit price |
| `holding_period_minutes` | `int` | Hold duration in minutes |
| `realized_return_pct` | `float` | Actual return percentage |
| `features` | `dict[str, float]` | Feature values (empty for now) |
| `metadata` | `dict[str, Any]` | Additional context |

### Metadata Fields

- `exit_reason`: Reason for exit (e.g., "stop_loss", "take_profit", "max_hold")
- `sl_hit`: Stop-loss hit flag (bool)
- `tp_hit`: Take-profit hit flag (bool)
- `max_hold_hit`: Max hold period hit flag (bool)
- `entry_fees`: Entry transaction fees
- `exit_fees`: Exit transaction fees
- `total_fees`: Total fees
- `position_value`: Total position value at entry
- `gross_pnl`: PnL before fees

## Performance Considerations

1. **Memory Usage**: Traces are buffered in memory until the buffer size is reached
2. **Disk I/O**: Writes are batched and atomic to prevent corruption
3. **File Rotation**: Automatic rotation when file size exceeds threshold
4. **Sampling**: Use `telemetry_sample_rate < 1.0` to reduce overhead for large backtests

## Future Enhancements

Potential improvements for future iterations:

1. **Feature Capture**: Populate the `features` field with actual feature values from the strategy
2. **Intraday Support**: Extend telemetry capture to intraday strategies
3. **Real-time Streaming**: Support streaming telemetry to external systems
4. **Enhanced Confidence**: Calculate actual confidence scores from strategy signals
5. **Async Writes**: Use async I/O for better performance
6. **Compression Options**: Support additional compression formats (zstd, lz4)

## Error Handling

The integration includes comprehensive error handling:

- Telemetry failures are logged but don't interrupt the backtest
- Writer close failures are caught and logged
- Empty/invalid telemetry files are handled gracefully during loading
- Sample rate edge cases (0.0, 1.0) are handled explicitly

## Backward Compatibility

The integration is fully backward compatible:

- All parameters are optional with sensible defaults
- Existing code continues to work without modification
- Telemetry is disabled by default
- All existing tests pass without changes

## Example Script

See `/home/gogi/Desktop/SenseQuant/examples/backtest_with_telemetry.py` for a complete working example demonstrating:
- Backtest configuration with telemetry
- Running the backtest
- Loading and analyzing telemetry data
- Generating reports and visualizations
