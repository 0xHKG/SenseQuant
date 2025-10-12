# Telemetry Dashboard

Real-time monitoring dashboard for trading strategy performance using Streamlit.

## Features

- **Strategy Overview**: Precision, recall, F1, Sharpe ratio cards for intraday and swing strategies
- **Cumulative Returns**: Interactive line chart showing performance over time
- **Confusion Matrices**: Side-by-side heatmaps for prediction accuracy
- **Alert Panel**: Automatic warnings when metrics fall below thresholds
- **Auto-Refresh**: Configurable refresh interval for real-time monitoring
- **No Live API Calls**: Uses cached telemetry data only

## Installation

```bash
pip install streamlit plotly
```

## Usage

### Basic Launch

```bash
streamlit run dashboards/telemetry_dashboard.py
```

### With Custom Options

```bash
streamlit run dashboards/telemetry_dashboard.py -- \
  --telemetry-dir data/analytics \
  --refresh-interval 30 \
  --alert-precision-threshold 0.55
```

### Options

- `--telemetry-dir`: Directory containing telemetry CSV files (default: `data/analytics`)
- `--refresh-interval`: Auto-refresh interval in seconds (default: `30`)
- `--alert-precision-threshold`: Precision threshold for alerts (default: `0.55`)

## Workflow

1. **Run Backtest with Telemetry**:
   ```bash
   python scripts/backtest.py \
     --symbols RELIANCE TCS \
     --start-date 2024-01-01 \
     --end-date 2024-12-31 \
     --strategy both \
     --enable-telemetry
   ```

2. **Launch Dashboard**:
   ```bash
   streamlit run dashboards/telemetry_dashboard.py
   ```

3. **Monitor Performance**:
   - Dashboard auto-refreshes every 30 seconds (configurable)
   - Alerts appear when precision < threshold or Sharpe < 0.5
   - Use sidebar to adjust settings without restarting

## Dashboard Panels

### 1. Strategy Overview

Two cards showing key metrics:
- Precision (LONG predictions)
- Recall (LONG predictions)
- F1 Score
- Sharpe Ratio (annualized)
- Win Rate
- Average Return
- Total Trades
- Average Holding Period

### 2. Performance

Cumulative returns chart with:
- Overlaid intraday and swing performance
- Interactive zoom and pan
- Timestamp-based x-axis

### 3. Prediction Accuracy

Confusion matrices showing:
- Predicted vs actual direction (LONG/SHORT/NOOP)
- Color-coded heatmap
- Raw trade counts

### 4. Alerts & Monitoring

Real-time alerts for:
- Precision below threshold (default: 55%)
- Sharpe ratio below 0.5
- Other degradation signals

## Architecture

```
dashboards/
├── telemetry_dashboard.py    # Main Streamlit app
├── components/                # UI components (future)
│   ├── strategy_cards.py
│   ├── performance_charts.py
│   ├── confusion_heatmap.py
│   └── alert_panel.py
├── utils/                     # Utility functions (future)
│   ├── data_loader.py
│   └── alert_rules.py
└── README.md                  # This file
```

## Performance

- **Initial Load**: < 3 seconds for 10,000 traces
- **Refresh**: < 1 second (uses caching)
- **Memory**: ~50-100 MB for typical datasets

## Troubleshooting

### No Data Available

**Problem**: Dashboard shows "No telemetry data found"

**Solution**:
1. Verify telemetry directory exists: `ls data/analytics`
2. Check that backtest was run with `--enable-telemetry`
3. Ensure CSV files are present: `ls data/analytics/*.csv`

### Slow Performance

**Problem**: Dashboard loads slowly or freezes

**Solution**:
1. Increase refresh interval: `--refresh-interval 60`
2. Limit telemetry to recent data (filter by date)
3. Use compression for large datasets

### Metrics Not Updating

**Problem**: Metrics don't change after new backtest

**Solution**:
1. Click "🔄 Refresh Now" button in sidebar
2. Restart dashboard: `Ctrl+C` then re-launch
3. Check telemetry file timestamps: `ls -lt data/analytics/`

## Future Enhancements

- WebSocket streaming for real-time updates
- Multi-symbol drill-down
- Historical playback mode
- Alert notifications (email/SMS)
- Export dashboard as PDF report
- Integration with Grafana/Datadog

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [US-017 Story](../docs/stories/us-017-intraday-telemetry.md)
