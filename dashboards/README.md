# SenseQuant Dashboards

This directory contains Streamlit dashboards for monitoring and visualization.

## US-024 Phase 6: Data Quality Dashboard

### Overview

The Data Quality Dashboard provides real-time monitoring of historical OHLCV and sentiment data quality.

### Installation

```bash
pip install streamlit plotly pandas
```

### Usage

```bash
streamlit run dashboards/data_quality_dashboard.py
```

Enable in `.env`:
```bash
DATA_QUALITY_DASHBOARD_ENABLED=true
```

### Dashboard Pages

1. **Overview** - Total symbols, files, last scan
2. **Quality Metrics** - Per-symbol metrics and validation errors
3. **Alerts** - Active quality alerts with filtering
4. **Batch Status** - Training pipeline execution status
5. **Fetch History** - Data fetch timeline

### Quality Metrics

- Missing files/bars
- Duplicate timestamps
- Zero-volume bars
- Sentiment gaps and low confidence
- Validation errors

### References

- [US-024 Story](../docs/stories/us-024-historical-data.md)
- [DataQualityService](../src/services/data_quality.py)
