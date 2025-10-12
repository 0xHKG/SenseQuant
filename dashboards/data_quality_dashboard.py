"""Data Quality Dashboard for US-024 Phase 6.

Streamlit dashboard showing data quality metrics for historical OHLCV and sentiment data:
- Missing bars/files
- Duplicate timestamps
- Zero-volume counts
- Sentiment gaps
- Retry failures
- Batch execution status

Usage:
    streamlit run dashboards/data_quality_dashboard.py

Requirements:
    pip install streamlit plotly
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import streamlit as st
except ImportError:
    print("Error: streamlit required for dashboard")
    print("Install with: pip install streamlit plotly")
    sys.exit(1)

from src.app.config import Settings
from src.services.data_quality import DataQualityService
from src.services.state_manager import StateManager


def load_state_data() -> tuple[StateManager, StateManager, StateManager]:
    """Load state managers for different data sources.

    Returns:
        Tuple of (historical_state, sentiment_state, teacher_batch_state)
    """
    historical_state = StateManager(Path("data/state/historical_fetch.json"))
    sentiment_state = StateManager(Path("data/state/sentiment_fetch.json"))
    teacher_batch_state = StateManager(Path("data/state/teacher_batch.json"))

    return historical_state, sentiment_state, teacher_batch_state


def render_overview(settings: Settings, quality_service: DataQualityService) -> None:
    """Render overview metrics."""
    st.header("📊 Data Quality Overview")

    # Get summary for all symbols
    summary = quality_service.get_summary_for_all_symbols()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Symbols", summary["total_symbols"])

    with col2:
        total_hist_files = sum(
            s["historical"].get("total_files", 0) for s in summary["symbols"].values()
        )
        st.metric("Historical Files", total_hist_files)

    with col3:
        total_sent_files = sum(
            s["sentiment"].get("total_files", 0) for s in summary["symbols"].values()
        )
        st.metric("Sentiment Files", total_sent_files)

    # Last scan timestamp
    st.text(f"Last scanned: {summary['scan_timestamp']}")


def render_quality_metrics(
    quality_service: DataQualityService, state_manager: StateManager
) -> None:
    """Render quality metrics by symbol."""
    st.header("📈 Quality Metrics by Symbol")

    # Get quality metrics from state
    quality_metrics = state_manager.get_quality_metrics()

    if not quality_metrics:
        st.info("No quality metrics available. Run data fetch to generate metrics.")
        return

    # Symbol selector
    symbols = sorted(quality_metrics.keys())
    selected_symbol = st.selectbox("Select Symbol", symbols)

    if selected_symbol:
        symbol_metrics = quality_metrics[selected_symbol]

        # Historical metrics
        if "historical" in symbol_metrics:
            st.subheader(f"Historical Data: {selected_symbol}")
            hist_metrics = symbol_metrics["historical"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Files", hist_metrics.get("total_files", 0))
            col2.metric("Total Bars", hist_metrics.get("total_bars", 0))
            col3.metric("Missing Files", hist_metrics.get("missing_files", 0))
            col4.metric("Duplicates", hist_metrics.get("duplicate_timestamps", 0))

            col5, col6 = st.columns(2)
            col5.metric("Zero Volume", hist_metrics.get("zero_volume_bars", 0))

            # Validation errors
            if hist_metrics.get("validation_errors"):
                st.warning(f"⚠️ {len(hist_metrics['validation_errors'])} validation errors")
                with st.expander("Show validation errors"):
                    for error in hist_metrics["validation_errors"]:
                        st.text(f"  • {error}")

        # Sentiment metrics
        if "sentiment" in symbol_metrics:
            st.subheader(f"Sentiment Data: {selected_symbol}")
            sent_metrics = symbol_metrics["sentiment"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Files", sent_metrics.get("total_files", 0))
            col2.metric("Total Snapshots", sent_metrics.get("total_snapshots", 0))
            col3.metric("Missing Files", sent_metrics.get("missing_files", 0))

            col4, col5 = st.columns(2)
            col4.metric("Invalid Scores", sent_metrics.get("invalid_scores", 0))
            col5.metric("Low Confidence", sent_metrics.get("low_confidence", 0))

            # Validation errors
            if sent_metrics.get("validation_errors"):
                st.warning(f"⚠️ {len(sent_metrics['validation_errors'])} validation errors")
                with st.expander("Show validation errors"):
                    for error in sent_metrics["validation_errors"]:
                        st.text(f"  • {error}")


def render_alerts(state_manager: StateManager) -> None:
    """Render quality alerts."""
    st.header("🚨 Quality Alerts")

    alerts = state_manager.get_quality_alerts()

    if not alerts:
        st.success("✅ No active quality alerts")
        return

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        severity_filter = st.selectbox("Filter by Severity", ["All", "error", "warning"])
    with col2:
        symbol_filter = st.selectbox(
            "Filter by Symbol",
            ["All"] + sorted(set(a["symbol"] for a in alerts)),
        )

    # Apply filters
    filtered_alerts = alerts
    if severity_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a["severity"] == severity_filter]
    if symbol_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a["symbol"] == symbol_filter]

    # Display alerts
    st.metric("Total Alerts", len(filtered_alerts))

    for alert in filtered_alerts:
        severity_icon = "🔴" if alert["severity"] == "error" else "⚠️"
        with st.expander(
            f"{severity_icon} [{alert['severity'].upper()}] {alert['symbol']} - {alert['metric']}"
        ):
            st.text(f"Message: {alert['message']}")
            st.text(f"Value: {alert['value']}")
            st.text(f"Threshold: {alert['threshold']}")
            st.text(f"Timestamp: {alert['timestamp']}")


def render_batch_status(batch_state: StateManager) -> None:
    """Render batch execution status."""
    st.header("🎯 Batch Training Status")

    batches = batch_state.state.get("batches", {})

    if not batches:
        st.info("No batch training runs recorded")
        return

    # Get most recent batch
    batch_ids = sorted(batches.keys(), reverse=True)
    selected_batch = st.selectbox("Select Batch", batch_ids)

    if selected_batch:
        batch = batches[selected_batch]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", batch.get("status", "unknown").upper())
        col2.metric("Completed", batch.get("completed", 0))
        col3.metric("Failed", batch.get("failed", 0))
        col4.metric("Total Tasks", batch.get("total_tasks", 0))

        # Failed tasks
        failed_tasks = batch.get("failed_tasks", [])
        if failed_tasks:
            st.warning(f"⚠️ {len(failed_tasks)} failed tasks requiring manual review")
            with st.expander("Show failed tasks"):
                for task in failed_tasks:
                    st.text(
                        f"  • {task['symbol']}/{task['window_label']}: "
                        f"{task['reason']} (after {task['attempts']} attempts)"
                    )


def render_fetch_history(hist_state: StateManager, sent_state: StateManager) -> None:
    """Render fetch history timeline."""
    st.header("📅 Fetch History")

    # Historical fetch history
    st.subheader("Historical Data Fetches")
    hist_symbols = hist_state.get_all_symbols()

    if hist_symbols:
        fetch_dates = []
        for symbol in hist_symbols:
            last_fetch = hist_state.get_last_fetch_date(symbol)
            if last_fetch:
                fetch_dates.append(
                    {
                        "symbol": symbol,
                        "last_fetch": last_fetch.strftime("%Y-%m-%d"),
                    }
                )

        if fetch_dates:
            import pandas as pd

            df = pd.DataFrame(fetch_dates)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No fetch history available")
    else:
        st.info("No historical fetches recorded")

    # Sentiment fetch history
    st.subheader("Sentiment Data Fetches")
    sent_symbols = sent_state.get_all_symbols()

    if sent_symbols:
        fetch_dates = []
        for symbol in sent_symbols:
            last_fetch = sent_state.get_last_fetch_date(symbol)
            if last_fetch:
                fetch_dates.append(
                    {
                        "symbol": symbol,
                        "last_fetch": last_fetch.strftime("%Y-%m-%d"),
                    }
                )

        if fetch_dates:
            import pandas as pd

            df = pd.DataFrame(fetch_dates)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No fetch history available")
    else:
        st.info("No sentiment fetches recorded")


def main() -> None:
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="SenseQuant Data Quality Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 SenseQuant Data Quality Dashboard")
    st.caption("US-024 Phase 6: Data Quality Monitoring & Alerts")

    # Load settings
    settings = Settings()  # type: ignore[call-arg]

    # Check if dashboard is enabled
    if not settings.data_quality_dashboard_enabled:
        st.warning(
            "⚠️ Dashboard is disabled. Set DATA_QUALITY_DASHBOARD_ENABLED=true in .env to enable."
        )
        st.stop()

    # Initialize services
    historical_dir = Path(settings.historical_data_output_dir)
    sentiment_dir = Path(settings.sentiment_snapshot_output_dir)
    quality_service = DataQualityService(historical_dir, sentiment_dir)

    # Load state managers
    hist_state, sent_state, batch_state = load_state_data()

    # Use combined state manager for quality metrics
    # In practice, we'll use the historical state manager to store quality metrics
    state_manager = hist_state

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "Overview",
            "Quality Metrics",
            "Alerts",
            "Batch Status",
            "Fetch History",
        ],
    )

    # Render selected page
    if page == "Overview":
        render_overview(settings, quality_service)
    elif page == "Quality Metrics":
        render_quality_metrics(quality_service, state_manager)
    elif page == "Alerts":
        render_alerts(state_manager)
    elif page == "Batch Status":
        render_batch_status(batch_state)
    elif page == "Fetch History":
        render_fetch_history(hist_state, sent_state)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("SenseQuant Data Quality Dashboard v1.0")
    st.sidebar.caption("US-024 Phase 6")


if __name__ == "__main__":
    main()
