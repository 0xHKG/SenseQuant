"""Real-time Telemetry Dashboard for Strategy Monitoring.

This Streamlit dashboard provides real-time monitoring of trading strategy performance
using cached telemetry data. It displays accuracy metrics, performance charts, confusion
matrices, and alerts for strategy degradation.

Usage:
    streamlit run dashboards/telemetry_dashboard.py -- \\
        --telemetry-dir data/analytics \\
        --refresh-interval 30 \\
        --alert-precision-threshold 0.55

Features:
    - Strategy overview cards (precision, recall, F1, Sharpe)
    - Rolling performance charts (cumulative returns)
    - Side-by-side confusion matrices (intraday vs swing)
    - Alert panel for degradation warnings
    - Symbol-level drill-down
    - Auto-refresh with configurable interval
    - Student model monitoring status (US-021)
    - Active release deployment tracking (US-023)
    - No live API calls (uses cached data only)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

# US-028 Phase 6x: Gracefully handle missing streamlit dependency
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    logger.warning("streamlit not installed - dashboard functionality will be limited")
    STREAMLIT_AVAILABLE = False

    # Create dummy streamlit object with no-op decorators
    class DummyStreamlit:
        """Dummy streamlit replacement when library not installed."""
        @staticmethod
        def cache_data(ttl=None):
            """No-op decorator replacement for st.cache_data."""
            def decorator(func):
                return func
            return decorator

    st = DummyStreamlit()  # type: ignore

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.accuracy_analyzer import AccuracyAnalyzer, AccuracyMetrics, PredictionTrace


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Telemetry Dashboard")
    parser.add_argument(
        "--telemetry-dir",
        type=str,
        default="data/analytics",
        help="Directory containing telemetry files",
    )
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=30,
        help="Auto-refresh interval in seconds",
    )
    parser.add_argument(
        "--alert-precision-threshold",
        type=float,
        default=0.55,
        help="Precision threshold for alerts",
    )
    return parser.parse_args()


@st.cache_data(ttl=30)
def load_telemetry_data(telemetry_dir: str) -> tuple[list[PredictionTrace], list[PredictionTrace]]:
    """Load telemetry data with caching (US-018 Phase 5: includes live/).

    Args:
        telemetry_dir: Directory containing telemetry files

    Returns:
        Tuple of (intraday_traces, swing_traces)
    """
    analyzer = AccuracyAnalyzer()
    telemetry_path = Path(telemetry_dir)

    if not telemetry_path.exists():
        logger.warning(f"Telemetry directory not found: {telemetry_path}")
        return [], []

    try:
        # Load from both backtest and live directories (US-018 Phase 5)
        all_traces = []

        # Load backtest traces (root directory)
        if telemetry_path.exists():
            backtest_traces = analyzer.load_traces(telemetry_path)
            all_traces.extend(backtest_traces)

        # Load live traces (live/ subdirectory)
        live_path = telemetry_path / "live"
        if live_path.exists():
            live_traces = analyzer.load_traces(live_path)
            all_traces.extend(live_traces)
            logger.info(f"Loaded {len(live_traces)} live traces")

        # Split by strategy
        intraday_traces = [t for t in all_traces if t.strategy.lower() == "intraday"]
        swing_traces = [t for t in all_traces if t.strategy.lower() == "swing"]

        logger.info(
            f"Loaded {len(intraday_traces)} intraday, {len(swing_traces)} swing traces (total: {len(all_traces)})"
        )

        return intraday_traces, swing_traces

    except Exception as e:
        logger.error(f"Failed to load telemetry data: {e}")
        return [], []


def is_live_mode(
    traces: list[PredictionTrace], threshold_minutes: int = 5
) -> tuple[bool, datetime | None]:
    """Detect if telemetry is in live mode (US-018 Phase 5).

    Args:
        traces: List of prediction traces
        threshold_minutes: Minutes threshold for considering "live" (default: 5)

    Returns:
        Tuple of (is_live, last_update_time)
    """
    if not traces:
        return False, None

    # Get most recent trace timestamp
    last_trace = max(traces, key=lambda t: t.timestamp)
    last_update = last_trace.timestamp

    # Check if within threshold
    now = datetime.now()
    elapsed = (now - last_update).total_seconds() / 60  # Convert to minutes

    is_live = elapsed < threshold_minutes

    return is_live, last_update


def compute_rolling_metrics(
    traces: list[PredictionTrace], window_size: int = 100
) -> tuple[AccuracyMetrics | None, AccuracyMetrics | None]:
    """Compute rolling vs all-time metrics (US-018 Phase 5).

    Args:
        traces: List of prediction traces
        window_size: Number of recent trades for rolling window (default: 100)

    Returns:
        Tuple of (rolling_metrics, all_time_metrics)
    """
    if not traces:
        return None, None

    # Sort by timestamp
    sorted_traces = sorted(traces, key=lambda t: t.timestamp)

    # All-time metrics
    analyzer = AccuracyAnalyzer()
    try:
        all_time_metrics = analyzer.compute_metrics(sorted_traces)
    except Exception as e:
        logger.error(f"Failed to compute all-time metrics: {e}")
        all_time_metrics = None

    # Rolling metrics (last N trades)
    rolling_traces = (
        sorted_traces[-window_size:] if len(sorted_traces) > window_size else sorted_traces
    )

    try:
        rolling_metrics = analyzer.compute_metrics(rolling_traces)
    except Exception as e:
        logger.error(f"Failed to compute rolling metrics: {e}")
        rolling_metrics = None

    return rolling_metrics, all_time_metrics


def detect_metric_degradation(
    rolling_metrics: AccuracyMetrics | None,
    all_time_metrics: AccuracyMetrics | None,
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Detect metric degradation (US-018 Phase 5).

    Args:
        rolling_metrics: Rolling window metrics
        all_time_metrics: All-time metrics
        thresholds: Optional thresholds dict (precision, win_rate, sharpe_ratio)

    Returns:
        List of alert messages
    """
    if thresholds is None:
        thresholds = {
            "precision_drop": 0.10,  # 10% drop
            "win_rate_drop": 0.10,  # 10% drop
            "sharpe_drop": 0.50,  # 0.5 drop
        }

    alerts = []

    if rolling_metrics is None or all_time_metrics is None:
        return alerts

    # Check precision drop
    rolling_precision = rolling_metrics.precision.get("LONG", 0.0)
    alltime_precision = all_time_metrics.precision.get("LONG", 0.0)
    if (
        alltime_precision > 0
        and (alltime_precision - rolling_precision) > thresholds["precision_drop"]
    ):
        alerts.append(
            f"⚠️ Precision drop: {rolling_precision:.2%} (rolling) vs {alltime_precision:.2%} (all-time)"
        )

    # Check win rate drop
    rolling_win_rate = rolling_metrics.win_rate
    alltime_win_rate = all_time_metrics.win_rate
    if alltime_win_rate > 0 and (alltime_win_rate - rolling_win_rate) > thresholds["win_rate_drop"]:
        alerts.append(
            f"⚠️ Win rate drop: {rolling_win_rate:.2%} (rolling) vs {alltime_win_rate:.2%} (all-time)"
        )

    # Check Sharpe drop
    rolling_sharpe = rolling_metrics.sharpe_ratio
    alltime_sharpe = all_time_metrics.sharpe_ratio
    if alltime_sharpe > 0 and (alltime_sharpe - rolling_sharpe) > thresholds["sharpe_drop"]:
        alerts.append(
            f"⚠️ Sharpe drop: {rolling_sharpe:.2f} (rolling) vs {alltime_sharpe:.2f} (all-time)"
        )

    return alerts


def compute_metrics_cached(
    traces: list[PredictionTrace],
) -> AccuracyMetrics | None:
    """Compute metrics with error handling.

    Args:
        traces: List of prediction traces

    Returns:
        AccuracyMetrics or None if computation fails
    """
    if not traces:
        return None

    analyzer = AccuracyAnalyzer()
    try:
        return analyzer.compute_metrics(traces)
    except Exception as e:
        logger.error(f"Failed to compute metrics: {e}")
        return None


def render_strategy_card(strategy_name: str, metrics: AccuracyMetrics | None, col: Any) -> None:
    """Render a strategy overview card.

    Args:
        strategy_name: Name of strategy (intraday/swing)
        metrics: Accuracy metrics
        col: Streamlit column object
    """
    with col:
        st.subheader(f"{strategy_name.title()} Strategy")

        if metrics is None:
            st.warning("No data available")
            return

        # Metrics grid
        metric_cols = st.columns(4)

        with metric_cols[0]:
            precision = metrics.precision.get("LONG", 0.0)
            st.metric("Precision", f"{precision:.2%}")

        with metric_cols[1]:
            recall = metrics.recall.get("LONG", 0.0)
            st.metric("Recall", f"{recall:.2%}")

        with metric_cols[2]:
            f1 = metrics.f1_score.get("LONG", 0.0)
            st.metric("F1 Score", f"{f1:.2%}")

        with metric_cols[3]:
            st.metric("Sharpe", f"{metrics.sharpe_ratio:.2f}")

        # Additional metrics row
        metric_cols2 = st.columns(4)

        with metric_cols2[0]:
            st.metric("Win Rate", f"{metrics.win_rate:.2%}")

        with metric_cols2[1]:
            st.metric("Avg Return", f"{metrics.avg_return:.2f}%")

        with metric_cols2[2]:
            st.metric("Total Trades", f"{metrics.total_trades}")

        with metric_cols2[3]:
            st.metric("Avg Hold (min)", f"{metrics.avg_holding_minutes:.0f}")


def render_confusion_matrix(strategy_name: str, metrics: AccuracyMetrics | None, col: Any) -> None:
    """Render confusion matrix heatmap.

    Args:
        strategy_name: Name of strategy
        metrics: Accuracy metrics
        col: Streamlit column object
    """
    with col:
        st.subheader(f"{strategy_name.title()} Confusion Matrix")

        if metrics is None or metrics.confusion_matrix is None:
            st.warning("No data available")
            return

        # Create heatmap
        labels = ["LONG", "SHORT", "NOOP"]
        fig = go.Figure(
            data=go.Heatmap(
                z=metrics.confusion_matrix,
                x=["Pred: " + label for label in labels],
                y=["Actual: " + label for label in labels],
                colorscale="Blues",
                text=metrics.confusion_matrix,
                texttemplate="%{text}",
                textfont={"size": 12},
            )
        )

        fig.update_layout(
            height=300,
            margin={"l": 50, "r": 50, "t": 30, "b": 50},
        )

        st.plotly_chart(fig, use_container_width=True)


def render_cumulative_returns(
    intraday_traces: list[PredictionTrace], swing_traces: list[PredictionTrace]
) -> None:
    """Render cumulative returns chart.

    Args:
        intraday_traces: Intraday prediction traces
        swing_traces: Swing prediction traces
    """
    st.subheader("Cumulative Returns")

    # Prepare data
    data_frames = []

    if intraday_traces:
        intraday_df = pd.DataFrame(
            [
                {
                    "timestamp": t.timestamp,
                    "return": t.realized_return_pct,
                    "strategy": "Intraday",
                }
                for t in sorted(intraday_traces, key=lambda x: x.timestamp)
            ]
        )
        intraday_df["cumulative_return"] = intraday_df["return"].cumsum()
        data_frames.append(intraday_df)

    if swing_traces:
        swing_df = pd.DataFrame(
            [
                {
                    "timestamp": t.timestamp,
                    "return": t.realized_return_pct,
                    "strategy": "Swing",
                }
                for t in sorted(swing_traces, key=lambda x: x.timestamp)
            ]
        )
        swing_df["cumulative_return"] = swing_df["return"].cumsum()
        data_frames.append(swing_df)

    if not data_frames:
        st.warning("No data available")
        return

    combined_df = pd.concat(data_frames, ignore_index=True)

    # Create line chart
    fig = px.line(
        combined_df,
        x="timestamp",
        y="cumulative_return",
        color="strategy",
        title="Cumulative Returns Over Time",
        labels={
            "timestamp": "Date",
            "cumulative_return": "Cumulative Return (%)",
            "strategy": "Strategy",
        },
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def load_active_release(
    releases_dir: str = "data/monitoring/releases",
) -> dict[str, Any]:
    """Load active release information (US-023).

    Args:
        releases_dir: Directory containing release tracking data

    Returns:
        Dictionary with active release info or empty dict if not available
    """
    try:
        active_release_path = Path(releases_dir) / "active_release.yaml"

        if not active_release_path.exists():
            return {}

        import yaml

        with open(active_release_path) as f:
            release_info = yaml.safe_load(f)

        # Auto-transition to normal monitoring after 48h
        heightened_end_str = release_info.get("heightened_monitoring_end")
        if heightened_end_str:
            from datetime import datetime

            heightened_end = datetime.fromisoformat(heightened_end_str)
            if datetime.now() >= heightened_end:
                release_info["heightened_monitoring_active"] = False

        return release_info

    except Exception as e:
        logger.error(f"Failed to load active release: {e}")
        return {}


def load_student_monitoring_status(
    monitoring_dir: str = "data/monitoring/student_model",
) -> dict[str, Any]:
    """Load student model monitoring status (US-021 Phase 3).

    Args:
        monitoring_dir: Directory containing student model monitoring data

    Returns:
        Dictionary with student model status or empty dict if not available
    """
    try:
        monitoring_path = Path(monitoring_dir)

        if not monitoring_path.exists():
            return {}

        # Load baseline metrics
        baseline_path = monitoring_path / "baseline_metrics.json"
        baseline_metrics = {}
        if baseline_path.exists():
            import json

            with open(baseline_path) as f:
                baseline_metrics = json.load(f)

        # Find most recent metrics snapshot
        snapshot_files = sorted(monitoring_path.glob("metrics_snapshot_*.json"), reverse=True)
        current_metrics = {}
        if snapshot_files:
            import json

            with open(snapshot_files[0]) as f:
                current_metrics = json.load(f)

        # Load recent alerts (if any)
        alert_file = Path("logs/alerts") / "alerts.jsonl"
        student_alerts = []
        if alert_file.exists():
            import json

            with open(alert_file) as f:
                for line in f:
                    try:
                        alert = json.loads(line)
                        if alert.get("rule", "").startswith("student_model_"):
                            student_alerts.append(alert)
                    except json.JSONDecodeError:
                        continue

        return {
            "enabled": bool(baseline_metrics or current_metrics),
            "baseline_metrics": baseline_metrics,
            "current_metrics": current_metrics,
            "recent_alerts": student_alerts[-5:],  # Last 5 alerts
        }

    except Exception as e:
        logger.error(f"Failed to load student monitoring status: {e}")
        return {}


def render_active_release(release_info: dict[str, Any]) -> None:
    """Render active release panel (US-023).

    Args:
        release_info: Active release information dictionary
    """
    st.subheader("Active Release")

    if not release_info:
        st.info("No active release registered")
        return

    # Release overview
    release_cols = st.columns(4)

    with release_cols[0]:
        release_id = release_info.get("release_id", "Unknown")
        st.metric("Release ID", release_id)

    with release_cols[1]:
        deployment_timestamp = release_info.get("deployment_timestamp", "")
        if deployment_timestamp:
            from datetime import datetime

            deploy_time = datetime.fromisoformat(deployment_timestamp)
            deploy_str = deploy_time.strftime("%Y-%m-%d %H:%M")
            st.metric("Deployed", deploy_str)
        else:
            st.metric("Deployed", "Unknown")

    with release_cols[2]:
        heightened_active = release_info.get("heightened_monitoring_active", False)
        if heightened_active:
            st.metric("Monitoring Mode", "🟡 Heightened")
        else:
            st.metric("Monitoring Mode", "🟢 Normal")

    with release_cols[3]:
        if heightened_active:
            heightened_end_str = release_info.get("heightened_monitoring_end", "")
            if heightened_end_str:
                from datetime import datetime

                heightened_end = datetime.fromisoformat(heightened_end_str)
                time_remaining = heightened_end - datetime.now()
                hours_remaining = int(time_remaining.total_seconds() / 3600)
                st.metric("Time Remaining", f"{hours_remaining}h")
            else:
                st.metric("Time Remaining", "N/A")
        else:
            st.metric("Time Remaining", "Completed")

    # Heightened monitoring details
    if heightened_active:
        st.markdown("**Heightened Monitoring Active:**")
        heightened_cols = st.columns(2)

        with heightened_cols[0]:
            st.markdown(
                """
                - Alert thresholds: **5%** (vs normal 10%)
                - Intraday window: **6 hours** (vs normal 24h)
                - Swing window: **24 hours** (vs normal 90 days)
                """
            )

        with heightened_cols[1]:
            heightened_hours = release_info.get("heightened_hours", 48)
            heightened_end_str = release_info.get("heightened_monitoring_end", "")
            if heightened_end_str:
                from datetime import datetime

                heightened_end = datetime.fromisoformat(heightened_end_str)
                st.markdown(
                    f"""
                    - Duration: **{heightened_hours} hours**
                    - Ends: **{heightened_end.strftime("%Y-%m-%d %H:%M")}**
                    """
                )

    # Manifest path
    manifest_path = release_info.get("manifest_path", "")
    if manifest_path:
        st.markdown(f"**Manifest:** `{manifest_path}`")

    # Rollback button with confirmation
    st.markdown("---")
    rollback_col1, rollback_col2 = st.columns([1, 3])

    with rollback_col1:
        if st.button("🔄 Rollback Release", type="primary"):
            st.session_state["confirm_rollback"] = True

    with rollback_col2:
        if st.session_state.get("confirm_rollback"):
            st.warning("⚠️ This will execute `make release-rollback`. Confirm in terminal.")
            if st.button("Cancel"):
                st.session_state["confirm_rollback"] = False


def render_student_model_status(status: dict[str, Any]) -> None:
    """Render student model monitoring status panel (US-021 Phase 3).

    Args:
        status: Student model status dictionary
    """
    st.subheader("Student Model Status")

    if not status.get("enabled"):
        st.info("Student model monitoring not enabled or no data available")
        return

    # Status overview
    status_cols = st.columns(4)

    current_metrics = status.get("current_metrics", {})
    baseline_metrics = status.get("baseline_metrics", {})

    with status_cols[0]:
        if current_metrics.get("model_version"):
            st.metric("Model Version", current_metrics["model_version"])
        else:
            st.metric("Model Version", "Unknown")

    with status_cols[1]:
        if "precision" in current_metrics:
            current_precision = current_metrics["precision"]
            baseline_precision = baseline_metrics.get("precision", 0.0)
            delta = current_precision - baseline_precision if baseline_precision else 0.0
            st.metric("Precision", f"{current_precision:.2%}", delta=f"{delta:+.2%}")
        else:
            st.metric("Precision", "N/A")

    with status_cols[2]:
        if "hit_ratio" in current_metrics:
            current_hit_ratio = current_metrics["hit_ratio"]
            baseline_hit_ratio = baseline_metrics.get("hit_ratio", 0.0)
            delta = current_hit_ratio - baseline_hit_ratio if baseline_hit_ratio else 0.0
            st.metric("Hit Ratio", f"{current_hit_ratio:.2%}", delta=f"{delta:+.2%}")
        else:
            st.metric("Hit Ratio", "N/A")

    with status_cols[3]:
        if "total_predictions" in current_metrics:
            st.metric("Predictions", current_metrics["total_predictions"])
        else:
            st.metric("Predictions", 0)

    # Recent alerts
    recent_alerts = status.get("recent_alerts", [])
    if recent_alerts:
        st.markdown("**Recent Alerts:**")
        for alert in recent_alerts:
            severity = alert.get("severity", "INFO")
            message = alert.get("message", "No message")
            timestamp = alert.get("timestamp", "")

            if severity == "CRITICAL":
                st.error(f"🚨 {timestamp}: {message}")
            elif severity == "WARNING":
                st.warning(f"⚠️ {timestamp}: {message}")
            else:
                st.info(f"ℹ️ {timestamp}: {message}")
    else:
        st.success("✅ No recent student model alerts")

    # Display baseline comparison if available
    if baseline_metrics and current_metrics:
        st.markdown("**Baseline Comparison:**")
        comparison_cols = st.columns(3)

        with comparison_cols[0]:
            baseline_precision = baseline_metrics.get("precision", 0.0)
            current_precision = current_metrics.get("precision", 0.0)
            precision_diff = current_precision - baseline_precision
            st.metric(
                "Precision Change",
                f"{precision_diff:+.2%}",
                delta=f"From baseline: {baseline_precision:.2%}",
            )

        with comparison_cols[1]:
            baseline_hit_ratio = baseline_metrics.get("hit_ratio", 0.0)
            current_hit_ratio = current_metrics.get("hit_ratio", 0.0)
            hit_ratio_diff = current_hit_ratio - baseline_hit_ratio
            st.metric(
                "Hit Ratio Change",
                f"{hit_ratio_diff:+.2%}",
                delta=f"From baseline: {baseline_hit_ratio:.2%}",
            )

        with comparison_cols[2]:
            window_hours = current_metrics.get("window_hours", 24)
            window_start = current_metrics.get("window_start", "")
            st.metric("Rolling Window", f"{window_hours}h", delta=f"Since {window_start[:10]}")


def render_alerts(
    intraday_metrics: AccuracyMetrics | None,
    swing_metrics: AccuracyMetrics | None,
    threshold: float,
) -> None:
    """Render alert panel.

    Args:
        intraday_metrics: Intraday accuracy metrics
        swing_metrics: Swing accuracy metrics
        threshold: Precision threshold for alerts
    """
    st.subheader("Alerts")

    alerts = []

    # Check intraday precision
    if intraday_metrics:
        intraday_precision = intraday_metrics.precision.get("LONG", 0.0)
        if intraday_precision < threshold:
            alerts.append(
                {
                    "severity": "WARNING",
                    "strategy": "Intraday",
                    "metric": "Precision",
                    "value": intraday_precision,
                    "threshold": threshold,
                    "message": f"Intraday precision ({intraday_precision:.2%}) below threshold ({threshold:.2%})",
                }
            )

        # Check Sharpe ratio
        if intraday_metrics.sharpe_ratio < 0.5:
            alerts.append(
                {
                    "severity": "WARNING",
                    "strategy": "Intraday",
                    "metric": "Sharpe Ratio",
                    "value": intraday_metrics.sharpe_ratio,
                    "threshold": 0.5,
                    "message": f"Intraday Sharpe ratio ({intraday_metrics.sharpe_ratio:.2f}) below 0.5",
                }
            )

    # Check swing precision
    if swing_metrics:
        swing_precision = swing_metrics.precision.get("LONG", 0.0)
        if swing_precision < threshold:
            alerts.append(
                {
                    "severity": "WARNING",
                    "strategy": "Swing",
                    "metric": "Precision",
                    "value": swing_precision,
                    "threshold": threshold,
                    "message": f"Swing precision ({swing_precision:.2%}) below threshold ({threshold:.2%})",
                }
            )

        # Check Sharpe ratio
        if swing_metrics.sharpe_ratio < 0.5:
            alerts.append(
                {
                    "severity": "WARNING",
                    "strategy": "Swing",
                    "metric": "Sharpe Ratio",
                    "value": swing_metrics.sharpe_ratio,
                    "threshold": 0.5,
                    "message": f"Swing Sharpe ratio ({swing_metrics.sharpe_ratio:.2f}) below 0.5",
                }
            )

    if alerts:
        for alert in alerts:
            st.warning(f"⚠️ {alert['message']}")
    else:
        st.success("✅ All metrics within acceptable range")


def main() -> None:
    """Main dashboard application."""
    # Parse args
    args = parse_args()

    # Page config
    st.set_page_config(
        page_title="Strategy Telemetry Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Title
    st.title("📊 Strategy Telemetry Dashboard")
    st.markdown("Real-time monitoring of trading strategy performance")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        telemetry_dir = st.text_input("Telemetry Directory", value=args.telemetry_dir)

        refresh_interval = st.slider(
            "Refresh Interval (seconds)", min_value=5, max_value=300, value=args.refresh_interval
        )

        alert_threshold = st.slider(
            "Alert Precision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=args.alert_precision_threshold,
            step=0.05,
        )

        st.markdown("---")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()

    # Load data
    with st.spinner("Loading telemetry data..."):
        intraday_traces, swing_traces = load_telemetry_data(telemetry_dir)

    # Compute metrics
    intraday_metrics = compute_metrics_cached(intraday_traces)
    swing_metrics = compute_metrics_cached(swing_traces)

    # Check if data is available
    if not intraday_traces and not swing_traces:
        st.error(
            f"No telemetry data found in {telemetry_dir}. "
            "Please run a backtest with --enable-telemetry first."
        )
        return

    # US-018 Phase 5: Live Mode Detection
    all_traces = intraday_traces + swing_traces
    is_live, last_update = is_live_mode(all_traces, threshold_minutes=5)

    # Live Mode Indicator (US-018 Phase 5)
    st.markdown("---")
    live_col1, live_col2, live_col3 = st.columns([1, 2, 3])

    with live_col1:
        if is_live:
            st.markdown("🟢 **LIVE MODE**")
        else:
            st.markdown("⚪ **HISTORICAL**")

    with live_col2:
        if last_update:
            elapsed_minutes = int((datetime.now() - last_update).total_seconds() / 60)
            st.markdown(f"**Last Update:** {last_update.strftime('%Y-%m-%d %H:%M:%S')}")

    with live_col3:
        if last_update:
            st.markdown(f"**Elapsed:** {elapsed_minutes} min ago")

    st.markdown("---")

    # US-018 Phase 5: Rolling vs All-Time Metrics
    st.header("Rolling Performance Analysis")

    # Compute rolling metrics for both strategies
    intraday_rolling, intraday_alltime = compute_rolling_metrics(intraday_traces, window_size=100)
    swing_rolling, swing_alltime = compute_rolling_metrics(swing_traces, window_size=100)

    # Display rolling comparison
    st.subheader("Intraday: Rolling (last 100) vs All-Time")
    if intraday_rolling and intraday_alltime:
        rolling_cols = st.columns(4)
        with rolling_cols[0]:
            st.metric(
                "Precision",
                f"{intraday_rolling.precision.get('LONG', 0.0):.2%}",
                delta=f"{(intraday_rolling.precision.get('LONG', 0.0) - intraday_alltime.precision.get('LONG', 0.0)):.2%}",
            )
        with rolling_cols[1]:
            st.metric(
                "Win Rate",
                f"{intraday_rolling.win_rate:.2%}",
                delta=f"{(intraday_rolling.win_rate - intraday_alltime.win_rate):.2%}",
            )
        with rolling_cols[2]:
            st.metric(
                "Sharpe",
                f"{intraday_rolling.sharpe_ratio:.2f}",
                delta=f"{(intraday_rolling.sharpe_ratio - intraday_alltime.sharpe_ratio):.2f}",
            )
        with rolling_cols[3]:
            st.metric(
                "Trades",
                f"{len(intraday_traces[-100:]) if len(intraday_traces) > 100 else len(intraday_traces)}",
                delta=f"of {intraday_alltime.total_trades}",
            )
    else:
        st.info("Insufficient intraday data for rolling analysis")

    st.subheader("Swing: Rolling (last 100) vs All-Time")
    if swing_rolling and swing_alltime:
        rolling_cols = st.columns(4)
        with rolling_cols[0]:
            st.metric(
                "Precision",
                f"{swing_rolling.precision.get('LONG', 0.0):.2%}",
                delta=f"{(swing_rolling.precision.get('LONG', 0.0) - swing_alltime.precision.get('LONG', 0.0)):.2%}",
            )
        with rolling_cols[1]:
            st.metric(
                "Win Rate",
                f"{swing_rolling.win_rate:.2%}",
                delta=f"{(swing_rolling.win_rate - swing_alltime.win_rate):.2%}",
            )
        with rolling_cols[2]:
            st.metric(
                "Sharpe",
                f"{swing_rolling.sharpe_ratio:.2f}",
                delta=f"{(swing_rolling.sharpe_ratio - swing_alltime.sharpe_ratio):.2f}",
            )
        with rolling_cols[3]:
            st.metric(
                "Trades",
                f"{len(swing_traces[-100:]) if len(swing_traces) > 100 else len(swing_traces)}",
                delta=f"of {swing_alltime.total_trades}",
            )
    else:
        st.info("Insufficient swing data for rolling analysis")

    st.markdown("---")

    # US-018 Phase 5: Metric Degradation Alerts
    st.header("Degradation Alerts")

    degradation_alerts = []

    if intraday_rolling and intraday_alltime:
        intraday_alerts = detect_metric_degradation(intraday_rolling, intraday_alltime)
        for alert in intraday_alerts:
            degradation_alerts.append(("Intraday", alert))

    if swing_rolling and swing_alltime:
        swing_alerts = detect_metric_degradation(swing_rolling, swing_alltime)
        for alert in swing_alerts:
            degradation_alerts.append(("Swing", alert))

    if degradation_alerts:
        for strategy, alert_msg in degradation_alerts:
            st.warning(f"**{strategy}:** {alert_msg}")
    else:
        st.success("✅ No metric degradation detected")

    st.markdown("---")

    # Strategy Overview Cards
    st.header("Strategy Overview")
    card_cols = st.columns(2)
    render_strategy_card("intraday", intraday_metrics, card_cols[0])
    render_strategy_card("swing", swing_metrics, card_cols[1])

    st.markdown("---")

    # Cumulative Returns
    st.header("Performance")
    render_cumulative_returns(intraday_traces, swing_traces)

    st.markdown("---")

    # Confusion Matrices
    st.header("Prediction Accuracy")
    matrix_cols = st.columns(2)
    render_confusion_matrix("intraday", intraday_metrics, matrix_cols[0])
    render_confusion_matrix("swing", swing_metrics, matrix_cols[1])

    st.markdown("---")

    # Alerts
    st.header("Alerts & Monitoring")
    render_alerts(intraday_metrics, swing_metrics, alert_threshold)

    st.markdown("---")

    # US-021 Phase 3: Student Model Status
    st.header("Student Model Monitoring")
    student_status = load_student_monitoring_status()
    render_student_model_status(student_status)

    st.markdown("---")

    # US-023: Active Release Panel
    st.header("Release Deployment")
    release_info = load_active_release()
    render_active_release(release_info)

    # Auto-refresh
    if refresh_interval > 0:
        import time

        time.sleep(refresh_interval)
        st.rerun()


def load_training_progress() -> dict[str, Any]:
    """Load training progress from StateManager (US-028 Phase 7 Initiative 4).

    Returns:
        Training progress dict for all phases
    """
    from src.services.state_manager import StateManager

    try:
        state_mgr = StateManager(Path("data/state/state.json"))
        return state_mgr.get_training_progress()
    except Exception as e:
        logger.warning(f"Failed to load training progress: {e}")
        return {}


def render_training_progress_page() -> None:
    """Render training progress monitoring page (US-028 Phase 7 Initiative 4).

    This is a standalone page that can be added to the dashboard to monitor
    long-running training pipelines in real-time.
    """
    if not STREAMLIT_AVAILABLE:
        logger.info("Streamlit not available - training progress monitoring disabled")
        logger.info("To enable, install streamlit: pip install streamlit")
        return

    st.title("Training Progress Monitor")
    st.markdown("Real-time progress tracking for historical training pipelines")

    progress_data = load_training_progress()

    if not progress_data:
        st.warning("No training progress data available. Start a training run to see progress.")
        return

    # Display progress for each phase
    for phase_name, phase_data in progress_data.items():
        with st.expander(f"📊 {phase_name.replace('_', ' ').title()}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Progress",
                    f"{phase_data.get('completed', 0)}/{phase_data.get('total', 0)}",
                )

            with col2:
                st.metric(
                    "Percent Complete",
                    f"{phase_data.get('percent_complete', 0):.1f}%",
                )

            with col3:
                eta = phase_data.get("eta_minutes")
                if eta:
                    st.metric("ETA", f"{eta:.1f} min")
                else:
                    st.metric("ETA", "N/A")

            with col4:
                st.metric("Status", phase_data.get("status", "unknown").upper())

            # Show additional metadata
            if "trained" in phase_data:
                st.write(f"✅ Trained: {phase_data['trained']}")
            if "skipped" in phase_data:
                st.write(f"⊘ Skipped: {phase_data['skipped']}")
            if "failed" in phase_data:
                st.write(f"✗ Failed: {phase_data['failed']}")

            # Progress bar
            percent = phase_data.get("percent_complete", 0)
            st.progress(percent / 100.0)

            # Timestamp
            timestamp = phase_data.get("timestamp", "N/A")
            st.caption(f"Last updated: {timestamp}")

    # Auto-refresh button
    if st.button("🔄 Refresh Progress"):
        st.rerun()


if __name__ == "__main__":
    main()
