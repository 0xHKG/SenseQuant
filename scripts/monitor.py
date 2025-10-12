#!/usr/bin/env python3
"""Enterprise CLI tool for monitoring and alerts (v2)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta

from loguru import logger

from src.app.config import settings
from src.services.monitoring import MonitoringService


def setup_logging() -> None:
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


# =============================================================================
# ALERTS SUBCOMMAND
# =============================================================================


def cmd_alerts_list(monitoring: MonitoringService, args: argparse.Namespace) -> None:
    """List active alerts with optional filtering."""
    alerts = monitoring.get_active_alerts(hours=args.hours)

    # Filter by severity if specified
    if args.severity:
        alerts = [a for a in alerts if a.severity == args.severity.upper()]

    if not alerts:
        print(f"No active alerts in last {args.hours} hours")
        return

    print(f"\n=== Active Alerts (Last {args.hours} Hours) ===\n")
    for i, alert in enumerate(alerts, 1):
        severity_icon = {"INFO": "ℹ️ ", "WARNING": "⚠️ ", "CRITICAL": "🚨"}.get(alert.severity, "• ")
        ack_marker = " [ACKED]" if monitoring._is_acknowledged(alert.rule) else ""
        print(f"{i}. {severity_icon}[{alert.severity}] {alert.rule}{ack_marker}")
        print(f"   Time: {alert.timestamp}")
        print(f"   Message: {alert.message}")
        if alert.context and args.verbose:
            print(f"   Context: {json.dumps(alert.context, indent=4)}")
        print()


def cmd_alerts_ack(monitoring: MonitoringService, args: argparse.Namespace) -> None:
    """Acknowledge alert(s)."""
    if args.all:
        # Acknowledge all active alerts
        alerts = monitoring.get_active_alerts(hours=24)
        for alert in alerts:
            monitoring.acknowledge_alert(
                alert.rule, acknowledged_by=args.operator, reason=args.reason
            )
        print(f"✅ Acknowledged {len(alerts)} alerts")
    else:
        # Acknowledge specific rule
        monitoring.acknowledge_alert(args.rule, acknowledged_by=args.operator, reason=args.reason)
        print(f"✅ Acknowledged alert: {args.rule}")


def cmd_alerts_clear(monitoring: MonitoringService, args: argparse.Namespace) -> None:
    """Clear acknowledgement for alert rule."""
    monitoring.clear_acknowledgement(args.rule)
    print(f"✅ Cleared acknowledgement: {args.rule}")


def cmd_alerts(monitoring: MonitoringService, args: argparse.Namespace) -> None:
    """Handle alerts subcommand."""
    if args.alerts_action == "list":
        cmd_alerts_list(monitoring, args)
    elif args.alerts_action == "ack":
        cmd_alerts_ack(monitoring, args)
    elif args.alerts_action == "clear":
        cmd_alerts_clear(monitoring, args)


# =============================================================================
# METRICS SUBCOMMAND
# =============================================================================


def cmd_metrics_show(monitoring: MonitoringService, args: argparse.Namespace) -> None:
    """Show aggregated metrics."""
    # Parse interval
    interval_map = {"5m": 5, "15m": 15, "1h": 60, "6h": 360, "1d": 1440}
    hours = interval_map.get(args.interval, 60) / 60

    rollups = monitoring.get_aggregated_metrics(start_time=datetime.now() - timedelta(hours=hours))

    if not rollups:
        print(f"No aggregated metrics for interval {args.interval}")
        return

    print(f"\n=== Aggregated Metrics ({args.interval}) ===\n")
    print(f"Rollups: {len(rollups)}")
    print()

    # Display latest rollup stats
    if rollups:
        latest = rollups[-1]
        print("Latest Rollup:")
        print(f"  Period: {latest.interval_start} to {latest.interval_end}")
        print(f"  Duration: {latest.interval_seconds}s")
        print()

        for metric_name, stats in latest.metrics.items():
            print(f"  {metric_name}:")
            print(f"    Min: {stats.min:.2f}")
            print(f"    Max: {stats.max:.2f}")
            print(f"    Avg: {stats.avg:.2f}")
            print(f"    Count: {stats.count}")
            print()


def cmd_metrics_export(monitoring: MonitoringService, args: argparse.Namespace) -> None:
    """Export metrics to CSV or JSON."""
    hours = args.hours
    rollups = monitoring.get_aggregated_metrics(start_time=datetime.now() - timedelta(hours=hours))

    if not rollups:
        print(f"No metrics to export for last {hours} hours")
        return

    output_file = (
        args.output or f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}"
    )

    if args.format == "csv":
        # Export as CSV
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(
                [
                    "interval_start",
                    "interval_end",
                    "metric_name",
                    "min",
                    "max",
                    "avg",
                    "count",
                    "sum",
                ]
            )
            # Write data
            for rollup in rollups:
                for metric_name, stats in rollup.metrics.items():
                    writer.writerow(
                        [
                            rollup.interval_start,
                            rollup.interval_end,
                            metric_name,
                            stats.min,
                            stats.max,
                            stats.avg,
                            stats.count,
                            stats.sum,
                        ]
                    )
    else:
        # Export as JSON
        data = []
        for rollup in rollups:
            rollup_dict = {
                "interval_start": rollup.interval_start,
                "interval_end": rollup.interval_end,
                "interval_seconds": rollup.interval_seconds,
                "metrics": {
                    name: {
                        "min": stats.min,
                        "max": stats.max,
                        "avg": stats.avg,
                        "count": stats.count,
                        "sum": stats.sum,
                    }
                    for name, stats in rollup.metrics.items()
                },
            }
            data.append(rollup_dict)

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    print(f"✅ Exported {len(rollups)} rollups to {output_file}")


def cmd_metrics_summary(monitoring: MonitoringService, args: argparse.Namespace) -> None:
    """Show statistical summary of metrics."""
    hours = args.hours
    rollups = monitoring.get_aggregated_metrics(start_time=datetime.now() - timedelta(hours=hours))

    if not rollups:
        print(f"No metrics for last {hours} hours")
        return

    print(f"\n=== Metrics Summary (Last {hours} Hours) ===\n")

    # Aggregate across all rollups
    metric_aggregates: dict[str, dict[str, list[float]]] = {}

    for rollup in rollups:
        for metric_name, stats in rollup.metrics.items():
            if metric_name not in metric_aggregates:
                metric_aggregates[metric_name] = {
                    "mins": [],
                    "maxs": [],
                    "avgs": [],
                    "counts": [],
                }
            metric_aggregates[metric_name]["mins"].append(stats.min)
            metric_aggregates[metric_name]["maxs"].append(stats.max)
            metric_aggregates[metric_name]["avgs"].append(stats.avg)
            metric_aggregates[metric_name]["counts"].append(stats.count)

    # Display summary statistics
    for metric_name, data in metric_aggregates.items():
        print(f"{metric_name}:")
        print(f"  Overall Min: {min(data['mins']):.2f}")
        print(f"  Overall Max: {max(data['maxs']):.2f}")
        print(f"  Avg of Avgs: {sum(data['avgs']) / len(data['avgs']):.2f}")
        print(f"  Total Samples: {sum(data['counts'])}")
        print()


def cmd_metrics(monitoring: MonitoringService, args: argparse.Namespace) -> None:
    """Handle metrics subcommand."""
    if args.metrics_action == "show":
        cmd_metrics_show(monitoring, args)
    elif args.metrics_action == "export":
        cmd_metrics_export(monitoring, args)
    elif args.metrics_action == "summary":
        cmd_metrics_summary(monitoring, args)


# =============================================================================
# WATCH SUBCOMMAND
# =============================================================================


def cmd_watch(monitoring: MonitoringService, args: argparse.Namespace) -> None:
    """Watch alerts in real-time."""
    print("=== Watching Alerts (Ctrl+C to stop) ===\n")

    # Try to use watchdog for filesystem events
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        class AlertFileHandler(FileSystemEventHandler):
            def __init__(self, monitoring_service: MonitoringService):
                self.monitoring = monitoring_service
                self.last_size = 0
                self.alert_file = None

            def on_modified(self, event):
                if event.is_directory:
                    return
                if not event.src_path.endswith(".jsonl"):
                    return

                # Read new alerts
                try:
                    with open(event.src_path) as f:
                        f.seek(self.last_size)
                        new_lines = f.readlines()
                        self.last_size = f.tell()

                        for line in new_lines:
                            if line.strip():
                                alert_dict = json.loads(line)
                                # Filter by severity if specified
                                if (
                                    args.severity
                                    and alert_dict["severity"] != args.severity.upper()
                                ):
                                    continue

                                severity_icon = {
                                    "INFO": "ℹ️ ",
                                    "WARNING": "⚠️ ",
                                    "CRITICAL": "🚨",
                                }.get(alert_dict["severity"], "• ")

                                print(
                                    f"\n{severity_icon}[{alert_dict['severity']}] {alert_dict['rule']}"
                                )
                                print(f"  Time: {alert_dict['timestamp']}")
                                print(f"  Message: {alert_dict['message']}")
                                if args.verbose and alert_dict.get("context"):
                                    print(
                                        f"  Context: {json.dumps(alert_dict['context'], indent=4)}"
                                    )
                except Exception as e:
                    print(f"Error reading alert file: {e}")

        # Set up filesystem watcher
        handler = AlertFileHandler(monitoring)
        observer = Observer()
        observer.schedule(handler, str(monitoring.alerts_dir), recursive=False)
        observer.start()

        # Initialize last_size for today's alert file
        alert_date = datetime.now().strftime("%Y-%m-%d")
        alert_file = monitoring.alerts_dir / f"{alert_date}.jsonl"
        if alert_file.exists():
            handler.last_size = alert_file.stat().st_size

        print("Watching for new alerts using filesystem events...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\nStopped watching alerts")

        observer.join()

    except ImportError:
        # Fallback to polling if watchdog not available
        print("Note: Install 'watchdog' package for better real-time monitoring")
        print("Falling back to polling mode...\n")

        seen_alerts = set()

        try:
            while True:
                time.sleep(args.poll_interval)
                alerts = monitoring.get_active_alerts(hours=1)

                # Filter by severity if specified
                if args.severity:
                    alerts = [a for a in alerts if a.severity == args.severity.upper()]

                # Show only new alerts
                for alert in alerts:
                    alert_id = f"{alert.timestamp}:{alert.rule}"
                    if alert_id not in seen_alerts:
                        seen_alerts.add(alert_id)
                        severity_icon = {
                            "INFO": "ℹ️ ",
                            "WARNING": "⚠️ ",
                            "CRITICAL": "🚨",
                        }.get(alert.severity, "• ")

                        print(f"\n{severity_icon}[{alert.severity}] {alert.rule}")
                        print(f"  Time: {alert.timestamp}")
                        print(f"  Message: {alert.message}")
                        if args.verbose and alert.context:
                            print(f"  Context: {json.dumps(alert.context, indent=4)}")

        except KeyboardInterrupt:
            print("\nStopped watching alerts")


# =============================================================================
# STATUS SUBCOMMAND
# =============================================================================


def cmd_status(monitoring: MonitoringService, args: argparse.Namespace) -> None:
    """Show aggregated system status dashboard."""
    print("\n" + "=" * 70)
    print(" " * 20 + "SENSQUANT SYSTEM STATUS")
    print("=" * 70 + "\n")

    # Alert Summary
    print("📊 ALERT SUMMARY")
    print("-" * 70)
    alerts = monitoring.get_active_alerts(hours=24)
    critical = [a for a in alerts if a.severity == "CRITICAL"]
    warning = [a for a in alerts if a.severity == "WARNING"]
    info = [a for a in alerts if a.severity == "INFO"]

    print(f"  Last 24 Hours: {len(alerts)} total alerts")
    if critical:
        print(f"  🚨 CRITICAL: {len(critical)}")
        for alert in critical[:3]:  # Show first 3
            print(f"     - {alert.rule}: {alert.message}")
    if warning:
        print(f"  ⚠️  WARNING: {len(warning)}")
    if info:
        print(f"  ℹ️  INFO: {len(info)}")
    if not alerts:
        print("  ✅ No active alerts")
    print()

    # Health Checks
    print("🏥 HEALTH CHECKS")
    print("-" * 70)
    health_results = monitoring.run_health_checks()

    for result in health_results:
        status_icon = {"OK": "✅", "WARNING": "⚠️", "ERROR": "🚨"}.get(result.status, "• ")
        print(f"  {status_icon} {result.check_name}: {result.message}")
        if result.details and result.status != "OK" and args.verbose:
            for key, value in result.details.items():
                print(f"     {key}: {value}")
    print()

    # Performance Metrics (if available)
    if monitoring.settings.monitoring_enable_performance_tracking:
        print("⚡ PERFORMANCE METRICS (Last Hour)")
        print("-" * 70)
        perf_stats = monitoring._aggregate_performance_stats()

        if perf_stats:
            for metric_name, stats in perf_stats.items():
                status = (
                    "✅"
                    if stats["avg"] < monitoring.settings.monitoring_performance_alert_threshold_ms
                    else "⚠️"
                )
                print(f"  {status} {metric_name}:")
                print(f"     Avg: {stats['avg']:.2f}ms")
                print(f"     Min: {stats['min']:.2f}ms")
                print(f"     Max: {stats['max']:.2f}ms")
                print(f"     Samples: {stats['count']}")
        else:
            print("  No performance data available")
        print()

    # Recent Metrics
    if monitoring.metrics_history:
        print("📈 RECENT METRICS")
        print("-" * 70)
        latest = monitoring.metrics_history[-1]

        # Positions
        positions = latest.get("positions", {})
        print(f"  Positions: {positions.get('count', 0)} open")
        if positions.get("symbols"):
            print(f"    Symbols: {', '.join(positions.get('symbols', []))}")

        # PnL
        pnl = latest.get("pnl", {})
        daily_pnl = pnl.get("daily", 0.0)
        daily_loss_pct = pnl.get("daily_loss_pct", 0.0)
        pnl_icon = "✅" if daily_pnl >= 0 else "📉"
        print(f"  {pnl_icon} Daily PnL: ₹{daily_pnl:,.2f} ({daily_loss_pct:+.2f}%)")

        # Risk
        risk = latest.get("risk", {})
        if risk.get("circuit_breaker_active"):
            print("  🚨 Circuit Breaker: ACTIVE")
        else:
            print("  ✅ Circuit Breaker: Inactive")

        # Heartbeat
        if monitoring.heartbeat_timestamp:
            seconds_ago = (datetime.now() - monitoring.heartbeat_timestamp).total_seconds()
            print(f"  💓 Last Heartbeat: {seconds_ago:.0f}s ago")
        print()

    # Acknowledgements
    active_acks = [ack for ack in monitoring.acknowledgements.values() if not ack.is_expired()]
    if active_acks:
        print("✓ ACTIVE ACKNOWLEDGEMENTS")
        print("-" * 70)
        for ack in active_acks:
            print(f"  • {ack.rule}")
            print(f"    Acknowledged by: {ack.acknowledged_by}")
            print(f"    Time: {ack.acknowledged_at}")
            if ack.reason:
                print(f"    Reason: {ack.reason}")
        print()

    # System Health Score
    print("🎯 SYSTEM HEALTH SCORE")
    print("-" * 70)
    score = 100
    if critical:
        score -= len(critical) * 20
    if warning:
        score -= len(warning) * 5
    score = max(0, score)

    if score >= 90:
        health_status = "🟢 EXCELLENT"
    elif score >= 70:
        health_status = "🟡 GOOD"
    elif score >= 50:
        health_status = "🟠 FAIR"
    else:
        health_status = "🔴 POOR"

    print(f"  {health_status} - Score: {score}/100")
    print()
    print("=" * 70 + "\n")


# =============================================================================
# HEALTH SUBCOMMAND (Legacy)
# =============================================================================


def cmd_health(monitoring: MonitoringService, args: argparse.Namespace) -> None:
    """Run and display health checks (legacy command)."""
    print("\n=== Running Health Checks ===\n")
    results = monitoring.run_health_checks()

    all_ok = True
    for result in results:
        status_icon = {"OK": "✅", "WARNING": "⚠️", "ERROR": "🚨"}.get(result.status, "• ")
        print(f"{status_icon} {result.check_name}: {result.message}")

        if result.details:
            for key, value in result.details.items():
                print(f"   {key}: {value}")

        if result.status in ["WARNING", "ERROR"]:
            all_ok = False

    print()
    if all_ok:
        print("✅ All checks passed")
        sys.exit(0)
    else:
        print("⚠️  Some checks failed or have warnings")
        sys.exit(1)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SenseQuant Enterprise Monitoring CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Alerts
  python scripts/monitor.py alerts list
  python scripts/monitor.py alerts list --severity CRITICAL --hours 48
  python scripts/monitor.py alerts ack circuit_breaker_triggered
  python scripts/monitor.py alerts ack --all --reason "Investigating"
  python scripts/monitor.py alerts clear daily_loss_high

  # Metrics
  python scripts/monitor.py metrics show --interval 1h
  python scripts/monitor.py metrics export --format csv --hours 24
  python scripts/monitor.py metrics summary --hours 48

  # Watch
  python scripts/monitor.py watch
  python scripts/monitor.py watch --severity CRITICAL

  # Status
  python scripts/monitor.py status
  python scripts/monitor.py status --verbose

  # Health (legacy)
  python scripts/monitor.py health
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ALERTS subcommand
    alerts_parser = subparsers.add_parser("alerts", help="Manage alerts")
    alerts_subparsers = alerts_parser.add_subparsers(dest="alerts_action", help="Alerts action")

    # alerts list
    list_parser = alerts_subparsers.add_parser("list", help="List active alerts")
    list_parser.add_argument("--hours", type=int, default=24, help="Time window in hours")
    list_parser.add_argument(
        "--severity", choices=["INFO", "WARNING", "CRITICAL"], help="Filter by severity"
    )
    list_parser.add_argument("--verbose", "-v", action="store_true", help="Show full context")

    # alerts ack
    ack_parser = alerts_subparsers.add_parser("ack", help="Acknowledge alert(s)")
    ack_parser.add_argument("rule", nargs="?", help="Alert rule to acknowledge")
    ack_parser.add_argument("--all", action="store_true", help="Acknowledge all active alerts")
    ack_parser.add_argument("--operator", default="cli_user", help="Operator name")
    ack_parser.add_argument("--reason", help="Reason for acknowledgement")

    # alerts clear
    clear_parser = alerts_subparsers.add_parser("clear", help="Clear acknowledgement")
    clear_parser.add_argument("rule", help="Alert rule to clear")

    # METRICS subcommand
    metrics_parser = subparsers.add_parser("metrics", help="View and export metrics")
    metrics_subparsers = metrics_parser.add_subparsers(dest="metrics_action", help="Metrics action")

    # metrics show
    show_parser = metrics_subparsers.add_parser("show", help="Show aggregated metrics")
    show_parser.add_argument(
        "--interval", choices=["5m", "15m", "1h", "6h", "1d"], default="1h", help="Time interval"
    )

    # metrics export
    export_parser = metrics_subparsers.add_parser("export", help="Export metrics")
    export_parser.add_argument(
        "--format", choices=["csv", "json"], default="csv", help="Export format"
    )
    export_parser.add_argument("--hours", type=int, default=24, help="Time window in hours")
    export_parser.add_argument("--output", "-o", help="Output file path")

    # metrics summary
    summary_parser = metrics_subparsers.add_parser("summary", help="Show summary statistics")
    summary_parser.add_argument("--hours", type=int, default=24, help="Time window in hours")

    # WATCH subcommand
    watch_parser = subparsers.add_parser("watch", help="Watch alerts in real-time")
    watch_parser.add_argument(
        "--severity", choices=["INFO", "WARNING", "CRITICAL"], help="Filter by severity"
    )
    watch_parser.add_argument(
        "--poll-interval", type=int, default=5, help="Polling interval (seconds)"
    )
    watch_parser.add_argument("--verbose", "-v", action="store_true", help="Show full context")

    # STATUS subcommand
    status_parser = subparsers.add_parser("status", help="Show system status dashboard")
    status_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")

    # HEALTH subcommand (legacy)
    subparsers.add_parser("health", help="Run health checks")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging()

    # Initialize monitoring service
    try:
        monitoring = MonitoringService(settings)
    except Exception as e:
        print(f"Failed to initialize monitoring: {e}")
        sys.exit(1)

    # Execute command
    try:
        if args.command == "alerts":
            if not args.alerts_action:
                alerts_parser.print_help()
                sys.exit(1)
            cmd_alerts(monitoring, args)
        elif args.command == "metrics":
            if not args.metrics_action:
                metrics_parser.print_help()
                sys.exit(1)
            cmd_metrics(monitoring, args)
        elif args.command == "watch":
            cmd_watch(monitoring, args)
        elif args.command == "status":
            cmd_status(monitoring, args)
        elif args.command == "health":
            cmd_health(monitoring, args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
