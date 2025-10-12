#!/usr/bin/env python3
"""CLI entry point for running backtests.

Usage:
    python scripts/backtest.py --symbols RELIANCE TCS --start-date 2024-01-01 --end-date 2024-12-31 --strategy swing
    python scripts/backtest.py --symbols INFY --start-date 2024-06-01 --end-date 2024-12-31 --strategy both --initial-capital 500000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.adapters.breeze_client import BreezeClient
from src.app.config import settings
from src.domain.types import BacktestConfig
from src.services.backtester import Backtester
from src.services.data_feed import BreezeDataFeed, CSVDataFeed, HybridDataFeed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtests on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest swing strategy on RELIANCE
  python scripts/backtest.py --symbols RELIANCE --start-date 2024-01-01 --end-date 2024-12-31 --strategy swing

  # Backtest both strategies on multiple symbols
  python scripts/backtest.py --symbols RELIANCE TCS INFY --start-date 2024-01-01 --end-date 2024-12-31 --strategy both

  # Use CSV data source
  python scripts/backtest.py --symbols TEST --start-date 2024-01-01 --end-date 2024-12-31 --strategy swing --data-source csv --csv data/historical

  # Use hybrid data source (API with cache fallback)
  python scripts/backtest.py --symbols RELIANCE --start-date 2024-01-01 --end-date 2024-12-31 --strategy swing --data-source hybrid --csv data/historical

  # Export results
  python scripts/backtest.py --symbols RELIANCE --start-date 2024-01-01 --end-date 2024-12-31 --strategy swing --export results/
        """,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Stock symbols to backtest (space-separated)",
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Backtest start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        required=True,
        help="Backtest end date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--strategy",
        choices=["intraday", "swing", "both"],
        default="swing",
        help="Strategy to backtest (default: swing)",
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1000000.0,
        help="Initial capital in INR (default: 1000000)",
    )

    parser.add_argument(
        "--data-source",
        choices=["breeze", "csv", "teacher", "hybrid"],
        default="breeze",
        help="Data source for historical bars (default: breeze)",
    )

    parser.add_argument(
        "--csv",
        "--csv-directory",
        dest="csv_directory",
        help="Path to CSV directory (for csv/hybrid data sources)",
    )

    parser.add_argument(
        "--csv-path",
        help="(Deprecated) Use --csv-directory instead",
    )

    parser.add_argument(
        "--interval",
        choices=["1minute", "5minute", "1day"],
        default="1day",
        help="Bar interval for data fetch (default: 1day)",
    )

    parser.add_argument(
        "--minute-data",
        action="store_true",
        help="Enable minute-level backtesting (shortcut for --interval 1minute) (US-018)",
    )

    parser.add_argument(
        "--export",
        help="Export results to specified directory",
    )

    parser.add_argument(
        "--teacher-labels",
        help="Path to Teacher labels CSV if data-source=teacher",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    # Telemetry & Accuracy Audit flags
    parser.add_argument(
        "--enable-telemetry",
        action="store_true",
        help="Enable prediction telemetry capture for accuracy analysis",
    )

    parser.add_argument(
        "--telemetry-dir",
        type=str,
        help="Custom directory for telemetry files (default: data/analytics)",
    )

    parser.add_argument(
        "--telemetry-sample-rate",
        type=float,
        help="Telemetry sampling rate 0.0-1.0 (default: from settings)",
    )

    parser.add_argument(
        "--export-metrics",
        action="store_true",
        help="Export accuracy metrics after backtest completion",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Configure loguru logging."""
    logger.remove()  # Remove default handler

    log_level = "DEBUG" if verbose else "INFO"

    # Console handler with structured format
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[component]}</cyan> | <level>{message}</level>",
    )

    # File handler for backtests
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "backtest_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[component]} | {message}",
    )


def validate_config(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    # Validate date format
    from datetime import datetime

    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}", extra={"component": "backtest_cli"})
        sys.exit(1)

    # Validate start < end
    if args.start_date >= args.end_date:
        logger.error(
            "Start date must be before end date",
            extra={"component": "backtest_cli"},
        )
        sys.exit(1)

    # Validate data source specific args
    if args.data_source in ("csv", "hybrid"):
        csv_dir = args.csv_directory or args.csv_path
        if not csv_dir:
            logger.error(
                f"--csv-directory required when --data-source={args.data_source}",
                extra={"component": "backtest_cli"},
            )
            sys.exit(1)

        # Update args to use csv_directory consistently
        if args.csv_path and not args.csv_directory:
            logger.warning(
                "--csv-path is deprecated, use --csv-directory instead",
                extra={"component": "backtest_cli"},
            )
            args.csv_directory = args.csv_path

    if args.data_source == "teacher" and not args.teacher_labels:
        logger.error(
            "--teacher-labels required when --data-source=teacher",
            extra={"component": "backtest_cli"},
        )
        sys.exit(1)

    # Validate directories exist (for CSV/hybrid)
    if args.csv_directory:
        csv_path = Path(args.csv_directory)
        if not csv_path.exists():
            logger.warning(
                f"CSV directory does not exist: {args.csv_directory}",
                extra={"component": "backtest_cli"},
            )

    if args.teacher_labels and not Path(args.teacher_labels).exists():
        logger.error(
            f"Teacher labels file not found: {args.teacher_labels}",
            extra={"component": "backtest_cli"},
        )
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger.info(
        "Starting backtest",
        extra={
            "component": "backtest_cli",
            "symbols": args.symbols,
            "start": args.start_date,
            "end": args.end_date,
            "strategy": args.strategy,
            "capital": args.initial_capital,
        },
    )

    # Validate configuration
    validate_config(args)

    # Handle --minute-data convenience flag (US-018)
    interval = args.interval
    if args.minute_data:
        interval = "1minute"
        logger.info(
            "Minute-data mode enabled, using 1-minute bars",
            extra={"component": "backtest_cli"},
        )

    # Create backtest config
    config = BacktestConfig(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        strategy=args.strategy,
        initial_capital=args.initial_capital,
        data_source=args.data_source,
        random_seed=args.random_seed,
        csv_path=args.csv_path,
        teacher_labels_path=args.teacher_labels,
        resolution=interval,  # US-018: Bar resolution
    )

    # Initialize data feed based on data source
    data_feed = None
    client = None

    if config.data_source == "csv":
        logger.info(
            f"Using CSV data feed from: {args.csv_directory}",
            extra={"component": "backtest_cli"},
        )
        data_feed = CSVDataFeed(args.csv_directory)

    elif config.data_source == "breeze":
        logger.info("Authenticating with Breeze API", extra={"component": "backtest_cli"})
        client = BreezeClient(
            api_key=settings.breeze_api_key,
            api_secret=settings.breeze_api_secret,
            session_token=settings.breeze_session_token,
            dry_run=True,
        )
        client.authenticate()
        data_feed = BreezeDataFeed(client, settings)

    elif config.data_source == "hybrid":
        logger.info(
            f"Using hybrid data feed (Breeze API + CSV cache: {args.csv_directory})",
            extra={"component": "backtest_cli"},
        )
        client = BreezeClient(
            api_key=settings.breeze_api_key,
            api_secret=settings.breeze_api_secret,
            session_token=settings.breeze_session_token,
            dry_run=True,
        )
        client.authenticate()

        # Temporarily update settings for hybrid mode
        original_csv_dir = settings.data_feed_csv_directory
        settings.data_feed_csv_directory = args.csv_directory

        data_feed = HybridDataFeed(client, settings)

        # Restore original setting
        settings.data_feed_csv_directory = original_csv_dir

    elif config.data_source == "teacher":
        logger.error(
            "Teacher data source not yet implemented",
            extra={"component": "backtest_cli"},
        )
        sys.exit(1)

    # Override telemetry sample rate if specified
    if args.telemetry_sample_rate is not None:
        if 0.0 <= args.telemetry_sample_rate <= 1.0:
            settings.telemetry_sample_rate = args.telemetry_sample_rate
        else:
            logger.warning(
                f"Invalid telemetry sample rate {args.telemetry_sample_rate}, using default",
                extra={"component": "backtest_cli"},
            )

    # Prepare telemetry directory
    telemetry_dir = None
    if args.enable_telemetry:
        if args.telemetry_dir:
            telemetry_dir = Path(args.telemetry_dir)
        else:
            # Use timestamped directory under default path
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            telemetry_dir = Path(settings.telemetry_storage_path) / f"backtest_{timestamp}"

        telemetry_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Telemetry enabled, output directory: {telemetry_dir}",
            extra={"component": "backtest_cli"},
        )

    # Create backtester with DataFeed and telemetry
    backtester = Backtester(
        config=config,
        client=client,  # For backward compatibility
        data_feed=data_feed,
        settings=settings,
        enable_telemetry=args.enable_telemetry,
        telemetry_dir=telemetry_dir,
    )

    # Run backtest
    try:
        logger.info("Running backtest...", extra={"component": "backtest_cli"})
        result = backtester.run()

        # Print summary
        logger.info("=" * 80, extra={"component": "backtest_cli"})
        logger.info("BACKTEST COMPLETE", extra={"component": "backtest_cli"})
        logger.info("=" * 80, extra={"component": "backtest_cli"})

        logger.info(
            f"Symbols: {', '.join(config.symbols)}",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"Period: {config.start_date} to {config.end_date}",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"Strategy: {config.strategy}",
            extra={"component": "backtest_cli"},
        )
        logger.info("", extra={"component": "backtest_cli"})

        logger.info("PERFORMANCE METRICS:", extra={"component": "backtest_cli"})
        logger.info("-" * 80, extra={"component": "backtest_cli"})

        metrics = result.metrics
        logger.info(
            f"Total Return:       {metrics['total_return_pct']:.2f}%",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"CAGR:               {metrics['cagr_pct']:.2f}%",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"Win Rate:           {metrics['win_rate_pct']:.2f}%",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"Total Trades:       {metrics['total_trades']:.0f}",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"Avg Win:            ₹{metrics['avg_win']:.2f}",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"Avg Loss:           ₹{metrics['avg_loss']:.2f}",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"Exposure:           {metrics['exposure_pct']:.2f}%",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"Total Fees:         ₹{metrics['total_fees']:.2f}",
            extra={"component": "backtest_cli"},
        )
        logger.info("", extra={"component": "backtest_cli"})

        logger.info("ARTIFACTS SAVED:", extra={"component": "backtest_cli"})
        logger.info("-" * 80, extra={"component": "backtest_cli"})
        logger.info(
            f"Summary:      {result.summary_path}",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"Equity Curve: {result.equity_path}",
            extra={"component": "backtest_cli"},
        )
        logger.info(
            f"Trades Log:   {result.trades_path}",
            extra={"component": "backtest_cli"},
        )

        # Export accuracy metrics if requested and telemetry was enabled
        if args.export_metrics and args.enable_telemetry and telemetry_dir:
            try:
                from src.services.accuracy_analyzer import AccuracyAnalyzer

                logger.info(
                    "Exporting accuracy metrics...",
                    extra={"component": "backtest_cli"},
                )

                analyzer = AccuracyAnalyzer()
                traces = analyzer.load_traces(telemetry_dir)

                if traces:
                    metrics_data = analyzer.compute_metrics(traces)
                    metrics_path = telemetry_dir / "accuracy_metrics.json"
                    analyzer.export_report(metrics_data, metrics_path)

                    logger.info(
                        f"Accuracy Metrics: {metrics_path}",
                        extra={"component": "backtest_cli"},
                    )
                    logger.info(
                        f"  - Precision (LONG): {metrics_data.precision.get('LONG', 0.0):.2%}",
                        extra={"component": "backtest_cli"},
                    )
                    logger.info(
                        f"  - Hit Ratio: {metrics_data.hit_ratio:.2%}",
                        extra={"component": "backtest_cli"},
                    )
                    logger.info(
                        f"  - Total Trades: {metrics_data.total_trades}",
                        extra={"component": "backtest_cli"},
                    )
                else:
                    logger.warning(
                        "No telemetry traces found to analyze",
                        extra={"component": "backtest_cli"},
                    )

            except Exception as e:
                logger.error(
                    f"Failed to export accuracy metrics: {e}",
                    extra={"component": "backtest_cli"},
                )

        logger.info("=" * 80, extra={"component": "backtest_cli"})

    except Exception as e:
        logger.exception(
            f"Backtest failed: {e}",
            extra={"component": "backtest_cli"},
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
