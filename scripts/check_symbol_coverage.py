"""Check symbol coverage for NIFTY100 data readiness audit.

US-028 Phase 7 Initiative 1: Audit historical data availability before bulk ingestion.

This script checks:
1. Whether historical OHLCV data exists locally for each symbol
2. If missing, attempts a minimal test fetch (1 recent trading day)
3. Records status: "ok", "fetched", "missing_api_data", "error"
4. Logs results to coverage_report_<timestamp>.jsonl

Usage:
    python scripts/check_symbol_coverage.py --constituents-file data/historical/metadata/nifty100_constituents.json
    python scripts/check_symbol_coverage.py --symbols RELIANCE TCS INFY --test-fetch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_local_coverage(
    symbol: str, data_dir: Path, intervals: list[str] = None
) -> dict[str, Any]:
    """Check if historical data exists locally for a symbol.

    Args:
        symbol: Stock symbol to check
        data_dir: Root data directory (data/historical)
        intervals: List of intervals to check (default: ['1minute', '5minute', '1day'])

    Returns:
        Dict with status and details
    """
    if intervals is None:
        intervals = ["1minute", "5minute", "1day"]

    symbol_dir = data_dir / symbol
    if not symbol_dir.exists():
        return {
            "status": "missing_local",
            "reason": "Symbol directory does not exist",
            "intervals_found": [],
        }

    intervals_found = []
    total_files = 0
    total_rows = 0

    for interval in intervals:
        interval_dir = symbol_dir / interval
        if interval_dir.exists():
            csv_files = list(interval_dir.glob("*.csv"))
            if csv_files:
                intervals_found.append(interval)
                total_files += len(csv_files)

                # Count rows in sample file (first file)
                sample_file = csv_files[0]
                try:
                    with open(sample_file) as f:
                        rows = len(f.readlines()) - 1  # -1 for header
                        total_rows += rows
                except Exception:
                    pass

    if not intervals_found:
        return {
            "status": "missing_local",
            "reason": "No data files found",
            "intervals_found": [],
        }

    return {
        "status": "ok",
        "intervals_found": intervals_found,
        "total_files": total_files,
        "sample_rows": total_rows,
    }


def test_fetch_symbol(
    symbol: str, breeze_client: Any = None, rate_limit_delay: float = 2.0
) -> dict[str, Any]:
    """Attempt a minimal test fetch for a symbol (1 recent trading day).

    Args:
        symbol: Stock symbol to test
        breeze_client: Initialized BreezeClient instance
        rate_limit_delay: Seconds to wait after fetch (respect rate limits)

    Returns:
        Dict with fetch status and details
    """
    if breeze_client is None:
        logger.warning(f"No BreezeClient provided, skipping test fetch for {symbol}")
        return {"status": "skip", "reason": "No API client"}

    try:
        # Fetch 1 day of 1minute data (recent trading day)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Look back 7 days to find a trading day

        logger.info(f"Test fetching {symbol} from {start_date.date()} to {end_date.date()}")

        # Use the fetch_historical_chunk method
        bars = breeze_client.fetch_historical_chunk(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="1minute",
        )

        # Rate limiting
        time.sleep(rate_limit_delay)

        if bars is None or bars.empty:
            return {
                "status": "missing_api_data",
                "reason": "API returned no data",
                "bars_fetched": 0,
            }

        return {
            "status": "fetched",
            "bars_fetched": len(bars),
            "date_range": f"{bars['timestamp'].min()} to {bars['timestamp'].max()}",
        }

    except Exception as e:
        logger.error(f"Test fetch failed for {symbol}: {e}")
        return {"status": "error", "reason": str(e)}


def audit_symbol(
    symbol: str,
    data_dir: Path,
    breeze_client: Any = None,
    test_fetch: bool = False,
    rate_limit_delay: float = 2.0,
) -> dict[str, Any]:
    """Audit a single symbol's data coverage.

    Args:
        symbol: Stock symbol to audit
        data_dir: Root data directory
        breeze_client: BreezeClient instance (if test_fetch enabled)
        test_fetch: Whether to attempt test fetch if data missing
        rate_limit_delay: Seconds to wait between fetches

    Returns:
        Dict with audit results
    """
    logger.info(f"Auditing {symbol}...")

    # Check local coverage
    local_result = check_local_coverage(symbol, data_dir)

    if local_result["status"] == "ok":
        return {
            "symbol": symbol,
            "status": "ok",
            "local_data": local_result,
            "test_fetch": None,
        }

    # If data missing and test_fetch enabled, try fetching
    if test_fetch and breeze_client:
        fetch_result = test_fetch_symbol(symbol, breeze_client, rate_limit_delay)
        return {
            "symbol": symbol,
            "status": fetch_result["status"],
            "local_data": local_result,
            "test_fetch": fetch_result,
        }

    return {
        "symbol": symbol,
        "status": "missing_local",
        "local_data": local_result,
        "test_fetch": None,
    }


def run_audit(
    symbols: list[str],
    data_dir: Path,
    output_dir: Path,
    test_fetch: bool = False,
    rate_limit_delay: float = 2.0,
    breeze_api_key: str | None = None,
    breeze_api_secret: str | None = None,
    breeze_session_token: str | None = None,
) -> tuple[Path, dict[str, int]]:
    """Run coverage audit across all symbols.

    Args:
        symbols: List of symbols to audit
        data_dir: Root data directory
        output_dir: Output directory for coverage report
        test_fetch: Whether to test fetch missing symbols
        rate_limit_delay: Seconds between API requests
        breeze_api_key: Breeze API key (if test_fetch enabled)
        breeze_api_secret: Breeze API secret (if test_fetch enabled)
        breeze_session_token: Breeze session token (if test_fetch enabled)

    Returns:
        Tuple of (report_path, status_counts)
    """
    # Initialize Breeze client if test_fetch enabled
    breeze_client = None
    if test_fetch:
        if not all([breeze_api_key, breeze_api_secret, breeze_session_token]):
            logger.warning(
                "Test fetch enabled but Breeze credentials missing. Loading from .env..."
            )
            from src.app.config import Settings

            settings = Settings()
            breeze_api_key = settings.breeze_api_key
            breeze_api_secret = settings.breeze_api_secret
            breeze_session_token = settings.breeze_session_token

        try:
            from src.adapters.breeze_client import BreezeClient

            breeze_client = BreezeClient(
                api_key=breeze_api_key,
                api_secret=breeze_api_secret,
                session_token=breeze_session_token,
                dry_run=False,
            )
            breeze_client.authenticate()
            logger.info("Breeze API client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Breeze client: {e}")
            test_fetch = False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate report file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"coverage_report_{timestamp}.jsonl"

    logger.info(f"Starting audit of {len(symbols)} symbols")
    logger.info(f"Report will be saved to: {report_path}")

    status_counts = {
        "ok": 0,
        "missing_local": 0,
        "fetched": 0,
        "missing_api_data": 0,
        "error": 0,
    }

    with open(report_path, "w") as f:
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Auditing {symbol}")

            result = audit_symbol(
                symbol=symbol,
                data_dir=data_dir,
                breeze_client=breeze_client,
                test_fetch=test_fetch,
                rate_limit_delay=rate_limit_delay,
            )

            # Add timestamp
            result["timestamp"] = datetime.now().isoformat()

            # Write to JSONL
            f.write(json.dumps(result) + "\n")

            # Update counts
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

            logger.info(f"  Status: {status}")

    logger.info(f"Audit complete. Report saved to {report_path}")
    return report_path, status_counts


def generate_summary(report_path: Path) -> dict[str, Any]:
    """Generate summary statistics from coverage report.

    Args:
        report_path: Path to coverage report JSONL file

    Returns:
        Dict with summary statistics
    """
    status_counts = {}
    symbols_by_status = {}
    total_symbols = 0

    with open(report_path) as f:
        for line in f:
            entry = json.loads(line)
            symbol = entry["symbol"]
            status = entry["status"]

            total_symbols += 1
            status_counts[status] = status_counts.get(status, 0) + 1

            if status not in symbols_by_status:
                symbols_by_status[status] = []
            symbols_by_status[status].append(symbol)

    summary = {
        "total_symbols": total_symbols,
        "status_counts": status_counts,
        "symbols_by_status": symbols_by_status,
        "coverage_rate": (
            status_counts.get("ok", 0) + status_counts.get("fetched", 0)
        )
        / total_symbols
        * 100
        if total_symbols > 0
        else 0,
    }

    return summary


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check symbol coverage for NIFTY100 data readiness audit"
    )

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--constituents-file",
        type=str,
        help="Path to NIFTY100 constituents JSON file",
    )
    group.add_argument(
        "--symbols", nargs="+", help="List of symbols to audit (space-separated)"
    )

    # Audit options
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/historical",
        help="Root data directory (default: data/historical)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/historical/metadata",
        help="Output directory for coverage report (default: data/historical/metadata)",
    )
    parser.add_argument(
        "--test-fetch",
        action="store_true",
        help="Test fetch missing symbols via Breeze API (1 recent day)",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=2.0,
        help="Delay between API requests in seconds (default: 2.0)",
    )

    # API credentials (optional, will load from .env if not provided)
    parser.add_argument("--breeze-api-key", type=str, help="Breeze API key")
    parser.add_argument("--breeze-api-secret", type=str, help="Breeze API secret")
    parser.add_argument("--breeze-session-token", type=str, help="Breeze session token")

    args = parser.parse_args()

    # Load symbols
    if args.constituents_file:
        logger.info(f"Loading symbols from {args.constituents_file}")
        with open(args.constituents_file) as f:
            constituents = json.load(f)
            symbols = constituents.get("symbols", [])
            logger.info(f"Loaded {len(symbols)} symbols from constituents file")
    else:
        symbols = args.symbols
        logger.info(f"Using {len(symbols)} symbols from command line")

    # Run audit
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    report_path, status_counts = run_audit(
        symbols=symbols,
        data_dir=data_dir,
        output_dir=output_dir,
        test_fetch=args.test_fetch,
        rate_limit_delay=args.rate_limit_delay,
        breeze_api_key=args.breeze_api_key,
        breeze_api_secret=args.breeze_api_secret,
        breeze_session_token=args.breeze_session_token,
    )

    # Generate summary
    summary = generate_summary(report_path)

    logger.info("=" * 80)
    logger.info("COVERAGE AUDIT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total symbols: {summary['total_symbols']}")
    logger.info(f"Coverage rate: {summary['coverage_rate']:.1f}%")
    logger.info("")
    logger.info("Status counts:")
    for status, count in summary["status_counts"].items():
        logger.info(f"  {status}: {count}")
    logger.info("")

    # Save summary
    summary_path = output_dir / f"coverage_summary_{report_path.stem.split('_', 2)[2]}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
