#!/usr/bin/env python3
"""
Test historical data availability for specific symbols via Breeze API.
Tests different date ranges to identify when data becomes available.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.adapters.breeze_client import BreezeClient

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

def test_symbol_availability(symbol: str, isec_code: str):
    """
    Test data availability for a symbol across different date ranges.
    """
    print(f"\n{'='*80}")
    print(f"Testing: {symbol} (ISEC: {isec_code})")
    print(f"{'='*80}")

    # Initialize Breeze client with credentials from .env
    client = BreezeClient(
        api_key=os.getenv("BREEZE_API_KEY"),
        api_secret=os.getenv("BREEZE_API_SECRET"),
        session_token=os.getenv("BREEZE_SESSION_TOKEN"),
        dry_run=False
    )
    client.authenticate()

    # Test different date ranges
    test_ranges = [
        ("2024-01-01", "2024-12-31", "Latest year (2024)"),
        ("2023-01-01", "2023-12-31", "Previous year (2023)"),
        ("2022-01-01", "2022-12-31", "2022"),
        ("2024-10-01", "2024-10-28", "Recent month"),
        ("2024-01-01", "2024-01-31", "Jan 2024 (1 month)"),
    ]

    results = []

    for start_str, end_str, description in test_ranges:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_str, "%Y-%m-%d").date()

        logger.info(f"Testing {description}: {start_str} to {end_str}")

        try:
            df = client.get_historical(
                symbol=symbol,
                stock_code=isec_code,
                exchange="NSE",
                from_date=start_date,
                to_date=end_date,
                interval="1day"
            )

            if df is not None and len(df) > 0:
                row_count = len(df)
                date_range = f"{df.index[0]} to {df.index[-1]}"
                status = f"✓ {row_count} rows"
                logger.info(f"  SUCCESS: {row_count} rows ({date_range})")
                results.append((description, True, row_count, date_range))
            else:
                status = "✗ 0 rows"
                logger.warning("  EMPTY: No data returned")
                results.append((description, False, 0, "N/A"))

        except Exception as e:
            status = f"✗ Error: {str(e)[:50]}"
            logger.error(f"  ERROR: {e}")
            results.append((description, False, 0, f"Error: {str(e)[:30]}"))

    # Print summary
    print(f"\n{'-'*80}")
    print(f"Summary for {symbol}:")
    print(f"{'-'*80}")
    for desc, success, rows, info in results:
        status_symbol = "✓" if success else "✗"
        print(f"  {status_symbol} {desc:25s}: {rows:5d} rows  {info}")
    print(f"{'='*80}\n")

    return results

def main():
    """Test the 4 problematic symbols."""

    symbols_to_test = [
        ("ADANIGREEN", "ADAGRE"),
        ("IDEA", "IDECEL"),
        ("APLAPOLLO", "APLAPO"),
        ("DIXON", "DIXTEC"),
    ]

    all_results = {}

    for symbol, isec_code in symbols_to_test:
        results = test_symbol_availability(symbol, isec_code)
        all_results[symbol] = results

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - DATA AVAILABILITY FOR 4 SYMBOLS")
    print(f"{'='*80}")

    for symbol in all_results:
        any_success = any(success for _, success, _, _ in all_results[symbol])
        status = "✓ DATA AVAILABLE" if any_success else "✗ NO DATA FOUND"
        print(f"\n{symbol:12s}: {status}")

        if any_success:
            successful_ranges = [(desc, rows) for desc, success, rows, _ in all_results[symbol] if success]
            for desc, rows in successful_ranges:
                print(f"  - {desc}: {rows} rows")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
