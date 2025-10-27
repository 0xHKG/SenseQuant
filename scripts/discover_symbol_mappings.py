"""Discover Breeze API symbol mappings using get_names() method.

US-028 Phase 7 Initiative 1: Build comprehensive NSE→ISEC stock code mapping.

This script:
1. Loads symbols from nifty100_constituents.json
2. Queries Breeze API using get_names() for each symbol
3. Captures the ISEC stock code returned by the API
4. Generates symbol_mappings.json with verified mappings
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.breeze_client import BreezeClient
from src.app.config import Settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def query_symbol_info(
    breeze_client: Any, symbol: str, exchange: str = "NSE"
) -> dict[str, Any]:
    """Query Breeze API for symbol information using get_names().

    Args:
        breeze_client: Authenticated BreezeClient instance
        symbol: NSE stock symbol (e.g., "RELIANCE", "TCS")
        exchange: Exchange code (default: "NSE")

    Returns:
        Dict with symbol info:
        {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "status": "success" | "error",
            "isec_code": "RELIND",  # if available
            "response": {...},  # raw API response
            "error": "..."  # if status == "error"
        }
    """
    try:
        logger.info(f"Querying symbol info for {symbol} on {exchange}")

        # Call get_names() method
        # Response format (actual from API):
        # {
        #   'exchange_code': 'NSE',
        #   'exchange_stock_code': 'INFY',
        #   'isec_stock_code': 'INFTEC',
        #   'isec_token': '1594',
        #   'company name': 'INFOSYS LTD',
        #   ...
        # }
        response = breeze_client._client.get_names(
            exchange_code=exchange, stock_code=symbol
        )

        # Check if response is valid (has isec_stock_code)
        if not isinstance(response, dict) or "isec_stock_code" not in response:
            error_msg = str(response) if response else "No response"
            logger.warning(f"Invalid API response for {symbol}: {error_msg}")
            return {
                "symbol": symbol,
                "exchange": exchange,
                "status": "error",
                "error": f"Invalid response: {error_msg}",
                "response": response,
            }

        # Extract ISEC stock code from response
        isec_code = response.get("isec_stock_code", symbol)

        logger.info(f"  {symbol} -> {isec_code} (ISEC code)")

        return {
            "symbol": symbol,
            "exchange": exchange,
            "status": "success",
            "isec_code": isec_code,
            "company_name": response.get("company name", ""),
            "isec_token": response.get("isec_token", ""),
            "response": response,
        }

    except Exception as e:
        logger.error(f"Exception querying {symbol}: {e}")
        return {
            "symbol": symbol,
            "exchange": exchange,
            "status": "error",
            "error": str(e),
        }


def discover_mappings(
    symbols: list[str],
    breeze_api_key: str | None = None,
    breeze_api_secret: str | None = None,
    breeze_session_token: str | None = None,
    rate_limit_delay: float = 1.0,
) -> dict[str, Any]:
    """Discover symbol mappings for all symbols.

    Args:
        symbols: List of NSE symbols to query
        breeze_api_key: Breeze API key (if None, load from .env)
        breeze_api_secret: Breeze API secret (if None, load from .env)
        breeze_session_token: Breeze session token (if None, load from .env)
        rate_limit_delay: Seconds to wait between queries

    Returns:
        Dict with mapping results
    """
    # Initialize Breeze client
    if not all([breeze_api_key, breeze_api_secret, breeze_session_token]):
        logger.warning("Breeze credentials missing. Loading from .env...")
        settings = Settings()
        breeze_api_key = settings.breeze_api_key
        breeze_api_secret = settings.breeze_api_secret
        breeze_session_token = settings.breeze_session_token

    breeze_client = BreezeClient(
        api_key=breeze_api_key,
        api_secret=breeze_api_secret,
        session_token=breeze_session_token,
        dry_run=False,  # Need live API access
    )
    breeze_client.authenticate()  # Establish session
    logger.info("Breeze API client initialized and authenticated")

    # Query each symbol
    results = []
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Querying {symbol}")
        result = query_symbol_info(breeze_client, symbol)
        results.append(result)

        # Rate limiting
        if i < len(symbols):
            time.sleep(rate_limit_delay)

    # Build mapping dictionary
    mapping = {}
    success_count = 0
    error_count = 0

    for result in results:
        if result["status"] == "success":
            nse_symbol = result["symbol"]
            isec_code = result["isec_code"]

            # Only add to mapping if codes are different
            if nse_symbol != isec_code:
                mapping[nse_symbol] = isec_code
                logger.info(f"  Mapping: {nse_symbol} -> {isec_code}")
            else:
                logger.info(f"  {nse_symbol} uses same code (no mapping needed)")

            success_count += 1
        else:
            error_count += 1
            logger.warning(f"  Failed: {result['symbol']} - {result.get('error', 'Unknown')}")

    logger.info(f"\nDiscovery complete:")
    logger.info(f"  Total symbols: {len(symbols)}")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Mappings found: {len(mapping)}")

    return {
        "total_symbols": len(symbols),
        "success_count": success_count,
        "error_count": error_count,
        "mappings": mapping,
        "all_results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Discover Breeze API symbol mappings using get_names()"
    )
    parser.add_argument(
        "--constituents-file",
        type=str,
        default="data/historical/metadata/nifty100_constituents.json",
        help="Path to NIFTY100 constituents file",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (overrides constituents file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/historical/metadata/symbol_mappings.json",
        help="Output file for symbol mappings",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between API queries (default: 1.0)",
    )

    args = parser.parse_args()

    # Load symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
        logger.info(f"Using {len(symbols)} symbols from command line")
    else:
        constituents_path = Path(args.constituents_file)
        if not constituents_path.exists():
            logger.error(f"Constituents file not found: {constituents_path}")
            sys.exit(1)

        with open(constituents_path) as f:
            constituents = json.load(f)

        symbols = constituents.get("symbols", [])
        logger.info(f"Loaded {len(symbols)} symbols from {constituents_path}")

    if not symbols:
        logger.error("No symbols to query")
        sys.exit(1)

    # Discover mappings
    discovery_results = discover_mappings(
        symbols=symbols, rate_limit_delay=args.rate_limit_delay
    )

    # Create output structure
    output = {
        "last_updated": "2025-10-15",
        "source": "Breeze API get_names() method",
        "discovery_stats": {
            "total_symbols": discovery_results["total_symbols"],
            "success_count": discovery_results["success_count"],
            "error_count": discovery_results["error_count"],
            "mappings_found": len(discovery_results["mappings"]),
        },
        "mappings": discovery_results["mappings"],
        "symbol_details": [
            {
                "nse_symbol": r["symbol"],
                "isec_code": r.get("isec_code", r["symbol"]),
                "isec_token": r.get("isec_token", ""),
                "company_name": r.get("company_name", ""),
                "status": r["status"],
                "error": r.get("error", None),
            }
            for r in discovery_results["all_results"]
        ],
    }

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nSymbol mappings saved to: {output_path}")
    logger.info(f"Total mappings: {len(discovery_results['mappings'])}")

    # Print summary
    print("\n" + "=" * 80)
    print("SYMBOL MAPPING DISCOVERY SUMMARY")
    print("=" * 80)
    print(f"Total symbols queried: {discovery_results['total_symbols']}")
    print(f"Successful queries: {discovery_results['success_count']}")
    print(f"Failed queries: {discovery_results['error_count']}")
    print(f"Mappings found (NSE != ISEC): {len(discovery_results['mappings'])}")
    print(f"\nOutput file: {output_path}")
    print("=" * 80)

    if discovery_results["mappings"]:
        print("\nMappings found:")
        for nse, isec in sorted(discovery_results["mappings"].items()):
            print(f"  {nse:15} -> {isec}")


if __name__ == "__main__":
    main()
