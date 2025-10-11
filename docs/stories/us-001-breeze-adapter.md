# US-001 — Breeze Adapter & Config

## Goal
Wrap ICICI Breeze API for auth, quotes, historical bars, and order placement. Provide .env-driven config.

## Context (PRD/Arch)
- Must support dry-run and live modes
- Expose: authenticate(), get_ltp(symbol), get_historical(symbol, interval, from, to), place_order(args)
- Handle errors, rate limits, reconnection; log all API ops

## Tasks
1. Implement src/config.py to load env (API keys, symbols, risk, mode)
2. Implement src/breeze_client.py with BreezeClient wrapper
3. Add basic tests with mocks for Breeze client methods

## Acceptance
- AC1: `BreezeClient.authenticate()` establishes session without hardcoding secrets
- AC2: `get_ltp("RELIANCE")` returns float (mocked in tests)
- AC3: `get_historical(..., interval="1minute")` returns pandas DataFrame with datetime index
- AC4: `place_order(..., dry_run=True)` logs intent instead of hitting API
- AC5: errors are caught and logged; methods never crash caller

