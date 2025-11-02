#!/usr/bin/env python3
"""Quick test of Breeze API to diagnose HTTP 500 errors."""

from breeze_connect import BreezeConnect

from src.app.config import Settings

settings = Settings()

print(f"Session token: {settings.breeze_session_token}")
print(f"Mode: {settings.mode}")
print()

# Initialize
breeze = BreezeConnect(api_key=settings.breeze_api_key)

# Authenticate
print("Authenticating...")
try:
    breeze.generate_session(
        api_secret=settings.breeze_api_secret,
        session_token=settings.breeze_session_token
    )
    print("✓ Authentication successful")
except Exception as e:
    print(f"✗ Authentication failed: {e}")
    exit(1)

# Test historical data v1
print("\nTesting get_historical_data (v1)...")
try:
    result = breeze.get_historical_data(
        interval="1day",
        from_date="2024-11-01T07:00:00.000Z",
        to_date="2024-11-30T07:00:00.000Z",
        stock_code="RELIANCE",
        exchange_code="NSE",
        product_type="cash"
    )
    print(f"✓ v1 Success: {type(result)} with {len(result.get('Success', [])) if isinstance(result, dict) else 'unknown'} records")
except Exception as e:
    print(f"✗ v1 Failed: {e}")

# Test historical data v2
print("\nTesting get_historical_data_v2 (v2)...")
try:
    result = breeze.get_historical_data_v2(
        interval="1day",
        from_date="2024-11-01T07:00:00.000Z",
        to_date="2024-11-30T07:00:00.000Z",
        stock_code="RELIANCE",
        exchange_code="NSE",
        product_type="cash"
    )
    print(f"✓ v2 Success: {type(result)} with {len(result.get('Success', [])) if isinstance(result, dict) else 'unknown'} records")
except Exception as e:
    print(f"✗ v2 Failed: {e}")
