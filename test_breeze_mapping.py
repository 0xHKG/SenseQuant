#!/usr/bin/env python3
"""Test if BreezeClient stock code mapping is working."""


import pandas as pd

from src.adapters.breeze_client import BreezeClient
from src.app.config import Settings

settings = Settings()

# Initialize BreezeClient
client = BreezeClient(
    api_key=settings.breeze_api_key,
    api_secret=settings.breeze_api_secret,
    session_token=settings.breeze_session_token,
    dry_run=False
)

print("Authenticating...")
client.authenticate()
print("✓ Authentication successful\n")

# Test fetching with RELIANCE (should map to RELIND internally)
print("Testing fetch_historical_chunk with RELIANCE...")
try:
    df = client.fetch_historical_chunk(
        symbol="RELIANCE",
        start_date=pd.Timestamp("2024-11-01", tz="UTC"),
        end_date=pd.Timestamp("2024-11-30", tz="UTC"),
        interval="1day"
    )
    print(f"✓ Success: {len(df)} records")
    if len(df) > 0:
        print(f"  Sample: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test fetching with TCS
print("Testing fetch_historical_chunk with TCS...")
try:
    df = client.fetch_historical_chunk(
        symbol="TCS",
        start_date=pd.Timestamp("2024-11-01", tz="UTC"),
        end_date=pd.Timestamp("2024-11-30", tz="UTC"),
        interval="1day"
    )
    print(f"✓ Success: {len(df)} records")
    if len(df) > 0:
        print(f"  Sample: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
