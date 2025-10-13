#!/usr/bin/env python3
"""Test different stock codes with Breeze API."""

from breeze_connect import BreezeConnect
from src.app.config import Settings

settings = Settings()

# Initialize and authenticate
breeze = BreezeConnect(api_key=settings.breeze_api_key)
breeze.generate_session(
    api_secret=settings.breeze_api_secret,
    session_token=settings.breeze_session_token
)
print("✓ Authentication successful\n")

# Test different stock codes
test_cases = [
    ("RELIANCE", "NSE"),
    ("RELIND", "NSE"),
    ("TCS", "NSE"),
]

for stock_code, exchange in test_cases:
    print(f"Testing {stock_code} on {exchange}...")
    try:
        result = breeze.get_historical_data_v2(
            interval="1day",
            from_date="2024-11-01T07:00:00.000Z",
            to_date="2024-11-30T07:00:00.000Z",
            stock_code=stock_code,
            exchange_code=exchange,
            product_type="cash"
        )

        if isinstance(result, dict):
            success_data = result.get('Success', [])
            status = result.get('Status')
            error = result.get('Error')

            print(f"  Status: {status}")
            print(f"  Records: {len(success_data)}")
            if error:
                print(f"  Error: {error}")
            if success_data:
                print(f"  Sample: {success_data[0]}")
        else:
            print(f"  Unexpected type: {type(result)}")
            print(f"  Value: {result}")
    except Exception as e:
        print(f"  Exception: {e}")
    print()
