#!/usr/bin/env python3
"""Inspect cached historical data CSVs."""

import pandas as pd
from pathlib import Path

data_dir = Path("data/historical")
symbols = ["RELIANCE", "TCS"]

for symbol in symbols:
    print(f"\n{'='*70}")
    print(f"Symbol: {symbol}")
    print('='*70)

    symbol_dir = data_dir / symbol / "1day"

    if not symbol_dir.exists():
        print(f"❌ Directory not found: {symbol_dir}")
        continue

    # Collect all CSV files
    csv_files = sorted(symbol_dir.glob("*.csv"))

    if not csv_files:
        print(f"❌ No CSV files found in {symbol_dir}")
        continue

    print(f"Total CSV files: {len(csv_files)}")

    # Load all data
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"⚠ Error reading {csv_file.name}: {e}")

    if not all_data:
        print("❌ No data could be loaded")
        continue

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'], format='ISO8601')
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    print(f"Total rows: {len(combined)}")
    print(f"\nDate range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

    # Show first 5 rows
    print(f"\nFirst 5 timestamps:")
    print(combined[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head(5).to_string(index=False))

    # Show last 5 rows
    print(f"\nLast 5 timestamps:")
    print(combined[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(5).to_string(index=False))

    # Check for gaps
    combined['date'] = combined['timestamp'].dt.date
    dates = sorted(combined['date'].unique())

    gaps = []
    for i in range(len(dates) - 1):
        current = pd.Timestamp(dates[i])
        next_date = pd.Timestamp(dates[i + 1])
        diff_days = (next_date - current).days

        # Count business days (Mon-Fri) in gap
        if diff_days > 5:  # More than a week gap
            gaps.append((dates[i], dates[i + 1], diff_days))

    if gaps:
        print(f"\n⚠ Found {len(gaps)} significant gaps (>5 days):")
        for start, end, days in gaps[:5]:  # Show first 5 gaps
            print(f"  {start} → {end} ({days} days)")
    else:
        print("\n✓ No significant gaps found")

    # Check for NaN values
    nan_counts = combined.isna().sum()
    if nan_counts.any():
        print(f"\n⚠ NaN values found:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  {col}: {count} NaNs ({count/len(combined)*100:.1f}%)")
    else:
        print("\n✓ No NaN values")

    # Basic stats
    print(f"\nPrice statistics:")
    print(f"  Open:  min={combined['open'].min():.2f}, max={combined['open'].max():.2f}, mean={combined['open'].mean():.2f}")
    print(f"  Close: min={combined['close'].min():.2f}, max={combined['close'].max():.2f}, mean={combined['close'].mean():.2f}")
    print(f"  Volume: min={combined['volume'].min():.0f}, max={combined['volume'].max():.0f}, mean={combined['volume'].mean():.0f}")

print(f"\n{'='*70}\n")
