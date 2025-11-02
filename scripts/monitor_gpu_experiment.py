#!/usr/bin/env python3
"""GPU utilization monitor for training experiments.

Samples nvidia-smi periodically and logs GPU metrics for later analysis.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_nvidia_smi() -> dict[str, dict[str, str]]:
    """Parse nvidia-smi output and extract GPU metrics.

    Returns:
        Dictionary mapping GPU ID to metrics dict
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        gpus = {}
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 6:
                gpu_id, gpu_util, mem_util, mem_used, mem_total, temp = parts
                gpus[gpu_id] = {
                    "gpu_utilization": gpu_util,
                    "memory_utilization": mem_util,
                    "memory_used_mb": mem_used,
                    "memory_total_mb": mem_total,
                    "temperature_c": temp,
                }
        return gpus
    except Exception as e:
        print(f"Error parsing nvidia-smi: {e}", file=sys.stderr)
        return {}


def log_sample(output_file: Path, gpus: dict[str, dict[str, str]]) -> None:
    """Log GPU metrics sample to file.

    Args:
        output_file: Path to output CSV file
        gpus: Dictionary of GPU metrics from parse_nvidia_smi()
    """
    timestamp = datetime.now().isoformat()

    # Write header if file doesn't exist
    if not output_file.exists():
        with open(output_file, "w") as f:
            f.write("timestamp,gpu_id,gpu_util_%,mem_util_%,mem_used_mb,mem_total_mb,temp_c\n")

    # Append metrics
    with open(output_file, "a") as f:
        for gpu_id, metrics in sorted(gpus.items()):
            f.write(
                f"{timestamp},{gpu_id},"
                f"{metrics['gpu_utilization']},"
                f"{metrics['memory_utilization']},"
                f"{metrics['memory_used_mb']},"
                f"{metrics['memory_total_mb']},"
                f"{metrics['temperature_c']}\n"
            )


def display_sample(gpus: dict[str, dict[str, str]]) -> None:
    """Display GPU metrics to console.

    Args:
        gpus: Dictionary of GPU metrics from parse_nvidia_smi()
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] GPU Utilization:")
    for gpu_id, metrics in sorted(gpus.items()):
        print(
            f"  GPU {gpu_id}: {metrics['gpu_utilization']:>3}% util | "
            f"{metrics['memory_used_mb']:>5}MB / {metrics['memory_total_mb']:>5}MB | "
            f"{metrics['temperature_c']:>2}°C"
        )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor GPU utilization during training experiments"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Sampling interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gpu_metrics.csv",
        help="Output CSV file path (default: gpu_metrics.csv)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Total monitoring duration in seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (no console output, only log to file)",
    )

    args = parser.parse_args()
    output_file = Path(args.output)

    if not args.quiet:
        print("GPU Monitoring Started")
        print(f"  Output: {output_file}")
        print(f"  Interval: {args.interval}s")
        if args.duration:
            print(f"  Duration: {args.duration}s")
        else:
            print("  Duration: until Ctrl+C")
        print("\nPress Ctrl+C to stop\n")

    start_time = time.time()
    samples_collected = 0

    try:
        while True:
            # Parse GPU metrics
            gpus = parse_nvidia_smi()

            if gpus:
                # Log to file
                log_sample(output_file, gpus)
                samples_collected += 1

                # Display to console
                if not args.quiet:
                    display_sample(gpus)

            # Check duration limit
            if args.duration and (time.time() - start_time) >= args.duration:
                break

            # Sleep until next sample
            time.sleep(args.interval)

    except KeyboardInterrupt:
        if not args.quiet:
            print("\n\nMonitoring stopped by user")

    elapsed = time.time() - start_time
    if not args.quiet:
        print(f"\nCollected {samples_collected} samples over {elapsed:.1f}s")
        print(f"Results saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
