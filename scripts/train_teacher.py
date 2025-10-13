#!/usr/bin/env python
"""CLI entry point for Teacher model training.

Usage:
    python scripts/train_teacher.py --symbol RELIANCE --start 2023-01-01 --end 2024-10-01
    python scripts/train_teacher.py --symbol TCS --window 10 --threshold 0.03 --seed 123
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Add project root to sys.path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.breeze_client import BreezeClient
from src.app.config import settings
from src.domain.types import TrainingConfig
from src.services.teacher_student import TeacherLabeler


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Teacher model on historical data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Stock symbol to train on (e.g., RELIANCE, TCS)",
    )

    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date for training data (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date for training data (YYYY-MM-DD)",
    )

    # Optional arguments
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Forward-looking window in days for label generation",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Return threshold (as decimal) for positive label (e.g., 0.02 = 2%%)",
    )

    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train/validation split ratio (e.g., 0.8 = 80%% train, 20%% val)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--estimators",
        type=int,
        default=100,
        help="Number of boosting iterations for LightGBM",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum tree depth for LightGBM",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for LightGBM",
    )

    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Force dry-run mode (mock data) regardless of MODE setting",
    )

    return parser.parse_args()


def validate_dates(start_date: str, end_date: str) -> None:
    """Validate date format and range."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    if start >= end:
        logger.error(f"Start date ({start_date}) must be before end date ({end_date})")
        sys.exit(1)

    # Check if date range is reasonable (at least 6 months)
    if (end - start).days < 180:
        logger.warning(
            f"Date range is less than 6 months ({(end - start).days} days). "
            "Consider using more data for better model training."
        )


def main() -> None:
    """Main entry point for Teacher training."""
    args = parse_args()

    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[component]}</cyan> | {message}",
        level="INFO",
    )
    logger.configure(extra={"component": "teacher"})

    logger.info("=" * 80)
    logger.info("Teacher Model Training")
    logger.info("=" * 80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Date Range: {args.start} to {args.end}")
    logger.info(f"Label Window: {args.window} days")
    logger.info(f"Return Threshold: {args.threshold * 100:.1f}%")
    logger.info(f"Train/Val Split: {args.split * 100:.0f}/{(1 - args.split) * 100:.0f}")
    logger.info(f"Random Seed: {args.seed}")
    logger.info("=" * 80)

    # Validate inputs
    validate_dates(args.start, args.end)

    if not (0.5 <= args.split <= 0.95):
        logger.error(f"Invalid train split: {args.split}. Must be between 0.5 and 0.95")
        sys.exit(1)

    # Create training configuration
    model_params = {
        "n_estimators": args.estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
    }

    config = TrainingConfig(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        label_window_days=args.window,
        label_threshold_pct=args.threshold,
        train_split=args.split,
        random_seed=args.seed,
        model_params=model_params,
    )

    # Initialize BreezeClient
    try:
        # Determine dry_run mode: CLI flag overrides, else respect settings.mode
        use_dry_run = args.dryrun if args.dryrun else (settings.mode != "live")

        logger.info("Initializing BreezeClient...")
        logger.info(f"MODE: {'DRYRUN (mock data)' if use_dry_run else 'LIVE (real data)'}")
        if args.dryrun:
            logger.info("  → Forced by --dryrun flag")
        else:
            logger.info(f"  → Determined by MODE={settings.mode} in .env")

        client = BreezeClient(
            api_key=settings.breeze_api_key,
            api_secret=settings.breeze_api_secret,
            session_token=settings.breeze_session_token,
            dry_run=use_dry_run,
        )
        client.authenticate()
    except Exception as e:
        logger.error(f"Failed to initialize BreezeClient: {e}")
        sys.exit(1)

    # Create TeacherLabeler
    teacher = TeacherLabeler(config, client=client)

    # Run training pipeline
    try:
        logger.info("Starting Teacher training pipeline...")
        result = teacher.run_full_pipeline()

        # Print results
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Model Path: {result.model_path}")
        logger.info(f"Labels Path: {result.labels_path}")
        logger.info(f"Importance Path: {result.importance_path}")
        logger.info(f"Metadata Path: {result.metadata_path}")
        logger.info("-" * 80)
        logger.info(f"Features: {result.feature_count}")
        logger.info(f"Training Samples: {result.train_samples}")
        logger.info(f"Validation Samples: {result.val_samples}")
        logger.info("-" * 80)
        logger.info("Validation Metrics:")
        logger.info(f"  Accuracy:  {result.metrics['val_accuracy']:.4f}")
        logger.info(f"  Precision: {result.metrics['val_precision']:.4f}")
        logger.info(f"  Recall:    {result.metrics['val_recall']:.4f}")
        logger.info(f"  F1 Score:  {result.metrics['val_f1']:.4f}")
        logger.info(f"  AUC-ROC:   {result.metrics['val_auc']:.4f}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
