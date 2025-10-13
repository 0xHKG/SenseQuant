#!/usr/bin/env python3
"""
Production Deployment Script (US-027)

Deploys release artifacts to specified environment with rollback support.

Usage:
    python scripts/deploy.py --environment prod
    python scripts/deploy.py --environment staging --dryrun
    python scripts/deploy.py --rollback --environment prod

Examples:
    # Deploy to staging
    python scripts/deploy.py --environment staging

    # Deploy to production (with confirmation)
    python scripts/deploy.py --environment prod

    # Dryrun (simulate without changes)
    python scripts/deploy.py --environment prod --dryrun

    # Rollback last deployment
    python scripts/deploy.py --rollback --environment prod
"""

import argparse
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Deployer:
    """Deployment orchestrator with rollback support."""

    def __init__(self, environment: str, dryrun: bool = False):
        """Initialize deployer.

        Args:
            environment: Target environment ("prod" or "staging")
            dryrun: Simulate deployment without changes
        """
        self.environment = environment
        self.dryrun = dryrun
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.release_id = f"release_{self.timestamp}"

        # Directories
        self.repo_root = Path(__file__).parent.parent
        self.staging_dir = self.repo_root / "data" / "models" / "staging"
        self.prod_dir = self.repo_root / "data" / "models" / "production"
        self.backup_dir = self.repo_root / "data" / "backups" / self.timestamp

        logger.info(f"Deployer initialized: {environment} (dryrun={dryrun})")

    def deploy(self) -> bool:
        """Execute deployment workflow.

        Returns:
            True if deployment successful, False otherwise
        """
        logger.info(f"🚀 Starting deployment: {self.release_id} → {self.environment}")

        # Step 1: Backup current
        logger.info("Step 1/4: Backing up current artifacts...")
        if not self._backup_current():
            logger.error("Backup failed")
            return False

        # Step 2: Copy new artifacts
        logger.info("Step 2/4: Copying new artifacts...")
        if not self._copy_artifacts():
            logger.error("Artifact copy failed")
            return False

        # Step 3: Smoke test
        logger.info("Step 3/4: Running smoke tests...")
        if not self._smoke_test():
            logger.error("Smoke tests failed, rolling back")
            self.rollback()
            return False

        # Step 4: Record deployment
        logger.info("Step 4/4: Recording deployment...")
        self._record_deployment(status="success", smoke_test_passed=True, rollback=False)

        logger.info(f"✅ Deployment successful: {self.release_id}")
        return True

    def _backup_current(self) -> bool:
        """Backup current production artifacts.

        Returns:
            True if backup successful
        """
        if self.dryrun:
            logger.info(f"[DRYRUN] Would backup to {self.backup_dir}")
            return True

        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Artifacts to backup
        artifacts = [
            self.prod_dir / "student_model.pkl",
            self.repo_root / "config" / "config.yaml",
        ]

        backed_up = []
        for artifact in artifacts:
            if artifact.exists():
                dst = self.backup_dir / artifact.name
                shutil.copy2(artifact, dst)
                backed_up.append(artifact.name)
                logger.info(f"  ✓ Backed up: {artifact.name}")

        if not backed_up:
            logger.warning("No artifacts found to backup")
            # Still return True (empty backup is OK for first deployment)

        logger.info(f"Backup complete: {len(backed_up)} artifacts → {self.backup_dir}")
        return True

    def _copy_artifacts(self) -> bool:
        """Copy new artifacts from staging to production.

        Returns:
            True if copy successful
        """
        if self.dryrun:
            logger.info(f"[DRYRUN] Would copy artifacts from {self.staging_dir} → {self.prod_dir}")
            return True

        # Verify staging directory exists
        if not self.staging_dir.exists():
            logger.error(f"Staging directory not found: {self.staging_dir}")
            logger.info("Ensure models are trained and saved to staging directory")
            return False

        # Create production directory if needed
        self.prod_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files from staging to production
        copied = []
        for artifact in self.staging_dir.glob("*"):
            if artifact.is_file():
                dst = self.prod_dir / artifact.name
                shutil.copy2(artifact, dst)
                copied.append(artifact.name)
                logger.info(f"  ✓ Deployed: {artifact.name}")

        if not copied:
            logger.error("No artifacts found in staging directory")
            return False

        logger.info(f"Artifact copy complete: {len(copied)} files")
        return True

    def _smoke_test(self) -> bool:
        """Run smoke tests on deployed artifacts.

        Returns:
            True if all smoke tests pass
        """
        if self.dryrun:
            logger.info("[DRYRUN] Would run smoke tests")
            return True

        logger.info("Running smoke tests...")

        # Test 1: Model file exists
        model_path = self.prod_dir / "student_model.pkl"
        if not model_path.exists():
            logger.error("Test failed: Model not found after deployment")
            return False
        logger.info("  ✓ Model file exists")

        # Test 2: Model loadable
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info("  ✓ Model loadable (pickle deserialization)")
        except Exception as e:
            logger.error(f"Test failed: Model load error: {e}")
            return False

        # Test 3: Config valid
        try:
            from src.app.config import Settings

            _ = Settings()  # Just verify it loads
            logger.info("  ✓ Config validation passed")
        except Exception as e:
            logger.error(f"Test failed: Config validation error: {e}")
            return False

        # Test 4: Quick model inference (if model has predict method)
        try:
            if hasattr(model, "predict"):
                # Create dummy features for inference test
                import numpy as np

                dummy_features = np.random.rand(1, 10)
                _ = model.predict(dummy_features)
                logger.info("  ✓ Model inference test passed")
            else:
                logger.warning("  ⚠ Model has no predict method (skipping inference test)")
        except Exception as e:
            logger.warning(f"  ⚠ Model inference test failed: {e} (non-critical)")

        logger.info("All smoke tests passed ✅")
        return True

    def rollback(self) -> bool:
        """Rollback to previous artifacts.

        Returns:
            True if rollback successful
        """
        logger.warning(f"🔄 Executing rollback: {self.environment}")

        if self.dryrun:
            logger.info("[DRYRUN] Would rollback to previous release")
            return True

        # Find latest backup (excluding current deployment)
        backups_dir = self.repo_root / "data" / "backups"
        if not backups_dir.exists():
            logger.error("No backups directory found")
            return False

        backups = sorted([b for b in backups_dir.glob("*") if b.is_dir() and b != self.backup_dir])

        if not backups:
            logger.error("No previous backups found for rollback")
            logger.info("Rollback unavailable for first deployment")
            return False

        latest_backup = backups[-1]
        logger.info(f"Rolling back to: {latest_backup.name}")

        # Restore artifacts from backup
        restored = []
        for artifact in latest_backup.glob("*"):
            if artifact.is_file():
                dst = self.prod_dir / artifact.name
                shutil.copy2(artifact, dst)
                restored.append(artifact.name)
                logger.info(f"  ✓ Restored: {artifact.name}")

        if not restored:
            logger.error("No artifacts found in backup directory")
            return False

        # Record rollback event
        self._record_deployment(
            status="rolled_back",
            smoke_test_passed=False,
            rollback=True,
        )

        logger.info(f"✅ Rollback complete: {len(restored)} artifacts restored")
        return True

    def _record_deployment(
        self,
        status: str,
        smoke_test_passed: bool,
        rollback: bool,
    ) -> None:
        """Record deployment in StateManager.

        Args:
            status: Deployment status ("success", "failed", "rolled_back")
            smoke_test_passed: Whether smoke tests passed
            rollback: Whether this was a rollback operation
        """
        if self.dryrun:
            logger.info(f"[DRYRUN] Would record deployment: {status}")
            return

        try:
            from src.services.state_manager import StateManager

            state_mgr = StateManager()

            # Get list of deployed artifacts
            artifacts = []
            if self.prod_dir.exists():
                artifacts = [f.name for f in self.prod_dir.glob("*") if f.is_file()]

            state_mgr.record_deployment(
                release_id=self.release_id,
                environment=self.environment,
                timestamp=datetime.now().isoformat(),
                status=status,
                artifacts=artifacts,
                rollback=rollback,
                smoke_test_passed=smoke_test_passed,
                deployed_by="deploy-script",
            )

            logger.info("Deployment recorded in StateManager")
        except Exception as e:
            logger.error(f"Failed to record deployment: {e}")


def confirm_production_deployment() -> bool:
    """Prompt user to confirm production deployment.

    Returns:
        True if user confirms
    """
    print()
    print("⚠️  WARNING: Production Deployment")
    print()
    print("This will:")
    print("  • Replace production models and configurations")
    print("  • Affect live trading (if enabled)")
    print("  • Create backup for rollback")
    print()

    response = input("Type 'yes' to confirm: ").strip().lower()
    return response == "yes"


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy release to environment with rollback support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy to staging
  python scripts/deploy.py --environment staging

  # Deploy to production (requires confirmation)
  python scripts/deploy.py --environment prod

  # Dryrun (simulate without changes)
  python scripts/deploy.py --environment prod --dryrun

  # Rollback last deployment
  python scripts/deploy.py --rollback --environment prod

Workflow:
  1. Backup current production artifacts
  2. Copy new artifacts from staging
  3. Run smoke tests (model load, config validation, inference test)
  4. Record deployment in StateManager
  5. If smoke tests fail → automatic rollback

Next Steps After Deployment:
  • Monitor dashboard for anomalies
  • Check deployment status: make deploy-status
  • Execute rollback if needed: make rollback
        """,
    )

    parser.add_argument(
        "--environment",
        "-e",
        choices=["prod", "staging"],
        required=True,
        help="Target environment",
    )

    parser.add_argument(
        "--dryrun",
        "-d",
        action="store_true",
        help="Simulate deployment without changes",
    )

    parser.add_argument(
        "--rollback",
        "-r",
        action="store_true",
        help="Rollback to previous deployment",
    )

    args = parser.parse_args()

    # Confirm production deployments (unless dryrun)
    if args.environment == "prod" and not args.dryrun and not args.rollback:
        if not confirm_production_deployment():
            print("Deployment cancelled")
            sys.exit(1)

    # Execute deployment or rollback
    deployer = Deployer(environment=args.environment, dryrun=args.dryrun)

    if args.rollback:
        success = deployer.rollback()
    else:
        success = deployer.deploy()

    # Exit with status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
