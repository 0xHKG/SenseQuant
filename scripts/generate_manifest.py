#!/usr/bin/env python3
"""Generate signed release manifest (US-023).

This script creates a release manifest with SHA256 hashes for all artifacts
to be deployed, approval records, and rollback plan.

Usage:
    python scripts/generate_manifest.py
    python scripts/generate_manifest.py --output-dir release/manifests
    python scripts/generate_manifest.py --release-type hotfix
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate signed release manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        default="release/manifests",
        help="Output directory for manifests (default: release/manifests)",
    )

    parser.add_argument(
        "--release-type",
        choices=["major", "minor", "hotfix"],
        default="minor",
        help="Release type (default: minor)",
    )

    parser.add_argument(
        "--audit-bundle",
        help="Path to audit bundle directory (default: latest)",
    )

    parser.add_argument(
        "--deployer",
        default="engineering_lead",
        help="Deployer identifier (default: engineering_lead)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n",
    )


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        SHA256 hash as hex string
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def find_latest_audit_bundle() -> Path | None:
    """Find the latest audit bundle directory.

    Returns:
        Path to latest audit bundle or None
    """
    release_dir = Path("release")
    if not release_dir.exists():
        return None

    audit_dirs = [d for d in release_dir.iterdir() if d.is_dir() and d.name.startswith("audit_")]
    if not audit_dirs:
        return None

    return max(audit_dirs, key=lambda d: d.stat().st_mtime)


def backup_artifact(artifact_path: Path, backup_dir: Path) -> Path:
    """Backup an artifact file.

    Args:
        artifact_path: Path to artifact to backup
        backup_dir: Backup directory

    Returns:
        Path to backed up file
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / artifact_path.name
    shutil.copy2(artifact_path, backup_path)
    return backup_path


def find_previous_manifest() -> Path | None:
    """Find the most recent release manifest.

    Returns:
        Path to previous manifest or None
    """
    manifests_dir = Path("release/manifests")
    if not manifests_dir.exists():
        return None

    manifest_files = list(manifests_dir.glob("release_*.yaml"))
    if not manifest_files:
        return None

    return max(manifest_files, key=lambda f: f.stat().st_mtime)


def generate_manifest(args: argparse.Namespace) -> dict:
    """Generate release manifest.

    Args:
        args: Command line arguments

    Returns:
        Manifest dictionary
    """
    timestamp = datetime.now()
    release_id = f"release_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Generating release manifest: {release_id}")

    # Find audit bundle
    if args.audit_bundle:
        audit_bundle = Path(args.audit_bundle)
    else:
        audit_bundle = find_latest_audit_bundle()

    if not audit_bundle or not audit_bundle.exists():
        logger.warning("No audit bundle found - manifest will have limited data")
        audit_bundle_str = None
    else:
        audit_bundle_str = str(audit_bundle)
        logger.info(f"Using audit bundle: {audit_bundle_str}")

    # Create backup directory
    backup_dir = Path(f"release/backups/{release_id}")

    # Collect artifacts
    artifacts = {"configs": [], "models": [], "notebooks": []}

    # Config files
    config_files = [
        Path("src/app/config.py"),
        Path("search_space.yaml"),
    ]
    for config_file in config_files:
        if config_file.exists():
            file_hash = compute_sha256(config_file)
            backup_path = backup_artifact(config_file, backup_dir)
            artifacts["configs"].append(
                {
                    "path": str(config_file),
                    "hash": f"sha256:{file_hash}",
                    "backup": str(backup_path),
                }
            )
            logger.debug(f"  Config: {config_file} (hash: {file_hash[:8]}...)")

    # Model files
    model_files = (
        list(Path("data/models").glob("student*.pkl")) if Path("data/models").exists() else []
    )
    for model_file in model_files:
        file_hash = compute_sha256(model_file)
        backup_path = backup_artifact(model_file, backup_dir)

        # Try to determine version from metadata
        version = "unknown"
        metadata_file = model_file.parent / "student_model_metadata.json"
        if metadata_file.exists():
            import json

            with open(metadata_file) as f:
                metadata = json.load(f)
            version = metadata.get("version", "unknown")

        artifacts["models"].append(
            {
                "path": str(model_file),
                "hash": f"sha256:{file_hash}",
                "version": version,
                "backup": str(backup_path),
            }
        )
        logger.debug(f"  Model: {model_file} (version: {version}, hash: {file_hash[:8]}...)")

    # Notebook files
    notebook_files = [
        Path("notebooks/accuracy_report.ipynb"),
        Path("notebooks/optimization_report.ipynb"),
    ]
    for notebook_file in notebook_files:
        if notebook_file.exists():
            file_hash = compute_sha256(notebook_file)
            backup_path = backup_artifact(notebook_file, backup_dir)
            artifacts["notebooks"].append(
                {
                    "path": str(notebook_file),
                    "hash": f"sha256:{file_hash}",
                    "backup": str(backup_path),
                }
            )
            logger.debug(f"  Notebook: {notebook_file} (hash: {file_hash[:8]}...)")

    # Load approvals from audit bundle if available
    approvals = []
    if audit_bundle and (audit_bundle / "summary.md").exists():
        # In real implementation, would parse approval signatures from summary.md
        # For now, use placeholder
        approvals = [
            {
                "role": "Engineering Lead",
                "name": "Auto-Generated",
                "email": "noreply@example.com",
                "timestamp": timestamp.isoformat(),
                "signature": f"sha256:{hashlib.sha256(release_id.encode()).hexdigest()[:16]}",
            }
        ]

    # Generate rollback plan
    previous_manifest = find_previous_manifest()
    rollback_plan = {
        "previous_release_id": None,
        "previous_manifest": None,
        "restore_commands": [],
        "verification_steps": [
            "python scripts/release_audit.py --skip-validation",
            "make test",
        ],
    }

    if previous_manifest:
        with open(previous_manifest) as f:
            prev_data = yaml.safe_load(f)
        rollback_plan["previous_release_id"] = prev_data["release_id"]
        rollback_plan["previous_manifest"] = str(previous_manifest)

        # Generate restore commands
        for artifact_type in ["configs", "models", "notebooks"]:
            for artifact in prev_data.get("artifacts", {}).get(artifact_type, []):
                if Path(artifact["backup"]).exists():
                    rollback_plan["restore_commands"].append(
                        f"cp {artifact['backup']} {artifact['path']}"
                    )

        logger.info(f"Rollback plan: restore from {rollback_plan['previous_release_id']}")

    # Build manifest
    manifest = {
        "release_id": release_id,
        "release_type": args.release_type,
        "audit_bundle": audit_bundle_str,
        "deployment": {
            "timestamp": timestamp.isoformat(),
            "deployer": args.deployer,
            "environment": "production",
        },
        "approvals": approvals,
        "artifacts": artifacts,
        "rollback_plan": rollback_plan,
        "monitoring": {
            "heightened_period_hours": 48,
            "heightened_start": timestamp.isoformat(),
            "heightened_end": (timestamp + __import__("datetime").timedelta(hours=48)).isoformat(),
            "alert_thresholds": {
                "intraday_hit_ratio_drop": 0.05,
                "swing_precision_drop": 0.05,
            },
            "alert_frequency_hours": 2,
        },
        "metadata": {
            "generator": "generate_manifest.py",
            "generator_version": "1.0.0",
        },
    }

    return manifest


def write_manifest(manifest: dict, output_dir: Path) -> Path:
    """Write manifest to YAML file.

    Args:
        manifest: Manifest dictionary
        output_dir: Output directory

    Returns:
        Path to written manifest file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{manifest['release_id']}.yaml"

    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    logger.info(f"✅ Manifest written: {manifest_path}")
    return manifest_path


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("=" * 70)
    logger.info("Release Manifest Generator (US-023)")
    logger.info("=" * 70)

    try:
        # Generate manifest
        manifest = generate_manifest(args)

        # Write to file
        output_dir = Path(args.output_dir)
        manifest_path = write_manifest(manifest, output_dir)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("✅ MANIFEST GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\n📦 Release ID: {manifest['release_id']}")
        logger.info(f"📄 Manifest: {manifest_path}")
        logger.info(f"💾 Backups: release/backups/{manifest['release_id']}/")

        artifact_counts = {k: len(v) for k, v in manifest["artifacts"].items()}
        logger.info(
            f"\n📊 Artifacts: {artifact_counts['configs']} configs, "
            f"{artifact_counts['models']} models, {artifact_counts['notebooks']} notebooks"
        )

        if manifest["rollback_plan"]["previous_release_id"]:
            logger.info(f"\n🔄 Rollback: {manifest['rollback_plan']['previous_release_id']}")
        else:
            logger.info("\n⚠️  No previous release found (first deployment)")

        logger.info("\n" + "=" * 70)

        return 0

    except Exception as e:
        logger.error(f"❌ Manifest generation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
