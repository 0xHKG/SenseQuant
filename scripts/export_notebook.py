#!/usr/bin/env python3
"""
Notebook Export Helper Script (US-019 Phase 4)

Converts Jupyter notebooks to various formats using nbconvert.

Usage:
    python scripts/export_notebook.py <notebook_name> [--format html|markdown|pdf]

Examples:
    python scripts/export_notebook.py optimization_report
    python scripts/export_notebook.py optimization_report --format html
    python scripts/export_notebook.py accuracy_report --format markdown
"""

import argparse
import subprocess
import sys
from pathlib import Path


def export_notebook(notebook_name: str, output_format: str = "html") -> None:
    """Export a Jupyter notebook to specified format using nbconvert.

    Args:
        notebook_name: Name of notebook (without .ipynb extension)
        output_format: Output format (html, markdown, or pdf)
    """
    # Resolve paths
    repo_root = Path(__file__).parent.parent
    notebooks_dir = repo_root / "notebooks"
    output_dir = repo_root / "data" / "reports"

    notebook_path = notebooks_dir / f"{notebook_name}.ipynb"

    # Validate notebook exists
    if not notebook_path.exists():
        print(f"❌ Error: Notebook not found: {notebook_path}")
        print("\nAvailable notebooks:")
        for nb in notebooks_dir.glob("*.ipynb"):
            print(f"  - {nb.stem}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build nbconvert command
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        output_format,
        str(notebook_path),
        "--output-dir",
        str(output_dir),
    ]

    # Add format-specific options
    if output_format == "html":
        cmd.extend(["--template", "classic"])
    elif output_format == "pdf":
        print("⚠️  PDF export requires pandoc and texlive-xetex")
        print("   Install: sudo apt-get install pandoc texlive-xetex")

    print(f"📓 Exporting {notebook_name}.ipynb to {output_format.upper()}...")
    print(f"   Input:  {notebook_path}")
    print(f"   Output: {output_dir}")

    try:
        # Run nbconvert
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Determine output filename
        output_file = output_dir / f"{notebook_name}.{output_format}"

        print("✅ Export successful!")
        print(f"   Generated: {output_file}")

        if output_format == "html":
            print(f"\n🌐 Open in browser: file://{output_file.absolute()}")

    except subprocess.CalledProcessError as e:
        print("❌ Export failed!")
        print("\nError output:")
        print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: jupyter nbconvert not found")
        print("\nInstall with: pip install nbconvert")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export Jupyter notebooks to HTML/Markdown/PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export optimization report to HTML (default)
  python scripts/export_notebook.py optimization_report

  # Export accuracy report to Markdown
  python scripts/export_notebook.py accuracy_report --format markdown

  # Export to PDF (requires pandoc and texlive)
  python scripts/export_notebook.py optimization_report --format pdf

Available notebooks:
  - optimization_report (US-019 Phase 4)
  - accuracy_report (US-016)
        """,
    )

    parser.add_argument("notebook", help="Notebook name (without .ipynb extension)")

    parser.add_argument(
        "--format",
        "-f",
        choices=["html", "markdown", "pdf"],
        default="html",
        help="Output format (default: html)",
    )

    args = parser.parse_args()

    export_notebook(args.notebook, args.format)


if __name__ == "__main__":
    main()
