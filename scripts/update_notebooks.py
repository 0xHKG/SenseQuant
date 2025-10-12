#!/usr/bin/env python3
"""Update notebooks with Release Audit sections (US-022)."""

import json
from pathlib import Path


def add_release_audit_to_accuracy_report() -> None:
    """Add Release Audit section to accuracy_report.ipynb."""
    notebook_path = Path("notebooks/accuracy_report.ipynb")

    with open(notebook_path) as f:
        nb = json.load(f)

    # Create new Release Audit section cells
    release_audit_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Release Audit Summary (US-022)\n\n",
                "This section consolidates telemetry data, optimization results, and student model metrics\n",
                "for release readiness assessment. It provides a unified view of baseline vs optimized performance,\n",
                "student model validation status, and monitoring KPIs.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load latest release audit bundle (if available)\n",
                "import glob\n",
                "from datetime import datetime\n\n",
                "# Find latest audit bundle\n",
                'audit_dirs = sorted(glob.glob("../release/audit_*"), reverse=True)\n\n',
                "if audit_dirs:\n",
                "    latest_audit_dir = Path(audit_dirs[0])\n",
                '    metrics_path = latest_audit_dir / "metrics.json"\n',
                "    \n",
                "    if metrics_path.exists():\n",
                "        with open(metrics_path) as f:\n",
                "            audit_metrics = json.load(f)\n",
                "        \n",
                "        print(f\"\\n{'='*70}\")\n",
                '        print("RELEASE AUDIT SUMMARY")\n',
                "        print(f\"{'='*70}\")\n",
                "        print(f\"\\nAudit ID: {audit_metrics['audit_id']}\")\n",
                "        print(f\"Audit Date: {audit_metrics['audit_timestamp']}\")\n",
                "        print(f\"Deployment Ready: {'YES' if audit_metrics['deployment_ready'] else 'NO'}\")\n",
                "        \n",
                "        if audit_metrics.get('risk_flags'):\n",
                "            print(f\"\\nRisk Flags ({len(audit_metrics['risk_flags'])}):\")\n",
                "            for flag in audit_metrics['risk_flags']:\n",
                '                print(f"   - {flag}")\n',
                "    else:\n",
                '        print("Latest audit bundle found but metrics.json is missing")\n',
                "        audit_metrics = None\n",
                "else:\n",
                '    print("No release audit bundles found. Run: python scripts/release_audit.py")\n',
                "    audit_metrics = None",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Baseline vs Optimized Comparison\n",
                "if audit_metrics and 'baseline' in audit_metrics and 'optimized' in audit_metrics:\n",
                "    baseline = audit_metrics['baseline']\n",
                "    optimized = audit_metrics['optimized']\n",
                "    deltas = audit_metrics.get('deltas', {})\n",
                "    \n",
                '    print("\\n" + "="*70)\n',
                '    print("BASELINE vs OPTIMIZED CONFIGURATION")\n',
                '    print("="*70)\n',
                "    \n",
                "    comparison_data = [\n",
                "        ['Sharpe Ratio', baseline['sharpe_ratio'], optimized['sharpe_ratio'], \n",
                "         deltas.get('sharpe_ratio_delta', 0.0)],\n",
                "        ['Total Return (%)', baseline['total_return_pct'], optimized['total_return_pct'], \n",
                "         deltas.get('total_return_delta_pct', 0.0)],\n",
                "        ['Win Rate (%)', baseline['win_rate_pct'], optimized['win_rate_pct'], \n",
                "         deltas.get('win_rate_delta_pct', 0.0)],\n",
                "        ['Hit Ratio (%)', baseline['hit_ratio_pct'], optimized['hit_ratio_pct'], \n",
                "         deltas.get('hit_ratio_delta_pct', 0.0)],\n",
                "    ]\n",
                "    \n",
                "    comp_df = pd.DataFrame(comparison_data, \n",
                "                          columns=['Metric', 'Baseline', 'Optimized', 'Delta'])\n",
                "    \n",
                '    print("\\n" + comp_df.to_string(index=False))\n',
                "    \n",
                "    # Visualization\n",
                "    fig, ax = plt.subplots(figsize=(14, 6))\n",
                "    \n",
                "    x = np.arange(len(comparison_data))\n",
                "    width = 0.35\n",
                "    \n",
                "    baseline_vals = [row[1] for row in comparison_data]\n",
                "    optimized_vals = [row[2] for row in comparison_data]\n",
                "    \n",
                "    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',\n",
                "                   color='#E74C3C', alpha=0.8, edgecolor='black')\n",
                "    bars2 = ax.bar(x + width/2, optimized_vals, width, label='Optimized',\n",
                "                   color='#27AE60', alpha=0.8, edgecolor='black')\n",
                "    \n",
                "    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')\n",
                "    ax.set_ylabel('Value', fontsize=12, fontweight='bold')\n",
                "    ax.set_title('Release Audit: Baseline vs Optimized Performance',\n",
                "                fontsize=14, fontweight='bold')\n",
                "    ax.set_xticks(x)\n",
                "    ax.set_xticklabels([row[0] for row in comparison_data], rotation=15, ha='right')\n",
                "    ax.legend()\n",
                "    ax.grid(True, alpha=0.3, axis='y')\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    \n",
                '    output_path = Path(output_dir) / "release_audit_comparison.png"\n',
                "    plt.savefig(output_path, dpi=300, bbox_inches='tight')\n",
                '    print(f"\\nSaved comparison chart to: {output_path}")\n',
                "    \n",
                "    plt.show()\n",
                "else:\n",
                '    print("Baseline/optimized metrics not available in audit bundle")',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Student Model Status\n",
                "if audit_metrics and 'student_model' in audit_metrics:\n",
                "    student = audit_metrics['student_model']\n",
                "    \n",
                '    print("\\n" + "="*70)\n',
                '    print("STUDENT MODEL VALIDATION STATUS")\n',
                '    print("="*70)\n',
                "    \n",
                "    print(f\"\\nDeployed: {'YES' if student.get('deployed') else 'NO'}\")\n",
                "    if student.get('deployed'):\n",
                "        print(f\"Version: {student.get('version', 'unknown')}\")\n",
                "        print(f\"Validation Precision: {student.get('validation_precision', 0.0):.2%}\")\n",
                "        print(f\"Validation Recall: {student.get('validation_recall', 0.0):.2%}\")\n",
                "        print(f\"Test Accuracy: {student.get('test_accuracy', 0.0):.2%}\")\n",
                "        print(f\"Feature Count: {student.get('feature_count', 0)}\")\n",
                "        print(f\"Training Samples: {student.get('training_samples', 0):,}\")\n",
                "else:\n",
                '    print("\\nNo student model metrics in audit bundle")',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Monitoring KPIs\n",
                "if audit_metrics and 'monitoring' in audit_metrics:\n",
                "    monitoring = audit_metrics['monitoring']\n",
                "    \n",
                '    print("\\n" + "="*70)\n',
                '    print("MONITORING KPIs (Rolling Windows)")\n',
                '    print("="*70)\n',
                "    \n",
                "    if 'intraday_30day' in monitoring:\n",
                "        intra = monitoring['intraday_30day']\n",
                '        print("\\nIntraday Strategy (30-day window):")\n',
                "        print(f\"   Hit Ratio: {intra.get('hit_ratio', 0.0):.2%}\")\n",
                "        print(f\"   Sharpe Ratio: {intra.get('sharpe_ratio', 0.0):.2f}\")\n",
                "        print(f\"   Alert Count: {intra.get('alert_count', 0)}\")\n",
                "        print(f\"   Degradation: {'YES' if intra.get('degradation_detected') else 'NO'}\")\n",
                "    \n",
                "    if 'swing_90day' in monitoring:\n",
                "        swing = monitoring['swing_90day']\n",
                '        print("\\nSwing Strategy (90-day window):")\n',
                "        print(f\"   Precision (LONG): {swing.get('precision_long', 0.0):.2%}\")\n",
                "        print(f\"   Recall (LONG): {swing.get('recall_long', 0.0):.2%}\")\n",
                "        print(f\"   Max Drawdown: {swing.get('max_drawdown_pct', 0.0):.1f}%\")\n",
                "        print(f\"   Alert Count: {swing.get('alert_count', 0)}\")\n",
                "        print(f\"   Degradation: {'YES' if swing.get('degradation_detected') else 'NO'}\")\n",
                "else:\n",
                '    print("\\nNo monitoring metrics in audit bundle")',
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Release Audit Recommendations\n\n",
                "Based on the consolidated audit metrics:\n\n",
                "1. **If Deployment Ready**: Proceed with gradual rollout as per deployment plan\n",
                "2. **If Risk Flags Present**: Address each flag before production deployment\n",
                "3. **Monitor KPIs**: Continue tracking rolling window metrics for early degradation detection\n",
                "4. **Schedule Next Audit**: Plan monthly audits to maintain release readiness\n\n",
                "For full audit details, review:\n",
                "- `release/audit_<timestamp>/summary.md` - Executive summary\n",
                "- `release/audit_<timestamp>/metrics.json` - Complete metrics\n",
                "- `release/audit_<timestamp>/plots/` - All visualizations",
            ],
        },
    ]

    # Find conclusion cell (last cell)
    conclusion_idx = len(nb["cells"]) - 1

    # Insert new cells before conclusion
    for i, cell in enumerate(release_audit_cells):
        nb["cells"].insert(conclusion_idx + i, cell)

    # Write back
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"✅ Updated {notebook_path} with {len(release_audit_cells)} new cells")
    print(f"   Total cells: {len(nb['cells'])}")


def main() -> None:
    """Main entry point."""
    add_release_audit_to_accuracy_report()
    print("\n✅ All notebooks updated!")


if __name__ == "__main__":
    main()
