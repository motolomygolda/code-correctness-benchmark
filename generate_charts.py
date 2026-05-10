#!/usr/bin/env python3
"""
generate_charts.py — Produces all publication-quality charts from results.

Charts produced:
  1. pass@k grouped bar chart by tier and model (with bootstrap CI error bars)
  2. Stacked bar chart of failure-type proportions by (model, tier)
  3. Heatmap of cross-model failure overlap
  4. pass@3 recovery rates by failure type
  5. Overall pass-rate comparison (simple bar)

Usage:
    python generate_charts.py                 # reads results/analysis.json
    python generate_charts.py --csv results/results_raw.csv   # recompute from CSV
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "gpt35":      "#4285F4",
    "codellama":  "#EA4335",
    "Syntactic":  "#F4B400",
    "Runtime":    "#DB4437",
    "Algorithmic":"#4285F4",
    "Edge-case":  "#0F9D58",
    "Pass":       "#CCCCCC",
}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f9f9f9",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.15,
})

TIERS = ["Easy", "Medium", "Hard"]
FAIL_CATS = ["Syntactic", "Runtime", "Algorithmic", "Edge-case"]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_analysis(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["passed"] = row["passed"] == "True"
            row["sample_idx"] = int(row["sample_idx"])
            for k in ("standard_passed","standard_total","boundary_passed",
                       "boundary_total","adversarial_passed","adversarial_total"):
                row[k] = int(row[k])
            rows.append(row)
    return rows


def _model_label(key: str) -> str:
    return config.MODELS.get(key, {}).get("name", key)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — pass@k grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def chart_pass_at_k(analysis: dict, out_dir: str):
    """Grouped bar chart: pass@1 and pass@3 per (model, tier) with CI error bars."""
    pak = analysis["pass_at_k"]
    models = list(pak.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax_idx, k in enumerate([1, 3]):
        ax = axes[ax_idx]
        x = np.arange(len(TIERS))
        width = 0.35

        for i, model in enumerate(models):
            means  = [pak[model][t][f"pass@{k}"]["mean"] for t in TIERS]
            ci_lo  = [pak[model][t][f"pass@{k}"]["ci_lower"] for t in TIERS]
            ci_hi  = [pak[model][t][f"pass@{k}"]["ci_upper"] for t in TIERS]
            errs   = [[m - lo for m, lo in zip(means, ci_lo)],
                      [hi - m for m, hi in zip(means, ci_hi)]]

            bars = ax.bar(x + (i - 0.5) * width, means, width,
                          yerr=errs, capsize=4,
                          label=_model_label(model),
                          color=COLORS.get(model, f"C{i}"),
                          alpha=0.85, edgecolor="white", linewidth=0.5)
            # value labels
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(TIERS)
        ax.set_ylabel("Pass Rate" if ax_idx == 0 else "")
        ax.set_title(f"pass@{k}")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Pass@k by Difficulty Tier and Model", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "pass_at_k.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — failure taxonomy stacked bars
# ═══════════════════════════════════════════════════════════════════════════════

def chart_failure_taxonomy(analysis: dict, out_dir: str):
    """Stacked bar chart of failure proportions by (model, tier)."""
    tbt = analysis["taxonomy_by_tier"]
    models = list(tbt.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        bottoms = np.zeros(len(TIERS))
        for cat in FAIL_CATS:
            vals = []
            for tier in TIERS:
                total_fail = tbt[model][tier]["total_failures"]
                count = tbt[model][tier]["distribution"].get(cat, 0)
                vals.append(count / total_fail if total_fail else 0)
            ax.bar(TIERS, vals, bottom=bottoms, label=cat,
                   color=COLORS[cat], edgecolor="white", linewidth=0.5)
            bottoms += np.array(vals)

        ax.set_title(_model_label(model))
        ax.set_ylabel("Proportion of Failures" if ax == axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Failure Type Distribution by Tier", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "failure_taxonomy.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — cross-model overlap heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def chart_overlap_heatmap(analysis: dict, out_dir: str):
    """Heatmap of (model1-failure-type × model2-failure-type) co-occurrence."""
    if "overlap" not in analysis:
        print("  Skipping overlap heatmap (need exactly 2 models)")
        return

    overlap = analysis["overlap"]
    matrix_raw = overlap["overlap_matrix"]
    labels = ["Pass"] + FAIL_CATS

    matrix = np.zeros((len(labels), len(labels)))
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            matrix[i][j] = matrix_raw.get(l1, {}).get(l2, 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    models = list(analysis["pass_at_k"].keys())
    ax.set_xlabel(_model_label(models[1]) if len(models) > 1 else "Model 2")
    ax.set_ylabel(_model_label(models[0]))

    # annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = int(matrix[i][j])
            if val > 0:
                color = "white" if matrix[i][j] > matrix.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Problem Count")
    ax.set_title(f"Cross-Model Failure Overlap (Jaccard={overlap['jaccard_similarity']:.3f})",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "overlap_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — pass@3 recovery
# ═══════════════════════════════════════════════════════════════════════════════

def chart_recovery(analysis: dict, out_dir: str):
    """Grouped bar chart of pass@3 recovery rates by failure type."""
    recovery = analysis.get("pass3_recovery", {})
    if not recovery:
        print("  Skipping recovery chart (no data)")
        return

    models = list(recovery.keys())
    cats = set()
    for m in models:
        cats.update(recovery[m].keys())
    cats = sorted(cats)

    if not cats:
        print("  Skipping recovery chart (no failure categories)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cats))
    width = 0.35

    for i, model in enumerate(models):
        rates = [recovery[model].get(c, {}).get("rate", 0) for c in cats]
        totals = [recovery[model].get(c, {}).get("total", 0) for c in cats]
        bars = ax.bar(x + (i - 0.5) * width, rates, width,
                      label=_model_label(model),
                      color=COLORS.get(model, f"C{i}"),
                      alpha=0.85, edgecolor="white")
        for bar, rate, total in zip(bars, rates, totals):
            if total > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{rate:.0%}\n(n={total})", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Recovery Rate")
    ax.set_title("pass@3 Recovery Rates by Failure Type")
    ax.set_ylim(0, 1.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "recovery_rates.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — overall pass-rate comparison
# ═══════════════════════════════════════════════════════════════════════════════

def chart_overall(analysis: dict, out_dir: str):
    """Simple bar chart comparing overall pass@1 and pass@3 for each model."""
    pak = analysis["pass_at_k"]
    models = list(pak.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.3

    for ki, k in enumerate([1, 3]):
        means = []
        for model in models:
            # Average across tiers
            all_vals = []
            for tier in TIERS:
                all_vals.extend(pak[model][tier][f"pass@{k}"]["values"])
            means.append(np.mean(all_vals) if all_vals else 0)

        bars = ax.bar(x + (ki - 0.5) * width, means, width,
                      label=f"pass@{k}",
                      color=["#4285F4", "#34A853"][ki],
                      alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(m) for m in models])
    ax.set_ylabel("Overall Pass Rate")
    ax.set_title("Overall Model Performance")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "overall_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6 — failure count heat-grid (model × tier × category)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_failure_heatgrid(analysis: dict, out_dir: str):
    """Grid of failure counts: rows = (model, tier), columns = failure categories."""
    tbt = analysis["taxonomy_by_tier"]
    models = list(tbt.keys())

    row_labels = []
    data = []
    for model in models:
        for tier in TIERS:
            row_labels.append(f"{_model_label(model)}\n{tier}")
            row_data = [tbt[model][tier]["distribution"].get(cat, 0) for cat in FAIL_CATS]
            data.append(row_data)

    data = np.array(data, dtype=float)

    fig, ax = plt.subplots(figsize=(8, max(4, len(row_labels) * 0.7)))
    im = ax.imshow(data, cmap="OrRd", aspect="auto")

    ax.set_xticks(range(len(FAIL_CATS)))
    ax.set_xticklabels(FAIL_CATS, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = int(data[i, j])
            color = "white" if data[i, j] > data.max() * 0.55 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.7, label="Failure Count")
    ax.set_title("Failure Counts by Model, Tier, and Category", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "failure_heatgrid.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FINDINGS REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_findings(analysis: dict, out_dir: str):
    """Write findings.md summarizing the analysis."""
    pak = analysis["pass_at_k"]
    models = list(pak.keys())
    lines = []
    lines.append("# Findings Report")
    lines.append("")
    lines.append("## RQ1: Does pass@1 decline monotonically with difficulty?")
    lines.append("")
    for model in models:
        vals = [pak[model][t]["pass@1"]["mean"] for t in TIERS]
        mono = vals[0] >= vals[1] >= vals[2]
        lines.append(f"**{_model_label(model)}**: pass@1 across Easy/Medium/Hard = "
                      f"{vals[0]:.3f} / {vals[1]:.3f} / {vals[2]:.3f} — "
                      f"{'monotonically declining' if mono else 'NOT monotonically declining'}.")
    lines.append("")

    lines.append("## RQ2: Does failure composition shift across tiers?")
    lines.append("")
    tbt = analysis["taxonomy_by_tier"]
    for model in models:
        lines.append(f"**{_model_label(model)}**:")
        for tier in TIERS:
            t = tbt[model][tier]
            parts = []
            for cat in FAIL_CATS:
                pct = t["proportions"].get(cat, 0) * 100
                if pct > 0:
                    parts.append(f"{cat} {pct:.0f}%")
            lines.append(f"  - {tier}: {', '.join(parts) if parts else 'no failures'}")
        lines.append("")

    lines.append("## RQ3: Does pass@3 preferentially recover specific failure types?")
    lines.append("")
    recovery = analysis.get("pass3_recovery", {})
    for model in models:
        if model in recovery:
            lines.append(f"**{_model_label(model)}**:")
            for cat, stats in recovery[model].items():
                lines.append(f"  - {cat}: {stats['recovered']}/{stats['total']} "
                              f"recovered ({stats['rate']*100:.0f}%)")
        else:
            lines.append(f"**{_model_label(model)}**: no recovery data.")
    lines.append("")

    lines.append("## RQ4: Model-scale effect on failure distribution?")
    lines.append("")
    if len(models) == 2:
        tax = analysis["taxonomy"]
        for model in models:
            t = tax[model]
            parts = [f"{cat} {t['proportions'].get(cat,0)*100:.0f}%"
                     for cat in FAIL_CATS if t['distribution'].get(cat, 0) > 0]
            lines.append(f"**{_model_label(model)}**: {', '.join(parts)}")
        if "overlap" in analysis:
            lines.append(f"\nCross-model Jaccard similarity: "
                          f"{analysis['overlap']['jaccard_similarity']:.3f}")
    lines.append("")

    path = os.path.join(out_dir, "findings.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate charts and findings report")
    parser.add_argument("--analysis", default="results/analysis.json",
                        help="Path to analysis.json (from eval_harness.py)")
    parser.add_argument("--out", default=config.CHARTS_DIR,
                        help="Output directory for charts")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading analysis...")
    analysis = _load_analysis(args.analysis)

    print("Generating charts...")
    chart_pass_at_k(analysis, args.out)
    chart_failure_taxonomy(analysis, args.out)
    chart_overlap_heatmap(analysis, args.out)
    chart_recovery(analysis, args.out)
    chart_overall(analysis, args.out)
    chart_failure_heatgrid(analysis, args.out)

    print("Generating findings report...")
    generate_findings(analysis, args.out)

    print(f"\nAll outputs saved to {args.out}/")


if __name__ == "__main__":
    main()
