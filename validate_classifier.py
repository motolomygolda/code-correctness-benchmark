#!/usr/bin/env python3
"""
validate_classifier.py — Sample and export failures for manual validation.

Draws a stratified 10% random sample of classified failures, exports them
to a CSV for human review, and computes inter-rater agreement (Cohen's kappa)
once manual labels are filled in.

Usage:
    # Step 1: Export sample for manual review
    python validate_classifier.py --export

    # Step 2: After filling in 'manual_label' column in the CSV
    python validate_classifier.py --score
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

FAIL_CATS = ["Syntactic", "Runtime", "Algorithmic", "Edge-case"]


def load_results(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["passed"] = row["passed"] == "True"
            rows.append(row)
    return rows


def export_sample(results_path: str, output_path: str, sample_rate: float = 0.10, seed: int = 42):
    """
    Draw a stratified sample of failures and export for manual review.

    Stratified by failure_type to ensure representation of all categories.
    """
    random.seed(seed)
    rows = load_results(results_path)

    # Group failures by category
    by_cat = defaultdict(list)
    for row in rows:
        if row["failure_type"] in FAIL_CATS:
            by_cat[row["failure_type"]].append(row)

    sampled = []
    for cat, cat_rows in by_cat.items():
        n = max(1, int(len(cat_rows) * sample_rate))
        chosen = random.sample(cat_rows, min(n, len(cat_rows)))
        for row in chosen:
            sampled.append({
                "problem_id":       row["problem_id"],
                "tier":             row["tier"],
                "model":            row["model"],
                "sample_idx":       row["sample_idx"],
                "auto_label":       row["failure_type"],
                "error_type":       row.get("error_type", ""),
                "error_message":    row.get("error_message", "")[:150],
                "standard_passed":  row.get("standard_passed", ""),
                "boundary_passed":  row.get("boundary_passed", ""),
                "adversarial_passed": row.get("adversarial_passed", ""),
                "manual_label":     "",  # <-- human fills this in
                "notes":            "",  # <-- optional reviewer notes
            })

    random.shuffle(sampled)  # randomize order to avoid bias

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sampled[0].keys()))
        writer.writeheader()
        writer.writerows(sampled)

    print(f"Exported {len(sampled)} samples to {output_path}")
    print(f"  Syntactic:   {sum(1 for s in sampled if s['auto_label'] == 'Syntactic')}")
    print(f"  Runtime:     {sum(1 for s in sampled if s['auto_label'] == 'Runtime')}")
    print(f"  Algorithmic: {sum(1 for s in sampled if s['auto_label'] == 'Algorithmic')}")
    print(f"  Edge-case:   {sum(1 for s in sampled if s['auto_label'] == 'Edge-case')}")
    print(f"\nFill in the 'manual_label' column with one of: {FAIL_CATS}")
    print(f"Then run: python validate_classifier.py --score")


def compute_agreement(sample_path: str):
    """
    Compute agreement between automated and manual labels.

    Reports:
    - Overall percent agreement
    - Cohen's kappa
    - Per-category precision/recall
    - Confusion matrix
    """
    rows = []
    with open(sample_path) as f:
        for row in csv.DictReader(f):
            if row.get("manual_label", "").strip():
                rows.append(row)

    if not rows:
        print("ERROR: No manual labels found. Fill in the 'manual_label' column first.")
        return

    total = len(rows)
    agree = sum(1 for r in rows if r["auto_label"] == r["manual_label"])
    pct = agree / total * 100

    print(f"\n{'='*50}")
    print(f"CLASSIFIER VALIDATION REPORT")
    print(f"{'='*50}")
    print(f"Samples reviewed:    {total}")
    print(f"Agreement:           {agree}/{total} ({pct:.1f}%)")

    # Cohen's kappa
    labels = FAIL_CATS
    label_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)

    # Build confusion matrix
    conf = [[0] * n for _ in range(n)]
    for r in rows:
        auto = r["auto_label"]
        manual = r["manual_label"]
        if auto in label_idx and manual in label_idx:
            conf[label_idx[auto]][label_idx[manual]] += 1

    # Print confusion matrix
    print(f"\nConfusion Matrix (rows=auto, cols=manual):")
    header = "            " + "".join(f"{l:>12}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row_str = f"{label:>12}" + "".join(f"{conf[i][j]:>12}" for j in range(n))
        print(row_str)

    # Compute kappa
    po = agree / total  # observed agreement
    # Expected agreement
    pe = 0
    for k in range(n):
        row_sum = sum(conf[k][j] for j in range(n))
        col_sum = sum(conf[i][k] for i in range(n))
        pe += (row_sum / total) * (col_sum / total)

    kappa = (po - pe) / (1 - pe) if pe < 1 else 0
    print(f"\nCohen's kappa:       {kappa:.3f}")

    # Interpretation
    if kappa >= 0.81:
        interp = "Almost perfect"
    elif kappa >= 0.61:
        interp = "Substantial"
    elif kappa >= 0.41:
        interp = "Moderate"
    elif kappa >= 0.21:
        interp = "Fair"
    else:
        interp = "Slight/Poor"
    print(f"Interpretation:      {interp}")

    # Per-category precision/recall
    print(f"\nPer-Category Performance:")
    print(f"{'Category':>15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for i, label in enumerate(labels):
        tp = conf[i][i]
        fp = sum(conf[i][j] for j in range(n)) - tp
        fn = sum(conf[j][i] for j in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{label:>15} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description="Classifier validation tool")
    parser.add_argument("--export", action="store_true",
                        help="Export sample for manual review")
    parser.add_argument("--score", action="store_true",
                        help="Score manual labels and compute agreement")
    parser.add_argument("--results", default="results/results_raw.csv")
    parser.add_argument("--sample-file", default="results/validation_sample.csv")
    parser.add_argument("--rate", type=float, default=config.MANUAL_VALIDATION_SAMPLE_RATE)
    args = parser.parse_args()

    if args.export:
        export_sample(args.results, args.sample_file, args.rate)
    elif args.score:
        compute_agreement(args.sample_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
