#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# run.sh — Full pipeline for the Code Correctness Benchmark
#
# Usage:
#   ./run.sh mock     — Run with simulated API calls (testing)
#   ./run.sh full     — Run with real API calls (requires keys in .env)
#   ./run.sh analyze  — Re-run analysis on existing results_raw.csv
#   ./run.sh charts   — Regenerate charts from existing analysis.json
#   ./run.sh validate — Export classifier validation sample
#   ./run.sh all      — Full pipeline end-to-end (mock mode)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODE="${1:-all}"
cd "$(dirname "$0")"

# Load API keys if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "================================================"
echo "  Code Correctness Benchmark Pipeline"
echo "  Mode: $MODE"
echo "================================================"

case "$MODE" in
    mock|full)
        echo ""
        echo "[Step 1/4] Running evaluation harness ($MODE mode)..."
        python3 eval_harness.py --mode "$MODE"

        echo ""
        echo "[Step 2/4] Generating charts..."
        python3 generate_charts.py

        echo ""
        echo "[Step 3/4] Exporting classifier validation sample..."
        python3 validate_classifier.py --export

        echo ""
        echo "[Step 4/4] Done!"
        echo "  Results:    results/results_raw.csv"
        echo "  Analysis:   results/analysis.json"
        echo "  Charts:     charts/"
        echo "  Findings:   charts/findings.md"
        echo "  Validation: results/validation_sample.csv"
        ;;

    analyze)
        echo ""
        echo "[1/2] Re-running analysis..."
        python3 eval_harness.py --mode analyze

        echo ""
        echo "[2/2] Regenerating charts..."
        python3 generate_charts.py
        ;;

    charts)
        echo ""
        echo "Regenerating charts from results/analysis.json..."
        python3 generate_charts.py
        ;;

    validate)
        echo ""
        echo "Exporting validation sample..."
        python3 validate_classifier.py --export
        ;;

    all)
        echo ""
        echo "[Step 1/4] Running evaluation harness (mock)..."
        python3 eval_harness.py --mode mock

        echo ""
        echo "[Step 2/4] Generating charts..."
        python3 generate_charts.py

        echo ""
        echo "[Step 3/4] Exporting classifier validation sample..."
        python3 validate_classifier.py --export

        echo ""
        echo "[Step 4/4] Pipeline complete!"
        echo ""
        echo "  Outputs:"
        echo "    results/results_raw.csv          — 540 rows of raw results"
        echo "    results/analysis.json             — full analysis data"
        echo "    charts/pass_at_k.png              — pass@k bar chart"
        echo "    charts/failure_taxonomy.png        — failure distribution"
        echo "    charts/overlap_heatmap.png         — cross-model overlap"
        echo "    charts/recovery_rates.png          — pass@3 recovery"
        echo "    charts/overall_comparison.png      — overall comparison"
        echo "    charts/failure_heatgrid.png         — failure count grid"
        echo "    charts/findings.md                 — written findings"
        echo "    results/validation_sample.csv      — classifier validation"
        ;;

    *)
        echo "Usage: ./run.sh {mock|full|analyze|charts|validate|all}"
        exit 1
        ;;
esac
