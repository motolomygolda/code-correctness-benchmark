#!/usr/bin/env python3
"""
Evaluation Harness — main pipeline orchestrator.

Usage:
    python eval_harness.py --mode mock      # simulated (no APIs)
    python eval_harness.py --mode full      # real API calls
    python eval_harness.py --mode analyze   # reuse existing CSV
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import config
from src.prompt_formatter import format_prompt
from src.response_parser import parse_model_response
from src.executor import execute_all_tests
from src.failure_classifier import classify_failure, taxonomy_summary
from src.metrics import (
    compute_metrics_by_tier,
    compute_failure_overlap,
    compute_pass3_recovery,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("eval_harness.log"),
    ],
)
logger = logging.getLogger("eval_harness")


def load_benchmark(path: str) -> list[dict]:
    with open(path) as f:
        problems = json.load(f)
    logger.info(f"Loaded {len(problems)} problems from {path}")
    tiers = defaultdict(int)
    for p in problems:
        assert "problem_id" in p and "tier" in p
        tiers[p["tier"]] += 1
    for tier, count in sorted(tiers.items()):
        logger.info(f"  {tier}: {count} problems")
    return problems


def generate_all_solutions(problems, mode="mock"):
    if mode == "mock":
        from src.api_caller import MockAPICaller
        caller = MockAPICaller(config, failure_rate=0.35)
    else:
        from src.api_caller import APICaller
        caller = APICaller(config)

    all_responses = {}
    for model_key in config.MODELS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating with {config.MODELS[model_key]['name']}")
        logger.info(f"{'='*60}")
        all_responses[model_key] = {}
        n_samples = config.MODELS[model_key]["samples_per_problem"]
        for i, problem in enumerate(problems):
            logger.info(f"[{i+1}/{len(problems)}] {problem['problem_id']} ({problem['tier']})")
            responses = caller.generate_solutions(
                problem=problem, model_key=model_key,
                num_samples=n_samples, save_dir=config.RAW_RESPONSES_DIR,
            )
            all_responses[model_key][problem["problem_id"]] = responses
    return all_responses


def evaluate_all(problems, all_responses):
    problem_map = {p["problem_id"]: p for p in problems}
    results = []

    for model_key in all_responses:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {config.MODELS[model_key]['name']}")
        logger.info(f"{'='*60}")

        for pid, responses in all_responses[model_key].items():
            problem = problem_map[pid]
            for sample_idx, raw_response in enumerate(responses):
                parsed_code = parse_model_response(raw_response, problem)
                exec_result = execute_all_tests(parsed_code, problem, timeout=config.EXECUTION_TIMEOUT)
                failure_type = classify_failure(exec_result)

                row = {
                    "problem_id": pid,
                    "tier": problem["tier"],
                    "source": problem.get("source", ""),
                    "model": model_key,
                    "model_name": config.MODELS[model_key]["name"],
                    "sample_idx": sample_idx,
                    "passed": exec_result["passed"],
                    "failure_type": failure_type,
                    "error_type": exec_result.get("error_type", ""),
                    "error_message": str(exec_result.get("error_message", ""))[:200],
                    "syntax_valid": exec_result["syntax_valid"],
                    "timed_out": exec_result.get("timed_out", False),
                    "standard_passed": exec_result["standard_passed"],
                    "standard_total": exec_result["standard_total"],
                    "boundary_passed": exec_result["boundary_passed"],
                    "boundary_total": exec_result["boundary_total"],
                    "adversarial_passed": exec_result["adversarial_passed"],
                    "adversarial_total": exec_result["adversarial_total"],
                }
                results.append(row)
                status = "PASS" if exec_result["passed"] else f"FAIL ({failure_type})"
                logger.info(f"  {pid} sample {sample_idx}: {status}")
    return results


def save_results_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "problem_id", "tier", "source", "model", "model_name",
        "sample_idx", "passed", "failure_type", "error_type",
        "error_message", "syntax_valid", "timed_out",
        "standard_passed", "standard_total",
        "boundary_passed", "boundary_total",
        "adversarial_passed", "adversarial_total",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    logger.info(f"Saved {len(results)} results to {path}")


def load_results_csv(path):
    results = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["sample_idx"] = int(row["sample_idx"])
            row["passed"] = row["passed"] == "True"
            row["syntax_valid"] = row["syntax_valid"] == "True"
            row["timed_out"] = row["timed_out"] == "True"
            for k in ("standard_passed","standard_total","boundary_passed",
                       "boundary_total","adversarial_passed","adversarial_total"):
                row[k] = int(row[k])
            results.append(row)
    logger.info(f"Loaded {len(results)} results from {path}")
    return results


def run_analysis(results):
    analysis = {}

    # Pass@k
    logger.info("\nComputing pass@k metrics...")
    metrics = compute_metrics_by_tier(results, k_values=[1, 3], n_resamples=config.BOOTSTRAP_RESAMPLES)
    analysis["pass_at_k"] = metrics

    print("\n" + "="*80)
    print("PASS@K RESULTS")
    print("="*80)
    for model in metrics:
        print(f"\n  Model: {config.MODELS[model]['name']}")
        print(f"  {'Tier':<10} {'pass@1':>25} {'pass@3':>25}")
        print(f"  {'-'*60}")
        for tier in ["Easy", "Medium", "Hard"]:
            p1 = metrics[model][tier]["pass@1"]
            p3 = metrics[model][tier]["pass@3"]
            print(f"  {tier:<10} {p1['mean']:.3f} [{p1['ci_lower']:.3f}, {p1['ci_upper']:.3f}]"
                  f"  {p3['mean']:.3f} [{p3['ci_lower']:.3f}, {p3['ci_upper']:.3f}]")

    # Taxonomy
    logger.info("\nComputing failure taxonomy...")
    taxonomy = {}
    for model in config.MODELS:
        model_results = [r for r in results if r["model"] == model]
        classifications = [r["failure_type"] for r in model_results]
        problem_ids = [r["problem_id"] for r in model_results]
        taxonomy[model] = taxonomy_summary(classifications, problem_ids)
    analysis["taxonomy"] = taxonomy

    print("\n" + "="*80)
    print("FAILURE TAXONOMY")
    print("="*80)
    for model in taxonomy:
        t = taxonomy[model]
        print(f"\n  Model: {config.MODELS[model]['name']}")
        print(f"  Total: {t['total']} | Pass: {t['total_pass']} | Fail: {t['total_failures']}")
        for cat in ["Syntactic", "Runtime", "Algorithmic", "Edge-case"]:
            count = t["distribution"].get(cat, 0)
            pct = t["proportions"].get(cat, 0) * 100
            print(f"    {cat:<15} {count:>5}  ({pct:.1f}%)")

    # Taxonomy by tier
    taxonomy_by_tier = {}
    for model in config.MODELS:
        taxonomy_by_tier[model] = {}
        for tier in ["Easy", "Medium", "Hard"]:
            tier_results = [r for r in results if r["model"] == model and r["tier"] == tier]
            classifications = [r["failure_type"] for r in tier_results]
            taxonomy_by_tier[model][tier] = taxonomy_summary(classifications)
    analysis["taxonomy_by_tier"] = taxonomy_by_tier

    print("\n" + "="*80)
    print("FAILURE TAXONOMY BY TIER")
    print("="*80)
    for model in taxonomy_by_tier:
        print(f"\n  Model: {config.MODELS[model]['name']}")
        for tier in ["Easy", "Medium", "Hard"]:
            t = taxonomy_by_tier[model][tier]
            print(f"    {tier} (Fail: {t['total_failures']}/{t['total']})")
            for cat in ["Syntactic", "Runtime", "Algorithmic", "Edge-case"]:
                count = t["distribution"].get(cat, 0)
                pct = t["proportions"].get(cat, 0) * 100
                if count > 0:
                    print(f"      {cat:<15} {count:>4} ({pct:.1f}%)")

    # Cross-model overlap
    if len(config.MODELS) == 2:
        classifications_by_model = {}
        for model in config.MODELS:
            classifications_by_model[model] = defaultdict(list)
            for r in results:
                if r["model"] == model:
                    classifications_by_model[model][r["problem_id"]].append(r["failure_type"])
        overlap = compute_failure_overlap(dict(classifications_by_model))
        analysis["overlap"] = overlap
        print(f"\n  Cross-model Jaccard overlap: {overlap['jaccard_similarity']:.3f}")

    # Pass@3 recovery
    cls_per_sample = {}
    for model in config.MODELS:
        cls_per_sample[model] = defaultdict(dict)
        for r in results:
            if r["model"] == model:
                cls_per_sample[model][r["problem_id"]][r["sample_idx"]] = r["failure_type"]
    recovery = compute_pass3_recovery(results, dict(cls_per_sample))
    analysis["pass3_recovery"] = recovery

    print("\n" + "="*80)
    print("PASS@3 RECOVERY RATES")
    print("="*80)
    for model in recovery:
        print(f"\n  Model: {config.MODELS[model]['name']}")
        for cat, stats in recovery[model].items():
            print(f"    {cat:<15} {stats['recovered']}/{stats['total']} ({stats['rate']*100:.1f}%)")

    # Save
    analysis_path = os.path.join(config.RESULTS_DIR, "analysis.json")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"\nAnalysis saved to {analysis_path}")
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Code Correctness Evaluation Harness")
    parser.add_argument("--mode", choices=["full", "mock", "analyze"], default="mock")
    parser.add_argument("--benchmark", default=config.BENCHMARK_PATH)
    args = parser.parse_args()

    results_csv_path = os.path.join(config.RESULTS_DIR, "results_raw.csv")

    if args.mode in ("full", "mock"):
        problems = load_benchmark(args.benchmark)
        all_responses = generate_all_solutions(problems, mode=args.mode)
        results = evaluate_all(problems, all_responses)
        save_results_csv(results, results_csv_path)

    if not os.path.exists(results_csv_path):
        logger.error(f"Results not found: {results_csv_path}")
        sys.exit(1)

    results = load_results_csv(results_csv_path)
    analysis = run_analysis(results)
    logger.info("\nPipeline complete!")
    return analysis


if __name__ == "__main__":
    main()
