"""
Metrics — pass@k computation, bootstrap confidence intervals, and
cross-model analysis for the code correctness benchmark.
"""

import numpy as np
from math import comb
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PASS@K METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k from Chen et al. (2021).

    Args:
        n: Total number of samples per problem
        c: Number of correct samples
        k: k value (1 or 3 in our study)

    Returns:
        Estimated pass@k probability for this problem
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def compute_pass_at_k_for_problem(results: list[bool], k: int) -> float:
    """
    Compute pass@k for a single problem given a list of pass/fail booleans.

    Args:
        results: List of True/False for each sample
        k: k value

    Returns:
        pass@k estimate
    """
    n = len(results)
    c = sum(results)
    return pass_at_k(n, c, k)


def compute_pass_at_k_batch(
    all_results: dict,  # {problem_id: [bool, bool, bool]}
    k: int,
) -> dict:
    """
    Compute pass@k for each problem.

    Returns:
        Dict mapping problem_id -> pass@k value
    """
    return {
        pid: compute_pass_at_k_for_problem(results, k)
        for pid, results in all_results.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        values: Sample values
        n_resamples: Number of bootstrap resamples
        ci: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        (mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)

    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    mean = np.mean(values)

    return float(mean), float(lower), float(upper)


def compute_metrics_by_tier(
    results_df: list[dict],
    k_values: list[int] = [1, 3],
    n_resamples: int = 1000,
) -> dict:
    """
    Compute pass@k metrics grouped by (model, tier) with bootstrap CIs.

    Args:
        results_df: List of result dicts, each containing:
            problem_id, tier, model, sample_idx, passed
        k_values: List of k values to compute
        n_resamples: Bootstrap resamples

    Returns:
        Nested dict: {model: {tier: {f"pass@{k}": {mean, ci_lower, ci_upper}}}}
    """
    # Group results by (model, problem_id)
    grouped = defaultdict(lambda: defaultdict(list))
    problem_tiers = {}

    for row in results_df:
        key = (row["model"], row["problem_id"])
        grouped[key[0]][row["problem_id"]].append(row["passed"])
        problem_tiers[row["problem_id"]] = row["tier"]

    metrics = {}
    for model in grouped:
        metrics[model] = {}

        # Group problems by tier
        tier_problems = defaultdict(list)
        for pid, results in grouped[model].items():
            tier = problem_tiers[pid]
            tier_problems[tier].append((pid, results))

        for tier in ["Easy", "Medium", "Hard"]:
            metrics[model][tier] = {}
            problems = tier_problems.get(tier, [])

            for k in k_values:
                # Compute pass@k for each problem in this tier
                pass_k_values = []
                for pid, results in problems:
                    pk = compute_pass_at_k_for_problem(results, k)
                    pass_k_values.append(pk)

                if pass_k_values:
                    mean, ci_lo, ci_hi = bootstrap_ci(
                        pass_k_values, n_resamples=n_resamples
                    )
                else:
                    mean = ci_lo = ci_hi = 0.0

                metrics[model][tier][f"pass@{k}"] = {
                    "mean": round(mean, 4),
                    "ci_lower": round(ci_lo, 4),
                    "ci_upper": round(ci_hi, 4),
                    "n_problems": len(pass_k_values),
                    "values": [round(v, 4) for v in pass_k_values],
                }

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-MODEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_failure_overlap(
    classifications: dict,  # {model: {problem_id: [classification per sample]}}
) -> dict:
    """
    Compute cross-model failure overlap.

    For each problem, check if both models fail with the same failure type.

    Returns:
        Dict with overlap statistics
    """
    models = list(classifications.keys())
    if len(models) != 2:
        raise ValueError("Expected exactly 2 models for overlap analysis")

    m1, m2 = models
    problems = set(classifications[m1].keys()) & set(classifications[m2].keys())

    overlap_matrix = defaultdict(lambda: defaultdict(int))
    both_fail = 0
    total = len(problems)

    for pid in problems:
        # Get dominant failure type for each model (mode of samples)
        types_m1 = classifications[m1][pid]
        types_m2 = classifications[m2][pid]

        dom_m1 = max(set(types_m1), key=types_m1.count)
        dom_m2 = max(set(types_m2), key=types_m2.count)

        overlap_matrix[dom_m1][dom_m2] += 1

        if dom_m1 != "Pass" and dom_m2 != "Pass":
            both_fail += 1

    # Compute Jaccard similarity for failures
    m1_fails = sum(1 for pid in problems
                   if max(set(classifications[m1][pid]), key=classifications[m1][pid].count) != "Pass")
    m2_fails = sum(1 for pid in problems
                   if max(set(classifications[m2][pid]), key=classifications[m2][pid].count) != "Pass")

    union_fails = m1_fails + m2_fails - both_fail
    jaccard = both_fail / union_fails if union_fails > 0 else 0.0

    return {
        "overlap_matrix": dict(overlap_matrix),
        "both_fail": both_fail,
        "total_problems": total,
        f"{m1}_fails": m1_fails,
        f"{m2}_fails": m2_fails,
        "jaccard_similarity": round(jaccard, 4),
    }


def compute_pass3_recovery(
    results_df: list[dict],
    classifications_per_sample: dict,  # {model: {problem_id: {sample_idx: classification}}}
) -> dict:
    """
    Analyze which failure types are recovered by pass@3 majority voting.

    For each problem where pass@1 fails but pass@3 succeeds:
    - Record the failure types of the failing samples
    - This tells us which errors are "recoverable" by sampling more

    Returns:
        Dict: {model: {failure_type: {"recovered": int, "total": int, "rate": float}}}
    """
    # Group results
    grouped = defaultdict(lambda: defaultdict(list))
    for row in results_df:
        grouped[row["model"]][row["problem_id"]].append(row["passed"])

    recovery = {}
    for model in grouped:
        recovery[model] = defaultdict(lambda: {"recovered": 0, "total": 0})

        for pid, results in grouped[model].items():
            p1 = compute_pass_at_k_for_problem(results, 1)
            p3 = compute_pass_at_k_for_problem(results, 3)

            # pass@1 fails but pass@3 succeeds
            if p1 < 1.0 and p3 == 1.0:
                # Get failure types of failing samples
                if model in classifications_per_sample and pid in classifications_per_sample[model]:
                    for sidx, cls in classifications_per_sample[model][pid].items():
                        if cls != "Pass":
                            recovery[model][cls]["recovered"] += 1
                            recovery[model][cls]["total"] += 1

            # pass@1 fails and pass@3 also fails
            elif p1 < 1.0 and p3 < 1.0:
                if model in classifications_per_sample and pid in classifications_per_sample[model]:
                    for sidx, cls in classifications_per_sample[model][pid].items():
                        if cls != "Pass":
                            recovery[model][cls]["total"] += 1

        # Compute rates
        for cls in recovery[model]:
            total = recovery[model][cls]["total"]
            recovered = recovery[model][cls]["recovered"]
            recovery[model][cls]["rate"] = round(recovered / total, 4) if total > 0 else 0.0

    return dict(recovery)
