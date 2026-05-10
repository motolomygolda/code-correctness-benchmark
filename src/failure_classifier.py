"""
Failure Classifier — assigns each failed solution to exactly one of four
mutually exclusive failure categories:

1. Syntactic   — SyntaxError or NameError before meaningful execution
2. Runtime     — uncaught exception during execution
3. Algorithmic — wrong output on standard test cases
4. Edge-case   — passes standard tests, fails on boundary/adversarial tests

Classification is hierarchical: earlier categories take priority.
"""

import logging

logger = logging.getLogger(__name__)

# Error types considered "syntactic" (pre-execution failures)
SYNTACTIC_ERRORS = {"SyntaxError", "IndentationError", "TabError", "NameError"}

# Error types considered "runtime" (execution-time crashes)
RUNTIME_ERRORS = {
    "TypeError", "ValueError", "IndexError", "KeyError",
    "AttributeError", "ZeroDivisionError", "RecursionError",
    "OverflowError", "StopIteration", "RuntimeError",
    "UnboundLocalError", "FileNotFoundError", "MemoryError",
    "ImportError", "ModuleNotFoundError", "Timeout",
}

# All four failure categories
FAILURE_CATEGORIES = ["Syntactic", "Runtime", "Algorithmic", "Edge-case"]


def classify_failure(execution_result: dict) -> str:
    """
    Classify a failed execution into one of four categories.

    Args:
        execution_result: Dict from executor.execute_all_tests()

    Returns:
        One of: "Pass", "Syntactic", "Runtime", "Algorithmic", "Edge-case"
    """
    # If it passed all tests, it's not a failure
    if execution_result["passed"]:
        return "Pass"

    # ── Step 1: Syntactic check ───────────────────────────────────────────
    if not execution_result["syntax_valid"]:
        return "Syntactic"

    error_type = execution_result.get("error_type", "")

    # NameError is syntactic (undefined variable/function)
    if error_type in SYNTACTIC_ERRORS:
        return "Syntactic"

    # Check stderr for syntactic errors
    stderr = execution_result.get("stderr_output", "")
    for syn_err in SYNTACTIC_ERRORS:
        if syn_err in stderr:
            return "Syntactic"

    # ── Step 2: Runtime check ─────────────────────────────────────────────
    if execution_result["timed_out"]:
        return "Runtime"

    if error_type in RUNTIME_ERRORS:
        return "Runtime"

    # Check if any individual test case threw a runtime exception
    test_results = execution_result.get("test_results", [])
    if test_results:
        for tr in test_results:
            et = tr.get("error_type", "")
            if et in RUNTIME_ERRORS:
                return "Runtime"

    # If there are no test results but there was an error, it's runtime
    if not test_results and error_type:
        return "Runtime"

    # ── Step 3: Algorithmic vs Edge-case ──────────────────────────────────
    # Check whether standard tests passed
    std_passed = execution_result.get("standard_passed", 0)
    std_total = execution_result.get("standard_total", 0)

    if std_total > 0 and std_passed < std_total:
        # Failed on at least one standard test → Algorithmic
        return "Algorithmic"

    # Standard tests all passed — check boundary and adversarial
    bnd_passed = execution_result.get("boundary_passed", 0)
    bnd_total = execution_result.get("boundary_total", 0)
    adv_passed = execution_result.get("adversarial_passed", 0)
    adv_total = execution_result.get("adversarial_total", 0)

    if (bnd_total > 0 and bnd_passed < bnd_total) or \
       (adv_total > 0 and adv_passed < adv_total):
        return "Edge-case"

    # ── Fallback ──────────────────────────────────────────────────────────
    # If we get here, something unexpected happened
    # Check test_results for any failed test with wrong output
    if test_results:
        n_std = std_total
        n_bnd = bnd_total
        for i, tr in enumerate(test_results):
            if not tr["passed"] and tr.get("error_type") is None:
                # Wrong output (not an exception)
                if i < n_std:
                    return "Algorithmic"
                else:
                    return "Edge-case"

    # Last resort: classify as Algorithmic
    logger.warning(
        f"Could not cleanly classify failure: "
        f"error_type={error_type}, "
        f"std={std_passed}/{std_total}, "
        f"bnd={bnd_passed}/{bnd_total}, "
        f"adv={adv_passed}/{adv_total}"
    )
    return "Algorithmic"


def classify_batch(results: list[dict]) -> list[str]:
    """Classify a batch of execution results."""
    return [classify_failure(r) for r in results]


def compute_taxonomy_distribution(classifications: list[str]) -> dict:
    """
    Compute the distribution of failure types.

    Returns:
        Dict mapping category -> count
    """
    dist = {cat: 0 for cat in FAILURE_CATEGORIES}
    dist["Pass"] = 0

    for c in classifications:
        dist[c] = dist.get(c, 0) + 1

    return dist


def taxonomy_summary(
    classifications: list[str],
    problem_ids: list[str] = None,
) -> dict:
    """
    Produce a summary report of the failure taxonomy.

    Returns dict with:
        - distribution: counts per category
        - proportions: fractions per category (failures only)
        - total_failures: count of non-Pass
        - total_pass: count of Pass
        - details: per-category list of problem_ids (if provided)
    """
    dist = compute_taxonomy_distribution(classifications)
    total = len(classifications)
    total_pass = dist.get("Pass", 0)
    total_fail = total - total_pass

    proportions = {}
    for cat in FAILURE_CATEGORIES:
        proportions[cat] = dist[cat] / total_fail if total_fail > 0 else 0.0

    details = {cat: [] for cat in FAILURE_CATEGORIES}
    if problem_ids:
        for pid, cls in zip(problem_ids, classifications):
            if cls != "Pass":
                details[cls].append(pid)

    return {
        "distribution": dist,
        "proportions": proportions,
        "total": total,
        "total_pass": total_pass,
        "total_failures": total_fail,
        "details": details,
    }
