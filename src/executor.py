"""
Sandboxed Executor — runs model-generated code in isolated subprocesses.

Each solution is executed in a separate Python subprocess with:
- Strict timeout enforcement (default 5 seconds)
- stdout/stderr capture for error analysis
- JSON-formatted test results
"""

import json
import subprocess
import tempfile
import os
import copy
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExecutionResult:
    """Result of executing a single solution against test cases."""

    def __init__(
        self,
        passed: bool,
        test_results: list[dict],
        error_type: str = None,
        error_message: str = None,
        stderr_output: str = "",
        timed_out: bool = False,
        syntax_valid: bool = True,
    ):
        self.passed = passed
        self.test_results = test_results  # per-test-case results
        self.error_type = error_type
        self.error_message = error_message
        self.stderr_output = stderr_output
        self.timed_out = timed_out
        self.syntax_valid = syntax_valid

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "test_results": self.test_results,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stderr_output": self.stderr_output[:500],  # truncate
            "timed_out": self.timed_out,
            "syntax_valid": self.syntax_valid,
        }


def check_syntax(code: str) -> tuple[bool, str]:
    """Check if code has syntax errors without executing it."""
    try:
        compile(code, "<solution>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def build_test_script(
    solution_code: str,
    entry_point: str,
    test_cases: list[dict],
) -> str:
    """
    Build a self-contained Python script that:
    1. Defines the model's function
    2. Runs each test case
    3. Outputs results as JSON to stdout
    """
    # Serialize test cases
    tc_json = json.dumps(test_cases)

    script = f'''
import json
import sys
import copy

# ── Model Solution ──
{solution_code}

# ── Test Runner ──
test_cases = json.loads('{tc_json.replace(chr(39), chr(92)+chr(39))}')
results = []

for i, tc in enumerate(test_cases):
    try:
        # Deep copy inputs to prevent mutation
        inputs = copy.deepcopy(tc["input"])
        expected = tc["expected"]

        actual = {entry_point}(*inputs)

        # Compare results
        passed = actual == expected
        results.append({{
            "test_idx": i,
            "passed": passed,
            "expected": repr(expected),
            "actual": repr(actual),
            "error": None,
            "error_type": None,
        }})
    except Exception as e:
        results.append({{
            "test_idx": i,
            "passed": False,
            "expected": repr(tc["expected"]),
            "actual": None,
            "error": str(e),
            "error_type": type(e).__name__,
        }})

print(json.dumps(results))
'''
    return script


def execute_solution(
    solution_code: str,
    problem: dict,
    timeout: int = 5,
) -> ExecutionResult:
    """
    Execute a solution against all test cases in a sandboxed subprocess.

    Args:
        solution_code: The model-generated Python code
        problem: Problem dict from benchmark.json
        timeout: Maximum execution time in seconds

    Returns:
        ExecutionResult with detailed per-test-case outcomes
    """
    entry_point = problem["entry_point"]

    # Step 1: Syntax check (fast, no subprocess needed)
    syntax_ok, syntax_err = check_syntax(solution_code)
    if not syntax_ok:
        return ExecutionResult(
            passed=False,
            test_results=[],
            error_type="SyntaxError",
            error_message=syntax_err,
            syntax_valid=False,
        )

    # Step 2: Flatten all test cases
    all_tests = (
        problem["test_cases"]["standard"]
        + problem["test_cases"]["boundary"]
        + problem["test_cases"]["adversarial"]
    )

    # Step 3: Build and run test script
    script = build_test_script(solution_code, entry_point, all_tests)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(script)
            temp_path = f.name

        result = subprocess.run(
            ["python3", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        # Parse results
        if result.returncode == 0 and result.stdout.strip():
            try:
                test_results = json.loads(result.stdout.strip())
                all_passed = all(r["passed"] for r in test_results)

                return ExecutionResult(
                    passed=all_passed,
                    test_results=test_results,
                    stderr_output=result.stderr,
                )
            except json.JSONDecodeError:
                return ExecutionResult(
                    passed=False,
                    test_results=[],
                    error_type="OutputParseError",
                    error_message=f"Could not parse test output: {result.stdout[:200]}",
                    stderr_output=result.stderr,
                )
        else:
            # Runtime error — extract error type from stderr
            error_type, error_msg = _parse_stderr(result.stderr)
            return ExecutionResult(
                passed=False,
                test_results=[],
                error_type=error_type,
                error_message=error_msg,
                stderr_output=result.stderr,
            )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            passed=False,
            test_results=[],
            error_type="Timeout",
            error_message=f"Execution exceeded {timeout}s timeout",
            timed_out=True,
        )
    except Exception as e:
        return ExecutionResult(
            passed=False,
            test_results=[],
            error_type="ExecutionError",
            error_message=str(e),
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except (OSError, NameError):
            pass


def _parse_stderr(stderr: str) -> tuple[str, str]:
    """Extract the error type and message from Python stderr output."""
    if not stderr:
        return "UnknownError", "No stderr output"

    # Look for the last traceback line (the actual error)
    lines = stderr.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if ':' in line and not line.startswith('File') and not line.startswith('Traceback'):
            parts = line.split(':', 1)
            error_type = parts[0].strip()
            error_msg = parts[1].strip() if len(parts) > 1 else ""
            return error_type, error_msg

    return "RuntimeError", stderr[-200:]


def execute_all_tests(
    solution_code: str,
    problem: dict,
    timeout: int = 5,
) -> dict:
    """
    Execute and return a structured result dict suitable for CSV output.
    """
    result = execute_solution(solution_code, problem, timeout)

    # Determine how many standard/boundary/adversarial tests passed
    n_std = len(problem["test_cases"]["standard"])
    n_bnd = len(problem["test_cases"]["boundary"])
    n_adv = len(problem["test_cases"]["adversarial"])

    std_pass = bnd_pass = adv_pass = 0
    if result.test_results:
        for i, tr in enumerate(result.test_results):
            if tr["passed"]:
                if i < n_std:
                    std_pass += 1
                elif i < n_std + n_bnd:
                    bnd_pass += 1
                else:
                    adv_pass += 1

    return {
        "passed": result.passed,
        "syntax_valid": result.syntax_valid,
        "timed_out": result.timed_out,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "stderr_output": result.stderr_output[:500] if result.stderr_output else "",
        "standard_passed": std_pass,
        "standard_total": n_std,
        "boundary_passed": bnd_pass,
        "boundary_total": n_bnd,
        "adversarial_passed": adv_pass,
        "adversarial_total": n_adv,
        "test_results": result.test_results,
    }
