"""
Response Parser — extracts executable Python function code from raw model outputs.

Handles: markdown fences, explanatory text, full-function vs body-only responses,
indentation issues, and common formatting artifacts.
"""

import re
import textwrap


def extract_code(raw_response: str, prompt: str, entry_point: str) -> str:
    """
    Extract a clean, executable Python function from the model's raw output.

    Strategy (in priority order):
    1. Look for ```python ... ``` fenced code blocks
    2. Look for ``` ... ``` generic fenced code blocks
    3. Look for a def statement matching the entry point
    4. Treat the entire response as code

    Then ensure the code contains a complete function definition.
    """
    if not raw_response or not raw_response.strip():
        return ""

    code = None

    # Strategy 1 & 2: Extract from markdown code fences
    # Try python-specific fences first, then generic
    patterns = [
        r'```python\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
        r'```python(.*?)```',
        r'```(.*?)```',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, raw_response, re.DOTALL)
        if matches:
            # Take the longest match (most likely to be the full solution)
            code = max(matches, key=len).strip()
            break

    # Strategy 3: Find the def statement
    if code is None:
        lines = raw_response.split('\n')
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith(f'def {entry_point}'):
                start_idx = i
                break
        if start_idx is not None:
            # Collect the function body: everything until the next unindented non-empty line
            func_lines = [lines[start_idx]]
            for line in lines[start_idx + 1:]:
                if line.strip() == '':
                    func_lines.append(line)
                elif line[0] in (' ', '\t'):
                    func_lines.append(line)
                else:
                    # Check if this is a new def or class (end of our function)
                    if line.strip().startswith(('def ', 'class ', '#')):
                        break
                    func_lines.append(line)
            code = '\n'.join(func_lines).rstrip()

    # Strategy 4: Use entire response
    if code is None:
        code = raw_response.strip()

    # Post-processing
    code = _post_process(code, prompt, entry_point)
    return code


def _post_process(code: str, prompt: str, entry_point: str) -> str:
    """Clean up extracted code and ensure it's a complete function."""

    # Remove any leading/trailing non-code text
    lines = code.split('\n')

    # If the code doesn't contain a def statement, try to wrap it as function body
    has_def = any(line.strip().startswith(f'def {entry_point}') for line in lines)

    if not has_def:
        # Check if the prompt has the function signature
        prompt_lines = prompt.split('\n')
        sig_line = None
        for line in prompt_lines:
            if line.strip().startswith(f'def {entry_point}'):
                sig_line = line
                break

        if sig_line:
            # The model returned just the body — reconstruct the full function
            # First, strip any docstring from prompt to avoid duplication
            body_lines = _strip_prompt_prefix(lines, prompt)

            # Ensure proper indentation
            indented_body = _ensure_indentation(body_lines)
            code = sig_line + '\n' + indented_body
        else:
            # Can't determine the function signature; return code as-is
            pass
    else:
        # Has a def statement — check for duplicate docstrings
        code = _remove_duplicate_docstring(code, prompt)

    # Remove trailing whitespace and ensure final newline
    code = code.rstrip() + '\n'

    # Final validation: try to compile
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError:
        # Try some common fixes
        code = _attempt_syntax_fix(code)

    return code


def _strip_prompt_prefix(lines: list[str], prompt: str) -> list[str]:
    """Remove lines that are duplicates from the prompt (docstring, etc.)."""
    prompt_lines = set(line.strip() for line in prompt.split('\n'))
    result = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if skip:
                skip = False
                continue
            if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                skip = True
                continue
        if skip:
            continue
        if stripped in prompt_lines and stripped.startswith(('def ', '"""', "'''")):
            continue
        result.append(line)
    return result


def _ensure_indentation(lines: list[str]) -> str:
    """Ensure all lines have at least 4 spaces of indentation."""
    result = []
    for line in lines:
        if line.strip():  # Non-empty line
            # Check if already indented
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
            if current_indent < 4:
                line = '    ' + stripped
        result.append(line)
    return '\n'.join(result)


def _remove_duplicate_docstring(code: str, prompt: str) -> str:
    """If the model included the prompt's docstring, avoid duplication."""
    # This is a light-touch check; just return as-is for now
    return code


def _attempt_syntax_fix(code: str) -> str:
    """Try common syntax fixes on code that doesn't compile."""
    # Fix 1: Remove trailing incomplete lines
    lines = code.rstrip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
        try:
            compile('\n'.join(lines[:i + 1]) + '\n', '<string>', 'exec')
            return '\n'.join(lines[:i + 1]) + '\n'
        except SyntaxError:
            continue

    # Fix 2: Remove 'return' at module level
    fixed = re.sub(r'^return\s+', '# return ', code, flags=re.MULTILINE)
    try:
        compile(fixed, '<string>', 'exec')
        return fixed
    except SyntaxError:
        pass

    return code  # Return original if no fix works


def parse_model_response(raw_text: str, problem: dict) -> str:
    """
    Main entry point: parse a model's raw text response into executable code.

    Args:
        raw_text: Raw model output string
        problem: Problem dict from benchmark.json

    Returns:
        Cleaned Python code string containing the function definition
    """
    return extract_code(raw_text, problem["prompt"], problem["entry_point"])
