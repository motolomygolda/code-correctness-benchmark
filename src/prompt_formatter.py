"""
Prompt Formatter — constructs model-specific prompts for code generation.

Each model has a different expected input format. This module handles
the translation from a benchmark problem to a properly formatted prompt.
"""


def format_prompt_openai(problem: dict) -> list[dict]:
    """
    Format prompt for OpenAI chat completion API (GPT-3.5-turbo).

    Returns a list of message dicts for the chat API.
    """
    system_msg = (
        "You are an expert Python programmer. Complete the given Python function. "
        "Return ONLY the complete function definition (including the def line). "
        "Do not include any explanation, examples, or test code. "
        "Do not wrap the code in markdown code fences."
    )

    user_msg = (
        f"Complete the following Python function:\n\n"
        f"{problem['prompt']}\n"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def format_prompt_codellama(problem: dict) -> str:
    """
    Format prompt for CodeLlama-7B-Instruct (Together.ai API).

    Uses the [INST] ... [/INST] template that CodeLlama expects.
    """
    instruction = (
        f"Complete the following Python function. "
        f"Return ONLY the complete function definition including the def line. "
        f"Do not include any explanation or test code.\n\n"
        f"{problem['prompt']}"
    )

    return f"[INST] {instruction} [/INST]\n"


def format_prompt(problem: dict, model_key: str) -> dict:
    """
    Route to the correct prompt formatter based on model key.

    Args:
        problem: Problem dict from benchmark.json
        model_key: Key from config.MODELS ("gpt35" or "codellama")

    Returns:
        Dict with 'messages' (for OpenAI-style) or 'prompt' (for completion-style)
    """
    if model_key == "gpt35":
        return {"messages": format_prompt_openai(problem)}
    elif model_key == "codellama":
        return {"messages": format_prompt_codellama_chat(problem)}
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def format_prompt_codellama_chat(problem: dict) -> list[dict]:
    """
    Format for Together.ai's OpenAI-compatible chat API.

    Together.ai supports the chat completions format, so we use
    the same message structure as OpenAI but with CodeLlama-specific
    system instructions.
    """
    system_msg = (
        "You are an expert Python programmer. Complete the given function. "
        "Return ONLY the complete function definition with the def line. "
        "No explanation, no tests, no markdown fences."
    )

    user_msg = f"Complete this Python function:\n\n{problem['prompt']}"

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
