"""
API Caller — handles requests to OpenAI and Together.ai APIs with retry logic.

Supports both GPT-3.5-turbo (OpenAI) and CodeLlama-7B-Instruct (Together.ai).
Both use OpenAI-compatible chat completion endpoints.
"""

import json
import os
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class APICaller:
    """Manages API calls to model providers with retry and caching."""

    def __init__(self, config):
        self.config = config
        self._clients = {}

    def _get_openai_client(self):
        """Lazy-initialize OpenAI client."""
        if "openai" not in self._clients:
            try:
                from openai import OpenAI
                self._clients["openai"] = OpenAI(api_key=self.config.OPENAI_API_KEY)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._clients["openai"]

    def _get_together_client(self):
        """Lazy-initialize Together.ai client (uses OpenAI-compatible API)."""
        if "together" not in self._clients:
            try:
                from openai import OpenAI
                self._clients["together"] = OpenAI(
                    api_key=self.config.TOGETHER_API_KEY,
                    base_url="https://api.together.xyz/v1",
                )
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._clients["together"]

    def call_model(
        self,
        model_key: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Make a single API call with retry logic.

        Args:
            model_key: Key from config.MODELS
            messages: Chat messages list
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Raw text response from the model
        """
        model_cfg = self.config.MODELS[model_key]
        model_id = model_cfg["model_id"]
        api_type = model_cfg["api"]

        client = (
            self._get_openai_client() if api_type == "openai"
            else self._get_together_client()
        )

        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            except Exception as e:
                wait_time = self.config.RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.config.MAX_RETRIES}): "
                    f"{type(e).__name__}: {e}. Retrying in {wait_time}s..."
                )
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.config.MAX_RETRIES} attempts failed for {model_key}")
                    raise

    def generate_solutions(
        self,
        problem: dict,
        model_key: str,
        num_samples: int = 3,
        save_dir: str = None,
    ) -> list[str]:
        """
        Generate multiple independent solutions for a problem.

        Args:
            problem: Problem dict from benchmark
            model_key: Model identifier
            num_samples: Number of independent samples to generate
            save_dir: Directory to save raw responses (None = don't save)

        Returns:
            List of raw text responses
        """
        from src.prompt_formatter import format_prompt

        prompt_data = format_prompt(problem, model_key)
        messages = prompt_data["messages"]
        model_cfg = self.config.MODELS[model_key]

        responses = []
        for sample_idx in range(num_samples):
            # Check cache first
            if save_dir:
                cache_path = Path(save_dir) / model_key / f"{problem['problem_id']}_sample_{sample_idx}.json"
                if cache_path.exists():
                    with open(cache_path) as f:
                        cached = json.load(f)
                    responses.append(cached["raw_response"])
                    logger.info(f"  Cache hit: {cache_path.name}")
                    continue

            # Make API call
            logger.info(
                f"  Calling {model_cfg['name']} for {problem['problem_id']} "
                f"sample {sample_idx}..."
            )
            raw_response = self.call_model(
                model_key=model_key,
                messages=messages,
                temperature=model_cfg["temperature"],
                max_tokens=model_cfg["max_tokens"],
            )
            responses.append(raw_response)

            # Save raw response
            if save_dir:
                cache_path = Path(save_dir) / model_key / f"{problem['problem_id']}_sample_{sample_idx}.json"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump({
                        "problem_id": problem["problem_id"],
                        "model_key": model_key,
                        "sample_idx": sample_idx,
                        "raw_response": raw_response,
                        "messages": messages,
                        "temperature": model_cfg["temperature"],
                        "timestamp": time.time(),
                    }, f, indent=2)

            # Small delay between calls to avoid rate limits
            time.sleep(0.5)

        return responses


class MockAPICaller:
    """
    Mock API caller for testing the pipeline without real API calls.
    Returns the canonical solution with random perturbations.
    """

    def __init__(self, config, failure_rate=0.3):
        self.config = config
        self.failure_rate = failure_rate

    def generate_solutions(
        self,
        problem: dict,
        model_key: str,
        num_samples: int = 3,
        save_dir: str = None,
    ) -> list[str]:
        import random

        responses = []
        for _ in range(num_samples):
            if random.random() < self.failure_rate:
                # Generate a realistic failure
                failure_type = random.choice([
                    "syntax", "runtime", "algorithmic", "edge_case"
                ])
                response = self._generate_failure(problem, failure_type)
            else:
                response = problem["canonical_solution"]
            responses.append(response)
        return responses

    def _generate_failure(self, problem: dict, failure_type: str) -> str:
        """Generate a realistically failing solution."""
        import random

        canonical = problem["canonical_solution"]
        entry_point = problem["entry_point"]

        if failure_type == "syntax":
            # Common syntax errors
            errors = [
                # Missing colon
                lambda c: c.replace(":\n", "\n", 1),
                # Unmatched parenthesis
                lambda c: c.replace(")", "", 1),
                # Invalid indentation
                lambda c: c.replace("    ", "  ", 1),
                # Undefined variable
                lambda c: c + f"\n    undefined_var += 1\n",
            ]
            return random.choice(errors)(canonical)

        elif failure_type == "runtime":
            # Add code that causes runtime errors
            lines = canonical.split('\n')
            body_start = next(i for i, l in enumerate(lines) if l.strip() and not l.strip().startswith('def'))
            runtime_errors = [
                "    _ = 1 / 0  # ZeroDivisionError",
                "    _ = [][99]  # IndexError",
                "    _ = int('abc')  # ValueError",
            ]
            lines.insert(body_start, random.choice(runtime_errors))
            return '\n'.join(lines)

        elif failure_type == "algorithmic":
            # Subtle logic errors
            code = canonical
            replacements = [
                ("<=", "<"),
                (">=", ">"),
                ("<", "<="),
                ("+1", "+2"),
                ("-1", ""),
                ("max", "min"),
            ]
            old, new = random.choice(replacements)
            if old in code:
                # Replace first occurrence only
                code = code.replace(old, new, 1)
            return code

        else:  # edge_case
            # Return a solution that works for standard cases but fails on edge cases
            # Typically: missing empty input check
            code = canonical
            # Remove the first 'if not' check if present
            code = code.replace("if not ", "if False and not ", 1)
            return code
