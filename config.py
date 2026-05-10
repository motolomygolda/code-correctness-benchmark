"""
Project configuration — API keys, model parameters, paths.
Set API keys via environment variables before running.
"""

import os

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# ── Model Settings ────────────────────────────────────────────────────────────
MODELS = {
    "gpt35": {
        "name": "GPT-3.5-Turbo",
        "api": "openai",
        "model_id": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1024,
        "samples_per_problem": 3,
    },
    "codellama": {
        "name": "CodeLlama-7B-Instruct",
        "api": "together",
        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
        "temperature": 0.7,
        "max_tokens": 1024,
        "samples_per_problem": 3,
    },
}

# ── Execution Settings ────────────────────────────────────────────────────────
EXECUTION_TIMEOUT = 5          # seconds per test execution
MAX_RETRIES = 3                # API call retries
RETRY_BACKOFF_BASE = 2         # exponential backoff base

# ── Paths ─────────────────────────────────────────────────────────────────────
BENCHMARK_PATH = "benchmark/benchmark.json"
RAW_RESPONSES_DIR = "raw_responses"
RESULTS_DIR = "results"
CHARTS_DIR = "charts"

# ── Analysis Settings ─────────────────────────────────────────────────────────
BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE_LEVEL = 0.95
MANUAL_VALIDATION_SAMPLE_RATE = 0.10
