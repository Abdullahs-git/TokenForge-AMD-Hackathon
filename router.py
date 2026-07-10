"""
TokenForge v8.0 — RTQ-Hybrid Query Router
Dynamically routes queries to local zero-token solvers or quality-maximized cloud models.
"""

import os
import re
import time
import logging
from typing import Optional, List
from openai import OpenAI
import local_solvers

logger = logging.getLogger(__name__)

# Quality-maximized system prompt for 100% accuracy and zero verbosity
SYSTEM_PROMPT = (
    "You are a highly accurate AI assistant. Make no mistakes and attain 100% accuracy "
    "for each question. Output only the direct, exact, correct answer without introductory "
    "filler, preambles, or meta-commentary."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_HINTS_RE = re.compile(
    r"\b(def |class |return |import |bug|fix|error|traceback|function|code|algorithm)\b|```",
    re.IGNORECASE
)


def select_best_model(prompt: str, allowed_models: List[str]) -> str:
    """
    Selects the optimal model from ALLOWED_MODELS based on task characteristics.
    Prioritizes models with superior instruction-following and zero-fluff formatting.
    """
    if not allowed_models:
        return "accounts/fireworks/models/llama-v3p1-8b-instruct"

    is_code_task = bool(_CODE_HINTS_RE.search(prompt))

    # Priority order for coding tasks
    if is_code_task:
        code_priorities = ["kimi-k2.7-code", "kimi-k2p7-code", "qwen2.5-coder", "minimax-m3", "gemma-4-31b-it"]
        for pref in code_priorities:
            for m in allowed_models:
                if pref in m.lower():
                    return m

    # Priority order for general reasoning / factual / NLP tasks
    general_priorities = ["minimax-m3", "minimax", "kimi-k2.6", "gemma-4-31b-it", "llama-v3p1-70b-instruct"]
    for pref in general_priorities:
        for m in allowed_models:
            if pref in m.lower():
                return m

    return allowed_models[0]


def sanitize_output(raw_text: str) -> str:
    """Strip internal chain-of-thought traces and extra whitespace."""
    if not raw_text:
        return ""
    text = _THINK_RE.sub("", raw_text).strip()
    return text


def solve_prompt(prompt: str, api_key: str, base_url: str, allowed_models: List[str]) -> str:
    """
    Full TokenForge v8.0 hybrid routing pipeline:
    1. Check Tier 0 Local Solver ($0 tokens)
    2. Select optimal cloud model from ALLOWED_MODELS
    3. Execute API call with strict accuracy system prompt & retry resilience
    4. Sanitize output
    """
    if not prompt or not prompt.strip():
        return "Unable to determine answer."

    # --- Tier 0: Zero-Token Local Arithmetic Solver ---
    local_ans = local_solvers.solve_math_expression(prompt)
    if local_ans is not None:
        logger.info("Tier 0 Local Solver HIT (0 API tokens): %s -> %s", prompt[:30], local_ans)
        return local_ans

    # --- Tier 1: Quality-Maximized Cloud Model Routing ---
    if not api_key or not base_url or not allowed_models:
        return "Unable to generate answer."

    model = select_best_model(prompt, allowed_models)
    logger.info("Routing to model: %s", model)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=1)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    delay = 1.0
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=800
            )
            raw_text = response.choices[0].message.content or ""
            cleaned = sanitize_output(raw_text)
            if cleaned:
                return cleaned
        except Exception as e:
            logger.warning("API attempt %d failed for model %s: %s", attempt + 1, model, e)
            if attempt < 2:
                time.sleep(delay)
                delay *= 2

    return "Unable to generate answer."
