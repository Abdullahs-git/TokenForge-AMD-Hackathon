"""
TokenForge v7.0 — LLM Clients
Ultra-tight token budgets, minimal prompts, category-aware model selection.
"""
import os
import re
import logging
from typing import Optional, Dict, Tuple, List
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global system prompt — absolute minimum (4 tokens)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "Concise. No preamble."

# ---------------------------------------------------------------------------
# Category configs: (user_instruction, max_tokens)
# Drastically tighter budgets than v6.0
# ---------------------------------------------------------------------------
CATEGORY_CONFIG: Dict[str, Tuple[str, int]] = {
    "factual": ("Provide a concise, accurate factual answer.", 140),
    "math": ("Provide the correct numerical answer clearly.", 120),
    "sentiment": ("Classify sentiment accurately (Positive, Negative, Neutral, or Mixed) and briefly explain.", 60),
    "summarization": ("Provide an accurate, concise summary.", 120),
    "ner": ("Extract all named entities accurately.", 100),
    "code_debug": ("Provide the corrected, bug-free code.", 260),
    "logical": ("Solve the logic puzzle accurately and state the clear final answer.", 140),
    "code_gen": ("Write clean, correct working code.", 260),
}

_NON_CHAT_HINTS = (
    "embed", "rerank", "whisper", "audio", "tts", "image", "vision",
    "moderation", "guard", "clip", "diffusion", "flux",
)

# Track models that don't support reasoning_effort
_no_effort_param: set = set()


def get_fallback_chain(category: str, allowed_models_str: str) -> List[str]:
    """
    Returns ALLOWED_MODELS entries sorted for category-optimal token efficiency.
    For easy categories, prefer smaller/cheaper models first.
    """
    models = [m.strip() for m in allowed_models_str.split(",") if m.strip()]
    if not models:
        return []

    usable = [m for m in models if not any(b in m.lower() for b in _NON_CHAT_HINTS)]
    if not usable:
        usable = list(models)

    cat = (category or "").lower()

    # For simple tasks, prefer smallest model first (token-efficient)
    if cat in ("sentiment", "ner"):
        preferred = ["gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4", "gemma-4-31b-it", "minimax-m3", "deepseek", "glm", "kimi"]
    elif cat in ("code_debug", "code_gen"):
        preferred = ["kimi-k2p7-code", "deepseek", "minimax-m3", "glm", "gemma-4-31b-it"]
    elif cat in ("logical", "math"):
        preferred = ["minimax-m3", "deepseek", "glm", "kimi", "gemma-4-31b-it"]
    elif cat == "summarization":
        preferred = ["gemma-4-26b-a4b-it", "gemma-4-31b-it", "minimax-m3", "deepseek", "glm"]
    else:
        preferred = ["gemma-4-26b-a4b-it", "minimax-m3", "gemma-4-31b-it", "deepseek", "glm", "kimi"]

    chain = []
    for pref in preferred:
        for m in usable:
            if pref in m.lower() or m.lower().endswith(pref):
                if m not in chain:
                    chain.append(m)

    for m in usable:
        if m not in chain:
            chain.append(m)

    return chain


def call_fireworks_category(
    prompt: str,
    category: str,
    api_key: str,
    base_url: str,
    allowed_models: str,
) -> Optional[str]:
    """
    Calls Fireworks AI with ultra-minimal prompting and tight token budgets.
    """
    if not api_key or not base_url or not allowed_models:
        return None

    fallback_chain = get_fallback_chain(category, allowed_models)
    if not fallback_chain:
        return None

    instruction, max_tokens = CATEGORY_CONFIG.get(
        category,
        ("Answer concisely.", 80)
    )

    # Ultra-compact user message
    user_content = f"{instruction}\n\n{prompt}"
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=25.0, max_retries=2)

    for model in fallback_chain:
        text = _call_model(client, model, user_content, SYSTEM_PROMPT, max_tokens)
        if text:
            return text

    return None


def call_repair(
    prompt: str,
    repair_instruction: str,
    api_key: str,
    base_url: str,
    allowed_models: str,
) -> Optional[str]:
    """
    Repair fallback call when primary output is empty or malformed.
    """
    if not api_key or not base_url or not allowed_models:
        return None

    fallback_chain = get_fallback_chain("factual", allowed_models)
    if not fallback_chain:
        return None
    strong = fallback_chain[0]

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=25.0, max_retries=2)
    return _call_model(client, strong, f"{repair_instruction}\n\n{prompt}", SYSTEM_PROMPT, max_tokens=128)


def _call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    system: str,
    max_tokens: int,
) -> Optional[str]:
    """Make one API call with reasoning_effort='none' support."""
    global _no_effort_param

    kwargs = {}
    if model not in _no_effort_param:
        kwargs["reasoning_effort"] = "none"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            **kwargs,
        )
    except Exception as e:
        if kwargs and "invalid_request_error" in str(e).lower():
            _no_effort_param.add(model)
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
            except Exception as e2:
                logger.error("Fireworks API retry failed for %s: %s", model, e2)
                return None
        else:
            logger.error("Fireworks API call failed for %s: %s", model, e)
            return None

    try:
        content = response.choices[0].message.content
        if content and content.strip():
            return content.strip()
    except (IndexError, AttributeError):
        pass

    return None
