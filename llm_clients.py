"""
TokenForge v6.0 — LLM Clients
Category-aware model fallback chain and ultra-concise prompt styling.
"""
import os
import re
import logging
from typing import Optional, Dict, Tuple, List
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category configs: (system_prompt, max_tokens)
# Highly optimized concise prompts. Models output exactly the answer, saving tokens.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "Be concise. English only. No preamble."

CATEGORY_CONFIG: Dict[str, Tuple[str, int]] = {
    "factual": (
        "Answer in one short, accurate sentence or a single fact. No extra commentary.",
        128,
    ),
    "math": (
        "Give ONLY the final numeric or algebraic answer (with units if relevant). No calculation steps.",
        64,
    ),
    "sentiment": (
        "Reply with exactly one label: Positive, Negative, Neutral, or Mixed. Add a reason only if the prompt asks for it.",
        32,
    ),
    "summarization": (
        "Summarize the text, strictly obeying the requested format and length constraints. No commentary.",
        160,
    ),
    "ner": (
        "List extracted entities grouped by type (PERSON, ORGANIZATION, LOCATION, DATE). No commentary.",
        128,
    ),
    "code_debug": (
        "Return only the corrected code block. Do not write explanations.",
        384,
    ),
    "logical": (
        "Give only the final answer or solution. No step-by-step reasoning.",
        96,
    ),
    "code_gen": (
        "Return only the requested code block. Do not write explanations.",
        384,
    ),
}

_NON_CHAT_HINTS = (
    "embed", "rerank", "whisper", "audio", "tts", "image", "vision",
    "moderation", "guard", "clip", "diffusion", "flux",
)

# Track models that don't support reasoning_effort
_no_effort_param: set = set()


def get_fallback_chain(category: str, allowed_models_str: str) -> List[str]:
    """
    Returns the exact ALLOWED_MODELS entries sorted in category-optimal accuracy fallback order.
    """
    import json
    raw = (allowed_models_str or "").strip()
    models = []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            models = [str(m).strip() for m in parsed if str(m).strip()]
        except Exception:
            pass
    if not models:
        models = [m.strip() for m in raw.split(",") if m.strip()]

    if not models:
        return []

    usable = [m for m in models if not any(b in m.lower() for b in _NON_CHAT_HINTS)]
    if not usable:
        usable = list(models)

    cat = (category or "").lower()

    # Preference lists by category domain
    if cat in ("code_debug", "code_gen"):
        preferred = ["kimi-k2p7-code", "minimax-m3", "deepseek", "glm", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]
    elif cat in ("logical", "math", "factual"):
        preferred = ["minimax-m3", "deepseek", "glm-5", "glm", "kimi", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]
    else:
        # NLP: sentiment, ner, summarization
        preferred = ["minimax-m3", "gemma-4-31b-it", "gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4", "glm", "deepseek", "kimi"]

    chain = []
    # 1. Add matching preferred models
    for pref in preferred:
        for m in usable:
            m_lower = m.lower()
            if pref in m_lower or m_lower.endswith(pref):
                if m not in chain:
                    chain.append(m)

    # 2. Append remaining usable models
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
    Calls Fireworks AI with category-specific compact instruction prompting.
    """
    if not api_key or not base_url or not allowed_models:
        return None

    fallback_chain = get_fallback_chain(category, allowed_models)
    if not fallback_chain:
        return None

    instruction, max_tokens = CATEGORY_CONFIG.get(
        category,
        ("Answer accurately and concisely.", 128)
    )

    # Clean redundant spaces and tabs to minimize input tokens
    lines = [line.strip() for line in prompt.splitlines()]
    prompt_clean = "\n".join(lines).strip()
    prompt_clean = re.sub(r'[ \t]+', ' ', prompt_clean)

    user_content = f"{instruction}\n\nTask: {prompt_clean}"
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=25.0, max_retries=3)

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

    # Clean redundant spaces and tabs to minimize input tokens
    lines = [line.strip() for line in prompt.splitlines()]
    prompt_clean = "\n".join(lines).strip()
    prompt_clean = re.sub(r'[ \t]+', ' ', prompt_clean)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=25.0, max_retries=2)
    return _call_model(client, strong, f"{repair_instruction}\n\nTask: {prompt_clean}", SYSTEM_PROMPT, max_tokens=256)


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
                logger.error("Fireworks API call failed for model %s (retry): %s", model, e2)
                return None
        else:
            logger.error("Fireworks API call failed for model %s: %s", model, e)
            return None

    try:
        content = response.choices[0].message.content
        if content and content.strip():
            return content.strip()
    except (IndexError, AttributeError):
        pass

    return None
