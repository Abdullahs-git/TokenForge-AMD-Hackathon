"""
TokenForge v5.0 — LLM Clients
Fireworks AI proxy client with dynamic model tiering.
"""
import os
import re
import logging
from typing import Optional, Dict, Tuple, List
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category configs: (system_prompt, max_tokens, preferred_tier)
# Concise prompts reduce input token cost while keeping accuracy high.
# ---------------------------------------------------------------------------

_BASE = "English only. Be concise; no preamble."

CATEGORY_CONFIG: Dict[str, Tuple[str, int, str]] = {
    "factual": (
        f"{_BASE} Answer accurately and clearly in under 120 words.",
        300,
        "strong",
    ),
    "math": (
        f"{_BASE} Brief steps, then 'Answer: <value>' on its own line.",
        400,
        "strong",
    ),
    "sentiment": (
        f"{_BASE} Label sentiment as Positive, Negative, Neutral, or Mixed, then one short justification.",
        120,
        "cheap",
    ),
    "summarization": (
        f"{_BASE} Output only the summary; obey any stated length or format constraint.",
        220,
        "cheap",
    ),
    "ner": (
        f"{_BASE} List each entity as 'label: value', one per line; labels: PERSON, ORGANIZATION, LOCATION, DATE.",
        260,
        "cheap",
    ),
    "code_debug": (
        f"{_BASE} Name the bug in one sentence, then give the corrected code in a fenced block.",
        520,
        "code",
    ),
    "logical": (
        f"{_BASE} Deduce in brief numbered steps checking every constraint, then 'Answer: <value>' on its own line.",
        420,
        "strong",
    ),
    "code_gen": (
        f"{_BASE} Output only the code in one fenced block, correct and self-contained.",
        520,
        "code",
    ),
}


# ---------------------------------------------------------------------------
# Dynamic model tiering from ALLOWED_MODELS
# ---------------------------------------------------------------------------

_MOE_PAT = re.compile(r"(\d+)\s*x\s*(\d+)\s*b\b")
_ACTIVE_PAT = re.compile(r"\ba(\d+)b\b")
_DENSE_PAT = re.compile(r"(\d+)\s*b\b")
_CODE_MODEL_PAT = re.compile(r"\bcode|coder|-code\b")
_QUANT_PAT = re.compile(r"nvfp4|fp4|fp8|int8|int4|awq|gptq|gguf")
_NON_CHAT_HINTS = (
    "embed", "rerank", "whisper", "audio", "tts", "image", "vision",
    "moderation", "guard", "clip", "diffusion", "flux",
)


def _total_params(model_id: str) -> int:
    mid = model_id.lower()
    if "deepseek" in mid:
        return 999
    moe = _MOE_PAT.search(mid)
    if moe:
        return int(moe.group(1)) * int(moe.group(2))
    sizes = [int(m.group(1)) for m in _DENSE_PAT.finditer(mid)]
    return max(sizes) if sizes else 100


def _active_params(model_id: str) -> int:
    m = _ACTIVE_PAT.search(model_id.lower())
    return int(m.group(1)) if m else _total_params(model_id)


class FireworksModelTierer:
    """Dynamically parses ALLOWED_MODELS and maps them into cheap, code, and strong tiers."""

    def __init__(self, allowed_models_str: str):
        models = [m.strip() for m in allowed_models_str.split(",") if m.strip()]
        self.tiers: Dict[str, str] = {}
        self.all_models: List[str] = []
        if not models:
            return

        usable = [m for m in models if not any(b in m.lower() for b in _NON_CHAT_HINTS)]
        if not usable:
            usable = list(models)
        self.all_models = usable

        general = [m for m in usable if not _CODE_MODEL_PAT.search(m.lower())] or usable

        strong = max(
            general,
            key=lambda m: (_total_params(m), not bool(_QUANT_PAT.search(m.lower())))
        )
        code_models = [m for m in usable if _CODE_MODEL_PAT.search(m.lower())]
        code = max(code_models, key=_total_params) if code_models else strong
        cheap = min(
            usable,
            key=lambda m: (_active_params(m), not bool(_QUANT_PAT.search(m.lower())))
        )

        self.tiers["cheap"] = cheap
        self.tiers["strong"] = strong
        self.tiers["code"] = code
        logger.info("Model tiers → cheap=%s | strong=%s | code=%s", cheap, strong, code)

    def get_model(self, tier: str) -> Optional[str]:
        return self.tiers.get(tier, self.tiers.get("strong"))

    def get_fallback_chain(self, preferred_tier: str) -> List[str]:
        """Build an ordered fallback chain: preferred → strong → all remaining."""
        chain = []
        primary = self.get_model(preferred_tier)
        if primary:
            chain.append(primary)
        strong = self.get_model("strong")
        if strong and strong not in chain:
            chain.append(strong)
        for m in self.all_models:
            if m not in chain:
                chain.append(m)
        return chain


# ---------------------------------------------------------------------------
# API call with reasoning_effort and full fallback chain
# ---------------------------------------------------------------------------

# Track which models don't support reasoning_effort so we don't retry them
_no_effort_param: set = set()


def call_fireworks_category(
    prompt: str,
    category: str,
    api_key: str,
    base_url: str,
    allowed_models: str,
) -> Optional[str]:
    """
    Calls Fireworks AI via the official OpenAI-compatible client.

    Features:
    - Category-specific concise system prompts
    - Strict max_tokens ceilings
    - reasoning_effort="none" for deterministic fast output (with graceful fallback)
    - Full model fallback chain (primary -> strong -> all remaining)
    - temperature=0.0 for exact deterministic answers
    """
    if not api_key or not base_url or not allowed_models:
        return None

    tierer = FireworksModelTierer(allowed_models)
    system_prompt, max_tokens, preferred_tier = CATEGORY_CONFIG.get(
        category,
        (f"{_BASE} Answer clearly.", 300, "strong")
    )

    fallback_chain = tierer.get_fallback_chain(preferred_tier)
    if not fallback_chain:
        return None

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=25.0, max_retries=3)

    for model in fallback_chain:
        text = _call_model(client, model, prompt, system_prompt, max_tokens)
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
    Repair call: re-asks the question with a tighter instruction prompt.
    Used when the primary answer was blank or invalid.
    """
    if not api_key or not base_url or not allowed_models:
        return None

    tierer = FireworksModelTierer(allowed_models)
    strong = tierer.get_model("strong")
    if not strong:
        return None

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=25.0, max_retries=2)
    return _call_model(client, strong, prompt, repair_instruction, max_tokens=400)


def _call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    system: str,
    max_tokens: int,
) -> Optional[str]:
    """Make one API call with reasoning_effort='none' and graceful fallback."""
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
        # If reasoning_effort is not supported by this model, retry without it
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
