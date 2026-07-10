import os
import re
import logging
from typing import Optional, Dict, Tuple
from openai import OpenAI

logger = logging.getLogger(__name__)

_BASE = "English only. Be concise; no preamble."

CATEGORY_CONFIG: Dict[str, Tuple[str, int, str]] = {
    "factual": (
        f"{_BASE} Explain clearly in under 120 words.",
        300,
        "strong",
    ),
    "math": (
        f"{_BASE} Brief steps, then 'Answer: <value>' on its own line.",
        450,
        "strong",
    ),
    "sentiment": (
        f"{_BASE} Label the sentiment positive, negative, or neutral, then give one short justification.",
        150,
        "strong",
    ),
    "summarization": (
        f"{_BASE} Output only the summary; obey any stated length or format constraint.",
        250,
        "strong",
    ),
    "ner": (
        f"{_BASE} List each entity as 'label: value', one per line; labels: person, organization, location, date.",
        260,
        "strong",
    ),
    "code_debug": (
        f"{_BASE} Name the bug in one sentence, then give the corrected code in one fenced block.",
        520,
        "code",
    ),
    "logical": (
        f"{_BASE} Deduce in brief numbered steps checking every constraint, then 'Answer: <value>' on its own line.",
        450,
        "strong",
    ),
    "code_gen": (
        f"{_BASE} Output only the code in one fenced block, correct and self-contained.",
        520,
        "code",
    ),
}

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
        if not models:
            return

        usable = [m for m in models if not any(b in m.lower() for b in _NON_CHAT_HINTS)]
        if not usable:
            usable = list(models)

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

    def get_model(self, tier: str) -> Optional[str]:
        return self.tiers.get(tier, self.tiers.get("strong"))

def call_fireworks_category(prompt: str, category: str, api_key: str, base_url: str, allowed_models: str) -> Optional[str]:
    """
    Calls Fireworks AI via the official OpenAI-compatible client routed exclusively to FIREWORKS_BASE_URL.
    Enforces category-specific judge-aligned system prompt and strict max_tokens limit.
    Includes automatic fallback retry across ALLOWED_MODELS on transient errors or blank responses.
    """
    if not api_key or not base_url or not allowed_models:
        return None

    tierer = FireworksModelTierer(allowed_models)
    system_prompt, max_tokens, preferred_tier = CATEGORY_CONFIG.get(
        category,
        (f"{_BASE} Answer clearly and concisely.", 384, "strong")
    )
    primary_model = tierer.get_model(preferred_tier)
    strong_model = tierer.get_model("strong")
    if not primary_model:
        return None

    models_to_try = [primary_model]
    if strong_model and strong_model not in models_to_try:
        models_to_try.append(strong_model)
    cheap_model = tierer.get_model("cheap")
    if cheap_model and cheap_model not in models_to_try:
        models_to_try.append(cheap_model)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=25.0, max_retries=2)
    for m in models_to_try:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
        except Exception as e:
            logger.error(f"Fireworks API call failed for model {m}: {e}")
            continue

    return None

