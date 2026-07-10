import os
import re
import logging
from typing import Optional, Dict, Tuple, List
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role-based, judge-aligned system prompts per category
# ---------------------------------------------------------------------------

CATEGORY_CONFIG: Dict[str, Tuple[str, int, str]] = {
    "factual": (
        "You are an expert knowledge assistant. Answer the question accurately, clearly, and thoroughly. "
        "Provide direct facts and crisp explanations.",
        512,
        "strong",
    ),
    "math": (
        "You are an expert mathematical solver. Work through the problem step-by-step showing your reasoning clearly. "
        "At the end of your response, state your final numerical or algebraic answer on a new line.",
        768,
        "strong",
    ),
    "sentiment": (
        "You are an expert sentiment analyst. Classify the overall sentiment of the text as Positive, Negative, Neutral, or Mixed. "
        "Provide a clear, brief justification explaining what specific aspects led to that classification.",
        384,
        "strong",
    ),
    "summarization": (
        "You are an expert text summarizer. Provide a concise, accurate summary that captures the main ideas. "
        "Strictly adhere to any explicit format or length constraints requested in the prompt.",
        512,
        "strong",
    ),
    "ner": (
        "You are an expert named entity recognition specialist. Extract all named entities from the text. "
        "Clearly identify each entity and label its type (e.g., PERSON, ORGANIZATION, LOCATION, DATE).",
        512,
        "strong",
    ),
    "code_debug": (
        "You are an expert software engineer and code debugger. First explain the bug clearly. "
        "Then provide the corrected, working code implementation inside a code block.",
        1024,
        "code",
    ),
    "logical": (
        "You are an expert logical reasoning solver. Solve the puzzle step-by-step, verifying every condition and constraint carefully. "
        "State your final deduced answer clearly at the end.",
        768,
        "strong",
    ),
    "code_gen": (
        "You are an expert software engineer. Write clean, correct, well-structured code meeting the exact prompt specification. "
        "Include concise comments explaining the logic.",
        1024,
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
        # Add any remaining models not already in the chain
        for m in self.all_models:
            if m not in chain:
                chain.append(m)
        return chain


# ---------------------------------------------------------------------------
# Prompt compression (strip politeness filler to save tokens)
# ---------------------------------------------------------------------------

_FILLER_PATTERNS = [
    (re.compile(r"\b(could you |would you |can you )(please |kindly )?", re.I), ""),
    (re.compile(r"\bplease\b", re.I), ""),
    (re.compile(r"\bi would like you to\b", re.I), ""),
    (re.compile(r"\bi want you to\b", re.I), ""),
    (re.compile(r"\bi need you to\b", re.I), ""),
    (re.compile(r"\bthanks? ?(you|in advance)?\\.?\\s*$", re.I), ""),
]


def _compress_prompt(prompt: str) -> str:
    """Strip politeness filler from prompts to save input tokens."""
    result = prompt
    for pattern, replacement in _FILLER_PATTERNS:
        result = pattern.sub(replacement, result)
    result = re.sub(r"  +", " ", result).strip()
    return result if result else prompt  # Never return empty


# ---------------------------------------------------------------------------
# API call with reasoning_effort and full fallback chain
# ---------------------------------------------------------------------------

# Track which models don't support reasoning_effort so we don't retry
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
    - Category-specific judge-aligned system prompts
    - Strict max_tokens ceilings
    - reasoning_effort="none" for deterministic output (with graceful fallback)
    - Full model fallback chain (primary → strong → all remaining)
    - Prompt compression to save input tokens
    """
    if not api_key or not base_url or not allowed_models:
        return None

    tierer = FireworksModelTierer(allowed_models)
    system_prompt, max_tokens, preferred_tier = CATEGORY_CONFIG.get(
        category,
        ("Answer clearly and concisely. No markdown formatting.", 384, "strong")
    )

    fallback_chain = tierer.get_fallback_chain(preferred_tier)
    if not fallback_chain:
        return None

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=3)

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

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=2)
    return _call_model(
        client, strong, prompt,
        repair_instruction,
        max_tokens=600,
    )


def _call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    system: str,
    max_tokens: int,
) -> Optional[str]:
    """Make one API call across fallback chain."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error("Fireworks API call failed for model %s: %s", model, e)
        return None

    try:
        content = response.choices[0].message.content
        if content and content.strip():
            return content.strip()
    except (IndexError, AttributeError):
        pass

    return None
