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
        "You are a factual knowledge expert. Give a clear, direct, accurate answer. "
        "No markdown formatting. No bold, no headers, no bullet points. "
        "Be thorough but concise (under 150 words).",
        300,
        "strong",
    ),
    "math": (
        "You are a precise mathematical solver. Solve the problem step-by-step. "
        "Show your work clearly. End your response with the final answer on its own line "
        "in the format: Answer: <value>. No markdown formatting.",
        450,
        "strong",
    ),
    "sentiment": (
        "You are a sentiment analyst. Classify the overall sentiment of the given text as exactly one of: "
        "Positive, Negative, Neutral, or Mixed. Then provide a brief one-sentence justification. "
        "No markdown formatting.",
        150,
        "strong",
    ),
    "summarization": (
        "You are a professional summarizer. Output ONLY the requested summary. "
        "Strictly obey any format or length constraints specified. "
        "No commentary, no preamble, no markdown formatting.",
        250,
        "strong",
    ),
    "ner": (
        "You are a named entity extraction specialist. Extract ALL named entities from the text. "
        "For each entity, output it as 'TYPE: name' on its own line. "
        "Valid types: PERSON, ORGANIZATION, LOCATION, DATE. "
        "If an entity could be multiple types, choose the most specific. "
        "No markdown formatting.",
        260,
        "strong",
    ),
    "code_debug": (
        "You are an expert code debugger. First, state the bug in one clear sentence. "
        "Then provide the complete corrected code. "
        "Do not use markdown bold or headers. You may use a code fence for the code block.",
        520,
        "code",
    ),
    "logical": (
        "You are a logic puzzle solver. Work through each constraint carefully in numbered steps. "
        "Check every condition. State your final answer clearly at the end on its own line "
        "in the format: Answer: <value>. No markdown formatting.",
        450,
        "strong",
    ),
    "code_gen": (
        "You are an expert code generator. Write correct, self-contained, well-structured code "
        "that meets the exact specification. Include brief inline comments. "
        "Output only the code. You may use a code fence.",
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

    # Compress the prompt to save input tokens
    compressed = _compress_prompt(prompt)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=3)

    for model in fallback_chain:
        text = _call_model(client, model, compressed, system_prompt, max_tokens)
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
        max_tokens=400,
    )


def _call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    system: str,
    max_tokens: int,
) -> Optional[str]:
    """Make one API call with reasoning_effort support and graceful fallback."""
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
        # If reasoning_effort is not supported, retry without it
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
                logger.error("Fireworks API call failed for model %s (no-effort retry): %s", model, e2)
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
