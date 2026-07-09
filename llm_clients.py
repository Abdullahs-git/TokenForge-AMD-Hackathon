import os
import logging
from typing import Optional, Dict, Tuple
from openai import OpenAI

logger = logging.getLogger(__name__)

# Category configs: (system_prompt, max_tokens, preferred_tier)
# ACCURACY-FIRST STRATEGY: Prioritize answer quality over token savings.
# The accuracy gate (80% = 16/19 tasks) is binary — below it, token efficiency is irrelevant.
CATEGORY_CONFIG: Dict[str, Tuple[str, int, str]] = {
    "factual": (
        "Answer in English. Provide a clear, direct, factual explanation. Be accurate and thorough but concise (under 150 words). Do not include unnecessary preamble.",
        300,
        "strong",
    ),
    "math": (
        "Answer in English. Solve the problem step-by-step showing your work clearly. End your response with the final answer on a new line in the format: Answer: <value>",
        512,
        "strong",
    ),
    "sentiment": (
        "Answer in English. Classify the overall sentiment of the text as Positive, Negative, Neutral, or Mixed. Provide a brief justification explaining which aspects of the text led to your classification.",
        256,
        "strong",
    ),
    "summarization": (
        "Answer in English. Provide only the requested summary, strictly adhering to any format or length constraints specified in the prompt. Do not add commentary.",
        300,
        "strong",
    ),
    "ner": (
        "Answer in English. Extract all named entities from the text. For each entity, provide its name and type (Person, Organization, Location, Date). Format as a clear list.",
        256,
        "strong",
    ),
    "code_debug": (
        "Answer in English. First identify the bug and explain what is wrong. Then provide the complete corrected code implementation.",
        512,
        "code",
    ),
    "logical": (
        "Answer in English. Solve the logic puzzle step by step. Consider each constraint carefully and show your reasoning. State your final answer clearly at the end.",
        512,
        "strong",
    ),
    "code_gen": (
        "Answer in English. Write a correct, well-structured, clean function that meets the exact specification. Include brief comments explaining the logic.",
        512,
        "code",
    ),
}

class FireworksModelTierer:
    """Dynamically parses ALLOWED_MODELS and maps them into cheap, code, and strong tiers."""
    def __init__(self, allowed_models_str: str):
        models = [m.strip() for m in allowed_models_str.split(",") if m.strip()]
        self.tiers: Dict[str, str] = {}
        if not models:
            return

        cheap_model = None
        code_model = None
        strong_model = None

        for m in models:
            m_lower = m.lower()
            if "gemma" in m_lower:
                cheap_model = m
            elif any(k in m_lower for k in ["8b", "small", "mini", "mixtral"]):
                if not cheap_model:
                    cheap_model = m
            elif any(k in m_lower for k in ["coder", "code", "qwen"]):
                if not code_model:
                    code_model = m
            elif any(k in m_lower for k in ["70b", "405b", "large", "pro"]):
                if not strong_model:
                    strong_model = m

        # Fallbacks if specific keywords aren't matched
        self.tiers["cheap"] = cheap_model or models[-1]
        self.tiers["strong"] = strong_model or models[0]
        self.tiers["code"] = code_model or self.tiers["strong"]

    def get_model(self, tier: str) -> Optional[str]:
        return self.tiers.get(tier, self.tiers.get("strong"))

def call_fireworks_category(prompt: str, category: str, api_key: str, base_url: str, allowed_models: str) -> Optional[str]:
    """
    Calls Fireworks AI via the official OpenAI-compatible client routed exclusively to FIREWORKS_BASE_URL.
    Enforces category-specific system prompt and strict max_tokens limit.
    Includes automatic fallback retry across ALLOWED_MODELS on transient errors.
    """
    if not api_key or not base_url or not allowed_models:
        return None

    tierer = FireworksModelTierer(allowed_models)
    system_prompt, max_tokens, preferred_tier = CATEGORY_CONFIG.get(
        category,
        ("Answer in English clearly and concisely.", 384, "strong")
    )
    model = tierer.get_model(preferred_tier)
    if not model:
        return None

    models_to_try = [model]
    cheap_model = tierer.get_model("cheap")
    if cheap_model and cheap_model not in models_to_try:
        models_to_try.append(cheap_model)

    client = OpenAI(api_key=api_key, base_url=base_url)
    for m in models_to_try:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            logger.error(f"Fireworks API call failed for model {m}: {e}")
            continue

    return None
