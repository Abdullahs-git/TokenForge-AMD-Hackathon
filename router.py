"""
TokenForge — AMD Judge-Aligned Hybrid Routing Engine
Aligned with the Official AMD Hackathon Judging FAQ & Public Validation Examples:
- Strictly obeys sentence/word/bullet counts for Summarization
- Acknowledges both positive & negative aspects for Sentiment Classification
- Accurately captures multi-part answers for Math & Factual Knowledge
- Exhaustively extracts entities for NER
"""

import re
import logging
from typing import Optional, List, Dict, Tuple, Any

logger = logging.getLogger("tokenforge.router")

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)

_FLUFF_PREFIXES = [
    r"^here is (?:the )?[^:]+:\s*",
    r"^sure[!,.\s]+(?:here is [^:]+:\s*)?",
    r"^certainly[!,.\s]+(?:here is [^:]+:\s*)?",
    r"^of course[!,.\s]+(?:here is [^:]+:\s*)?",
]

_FLUFF_SUFFIXES = [
    r"\s+hope this helps[!.]*$",
    r"\s+let me know if you need anything else[!.]*$",
    r"\s+feel free to ask[!.]*$",
]

# ---------------------------------------------------------------------------
# 1. Exact AMD Judge-Aligned System Prompts & Token Ceilings
# ---------------------------------------------------------------------------
_BASE = "English only. Be concise; no preamble."

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "factual": {
        "system_prompt": f"{_BASE} Explain clearly and accurately, directly addressing all parts of the prompt.",
        "max_tokens": 300,
        "tier": "strong",
    },
    "math": {
        "system_prompt": f"{_BASE} Provide clear step-by-step calculations and explicitly state all required answers accurately.",
        "max_tokens": 400,
        "tier": "strong",
    },
    "sentiment": {
        "system_prompt": f"{_BASE} State the sentiment label, then give a one-sentence justification acknowledging all key aspects (both positive and negative elements if present).",
        "max_tokens": 140,
        "tier": "cheap",
    },
    "summarization": {
        "system_prompt": f"{_BASE} Output only the summary. Strictly obey all sentence count, bullet point count, and word-limit constraints stated in the prompt.",
        "max_tokens": 240,
        "tier": "cheap",
    },
    "ner": {
        "system_prompt": f"{_BASE} Extract all named entities accurately and list each on its own line as 'LABEL: Entity Name'.",
        "max_tokens": 260,
        "tier": "cheap",
    },
    "code_debug": {
        "system_prompt": f"{_BASE} Name the bug concisely, then provide the corrected code in a single self-contained fenced block.",
        "max_tokens": 520,
        "tier": "code",
    },
    "logic": {
        "system_prompt": f"{_BASE} Deduce step-by-step verifying every constraint, then state the final answer clearly.",
        "max_tokens": 420,
        "tier": "strong",
    },
    "code_gen": {
        "system_prompt": f"{_BASE} Output only the complete, correct, self-contained code in a single fenced block.",
        "max_tokens": 520,
        "tier": "code",
    },
}

_BACKWARD_MAP = {
    "general": "factual",
    "short_fact": "factual",
    "classification": "sentiment",
    "code": "code_gen",
}

# ---------------------------------------------------------------------------
# 2. 8-Category Regex Classifier
# ---------------------------------------------------------------------------
_CLASSIFIER_PATTERNS: Dict[str, List[str]] = {
    "code_debug": [
        r"\bbug\b", r"\bdebug\b", r"\bfix (this|the|my|it)\b",
        r"what'?s wrong", r"why (does|is)n'?t (this|it|my)\b",
        r"error in (this|the|my)\b", r"traceback", r"stack ?trace",
        r"throws? an? (error|exception)", r"returns? \w+ instead",
        r"infinite loop", r"corrected (version|code)",
    ],
    "code_gen": [
        r"\b(write|create|implement|build|generate|produce|give me)\b.*"
        r"\b(function|method|class|program|script|routine)\b",
        r"```(python|js|javascript|java|cpp|c|go|rust|ts|typescript|sql)",
        r"def \w+\(", r"class \w+[:\(]",
    ],
    "math": [
        r"\b(calculate|compute|solve|equation|formula|percent(age)?|integral|derivative|algebra|arithmetic)\b",
        r"[\d]+\s*[\+\-\*\/\^]\s*[\d]+",
        r"\bhow many\b.*\b(units|cookies|dollars|cups|remain|cost)\b",
        r"\b(cost|price|total cost|recipe|stock|restock)\b",
    ],
    "sentiment": [
        r"\b(sentiment|classify.*sentiment|positive.*negative|review|tweet|customer review)\b",
        r"\bclassify the sentiment\b",
    ],
    "ner": [
        r"\b(named entit(y|ies)|extract.*entit(y|ies)|NER|PERSON|ORGANIZATION|LOCATION|DATE)\b",
        r"extract all named entities",
    ],
    "summarization": [
        r"\b(summarize|summarise|summary|tl;dr|in exactly \d+ (sentences|bullet points))\b",
        r"bullet points?, each no longer than",
    ],
    "logic": [
        r"\b(deduce|logical|puzzle|riddle|syllogism|statement.*true|infer|constraint)\b",
    ],
}


def classify_task(prompt: str) -> str:
    """Classify prompt into one of the 8 Judge-Aligned categories."""
    p_lower = prompt.lower()

    for category, patterns in _CLASSIFIER_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, p_lower):
                return category

    return "factual"


# ---------------------------------------------------------------------------
# 3. Dynamic Model Selection
# ---------------------------------------------------------------------------
def select_model(prompt: str, category: str, allowed_models: List[str]) -> str:
    """Select the optimal model from ALLOWED_MODELS based on category tier."""
    if not allowed_models:
        return "accounts/fireworks/models/llama-v3p1-70b-instruct"

    tier = TASK_CONFIG.get(category, TASK_CONFIG["factual"])["tier"]

    if tier == "code":
        priorities = ["kimi-k2.7-code", "qwen2.5-coder", "minimax-m3", "gemma-4-31b-it"]
    elif tier == "cheap":
        priorities = ["llama-v3p1-8b-instruct", "gemma-4-9b-it", "minimax-m3"]
    else:
        priorities = ["minimax-m3", "minimax", "llama-v3p1-70b-instruct", "gemma-4-31b-it"]

    for pref in priorities:
        for m in allowed_models:
            if pref in m.lower():
                return m

    return allowed_models[0]


# ---------------------------------------------------------------------------
# 4. Tier 0 Local Deterministic Solvers ($0 API cost, 0 tokens)
# ---------------------------------------------------------------------------
def solve_equation(prompt: str) -> Optional[str]:
    """Tier 0 solver for simple linear equations like '2*x + 5 = 15'."""
    match = re.search(
        r"(?:solve\s+(?:the\s+)?(?:equation\s+)?)(?:for\s+[a-z]\s*:\s*)?([0-9]+)\s*\*\s*([a-z])\s*([\+\-])\s*([0-9]+)\s*=\s*([0-9]+)",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return None

    try:
        a = int(match.group(1))
        var = match.group(2)
        op = match.group(3)
        b = int(match.group(4))
        c = int(match.group(5))

        if op == "+":
            rhs = c - b
        else:
            rhs = c + b

        if rhs % a == 0:
            val = rhs // a
            return f"Answer: {val}"
    except Exception:
        pass

    return None


def solve_arithmetic(prompt: str) -> Optional[str]:
    """Tier 0 solver for pure arithmetic expressions like 'Calculate 144 / 12'."""
    text = prompt.strip()
    match = re.match(
        r"^(?:what is|calculate|compute|evaluate)\s+([0-9\s\+\-\*\/\(\)\.]+)\??$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None

    expr = match.group(1).strip()
    if not re.match(r"^[0-9\s\+\-\*\/\(\)\.]+$", expr):
        return None

    try:
        val = eval(expr, {"__builtins__": {}}, {})
        if isinstance(val, (int, float)):
            if val == int(val):
                return f"Answer: {int(val)}"
            return f"Answer: {val:.4g}"
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# 5. Output Sanitization
# ---------------------------------------------------------------------------
def sanitize_output(raw_text: str) -> str:
    """Clean internal CoT reasoning traces and conversational fluff."""
    if not raw_text:
        return ""

    text = _THINK_RE.sub("", raw_text).strip()

    for pat in _FLUFF_PREFIXES:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

    for pat in _FLUFF_SUFFIXES:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

    return text


# ---------------------------------------------------------------------------
# 6. Pipeline Orchestration
# ---------------------------------------------------------------------------
def solve_prompt(prompt: str, api_key: str, base_url: str, allowed_models: List[str]) -> str:
    """Execute complete Judge-Aligned routing and evaluation pipeline."""
    if not prompt or not prompt.strip():
        return "Unable to determine answer."

    # --- Tier 0 Deterministic Solvers ---
    eq_res = solve_equation(prompt)
    if eq_res is not None:
        logger.info("Tier 0 Local Solver HIT: %s -> %s", prompt[:35], eq_res)
        return eq_res

    arith_res = solve_arithmetic(prompt)
    if arith_res is not None:
        logger.info("Tier 0 Local Solver HIT: %s -> %s", prompt[:35], arith_res)
        return arith_res

    # --- Tier 1 SOTA Cloud Model Execution ---
    category = classify_task(prompt)
    cfg = TASK_CONFIG.get(category, TASK_CONFIG["factual"])

    model = select_model(prompt, category, allowed_models)
    logger.info("Task Category: [%s] -> Routing to model: %s", category.upper(), model)

    if not api_key or not base_url or not allowed_models:
        logger.warning("API credentials missing during solve_prompt execution.")
        return "Unable to generate answer."

    from openai import OpenAI
    import time

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=1)

    messages = [
        {"role": "system", "content": cfg["system_prompt"]},
        {"role": "user", "content": prompt},
    ]

    delay = 1.0
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=cfg["max_tokens"],
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
