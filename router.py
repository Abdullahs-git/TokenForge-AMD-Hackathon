"""
TokenForge — 8-Category Judge-Aligned Hybrid Routing Engine
Combines:
1. Exact 100% accuracy strategy from Judge-Aligned Reference (8 categories, exact judge prompts)
2. Ultra-Lean Token Optimization (Tier 0 local solvers + tight token ceilings 1000-1500 target)
3. Dynamic Model Tiering (cheap / strong / code models)
4. Fallback safety net & output sanitization
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
# 1. Exact Judge-Aligned 8-Category System Prompts & Token Ceilings
# ---------------------------------------------------------------------------
_BASE = "English only. Be concise; no preamble."

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "factual": {
        "system_prompt": f"{_BASE} Explain clearly in under 120 words.",
        "max_tokens": 160,
        "tier": "strong",
    },
    "math": {
        "system_prompt": f"{_BASE} Brief steps, then 'Answer: <value>' on its own line.",
        "max_tokens": 160,
        "tier": "strong",
    },
    "sentiment": {
        "system_prompt": f"{_BASE} Label the sentiment positive, negative, or neutral, then give one short justification.",
        "max_tokens": 64,
        "tier": "cheap",
    },
    "summarization": {
        "system_prompt": f"{_BASE} Output only the summary; obey any stated length or format constraint.",
        "max_tokens": 120,
        "tier": "cheap",
    },
    "ner": {
        "system_prompt": f"{_BASE} List each entity as 'label: value', one per line; labels: person, organization, location, date.",
        "max_tokens": 80,
        "tier": "cheap",
    },
    "code_debug": {
        "system_prompt": f"{_BASE} Name the bug in one sentence, then give the corrected code in one fenced block.",
        "max_tokens": 350,
        "tier": "code",
    },
    "logic": {
        "system_prompt": f"{_BASE} Deduce in brief numbered steps checking every constraint, then 'Answer: <value>' on its own line.",
        "max_tokens": 180,
        "tier": "strong",
    },
    "code_gen": {
        "system_prompt": f"{_BASE} Output only the code in one fenced block, correct and self-contained.",
        "max_tokens": 350,
        "tier": "code",
    },
}

# Backward compatibility mapping for general/classification/short_fact
_BACKWARD_MAP = {
    "general": "factual",
    "short_fact": "factual",
    "classification": "sentiment",
    "code": "code_gen",
}

# ---------------------------------------------------------------------------
# 2. Exact Judge-Aligned 8-Category Regex Classifier
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
        r"\bfunction (that|to)\b", r"\bcode that\b",
        r"\bwrite (a|an|some) code\b", r"\bimplement (a|an|the)\b",
    ],
    "sentiment": [
        r"\bsentiment\b", r"positive or negative", r"positive, negative",
        r"classify the (tone|emotion|sentiment|mood)",
        r"(emotional )?tone of (this|the|that)",
        r"\b(positive|negative|neutral)\b.*\breview\b",
        r"how (positive|negative) ", r"is this (review|tweet|comment)\b",
    ],
    "ner": [
        r"named entit", r"\bner\b",
        r"extract (all )?(the )?(entit|name|person|people|organi|location|date)",
        r"(list|identify|find|pull out) (all )?(the )?"
        r"(people|persons?|organi[sz]ations?|locations?|dates?|entit)",
        r"(person|organization|location|date)\s*[:=]",
    ],
    "summarization": [
        r"summari[sz]e", r"\bsummary\b", r"\btl;?dr\b", r"\bcondense\b",
        r"\bshorten\b", r"in (one|a single|two|three|\d+) (sentences?|words?|lines?)",
        r"main (idea|point|takeaway)", r"\bthe gist\b", r"key points",
        r"boil .* down",
    ],
    "logic": [
        r"\bpuzzle\b", r"who (is|owns|sits|lives|has|drinks|likes)\b",
        r"if and only if", r"exactly one", r"at least one",
        r"the following (clues|facts|statements|conditions)",
        r"each (person|house|box|day|one) .*(different|exactly|only|one)",
        r"\bdeduc(e|tive|tion)\b", r"logically (follows?|true)",
        r"(definitely|necessarily) (true|follows)",
        r"knights? and knaves", r"truth[- ]?teller", r"\bliar\b",
    ],
    "math": [
        r"\bcalculate\b", r"\bcompute\b", r"how (much|many)\b", r"percent",
        r"\d+\s*%", r"\bsum of\b", r"\baverage\b", r"solve for\b",
        r"\d+\s*[+\-*/x×÷]\s*\d+", r"total (cost|price|amount|distance)",
        r"\b(interest|discount|ratio|profit)\b",
        r"find the (largest|smallest|value|angle|area|sum|total|average)",
        r"what is \d",
    ],
    "factual": [
        r"what (is|are|was|were)\b", r"who (is|was|were)\b",
        r"when (did|was|is)\b", r"where (is|was|are)\b",
        r"why (is|do|does|are)\b", r"how (do|does|can)\b",
        r"\bexplain\b", r"\bdefine\b", r"\bdescribe\b", r"what does .* mean",
    ],
}

_PRIORITY_ORDER = [
    "code_debug", "code_gen", "sentiment", "ner",
    "summarization", "logic", "math", "factual",
]

_COMPILED_PATTERNS = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in _CLASSIFIER_PATTERNS.items()
}

_CODE_FENCE = re.compile(r"```")
_CODE_HINT = re.compile(
    r"\b(def |class |return |import |#include|public |void |printf|"
    r"console\.log|System\.out)|=>|;\s*$",
    re.MULTILINE,
)


def classify_task(prompt: str) -> str:
    """Classify the prompt into one of the 8 judge-aligned categories."""
    text = prompt or ""
    for cat in _PRIORITY_ORDER:
        if any(rx.search(text) for rx in _COMPILED_PATTERNS[cat]):
            return cat
    if _CODE_FENCE.search(text) or _CODE_HINT.search(text):
        return "code_debug"
    return "factual"


detect_category = classify_task


# ---------------------------------------------------------------------------
# 3. Dynamic Model Tiering (cheap / strong / code)
# ---------------------------------------------------------------------------
_MOE_PAT = re.compile(r"(\d+)\s*x\s*(\d+)\s*b\b")
_DENSE_PAT = re.compile(r"(\d+)\s*b\b")
_CODE_MODEL_PAT = re.compile(r"\bcode|coder|-code\b")


def _total_params(model_id: str) -> int:
    mid = model_id.lower()
    if "deepseek" in mid:
        return 999
    moe = _MOE_PAT.search(mid)
    if moe:
        return int(moe.group(1)) * int(moe.group(2))
    sizes = [int(m.group(1)) for m in _DENSE_PAT.finditer(mid)]
    return max(sizes) if sizes else 100


def resolve_model_tiers(allowed_models: List[str]) -> Dict[str, str]:
    if not allowed_models:
        default = "accounts/fireworks/models/llama-v3p1-70b-instruct"
        return {"cheap": default, "strong": default, "code": default}

    general = [m for m in allowed_models if not _CODE_MODEL_PAT.search(m.lower())] or allowed_models
    strong = max(general, key=_total_params)
    code_models = [m for m in allowed_models if _CODE_MODEL_PAT.search(m.lower())]
    code = max(code_models, key=_total_params) if code_models else strong
    cheap = min(allowed_models, key=_total_params)
    return {"cheap": cheap, "strong": strong, "code": code}


def select_best_model(prompt: str, allowed_models: List[str], task_type: Optional[str] = None) -> str:
    """Select the appropriate model tier (cheap / strong / code) for the task."""
    if not allowed_models:
        return "accounts/fireworks/models/llama-v3p1-70b-instruct"

    if task_type is None:
        task_type = classify_task(prompt)

    cfg = TASK_CONFIG.get(task_type, TASK_CONFIG["factual"])
    target_tier = cfg.get("tier", "strong")
    tiers = resolve_model_tiers(allowed_models)
    return tiers.get(target_tier, tiers["strong"])


# ---------------------------------------------------------------------------
# 4. Output Sanitization & CoT Stripping
# ---------------------------------------------------------------------------
def sanitize_output(raw_text: str, task_type: Optional[str] = None) -> str:
    """Strip chain-of-thought blocks and conversational wrappers."""
    if not raw_text:
        return ""

    cleaned = _THINK_RE.sub("", raw_text).strip()

    for pattern in _FLUFF_PREFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    for pattern in _FLUFF_SUFFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned


# ---------------------------------------------------------------------------
# 5. Pipeline Orchestration & Fallback Execution
# ---------------------------------------------------------------------------
def solve_prompt(prompt: str, api_key: str, base_url: str, allowed_models: List[str]) -> str:
    """
    Execute task prompt through the full hybrid routing pipeline:
    1. Tier 0 Local Deterministic Solver (0 API tokens)
    2. 8-Category Judge-Aligned Classification
    3. Dynamic Model Tiering (cheap / strong / code)
    4. Strict Judge System Prompts + Tight max_tokens budget
    5. Fallback Safety Net & Output Sanitization
    """
    if not prompt or not prompt.strip():
        return ""

    # Step 1: Check zero-token deterministic local solver
    import local_solvers
    local_ans = local_solvers.solve(prompt)
    if local_ans is not None:
        logger.info("Tier 0 Local Solver HIT: %s -> %s", prompt[:40], local_ans)
        return str(local_ans)

    # Step 2: Classify into one of 8 judge-aligned categories
    category = classify_task(prompt)
    cfg = TASK_CONFIG.get(category, TASK_CONFIG["factual"])

    # Step 3: Pick appropriate model tier
    model = select_best_model(prompt, allowed_models, category)

    if not api_key or not base_url:
        logger.warning("API credentials missing during solve_prompt execution.")
        return "Unable to determine answer."

    # Step 4: Execute remote API call with judge-aligned system prompt
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_api(target_model: str, max_toks: int) -> str:
        resp = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": cfg["system_prompt"]},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_toks,
        )
        return resp.choices[0].message.content or ""

    raw_ans = ""
    try:
        raw_ans = _call_api(model, cfg["max_tokens"])
    except Exception as e:
        logger.warning("Primary API call failed for model %s: %s", model, e)

    # Step 5: Fallback safety net (retry with strong model if primary failed or returned blank)
    if not raw_ans or not raw_ans.strip():
        tiers = resolve_model_tiers(allowed_models)
        strong_model = tiers.get("strong", model)
        if strong_model != model or not raw_ans:
            logger.info("Triggering Fallback Safety Net to strong model: %s", strong_model)
            try:
                raw_ans = _call_api(strong_model, max(200, cfg["max_tokens"]))
            except Exception as e:
                logger.error("Fallback API call failed: %s", e)

    return sanitize_output(raw_ans, category)
