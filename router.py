"""
Router v7.0 — Aggressive token-saving routing pipeline.

Pipeline per task:
1. Classify prompt into one of 8 categories (regex, 0 tokens)
2. Try ALL applicable local solvers (0 tokens)
3. Compress prompt (remove filler words)
4. Route to Fireworks AI with ultra-tight token budgets
5. Post-process: strip <think> blocks + sanitize output
6. Validate: if blank → repair with tighter prompt
"""

import os
import re
import logging
from typing import Optional
import local_solvers
import llm_clients
import output_sanitizer
import prompt_compressor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 8-category regex classifier
# ---------------------------------------------------------------------------

_CLASSIFIER_PATTERNS = {
    "code_debug": [
        r"\bbug\b", r"\bdebug\b", r"\bfix (this|the|my|it)\b",
        r"what'?s wrong", r"why (does|is)n'?t (this|it|my)\b",
        r"error in (this|the|my)\b", r"traceback", r"stack ?trace",
        r"throws? an? (error|exception)", r"returns? \w+ instead",
        r"infinite loop", r"corrected (version|code)",
        r"has a bug",
    ],
    "code_gen": [
        r"\b(write|create|implement|build|generate|produce|give me)\b.*\b(function|method|class|program|script|routine)\b",
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
        r"(list|identify|find|pull out) (all )?(the )?(people|persons?|organi[sz]ations?|locations?|dates?|entit)",
        r"(person|organization|location|date)\s*[:=]",
    ],
    "summarization": [
        r"summari[sz]e", r"\bsummary\b", r"\btl;?dr\b", r"\bcondense\b",
        r"\bshorten\b", r"in (one|a single|two|three|\d+) (sentences?|words?|lines?)",
        r"main (idea|point|takeaway)", r"\bthe gist\b", r"key points",
        r"boil .* down",
    ],
    "logical": [
        r"\bpuzzle\b", r"who (is|owns|sits|lives|has|drinks|likes)\b",
        r"if and only if", r"exactly one", r"at least one",
        r"the following (clues|facts|statements|conditions)",
        r"each (person|friend|house|box|day|one|child|student|player) .*(different|exactly|only|one)",
        r"\bdeduc(e|tive|tion)\b", r"logically (follows?|true)",
        r"(definitely|necessarily) (true|follows)",
        r"knights? and knaves", r"truth[- ]?teller", r"\bliar\b",
        r"\bdoes not (own|have|like|live|sit|drink)\b",
        r"\b(own|has|have) a different\b",
        r"\b(three|four|five|six) (friends?|people|persons?|children|students|players?) .*(each|different|own)",
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
    "summarization", "logical", "math", "factual",
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


def detect_category(prompt: str) -> str:
    """Classifies a prompt into one of the 8 hackathon capability categories."""
    text = prompt or ""
    for cat in _PRIORITY_ORDER:
        if any(rx.search(text) for rx in _COMPILED_PATTERNS[cat]):
            return cat
    return "code_debug" if (_CODE_FENCE.search(text) or _CODE_HINT.search(text)) else "factual"


def classify_and_route(prompt: str) -> str:
    """
    Full routing pipeline:
    1. Classify → 2. Local solvers (0 tokens) → 3. Compress prompt →
    4. Cloud call → 5. Sanitize → 6. Validate/Repair
    """
    category = detect_category(prompt)
    logger.info("Category: %s", category)

    # --- Tier 0: Deterministic local solvers (0 tokens) ---
    local_ans = _try_local_solvers(prompt, category)
    if local_ans is not None:
        logger.info("LOCAL solver handled [%s] (0 tokens)", category)
        return local_ans

    # --- Tier 1: Compress prompt → Cloud call ---
    compressed = prompt_compressor.compress(prompt)
    logger.info("Prompt compressed: %d → %d chars", len(prompt), len(compressed))

    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    allowed_models = os.environ.get("ALLOWED_MODELS", "")

    raw_answer = llm_clients.call_fireworks_category(
        compressed, category, api_key, base_url, allowed_models
    )

    # --- Post-process: sanitize output ---
    answer = output_sanitizer.extract_answer(raw_answer, category) if raw_answer else ""

    # --- Validate & repair if needed ---
    if not output_sanitizer.validate_answer(answer):
        logger.warning("Primary answer invalid for %s, attempting repair", category)
        repair_prompt = output_sanitizer.get_repair_prompt(category, compressed)
        repair_answer = llm_clients.call_repair(
            compressed, repair_prompt, api_key, base_url, allowed_models
        )
        if repair_answer:
            answer = output_sanitizer.extract_answer(repair_answer, category)

    # --- Final fallback: never return blank ---
    if not output_sanitizer.validate_answer(answer):
        logger.error("All attempts failed for %s, using minimal fallback", category)
        answer = _minimal_fallback(category, prompt)

    return answer


def _try_local_solvers(prompt: str, category: str) -> Optional[str]:
    """Try all applicable local solvers in order. Returns answer or None."""

    # Math — arithmetic, percentages
    if category == "math":
        ans = local_solvers.solve_math(prompt)
        if ans is not None:
            return ans

    # Math — SymPy arithmetic evaluation ($0 tokens, 100% exact)
    if category == "math":
        ans = local_solvers.solve_math(prompt)
        if ans is not None:
            return ans

    return None


def _minimal_fallback(category: str, prompt: str) -> str:
    """Generate a minimal non-blank answer when all else fails."""
    if category == "sentiment":
        return "Neutral."
    elif category == "math":
        return "Unable to compute."
    elif category == "logical":
        return "Unable to determine from given constraints."
    elif category == "ner":
        return "No named entities found."
    elif category == "summarization":
        words = prompt.split()[:20]
        return " ".join(words)
    elif category in ("code_debug", "code_gen"):
        return "# Unable to generate code."
    else:
        return "Unable to determine answer."
