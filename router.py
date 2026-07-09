import os
import re
from typing import Optional
import local_solvers
import llm_clients

# Comprehensive regex patterns covering all 8 hackathon capability categories
_CLASSIFIER_PATTERNS = {
    "code_debug": [
        r"\bbug\b", r"\bdebug\b", r"\bfix (this|the|my|it)\b",
        r"what'?s wrong", r"why (does|is)n'?t (this|it|my)\b",
        r"error in (this|the|my)\b", r"traceback", r"stack ?trace",
        r"throws? an? (error|exception)", r"returns? \w+ instead",
        r"infinite loop", r"corrected (version|code)",
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
        r"each (person|house|box|day|one) .*(different|exactly|only|one)",
        r"\bdeduce\b", r"logically (follows?|true)",
        r"(definitely|necessarily) (true|follows)",
        r"knights? and knaves", r"truth[- ]?teller", r"\bliar\b",
    ],
    "math": [
        r"\bcalculate\b", r"\bcompute\b", r"how (much|many)\b", r"percent",
        r"\d+\s*%", r"\bsum of\b", r"\baverage\b", r"solve for\b",
        r"\d+\s*[+\-*/x×÷]\s*\d+", r"total (cost|price|amount)",
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

def detect_category(prompt: str) -> str:
    """Classifies a prompt into one of the 8 hackathon capability categories."""
    text = prompt or ""
    for cat in _PRIORITY_ORDER:
        if any(rx.search(text) for rx in _COMPILED_PATTERNS[cat]):
            return cat
    return "factual"

def classify_and_route(prompt: str) -> str:
    """
    Classifies prompt and routes to Tier 0 (Local Deterministic Solver) or Tier 1 (Fireworks AI Cloud).
    """
    category = detect_category(prompt)

    # 1. Tier 0: Deterministic Local Fast-Path ($0.00 Tokens)
    #    ONLY pure arithmetic goes here — SymPy is 100% precise on clean expressions.
    #    Sentiment & NER are routed to cloud for accuracy safety:
    #      - VADER misclassifies mixed-sentiment reviews (e.g. "great battery, bad screen" → "Positive")
    #      - spaCy en_core_web_sm mislabels entities (e.g. "Fireworks AI" → FAC instead of ORG)
    if category == "math":
        local_ans = local_solvers.solve_math(prompt)
        if local_ans is not None:
            return local_ans

    # 2. Tier 1: Optimal Fireworks Cloud Tiering
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    allowed_models = os.environ.get("ALLOWED_MODELS", "")

    cloud_ans = llm_clients.call_fireworks_category(prompt, category, api_key, base_url, allowed_models)
    if cloud_ans:
        return cloud_ans

    # 3. Graceful Fallback if API key is unconfigured during local test
    return f"[Local Offline Mode] Category detected: {category}. Prompt received: {prompt}"
