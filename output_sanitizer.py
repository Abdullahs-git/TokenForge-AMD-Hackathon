"""
Output Sanitizer & Answer Extractor for TokenForge v3.0

Post-processes raw LLM output to:
1. Strip ALL markdown/HTML formatting (bold, code fences, headers, links)
2. Extract structured answers for math/logic categories
3. Normalize sentiment labels
4. Validate non-empty output
5. Provide repair prompt generation for blank/invalid answers
"""

import re
import json
from typing import Optional


# ---------------------------------------------------------------------------
# Markdown & HTML stripping
# ---------------------------------------------------------------------------

# Order matters: strip code fences first (preserve inner content), then inline
_STRIP_PATTERNS = [
    # Fenced code blocks: keep inner content, remove fences
    (re.compile(r"```[\w]*\n?(.*?)```", re.DOTALL), r"\1"),
    # HTML tags
    (re.compile(r"<[^>]+>"), ""),
    # Bold / italic (*** ** * __ _)
    (re.compile(r"\*{1,3}(.*?)\*{1,3}"), r"\1"),
    (re.compile(r"_{1,3}(.*?)_{1,3}"), r"\1"),
    # Inline code backticks
    (re.compile(r"`([^`]+)`"), r"\1"),
    # Markdown headers (# ## ### etc.)
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    # Markdown links [text](url) → text
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    # Markdown images ![alt](url) → remove entirely
    (re.compile(r"!\[[^\]]*\]\([^)]+\)"), ""),
    # Horizontal rules
    (re.compile(r"^[\-\*_]{3,}\s*$", re.MULTILINE), ""),
    # Blockquotes
    (re.compile(r"^>\s?", re.MULTILINE), ""),
    # Unordered list markers at line start (but keep content)
    (re.compile(r"^[\*\-\+]\s+", re.MULTILINE), ""),
]

# Collapse excessive whitespace
_MULTI_SPACE = re.compile(r"  +")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def strip_formatting(text: str) -> str:
    """Remove all markdown and HTML formatting from LLM output."""
    result = text
    for pattern, replacement in _STRIP_PATTERNS:
        result = pattern.sub(replacement, result)
    result = _MULTI_SPACE.sub(" ", result)
    result = _MULTI_NEWLINE.sub("\n\n", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Answer extraction per category
# ---------------------------------------------------------------------------

# Math/Logic: extract "Answer: <value>" from the end of the response
_ANSWER_LINE = re.compile(
    r"(?:^|\n)\s*(?:Answer|ANSWER|Final Answer|final_answer|Result)\s*[:=]\s*(.+?)$",
    re.MULTILINE | re.IGNORECASE,
)

# JSON extraction for structured reasoning (optional)
_JSON_BLOCK = re.compile(r"\{[^{}]*\"final_answer\"\s*:\s*\"([^\"]+)\"[^{}]*\}", re.IGNORECASE)

# Sentiment normalization
_SENTIMENT_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "mixed": "Mixed",
}
_SENTIMENT_PAT = re.compile(
    r"\b(positive|negative|neutral|mixed)\b", re.IGNORECASE
)


def extract_answer(raw: str, category: str) -> str:
    """
    Post-process raw LLM output based on category.
    Returns a cleaned answer string ready for results.json.
    """
    if not raw or not raw.strip():
        return ""

    text = raw.strip()

    if category in ("math", "logical"):
        return _extract_math_logic(text)
    elif category == "sentiment":
        return _extract_sentiment(text)
    elif category == "ner":
        return _extract_ner(text)
    elif category in ("code_debug", "code_gen"):
        return _extract_code(text)
    elif category == "summarization":
        return strip_formatting(text)
    elif category == "factual":
        return strip_formatting(text)
    else:
        return strip_formatting(text)


def _extract_math_logic(text: str) -> str:
    """
    For math and logic: keep the full reasoning but ensure it's clean.
    The judge needs to see the reasoning steps AND the final answer.
    """
    cleaned = strip_formatting(text)
    return cleaned


def _extract_sentiment(text: str) -> str:
    """Normalize sentiment output to consistent format."""
    cleaned = strip_formatting(text)
    return cleaned


def _extract_ner(text: str) -> str:
    """Normalize NER output format."""
    cleaned = strip_formatting(text)
    return cleaned


def _extract_code(text: str) -> str:
    """
    For code tasks: preserve code blocks but strip outer markdown.
    Keep code fences for code_debug and code_gen since the judge expects them.
    """
    # For code output, we want to keep the fenced code blocks intact
    # but strip any OTHER markdown formatting around them
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text.strip()


# ---------------------------------------------------------------------------
# Repair prompt generation
# ---------------------------------------------------------------------------

_REPAIR_PROMPTS = {
    "factual": "Answer the question directly in plain text. No markdown formatting.",
    "math": "Solve the problem. Show brief steps. End with 'Answer: <value>' on its own line. No markdown.",
    "sentiment": "Classify the sentiment as Positive, Negative, Neutral, or Mixed. Give one sentence justification. No markdown.",
    "summarization": "Provide only the summary as plain text. No markdown.",
    "ner": "List each named entity as 'TYPE: name', one per line. Types: PERSON, ORGANIZATION, LOCATION, DATE. No markdown.",
    "code_debug": "Identify the bug and provide the corrected code. No markdown outside code blocks.",
    "logical": "Solve step by step. End with 'Answer: <value>' on its own line. No markdown.",
    "code_gen": "Write the requested code. Output only the code. No markdown outside code blocks.",
}


def get_repair_prompt(category: str, original_prompt: str) -> str:
    """Generate a repair prompt for when the primary answer is blank or invalid."""
    instruction = _REPAIR_PROMPTS.get(category, "Answer clearly in plain text. No markdown.")
    return f"{instruction}\n\nOriginal question: {original_prompt}"


def validate_answer(answer: str) -> bool:
    """Check if an answer is valid (non-empty, non-placeholder)."""
    if not answer or not answer.strip():
        return False
    # Reject known failure patterns
    bad_patterns = [
        "[Local Offline Mode]",
        "Error processing prompt",
        "Unable to generate response",
    ]
    return not any(bp in answer for bp in bad_patterns)
