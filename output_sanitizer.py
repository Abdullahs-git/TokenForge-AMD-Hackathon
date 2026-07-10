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
    """Clean trailing whitespace without mutating markdown or structure."""
    if not text:
        return ""
    return text.strip()


def extract_answer(raw: str, category: str) -> str:
    """
    Returns the cleaned raw LLM output ready for results.json.
    Preserves all lists, code blocks, and markdown structure for the LLM-Judge.
    """
    if not raw or not raw.strip():
        return ""
    return raw.strip()


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
