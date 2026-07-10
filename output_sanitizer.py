"""
TokenForge v5.0 — Output Sanitizer

Validates non-empty output and provides repair prompt generation.
Does NOT strip or mutate LLM output — raw model responses are passed
directly to results.json so the LLM-Judge receives them intact.
"""

from typing import Optional


def extract_answer(raw: str, category: str) -> str:
    """
    Returns the raw LLM output ready for results.json.
    Preserves all formatting (markdown, bullet points, code fences)
    so the LLM-Judge receives the answer exactly as the model wrote it.
    """
    if not raw or not raw.strip():
        return ""
    return raw.strip()


def strip_formatting(text: str) -> str:
    """Trim leading/trailing whitespace only. Does NOT strip markdown."""
    if not text:
        return ""
    return text.strip()


def validate_answer(answer: str) -> bool:
    """Check if an answer is valid (non-empty, non-placeholder)."""
    if not answer or not answer.strip():
        return False
    bad_patterns = [
        "[Local Offline Mode]",
        "Error processing prompt",
        "Unable to generate response",
    ]
    return not any(bp in answer for bp in bad_patterns)


_REPAIR_PROMPTS = {
    "factual": "Answer the question directly and accurately.",
    "math": "Solve the problem. Show brief steps. End with 'Answer: <value>' on its own line.",
    "sentiment": "Classify as Positive, Negative, Neutral, or Mixed. Give one sentence justification.",
    "summarization": "Provide only the summary. Obey any stated length or format constraint.",
    "ner": "List each entity as 'LABEL: name', one per line. Labels: PERSON, ORGANIZATION, LOCATION, DATE.",
    "code_debug": "Name the bug in one sentence, then provide the corrected code in a fenced block.",
    "logical": "Solve step by step. End with 'Answer: <value>' on its own line.",
    "code_gen": "Write the requested code in one fenced block, correct and self-contained.",
}


def get_repair_prompt(category: str, original_prompt: str) -> str:
    """Generate a repair prompt for when the primary answer is blank or invalid."""
    instruction = _REPAIR_PROMPTS.get(category, "Answer clearly.")
    return f"{instruction}\n\nOriginal question: {original_prompt}"
