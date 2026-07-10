"""
TokenForge v7.0 — Output Sanitizer

Cleans up model preambles, <think> blocks, and formatting to keep the answers
short, clean, and correct for the LLM-Judge evaluation.
"""
import re

_PREAMBLE_RE = re.compile(
    r"^(sure[,.!]?|okay[,.!]?|certainly[,.!]?|of course[,.!]?|"
    r"here'?s?( is)?( the)?( answer)?[,.!]?:?|"
    r"the answer is[,.!]?:?|the correct answer is[,.!]?:?|"
    r"final answer[,.!]?:?|answer[,.!]?:?|result[,.!]?:?)\s*",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)

# <think>...</think> blocks from reasoning models
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Unclosed <think> blocks (model hit max_tokens mid-reasoning)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning traces from model output."""
    if not text:
        return text
    # First remove closed blocks
    text = _THINK_BLOCK_RE.sub("", text)
    # Then remove unclosed blocks (truncated reasoning)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


def strip_preamble(text: str) -> str:
    """Recursively strip conversational headers from the start of the answer."""
    text = (text or "").strip()
    prev = None
    while prev != text:
        prev = text
        text = _PREAMBLE_RE.sub("", text).strip()
    return text


def strip_code_fences(text: str) -> str:
    """Strip wrapping ```code``` fences from the answer."""
    return _CODE_FENCE_RE.sub("", text or "").strip()


def extract_answer(raw: str, category: str) -> str:
    """
    Returns the cleaned raw LLM output ready for results.json.
    """
    if not raw or not raw.strip():
        return ""

    # ALWAYS strip <think> blocks first
    cleaned = strip_think_blocks(raw)

    if not cleaned.strip():
        return ""

    cleaned = cleaned.strip()

    if category not in ("code_gen", "code_debug"):
        cleaned = strip_code_fences(cleaned)
        cleaned = strip_preamble(cleaned)
    else:
        cleaned = strip_code_fences(cleaned)

    return cleaned.strip()


def strip_formatting(text: str) -> str:
    """Trim leading/trailing whitespace only."""
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
    "factual": "Answer directly.",
    "math": "Final answer only. Format: Answer: <value>",
    "sentiment": "One word: Positive, Negative, or Neutral.",
    "summarization": "Summarize. Obey length constraint.",
    "ner": "List entities as LABEL: name, one per line.",
    "code_debug": "Corrected code only.",
    "logical": "Final answer only.",
    "code_gen": "Code only, no explanation.",
}


def get_repair_prompt(category: str, original_prompt: str) -> str:
    """Generate a repair prompt for when the primary answer is blank or invalid."""
    instruction = _REPAIR_PROMPTS.get(category, "Answer clearly.")
    return f"{instruction}\n\n{original_prompt}"
