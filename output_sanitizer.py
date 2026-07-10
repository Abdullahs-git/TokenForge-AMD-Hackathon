"""
TokenForge v6.0 — Output Sanitizer

Cleans up model preambles and formatting to keep the answers short,
clean, and correct for the LLM-Judge evaluation.
"""
import re

_PREAMBLE_RE = re.compile(
    r"^(sure[,.!]?|okay[,.!]?|certainly[,.!]?|here'?s?( is)?( the)?( answer)?[,.!]?:?|the answer is[,.!]?:?|"
    r"the correct answer is[,.!]?:?|final answer[,.!]?:?|answer[,.!]?:?|result[,.!]?:?)\s*",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)

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
    Cleans wrapping fences and conversational preambles from non-code tasks
    so the final output remains compact and focused.
    """
    if not raw or not raw.strip():
        return ""
    
    cleaned = raw.strip()
    if category not in ("code_gen", "code_debug"):
        # For non-code tasks, clean wrapping fences and pleasantries
        cleaned = strip_code_fences(cleaned)
        cleaned = strip_preamble(cleaned)
    else:
        # For code tasks, only strip wrapping fences so indentation remains intact
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
    "factual": "Answer the question directly and accurately.",
    "math": "Solve the problem. Show brief steps. End with 'Answer: <value>' on its own line.",
    "sentiment": "Classify as Positive, Negative, Neutral, or Mixed. Give one sentence justification.",
    "summarization": "Provide only the summary. Obey any stated length or format constraint.",
    "ner": "List each entity as 'LABEL: name', one per line. Types: PERSON, ORGANIZATION, LOCATION, DATE.",
    "code_debug": "Name the bug in one sentence, then provide the corrected code in a fenced block.",
    "logical": "Solve step by step. End with 'Answer: <value>' on its own line.",
    "code_gen": "Write the requested code in one fenced block, correct and self-contained.",
}

def get_repair_prompt(category: str, original_prompt: str) -> str:
    """Generate a repair prompt for when the primary answer is blank or invalid."""
    instruction = _REPAIR_PROMPTS.get(category, "Answer clearly.")
    return f"{instruction}\n\nOriginal question: {original_prompt}"
