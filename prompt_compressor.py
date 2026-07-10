"""
TokenForge v7.0 — Prompt Compressor

Strips filler phrases and unnecessary verbosity from input prompts
to reduce input token count before sending to the LLM.
"""

import re
from typing import List

# Filler phrases to strip (ordered longest-first for greedy matching)
_FILLER_PHRASES: List[str] = [
    "could you please",
    "can you please",
    "i would like you to",
    "i want you to",
    "please help me",
    "please help me to",
    "i need you to",
    "i'd like you to",
    "would you be able to",
    "would you mind",
    "would you please",
    "thanks in advance",
    "thank you in advance",
    "thank you",
    "thanks",
    "if possible",
    "if you can",
    "if you could",
    "make sure to",
    "make sure that",
    "be sure to",
    "please note that",
    "note that",
    "keep in mind that",
    "it is important to",
    "it's important to",
    "please provide",
    "kindly provide",
    "kindly",
    "please",
]

# Pre-compile patterns (case-insensitive, word-bounded)
_FILLER_PATTERNS = [
    re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
    for phrase in _FILLER_PHRASES
]

_MULTI_SPACE = re.compile(r"\s{2,}")
_TRAILING_PUNCT = re.compile(r"[.!?]+$")


def compress(prompt: str) -> str:
    """Safe prompt compression: collapse multiple spaces and newlines without removing words."""
    if not prompt:
        return prompt

    text = prompt.strip()
    # Collapse multiple spaces
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()
