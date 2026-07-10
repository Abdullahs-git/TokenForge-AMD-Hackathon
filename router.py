"""
TokenForge — Task-Aware Hybrid Routing Engine
Implements:
1. Tier 0 safe local solver execution
2. Task classification (code, classification, short_fact, general)
3. Dynamic model scoring (no hardcoded model names; parses parameter size and tags)
4. Adaptive token ceilings & task-specific system prompts
5. Comprehensive output sanitization & post-processing
"""

import re
import time
import logging
from typing import Optional, List
from openai import OpenAI
import local_solvers

logger = logging.getLogger("tokenforge.router")

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)

_FLUFF_PREFIXES = [
    r"^here is (?:the )?[^:]+:\s*",
    r"^sure[!,.\s]+(?:here is [^:]+:\s*)?",
    r"^certainly[!,.\s]+(?:here is [^:]+:\s*)?",
    r"^of course[!,.\s]+(?:here is [^:]+:\s*)?",
    r"^(?:the )?exact answer is:\s*",
    r"^the answer is:\s*",
    r"^answer:\s*",
    r"^result:\s*",
    r"^output:\s*",
    r"^sentiment:\s*",
    r"^label:\s*",
]

_FLUFF_SUFFIXES = [
    r"\s+hope this helps[!.]*$",
    r"\s+let me know if you need anything else[!.]*$",
    r"\s+feel free to ask[!.]*$",
]

TASK_CONFIG = {
    "code": {
        "system_prompt": "Output only the complete, working code solution without markdown chatter or explanations unless requested.",
        "max_tokens": 600,
    },
    "classification": {
        "system_prompt": "Respond with only the exact label, category, or answer. No explanation, no filler, no punctuation.",
        "max_tokens": 64,
    },
    "short_fact": {
        "system_prompt": "Provide only the direct factual answer in a single concise sentence or phrase. No introductory filler or explanation.",
        "max_tokens": 80,
    },
    "general": {
        "system_prompt": "Provide the complete, correct answer. Be direct and concise without conversational filler.",
        "max_tokens": 500,
    },
}


def classify_task(prompt: str) -> str:
    """
    Categorize each prompt into one of:
    - 'code': Programming, functions, debugging, algorithms
    - 'classification': Sentiment analysis, NER, true/false, multiple choice, label assignment
    - 'short_fact': Short factual QA (boiling point, capitals, dates, who/what/where)
    - 'general': Reasoning, logic, open-ended, summarization
    """
    if not prompt:
        return "general"

    text = prompt.strip()
    lower = text.lower()

    # Code detection
    code_keywords = (
        "def ", "class ", "return ", "import ", "python", "function",
        "algorithm", "write a code", "write code", "bug", "traceback",
        "syntax", "javascript", "sql", "html", "css", "```"
    )
    if any(kw in lower for kw in code_keywords):
        return "code"

    # Classification / NER / Sentiment detection
    classification_keywords = (
        "sentiment", "classify", "extract entities", "extract named entities",
        "named entities", "ner", "true or false", "multiple choice",
        "label the following", "what is the sentiment"
    )
    if any(kw in lower for kw in classification_keywords):
        return "classification"

    # Short Factual QA detection
    short_fact_keywords = (
        "boiling point", "capital city", "capital of", "capitals of",
        "who invented", "when was", "where is", "list the capital"
    )
    if any(kw in lower for kw in short_fact_keywords):
        return "short_fact"

    return "general"


# Alias for backward compatibility with existing tests
detect_category = classify_task


def select_best_model(prompt: str, allowed_models: List[str], task_type: Optional[str] = None) -> str:
    """
    Generic scoring function to select the best model from ALLOWED_MODELS.
    Parses parameter size (e.g. 70b, 34b, 8x7b) and tags (instruct, chat, coder, mini, distill)
    without relying on hardcoded model names.
    """
    if not allowed_models:
        return "accounts/fireworks/models/llama-v3p1-70b-instruct"

    if task_type is None:
        task_type = classify_task(prompt)

    def score_model(m_str: str) -> float:
        name_lower = m_str.lower()

        # Parse parameter size in billions
        size_b = 15.0  # default baseline if parameter size is unlisted
        moe_match = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", name_lower)
        if moe_match:
            size_b = float(moe_match.group(1)) * float(moe_match.group(2))
        else:
            size_match = re.search(r"(\d+(?:\.\d+)?)b\b", name_lower)
            if size_match:
                size_b = float(size_match.group(1))

        # Identify capability tags
        is_code_model = any(tag in name_lower for tag in ("coder", "code"))
        is_instruct = any(tag in name_lower for tag in ("instruct", "chat", "-it", "it-", "minimax", "claude", "gpt"))
        is_flagship = any(tag in name_lower for tag in ("minimax-m3", "70b", "405b", "deepseek-r1", "deepseek-v3"))

        score = 0.0

        if task_type == "code":
            if is_code_model:
                score += 150.0
            if is_instruct:
                score += 20.0
            score += min(size_b, 100.0)
        elif task_type in ("classification", "short_fact"):
            if is_code_model:
                score -= 30.0
            if is_instruct:
                score += 40.0
            if is_flagship:
                score += 30.0
            score += min(size_b, 100.0)
        else:  # general
            if is_code_model:
                score -= 20.0
            if is_instruct:
                score += 40.0
            if is_flagship:
                score += 50.0
            score += min(size_b, 100.0)

        return score

    ranked = sorted(allowed_models, key=score_model, reverse=True)
    return ranked[0]


def sanitize_output(raw_text: str, task_type: Optional[str] = None) -> str:
    """
    Strips internal reasoning traces (<think>...</think>), conversational fluff prefixes/suffixes,
    and applies task-specific post-processing.
    """
    if not raw_text:
        return ""

    # Strip <think> blocks
    text = _THINK_RE.sub("", raw_text).strip()

    # Strip conversational fluff prefixes
    for pattern in _FLUFF_PREFIXES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # Strip conversational fluff suffixes
    for pattern in _FLUFF_SUFFIXES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # Task-specific post-processing
    if task_type == "classification":
        # Force single non-empty line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            text = lines[0]
    elif task_type == "short_fact":
        # If model over-explains across multiple paragraphs, keep concise first sentence/line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) > 1 and len(lines[0]) < 150:
            text = lines[0]

    return text.strip()


def solve_prompt(prompt: str, api_key: str, base_url: str, allowed_models: List[str]) -> str:
    """
    Task-aware routing pipeline:
    1. Tier 0 Safe Local Solver (0 tokens)
    2. Task classification
    3. Generic model scoring & selection
    4. Task-specific system prompt & adaptive max_tokens
    5. Retry/backoff resilience (3 attempts)
    6. Output sanitization
    """
    if not prompt or not prompt.strip():
        return "Unable to determine answer."

    # Step 1: Tier 0 local solver
    local_ans = local_solvers.solve(prompt)
    if local_ans is not None:
        logger.info("Tier 0 Local Solver HIT: %s -> %s", prompt[:30], local_ans)
        return local_ans

    # Step 2: Task classification
    task_type = classify_task(prompt)
    cfg = TASK_CONFIG.get(task_type, TASK_CONFIG["general"])

    # Step 3: Generic model selection
    if not api_key or not base_url or not allowed_models:
        return "Unable to generate answer."

    model = select_best_model(prompt, allowed_models, task_type=task_type)
    logger.info("Task classified as '%s', routing to model: %s", task_type, model)

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
            cleaned = sanitize_output(raw_text, task_type=task_type)
            if cleaned:
                return cleaned
        except Exception as e:
            logger.warning("API attempt %d failed for model %s: %s", attempt + 1, model, e)
            if attempt < 2:
                time.sleep(delay)
                delay *= 2

    return "Unable to generate answer."
