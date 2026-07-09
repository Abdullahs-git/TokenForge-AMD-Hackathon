import re
import sympy
from typing import Optional

_nlp = None

def clean_math_prompt(prompt: str) -> str:
    """Cleans English conversational prefixes/suffixes from algebraic expressions."""
    text = prompt.strip()
    # Remove common English prefixes like 'solve:', 'what is', 'calculate'
    text = re.sub(r'^(what is|calculate|compute|solve for \w+:?|solve:?|evaluate:?)\s*', '', text, flags=re.IGNORECASE)
    text = text.rstrip('?.!')
    return text.strip()

def solve_math(prompt: str) -> Optional[str]:
    """
    Deterministically solves pure arithmetic or linear equations using SymPy ($0 token cost).
    Returns None if prompt is a complex multi-step word problem requiring LLM reasoning.
    """
    try:
        cleaned = clean_math_prompt(prompt)
        if not cleaned:
            return None

        # Don't try to sympify long conversational word problems
        if len(cleaned.split()) > 10 and not ("=" in cleaned and len(cleaned.split("=")) == 2):
            return None

        if "=" in cleaned:
            parts = cleaned.split("=")
            if len(parts) == 2:
                lhs = sympy.sympify(parts[0].strip())
                rhs = sympy.sympify(parts[1].strip())
                eq = sympy.Eq(lhs, rhs)
                solutions = sympy.solve(eq)
                return f"Answer: {solutions[0]}" if len(solutions) == 1 else f"Answer: {solutions}"
        
        # Pure arithmetic evaluation
        expr = sympy.sympify(cleaned)
        # If it evaluated to a number or simplified expression
        return f"Answer: {expr}"
    except Exception:
        return None

def analyze_sentiment(prompt: str) -> Optional[str]:
    """
    Deterministically classifies sentiment and justifies it using VADER lexicon ($0 token cost).
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # Clean classification prompt boilerplate if present
        text = re.sub(
            r'^(classify the sentiment of this review:?|classify the sentiment:?|what is the sentiment of:?|sentiment of:?)\s*',
            '',
            prompt.strip(),
            flags=re.IGNORECASE
        )
        
        scores = analyzer.polarity_scores(text)
        compound = scores.get("compound", 0.0)
        
        if compound >= 0.05:
            return (
                f"Sentiment: Positive\n"
                f"Justification: The text exhibits an overall positive tone with positive lexical indicators "
                f"(VADER compound score: {compound:.2f})."
            )
        elif compound <= -0.05:
            return (
                f"Sentiment: Negative\n"
                f"Justification: The text exhibits an overall negative tone with negative lexical indicators "
                f"(VADER compound score: {compound:.2f})."
            )
        else:
            return (
                f"Sentiment: Neutral\n"
                f"Justification: The text uses balanced or objective language without strong emotional polarity "
                f"(VADER compound score: {compound:.2f})."
            )
    except Exception:
        return None

def extract_ner(prompt: str) -> Optional[str]:
    """
    Deterministically extracts named entities using spaCy ($0 token cost).
    Falls back gracefully if model is not present.
    """
    global _nlp
    try:
        import spacy
        if _nlp is None:
            _nlp = spacy.load("en_core_web_sm")
        doc = _nlp(prompt)
        entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
        if not entities:
            return None
        return "Named Entities: " + ", ".join(entities)
    except Exception:
        return None
