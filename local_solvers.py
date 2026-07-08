import sympy
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_nlp = None

def solve_math(prompt: str) -> str | None:
    try:
        if "=" in prompt:
            parts = prompt.split("=")
            if len(parts) == 2:
                lhs = sympy.sympify(parts[0].strip())
                rhs = sympy.sympify(parts[1].strip())
                eq = sympy.Eq(lhs, rhs)
                solutions = sympy.solve(eq)
                return str(solutions)
        expr = sympy.sympify(prompt.strip())
        return str(expr)
    except Exception:
        return None

def analyze_sentiment(prompt: str) -> str | None:
    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(prompt)
        compound = scores.get("compound", 0.0)
        if compound >= 0.05:
            return f"Positive. The overall sentiment is positive as VADER analysis detected positive emotion words (Compound Score: {compound})."
        elif compound <= -0.05:
            return f"Negative. The overall sentiment is negative as VADER analysis detected negative emotion words (Compound Score: {compound})."
        else:
            return f"Neutral. The overall sentiment is neutral since the compound score indicates lack of strong positive or negative language (Compound Score: {compound})."
    except Exception:
        return None

def extract_ner(prompt: str) -> str | None:
    global _nlp
    try:
        if _nlp is None:
            _nlp = spacy.load("en_core_web_sm")
        doc = _nlp(prompt)
        entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
        if not entities:
            return None
        return ", ".join(entities)
    except Exception:
        return None
