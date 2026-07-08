import re
from typing import Optional
import sympy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

# Global NLP engine loaded lazily to speed up startup
_nlp = None

def solve_math(prompt: str) -> Optional[str]:
    """Extracts and solves equations or evaluates arithmetic in the prompt using SymPy."""
    # Look for explicit equations: e.g., "x + 2 = 5"
    eq_match = re.search(r'([a-zA-Z0-9\s\+\-\*\/\(\)\^]+)\s*=\s*([a-zA-Z0-9\s\+\-\*\/\(\)\^]+)', prompt)
    if eq_match:
        lhs_str = eq_match.group(1).strip()
        rhs_str = eq_match.group(2).strip()
        try:
            # Find variable characters (single letters like x, y, z)
            vars_in_lhs = re.findall(r'\b[a-zA-Z]\b', lhs_str)
            vars_in_rhs = re.findall(r'\b[a-zA-Z]\b', rhs_str)
            variables = sorted(list(set(vars_in_lhs + vars_in_rhs)))
            
            lhs = sympy.sympify(lhs_str)
            rhs = sympy.sympify(rhs_str)
            
            if not variables:
                return str(lhs == rhs)
            
            eq = lhs - rhs
            symbols = [sympy.Symbol(v) for v in variables]
            solutions = sympy.solve(eq, symbols)
            return f"Solutions for {', '.join(variables)}: {solutions}"
        except Exception:
            return None

    # Fallback to evaluating mathematical expressions
    expr_match = re.findall(r'[\d\+\-\*\/\(\)\^\.]+', prompt)
    if expr_match:
        # Find the longest candidate math substring
        expr_str = max(expr_match, key=len).strip()
        # Clean up trailing/leading operators
        expr_str = re.sub(r'^[\+\-\*\/]+|[\+\-\*\/]+$', '', expr_str).strip()
        if len(expr_str) > 2 and re.search(r'[\+\-\*\/]', expr_str):
            try:
                res = sympy.sympify(expr_str)
                return f"Result: {res}"
            except Exception:
                return None
    return None

def analyze_sentiment(prompt: str) -> str:
    """Analyzes the sentiment of the prompt using VADER sentiment analysis."""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(prompt)
    compound = scores.get("compound", 0.0)
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return f"Sentiment: {sentiment} (Compound Score: {compound})"

def extract_ner(prompt: str) -> str:
    """Extracts Named Entities from the prompt using spaCy."""
    global _nlp
    if _nlp is None:
        # Load the small English model
        _nlp = spacy.load("en_core_web_sm")
    doc = _nlp(prompt)
    entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
    if not entities:
        return "No named entities found."
    return f"Entities: {', '.join(entities)}"
