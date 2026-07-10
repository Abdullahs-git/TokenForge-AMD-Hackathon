"""
TokenForge v9.0 — Safe Local Arithmetic Solver (Tier 0)
Deterministic execution for pure numerical arithmetic expressions ($0.00 API cost, 0 tokens).
"""

import re
from typing import Optional

# Only allow pure arithmetic numbers and operators (+ - * / ^)
_PURE_ARITHMETIC_RE = re.compile(r"^[0-9\s\.\+\-\*\/\(\)\^]+$")


def solve_math_expression(prompt: str) -> Optional[str]:
    """
    Attempts to evaluate strictly pure arithmetic expressions locally using SymPy.
    If the prompt contains words, percentages, units, or algebraic variables, returns None
    so it is safely handled by Tier 1 SOTA models with 100% accuracy.
    """
    text = prompt.strip()
    if not text:
        return None

    # Check for simple "What is X?" prefix
    expr_str = text
    lower = text.lower()
    if lower.startswith("what is "):
        expr_str = text[8:].strip()
    elif lower.startswith("calculate "):
        expr_str = text[10:].strip()
    elif lower.startswith("compute "):
        expr_str = text[8:].strip()

    expr_str = expr_str.rstrip("?").strip()

    # Must match strictly pure digits and arithmetic operators (+ - * / ^)
    if not _PURE_ARITHMETIC_RE.match(expr_str):
        return None

    # Must contain at least one digit and one operator
    if not any(c.isdigit() for c in expr_str) or not any(op in expr_str for op in "+-*/^"):
        return None

    try:
        import sympy
        py_expr = expr_str.replace("^", "**")
        result = sympy.sympify(py_expr)
        if result.is_real:
            float_val = float(result)
            if float_val == int(float_val):
                return str(int(float_val))
            return f"{float_val:.4g}".rstrip("0").rstrip(".")
    except Exception:
        pass

    return None
