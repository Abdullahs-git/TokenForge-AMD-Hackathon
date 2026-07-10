"""
TokenForge v8.0 — Zero-Token Local Solvers (Tier 0)
Deterministic execution for pure mathematical expressions ($0.00 API cost, 0 tokens).
"""

import re
from typing import Optional

# Safe mathematical symbols pattern
_SAFE_MATH_RE = re.compile(r"^[0-9\s\.\+\-\*\/\(\)\^\%]+$")


def solve_math_expression(prompt: str) -> Optional[str]:
    """
    Attempts to evaluate pure arithmetic mathematical expressions locally using SymPy.
    Returns the exact numerical answer if successful, or None if it requires NLP reasoning.
    """
    text = prompt.strip()
    if not text:
        return None

    # Check for simple "What is X?" or pure expression "X"
    expr_str = text
    lower = text.lower()
    if lower.startswith("what is "):
        expr_str = text[8:].strip()
    elif lower.startswith("calculate "):
        expr_str = text[10:].strip()
    elif lower.startswith("compute "):
        expr_str = text[8:].strip()

    expr_str = expr_str.rstrip("?").strip()

    # Verify safe arithmetic characters only
    if not _SAFE_MATH_RE.match(expr_str):
        return None

    # Must contain at least one digit and one operator
    if not any(c.isdigit() for c in expr_str) or not any(op in expr_str for op in "+-*/^%"):
        return None

    try:
        import sympy
        # Replace ^ with ** for exponentiation
        py_expr = expr_str.replace("^", "**")
        result = sympy.sympify(py_expr)
        if result.is_real:
            # Format cleanly as integer or decimal
            float_val = float(result)
            if float_val == int(float_val):
                return str(int(float_val))
            return f"{float_val:.4g}".rstrip("0").rstrip(".")
    except Exception:
        pass

    return None
