"""
TokenForge — Safe Deterministic Local Solvers (Tier 0)
Provides zero-token, 100% accurate solutions for strictly unambiguous tasks.
Every solver enforces fail-closed discipline: returns None on any ambiguity.
"""

import re
from typing import Optional

_PURE_ARITHMETIC_RE = re.compile(r"^[0-9\s\.\+\-\*\/\(\)\^]+$")
_PERCENT_OF_RE = re.compile(
    r"^(?:what\s+is|calculate|compute)?\s*(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)\s*\??$",
    re.IGNORECASE,
)
_STRING_REVERSE_RE = re.compile(
    r"^reverse\s+(?:the\s+)?(?:string|word|text)\s+[\"'](.*?)[\"']\s*\??$",
    re.IGNORECASE,
)
_STRING_UPPER_RE = re.compile(
    r"^convert\s+[\"'](.*?)[\"']\s+to\s+uppercase\s*\??$",
    re.IGNORECASE,
)
_STRING_LOWER_RE = re.compile(
    r"^convert\s+[\"'](.*?)[\"']\s+to\s+lowercase\s*\??$",
    re.IGNORECASE,
)
_STRING_LEN_RE = re.compile(
    r"^(?:count\s+(?:the\s+)?number\s+of\s+characters\s+in|length\s+of)\s+[\"'](.*?)[\"']\s*\??$",
    re.IGNORECASE,
)


def _format_number(val: float) -> str:
    """Format numeric result cleanly as int or compact float."""
    if val == int(val):
        return str(int(val))
    formatted = f"{val:.6g}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def solve_math_expression(prompt: str) -> Optional[str]:
    """
    Attempts to evaluate strictly pure arithmetic expressions locally using SymPy.
    Returns None on any ambiguity or non-arithmetic prompt.
    """
    text = prompt.strip()
    if not text:
        return None

    # Check for percentage expression e.g. "Calculate 15% of 240"
    m_pct = _PERCENT_OF_RE.match(text)
    if m_pct:
        pct = float(m_pct.group(1))
        num = float(m_pct.group(2))
        return _format_number((pct / 100.0) * num)

    # Strip simple command prefixes
    expr_str = text
    lower = text.lower()
    for prefix in ("what is ", "calculate ", "compute ", "solve equation: ", "solve: ", "solve "):
        if lower.startswith(prefix):
            expr_str = text[len(prefix):].strip()
            lower = expr_str.lower()

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
            return _format_number(float_val)
    except Exception:
        pass

    return None


def solve_string_operation(prompt: str) -> Optional[str]:
    """
    Attempts to evaluate strictly unambiguous string manipulation tasks.
    Returns None on any ambiguity.
    """
    text = prompt.strip()
    if not text:
        return None

    m_rev = _STRING_REVERSE_RE.match(text)
    if m_rev:
        return m_rev.group(1)[::-1]

    m_upper = _STRING_UPPER_RE.match(text)
    if m_upper:
        return m_upper.group(1).upper()

    m_lower = _STRING_LOWER_RE.match(text)
    if m_lower:
        return m_lower.group(1).lower()

    m_len = _STRING_LEN_RE.match(text)
    if m_len:
        return str(len(m_len.group(1)))

    return None


def solve(prompt: str) -> Optional[str]:
    """
    Main entry point for Tier 0 Deterministic Solvers.
    Executes fail-closed deterministic solvers in order.
    """
    math_ans = solve_math_expression(prompt)
    if math_ans is not None:
        return math_ans

    str_ans = solve_string_operation(prompt)
    if str_ans is not None:
        return str_ans

    return None
