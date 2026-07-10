import re
import sympy
from typing import Optional

_MATH_PATTERN = re.compile(
    r"^(?:what is|calculate|compute|solve|evaluate)?\s*(\d+(?:\.\d+)?)\s*"
    r"([+\-*/^x×÷])\s*(\d+(?:\.\d+)?)[^\d\w]*$",
    re.IGNORECASE,
)

def clean_math_prompt(prompt: str) -> str:
    """Cleans English conversational prefixes/suffixes from algebraic expressions."""
    text = prompt.strip()
    text = re.sub(r'^(what is|calculate|compute|solve for \w+:?|solve:?|evaluate:?)\s*', '', text, flags=re.IGNORECASE)
    text = text.rstrip('?.!')
    return text.strip()

def solve_math(prompt: str) -> Optional[str]:
    """
    Deterministically solves pure simple arithmetic expressions ($0 token cost).
    Returns None if prompt is a complex multi-step word problem requiring LLM reasoning.
    """
    try:
        text = prompt.strip()
        if len(text.split()) > 20:
            return None

        # Try fast regex match for standard binary operations (A op B)
        match = _MATH_PATTERN.search(text)
        if match:
            a_val, op, b_val = float(match.group(1)), match.group(2), float(match.group(3))
            if op == "+":
                res = a_val + b_val
            elif op == "-":
                res = a_val - b_val
            elif op in ("*", "x", "×"):
                res = a_val * b_val
            elif op in ("/", "÷"):
                if b_val == 0:
                    return None
                res = a_val / b_val
            elif op == "^":
                res = a_val ** b_val
            else:
                return None
            ans_str = str(int(res)) if res == int(res) else str(round(res, 6))
            return f"Answer: {ans_str}"

        cleaned = clean_math_prompt(prompt)
        if not cleaned or len(cleaned.split()) > 8:
            return None

        # Clean simple linear equation "a*x + b = c"
        if "=" in cleaned:
            parts = cleaned.split("=")
            if len(parts) == 2:
                lhs = sympy.sympify(parts[0].strip())
                rhs = sympy.sympify(parts[1].strip())
                eq = sympy.Eq(lhs, rhs)
                solutions = sympy.solve(eq)
                if len(solutions) == 1:
                    sol = solutions[0]
                    return f"Answer: {sol}"
        
        return None
    except Exception:
        return None

