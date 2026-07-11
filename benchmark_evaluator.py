"""
TokenForge Benchmark Evaluator
Evaluates token consumption and tier routing across representative hackathon tasks.
Architecture: Tier 0 Safe Deterministic Solvers -> Task-Classified Routing -> Dynamic Model Selection -> Sanitization
"""

import json
import os
import sys
from typing import Dict, Any, List
import local_solvers
import router


def estimate_tokens(text: str) -> int:
    """Estimate tokens (~3.7 chars per token for typical models)."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.7))


def run_benchmark():
    print("=" * 75)
    print("           TOKENFORGE TASK-AWARE ROUTER EVALUATION SCORECARD           ")
    print("=" * 75)

    tasks_file = os.path.join("input", "tasks.json")
    if not os.path.exists(tasks_file):
        print(f"Error: {tasks_file} not found.")
        sys.exit(1)

    with open(tasks_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    total_raw_input_tokens = 0
    total_est_tokens_spent = 0
    local_solvers_hit = 0

    print(f"\nEvaluating {len(tasks)} benchmark tasks...\n")
    print(f"{'Task ID':<16} {'Category':<16} {'Solver Tier':<18} {'Est. Tokens':<14} {'Status'}")
    print("-" * 80)

    for task in tasks:
        tid = str(task.get("task_id") or task.get("id") or "unknown")
        prompt = str(task.get("prompt") or task.get("question") or "")
        raw_in_tokens = estimate_tokens(prompt)
        total_raw_input_tokens += raw_in_tokens

        # Check safe Tier 0 Deterministic Solver
        local_ans = local_solvers.solve(prompt)

        if local_ans is not None:
            category = "deterministic"
            solver_tier = "Tier 0 (Local)"
            tokens_spent = 0
            local_solvers_hit += 1
            status = "[OK - 0 TOKENS]"
        else:
            category = router.classify_task(prompt)
            cfg = router.TASK_CONFIG.get(category, router.TASK_CONFIG.get("factual", {}))
            solver_tier = "Tier 1 (SOTA)"
            # System prompt tokens + estimated concise output
            sys_tokens = estimate_tokens(cfg["system_prompt"])
            est_out_tokens = min(40, cfg["max_tokens"])
            tokens_spent = raw_in_tokens + sys_tokens + est_out_tokens
            total_est_tokens_spent += tokens_spent
            status = f"[ROUTED: {category.upper()}]"

        print(f"{tid:<16} {category:<16} {solver_tier:<18} {tokens_spent:<14} {status}")

    print("-" * 80)
    print("\n=== TOKENFORGE ARCHITECTURAL SCORECARD ===")
    print(f"Total Benchmark Tasks Evaluated:     {len(tasks)}")
    print(f"Tier 0 Deterministic Solver Hits:    {local_solvers_hit} / {len(tasks)}")
    print(f"Estimated API Tokens Consumed:       {total_est_tokens_spent} tokens")

    avg_tokens = total_est_tokens_spent / len(tasks) if tasks else 0
    print(f"Average Estimated Tokens / Task:     ~{avg_tokens:.1f} tokens")
    print("Architecture: Tier 0 Deterministic Solvers -> Task Classification -> Generic Model Selection -> Sanitization")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark()
