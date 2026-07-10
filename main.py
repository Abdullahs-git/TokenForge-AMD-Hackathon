"""
TokenForge — Main Entry Point
Enterprise AI Cost Governor & Hybrid Token-Efficient Routing Agent
Built for AMD Developer Hackathon: ACT II — Track 1
"""

import os
import sys
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tokenforge.main")


def process_task(task: Dict[str, Any], api_key: str, base_url: str, allowed_models: List[str]) -> Dict[str, str]:
    """Process a single evaluation task through the hybrid TokenForge pipeline."""
    task_id = str(task.get("task_id") or task.get("id") or "unknown").strip()
    prompt = str(task.get("prompt") or task.get("question") or task.get("instruction") or "").strip()

    try:
        answer = router.solve_prompt(prompt, api_key, base_url, allowed_models)
        if not answer or not answer.strip():
            answer = "Unable to determine answer."
    except Exception as e:
        logger.error("Error processing task %s: %s", task_id, e)
        answer = "Unable to determine answer."

    return {
        "task_id": task_id,
        "id": task_id,
        "answer": answer.strip(),
    }


def main() -> None:
    start_time = time.monotonic()
    logger.info("TokenForge starting evaluation pipeline...")

    # Load environment configuration
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    allowed_models_str = os.environ.get("ALLOWED_MODELS", "")
    allowed_models = [m.strip() for m in allowed_models_str.split(",") if m.strip()]

    # Paths (container standard /input/tasks.json vs local development)
    input_path = "/input/tasks.json" if os.path.exists("/input/tasks.json") else "input/tasks.json"
    output_path = "/output/results.json" if os.path.exists("/output") else "output/results.json"

    if not os.path.exists(input_path):
        logger.error("Input file %s not found.", input_path)
        sys.exit(1)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        logger.error("Failed to read input JSON %s: %s", input_path, e)
        sys.exit(1)

    # Unwrap dict containers e.g. {"tasks": [...]} or {"results": [...]}
    if isinstance(tasks, dict):
        for key in ("tasks", "data", "results", "items"):
            if key in tasks and isinstance(tasks[key], list):
                tasks = tasks[key]
                break

    if not isinstance(tasks, list):
        logger.error("Input JSON must contain a list of task objects.")
        sys.exit(1)

    logger.info("Processing %d evaluation tasks...", len(tasks))

    results: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_task, task, api_key, base_url, allowed_models)
            for task in tasks
        ]
        for future in futures:
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                logger.error("Worker thread exception: %s", e)

    # Write output JSON safely
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(
            "Saved %d results to %s in %.2fs",
            len(results),
            output_path,
            time.monotonic() - start_time,
        )
    except Exception as e:
        logger.error("Failed to save results JSON: %s", e)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
