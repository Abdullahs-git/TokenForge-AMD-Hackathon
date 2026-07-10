"""
TokenForge v6.0 — Main Entry Point

Container contract:
- Read:  /input/tasks.json  → [{"task_id": "...", "prompt": "..."}, ...]
- Write: /output/results.json → [{"task_id": "...", "answer": "..."}, ...]
- Exit code 0 on success
- Maximum runtime: 10 minutes

Pipeline:
1. Load tasks from JSON
2. Process all tasks in parallel (8-worker ThreadPoolExecutor)
3. Schema-validate every result (task_id: str, answer: str, both non-empty)
4. Write validated results and exit 0
"""

import json
import os
import sys
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

from router import classify_and_route

# Configure logging to stderr (keep stdout clean)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("tokenforge")

# Maximum runtime safety margin (harness kills at 10 min)
MAX_RUNTIME_SECONDS = 570  # 9.5 minutes


class ResponseCache:
    """Thread-safe in-memory cache to deduplicate identical evaluation prompts for zero token cost."""

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        with self._lock:
            # Normalize key for better cache hits
            normalized = key.strip().lower()
            result = self._cache.get(normalized)
            if result is not None:
                self.hits += 1
            else:
                self.misses += 1
            return result

    def set(self, key: str, val: str) -> None:
        with self._lock:
            normalized = key.strip().lower()
            self._cache[normalized] = val


def validate_result(result: Dict[str, str]) -> Dict[str, str]:
    """
    Schema validation: ensure task_id and answer are both non-empty strings.
    Repairs invalid entries in-place.
    """
    if not result:
        return {"task_id": "unknown", "answer": "Unable to determine answer."}

    task_id = result.get("task_id", "")
    answer = result.get("answer", "")

    # Ensure task_id is a non-empty string
    if not isinstance(task_id, str) or not task_id.strip():
        task_id = "unknown"

    # Ensure answer is a non-empty string
    if not isinstance(answer, str) or not answer.strip():
        answer = "Unable to determine answer."

    return {"task_id": str(task_id).strip(), "answer": str(answer).strip()}


def process_single_task(task: Dict[str, Any], cache: ResponseCache) -> Dict[str, str] | None:
    """Process one task through the full pipeline with caching."""
    task_id = task.get("task_id") or task.get("id") or task.get("taskId") or ""
    if not task_id:
        return None

    prompt = str(task.get("prompt") or task.get("question") or task.get("input") or task.get("text") or "")
    if not prompt.strip():
        return validate_result({"task_id": str(task_id), "answer": ""})

    # Check response cache
    cached_ans = cache.get(prompt)
    if cached_ans is not None:
        logger.info("Task %s: cache HIT (0 tokens)", task_id)
        return validate_result({"task_id": str(task_id), "answer": cached_ans})

    # Route and obtain answer safely
    try:
        answer = classify_and_route(prompt)
        if not answer or not isinstance(answer, str):
            answer = "Unable to determine answer."
    except Exception as e:
        logger.error("Task %s: exception during routing: %s", task_id, e)
        answer = "Unable to determine answer."

    cache.set(prompt, answer)
    return validate_result({"task_id": str(task_id), "answer": answer})


def main() -> None:
    start_time = time.monotonic()
    logger.info("TokenForge v6.0 starting...")

    # 1. Define paths (container standard vs local fallback)
    input_path = "/input/tasks.json" if os.path.exists("/input/tasks.json") else "input/tasks.json"
    output_path = "/output/results.json" if os.path.exists("/output") else "output/results.json"

    if not os.path.exists(input_path):
        logger.error("Input file %s not found.", input_path)
        sys.exit(1)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        logger.error("Error reading input JSON: %s", e)
        sys.exit(1)

    if not isinstance(tasks, list):
        logger.error("Input JSON must be a list of tasks.")
        sys.exit(1)

    logger.info("Loaded %d tasks from %s", len(tasks), input_path)

    cache = ResponseCache()
    results: List[Dict[str, str]] = []
    seen_task_ids: set = set()

    deadline = start_time + MAX_RUNTIME_SECONDS

    # 2. Parallel 8-Worker Execution
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_task, t, cache) for t in tasks]
        for i, f in enumerate(futures):
            try:
                # Ensure we leave enough time to write results.json
                timeout = max(1.0, deadline - time.monotonic())
                res = f.result(timeout=timeout)
                if res:
                    results.append(res)
                    seen_task_ids.add(res["task_id"])
            except Exception as e:
                logger.error("Worker execution exception for task %d: %s", i, e)
                # Ensure we still have an entry for this task
                task_id = tasks[i].get("task_id", "") if i < len(tasks) else ""
                if task_id and task_id not in seen_task_ids:
                    results.append(validate_result({"task_id": str(task_id), "answer": ""}))
                    seen_task_ids.add(str(task_id))

    # 3. Guarantee every input task_id has a result (no missing entries)
    for task in tasks:
        task_id = str(task.get("task_id") or task.get("id") or task.get("taskId") or "")
        if task_id and task_id not in seen_task_ids:
            logger.warning("Task %s missing from results, adding fallback", task_id)
            results.append(validate_result({"task_id": task_id, "answer": ""}))
            seen_task_ids.add(task_id)

    # 4. Final schema validation pass
    validated_results = [validate_result(r) for r in results]

    # 5. Write results and exit 0
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "output", exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(validated_results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Error writing output JSON: %s", e)
        sys.exit(1)

    elapsed = time.monotonic() - start_time
    logger.info(
        "Done: %d tasks processed in %.1fs | Cache hits: %d/%d",
        len(validated_results), elapsed, cache.hits, cache.hits + cache.misses,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
