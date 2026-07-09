import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
from router import classify_and_route

class ResponseCache:
    """Thread-safe in-memory cache to deduplicate identical evaluation prompts for zero token cost."""
    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, val: str) -> None:
        with self._lock:
            self._cache[key] = val

def process_single_task(task: Dict[str, Any], cache: ResponseCache) -> Dict[str, str] | None:
    task_id = task.get("task_id")
    if not task_id:
        return None
    prompt = str(task.get("prompt", ""))

    # Check response cache
    cached_ans = cache.get(prompt)
    if cached_ans is not None:
        return {"task_id": str(task_id), "answer": cached_ans}

    # Route and obtain answer safely
    try:
        answer = classify_and_route(prompt)
        if not answer or not isinstance(answer, str):
            answer = "Unable to generate response for prompt."
    except Exception as e:
        answer = f"Error processing prompt: {e}"

    cache.set(prompt, answer)
    return {"task_id": str(task_id), "answer": answer}

def main() -> None:
    # 1. Define paths (container standard vs local fallback)
    input_path = "/input/tasks.json" if os.path.exists("/input/tasks.json") else "input/tasks.json"
    output_path = "/output/results.json" if os.path.exists("/output") else "output/results.json"

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"Error reading input JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(tasks, list):
        print("Error: Input JSON must be a list of tasks.", file=sys.stderr)
        sys.exit(1)

    cache = ResponseCache()
    results: List[Dict[str, str]] = []

    # 2. Parallel 8-Worker Execution
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_task, t, cache) for t in tasks]
        for f in futures:
            try:
                res = f.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"Worker execution exception: {e}", file=sys.stderr)

    # 3. Write results and exit 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error writing output JSON: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
