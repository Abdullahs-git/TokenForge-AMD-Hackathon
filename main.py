import json
import os
import sys
from router import classify_and_route

def main() -> None:
    # Load required environment variables
    fireworks_api_key = os.environ.get("FIREWORKS_API_KEY", "")
    fireworks_base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    allowed_models = os.environ.get("ALLOWED_MODELS", "")

    # Define paths (fallback to local directory if container paths do not exist)
    input_path = "/input/tasks.json" if os.path.exists("/input/tasks.json") else "input/tasks.json"
    output_path = "/output/results.json" if os.path.exists("/output") else "output/results.json"

    # Read input tasks
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing input JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(tasks, list):
        print("Error: Input JSON must be an array/list of tasks.", file=sys.stderr)
        sys.exit(1)

    results = []

    # Process each task
    for task in tasks:
        task_id = task.get("task_id")
        prompt = task.get("prompt", "")
        
        if not task_id:
            continue

        # Route prompt and obtain answer
        answer = classify_and_route(prompt)
        results.append({
            "task_id": task_id,
            "answer": answer
        })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write output results
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error writing output JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Exit cleanly
    sys.exit(0)

if __name__ == "__main__":
    main()
