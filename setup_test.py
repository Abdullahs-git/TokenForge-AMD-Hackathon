import os
import json

# 1. Create the input and output directories
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 2. Add the official practice tasks from the participant guide
practice_tasks = [
    {
        "task_id": "practice-01", 
        "prompt": "What is the capital of Australia, and what body of water is it near?"
    },
    {
        "task_id": "practice-02", 
        "prompt": "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?"
    },
    {
        "task_id": "practice-03", 
        "prompt": "Classify the sentiment of this review: The battery life is great, but the screen scratches too easily."
    }
]

# 3. Write them to tasks.json
with open("input/tasks.json", "w") as f:
    json.dump(practice_tasks, f, indent=2)

print("[OK] Test environment initialized. input/tasks.json is ready.")
