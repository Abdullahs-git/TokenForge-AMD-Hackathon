<div align="center">
  <img src="assets/no-bg-logo.png" alt="TokenForge Logo" width="260" />
  <h1>TokenForge — Hybrid Token-Efficient Routing Agent</h1>
  <p><strong>AMD Developer Hackathon ACT II — Track 1</strong></p>

  [![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Docker linux/amd64](https://img.shields.io/badge/Docker-linux%2Famd64-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 🚀 Architectural Overview

TokenForge is a task-aware hybrid routing agent designed to minimize token consumption while maintaining strict accuracy across varied evaluation benchmarks.

### Core Architecture

1. **Tier 0 Fail-Closed Deterministic Solvers (`local_solvers.py`):**
   - Pure numerical arithmetic expressions are solved locally via zero-token deterministic evaluation (`SymPy`).
   - Unambiguous deterministic operations execute locally at zero API cost.
   - Strict fail-closed discipline: returns `None` on any ambiguity or potential edge case to ensure zero false-positive local guesses.

2. **Task Classification (`router.classify_task`):**
   - Automatically categorizes incoming prompts into `code`, `classification`, `short_fact`, and `general` reasoning.
   - Assigns tailored, strict format system prompts per task category to prevent verbose preambles or post-completion conversational fluff.

3. **Dynamic Generic Model Scoring (`router.select_best_model`):**
   - Parses parameter sizes and capability tags (`coder`, `instruct`, `chat`) directly from the runtime `ALLOWED_MODELS` environment variable.
   - Ranks models dynamically without relying on brittle hardcoded model name strings.

4. **Adaptive Token Ceilings & Sanitization (`router.sanitize_output`):**
   - Applies strict category-specific `max_tokens` ceilings (`64` for classification, `80` for short facts, `600` for code).
   - Automatically strips chain-of-thought `<think>...</think>` tags and conversational prefixes/suffixes.

---

## 📦 Container Contract & Evaluation Schema

The container strictly adheres to the AMD evaluation pipeline specifications:
- **Platform:** Explicitly built for `linux/amd64`.
- **Input:** Reads `/input/tasks.json` containing task objects (`task_id` / `id` and `prompt` / `question`).
- **Output:** Writes schema-compliant records to `/output/results.json`:
  ```json
  [
    {
      "task_id": "example_01",
      "id": "example_01",
      "answer": "Direct answer text"
    }
  ]
  ```

---

## 🛠️ Local Verification & Testing

### Run Automated Sanity Suite
```bash
python test_pipeline.py
```

### Run Benchmark Evaluator
```bash
python benchmark_evaluator.py
```

### Build & Verify Docker Container (`linux/amd64`)
```bash
docker buildx build --platform linux/amd64 -t tokenforge:latest .

mkdir -p input output
echo '[{"task_id":"t1","prompt":"What is 144 / 12?"}]' > input/tasks.json

docker run --rm --platform linux/amd64 \
  -v $(pwd)/input:/input -v $(pwd)/output:/output \
  -e ALLOWED_MODELS="accounts/fireworks/models/llama-v3p1-70b-instruct" \
  tokenforge:latest

cat output/results.json
```
