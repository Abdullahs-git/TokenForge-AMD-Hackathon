# TokenForge — Fix & Win Track 1 (AMD Developer Hackathon ACT II)

## Context

Repo: https://github.com/Abdullahs-git/TokenForge-AMD-Hackathon
Track: AMD Developer Hackathon ACT II — Track 1: "Hybrid Token-Efficient Routing Agent"
Scoring: **fewest total tokens wins, subject to a minimum accuracy gate**. Submissions are Docker containers that read `/input/tasks.json`, call an LLM via the Fireworks AI OpenAI-compatible endpoint as needed, and write `/output/results.json`.

**Current state (verified on the live leaderboard):** Two of my submissions from this repo are failing before they even get scored:
- One shows `RUNTIME_ERROR`: *"container was pulled but crashed during evaluation. Check your entrypoint/CMD and that it runs on linux/amd64."*
- One shows `INVALID_RESULTS_SCHEMA`.

Only 2 submissions total have cleared the accuracy gate on the whole leaderboard so far (both ~84.2% accuracy, ~5,200–5,600 total tokens). This tells us the real per-run token budget is much larger than this repo's README assumes (README claims a ~1,285-token target across "19 tasks" — that number is not grounded in the real eval set and should not be trusted).

**Priority order — fix in this exact sequence, verifying each step before moving to the next:**

---

## Step 1 — Diagnose and fix why the container doesn't run at all

This is the actual blocker, above all else. Nothing else matters until this is fixed.

1. Read the existing `Dockerfile`, `main.py`, `router.py`, `local_solvers.py`, `requirements.txt`, and anything in `/input` (there's an `input/` folder in the repo — check if it holds a sample `tasks.json`; if so, use it as the ground-truth shape of what this container will actually receive).
2. Also check `benchmark_evaluator.py` and `setup_test.py` — these look like local testing scaffolding the previous author wrote; read them for clues about what output shape was expected, then decide whether to keep, fix, or delete them (see cleanup section below).
3. **Most likely root cause of `RUNTIME_ERROR`:** the image was built and pushed without targeting `linux/amd64` (e.g. built on an Apple Silicon Mac), so it fails to start at all on AMD's x86-64 evaluation infrastructure — this happens at the container-runtime level, before any Python code executes, which is consistent with there being no useful stack trace.
   - Rebuild and push with the platform explicitly forced:
     ```bash
     docker buildx build --platform linux/amd64 -t <registry>/<image>:latest --push .
     ```
   - Verify locally under the same emulated platform before resubmitting:
     ```bash
     mkdir -p input output
     echo '[{"task_id":"t1","prompt":"What is 12*8?"}]' > input/tasks.json
     docker run --rm --platform linux/amd64 \
       -v $(pwd)/input:/input -v $(pwd)/output:/output \
       -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
       -e FIREWORKS_BASE_URL=$FIREWORKS_BASE_URL \
       -e ALLOWED_MODELS=$ALLOWED_MODELS \
       <registry>/<image>:latest
     cat output/results.json
     ```
   - If it crashes even locally under `--platform linux/amd64` emulation, capture the actual Python traceback (add `set -x` / ensure `PYTHONUNBUFFERED=1` is set, which it already is) and fix the real code bug — don't assume it's only the platform issue.
4. **`INVALID_RESULTS_SCHEMA` diagnosis:** the container currently writes:
   ```json
   [{"task_id": "...", "answer": "..."}, ...]
   ```
   Search this repo (README, any docs folder, `.github/workflows`, comments in `benchmark_evaluator.py`) and the AMD Developer Hackathon ACT II lablab.ai submission docs / Discord for the **literal expected field names** of `results.json` (candidates to check for: `task_id` vs `id`, `answer` vs `response`/`prediction`/`output`, whether the top level should be a bare array vs `{"results": [...]}`). If you cannot find an explicit spec, treat the field names currently in `main.py` as unverified — flag this clearly in your final report to me rather than silently guessing, and if the repo's own `input/tasks.json` sample or `benchmark_evaluator.py` implies a specific schema, match that exactly.
5. Do not proceed past this step until a local Docker run under `--platform linux/amd64` produces a valid, schema-correct `output/results.json`.

---

## Step 2 — Replace router.py and local_solvers.py with the task-aware architecture below

Do not hardcode specific model name strings (e.g. `"kimi-k2.7-code"`, `"gemma-4-31b-it"`) — we don't know what `ALLOWED_MODELS` will actually contain at eval time, and silently falling back to `allowed_models[0]` when no hardcoded string matches is a major accuracy risk. Use **generic scoring** instead: parse parameter size (e.g. `70b`, `34b`) and tags (`instruct`, `chat`, `coder`, `mini`, `distill`) out of whatever model strings are actually provided, and rank dynamically.

Implement:

### `local_solvers.py` — Tier 0, zero-token deterministic solvers
- Pure arithmetic (existing capability — keep and keep it strict).
- Any other **fully deterministic, unambiguous** pattern you can safely detect (date differences in ISO format, quoted-string reverse/case/count operations, etc.).
- **Fail-closed discipline is mandatory**: every solver function must return `None` on any ambiguity whatsoever. A wrong zero-token answer costs an accuracy-gate failure, which is far worse than just paying for the API call. Never guess.

### `router.py` — task classification + adaptive token ceilings + generic model scoring
- `classify_task(prompt) -> str`: categorize each prompt into at least `code`, `classification` (sentiment/NER/labeling/true-false/multiple-choice), `short_fact` (who/what/when/where one-liners), and `general` (everything else, reasoning/summarization/open-ended).
- Give each category its own `max_tokens` ceiling and its own system prompt that states the **exact expected output shape** (e.g. classification: "respond with only the exact label, no explanation, no punctuation"; code: "output only the complete code"). A soft "be concise" instruction is not sufficient — the prompt needs to state the literal output format per category, because that's what actually constrains completion length, not just hoping the model complies.
- `select_best_model(prompt, allowed_models, task_type) -> str`: generic scoring function — extract size/tags from the actual `allowed_models` list at runtime, rank, no hardcoded specific model names.
- `sanitize_output(raw_text, task_type) -> str`: strip `<think>...</think>` blocks, strip a broader set of fluff prefixes/suffixes than just 4 fixed patterns, and apply task-specific post-processing (force single-line + strip punctuation for `classification`, keep only first sentence if the model over-explains for `short_fact`).
- `solve_prompt(...)`: Tier 0 local solver first (0 tokens) → task classification → generic model selection → task-tuned system prompt + `max_tokens` → sanitize. Keep the existing retry/backoff logic (3 attempts, exponential backoff) — that's sound.
- Keep `main.py`'s function signatures unchanged so it remains a drop-in replacement — don't require changes to `main.py` unless the schema fix in Step 1 requires it.

Write unit-style sanity checks (a small script or `pytest` file) that:
- Feeds each task category a sample prompt and asserts the classifier picks the right category.
- Feeds a fake `allowed_models` list and asserts the scorer picks a sensible model per category.
- Confirms `local_solvers` never returns a wrong answer on any of your test prompts, and returns `None` (not a guess) on ambiguous ones.

---

## Step 3 — Clean up repo cruft that could confuse evaluation or graders

- `run.sh` currently starts `ollama serve` — this is dead code from an earlier architecture and is not referenced by the Dockerfile's `CMD`. Either delete it or update it to match the real entrypoint; leaving contradictory unused files in a hackathon submission repo is a credibility/judging risk if a human reviews the repo.
- Check `cleanup.ps1`, `test_v7_solvers.py`, `setup_test.py`, `benchmark_evaluator.py` for the same kind of staleness (version-numbered filenames like `test_v7_solvers.py` next to a `router.py` that's now on v11 logic is a signal of drift — reconcile or remove).
- Update `README.md` to remove the unverified "100% accuracy" and "~1,285 tokens" claims. Replace with an honest description of the actual architecture (Tier 0 local solvers → task-classified routing → generic model scoring → sanitization) and, once you have a real scored submission, the real measured token/accuracy numbers from the leaderboard. An honest README that matches working code is worth more to judges than an aspirational one that doesn't match a crashing container.
- Make sure `requirements.txt` pins reasonably safe versions if you're not already (bare `openai` / `sympy` with no version floor is a minor future-breakage risk, low priority compared to Steps 1–2).

---

## Step 4 — Re-verify end-to-end before resubmitting

1. Local Docker run (Step 1's command) with a small hand-written `tasks.json` covering: one arithmetic task, one classification/sentiment task, one code-generation task, one factual QA task — confirm each gets routed to the right category, gets a sensible model, and produces a clean, schema-correct answer.
2. Confirm `results.json` output matches whatever schema you settled on in Step 1.4.
3. Push the image, resubmit on lablab.ai, and check the dashboard for the actual error state (or an actual accuracy % + token count) rather than assuming success.
4. Report back: paste the exact new leaderboard status (accuracy %, token count, or the exact new error text if it still fails) so we can iterate — don't guess at what fixed it without seeing the real result.

---

## Constraints / things not to do

- Don't hardcode specific Fireworks model names anywhere — always derive behavior from the real `ALLOWED_MODELS` env var at runtime.
- Don't claim accuracy or token numbers in the README that haven't been measured against the actual leaderboard.
- Don't add local-solver logic that could produce a wrong answer with any confidence less than certain — every Tier-0 solver must fail closed (`return None`) rather than guess.
- Don't remove the retry/backoff logic in the API call path — it's a legitimate reliability feature, not a token-waste issue.
- Keep changes minimal and legible — this is a hackathon judged partly on code quality; don't introduce unnecessary abstraction layers beyond what's described above.
