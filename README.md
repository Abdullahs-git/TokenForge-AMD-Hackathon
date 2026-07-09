<div align="center">
  <img src="assets/no-bg-logo.png" alt="TokenForge Logo" width="260" />
  <h1>TokenForge</h1>
  <p><strong>Enterprise AI Cost Governor & Hybrid Token-Efficient Routing Agent</strong></p>
  <p><em>AMD Developer Hackathon: ACT II — Track 1: General-Purpose AI Agent</em></p>

  [![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Docker linux/amd64](https://img.shields.io/badge/Docker-linux%2Famd64-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
  [![Fireworks AI](https://img.shields.io/badge/Fireworks_AI-Official_Proxy-FF6B6B?style=for-the-badge)](https://fireworks.ai/)
  [![Zero Token Local Solvers](https://img.shields.io/badge/Deterministic_Solvers-%240.00_Token_Cost-00C853?style=for-the-badge)](#tier-0-zero-token-deterministic-local-solvers-000-cost)
  [![Google Gemma Prize Eligible](https://img.shields.io/badge/Google_Gemma-Track_1_Bonus_Eligible-4285F4?style=for-the-badge&logo=google&logoColor=white)](#-gemma-bonus-prize-eligibility-1000-track-1-award)
  [![CI/CD](https://github.com/Abdullahs-git/TokenForge-AMD-Hackathon/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdullahs-git/TokenForge-AMD-Hackathon/actions/workflows/ci.yml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 🚀 Executive Overview

**TokenForge** is an enterprise-grade AI Cost Governor and Hybrid Routing Engine engineered specifically to win **Track 1 of the AMD Developer Hackathon: ACT II**.

Designed strictly around the official headless evaluation container specification, TokenForge employs an **Accuracy-First Strategy** to clear the **80% accuracy gate (16/19 tasks)** and then minimize token consumption:

1. **Accuracy Gate (80%):** Routes all complex reasoning (sentiment, NER, logical, factual, summarization, code) to the `strong` tier of Fireworks AI cloud models via `FIREWORKS_BASE_URL`, with category-tailored system prompts optimized for LLM-Judge scoring.
2. **Token Efficiency Leaderboard:** Intercepts pure arithmetic locally via **SymPy** ($0.00 tokens), deduplicates identical prompts via **ResponseCache** ($0.00 tokens), and enforces strict per-category `max_tokens` ceilings to prevent verbose output inflation.

---

## 🏆 Gemma Bonus Prize Eligibility ($1,000 Track 1 Award)

**TokenForge is architected to compete for the Best Use of Google DeepMind Gemma Models via Fireworks AI ($1,000 Track 1 Bonus Prize):**
- **Prioritized Gemma Tiering:** Our `FireworksModelTierer` engine automatically inspects the runtime `ALLOWED_MODELS` environment variable and prioritizes open-weight **Google Gemma models** (`gemma`, `gemma2-9b-it`) as our primary cost-and-token-efficient cloud tier.
- **On-Demand Compatible:** Per organizer guidance, Gemma is on-demand via Fireworks. Our tierer activates Gemma routing only when it appears in `ALLOWED_MODELS`, with safe fallback to other available models.

---

## 🛠️ Tools & Technologies Used

| Technology / Tool | Version / Library | Purpose in TokenForge Architecture | Cost & Token Impact |
| :--- | :--- | :--- | :--- |
| **Python** | `3.11-slim` | Core headless runtime engine built for containerized efficiency | Lightweight `< 500 MB` footprint |
| **SymPy** | `1.14.0` | **Deterministic Algebraic Solver:** Parses natural language math prompts and solves arithmetic / algebraic equations locally | **$0.00 Token Spend (0 API Calls)** |
| **OpenAI Python SDK** | `2.44.0` | **Fireworks AI Proxy Client:** Routes external requests exclusively via `FIREWORKS_BASE_URL` with automatic multi-model retry fallback | Optimal dynamic tiering |
| **Concurrent Futures** | `ThreadPoolExecutor` | **8-Worker Parallel Concurrency:** Evaluates batch input tasks simultaneously | Sub-5 second total execution |
| **Docker Buildx** | `linux/amd64` | **Multi-Platform Container Builder:** Compiles compliant headless judging VM container | 100% judging proxy compliance |
| **GitHub Actions** | CI/CD Pipeline | **Automated Testing & Deployment:** Runs 8-category classifier tests, JSON schema validation, and auto-pushes Docker images on every commit | Continuous quality assurance |

---

## 🔥 Architecture Visual Explained

TokenForge employs an **Accuracy-First 2-Tier Cost Governor Pipeline**. Every incoming prompt flows through an in-memory cache and a high-precision regex classifier before being routed either to a **Zero-Token Local Solver (Tier 0)** for pure arithmetic or to the **Fireworks AI Cloud (Tier 1)** for all other categories:

```
                       [ Input Tasks: /input/tasks.json ]
                                         │
                              ┌──────────┴──────────┐
                              │  8-Worker Parallel   │
                              │    ThreadPoolExecutor │
                              └──────────┬──────────┘
                                         │
                                         ▼
                             [ In-Memory ResponseCache ]
                      (Instant cache hit = $0 cost, 0 tokens, <1ms)
                                         │
                                         ▼
                     [ Comprehensive 8-Category Classifier ]
              (code_debug, code_gen, sentiment, ner, summarization,
                         logical, math, factual)
                                         │
          ┌──────────────────────────────┴──────────────────────────────┐
          ▼                                                             ▼
   [ Tier 0: Local Solver ]                                [ Tier 1: Dynamic Cloud Tiering ]
   (Pure Arithmetic Only)                                    (Fireworks AI via FIREWORKS_BASE_URL)
          │                                                             │
          └─► SymPy Algebra ($0 Tokens)                                 ├─► 'strong' Tier (Factual, Math, Sentiment,
                                                                        │    NER, Summarization, Logical)
                                                                        └─► 'code' Tier (Code Gen, Code Debug)
          │                                                             │
          └──────────────────────────────┬──────────────────────────────┘
                                         │
                                         ▼
                     [ Output Results: /output/results.json ]
                                   (Exit code 0)
```

### How the Pipeline Works:
1. **Input Ingestion & Parallel Dispatch:** On container startup, `main.py` reads `/input/tasks.json` and dispatches all tasks concurrently across an **8-worker thread pool**.
2. **Response Cache:** Each prompt is checked against the thread-safe `ResponseCache`. Identical prompts return cached answers instantly ($0.00 cost, 0 tokens).
3. **8-Category Regex Classifier:** `router.py` evaluates the prompt against a comprehensive regex engine covering all 8 official hackathon categories: `code_debug`, `code_gen`, `sentiment`, `ner`, `summarization`, `logical`, `math`, and `factual`.
4. **Tier 0 — Deterministic Math Solver (Accuracy-First):**
   - Only **pure arithmetic expressions** (e.g., `144 / 12`) are solved locally by `SymPy` with 100% mathematical precision and $0.00 token cost.
   - Sentiment and NER are routed to cloud for accuracy safety (VADER misclassifies mixed-sentiment reviews; spaCy mislabels certain entity types).
5. **Tier 1 — Dynamic Cloud Tiering (`ALLOWED_MODELS`):**
   - `llm_clients.py` dynamically parses `ALLOWED_MODELS` at runtime into `strong` and `code` tiers.
   - Prioritizes **Google Gemma** models when available in `ALLOWED_MODELS`.
   - Category-tailored system prompts and strict `max_tokens` ceilings (256–512 tokens) prevent verbose output inflation.
   - Automatic multi-model fallback retries on transient API errors.

---

## 📋 Hackathon Category Coverage & Token Budgets

> **Accuracy-First Strategy:** All categories except pure arithmetic route to `strong` tier to maximize LLM-Judge scores and clear the 80% accuracy gate (16/19 tasks).

| # | Official Category | Target Track Coverage | Routing Tier | Output Cap (`max_tokens`) |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Factual knowledge** | Explaining concepts, definitions, how things work | `strong` | `300` |
| **2** | **Mathematical reasoning** | Arithmetic, percentages, word problems, projections | `SymPy` / `strong` | `512` |
| **3** | **Sentiment classification** | Labelling sentiment and justifying classification | `strong` | `256` |
| **4** | **Text summarisation** | Condensing passages to format/length constraint | `strong` | `300` |
| **5** | **Named entity recognition** | Extracting entities (`PERSON`, `ORG`, `LOCATION`, `DATE`) | `strong` | `256` |
| **6** | **Code debugging** | Identifying bugs and providing corrected code | `code` | `512` |
| **7** | **Logical reasoning** | Constraint-based puzzles where conditions must hold | `strong` | `512` |
| **8** | **Code generation** | Writing correct, well-structured functions from spec | `code` | `512` |

---

## 🔄 CI/CD Pipeline (GitHub Actions)

Every push to `main` triggers an automated pipeline (`.github/workflows/ci.yml`):

| Stage | What It Does |
| :--- | :--- |
| **🧪 Module Import Test** | Verifies `main.py`, `router.py`, `llm_clients.py`, `local_solvers.py` all import cleanly on Python 3.11 |
| **🎯 8-Category Classifier Test** | Runs 8 representative prompts through `detect_category()` and asserts all 8 categories classify correctly |
| **🏃 Headless Evaluation Run** | Executes `python main.py` against practice tasks and confirms exit code `0` |
| **📋 JSON Schema Validation** | Asserts `output/results.json` is a valid `[{"task_id": "...", "answer": "..."}]` array |
| **🐳 Docker Build & Push** | Builds `linux/amd64` image and pushes `pandabutt/amd-act2-router:latest` to Docker Hub |

---

## 📖 Step-by-Step Guide: Setup, Execution & Docker Deployment

### Step 1: Clone & Install Dependencies Locally
Ensure Python 3.11+ is installed on your machine:

```bash
git clone https://github.com/Abdullahs-git/TokenForge-AMD-Hackathon.git
cd TokenForge-AMD-Hackathon

# Create virtual environment & install dependencies
python -m venv venv
# On Windows: venv\Scripts\activate | On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

# Download local spaCy NLP model (used as fallback)
python -m spacy download en_core_web_sm
```

### Step 2: Run Local Headless Verification
Test the headless container engine locally against sample practice tasks in `input/tasks.json`:

```bash
python main.py
```

Inspect the generated `output/results.json` to verify zero-token deterministic math answers (`Answer: 12`) and structured JSON compliance.

### Step 3: Build Multi-Platform Docker Container (`linux/amd64`)
Per Track 1 rules, the judging VM runs `linux/amd64`. Build the container image using Docker Buildx:

```bash
docker buildx build --platform linux/amd64 -t pandabutt/amd-act2-router:latest .
```

### Step 4: Run Headless Docker Container Locally
Simulate the judging VM execution harness by mounting local `/input` and `/output` directories and injecting environment variables:

```bash
docker run --rm \
  -e FIREWORKS_API_KEY="your-api-key" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="accounts/fireworks/models/llama-v3p1-8b-instruct,accounts/fireworks/models/llama-v3p1-70b-instruct" \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  pandabutt/amd-act2-router:latest
```

### Step 5: Push Container to Registry for Submission
Push your `linux/amd64` Docker container to Docker Hub:

```bash
docker buildx build --platform linux/amd64 -t pandabutt/amd-act2-router:latest --push .
```

Submit `pandabutt/amd-act2-router:latest` on the AMD Developer Hackathon submission portal!

---

## 🛡️ Hackathon Rule Compliance Summary

| Rule | TokenForge Compliance | Status |
| :--- | :--- | :---: |
| Exit code `0` on success | All workers wrapped in `try...except`; container always writes results and exits `0` | ✅ |
| Read from `/input/tasks.json` | `main.py` reads on startup with container/local path fallback | ✅ |
| Write to `/output/results.json` | Valid JSON array `[{"task_id": "...", "answer": "..."}]` written before exit | ✅ |
| All API calls via `FIREWORKS_BASE_URL` | OpenAI client configured exclusively with `base_url=os.environ["FIREWORKS_BASE_URL"]` | ✅ |
| Only `ALLOWED_MODELS` used | `FireworksModelTierer` parses `os.environ["ALLOWED_MODELS"]` at runtime; no hardcoded IDs | ✅ |
| Maximum runtime < 10 minutes | 8-worker parallel execution completes batch in < 5 seconds | ✅ |
| Image < 10 GB compressed | `3.17 GB` total size | ✅ |
| `linux/amd64` manifest | Built with `docker buildx build --platform linux/amd64` | ✅ |
| No hardcoded/cached answers | All answers generated dynamically per prompt at runtime | ✅ |
