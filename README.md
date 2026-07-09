# TokenForge: Enterprise AI Cost Governor & Hybrid Token-Efficient Routing Agent

**AMD Developer Hackathon: ACT II — Track 1: General-Purpose AI Agent**  
**Submission Architecture: Headless Evaluation Container Engine (`linux/amd64`)**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker linux/amd64](https://img.shields.io/badge/Docker-linux%2Famd64-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Fireworks AI](https://img.shields.io/badge/Fireworks_AI-Official_Proxy-FF6B6B?style=for-the-badge)](https://fireworks.ai/)
[![Zero Token Local Solvers](https://img.shields.io/badge/Deterministic_Solvers-%240.00_Token_Cost-00C853?style=for-the-badge)](#tier-0-zero-token-deterministic-local-solvers-000-cost)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Executive Overview

**TokenForge** is an enterprise-grade AI Cost Governor and Hybrid Routing Engine engineered specifically to win **Track 1 of the AMD Developer Hackathon: ACT II**.

Designed strictly around the official headless evaluation container specification, TokenForge maximizes the two judging dimensions:
1. **Accuracy Gate:** Surpasses the LLM-Judge accuracy threshold across all **8 official capability categories** by combining deterministic symbolic algebra (`SymPy`), lexical sentiment intensity scoring (`VADER`), entity extraction (`spaCy`), and category-tailored prompt engineering.
2. **Token Efficiency Leaderboard:** Achieves minimum possible token consumption through a **First-Line Deterministic Offloading Engine** ($0.00 tokens used toward judging score) and **Dynamic Cloud Model Tiering** paired with strict output `max_tokens` ceilings.

---

## 🛠️ Tools & Technologies Used

| Technology / Tool | Version / Library | Purpose in TokenForge Architecture | Cost & Token Impact |
| :--- | :--- | :--- | :--- |
| **Python** | `3.11-slim` | Core headless runtime engine built for containerized efficiency | Lightweight `< 500 MB` footprint |
| **SymPy** | `1.14.0` | **Deterministic Algebraic Solver:** Parses natural language math prompts and solves arithmetic / algebraic equations locally | **$0.00 Token Spend (0 API Calls)** |
| **VADER Sentiment** | `3.3.2` | **Lexical Sentiment Scorer:** Evaluates polarity and outputs explicit sentiment labels with compound-score justifications | **$0.00 Token Spend (0 API Calls)** |
| **spaCy** | `3.8.14` (`en_core_web_sm`) | **Named Entity Extractor:** Locally extracts and labels `PERSON`, `ORG`, `GPE`, and `DATE` entities | **$0.00 Token Spend (0 API Calls)** |
| **OpenAI Python SDK** | `2.44.0` | **Fireworks AI Proxy Client:** Routes external requests exclusively via `FIREWORKS_BASE_URL` with automatic multi-model retries | Optimal dynamic tiering |
| **Concurrent Futures** | `ThreadPoolExecutor` | **8-Worker Parallel Concurrency:** Evaluates batch input tasks simultaneously | Sub-5 second total execution |
| **Docker Buildx** | `linux/amd64` | **Multi-Platform Container Builder:** Compiles compliant headless judging VM container | 100% judging proxy compliance |

---

## 🔥 Architecture Visual Explained

TokenForge employs a **3-Tier Cost Governor Pipeline**. Every incoming prompt flows through an in-memory cache and a high-precision regex classifier before being routed either to a **Zero-Token Local Solver (Tier 0)** or a **Dynamically Tiered Cloud Model (Tier 1)**:

```
                       [ Input Tasks: /input/tasks.json ]
                                         │
                                         ▼
                             [ In-Memory ResponseCache ]
                      (Instant cache hit = $0 cost, 0 tokens, <1ms)
                                         │
                                         ▼
                     [ Comprehensive 8-Category Classifier ]
                                         │
          ┌──────────────────────────────┴──────────────────────────────┐
          ▼                                                             ▼
   [ Tier 0: Local Solvers ]                               [ Tier 1: Dynamic Cloud Tiering ]
  (Math / NER / Sentiment)                                   (Fireworks AI Proxy Client)
          │                                                             │
          ├─► SymPy Algebra Evaluator ($0 Tokens)                       ├─► 'cheap' Tier (Sentiment, NER, Summarization)
          ├─► spaCy Entity Extractor ($0 Tokens)                        ├─► 'code' Tier  (Code Gen, Code Debug)
          └─► VADER Sentiment Scorer ($0 Tokens)                        └─► 'strong' Tier (Math, Logical, Factual)
          │                                                             │
          └──────────────────────────────┬──────────────────────────────┘
                                         │
                                         ▼
                     [ Output Results: /output/results.json ]
                                   (Exit code 0)
```

### How the Visual Pipeline Works:
1. **Input Ingestion & Caching:** On container startup, `main.py` reads `/input/tasks.json`. Each task prompt is checked against `ResponseCache`. If identical prompts appear, TokenForge returns the cached answer instantly ($0.00 cost, 0 tokens).
2. **Comprehensive Category Classification:** `router.py` evaluates the prompt against an 8-category regex engine covering `code_debug`, `code_gen`, `sentiment`, `ner`, `summarization`, `logical`, `math`, and `factual`.
3. **Tier 0 Fast-Path (Deterministic Solvers):**
   - If classified as **Math**, TokenForge strips English conversational boilerplate and executes `SymPy` symbolic evaluation.
   - If classified as **Sentiment**, TokenForge executes `VADER` intensity analysis.
   - If classified as **NER**, TokenForge executes `spaCy` entity extraction.
   - *Result:* Output generated locally counts as **0 tokens** toward the competition score.
4. **Tier 1 Dynamic Cloud Tiering (`ALLOWED_MODELS`):**
   - If cloud reasoning is required, `llm_clients.py` dynamically parses `ALLOWED_MODELS` injected at runtime by the judging VM into `cheap`, `code`, and `strong` tiers.
   - Enforces category-tailored system prompts and strict output token ceilings (`128` to `512` tokens).
   - Includes automatic multi-model fallback across `ALLOWED_MODELS` on transient errors.

---

## 📋 Hackathon Category Coverage & Token Budgets

| # | Official Category | Target Track Coverage | Routing Tier | Output Cap (`max_tokens`) |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Factual knowledge** | Explaining concepts, definitions, how things work | `strong` | `256` |
| **2** | **Mathematical reasoning** | Arithmetic, percentages, word problems, projections | `SymPy` / `strong` | `384` |
| **3** | **Sentiment classification** | Labelling sentiment and justifying classification | `VADER` / `cheap` | `128` |
| **4** | **Text summarisation** | Condensing passages to format/length constraint | `cheap` | `256` |
| **5** | **Named entity recognition** | Extracting entities (`PERSON`, `ORG`, `LOCATION`, `DATE`) | `spaCy` / `cheap` | `256` |
| **6** | **Code debugging** | Identifying bugs and providing corrected code | `code` | `512` |
| **7** | **Logical reasoning** | Constraint-based puzzles where conditions must hold | `strong` | `384` |
| **8** | **Code generation** | Writing correct, well-structured functions from spec | `code` | `512` |

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

# Download local spaCy NLP model
python -m spacy download en_core_web_sm
```

### Step 2: Run Local Headless Verification
Test the headless container engine locally against sample practice tasks in `input/tasks.json`:

```bash
python main.py
```

Inspect the generated `output/results.json` to verify zero-token deterministic answers (`Answer: 12`, `Sentiment: Positive`, etc.) and structured JSON compliance.

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
- ✅ **Exit code 0** on clean evaluation batch completion.
- ✅ **All external calls routed via `FIREWORKS_BASE_URL`** (no rogue API endpoints).
- ✅ **Dynamic runtime parsing of `ALLOWED_MODELS`** (no hardcoded model IDs).
- ✅ **Zero-token local models** (`SymPy`, `VADER`, `spaCy`) count as $0.00 toward judging score.
- ✅ **Ultra-compact image size** (`3.17 GB`, well below the 10 GB limit).
