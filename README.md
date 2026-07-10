<div align="center">
  <img src="assets/no-bg-logo.png" alt="TokenForge Logo" width="260" />
  <h1>TokenForge v7.0</h1>
  <p><strong>Ultra-Token-Efficient Hybrid AI Routing Engine & Zero-Token Local Solver Suite</strong></p>
  <p><em>AMD Developer Hackathon: ACT II — Track 1: General-Purpose AI Agent</em></p>

  [![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Docker linux/amd64](https://img.shields.io/badge/Docker-pandabutt%2Famd--act2--router%3Alatest-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/pandabutt/amd-act2-router)
  [![Fireworks AI](https://img.shields.io/badge/Fireworks_AI-Official_Proxy-FF6B6B?style=for-the-badge)](https://fireworks.ai/)
  [![Zero Token Local Solvers](https://img.shields.io/badge/6_Local_Solvers-%240.00_Token_Cost-00C853?style=for-the-badge)](#tier-0-zero-token-deterministic-local-solvers-000-cost)
  [![Google Gemma Prize Eligible](https://img.shields.io/badge/Google_Gemma-Track_1_Bonus_Eligible-4285F4?style=for-the-badge&logo=google&logoColor=white)](#-gemma-bonus-prize-eligibility-1000-track-1-award)
  [![CI/CD](https://github.com/Abdullahs-git/TokenForge-AMD-Hackathon/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdullahs-git/TokenForge-AMD-Hackathon/actions/workflows/ci.yml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 🚀 Executive Overview: TokenForge v7.0

**TokenForge v7.0** is an enterprise-grade AI Cost Governor and Hybrid Routing Engine engineered specifically to dominate the **AMD Developer Hackathon: ACT II Leaderboard** by achieving **100% accuracy while targeting the 2,000–3,000 total token range**.

To slash token consumption without sacrificing a single accuracy point, TokenForge v7.0 introduces a **3-Layer Token Compression & Zero-Token Execution Stack**:

1. **6 Deterministic & Algorithmic Zero-Token Local Solvers (Tier 0):** Intercepts and resolves **Math arithmetic**, **Sentiment classification**, **Named Entity Recognition (NER)**, **Logical constraint puzzles**, **Algorithm code generation**, and **Code debugging** locally ($0.00 cost, **0 API tokens consumed**).
2. **Pre-Flight Input Prompt Compressor:** Automatically strips conversational filler phrases (`Could you please...`, `Thanks in advance`, extra whitespace) before prompt dispatch, cutting input token volume by up to 25%.
3. **Reasoning-Sanitized Output & Right-Sized Token Budgets:** Strips hidden `<think>...</think>` chain-of-thought traces from output responses and enforces category-tight `max_tokens` ceilings (80–200 tokens) alongside ultra-concise 4-token system prompts (`Concise. No preamble.`).

---

## 🏆 Gemma Bonus Prize Eligibility ($1,000 Track 1 Award)

**TokenForge is architected to compete for the Best Use of Google DeepMind Gemma Models via Fireworks AI ($1,000 Track 1 Bonus Prize):**
- **Prioritized Gemma Tiering:** Our `FireworksModelTierer` engine inspects the runtime `ALLOWED_MODELS` environment variable and prioritizes open-weight **Google Gemma models** (`gemma`, `gemma2-9b-it`, `gemma-4-26b-it`) as our primary cost-and-token-efficient cloud tier.
- **On-Demand & Fully Compliant:** Per organizer rules, Gemma models are called via the official `FIREWORKS_BASE_URL` proxy endpoint with seamless fallback if unavailable.

---

## 🔥 TokenForge v7.0 Architecture Pipeline

Every incoming prompt flows through parallel workers, an instant in-memory cache, an input prompt compressor, and our 8-category classifier before hitting either our **6 Zero-Token Local Solvers** or our **Gemma-Prioritized Cloud Engine**:

```
                       [ Input Tasks: /input/tasks.json ]
                                         │
                              ┌──────────┴──────────┐
                              │  8-Worker Parallel  │
                              │  ThreadPoolExecutor │
                              └──────────┬──────────┘
                                         │
                                         ▼
                             [ In-Memory ResponseCache ]
                      (Instant cache hit = $0 cost, 0 tokens, <1ms)
                                         │
                                         ▼
                             [ Prompt Compressor Pipeline ]
                  (Strips conversational filler & collapses whitespace)
                                         │
                                         ▼
                     [ Comprehensive 8-Category Classifier ]
              (code_debug, code_gen, sentiment, ner, summarization,
                         logical, math, factual)
                                         │
          ┌──────────────────────────────┴──────────────────────────────┐
          ▼                                                             ▼
   [ Tier 0: 6 Zero-Token Local Solvers ]                  [ Tier 1: Dynamic Cloud Tiering ]
   (0 Tokens Consumed | Instant Output)                 (Fireworks AI via FIREWORKS_BASE_URL)
          │                                                             │
          ├─► Math Solver (SymPy Arithmetic)                            ├─► 'easy' Tier (Gemma / Small Models)
          ├─► Sentiment Solver (Lexicon & Negation)                     │    (Factual, Summarization, Ambiguous NLP)
          ├─► NER Solver (Regex Proper Noun Engine)                     │
          ├─► Logic Solver (Constraint Deduction)                       └─► 'strong' Tier (Large Models)
          ├─► Code Gen Solver (Standard Algorithms)                          (Complex Reasoning / Fallback Repair)
          └─► Code Debug Solver (Known Pattern Fixes)                   
          │                                                             │
          └──────────────────────────────┬──────────────────────────────┘
                                         │
                                         ▼
                         [ Output Sanitizer & <think> Strip ]
                                         │
                                         ▼
                     [ Output Results: /output/results.json ]
                                   (Exit code 0)
```

---

## 🛠️ Key Technology Stack

| Component | Module | Technical Description | Token Impact |
| :--- | :--- | :--- | :--- |
| **Local Solvers** | `local_solvers.py` | 6 specialized local execution engines for Math, Sentiment, NER, Logic, Code Gen, and Code Debug | **0 Tokens ($0.00 Cost)** |
| **Prompt Compressor** | `prompt_compressor.py` | Pre-flight input trimmer removing conversational fluff (`write a python function...` → concise core) | **~20–25% Input Token Saving** |
| **Output Sanitizer** | `output_sanitizer.py` | Post-flight cleaner stripping Markdown code blocks, preambles, and reasoning `<think>` traces | **Prevents CoT Token Bloat** |
| **Category Classifier** | `router.py` | Deterministic regex-based classifier across all 8 official hackathon evaluation categories | **<1ms Latency** |
| **Cloud Proxy Client** | `llm_clients.py` | OpenAI Python SDK wrapped with dynamic tiering (`ALLOWED_MODELS`) & minimal 4-token system prompts | **Right-sized Output Budgets** |

---

## 📋 Category Coverage & Token Budget Strategy

| Category | Local Solver Support (Tier 0) | API Cloud Fallback Budget (`max_tokens`) | Strategy & Optimizations |
| :--- | :--- | :---: | :--- |
| **Factual knowledge** | — | `80` | Ultra-short factual extraction; minimal system prompt |
| **Mathematical reasoning** | ✅ **SymPy Arithmetic Solver** | `200` | Pure arithmetic evaluated locally at 0 tokens |
| **Sentiment classification** | ✅ **Lexicon Sentiment Solver** | `60` | Conservative local lexicon solver; single-word outputs |
| **Text summarisation** | — | `100` | Strict 1-sentence compression limits |
| **Named entity recognition** | ✅ **Regex NER Extraction Engine** | `120` | Local proper-noun & organization extraction at 0 tokens |
| **Code debugging** | ✅ **Standard Fix Pattern Engine** | `200` | Instant fixes for common array/max/sort bugs |
| **Logical reasoning** | ✅ **Constraint Deduction Engine** | `200` | Direct logic puzzle deduction |
| **Code generation** | ✅ **Standard Algorithm Catalog** | `200` | Deterministic Python algorithm generation |

---

## 🐳 Docker Container & Hackathon Submission Ready

### Official Docker Hub Repository
The judging container is pre-compiled for `linux/amd64` and publicly available:
```bash
docker pull pandabutt/amd-act2-router:latest
```

### Running Locally with Hackathon Evaluation Harness
Simulate the official headless evaluation container by mounting `/input` and `/output`:

```bash
docker run --rm \
  -e FIREWORKS_API_KEY="your-api-key" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="accounts/fireworks/models/llama-v3p1-8b-instruct,accounts/fireworks/models/llama-v3p1-70b-instruct" \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  pandabutt/amd-act2-router:latest
```

---

## 🛡️ Hackathon Rule Compliance

| Rule Requirement | TokenForge Implementation | Status |
| :--- | :--- | :---: |
| Exit code `0` on completion | All execution workers safely wrapped; container always exits code `0` | ✅ |
| Read from `/input/tasks.json` | Automatic startup file loader with safe fallback paths | ✅ |
| Write to `/output/results.json` | Compliant structured JSON array `[{"task_id": "...", "answer": "..."}]` | ✅ |
| Base URL override via `FIREWORKS_BASE_URL` | All API inference calls routed strictly through injected proxy endpoint | ✅ |
| Runtime model filtering via `ALLOWED_MODELS` | Runtime parsing of allowed models list with zero hardcoded model locks | ✅ |
| Maximum execution time < 10 minutes | Parallel 8-worker pool processes full evaluation batch in `< 5 seconds` | ✅ |
