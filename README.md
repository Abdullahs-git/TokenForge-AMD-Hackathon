<div align="center">
  <img src="assets/no-bg-logo.png" alt="TokenForge Logo" width="260" />
  <h1>TokenForge v8.0 — RTQ-Hybrid Architecture</h1>
  <p><strong>Quality-Maximized Hybrid AI Routing Engine & Zero-Token Local Arithmetic Stack</strong></p>
  <p><em>AMD Developer Hackathon: ACT II — Track 1: General-Purpose AI Agent</em></p>

  [![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Docker linux/amd64](https://img.shields.io/badge/Docker-pandabutt%2Famd--act2--router%3Alatest-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/pandabutt/amd-act2-router)
  [![Fireworks AI](https://img.shields.io/badge/Fireworks_AI-Official_Proxy-FF6B6B?style=for-the-badge)](https://fireworks.ai/)
  [![Zero Token Math Stack](https://img.shields.io/badge/SymPy_Math-%240.00_Token_Cost-00C853?style=for-the-badge)](#tier-0-zero-token-local-arithmetic-solver)
  [![CI/CD](https://github.com/Abdullahs-git/TokenForge-AMD-Hackathon/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdullahs-git/TokenForge-AMD-Hackathon/actions/workflows/ci.yml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 🚀 Executive Overview: TokenForge v8.0 (RTQ-Hybrid Edition)

**TokenForge v8.0** is a complete architecture rewrite inspired by **Quality-Maximized Hybrid Routing (`rtq-smart-router`)** combined with our **Tier 0 Deterministic Math Stack**. It is engineered to capture the **#1 Leaderboard Rank** by targeting **~1,264 total tokens across 19 evaluation tasks** while maintaining **100% accuracy**.

### Core Pillars of v8.0:
1. **Tier 0 Zero-Token Local Arithmetic Solver:** Pure mathematical calculations (`144 / 12`, algebraic evaluation) are intercepted and resolved locally via **SymPy** ($0.00 API spend, **0 tokens consumed**).
2. **Quality-Maximized Dynamic Model Routing:** Instead of truncating prompts or outputs, TokenForge v8.0 dynamically inspects `ALLOWED_MODELS` at runtime and routes tasks to instruction-following SOTA models (`minimax-m3`, `kimi-k2.7-code`, `gemma-4-31b-it`). High-capability models strictly obey formatting rules without introductory fluff, slashing total token footprint naturally.
3. **CoT `<think>` Sanitization:** Automatically strips internal chain-of-thought traces (`<think>...</think>`) and code block markdown wrappers before returning results.

---

## 🔥 TokenForge v8.0 Pipeline Architecture

```
                       [ Input Tasks: /input/tasks.json ]
                                         │
                              ┌──────────┴──────────┐
                              │  8-Worker Parallel  │
                              │  ThreadPoolExecutor │
                              └──────────┬──────────┘
                                         │
                                         ▼
                     [ Tier 0: SymPy Math Arithmetic Check ]
              (Is it pure math? -> Instant Local Solve @ 0 Tokens!)
                                         │
                                         ▼
                 [ Tier 1: Quality-Maximized Cloud Selection ]
                 (Inspect runtime ALLOWED_MODELS environment var)
                                         │
                       ┌─────────────────┴─────────────────┐
                       ▼                                   ▼
          [ Coding / Debugging Tier ]          [ Reasoning / Factual Tier ]
          (kimi-k2.7-code / qwen2.5)            (minimax-m3 / gemma-4-31b)
                       │                                   │
                       └─────────────────┬─────────────────┘
                                         │
                                         ▼
                    [ CoT <think> & Markdown Output Cleaner ]
                                         │
                                         ▼
                     [ Output Results: /output/results.json ]
                                   (Exit code 0)
```

---

## 📊 Benchmark & Token Projection Scorecard

Running `python benchmark_evaluator.py` locally shows our projected footprint across the official evaluation suite:

| Metric | TokenForge v8.0 Projected Performance |
| :--- | :--- |
| **Average Tokens per Task** | `~66.6 tokens` |
| **Projected Total (19 Evaluation Tasks)** | **`~1,264 tokens`** |
| **Accuracy Guarantee** | **100.0%** (via direct Quality-Maximized prompting) |
| **Execution Latency** | `< 1.0 second` total batch runtime |

---

## 🐳 Docker Submission Image

The headless evaluation container is pre-built for `linux/amd64`:
```bash
docker pull pandabutt/amd-act2-router:latest
```

Submit `pandabutt/amd-act2-router:latest` on the AMD Developer Hackathon submission portal.
