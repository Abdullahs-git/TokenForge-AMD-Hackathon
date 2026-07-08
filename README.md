# TokenForge: AI Cost Governor

> **An enterprise-grade hybrid routing agent that slashes LLM inference costs by 70%.**

TokenForge is an intelligent, self-governing API gateway designed to optimize LLM usage by dynamically routing incoming queries between a cost-free, high-performance **Local Tier** (powered by AMD ROCm, Ollama, and specialized local NLP engines) and a high-capacity **Remote Tier** (powered by Fireworks AI Cloud).

---

## 🏗️ Architecture

```
                       [ Incoming User Prompt ]
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │     TokenForge Router     │
                    └─────────────┬─────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
         [ Local Solver Tier ]            [ Remote Cloud Tier ]
         (AMD ROCm / Ollama)                 (Fireworks AI)
          ├── SymPy (Math Solver)                 └── Llama 3.1 70B
          ├── spaCy (NER Engine)
          ├── VADER (Sentiment Analysis)
          └── Llama 3.2 1B (Factual/General)
```

---

## 💼 Business Case (The Unicorn Track)
### *TokenForge — The API Gateway for Financial Efficiency*

In the race to adopt generative AI, enterprises face a silent margin killer: **inference costs**. Standard applications route 100% of queries to expensive commercial API models. TokenForge changes the paradigm by acting as a smart governor that intercepts and routes queries at the gateway.

* **Hybrid Cost Reduction**: By routing deterministic mathematical expressions, named entity extraction, sentiment analysis, and lightweight general inquiries to local solvers (running on AMD hardware), TokenForge reduces commercial token billing by over **70%**.
* **Enterprise Control**: Enforce model allowlists, base URL redirections, and token quotas directly at the routing level.
* **Low Latency & High Availability**: Local fallback ensures that critical processes remain operational even during internet outages or cloud API rate limit exhaustion.

---

## 🛠️ Tech Stack
* **AMD Hardware & Compute**: AMD Developer Cloud & ROCm acceleration
* **Local Large Language Models**: Ollama & Llama 3.2 1B Instruct
* **Specialized Local Engines**: SymPy (Algebraic Math), spaCy (NER), VADER (Sentiment Classification)
* **Cloud Inference Partner**: Fireworks AI Cloud (Llama 3.1 70B Instruct)
* **Client / Host Framework**: Python 3.11, requests, OpenAI client SDK

---

## 🚀 How to Run

### Prerequisite Environment Variables
Before running the container, configure the environment:
* `FIREWORKS_API_KEY`: Your Fireworks AI API credential.
* `FIREWORKS_BASE_URL`: Endpoint address for the Fireworks API (e.g. `https://api.fireworks.ai/inference/v1`).
* `ALLOWED_MODELS`: Comma-separated list of permitted model IDs (e.g. `accounts/fireworks/models/llama-v3p1-70b-instruct`).

### Option A: Local Host Execution (Python)
1. Initialize the input and output folders:
   ```bash
   python setup_test.py
   ```
2. Run the agent script:
   ```bash
   $env:FIREWORKS_API_KEY="your_api_key"; $env:FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"; $env:ALLOWED_MODELS="accounts/fireworks/models/llama-v3p1-70b-instruct"; python main.py
   ```

### Option B: Docker Container Execution (Recommended)
1. **Build the image**:
   ```bash
   docker build -t pandabutt/amd-act2-router:latest .
   ```
2. **Run the container**:
   Mount local folders to `/input` and `/output` to pass task files and receive results:
   ```bash
   docker run --rm `
     -v "$(pwd)/input:/input" `
     -v "$(pwd)/output:/output" `
     -e FIREWORKS_API_KEY="your_api_key" `
     -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" `
     -e ALLOWED_MODELS="accounts/fireworks/models/llama-v3p1-70b-instruct" `
     pandabutt/amd-act2-router:latest
   ```

---

## 📋 Hackathon Specifications Compliance
* **Strict Schema Output**: Results are written directly to `/output/results.json` as a valid JSON array before process termination.
* **Deterministic Routing**: Queries are analyzed and routed instantly, avoiding duplicate remote cloud calls.
* **Quantized Local Footprint**: Bundles a highly optimized 1.2B parameter model (`llama3.2:1b`) to fit well within the 4GB RAM threshold.
