import os
import re
from typing import Optional
import local_solvers
import llm_clients

def classify_and_route(prompt: str) -> str:
    """Classifies a prompt and routes it to the appropriate local solver or LLM client."""
    prompt_lower = prompt.lower()
    
    # Basic classification rules using keywords/regex
    if re.search(r'\b(solve|equation|plus|minus|multiplied|divided|sum|subtract|\+|-|\*|/|=)\b', prompt_lower):
        category = "math"
    elif re.search(r'\b(sentiment|happy|sad|angry|feel|emotion|love|hate|vibe|review)\b', prompt_lower):
        category = "sentiment"
    elif re.search(r'\b(ner|named entity|entities|person|location|organization|extract names)\b', prompt_lower):
        category = "ner"
    elif re.search(r'\b(debug|error|exception|stack trace|bug|fix code|compile)\b', prompt_lower):
        category = "code_debug"
    elif re.search(r'\b(logic|riddle|puzzle|if then|prove|deduct|reasoning)\b', prompt_lower):
        category = "logical"
    elif re.search(r'\b(summarize|summary|tldr|shorten|gist|condense)\b', prompt_lower):
        category = "summarization"
    elif re.search(r'\b(code|generate|function|class|write script|python|javascript|c\+\+|html)\b', prompt_lower):
        category = "code_gen"
    else:
        category = "factual"

    # Route math, sentiment, ner to local solvers first
    result: Optional[str] = None
    if category == "math":
        result = local_solvers.solve_math(prompt)
    elif category == "sentiment":
        result = local_solvers.analyze_sentiment(prompt)
    elif category == "ner":
        result = local_solvers.extract_ner(prompt)

    # If the local solver successfully returned a value, return it
    if result is not None:
        return result

    # Fallback to local Ollama (llama3.2:1b)
    result = llm_clients.call_local_ollama(prompt)
    if result is not None:
        return result

    # Fallback to Fireworks API
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    allowed_models = os.environ.get("ALLOWED_MODELS", "")

    result = llm_clients.call_fireworks(prompt, api_key, base_url, allowed_models)
    if result is not None:
        return result

    return "Error: Failed to process prompt using local solvers, Ollama, or Fireworks API."
