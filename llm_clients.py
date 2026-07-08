import requests
from typing import Optional
from openai import OpenAI

def call_local_ollama(prompt: str) -> Optional[str]:
    """Calls local Ollama instance running llama3.2:1b."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("response")
    except Exception:
        return None

def call_fireworks(prompt: str, api_key: str, base_url: str, allowed_models: str) -> Optional[str]:
    """Calls Fireworks API via OpenAI client using the first model in allowed_models."""
    if not allowed_models:
        return None
    models = [m.strip() for m in allowed_models.split(",") if m.strip()]
    if not models:
        return None
    model = models[0]
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception:
        return None
