import os
import requests

LMSTUDIO_URL = os.getenv(
    "LMSTUDIO_URL",
    "http://host.docker.internal:1234/v1/chat/completions"
)
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "txgemma-9b-chat")

def generate_report_lmstudio(prompt: str, temperature: float = 0.0, max_tokens: int = 900) -> str:
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = requests.post(LMSTUDIO_URL, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]