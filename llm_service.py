import httpx
from dotenv import load_dotenv
import os


OLLAMA_URL = "http://172.17.0.1:11434/api/chat"

load_dotenv("/home/ubuntu/open-tutor-ai-CE/testing/benchmark.env")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

api_key = os.getenv("OPENAI_API_KEY")

# ─────────────────────────────
# ROUTING MODEL
# ─────────────────────────────

def resolve_provider(model: str) -> str:
    api_models_prefix = ("gpt", "o1", "o3", "gpt-4", "gpt-5")

    if model.startswith(api_models_prefix):
        return "openai"

    return "ollama"


# ─────────────────────────────
# MAIN LLM CALL
# ─────────────────────────────

async def call_llm(model: str, prompt: str, max_tokens: int = 500):

    provider = resolve_provider(model)

    messages = [
        {"role": "user", "content": prompt}
    ]

    # ─────────────
    # OLLAMA LOCAL
    # ─────────────
    if provider == "ollama":
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                res = await client.post(
                    OLLAMA_URL,
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False
                    }
                )
                res.raise_for_status()
                data = res.json()

                return {
                    "provider": "ollama",
                    "model": model,
                    "response": data.get("message", {}).get("content", ""),
                    "raw": data
                }

            except Exception as e:
                return {
                    "provider": "ollama",
                    "error": str(e),
                    "response": ""
                }

    # ─────────────
    # OPENAI API
    # ─────────────
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            res = await client.post(
                OPENAI_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens
                }
            )
            res.raise_for_status()
            data = res.json()

            return {
                "provider": "openai",
                "model": model,
                "response": data["choices"][0]["message"]["content"],
                "raw": data
            }

        except Exception as e:
            return {
                "provider": "openai",
                "error": str(e),
                "response": ""
            }
