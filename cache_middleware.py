"""
cache_middleware.py — Façade légère sur cache_service.py
Rôle : appel LLM + orchestration du cache (exact → sémantique → LLM)
Toute la logique Redis est dans cache_service.py (source unique)
"""
import time
import requests
from typing import Optional

from open_tutorai.services.cache_service import (
    get_cache,
    store_cache,
    normalize_prompt,
    build_scope,
    get_exact_cache,
    set_exact_cache,
    get_semantic_cache,
    set_semantic_cache,
)

# =========================
# CONFIG LLM
# =========================
OLLAMA_URL = "http://172.20.0.1:11434/api/generate"
MODEL_NAME = "mistral"

# =========================
# LLM CALL
# =========================
def call_llm(prompt: str, model: str = MODEL_NAME) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        return response.json().get("response", "")
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return "Error generating response"

# =========================
# CONSTRUCTION DE LA CACHE KEY EXACTE
# Format : user:{user_id}:scope:{scope}:{hash_prompt}
# (le hash est fait dans cache_service)
# =========================
def build_exact_key(user_id: str, role: str, support_id: str, model_name: str, prompt: str) -> str:
    """
    Clé pour le cache exact.
    Inclut user_id pour isoler les réponses personnalisées.
    Le scope dans la clé permet le scan sémantique filtré.
    """
    scope = build_scope(role, support_id, model_name)
    normalized = normalize_prompt(prompt)
    return f"user:{user_id}:{scope}:{normalized}"

# =========================
# FLUX PRINCIPAL
# =========================
def get_response_with_cache(
    prompt: str,
    user_id: str = "anonymous",
    role: str = "student",
    support_id: str = None,
    model_name: str = MODEL_NAME,
) -> dict:
    start = time.time()
    safe_support = support_id or "unknown"
    normalized = normalize_prompt(prompt)

    # Clé exact isolée par user
    exact_key = build_exact_key(user_id, role, safe_support, model_name, normalized)

    print(f"\n[CACHE] user={user_id} role={role} support={safe_support} model={model_name}")
    print(f"[CACHE] prompt={normalized[:80]}")

    # ── Appel unifié cache_service ──────────────────────
    response, cache_type = get_cache(
        cache_key=exact_key,
        prompt=normalized,
        role=role,
        support_id=safe_support,
        model_name=model_name,
    )

    if response:
        source = "exact_cache" if cache_type == "exact" else "semantic_cache"
        print(f"[CACHE] HIT {source}")
        return {
            "response": response,
            "source": source,
            "latency": round(time.time() - start, 4),
        }

    print("[CACHE] MISS → calling LLM")

    # ── LLM ────────────────────────────────────────────
    llm_response = call_llm(normalized, model_name)

    # ── Store via cache_service ─────────────────────────
    store_cache(
        cache_key=exact_key,
        prompt=normalized,
        response=llm_response,
        role=role,
        support_id=safe_support,
        model_name=model_name,
        user_id=user_id,
    )
    print("[CACHE] Stored")

    return {
        "response": llm_response,
        "source": "llm",
        "latency": round(time.time() - start, 4),
    }
