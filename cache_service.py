"""
cache_service.py — Service de cache Redis (Exact + Sémantique)
Isolation complète par user_id, support_id et model_name.
"""
import redis
import hashlib
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional

# =========================
# CONFIG
# =========================
REDIS_HOST           = "redis-cache"
REDIS_PORT           = 6379
CACHE_TTL            = 86400        # 24h
SIMILARITY_THRESHOLD = 0.70
SEMANTIC_SCAN_LIMIT  = 100

# =========================
# REDIS CONNECTION
# =========================
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

def get_redis() -> redis.Redis:
    return redis_client

# =========================
# EMBEDDING MODEL (lazy)
# =========================
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print("[cache_service] Loading embedding model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

# =========================
# UTILS
# =========================
def normalize_prompt(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    aliases = {
        " ml ":  " machine learning ",
        " ai ":  " artificial intelligence ",
        " dl ":  " deep learning ",
        " nlp ": " natural language processing ",
    }
    for k, v in aliases.items():
        text = text.replace(k, v)
    return text

def hash_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

def cosine_similarity(vec1, vec2) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# =========================
# SCOPE — source unique de vérité
# Isolé par user_id + support_id + model_name
# =========================
def build_scope(
    role: str,
    support_id: str,
    model_name: str,
    user_id: str = "anonymous",
) -> str:
    """
    Construit le scope de cache.
    - student  → isolé par user + support + model
    - admin/teacher → partagé globalement par support + model
    """
    if role == "student":
        return f"student:{user_id}:{support_id}:{model_name}"
    return f"global:{support_id}:{model_name}"

# =========================
# EXACT CACHE
# Clé : exact:{hash(cache_key)}
# cache_key inclut déjà user_id depuis le appelant
# =========================
def get_exact_cache(cache_key: str) -> Optional[str]:
    try:
        key = f"exact:{hash_key(cache_key)}"
        return get_redis().get(key)
    except Exception as e:
        print(f"[exact_cache] GET error: {e}")
        return None

def set_exact_cache(cache_key: str, response: str) -> None:
    try:
        key = f"exact:{hash_key(cache_key)}"
        get_redis().setex(key, CACHE_TTL, response)
    except Exception as e:
        print(f"[exact_cache] SET error: {e}")

# =========================
# SEMANTIC CACHE
# Clé stable : semantic:{hash(scope + prompt)} → pas de doublons
# Scan filtré par scope dans la valeur JSON
# =========================
def get_semantic_cache(
    prompt: str,
    role: str,
    support_id: str,
    model_name: str,
    user_id: str = "anonymous",
) -> Optional[str]:
    """
    Recherche sémantique dans Redis.
    Filtre par scope (user + support + model).
    Retourne la meilleure réponse si sim >= SIMILARITY_THRESHOLD.
    """
    try:
        r = get_redis()
        scope      = build_scope(role, support_id, model_name, user_id)
        normalized = normalize_prompt(prompt)
        model      = get_embedding_model()
        query_emb  = model.encode(normalized, normalize_embeddings=True)

        best_score    = 0.0
        best_response = None

        for i, key in enumerate(r.scan_iter("semantic:*")):
            if i >= SEMANTIC_SCAN_LIMIT:
                break
            raw = r.get(key)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if "embedding" not in data or "scope" not in data or "response" not in data:
                continue

            # Filtre par scope — ne compare que les entrées du même user + cours
            if data["scope"] != scope:
                continue

            sim = cosine_similarity(query_emb, data["embedding"])
            print(f"[semantic_cache] sim={sim:.3f} key={key[-8:]}")

            if sim > best_score:
                best_score    = sim
                best_response = data["response"]

        if best_score >= SIMILARITY_THRESHOLD:
            print(f"[semantic_cache] HIT score={best_score:.3f}")
            return best_response

    except Exception as e:
        print(f"[semantic_cache] GET error: {e}")

    return None

def set_semantic_cache(
    prompt: str,
    response: str,
    role: str,
    support_id: str,
    model_name: str,
    user_id: str = "anonymous",
) -> None:
    """
    Stocke dans le cache sémantique.
    Clé stable basée sur hash(scope + prompt) → évite les doublons.
    """
    try:
        r          = get_redis()
        scope      = build_scope(role, support_id, model_name, user_id)
        normalized = normalize_prompt(prompt)
        model      = get_embedding_model()
        embedding  = model.encode(normalized, normalize_embeddings=True).tolist()

        # Clé stable (pas de timestamp) → un seul enregistrement par prompt/scope
        key = f"semantic:{hash_key(scope + ':' + normalized)}"
        value = {
            "scope":      scope,
            "prompt":     normalized,
            "response":   response,
            "embedding":  embedding,
            "model":      model_name,
            "created_at": int(time.time()),
        }
        r.setex(key, CACHE_TTL, json.dumps(value))
        print(f"[semantic_cache] SET scope={scope}")

    except Exception as e:
        print(f"[semantic_cache] SET error: {e}")

# =========================
# API UNIFIÉE
# Point d'entrée unique pour patches.py, cache_router.py, cache_middleware.py
# =========================
def get_cache(
    cache_key: str,
    prompt: str,
    role: str,
    support_id: str,
    model_name: str,
    user_id: str = "anonymous",
) -> tuple[Optional[str], Optional[str]]:
    """
    Vérifie exact puis sémantique.
    Retourne (response, cache_type) où cache_type = 'exact' | 'semantic' | None
    """
    # 1. Exact cache
    exact = get_exact_cache(cache_key)
    if exact:
        print("[cache] HIT exact")
        return exact, "exact"

    # 2. Semantic cache
    semantic = get_semantic_cache(prompt, role, support_id, model_name, user_id)
    if semantic:
        print("[cache] HIT semantic → backfill exact")
        # Promotion en exact pour accélérer les prochains appels identiques
        set_exact_cache(cache_key, semantic)
        return semantic, "semantic"

    print("[cache] MISS")
    return None, None

def store_cache(
    cache_key: str,
    prompt: str,
    response: str,
    role: str,
    support_id: str,
    model_name: str,
    user_id: str = "anonymous",
) -> None:
    """
    Stocke dans exact ET sémantique après un appel LLM réel.
    """
    set_exact_cache(cache_key, response)
    set_semantic_cache(prompt, response, role, support_id, model_name, user_id)

# =========================
# STATS
# =========================
def get_cache_stats() -> dict:
    """Stats Redis pour debug et monitoring."""
    try:
        r = get_redis()
        exact_keys    = sum(1 for _ in r.scan_iter("exact:*"))
        semantic_keys = sum(1 for _ in r.scan_iter("semantic:*"))
        return {
            "exact_entries":    exact_keys,
            "semantic_entries": semantic_keys,
            "total":            exact_keys + semantic_keys,
            "ttl_hours":        CACHE_TTL // 3600,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        }
    except Exception as e:
        return {"error": str(e)}
