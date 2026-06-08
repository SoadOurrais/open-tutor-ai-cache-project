# OpenTutorAI — Semantic Cache Layer

Optimisation du déploiement LLM : caching hybride + mémoire pédagogique

---

## Architecture

```
Requête utilisateur
        ↓
cache_middleware.py
        ↓
┌─────────────────────────────────────┐
│           Redis Cache Layer         │
│  1. Cache exact   (SHA-256)         │ → HIT → réponse directe
│  2. Cache sémantique (cosinus 0.70) │ → HIT → réponse directe
│  3. MISS                            │ → LLM
└─────────────────────────────────────┘
        ↓ MISS
┌─────────────────────────────────────┐
│         llm_service.py              │
│  Ollama (local)  │  OpenAI (cloud)  │
│  Mistral Q4_K_M  │  gpt-4o-mini     │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│         memory_service.py           │
│  Session  │ Working  │  Context     │
│  TTL 1h   │ TTL 24h  │  TTL 30j    │
└─────────────────────────────────────┘
```

---

## Fichiers

| Fichier | Rôle |
|---------|------|
| `cache_service.py` | Cache exact MD5 + cache sémantique cosinus · Redis 7 · isolation user_id |
| `cache_middleware.py` | Orchestration du pipeline complet |
| `cache_router.py` | Endpoints FastAPI `/api/cache/generate` · `/api/cache/stats` |
| `llm_service.py` | Routage automatique Ollama / OpenAI + fallback |
| `memory_service.py` | Mémoire pédagogique 3 couches Redis |
| `benchmark_v3_clean.py` | Benchmark comparatif 5 modèles × 3 scénarios |

---

## Configuration

```python
REDIS_HOST           = "redis-cache"
REDIS_PORT           = 6379
CACHE_TTL            = 86400        # 24h
SIMILARITY_THRESHOLD = 0.70
SEMANTIC_SCAN_LIMIT  = 100
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"  # 384 dims
```

---

## Cache exact (SHA-256)

```python
def get_exact_cache(cache_key: str) -> Optional[str]:
    key = f"exact:{hash_key(cache_key)}"
    return get_redis().get(key)

def set_exact_cache(cache_key: str, response: str) -> None:
    key = f"exact:{hash_key(cache_key)}"
    get_redis().setex(key, CACHE_TTL, response)
```

---

## Cache sémantique (cosinus)

```python
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_semantic_cache(prompt, role, support_id, model_name, user_id):
    scope     = build_scope(role, support_id, model_name, user_id)
    query_emb = model.encode(normalize_prompt(prompt), normalize_embeddings=True)

    for key in redis.scan_iter("semantic:*"):
        data = json.loads(redis.get(key))
        if data["scope"] != scope:
            continue
        sim = cosine_similarity(query_emb, data["embedding"])
        if sim >= SIMILARITY_THRESHOLD:
            return data["response"]
    return None
```

---

## Isolation par user_id

```python
def build_scope(role, support_id, model_name, user_id="anonymous") -> str:
    if role == "student":
        return f"student:{user_id}:{support_id}:{model_name}"
    return f"global:{support_id}:{model_name}"
```

---

## LLM Routing + fallback

```python
def resolve_provider(model: str) -> str:
    if model.startswith(("gpt", "o1", "o3")):
        return "openai"
    return "ollama"

# Fallback automatique si OpenAI → 429 TPM
# retry 2s → 4s → 8s → Ollama
```

---

## Mémoire pédagogique

```python
# 3 couches Redis
SESSION_TTL = 3600      # 1h  — session courante
WORKING_TTL = 86400     # 24h — concepts vus
CONTEXT_TTL = 2592000   # 30j — profil long terme

def update_all_memory(user_id, support_id, prompt, session_id):
    update_session_memory(user_id, support_id, prompt)
    update_working_memory(user_id, support_id, session_id, prompt)
    update_context_store(user_id)
```

---

## Installation

```bash
# Dépendances Python
pip install redis sentence-transformers numpy httpx openai python-dotenv

# Configuration (renseigner OPENAI_API_KEY dans .env)
cp .env.example .env

# Lancer Redis localement
docker run -d --name redis-cache -p 6379:6379 redis:7
```

## Benchmark

`benchmark_v3_clean.py` évalue et compare 5 modèles LLM
(2 locaux via Ollama + 3 cloud via OpenAI) sur 22 prompts
pédagogiques organisés en 7 groupes thématiques,
dans 3 scénarios : `no_cache`, `exact_cache`, `semantic_cache`.

```bash
python benchmark_v3_clean.py
# Résultats exportés dans :
#   benchmark_v3_results.csv  — toutes les lignes
#   benchmark_v3_summary.csv  — résumé par modèle/scénario
```
## Tester

```bash
# Stats du cache
curl http://localhost:8080/api/cache/stats

# Générer avec cache
curl -X POST http://localhost:8080/api/cache/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral","prompt":"quest ce que python",
       "user_id":"user1","support_id":"s1"}'

# Voir les clés Redis
docker exec redis-cache redis-cli KEYS "*"
```

---

## Infrastructure

```
VPS      : Ubuntu Linux · 22 Go RAM · 8 vCPU · sans GPU
Redis    : version 7 · port 6379
Ollama   : Mistral 7B Q4_K_M · 4,1 Go RAM
OpenAI   : gpt-4o-mini · fallback cloud
Embedding: all-MiniLM-L6-v2 · 384 dims · ~2ms/encodage
```

---

*Résultats détaillés disponibles dans le draft paper scientifique.*
