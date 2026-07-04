# OpenTutorAI — Semantic Cache Layer

Optimisation du déploiement LLM sur infrastructure CPU-only : caching hybride sémantique + mémoire pédagogique multi-couches.

> **Version production** : pipeline actif basé sur `patches.py` + `cache_service.py`.  
> `cache_middleware.py` et `llm_service.py` appartiennent à la version initiale du système — conservés dans le dépôt mais non actifs en production.

---

## Architecture de production

```
Requête utilisateur
        ↓
Open WebUI (port 8080)
        ↓
patches.py  ← intercepteur actif (monkey-patch de generate_chat_completion)
        ↓
┌─────────────────────────────────────┐
│           cache_service.py          │  ← source unique de vérité Redis
│  1. Cache exact   (hash MD5)        │ → HIT → réponse directe
│  2. Cache sémantique (cosinus 0.70) │ → HIT → réponse directe
│  3. MISS                            │ → appel LLM via Open WebUI
└─────────────────────────────────────┘
        ↓ MISS
┌─────────────────────────────────────┐
│  Ollama (local)  │  OpenAI (cloud)  │
│  Mistral Q4_K_M  │  gpt-4o-mini     │
└─────────────────────────────────────┘
        ↓
Réponse stockée dans Redis (exact + sémantique)
        ↓
Retournée à l'utilisateur
```

---

## Fichiers

| Fichier | Rôle | Statut |
|---------|------|--------|
| `patches.py` | Intercepteur actif — monkey-patch de `generate_chat_completion` | ✅ Production |
| `cache_service.py` | Source unique Redis — cache exact MD5 + cache sémantique cosinus | ✅ Production |
| `cache_router.py` | Endpoints FastAPI `/api/cache/generate` · `/api/cache/stats` | ✅ Production |
| `memory_service.py` | Mémoire pédagogique 3 couches Redis | ✅ Production |
| `benchmark_v3_clean.py` | Benchmark comparatif 5 modèles × 3 scénarios × 22 prompts | ✅ Production |
| `cache_middleware.py` | Ancienne orchestration du pipeline | ⚠️ Dormant |
| `llm_service.py` | Ancien routage Ollama / OpenAI | ⚠️ Dormant |

---

## Configuration

```python
REDIS_HOST           = "redis-cache"
REDIS_PORT           = 6379
CACHE_TTL            = 86400        # 24h
SIMILARITY_THRESHOLD = 0.70         # production / 0.75 benchmark
SEMANTIC_SCAN_LIMIT  = 100
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"  # 384 dims
```

---

## Cache exact (MD5)

```python
def get_exact_cache(cache_key: str) -> Optional[str]:
    key = f"exact:{hash_key(cache_key)}"   # hash_key() utilise MD5
    return get_redis().get(key)

def set_exact_cache(cache_key: str, response: str) -> None:
    key = f"exact:{hash_key(cache_key)}"
    get_redis().setex(key, CACHE_TTL, response)
```

---

## Cache sémantique (cosinus)

```python
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims

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

## API unifiée — cache_service.py

```python
# Point d'entrée unique pour patches.py et cache_router.py

def get_cache(cache_key, prompt, role, support_id, model_name, user_id):
    """Vérifie exact puis sémantique. Retourne (response, cache_type)."""
    # 1. Exact cache
    exact = get_exact_cache(cache_key)
    if exact:
        return exact, "exact"

    # 2. Semantic cache + backfill automatique vers exact
    semantic = get_semantic_cache(prompt, role, support_id, model_name, user_id)
    if semantic:
        set_exact_cache(cache_key, semantic)   # promotion exact
        return semantic, "semantic"

    return None, None

def store_cache(cache_key, prompt, response, role, support_id, model_name, user_id):
    """Stocke dans exact ET sémantique après un appel LLM réel."""
    set_exact_cache(cache_key, response)
    set_semantic_cache(prompt, response, role, support_id, model_name, user_id)
```

---

## Mémoire pédagogique

```python
SESSION_TTL = 3600      # 1h  — session courante
WORKING_TTL = 86400     # 24h — concepts vus dans la journée
CONTEXT_TTL = 2592000   # 30j — profil long terme de l'apprenant

def update_all_memory(user_id, support_id, prompt, session_id):
    update_session_memory(user_id, support_id, prompt)
    update_working_memory(user_id, support_id, session_id, prompt)
    update_context_store(user_id)
```

> `get_full_student_profile()` est implémenté mais non activé — perspective future.

---

## Benchmark

`benchmark_v3_clean.py` évalue 5 modèles LLM (2 locaux via Ollama + 3 cloud via OpenAI)  
sur 22 prompts pédagogiques organisés en 7 groupes thématiques,  
dans 3 scénarios isolés par préfixes Redis : `no_cache`, `exact_cache`, `semantic_cache`.

```bash
python benchmark_v3_clean.py
# Résultats exportés dans :
#   benchmark_v3_results.csv  — détail ligne par ligne
#   benchmark_v3_summary.csv  — résumé par modèle/scénario
```

---

## Installation

```bash
# Dépendances Python
pip install redis sentence-transformers numpy httpx openai python-dotenv

# Configuration
cp .env.example .env
# Renseigner OPENAI_API_KEY dans .env

# Lancer Redis
docker run -d --name redis-cache -p 6379:6379 redis:7
```

---

## Tester

```bash
# Stats du cache
curl http://localhost:8080/api/cache/stats

# Générer avec cache
curl -X POST http://localhost:8080/api/cache/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral","prompt":"quest ce que python",
       "user_id":"user1","support_id":"s1"}'

# Inspecter les clés Redis
docker exec redis-cache redis-cli KEYS "*"
```

---

## Infrastructure

```
VPS      : Ubuntu 22.04 · 22 Go RAM · 8 vCPU · sans GPU
Redis    : version 7 · port 6379
Ollama   : Mistral 7B Q4_K_M · ~4.1 Go RAM (réduction 70% vs float16)
OpenAI   : gpt-4o-mini · fallback cloud
Embedding: all-MiniLM-L6-v2 · 384 dims · ~2ms/encodage
```

---

## Évolution de l'architecture

| Version | Pipeline | Statut |
|---------|----------|--------|
| v1 | `cache_middleware.py` → `llm_service.py` | ⚠️ Dormant |
| v2 (production) | `patches.py` → `cache_service.py` | ✅ Actif |
