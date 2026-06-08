
"""
memory_service.py — OpenTutorAI
================================
Memory Manager pédagogique avec Redis.
Architecture 3 couches :
  - Session Memory  : historique session courante (TTL 1h)
  - Working Memory  : niveau inscrit + concepts vus (TTL 24h)
  - Context Store   : profil long terme étudiant (TTL 30 jours)

Note : le niveau est défini à l'inscription dans le support,
pas détecté depuis les prompts.
"""
import redis
import json
from datetime import datetime

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
REDIS_HOST  = "redis-cache"
REDIS_PORT  = 6379
SESSION_TTL = 3600      # 1h   — session courante
WORKING_TTL = 86400     # 24h  — concepts de la journée
CONTEXT_TTL = 2592000   # 30j  — profil long terme

# ─────────────────────────────────────────
# REDIS CONNECTION
# ─────────────────────────────────────────
try:
    _redis = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True
    )
    _redis.ping()
    print("[memory_service] Redis connected")
except Exception as e:
    print(f"[memory_service] Redis unavailable: {e}")
    _redis = None

def _get_redis():
    return _redis

# ─────────────────────────────────────────
# SESSION MEMORY — TTL 1h
# Mémoire de la session courante
# ─────────────────────────────────────────
def _default_session() -> dict:
    return {
        "messages_count":        0,
        "session_start":         datetime.now().isoformat(),
        "last_prompt":           None,
        "concepts_this_session": [],
    }

def get_session_memory(user_id: str, support_id: str) -> dict:
    r = _get_redis()
    if not r:
        return _default_session()
    key = f"session:{user_id}:{support_id}"
    try:
        raw = r.get(key)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return _default_session()

def update_session_memory(user_id: str, support_id: str, prompt: str):
    r = _get_redis()
    if not r:
        return
    key     = f"session:{user_id}:{support_id}"
    session = get_session_memory(user_id, support_id)

    session["messages_count"] += 1
    session["last_prompt"]     = prompt[:100]

    try:
        r.setex(key, SESSION_TTL, json.dumps(session))
    except Exception as e:
        print(f"[memory_service] Session write error: {e}")

# ─────────────────────────────────────────
# WORKING MEMORY — TTL 24h
# Niveau inscrit + concepts vus + sessions
# Le niveau vient de l'inscription au support,
# il n'est pas détecté automatiquement.
# ─────────────────────────────────────────
def _default_working() -> dict:
    return {
        "concepts_seen":      [],
        "concepts_difficult": [],
        "nb_sessions":        0,
        "counted_sessions":   [],
    }

def get_working_memory(user_id: str, support_id: str) -> dict:
    r = _get_redis()
    if not r:
        return _default_working()
    key = f"working:{user_id}:{support_id}"
    try:
        raw = r.get(key)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return _default_working()

def update_working_memory(
    user_id: str,
    support_id: str,
    session_id: str,
    prompt: str,
) -> dict:
    r = _get_redis()
    if not r:
        return _default_working()

    key     = f"working:{user_id}:{support_id}"
    working = get_working_memory(user_id, support_id)

    # Comptage des sessions uniques
    counted = working.get("counted_sessions", [])
    if session_id and session_id not in counted:
        counted.append(session_id)
        counted = counted[-50:]  # limite à 50 sessions
        working["counted_sessions"] = counted
        working["nb_sessions"]      = len(counted)

    try:
        r.setex(key, WORKING_TTL, json.dumps(working))
    except Exception as e:
        print(f"[memory_service] Working write error: {e}")

    return working

# ─────────────────────────────────────────
# CONTEXT STORE — TTL 30j
# Profil long terme étudiant
# ─────────────────────────────────────────
def _default_context() -> dict:
    return {
        "preferred_style":  "balanced",
        "total_sessions":   0,
        "subjects_studied": [],
        "last_active":      None,
    }

def get_context_store(user_id: str) -> dict:
    r = _get_redis()
    if not r:
        return _default_context()
    key = f"context:{user_id}"
    try:
        raw = r.get(key)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return _default_context()

def update_context_store(user_id: str, support_subject: str = ""):
    r = _get_redis()
    if not r:
        return
    key     = f"context:{user_id}"
    context = get_context_store(user_id)

    context["total_sessions"] += 1
    context["last_active"]     = datetime.now().isoformat()

    if support_subject and support_subject not in context["subjects_studied"]:
        context["subjects_studied"].append(support_subject)

    try:
        r.setex(key, CONTEXT_TTL, json.dumps(context))
    except Exception as e:
        print(f"[memory_service] Context write error: {e}")

# ─────────────────────────────────────────
# PROFIL COMPLET
# Disponible pour une future intégration
# du context engineering
# ─────────────────────────────────────────
def get_full_student_profile(user_id: str, support_id: str) -> dict:
    """
    Retourne le profil complet combinant les 3 couches.
    Prêt pour le context engineering quand il sera réactivé.
    """
    session = get_session_memory(user_id, support_id)
    working = get_working_memory(user_id, support_id)
    context = get_context_store(user_id)

    return {
        "concepts_seen":         working.get("concepts_seen", []),
        "concepts_difficult":    working.get("concepts_difficult", []),
        "nb_sessions":           working.get("nb_sessions", 0),
        "messages_this_session": session.get("messages_count", 0),
        "total_sessions":        context.get("total_sessions", 0),
        "preferred_style":       context.get("preferred_style", "balanced"),
        "subjects_studied":      context.get("subjects_studied", []),
    }

# ─────────────────────────────────────────
# API PRINCIPALE
# Appelée par patches.py après chaque LLM
# ─────────────────────────────────────────
def update_all_memory(
    user_id:         str,
    support_id:      str,
    prompt:          str,
    support_subject: str = "",
    session_id:      str = "",
):
    """
    Met à jour les 3 couches mémoire après chaque échange LLM.
    Les hits cache ne déclenchent PAS cette fonction.
    """
    try:
        update_session_memory(user_id, support_id, prompt)
    except Exception as e:
        print(f"[memory_service] Session error: {e}")

    try:
        update_working_memory(user_id, support_id, session_id, prompt)
    except Exception as e:
        print(f"[memory_service] Working error: {e}")

    try:
        update_context_store(user_id, support_subject)
    except Exception as e:
        print(f"[memory_service] Context error: {e}")

    print(f"🧠 Memory updated | user={user_id} | support={support_id}")
