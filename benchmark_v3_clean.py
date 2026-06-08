"""
benchmark_v3_clean.py
=====================
Benchmark comparatif : modèles locaux vs API OpenAI
Cas d'usage : OpenTutorAI (système tutoriel intelligent)

Métriques mesurées :
  - Latence (secondes)
  - Coût estimé (USD)
  - Qualité pédagogique (score 1-5, LLM-as-judge)
  - Taux de cache hit (exact + sémantique)
  - Tokens (prompt / completion)

Usage :
  1. Créer un fichier .env avec : OPENAI_API_KEY=...
  2. Lancer Redis (docker run -d -p 6379:6379 redis)
  3. Lancer Ollama avec les modèles locaux
  4. python benchmark_v3_clean.py
"""
import time
import hashlib
import requests
import csv
import os
import json
import redis
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
load_dotenv("benchmark.env") # Charge OPENAI_API_KEY depuis .env 

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OLLAMA_URL         = "http://localhost:11434/v1/chat/completions"
REDIS_HOST         = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT         = int(os.getenv("REDIS_PORT", 6379))
CACHE_TTL          = 3600        # 1 heure (tests)
MAX_TOKENS         = 250
SEMANTIC_THRESHOLD = 0.75
CSV_FILE           = "benchmark_v3_results.csv"

# Modèles locaux (via Ollama)
LOCAL_MODELS = [
    "mistral:7b-instruct-q4_K_M",   # quantifié 4-bit — cas principal de l'étude
    "llama3:8b",                     # comparatif
]

# Modèles API — noms exacts et disponibles en mai 2026
API_MODELS = [
    {"name": "gpt-4o-mini", "provider": "openai"},   # léger, économique
    {"name": "gpt-4o",      "provider": "openai"},   # haute qualité
    {"name": "gpt-4-turbo", "provider": "openai"},   # compromis performance/coût
]

# Coûts par 1000 tokens (USD) — à vérifier sur platform.openai.com
COST_TABLE = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
    "gpt-4o":      {"input": 0.00250, "output": 0.01000},
    "gpt-4-turbo": {"input": 0.01000, "output": 0.03000},
}

# ─────────────────────────────────────────────
# 2. DATASET — représentatif d'OpenTutorAI
#    Couvre : simple, exact duplicate, sémantique,
#             complexe, pédagogique (tutor)
# ─────────────────────────────────────────────
PROMPTS = [
    # --- Groupe 1 : Machine Learning (base + paraphrases) ---
    {"id": "q01", "group": "ml_def",     "type": "simple",
     "text": "Qu'est-ce que le machine learning ?"},
    {"id": "q02", "group": "ml_def",     "type": "exact",
     "text": "Qu'est-ce que le machine learning ?"},          # hit exact attendu
    {"id": "q03", "group": "ml_def",     "type": "semantic",
     "text": "Explique le machine learning simplement."},     # hit sémantique attendu
    {"id": "q04", "group": "ml_def",     "type": "semantic",
     "text": "C'est quoi le ML ?"},                          # hit sémantique attendu

    # --- Groupe 2 : Overfitting ---
    {"id": "q05", "group": "overfit",    "type": "simple",
     "text": "Définis la notion de surapprentissage (overfitting)."},
    {"id": "q06", "group": "overfit",    "type": "semantic",
     "text": "Pourquoi un modèle fait du surapprentissage ?"},
    {"id": "q07", "group": "overfit",    "type": "semantic",
     "text": "Explique l'overfitting en machine learning."},

    # --- Groupe 3 : Fonction d'activation ---
    {"id": "q08", "group": "activation", "type": "simple",
     "text": "C'est quoi une fonction d'activation dans un réseau de neurones ?"},
    {"id": "q09", "group": "activation", "type": "semantic",
     "text": "À quoi sert une fonction d'activation ?"},
    {"id": "q10", "group": "activation", "type": "semantic",
     "text": "Pourquoi utilise-t-on ReLU ou sigmoid ?"},

    # --- Groupe 4 : Rétropropagation (complexe) ---
    {"id": "q11", "group": "backprop",   "type": "complex",
     "text": "Explique en 5 étapes comment fonctionne l'algorithme de rétropropagation."},
    {"id": "q12", "group": "backprop",   "type": "semantic",
     "text": "Comment fonctionne la backpropagation ?"},
    {"id": "q13", "group": "backprop",   "type": "semantic",
     "text": "Comment un réseau de neurones apprend avec la rétropropagation ?"},

    # --- Groupe 5 : Régression vs Classification ---
    {"id": "q14", "group": "reg_vs_clf", "type": "complex",
     "text": "Quelle est la différence entre la régression et la classification ? Donne un exemple concret de chaque."},
    {"id": "q15", "group": "reg_vs_clf", "type": "semantic",
     "text": "Régression vs classification : explique simplement."},
    {"id": "q16", "group": "reg_vs_clf", "type": "semantic",
     "text": "Quels sont les types de problèmes en machine learning ?"},

    # --- Groupe 6 : Pédagogie (tutor) ---
    {"id": "q17", "group": "tutor_cnn",  "type": "tutor",
     "text": "Un étudiant ne comprend pas les réseaux convolutifs. Comment lui expliquer simplement ?"},
    {"id": "q18", "group": "tutor_cnn",  "type": "semantic",
     "text": "Explique les CNN à un débutant."},
    {"id": "q19", "group": "tutor_cnn",  "type": "semantic",
     "text": "C'est quoi un réseau convolutif ?"},

    # --- Groupe 7 : Génération d'exercice (tutor) ---
    {"id": "q20", "group": "tutor_quiz", "type": "tutor",
     "text": "Génère un quiz de 3 questions sur les arbres de décision."},
    {"id": "q21", "group": "tutor_quiz", "type": "semantic",
     "text": "Donne-moi un petit test sur les arbres de décision."},
    {"id": "q22", "group": "tutor_quiz", "type": "semantic",
     "text": "Propose 3 questions pour tester les decision trees."},
]

# ─────────────────────────────────────────────
# 3. EMBEDDINGS
# ─────────────────────────────────────────────
_embed_model = None

def load_embed_model():
    global _embed_model
    if _embed_model is None:
        print("  [embed] Chargement sentence-transformers (all-MiniLM-L6-v2)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

# ─────────────────────────────────────────────
# 4. CACHES REDIS
# ─────────────────────────────────────────────
class ExactCacheRedis:
    """Cache exact : hit uniquement si prompt identique (hash MD5)."""

    def __init__(self, model_name):
        self.r   = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.pfx = f"exact:{model_name}:"

    def _key(self, prompt: str) -> str:
        return self.pfx + hashlib.md5(prompt.encode()).hexdigest()

    def get(self, prompt: str):
        return self.r.get(self._key(prompt))

    def set(self, prompt: str, response: str):
        self.r.setex(self._key(prompt), CACHE_TTL, response)


class SemanticCacheRedis:
    """Cache sémantique : hit si similarité cosinus >= seuil."""

    def __init__(self, model_name, threshold: float = SEMANTIC_THRESHOLD):
        self.r         = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.pfx       = f"sem:{model_name}:"
        self.threshold = threshold
        self.embedder  = load_embed_model()

    def get(self, prompt: str):
        q_emb    = self.embedder.encode([prompt])
        best_sim = -1.0
        best_val = None

        for k in self.r.scan_iter(self.pfx + "*"):
            raw = self.r.get(k)
            if raw is None:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            emb_vector = data.get("emb") or data.get("embedding")  
            if emb_vector is None:                                 
                continue    
            sim = float(cosine_similarity(q_emb,   [emb_vector])[0][0])
            if sim > best_sim:
                best_sim = sim
                best_val = data

        if best_val is not None and best_sim >= self.threshold:
            return best_val["response"], best_sim
        return None

    def set(self, prompt: str, response: str):
        emb = self.embedder.encode(prompt).tolist()
        key = self.pfx + hashlib.md5(prompt.encode()).hexdigest()
        self.r.setex(key, CACHE_TTL, json.dumps({
            "emb":      emb,
            "response": response,
            "prompt":   prompt,
        }))

# ─────────────────────────────────────────────
# 5. APPELS MODÈLES
# ─────────────────────────────────────────────
def call_local(model: str, prompt: str):
    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
    }
    t   = time.time()
    r   = requests.post(OLLAMA_URL, json=payload, timeout=180)
    lat = time.time() - t
    d   = r.json()
    if "error" in d:
        raise RuntimeError(d["error"])
    return d["choices"][0]["message"]["content"], lat, d.get("usage", {})


def call_openai(model: str, prompt: str):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY non définie dans .env")
    client = OpenAI(api_key=OPENAI_API_KEY)
    params = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
    }
    t    = time.time()
    resp = client.chat.completions.create(**params)
    lat  = time.time() - t
    usage = {
        "prompt_tokens":     resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens":      resp.usage.total_tokens,
    }
    return resp.choices[0].message.content, lat, usage


def call_model(model: str, provider: str, prompt: str):
    if provider == "local":
        return call_local(model, prompt)
    return call_openai(model, prompt)

# ─────────────────────────────────────────────
# 6. COÛT & QUALITÉ
# ─────────────────────────────────────────────
def compute_cost(model: str, usage: dict) -> float:
    if model not in COST_TABLE:
        return 0.0
    rates = COST_TABLE[model]
    return round(
        (usage.get("prompt_tokens",     0) / 1000) * rates["input"]  +
        (usage.get("completion_tokens", 0) / 1000) * rates["output"],
        8,
    )


def evaluate_quality(question: str, response: str) -> int:
    """
    LLM-as-judge : évalue la qualité pédagogique de la réponse.
    Retourne un score 1-5 (ou -1 si évaluation impossible).
    Appelé AUSSI sur les cache hits pour comparer la qualité réelle.
    """
    if not OPENAI_API_KEY or not response:
        return -1
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        judge  = f"""Tu es un évaluateur pédagogique expert. Note la réponse de 1 à 5 :
5 = Réponse complète, claire, pédagogiquement excellente
4 = Bonne réponse, quelques détails manquants
3 = Réponse correcte mais peu approfondie
2 = Réponse partielle ou confuse
1 = Réponse incorrecte ou hors sujet

Question  : {question}
Réponse   : {response}

Réponds UNIQUEMENT avec le chiffre (1, 2, 3, 4 ou 5). Rien d'autre."""
        r = client.chat.completions.create(
            model      = "gpt-4o-mini",
            messages   = [{"role": "user", "content": judge}],
            max_tokens = 5,
            temperature= 0,
        )
        s = r.choices[0].message.content.strip()
        return int(s) if s.isdigit() and 1 <= int(s) <= 5 else -1
    except Exception as e:
        print(f"   [qualité] erreur : {e}")
        return -1

# ─────────────────────────────────────────────
# 7. BENCHMARK — UN scénario, UN modèle
# ─────────────────────────────────────────────
def run_scenario(model: str, provider: str, scenario: str) -> list:
    """
    Parcourt tous les prompts séquentiellement.
    Le cache se remplit au fur et à mesure → simule un vrai flux pédagogique.
    La qualité est évaluée sur TOUTES les réponses (cache hit inclus).
    """
    ec   = ExactCacheRedis(model)
    sc   = SemanticCacheRedis(model)
    rows = []

    for p in PROMPTS:
        pt  = p["text"]
        row = {
            "Model": model, "Provider": provider,
            "PromptID": p["id"], "PromptGroup": p["group"],
            "PromptType": p["type"], "Scenario": scenario,
            "Latency_s": 0.0, "PromptTokens": 0, "CompletionTokens": 0,
            "Cost_USD": 0.0, "ResponseLength": 0,
            "QualityScore": -1, "CacheHit": False,
            "SemanticSimilarity": 0.0, "Error": "", "Prompt": pt,
        }

        try:
            content  = None
            latency  = 0.0
            usage    = {}
            cache_hit= False
            sem_sim  = 0.0

            if scenario == "no_cache":
                content, latency, usage = call_model(model, provider, pt)

            elif scenario == "exact_cache":
                hit = ec.get(pt)
                if hit is not None:
                    content, latency, cache_hit = hit, 0.0, True
                    print(f"      [HIT exact]  {p['id']}")
                else:
                    content, latency, usage = call_model(model, provider, pt)
                    ec.set(pt, content)

            elif scenario == "semantic_cache":
                hit = sc.get(pt)
                if hit is not None:
                    content, sem_sim = hit
                    latency, cache_hit = 0.0, True
                    print(f"      [HIT sem={sem_sim:.3f}]  {p['id']}")
                else:
                    content, latency, usage = call_model(model, provider, pt)
                    sc.set(pt, content)

            # Qualité évaluée sur TOUTES les réponses (hit ou non)
            score = evaluate_quality(pt, content) if content else -1
            cost  = compute_cost(model, usage)

            row.update({
                "Latency_s":          round(latency, 3),
                "PromptTokens":       usage.get("prompt_tokens",     0),
                "CompletionTokens":   usage.get("completion_tokens", 0),
                "Cost_USD":           cost,
                "ResponseLength":     len(content) if content else 0,
                "QualityScore":       score,
                "CacheHit":           cache_hit,
                "SemanticSimilarity": round(sem_sim, 4),
            })
            print(f"    [{p['id']}] lat={latency:.2f}s  cost=${cost:.6f}"
                  f"  q={score}  hit={cache_hit}")

        except requests.exceptions.Timeout:
            row["Error"] = "Timeout"
            print(f"    [{p['id']}] TIMEOUT")
        except Exception as e:
            row["Error"] = str(e)
            print(f"    [{p['id']}] ERREUR : {e}")

        rows.append(row)
    return rows

# ─────────────────────────────────────────────
# 8. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
CSV_COLUMNS = [
    "Model", "Provider", "PromptID", "PromptGroup", "PromptType", "Scenario",
    "Latency_s", "PromptTokens", "CompletionTokens", "Cost_USD", "ResponseLength",
    "QualityScore", "CacheHit", "SemanticSimilarity", "Error", "Prompt",
]

def run_benchmark():
    # Initialise le CSV
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_COLUMNS)

    all_models = (
        [{"name": m, "provider": "local"} for m in LOCAL_MODELS]
        + API_MODELS
    )

    for mc in all_models:
        mn, pv = mc["name"], mc["provider"]
        print(f"\n{'='*60}\n  Modèle : {mn}  [{pv}]\n{'='*60}")

        for scenario in ["no_cache", "exact_cache", "semantic_cache"]:
            print(f"\n  ── Scénario : {scenario} ──")
            rows = run_scenario(mn, pv, scenario)

            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for row in rows:
                    w.writerow([row[c] for c in CSV_COLUMNS])

    print(f"\n✅  Benchmark terminé → {CSV_FILE}")

# ─────────────────────────────────────────────
# 9. ANALYSE DES RÉSULTATS
# ─────────────────────────────────────────────
def analyze_results():
    """
    Affiche un résumé structuré pour le papier :
    latence, coût, qualité, taux de hit, temps économisé.
    """
    try:
        import pandas as pd
    except ImportError:
        print("pip install pandas  pour l'analyse.")
        return

    df = pd.read_csv(CSV_FILE)
    nc = df[df["Scenario"] == "no_cache"]

    print("\n" + "="*60)
    print("RÉSULTATS — Latence moyenne (no_cache, secondes)")
    print("="*60)
    print(nc.groupby(["Model", "Provider"])["Latency_s"]
            .mean().round(3).sort_values().to_string())

    print("\n" + "="*60)
    print("RÉSULTATS — Coût total estimé (no_cache, USD)")
    print("="*60)
    print(nc.groupby(["Model", "Provider"])["Cost_USD"]
            .sum().round(6).sort_values().to_string())

    print("\n" + "="*60)
    print("RÉSULTATS — Qualité pédagogique moyenne /5 (no_cache)")
    print("="*60)
    q = nc[nc["QualityScore"] > 0].groupby("Model")["QualityScore"].mean()
    print(q.round(2).sort_values(ascending=False).to_string())

    print("\n" + "="*60)
    print("RÉSULTATS — Qualité cache hits vs no_cache")
    print("="*60)
    for sc in ["exact_cache", "semantic_cache"]:
        hits = df[(df["Scenario"] == sc) & df["CacheHit"] & (df["QualityScore"] > 0)]
        if not hits.empty:
            print(f"\n  {sc} — qualité moyenne sur les hits :")
            print(hits.groupby("Model")["QualityScore"].mean().round(2).to_string())

    print("\n" + "="*60)
    print("RÉSULTATS — Taux de hit & temps économisé")
    print("="*60)
    for model in df["Model"].unique():
        md      = df[df["Model"] == model]
        lat_avg = md[md["Scenario"] == "no_cache"]["Latency_s"].mean()
        for sc in ["exact_cache", "semantic_cache"]:
            sdf  = md[md["Scenario"] == sc]
            hr   = sdf["CacheHit"].mean() * 100
            hits = int(sdf["CacheHit"].sum())
            n    = len(sdf)
            saved= round(lat_avg * hits, 1)
            print(f"  {model:<40} {sc:<18}"
                  f" hit={hr:.0f}% ({hits}/{n})  économisé≈{saved}s")

    print("\n" + "="*60)
    print("RÉSULTATS — Latence avec cache (sémantique, hits seulement)")
    print("="*60)
    sem_hits = df[(df["Scenario"] == "semantic_cache") & df["CacheHit"]]
    if not sem_hits.empty:
        print(sem_hits.groupby("Model")["SemanticSimilarity"].mean().round(3).to_string())

    # Export résumé
    summary_file = "benchmark_v3_summary.csv"
    summary = df.groupby(["Model", "Provider", "Scenario"]).agg(
        AvgLatency   =("Latency_s",    "mean"),
        TotalCost    =("Cost_USD",     "sum"),
        AvgQuality   =("QualityScore", lambda x: x[x > 0].mean()),
        HitRate      =("CacheHit",     "mean"),
        AvgSemSim    =("SemanticSimilarity", "mean"),
    ).round(4).reset_index()
    summary.to_csv(summary_file, index=False)
    print(f"\n📊  Résumé exporté → {summary_file}")


# ─────────────────────────────────────────────
# 10. POINT D'ENTRÉE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_benchmark()
    analyze_results()
