from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, Union
import time

from open_tutorai.services.cache_service import (
    get_cache,
    store_cache
)

from open_tutorai.services.llm_service import call_llm


router = APIRouter(prefix="/api/cache", tags=["Cache"])


# ─────────────────────────────
# SCHEMAS
# ─────────────────────────────

class PromptRequest(BaseModel):
    prompt: str
    model: str = "mistral"

   


class PromptResponse(BaseModel):
    hit: bool
    source: Literal["exact_cache", "semantic_cache", "ollama", "openai", "miss"]
    model: str
    model_type: Literal["local", "api"]
    latency: float
    response: str


# ─────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────

LOCAL_MODELS = {
    "mistral",
    "mistral:7b",
    "llama3",
    "llama3:8b",
    "phi"
}

API_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5.3",
    "o1",
    "o3-mini"
}


def resolve_model_type(model: str) -> str:
    """
    Détermine si le modèle est local (Ollama) ou API.
    """
    if model in LOCAL_MODELS:
        return "local"

    if model in API_MODELS or model.startswith(("gpt", "o1", "o3")):
        return "api"

    # fallback sécurisé
    return "local"


# ─────────────────────────────
# SOURCE MAPPING
# ─────────────────────────────

def map_source(cache_type: str, model_type: str):
    if cache_type == "exact":
        return "exact_cache"
    if cache_type == "semantic":
        return "semantic_cache"
    if model_type == "api":
        return "openai"
    return "ollama"


# ─────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────

@router.post("/generate", response_model=PromptResponse)
async def generate(request: PromptRequest):

    start = time.time()
    model_type = resolve_model_type(request.model)

    # =========================
    # 1. CACHE CHECK
    # =========================
    cached, cache_type = await get_cache(
        model=request.model,
        prompt=request.prompt
    )

    if cached:
        return PromptResponse(
            hit=True,
            source=map_source(cache_type, model_type),
            model=request.model,
            model_type=model_type,
            latency=round(time.time() - start, 3),
            response=str(cached),
        )

    # =========================
    # 2. LLM CALL (UNIFIED)
    # =========================

    try:
        llm_response = await call_llm(
            model=request.model,
            prompt=request.prompt
        )

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    # =========================
    # 3. NORMALISATION RESPONSE
    # =========================

    if isinstance(llm_response, dict):
        content = (
            llm_response.get("response")
            or llm_response.get("message", "")
            or llm_response.get("content", "")
        )
    else:
        content = str(llm_response)

    # =========================
    # 4. STORE CACHE (EXACT + SEMANTIC)
    # =========================

    await store_cache(
        model=request.model,
        prompt=request.prompt,
        response=content
    )

    # =========================
    # 5. FINAL RESPONSE
    # =========================

    return PromptResponse(
        hit=False,
        source=map_source("miss", model_type),
        model=request.model,
        model_type=model_type,
        latency=round(time.time() - start, 3),
        response=content,
    )
