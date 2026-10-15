import asyncio
import logging
import time

import config
from services import cache, embeddings, semantic_cache
from services.llm import ask_llm, get_model

logger = logging.getLogger(__name__)


async def _semantic_lookup(question: str, model: str, found: dict) -> dict | None:
    """Embed the question, then find the closest saved one. The vector goes into `found`
    so the caller can reuse it even if the lookup later fails or times out."""
    t = time.perf_counter()
    found["vector"] = await embeddings.embed(question)
    found["embed_ms"] = round((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    match = await asyncio.to_thread(semantic_cache.find_closest, found["vector"], model)
    found["db_ms"] = round((time.perf_counter() - t) * 1000)
    return match


async def _try_semantic(question: str, model: str, found: dict) -> dict | None:
    """The whole semantic step under a time limit. Never raises: any problem means 'no match'."""
    try:
        return await asyncio.wait_for(_semantic_lookup(question, model, found), config.SEMANTIC_TIMEOUT_SECONDS)
    except Exception as e:  # includes TimeoutError
        logger.warning("Semantic lookup failed, asking the LLM instead: %r", e)
        return None


async def answer_question(question: str) -> dict:
    """The gateway's main flow: exact cache -> semantic cache -> LLM.

    Later days add steps here (Gemini fallback) without touching the route.
    """
    model = get_model()
    start = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    # 1. Exact cache: same (normalized) question asked before?
    cached = await cache.get_cached(question, model)
    redis_ms = round((time.perf_counter() - start) * 1000)
    if cached is not None:
        logger.info("timing: redis=%sms -> exact hit", redis_ms)
        return {
            **cached,
            "prompt_tokens": 0,  # we didn't pay Groq for this one
            "completion_tokens": 0,
            "latency_ms": elapsed_ms(),  # real lookup time
            "cache_hit": True,
            "cache_type": "exact",
            "similarity": None,
        }

    # 2. Semantic cache: a *similar* question answered before? (embedded only once per request)
    found: dict = {}
    match = await _try_semantic(question, model, found)
    logger.info("timing: redis=%sms embed=%sms db=%sms (- = did not finish)",
                redis_ms, found.get("embed_ms", "-"), found.get("db_ms", "-"))
    if match is not None and match["similarity"] >= config.SEMANTIC_CUTOFF:
        return {
            "answer": match["answer"],
            "model": model,
            "provider": "cache",  # nobody was asked; the saved answer came from our cache
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": elapsed_ms(),
            "cache_hit": True,
            "cache_type": "semantic",
            "similarity": round(match["similarity"], 4),
        }

    # 3. Miss: ask the LLM (raises LLMError on failure -> route returns 502), then save it.
    result = await ask_llm(question)
    await cache.set_cached(question, model, result)
    return {
        **result,
        "cache_hit": False,
        "cache_type": None,
        "similarity": None,
        "_vector": found.get("vector"),  # reused for the semantic save, never sent to the user
    }


async def remember_answer(question: str, result: dict) -> None:
    """Runs in the background AFTER the user has their answer."""
    if result.get("cache_type") == "exact":
        return  # already stored when it was first answered
    if result.get("cache_type") == "semantic":
        # Next time this exact wording is asked, Redis answers it directly.
        await cache.set_cached(question, result["model"], {
            k: result[k] for k in ("answer", "model", "provider", "prompt_tokens", "completion_tokens", "latency_ms")
        })
        return
    # Fresh LLM answer: store it with its embedding so similar questions can match it.
    await semantic_cache.save_entry(question, result, result.get("_vector"))
