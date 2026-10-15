import time

from services import cache, semantic_cache
from services.llm import ask_llm, get_model


async def answer_question(question: str) -> dict:
    """The gateway's main flow: check the cache first, only ask the LLM on a miss.

    Later days add steps here (semantic cache, Gemini fallback) without touching the route.
    """
    model = get_model()

    # 1. Exact cache: same (normalized) question asked before?
    start = time.perf_counter()
    cached = await cache.get_cached(question, model)
    if cached is not None:
        return {
            **cached,
            "prompt_tokens": 0,  # we didn't pay Groq for this one
            "completion_tokens": 0,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),  # real lookup time
            "cache_hit": True,
        }

    # 2. Miss: ask the LLM (raises LLMError on failure -> route returns 502), then save it.
    result = await ask_llm(question)
    await cache.set_cached(question, model, result)
    return {**result, "cache_hit": False}


async def remember_answer(question: str, result: dict) -> None:
    """Runs in the background AFTER the user has their answer.

    Fresh LLM answers are stored with their embedding, so Day 6 can match similar questions.
    """
    if result["cache_hit"]:
        return  # already stored when it was first answered
    await semantic_cache.save_entry(question, result)
