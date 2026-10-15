import asyncio
import logging
import time

import config
from services import breaker, budget, cache, embeddings, match_guard, semantic_cache
from services.call_log import calculate_cost
from services.gemini import ask_gemini
from services.llm import LLMError, ask_llm, get_model

logger = logging.getLogger(__name__)


def _cache_scope(user_id: int | None) -> str | None:
    """Redis key scope: 'user:5' when caches are per user, None when everyone shares one cache."""
    return f"user:{user_id}" if config.CACHE_SCOPE == "per_user" and user_id is not None else None


def _lookup_user_id(user_id: int | None) -> int | None:
    """Which user's rows the semantic lookup may match (None = anyone's)."""
    return user_id if config.CACHE_SCOPE == "per_user" else None


async def _semantic_lookup(question: str, model: str, found: dict, user_id: int | None) -> dict | None:
    """Embed the question, then find the closest saved one. The vector goes into `found`
    so the caller can reuse it even if the lookup later fails or times out."""
    t = time.perf_counter()
    found["vector"] = await embeddings.embed(question)
    found["embed_ms"] = round((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    match = await asyncio.to_thread(semantic_cache.find_closest, found["vector"], model, _lookup_user_id(user_id))
    found["db_ms"] = round((time.perf_counter() - t) * 1000)
    return match


async def _try_semantic(question: str, model: str, found: dict, user_id: int | None) -> dict | None:
    """The whole semantic step under a time limit. Never raises: any problem means 'no match'."""
    try:
        return await asyncio.wait_for(_semantic_lookup(question, model, found, user_id), config.SEMANTIC_TIMEOUT_SECONDS)
    except Exception as e:  # includes TimeoutError
        logger.warning("Semantic lookup failed, asking the LLM instead: %r", e)
        return None


def _is_reusable(question: str, match: dict | None) -> bool:
    """Close enough in meaning (cutoff) AND the numbers/years agree (guard)?"""
    if match is None or match["similarity"] < config.SEMANTIC_CUTOFF:
        return False
    reason = match_guard.check_numbers(question, match["question"])
    if reason:
        logger.info("Semantic match %.3f blocked by guard (%s): %r vs %r",
                    match["similarity"], reason, question, match["question"])
    return reason is None


def _short(text: object, limit: int = 200) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "..."


async def _ask_with_fallback(question: str) -> dict:
    """Groq first. Gemini ONLY if Groq fails, can't connect, or hangs; never just because an answer is long.

    After BREAKER_FAILURES Groq failures in a row, Groq is skipped for a while (see services/breaker.py).
    Raises LLMError if both providers fail (the route turns that into a 502).
    """
    reason = None
    if breaker.groq_breaker.is_open():
        reason = "circuit_open"
    else:
        try:
            result = await asyncio.wait_for(ask_llm(question), config.GROQ_TOTAL_TIMEOUT_SECONDS)
        except TimeoutError:
            reason = "groq_timeout"
        except LLMError as e:
            reason = e.reason
        except Exception:  # a bug or surprise inside the Groq call must not take the whole service down
            logger.exception("Unexpected error while calling Groq")
            reason = "groq_error"
        else:
            breaker.groq_breaker.record_success()
            return {**result, "fallback_reason": None}
        breaker.groq_breaker.record_failure()

    logger.warning("Groq not used (%s), asking Gemini instead", reason)
    try:
        result = await ask_gemini(question)
    except Exception as e:
        raise LLMError(f"Both providers failed. Groq: {reason}. Gemini: {_short(e)}", "both_failed") from e
    return {**result, "fallback_reason": reason}


async def answer_question(question: str, use_semantic: bool = True, user_id: int | None = None,
                          daily_budget: int | None = None) -> dict:
    """The gateway's main flow: exact cache -> semantic cache -> LLM.

    use_semantic=False skips the semantic step (the exact cache still applies).
    user_id decides whose saved answers may be reused (see CACHE_SCOPE in config.py).
    daily_budget is the user's token limit; cache hits cost 0 tokens so they are always allowed.
    Raises budget.BudgetExceeded (route -> 429) before calling the LLM if the budget is used up.
    Later days add steps here (Gemini fallback) without touching the route.
    """
    model = get_model()
    scope = _cache_scope(user_id)
    start = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    # Ask Redis how much the user has used today, at the same time as the cache lookups below.
    usage_task = asyncio.create_task(budget.get_used(user_id)) if user_id is not None else None

    async def used_today() -> int | None:
        """For a real Groq call: wait for the counter (up to BUDGET_TIMEOUT_SECONDS) so the budget check is not skipped."""
        return await usage_task if usage_task is not None else None

    async def used_today_quick() -> int | None:
        """For free cache hits there is nothing to enforce, so don't make the user wait for the counter."""
        if usage_task is None:
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(usage_task), config.REDIS_TIMEOUT_SECONDS)
        except TimeoutError:
            return None

    # 1. Exact cache: same (normalized) question asked before?
    cached = await cache.get_cached(question, model, scope)
    redis_ms = round((time.perf_counter() - start) * 1000)
    if cached is not None:
        logger.info("timing: redis=%sms -> exact hit", redis_ms)
        used = await used_today_quick()
        return {
            **cached,
            "prompt_tokens": 0,  # we didn't pay Groq for this one
            "completion_tokens": 0,
            "latency_ms": elapsed_ms(),  # real lookup time
            "cache_hit": True,
            "provider": "cache",  # nobody was asked; the model field still names who first wrote the answer
            "fallback_reason": None,
            "cache_type": "exact",
            "similarity": None,
            "matched_question": None,
            "truncated": False,
            "tokens_used_today": used,
            "daily_budget": daily_budget,
            # The money this hit saved = what the saved answer cost when it was first produced.
            "saved_cost_usd": calculate_cost(cached.get("model", model), cached.get("prompt_tokens", 0),
                                             cached.get("completion_tokens", 0)),
        }

    # 2. Semantic cache: a *similar* question answered before? (embedded only once per request)
    found: dict = {}
    match = None
    if use_semantic:  # the caller can opt out per request
        match = await _try_semantic(question, model, found, user_id)
        logger.info("timing: redis=%sms embed=%sms db=%sms (- = did not finish)",
                    redis_ms, found.get("embed_ms", "-"), found.get("db_ms", "-"))
    if _is_reusable(question, match):
        used = await used_today_quick()
        return {
            "answer": match["answer"],
            "model": model,
            "provider": "cache",  # nobody was asked; the saved answer came from our cache
            "fallback_reason": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": elapsed_ms(),
            "cache_hit": True,
            "cache_type": "semantic",
            "similarity": round(match["similarity"], 4),
            "matched_question": match["question"],  # the saved question whose answer we reused
            "truncated": False,
            "tokens_used_today": used,
            "daily_budget": daily_budget,
            "saved_cost_usd": calculate_cost(model, match.get("prompt_tokens", 0), match.get("completion_tokens", 0)),
            # Kept so the exact-cache copy of this answer still remembers what it originally cost:
            "_original_tokens": (match.get("prompt_tokens", 0), match.get("completion_tokens", 0)),
        }

    # 3. Miss: this one costs real tokens, so check the budget first (raises BudgetExceeded -> 429).
    used = await used_today()
    if daily_budget is not None:
        budget.check(used, daily_budget)

    # Ask the LLM (Groq, or Gemini if Groq is down; LLMError if both fail -> route returns 502), then save the
    # answer and count the tokens. We only know the tokens AFTER the answer, so one request can go a little over.
    result = await _ask_with_fallback(question)
    tokens = result["prompt_tokens"] + result["completion_tokens"]
    truncated = bool(result.get("truncated"))
    if truncated:
        logger.warning("Answer was cut off at the length limit (%s completion tokens); it will NOT be cached",
                       result["completion_tokens"])
    # A cut-off answer is never saved: the next person asking would get half an answer. It still costs tokens.
    _, used_now = await asyncio.gather(
        asyncio.sleep(0) if truncated else cache.set_cached(question, model, result, scope),
        budget.add_usage(user_id, tokens) if user_id is not None else asyncio.sleep(0),
    )
    if used_now is None and used is not None:  # the counter write failed: best guess
        used_now = used + tokens
    return {
        **result,
        "cache_hit": False,
        "cache_type": None,
        "matched_question": None,
        # Even on a miss, keep the nearest score: it shows near-misses in real traffic.
        "similarity": round(match["similarity"], 4) if match is not None else None,
        "truncated": truncated,
        "saved_cost_usd": 0.0,
        "tokens_used_today": used_now,
        "daily_budget": daily_budget,
        "_vector": found.get("vector"),  # reused for the semantic save, never sent to the user
    }


async def remember_answer(question: str, result: dict, user_id: int | None = None) -> None:
    """Runs in the background AFTER the user has their answer."""
    if result.get("truncated"):
        return  # never keep a cut-off answer in the semantic cache either
    if result.get("cache_type") == "exact":
        return  # already stored when it was first answered
    if result.get("cache_type") == "semantic":
        # Next time this exact wording is asked, Redis answers it directly.
        prompt_tokens, completion_tokens = result["_original_tokens"]
        await cache.set_cached(question, result["model"], {
            **{k: result[k] for k in ("answer", "model", "provider", "latency_ms")},
            "prompt_tokens": prompt_tokens,  # the ORIGINAL cost (the response itself shows 0), for saved-money stats
            "completion_tokens": completion_tokens,
        }, _cache_scope(user_id))
        return
    # Fresh LLM answer: store it with its embedding so similar questions can match it.
    await semantic_cache.save_entry(question, result, result.get("_vector"), user_id, _lookup_user_id(user_id),
                                    model=get_model())  # same namespace the lookup uses, whoever answered
