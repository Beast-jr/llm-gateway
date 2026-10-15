"""GET /stats: numbers about how the gateway is doing. AGGREGATES ONLY.

We never select the question or answer columns, so no text can leak out of here by accident.
Counts and sums are done in SQL; medians and percentiles are done in Python (SQLite, which the tests
use, has no percentile function, and the lists are just numbers).
"""
import math
import statistics
from datetime import datetime, timedelta, timezone

from sqlalchemy import case, func, select

from models.call import Call
from services.db import get_session, retry_once_if_dropped

# One word per row, worked out in SQL. Rows from before the cache_type column existed (cache_hit true,
# cache_type empty) were all exact hits, so "any other hit" = exact.
KIND = case(
    (Call.cache_hit.is_(False), "miss"),
    (Call.cache_type == "semantic", "semantic"),
    else_="exact",
)
# Who produced the answer. Every cache hit counts as "cache", even old rows that logged the original provider.
PROVIDER = case((Call.cache_hit.is_(True), "cache"), else_=Call.provider)


def utcnow() -> datetime:
    """The current time in UTC. Tests replace this."""
    return datetime.now(timezone.utc)


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile: the smallest value that at least pct% of the values are <= to."""
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[max(rank, 1) - 1]


def latency_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "avg_ms": None, "median_ms": None, "p95_ms": None}
    return {
        "count": len(values),
        "avg_ms": round(sum(values) / len(values), 2),
        "median_ms": round(statistics.median(values), 2),
        "p95_ms": round(percentile(values, 95), 2),
    }


def share(count: int, total: int) -> dict:
    return {"count": count, "percent": round(100 * count / total, 1) if total else 0.0}


def format_ms(ms: float) -> str:
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


def make_headline(days: int, total: int, hits: int, saved_usd: float, hit_latencies: list[float],
                  llm_latencies: list[float]) -> str:
    if total == 0:
        return f"No requests in the last {days} days."
    if hits == 0:
        return f"No cache hits yet in {total} requests over the last {days} days."
    text = f"Cache answered {100 * hits / total:.0f}% of requests, saved ${saved_usd:.4f}"
    if hit_latencies and llm_latencies:
        fast, slow = statistics.median(hit_latencies), statistics.median(llm_latencies)
        if fast > 0:
            ratio = slow / fast
            text += (f", cached answers were {ratio:.0f}x faster" if ratio >= 10 else
                     f", cached answers were {ratio:.1f}x faster") + f" ({format_ms(fast)} vs {format_ms(slow)})"
    return text + "."


def _where(since: datetime, user_id: int | None) -> list:
    conditions = [Call.created_at >= since]
    if user_id is not None:
        conditions.append(Call.user_id == user_id)
    return conditions


def _query(since: datetime, user_id: int | None) -> tuple[list, list]:
    with get_session() as session:
        sums = session.execute(
            select(
                KIND, PROVIDER, Call.fallback_reason,
                func.count(), func.sum(Call.prompt_tokens), func.sum(Call.completion_tokens),
                func.sum(Call.cost_usd), func.sum(Call.saved_cost_usd),
                func.sum(case((Call.truncated.is_(True), 1), else_=0)),
            ).where(*_where(since, user_id)).group_by(KIND, PROVIDER, Call.fallback_reason)
        ).all()
        latencies = session.execute(
            select(KIND, PROVIDER, Call.latency_ms).where(*_where(since, user_id))
        ).all()
    return sums, latencies


def compute_stats(days: int, user_id: int | None = None) -> dict:
    """Stats for the last `days` days (a rolling window ending now, in UTC). user_id limits it to one user."""
    until = utcnow()
    since = until - timedelta(days=days)
    sums, latencies = retry_once_if_dropped(lambda: _query(since, user_id))

    kinds = {"exact": 0, "semantic": 0, "miss": 0}
    providers = {"groq": 0, "gemini": 0, "cache": 0}
    fallbacks: dict[str, int] = {}
    prompt = completion = truncated = 0
    spent = saved = 0.0
    for kind, provider, reason, count, p_tok, c_tok, cost, saved_cost, n_truncated in sums:
        kinds[kind] += count
        providers[provider] = providers.get(provider, 0) + count
        if reason:
            fallbacks[reason] = fallbacks.get(reason, 0) + count
        prompt += p_tok or 0
        completion += c_tok or 0
        spent += cost or 0.0
        saved += saved_cost or 0.0
        truncated += n_truncated or 0

    groups: dict[str, list[float]] = {"exact": [], "semantic": [], "groq": [], "gemini": []}
    for kind, provider, latency in latencies:
        if kind in ("exact", "semantic"):
            groups[kind].append(latency)
        elif provider in groups:
            groups[provider].append(latency)

    total = sum(kinds.values())
    hits = kinds["exact"] + kinds["semantic"]
    hit_latencies = groups["exact"] + groups["semantic"]
    llm_latencies = groups["groq"] + groups["gemini"]
    return {
        "headline": make_headline(days, total, hits, saved, hit_latencies, llm_latencies),
        "period": {"days": days, "since": since, "until": until, "scope": "mine" if user_id is not None else "all"},
        "requests": {
            "total": total,
            "exact_hits": share(kinds["exact"], total),
            "semantic_hits": share(kinds["semantic"], total),
            "misses": share(kinds["miss"], total),
            "cache_hit_rate_percent": round(100 * hits / total, 1) if total else 0.0,
        },
        "by_provider": providers,
        "fallbacks": {"total": sum(fallbacks.values()), "by_reason": fallbacks},
        "latency_ms": {
            "exact_hits": latency_summary(groups["exact"]),
            "semantic_hits": latency_summary(groups["semantic"]),
            "groq": latency_summary(groups["groq"]),
            "gemini": latency_summary(groups["gemini"]),
        },
        "tokens": {"prompt": prompt, "completion": completion, "total": prompt + completion},
        "cost_usd": {"spent": round(spent, 6), "saved": round(saved, 6)},
        "truncated_answers": truncated,
    }
