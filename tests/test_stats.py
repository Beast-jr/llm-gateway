import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import services.stats
from api.main import app
from models.call import Call
from services.auth import create_user, current_user
from services.call_log import calculate_cost
from services.llm import get_model
from services.stats import make_headline, percentile

client = TestClient(app)
NOW = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def fixed_now(monkeypatch):
    monkeypatch.setattr(services.stats, "utcnow", lambda: NOW)


def add_call(session, kind="miss", provider="groq", latency=100.0, prompt=10, completion=20, cost=0.001,
             saved=0.0, truncated=False, reason=None, user_id=1, age_days=0.0):
    """One row in the calls table. kind: "miss", "exact" or "semantic"."""
    hit = kind != "miss"
    session.add(Call(
        question="SECRET QUESTION TEXT", answer="SECRET ANSWER TEXT", model="m", provider="cache" if hit else provider,
        prompt_tokens=0 if hit else prompt, completion_tokens=0 if hit else completion, latency_ms=latency,
        cost_usd=0.0 if hit else cost, cache_hit=hit, cache_type=kind if hit else None,
        saved_cost_usd=saved, truncated=truncated, fallback_reason=reason, user_id=user_id,
        created_at=NOW - timedelta(days=age_days, minutes=1),
    ))


@pytest.fixture
def traffic(test_db):
    """14 requests with numbers picked so every statistic can be checked by hand:
    2 exact hits (20ms, 40ms), 1 semantic hit (60ms), 10 Groq answers (100ms ... 1000ms, the slowest cut off),
    and 1 Gemini answer (600ms) that happened because Groq timed out."""
    with test_db() as session:
        add_call(session, "exact", latency=20, saved=0.001)
        add_call(session, "exact", latency=40, saved=0.002)
        add_call(session, "semantic", latency=60, saved=0.003)
        for i in range(1, 11):
            add_call(session, latency=100 * i, truncated=(i == 10))  # 10 + 20 tokens, $0.001 each
        add_call(session, provider="gemini", latency=600, prompt=5, completion=5, cost=0.002, reason="groq_timeout")
        session.commit()


def get_stats(**params):
    res = client.get("/stats", params=params)
    assert res.status_code == 200, res.text
    return res.json()


# ---------- the maths ----------

def test_request_counts_and_percentages(traffic):
    r = get_stats()["requests"]
    assert r["total"] == 14
    assert r["exact_hits"] == {"count": 2, "percent": 14.3}
    assert r["semantic_hits"] == {"count": 1, "percent": 7.1}
    assert r["misses"] == {"count": 11, "percent": 78.6}
    assert r["cache_hit_rate_percent"] == 21.4


def test_providers_and_fallbacks(traffic):
    body = get_stats()
    assert body["by_provider"] == {"groq": 10, "gemini": 1, "cache": 3}
    assert body["fallbacks"] == {"total": 1, "by_reason": {"groq_timeout": 1}}


def test_latency_average_median_and_p95_per_group(traffic):
    lat = get_stats()["latency_ms"]
    assert lat["exact_hits"] == {"count": 2, "avg_ms": 30.0, "median_ms": 30.0, "p95_ms": 40.0}
    assert lat["semantic_hits"] == {"count": 1, "avg_ms": 60.0, "median_ms": 60.0, "p95_ms": 60.0}
    # 100, 200, ... 1000: the middle is between 500 and 600; p95 of 10 values is the 10th
    assert lat["groq"] == {"count": 10, "avg_ms": 550.0, "median_ms": 550.0, "p95_ms": 1000.0}
    assert lat["gemini"] == {"count": 1, "avg_ms": 600.0, "median_ms": 600.0, "p95_ms": 600.0}


def test_tokens_cost_saved_and_truncated(traffic):
    body = get_stats()
    assert body["tokens"] == {"prompt": 105, "completion": 205, "total": 310}  # cache hits spent no tokens
    assert body["cost_usd"] == {"spent": 0.012, "saved": 0.006}
    assert body["truncated_answers"] == 1


def test_headline(traffic):
    # 3 of 14 = 21%; cache hits' median is 40ms; LLM answers' median is 600ms -> 15x
    assert get_stats()["headline"] == "Cache answered 21% of requests, saved $0.0060, cached answers were 15x faster (40ms vs 600ms)."


def test_empty_database_has_no_division_by_zero():
    body = get_stats()
    assert body["requests"]["total"] == 0
    assert body["requests"]["cache_hit_rate_percent"] == 0.0
    assert body["latency_ms"]["groq"]["median_ms"] is None
    assert body["headline"] == "No requests in the last 7 days."


def test_old_exact_hits_without_a_cache_type_still_count(test_db):
    with test_db() as session:  # Day 4 style row: cache_hit true, nothing in cache_type, provider still "groq"
        session.add(Call(question="q", answer="a", model="m", provider="groq", prompt_tokens=0, completion_tokens=0,
                         latency_ms=25, cost_usd=0, cache_hit=True, created_at=NOW - timedelta(hours=1)))
        session.commit()
    body = get_stats()
    assert body["requests"]["exact_hits"]["count"] == 1
    assert body["by_provider"]["cache"] == 1 and body["by_provider"]["groq"] == 0


def test_percentile_is_nearest_rank():
    assert percentile([], 95) is None
    assert percentile([7], 95) == 7
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile([30, 10, 20], 50) == 20  # order of the input doesn't matter


def test_headline_edge_cases():
    assert make_headline(7, 5, 0, 0, [], [900]) == "No cache hits yet in 5 requests over the last 7 days."
    assert "1.5x faster (400ms vs 600ms)" in make_headline(7, 10, 5, 0.5, [400], [600])
    assert "(20ms vs 1.2s)" in make_headline(7, 10, 5, 0.5, [20], [1200])
    assert make_headline(7, 4, 2, 0.1, [], [500]).endswith("saved $0.1000.")  # nothing to compare speed with


# ---------- who sees what, and for how long ----------

def test_no_question_or_answer_text_ever_appears(traffic):
    res = client.get("/stats")

    assert "SECRET" not in res.text

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)

    assert not [k for k in keys(res.json()) if k in ("question", "answer", "matched_question")]


def test_days_filter(test_db):
    with test_db() as session:
        add_call(session, age_days=1)
        add_call(session, age_days=1)
        add_call(session, age_days=10)
        session.commit()

    assert get_stats()["requests"]["total"] == 2  # default is 7 days
    assert get_stats(days=7)["requests"]["total"] == 2
    assert get_stats(days=30)["requests"]["total"] == 3
    assert get_stats(days=1)["requests"]["total"] == 0  # both recent rows are a little over 1 day old
    period = get_stats(days=30)["period"]
    assert period["days"] == 30 and period["scope"] == "all"
    assert period["until"].startswith("2026-03-10T12:00:00")
    assert period["since"].startswith("2026-02-08T12:00:00")  # exactly 30 days earlier


@pytest.mark.parametrize("days", [0, -1, 366, "abc"])
def test_bad_days_value_is_422(days):
    assert client.get("/stats", params={"days": days}).status_code == 422


def test_mine_only_counts_the_callers_own_requests(test_db):
    with test_db() as session:
        add_call(session, user_id=1)
        add_call(session, user_id=1)
        add_call(session, user_id=2)
        add_call(session, kind="exact", user_id=2, saved=0.5)
        session.commit()

    everyone = get_stats()
    mine = get_stats(mine="true")
    assert everyone["requests"]["total"] == 4 and everyone["period"]["scope"] == "all"
    assert mine["requests"]["total"] == 2 and mine["period"]["scope"] == "mine"
    assert mine["cost_usd"]["saved"] == 0  # user 2's saved money is not in user 1's numbers


# ---------- login ----------

def test_stats_needs_a_valid_key(test_db):
    app.dependency_overrides.pop(current_user, None)  # undo the conftest shortcut: use real login

    assert client.get("/stats").status_code == 401
    assert client.get("/stats", headers={"Authorization": "Bearer gw_notarealkey"}).status_code == 401

    _, key = create_user("alice", 500)
    ok = client.get("/stats", headers={"Authorization": f"Bearer {key}"})
    assert ok.status_code == 200


def test_stats_failure_is_a_clean_503(monkeypatch):
    def broken(days, user_id=None):
        raise ConnectionError("database is down")

    monkeypatch.setattr("api.main.compute_stats", broken)
    res = client.get("/stats")
    assert res.status_code == 503 and "temporarily unavailable" in res.json()["detail"]


# ---------- saved money is recorded when the hit happens ----------

def chat(question="What is RAG?", **extra):
    return client.post("/chat", json={"question": question, **extra})


def calls(test_db):
    with test_db() as session:
        return session.query(Call).order_by(Call.id).all()


def test_exact_hit_records_what_the_original_answer_cost(fake_llm, test_db):
    chat()
    chat()
    first, second = calls(test_db)
    assert first.saved_cost_usd == 0
    assert second.saved_cost_usd == pytest.approx(calculate_cost(get_model(), 5, 7))  # the fake answer's 5 + 7 tokens
    assert second.saved_cost_usd > 0 and second.cost_usd == 0


def test_semantic_hit_records_the_saved_answers_cost_and_remembers_it_for_later_exact_hits(fake_llm, fake_lookup, test_db):
    fake_lookup.match = {"question": "What is RAG?", "answer": "saved", "similarity": 0.99,
                         "prompt_tokens": 100, "completion_tokens": 200}
    chat("Explain RAG to me")  # semantic hit
    fake_lookup.match = None
    chat("Explain RAG to me")  # now an exact hit, straight from Redis

    semantic, exact = calls(test_db)
    expected = calculate_cost(get_model(), 100, 200)
    assert (semantic.cache_type, exact.cache_type) == ("semantic", "exact")
    assert semantic.saved_cost_usd == pytest.approx(expected)
    assert exact.saved_cost_usd == pytest.approx(expected)  # the original cost travelled with the Redis copy
    assert fake_llm.calls == 0


def test_truncated_answers_are_logged_and_counted(monkeypatch, test_db):
    from tests.conftest import FAKE_RESULT

    async def cut_off(question):
        return {**FAKE_RESULT, "truncated": True}

    monkeypatch.setattr("services.gateway.ask_llm", cut_off)
    chat()
    assert calls(test_db)[0].truncated is True
    assert get_stats()["truncated_answers"] == 1


def test_stats_after_real_chat_traffic(fake_llm):
    chat("Question one?")
    chat("Question one?")  # exact hit
    chat("Question two?")
    body = get_stats()

    assert body["requests"]["total"] == 3
    assert body["requests"]["exact_hits"]["count"] == 1
    assert body["cost_usd"]["saved"] == pytest.approx(calculate_cost(get_model(), 5, 7), abs=1e-6)
    assert body["headline"].startswith("Cache answered 33% of requests")
    assert json.dumps(body).count("Question") == 0  # still no text
