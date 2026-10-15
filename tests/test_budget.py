import asyncio
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import config
import services.budget
import services.cache
from api.main import app
from models.call import Call
from services.auth import AuthUser, current_user
from services.budget import seconds_until_reset, usage_key

client = TestClient(app)
TOKENS_PER_ANSWER = 12  # the fake LLM answers with 5 prompt + 7 completion tokens


class Clock:
    """A fake 'now' that tests move by hand, so nothing ever waits for a real midnight."""
    now = datetime(2026, 3, 1, 22, 30, 0, tzinfo=timezone.utc)  # 1h30m before midnight UTC


@pytest.fixture
def clock(monkeypatch):
    Clock.now = datetime(2026, 3, 1, 22, 30, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(services.budget, "utcnow", lambda: Clock.now)
    return Clock


@pytest.fixture
def as_user():
    """Log in as a user with a chosen id and daily budget (instead of the default big-budget test user)."""
    def login(budget, user_id=1):
        app.dependency_overrides[current_user] = lambda: AuthUser(user_id, f"user{user_id}", budget, True)
    return login


def ask(question):
    return client.post("/chat", json={"question": question})


def counter(user_id=1, now=None):
    """Read the raw Redis counter, the way the app writes it."""
    redis = services.cache._get_redis()
    return asyncio.run(redis.get(usage_key(user_id, now or Clock.now)))


def test_under_budget_is_200_and_reports_usage(fake_llm, clock, as_user):
    as_user(500)
    body = ask("What is RAG?").json()
    assert body["answer"] == fake_llm.result["answer"]
    assert body["tokens_used_today"] == TOKENS_PER_ANSWER
    assert body["daily_budget"] == 500


def test_usage_adds_up_in_redis(fake_llm, clock, as_user):
    as_user(500)
    ask("First question?")
    body = ask("A totally different question?").json()
    assert body["tokens_used_today"] == 2 * TOKENS_PER_ANSWER
    assert counter() == str(2 * TOKENS_PER_ANSWER)


def test_counter_key_is_per_user_per_utc_day_and_expires(fake_llm, clock, as_user):
    as_user(500)
    ask("What is RAG?")
    assert services.budget.usage_key(1, Clock.now) == "budget:1:2026-03-01"

    redis = services.cache._get_redis()
    ttl = asyncio.run(redis.ttl("budget:1:2026-03-01"))
    assert 47 * 3600 < ttl <= 48 * 3600  # about 48 hours, then it cleans itself up


def test_over_budget_is_429_with_retry_after(fake_llm, clock, as_user):
    as_user(20)
    assert ask("Question one?").status_code == 200  # 12 used
    assert ask("Question two?").status_code == 200  # 24 used: allowed, because 12 < 20 when it started
    calls_before = fake_llm.calls

    res = ask("Question three?")  # 24 >= 20 -> blocked
    assert res.status_code == 429
    assert "Daily token budget used up (24 of 20 tokens)" in res.json()["detail"]
    assert "Cached answers still work" in res.json()["detail"]
    assert res.headers["retry-after"] == str(90 * 60)  # 22:30 -> midnight = 1h30m
    assert fake_llm.calls == calls_before  # Groq was NOT called


def test_one_request_may_overshoot_the_budget(fake_llm, clock, as_user):
    as_user(5)  # smaller than a single answer (12 tokens)
    first = ask("Question one?")
    assert first.status_code == 200  # we only learn the cost after answering
    assert first.json()["tokens_used_today"] == TOKENS_PER_ANSWER
    assert ask("Question two?").status_code == 429


def test_using_exactly_the_budget_is_blocked(fake_llm, clock, as_user):
    as_user(TOKENS_PER_ANSWER)
    assert ask("Question one?").status_code == 200
    assert ask("Question two?").status_code == 429


def test_exact_cache_hit_is_free_even_when_over_budget(fake_llm, clock, as_user):
    as_user(500)
    ask("What is RAG?")  # answered by Groq and saved
    as_user(5)  # now the same user is over budget

    assert ask("A brand new question?").status_code == 429
    hit = ask("What is RAG?")
    assert hit.status_code == 200
    assert hit.json()["cache_type"] == "exact"
    assert hit.json()["tokens_used_today"] == TOKENS_PER_ANSWER  # unchanged: hits cost nothing
    assert counter() == str(TOKENS_PER_ANSWER)


def test_semantic_cache_hit_is_free_even_when_over_budget(fake_llm, fake_lookup, clock, as_user):
    as_user(500)
    ask("Something else entirely?")  # uses 12 tokens
    as_user(5)
    fake_lookup.match = {"question": "What is RAG?", "answer": "saved answer", "similarity": 0.99}

    res = ask("Explain RAG to me")
    assert res.status_code == 200 and res.json()["cache_type"] == "semantic"
    assert counter() == str(TOKENS_PER_ANSWER)  # still only the first request's tokens


def test_new_utc_day_starts_fresh(fake_llm, clock, as_user):
    as_user(20)
    ask("Question one?")
    ask("Question two?")
    assert ask("Question three?").status_code == 429

    clock.now = datetime(2026, 3, 2, 0, 0, 1, tzinfo=timezone.utc)  # one second past midnight UTC
    res = ask("Question four?")
    assert res.status_code == 200
    assert res.json()["tokens_used_today"] == TOKENS_PER_ANSWER  # yesterday's 24 no longer counts
    assert counter(now=datetime(2026, 3, 1, tzinfo=timezone.utc)) == str(2 * TOKENS_PER_ANSWER)  # yesterday kept


def test_each_user_has_their_own_budget(fake_llm, clock, as_user):
    as_user(20, user_id=1)
    ask("Question one?")
    ask("Question two?")
    assert ask("Question three?").status_code == 429

    as_user(20, user_id=2)
    res = ask("Question four?")
    assert res.status_code == 200
    assert res.json()["tokens_used_today"] == TOKENS_PER_ANSWER


def test_redis_down_fails_open(fake_llm, fake_redis, clock, as_user, caplog):
    as_user(1)  # would be blocked immediately if the counter worked
    fake_redis.connected = False
    res = ask("What is RAG?")

    assert res.status_code == 200
    assert res.json()["tokens_used_today"] is None  # we can't know
    assert res.json()["daily_budget"] == 1
    assert "fail open" in caplog.text


def test_blocked_requests_are_not_sent_to_groq_or_logged_as_calls(fake_llm, clock, as_user, test_db):
    as_user(1)
    ask("Question one?")
    ask("Question two?")
    with test_db() as session:
        assert session.query(Call).count() == 1  # only the answered one


def test_seconds_until_reset():
    utc = timezone.utc
    assert seconds_until_reset(datetime(2026, 3, 1, 23, 59, 59, tzinfo=utc)) == 1
    assert seconds_until_reset(datetime(2026, 3, 1, 0, 0, 0, tzinfo=utc)) == 24 * 3600
    assert seconds_until_reset(datetime(2026, 12, 31, 12, 0, 0, tzinfo=utc)) == 12 * 3600  # across a year end


class SlowCounterRedis:
    """A Redis that takes 0.2s to answer. The budget counter says the user is far over their limit."""

    async def get(self, key):
        await asyncio.sleep(0.2)
        return "999" if key.startswith("budget:") else None


def test_slow_counter_read_does_not_skip_the_budget_check(fake_llm, clock, as_user, monkeypatch):
    # Slower than the cache timeout (0.05s) but within the budget timeout (2s): the check must still happen.
    monkeypatch.setattr(config, "REDIS_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(config, "BUDGET_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(services.cache, "_get_redis", lambda: SlowCounterRedis())
    as_user(500)

    res = ask("A brand new question?")
    assert res.status_code == 429
    assert fake_llm.calls == 0


def test_counter_that_never_answers_still_fails_open(fake_llm, clock, as_user, monkeypatch):
    monkeypatch.setattr(config, "REDIS_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(config, "BUDGET_TIMEOUT_SECONDS", 0.05)  # gives up before the 0.2s answer arrives
    monkeypatch.setattr(services.cache, "_get_redis", lambda: SlowCounterRedis())
    as_user(500)

    res = ask("A brand new question?")
    assert res.status_code == 200
    assert res.json()["tokens_used_today"] is None


def test_cache_hit_does_not_wait_for_a_slow_counter(fake_llm, clock, as_user, monkeypatch):
    as_user(500)
    ask("What is RAG?")  # saved in the (fake) Redis
    real = services.cache._get_redis()

    class SlowOnlyForBudget:
        """Behaves like the real fake Redis, except reading the budget counter takes 0.5s."""

        def __getattr__(self, name):
            return getattr(real, name)

        async def get(self, key):
            if key.startswith("budget:"):
                await asyncio.sleep(0.5)
            return await real.get(key)

    monkeypatch.setattr(services.cache, "_get_redis", lambda: SlowOnlyForBudget())
    monkeypatch.setattr(config, "REDIS_TIMEOUT_SECONDS", 0.1)

    res = ask("What is RAG?")
    assert res.status_code == 200 and res.json()["cache_type"] == "exact"
    assert res.json()["tokens_used_today"] is None  # we didn't make a free hit wait for the counter
    assert res.json()["latency_ms"] < 400
