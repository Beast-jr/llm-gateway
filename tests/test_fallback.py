import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google.genai import errors, types

import config
import services.breaker
import services.budget
import services.gateway
import services.gemini
import services.llm
from api.main import app
from models.call import Call
from models.semantic_cache import SemanticCacheEntry
from services.llm import LLMError
from tests.conftest import FAKE_GEMINI_RESULT, FAKE_RESULT

client = TestClient(app)
GEMINI_TOKENS = FAKE_GEMINI_RESULT["prompt_tokens"] + FAKE_GEMINI_RESULT["completion_tokens"]


@pytest.fixture
def groq(monkeypatch):
    """A controllable fake Groq. groq.error = a reason code makes it fail; groq.delay = seconds it takes."""
    class Groq:
        calls = 0
        error = None
        delay = 0

    async def fake_ask_llm(question):
        Groq.calls += 1
        if Groq.delay:
            await asyncio.sleep(Groq.delay)
        if Groq.error:
            raise LLMError("Groq request failed: boom", Groq.error)
        return dict(FAKE_RESULT)

    monkeypatch.setattr(services.gateway, "ask_llm", fake_ask_llm)
    return Groq


def ask(question="What is RAG?"):
    return client.post("/chat", json={"question": question})


# ---------- when Gemini takes over ----------

def test_groq_error_switches_to_gemini(groq, fake_gemini, test_db):
    groq.error = "groq_error_401"
    fake_gemini.working = True
    body = ask().json()

    assert body["provider"] == "gemini"
    assert body["fallback_reason"] == "groq_error_401"
    assert body["answer"] == FAKE_GEMINI_RESULT["answer"]
    assert body["model"] == "gemini-2.5-flash-lite"
    assert body["truncated"] is False
    with test_db() as session:
        row = session.query(Call).one()
    assert (row.provider, row.fallback_reason) == ("gemini", "groq_error_401")


def test_groq_connection_failure_switches_to_gemini(groq, fake_gemini):
    groq.error = "groq_connection_error"
    fake_gemini.working = True
    assert ask().json()["fallback_reason"] == "groq_connection_error"


def test_rate_limit_and_server_errors_switch_too(groq, fake_gemini):
    fake_gemini.working = True
    for i, reason in enumerate(["groq_error_429", "groq_error_503"]):
        groq.error = reason
        assert ask(f"Question {i}?").json()["fallback_reason"] == reason


def test_groq_slower_than_the_total_limit_switches_to_gemini(groq, fake_gemini, monkeypatch):
    monkeypatch.setattr(config, "GROQ_TOTAL_TIMEOUT_SECONDS", 0.05)
    groq.delay = 0.5
    fake_gemini.working = True
    body = ask().json()
    assert body["provider"] == "gemini"
    assert body["fallback_reason"] == "groq_timeout"


def test_long_but_healthy_groq_answer_stays_on_groq(groq, fake_gemini, monkeypatch):
    monkeypatch.setattr(config, "GROQ_TOTAL_TIMEOUT_SECONDS", 1.0)
    groq.delay = 0.3  # slow, but within the limit: this is the "healthy 7 second answer" case
    fake_gemini.working = True
    body = ask().json()

    assert body["provider"] == "groq"
    assert body["fallback_reason"] is None
    assert fake_gemini.calls == 0
    assert services.breaker.groq_breaker.failures == 0


def test_healthy_groq_never_touches_gemini(groq, fake_gemini):
    body = ask().json()
    assert body["provider"] == "groq" and body["fallback_reason"] is None
    assert fake_gemini.calls == 0


# ---------- Gemini answers are treated like any other answer ----------

def test_gemini_answer_is_cached_and_then_served_as_cache(groq, fake_gemini, test_db):
    groq.error = "groq_error_500"
    fake_gemini.working = True
    ask()
    groq.error = None  # Groq recovers, but the answer is already saved

    again = ask().json()
    assert again["cache_type"] == "exact" and again["provider"] == "cache"
    assert again["answer"] == FAKE_GEMINI_RESULT["answer"]
    assert groq.calls == 1 and fake_gemini.calls == 1

    with test_db() as session:
        saved = session.query(SemanticCacheEntry).one()
    assert saved.model == services.llm.get_model()  # stored where lookups will find it, not under "gemini-..."


def test_gemini_tokens_count_against_the_budget(groq, fake_gemini):
    groq.error = "groq_error_500"
    fake_gemini.working = True
    body = ask().json()
    assert body["tokens_used_today"] == GEMINI_TOKENS
    assert ask("A second, different question?").json()["tokens_used_today"] == 2 * GEMINI_TOKENS


def test_truncated_gemini_answer_is_not_cached(groq, fake_gemini, monkeypatch):
    groq.error = "groq_error_500"

    async def cut_off(question):
        return {**FAKE_GEMINI_RESULT, "truncated": True}

    monkeypatch.setattr(services.gateway, "ask_gemini", cut_off)
    assert ask().json()["truncated"] is True
    assert ask().json()["cache_hit"] is False  # asked again, nothing was saved


def test_cache_hit_reports_provider_cache(groq):
    ask()
    again = ask().json()
    assert again["provider"] == "cache" and again["fallback_reason"] is None


# ---------- circuit breaker ----------

def test_three_failures_open_the_circuit_and_groq_is_skipped(groq, fake_gemini):
    groq.error = "groq_error_500"
    fake_gemini.working = True
    for i in range(3):
        assert ask(f"Question {i}?").json()["fallback_reason"] == "groq_error_500"
    assert groq.calls == 3

    body = ask("Question 4?").json()
    assert body["provider"] == "gemini"
    assert body["fallback_reason"] == "circuit_open"
    assert groq.calls == 3  # Groq was not even tried


def test_two_failures_do_not_open_the_circuit(groq, fake_gemini):
    groq.error = "groq_error_500"
    fake_gemini.working = True
    ask("Question 1?")
    ask("Question 2?")
    groq.error = None
    assert ask("Question 3?").json()["provider"] == "groq"


def test_a_success_resets_the_failure_count(groq, fake_gemini):
    fake_gemini.working = True
    for step, error in enumerate(["groq_error_500", "groq_error_500", None, "groq_error_500", "groq_error_500"]):
        groq.error = error
        ask(f"Question {step}?")
    groq.error = None
    assert ask("Question last?").json()["provider"] == "groq"  # 2 + 2 failures in a row never reached 3


def test_groq_is_tried_again_after_the_cooldown(groq, fake_gemini, breaker_clock):
    groq.error = "groq_error_500"
    fake_gemini.working = True
    for i in range(3):
        ask(f"Question {i}?")
    assert ask("During cooldown?").json()["fallback_reason"] == "circuit_open"

    breaker_clock.now += config.BREAKER_COOLDOWN_SECONDS - 1
    assert ask("Almost over?").json()["fallback_reason"] == "circuit_open"

    breaker_clock.now += 2  # cooldown is over
    groq.error = None  # and Groq has recovered
    calls_before = groq.calls
    body = ask("After cooldown?").json()
    assert groq.calls == calls_before + 1  # tried again
    assert body["provider"] == "groq" and body["fallback_reason"] is None
    assert services.breaker.groq_breaker.failures == 0


def test_one_failure_after_the_cooldown_closes_the_door_again(groq, fake_gemini, breaker_clock):
    groq.error = "groq_error_500"
    fake_gemini.working = True
    for i in range(3):
        ask(f"Question {i}?")
    breaker_clock.now += config.BREAKER_COOLDOWN_SECONDS + 1

    assert ask("Trial request?").json()["fallback_reason"] == "groq_error_500"  # Groq tried, still broken
    assert ask("Next request?").json()["fallback_reason"] == "circuit_open"  # skipped again right away


# ---------- both providers down ----------

def test_both_fail_is_502_with_a_clear_message(groq, fake_gemini):
    groq.error = "groq_error_401"  # fake_gemini is 'down' by default
    res = ask()
    assert res.status_code == 502
    detail = res.json()["detail"]
    assert "Both providers failed" in detail
    assert "groq_error_401" in detail
    assert "Gemini" in detail


def test_both_fail_while_circuit_is_open(groq, fake_gemini):
    groq.error = "groq_error_500"
    for i in range(3):
        ask(f"Question {i}?")
    res = ask("Question 4?")
    assert res.status_code == 502
    assert "circuit_open" in res.json()["detail"]


def test_a_502_costs_no_budget_and_is_not_cached(groq, fake_gemini):
    groq.error = "groq_error_401"
    ask()
    groq.error = None
    body = ask().json()
    assert body["cache_hit"] is False and body["tokens_used_today"] == 12


# ---------- the real Groq client settings ----------

def test_groq_client_has_no_hidden_retries_and_a_short_connect_timeout(monkeypatch):
    monkeypatch.setattr(services.llm, "_client", None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_real")
    groq_client = services.llm._get_client()
    assert groq_client.max_retries == 0
    assert groq_client.timeout.connect == config.GROQ_CONNECT_TIMEOUT_SECONDS == 3
    assert groq_client.timeout.read == config.GROQ_TOTAL_TIMEOUT_SECONDS
    monkeypatch.setattr(services.llm, "_client", None)  # don't leave a fake-key client behind


def test_groq_errors_are_turned_into_reason_codes():
    import httpx
    import groq as groq_sdk

    request = httpx.Request("POST", "https://api.groq.com/x")
    unauthorized = groq_sdk.AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None)
    assert services.llm._groq_reason(unauthorized) == "groq_error_401"
    assert services.llm._groq_reason(groq_sdk.APITimeoutError(request=request)) == "groq_timeout"
    assert services.llm._groq_reason(groq_sdk.APIConnectionError(request=request)) == "groq_connection_error"
    rate_limited = groq_sdk.RateLimitError("slow down", response=httpx.Response(429, request=request), body=None)
    assert services.llm._groq_reason(rate_limited) == "groq_error_429"


# ---------- ask_gemini itself, with a fake Google client ----------

def fake_google(response=None, error=None, captured=None):
    async def generate_content(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        if error:
            raise error
        return response

    return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))


def google_response(text="hello", finish=types.FinishReason.STOP, prompt=3, answer_tokens=4, thinking=5):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=prompt, candidates_token_count=answer_tokens,
                                       thoughts_token_count=thinking),
        candidates=[SimpleNamespace(finish_reason=finish)],
    )


def run_gemini(monkeypatch, **kwargs):
    monkeypatch.setattr(services.gemini, "_get_client", lambda: fake_google(**kwargs))
    return asyncio.run(services.gemini.ask_gemini("hi"))


def test_ask_gemini_returns_the_same_fields_as_ask_llm(monkeypatch):
    captured = {}
    result = run_gemini(monkeypatch, response=google_response(), captured=captured)

    assert set(result) == set(FAKE_RESULT) | {"truncated"}
    assert result["provider"] == "gemini" and result["answer"] == "hello"
    assert result["prompt_tokens"] == 3
    assert result["completion_tokens"] == 4 + 5  # answer + hidden thinking tokens (both are billed)
    assert result["truncated"] is False
    assert captured["model"] == services.gemini.get_gemini_model()
    assert captured["config"].max_output_tokens == config.MAX_OUTPUT_TOKENS


def test_ask_gemini_flags_a_cut_off_answer(monkeypatch):
    result = run_gemini(monkeypatch, response=google_response(finish=types.FinishReason.MAX_TOKENS))
    assert result["truncated"] is True


def test_gemini_model_comes_from_env_then_config(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert services.gemini.get_gemini_model() == config.GEMINI_MODEL
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    assert services.gemini.get_gemini_model() == "gemini-test-model"


def test_ask_gemini_errors_become_llm_errors_with_reasons(monkeypatch):
    with pytest.raises(LLMError) as quota:
        run_gemini(monkeypatch, error=errors.ClientError(429, {"error": {"message": "quota exceeded"}}))
    assert quota.value.reason == "gemini_error_429"

    with pytest.raises(LLMError) as down:
        run_gemini(monkeypatch, error=ConnectionError("no network"))
    assert down.value.reason == "gemini_error"


def test_ask_gemini_with_no_answer_is_an_error(monkeypatch):
    with pytest.raises(LLMError) as blocked:
        run_gemini(monkeypatch, response=google_response(text=None, finish=types.FinishReason.SAFETY))
    assert blocked.value.reason == "gemini_no_answer"


def test_ask_gemini_times_out(monkeypatch):
    async def never_answers(**kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(config, "GEMINI_TIMEOUT_SECONDS", 0.05)
    client_ = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=never_answers)))
    monkeypatch.setattr(services.gemini, "_get_client", lambda: client_)
    with pytest.raises(LLMError) as slow:
        asyncio.run(services.gemini.ask_gemini("hi"))
    assert slow.value.reason == "gemini_timeout"


# ---------- surprises ----------

def test_unexpected_crash_inside_groq_still_falls_back(fake_gemini, monkeypatch):
    async def explodes(question):
        raise RuntimeError("a bug nobody expected")

    monkeypatch.setattr(services.gateway, "ask_llm", explodes)
    fake_gemini.working = True
    body = ask().json()
    assert body["provider"] == "gemini" and body["fallback_reason"] == "groq_error"


def test_real_ask_llm_turns_groq_sdk_errors_into_reason_codes(monkeypatch):
    import httpx
    import groq as groq_sdk

    request = httpx.Request("POST", "https://api.groq.com/x")

    async def create(**kwargs):
        raise groq_sdk.AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None)

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(services.llm, "_get_client", lambda: fake)
    with pytest.raises(LLMError) as failure:
        asyncio.run(services.llm.ask_llm("hi"))
    assert failure.value.reason == "groq_error_401"


def test_if_the_counter_write_fails_usage_is_estimated(groq, monkeypatch):
    async def write_fails(user_id, tokens):
        return None

    monkeypatch.setattr(services.budget, "add_usage", write_fails)
    assert ask().json()["tokens_used_today"] == 12  # 0 used before + this answer's 5 + 7 tokens
