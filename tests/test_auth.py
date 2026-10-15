import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

import config
import services.auth
from api.main import app
from models.call import Call
from models.semantic_cache import SemanticCacheEntry
from models.user import User
from services.auth import create_user, current_user
from services.semantic_cache import closest_query
from tests.conftest import FAKE_VECTOR

client = TestClient(app)


@pytest.fixture(autouse=True)
def real_login():
    """Undo the conftest shortcut: in this file /chat checks real keys against the (test) database."""
    app.dependency_overrides.pop(current_user, None)


def ask(key, question="What is RAG?", **extra):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return client.post("/chat", json={"question": question, **extra}, headers=headers)


def make_user(name="alice", budget=500):
    return create_user(name, budget)  # returns (user, plain key)


# ---------- who gets in ----------

def test_health_is_public():
    assert client.get("/health").status_code == 200


def test_no_key_is_401():
    res = ask(None)
    assert res.status_code == 401
    assert "Missing API key" in res.json()["detail"]
    assert res.headers["www-authenticate"] == "Bearer"


def test_wrong_key_is_401(fake_llm):
    make_user()
    res = ask("gw_thisIsNotARealKey")
    assert res.status_code == 401
    assert res.json()["detail"] == "Invalid API key."
    assert fake_llm.calls == 0  # nothing was sent to Groq


def test_key_without_our_prefix_is_401(monkeypatch):
    def must_not_run(key_hash):
        raise AssertionError("should not touch the database for an obviously wrong key")

    monkeypatch.setattr(services.auth, "_find_user", must_not_run)
    assert ask("sk-something-else").status_code == 401


def test_wrong_header_style_is_401(fake_llm):
    res = client.post("/chat", json={"question": "Hi"}, headers={"Authorization": "Basic abc123"})
    assert res.status_code == 401


def test_inactive_key_is_401(fake_llm, test_db):
    user, key = make_user()
    with test_db() as session:
        session.get(User, user.id).is_active = False
        session.commit()
    res = ask(key)
    assert res.status_code == 401
    assert "disabled" in res.json()["detail"]
    assert fake_llm.calls == 0


def test_valid_key_is_200_and_call_is_logged_to_that_user(fake_llm, test_db):
    user, key = make_user()
    res = ask(key)
    assert res.status_code == 200
    assert res.json()["answer"] == fake_llm.result["answer"]
    with test_db() as session:
        assert session.query(Call).one().user_id == user.id


def test_database_down_during_login_is_503_not_a_free_pass(fake_llm, monkeypatch):
    _, key = make_user()
    services.auth._cache.clear()

    def broken(key_hash):
        raise ConnectionError("database is down")

    monkeypatch.setattr(services.auth, "_find_user", broken)
    assert ask(key).status_code == 503
    assert fake_llm.calls == 0


# ---------- keys are stored safely ----------

def test_plain_key_is_never_stored(test_db):
    user, key = make_user("bob")
    assert key.startswith("gw_") and len(key) > 40

    with test_db() as session:
        row = session.get(User, user.id)
        stored = [str(getattr(row, col.name)) for col in User.__table__.columns]
    assert not any(key in value for value in stored)  # not in ANY column
    assert row.api_key_hash == hashlib.sha256(key.encode()).hexdigest()
    assert row.key_prefix == key[:8]
    assert row.daily_token_budget == 500 and row.is_active is True


def test_every_key_is_different():
    assert services.auth.generate_key() != services.auth.generate_key()


# ---------- the in-memory key cache ----------

def test_key_lookup_is_remembered_in_memory(fake_llm, monkeypatch):
    _, key = make_user()
    real = services.auth._find_user
    calls = []
    monkeypatch.setattr(services.auth, "_find_user", lambda h: calls.append(1) or real(h))

    for _ in range(3):
        assert ask(key).status_code == 200
    assert len(calls) == 1  # only the first request asked the database


def test_cache_expires(fake_llm, monkeypatch):
    _, key = make_user()
    real = services.auth._find_user
    calls = []
    monkeypatch.setattr(services.auth, "_find_user", lambda h: calls.append(1) or real(h))
    monkeypatch.setattr(config, "AUTH_CACHE_SECONDS", 0)  # remember for 0 seconds = never

    ask(key)
    ask(key)
    assert len(calls) == 2


def test_wrong_keys_are_not_remembered(monkeypatch):
    ask("gw_nope")
    assert services.auth._cache == {}


# ---------- one user never sees another user's answers ----------

def test_exact_cache_is_private_per_user(fake_llm):
    _, key_a = make_user("alice")
    _, key_b = make_user("bob")

    a1 = ask(key_a).json()
    b1 = ask(key_b).json()  # same question, different user
    a2 = ask(key_a).json()

    assert a1["cache_hit"] is False
    assert b1["cache_hit"] is False  # NOT served alice's saved answer
    assert a2["cache_type"] == "exact"  # alice still gets her own
    assert fake_llm.calls == 2


def test_shared_scope_lets_users_share_exact_answers(fake_llm, monkeypatch):
    monkeypatch.setattr(config, "CACHE_SCOPE", "shared")
    _, key_a = make_user("alice")
    _, key_b = make_user("bob")

    ask(key_a)
    b = ask(key_b).json()
    assert b["cache_type"] == "exact"
    assert fake_llm.calls == 1


def test_semantic_lookup_is_limited_to_the_callers_rows(fake_llm, fake_lookup):
    user_a, key_a = make_user("alice")
    user_b, key_b = make_user("bob")

    ask(key_a, "What is RAG?")
    assert fake_lookup.last_user_id == user_a.id
    ask(key_b, "Explain RAG to me")
    assert fake_lookup.last_user_id == user_b.id


def test_shared_scope_semantic_lookup_has_no_user_limit(fake_llm, fake_lookup, monkeypatch):
    monkeypatch.setattr(config, "CACHE_SCOPE", "shared")
    _, key = make_user()
    ask(key)
    assert fake_lookup.last_user_id is None


def test_semantic_rows_remember_their_owner(fake_llm, test_db):
    user, key = make_user()
    ask(key)
    with test_db() as session:
        assert session.query(SemanticCacheEntry).one().user_id == user.id


def test_semantic_hit_is_stored_for_that_user_only(fake_llm, fake_lookup):
    _, key_a = make_user("alice")
    _, key_b = make_user("bob")
    fake_lookup.match = {"question": "What is RAG?", "answer": "saved", "similarity": 0.99}

    assert ask(key_a, "Explain RAG to me").json()["cache_type"] == "semantic"
    fake_lookup.match = None
    assert ask(key_a, "Explain RAG to me").json()["cache_type"] == "exact"  # alice: Redis has it
    assert ask(key_b, "Explain RAG to me").json()["cache_hit"] is False  # bob: nothing of alice's


def test_the_real_sql_filters_by_user():
    """SQLite can't run vector SQL, so check the query text instead."""
    def sql(user_id):
        return str(closest_query(FAKE_VECTOR, "m", user_id).compile(dialect=postgresql.dialect()))

    where = lambda user_id: sql(user_id).split("WHERE")[1].split("ORDER BY")[0]
    assert "semantic_cache.user_id =" in where(7)  # also means old rows with no owner never match
    assert "user_id" not in where(None)  # shared scope: no owner limit
    assert "<=>" in sql(7)  # still the pgvector distance, so the HNSW index is used
