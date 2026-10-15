"""Tests for the 'plumbing': startup, database/Redis/embedding setup. All fakes, no real services or models."""
import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import literal, select

import config
import services.auth
import services.budget
import services.cache
import services.db
import services.embeddings
import services.semantic_cache
from api.main import app
from models.semantic_cache import SemanticCacheEntry
from tests.conftest import FAKE_VECTOR, REAL_EMBED, REAL_FIND_CLOSEST, REAL_GET_REDIS


# ---------- server startup ----------

def test_startup_warms_up_all_three_and_survives_one_failing(monkeypatch, caplog):
    started = []

    async def model_fails():
        raise RuntimeError("model download failed")

    async def redis_ok():
        started.append("redis")

    monkeypatch.setattr(services.embeddings, "warm_up", model_fails)
    monkeypatch.setattr(services.cache, "warm_up", redis_ok)
    monkeypatch.setattr(services.db, "warm_up", lambda: started.append("db"))

    with TestClient(app) as running:  # "with" makes FastAPI run its startup code
        assert running.get("/health").status_code == 200  # the server is up despite the failure

    assert sorted(started) == ["db", "redis"]  # the others still ran
    assert "Could not warm up embedding model" in caplog.text


# ---------- database setup ----------

def test_database_url_gets_the_psycopg_driver(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db?sslmode=require")
    assert services.db.get_database_url() == "postgresql+psycopg://u:p@h/db?sslmode=require"
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@h/db")
    assert services.db.get_database_url() == "postgresql+psycopg://u:p@h/db"
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    assert services.db.get_database_url() == "sqlite://"  # anything else is left alone


def test_engine_is_created_once_and_recycles_connections(monkeypatch):
    monkeypatch.setattr(services.db, "_SessionLocal", None)
    monkeypatch.setattr(services.db, "_engine", None)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")  # never connected to: engines connect lazily

    services.db.get_session().close()
    engine = services.db._engine
    services.db.get_session().close()

    assert services.db._engine is engine
    assert engine.url.drivername == "postgresql+psycopg"
    assert engine.pool._recycle == 240


def test_database_warm_up_runs_a_query(test_db, monkeypatch):
    monkeypatch.setattr(services.db, "get_session", test_db)
    services.db.warm_up()  # would raise if the connection were broken


# ---------- embedding model setup ----------

def test_embed_many_scales_every_vector_to_length_one(monkeypatch):
    class FakeModel:
        def embed(self, texts):
            return [np.array([3.0, 4.0]), np.array([0.0, 2.0])]

    monkeypatch.setattr(services.embeddings, "_get_model", lambda: FakeModel())
    vectors = services.embeddings.embed_many(["a", "b"])
    assert vectors == [pytest.approx([0.6, 0.8]), pytest.approx([0.0, 1.0])]


def test_embed_returns_one_vector(monkeypatch):
    monkeypatch.setattr(services.embeddings, "embed_many", lambda texts: [[0.1, 0.2]])
    assert asyncio.run(REAL_EMBED("hello")) == [0.1, 0.2]


def test_model_is_loaded_once_from_config(monkeypatch):
    loads = []

    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir):
            loads.append((model_name, cache_dir))

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    monkeypatch.setattr(services.embeddings, "_model", None)

    first = services.embeddings._get_model()
    assert services.embeddings._get_model() is first
    assert loads == [(config.EMBEDDING_MODEL, config.EMBEDDING_CACHE_DIR)]  # loaded exactly once


def test_embedding_warm_up_loads_the_model(monkeypatch):
    loaded = []
    monkeypatch.setattr(services.embeddings, "_get_model", lambda: loaded.append(True))
    asyncio.run(services.embeddings.warm_up())
    assert loaded == [True]


# ---------- Redis setup ----------

def test_no_redis_url_means_no_cache(monkeypatch):
    monkeypatch.setattr(services.cache, "_redis", None)
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert REAL_GET_REDIS() is None


def test_redis_client_is_built_lazily_with_keepalive_and_a_retry(monkeypatch):
    monkeypatch.setattr(services.cache, "_redis", None)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")  # never connected to

    client = REAL_GET_REDIS()
    assert REAL_GET_REDIS() is client  # created once, then reused
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["socket_keepalive"] is True
    assert kwargs["health_check_interval"] == 30
    monkeypatch.setattr(services.cache, "_redis", None)  # don't leave this client behind


def test_redis_warm_up_pings_and_is_a_noop_without_redis(monkeypatch):
    asyncio.run(services.cache.warm_up())  # the (fake) Redis from conftest answers the ping
    monkeypatch.setattr(services.cache, "_get_redis", lambda: None)
    asyncio.run(services.cache.warm_up())


def test_budget_functions_do_nothing_without_redis(monkeypatch):
    monkeypatch.setattr(services.cache, "_get_redis", lambda: None)
    assert asyncio.run(services.budget.get_used(1)) is None
    assert asyncio.run(services.budget.add_usage(1, 50)) is None


# ---------- semantic lookup: turning a database row into an answer ----------

def test_find_closest_turns_distance_into_similarity_and_returns_the_saved_cost(test_db, monkeypatch):
    with test_db() as session:
        session.add(SemanticCacheEntry(question="What is RAG?", embedding=FAKE_VECTOR, answer="a saved answer",
                                       model="m", prompt_tokens=11, completion_tokens=22, user_id=1))
        session.commit()
    # SQLite can't do vector distance, so swap the query for a simple one that pretends the distance is 0.08
    monkeypatch.setattr(services.semantic_cache, "closest_query",
                        lambda vector, model, user_id: select(SemanticCacheEntry, literal(0.08).label("distance")).limit(1))

    found = REAL_FIND_CLOSEST(FAKE_VECTOR, "m", 1)
    assert found == {"question": "What is RAG?", "answer": "a saved answer", "similarity": pytest.approx(0.92),
                     "prompt_tokens": 11, "completion_tokens": 22}


def test_find_closest_with_nothing_saved_returns_none(test_db, monkeypatch):
    monkeypatch.setattr(services.semantic_cache, "closest_query",
                        lambda vector, model, user_id: select(SemanticCacheEntry, literal(0.0).label("distance")).limit(1))
    assert REAL_FIND_CLOSEST(FAKE_VECTOR, "m", 1) is None


# ---------- the login cache stays small ----------

def test_login_cache_is_emptied_when_it_gets_too_big(monkeypatch):
    monkeypatch.setattr(services.auth, "_CACHE_MAX", 2)
    keys = [services.auth.create_user(f"user{i}", 100)[1] for i in range(3)]
    for key in keys:
        assert asyncio.run(services.auth.authenticate(key)) is not None
    assert len(services.auth._cache) <= 2
    assert services.auth.hash_key(keys[-1]) in services.auth._cache  # the newest one is still remembered
