import asyncio
import logging

from sqlalchemy import select

import config
from models.semantic_cache import SemanticCacheEntry
from services import embeddings, match_guard
from services.db import get_session, retry_once_if_dropped

logger = logging.getLogger(__name__)


def _insert(question: str, vector: list[float], result: dict, user_id: int | None, model: str) -> None:
    with get_session() as session:
        session.add(SemanticCacheEntry(
            question=question,
            embedding=vector,
            answer=result["answer"],
            model=model,
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            user_id=user_id,
        ))
        session.commit()


def closest_query(vector: list[float], model: str, user_id: int | None):
    """The SQL: closest saved question for this model, and for this user if user_id is given.

    user_id=None means "everyone's rows" (CACHE_SCOPE = "shared"). A given user_id never matches
    rows with no owner (old rows), because NULL = 5 is never true.
    """
    # <=> is pgvector's cosine distance; ordering by it lets the HNSW index do the search.
    distance = SemanticCacheEntry.embedding.cosine_distance(vector)
    query = select(SemanticCacheEntry, distance.label("distance")).where(SemanticCacheEntry.model == model)
    if user_id is not None:
        query = query.where(SemanticCacheEntry.user_id == user_id)
    return query.order_by(distance).limit(1)


def find_closest(vector: list[float], model: str, user_id: int | None = None) -> dict | None:
    """Closest saved question (see closest_query), or None if there is no row for it.

    Sync (the DB driver is sync): callers run it with asyncio.to_thread.
    """
    query = closest_query(vector, model, user_id)

    def run():
        with get_session() as session:
            # AUTOCOMMIT: a plain read needs no transaction, which saves extra network round trips.
            session.connection(execution_options={"isolation_level": "AUTOCOMMIT"})
            return session.execute(query).first()

    row = retry_once_if_dropped(run)
    if row is None:
        return None
    entry, dist = row
    return {"question": entry.question, "answer": entry.answer, "similarity": 1 - float(dist)}


async def save_entry(question: str, result: dict, vector: list[float] | None = None,
                     user_id: int | None = None, lookup_user_id: int | None = None,
                     model: str | None = None) -> None:
    """Store a question with its answer. Pass the vector if we already have it. Never raises.

    user_id = the owner stored on the row. lookup_user_id = whose rows count as duplicates
    (the owner when scope is per_user, None when shared).
    model = the cache namespace to store under (the primary model's name, so lookups find answers
    from ANY provider). Defaults to the model that wrote the answer.
    """
    try:
        model = model or result["model"]
        if vector is None:
            vector = await embeddings.embed(question)
        # Skip near-duplicates. Checked here (in the background, so it costs the user nothing),
        # because the request's own lookup may have timed out and then knows nothing.
        closest = await asyncio.to_thread(find_closest, vector, model, lookup_user_id)
        if (closest is not None and closest["similarity"] >= config.SEMANTIC_DUPLICATE_SIMILARITY
                and match_guard.check_numbers(question, closest["question"]) is None):
            return
        await asyncio.to_thread(retry_once_if_dropped, lambda: _insert(question, vector, result, user_id, model))  # off the main thread
    except Exception:
        logger.exception("Could not save to semantic cache (the user still got their answer)")
