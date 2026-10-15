import asyncio
import logging

from sqlalchemy import select

from models.semantic_cache import SemanticCacheEntry
from services import embeddings
from services.db import get_session, retry_once_if_dropped

logger = logging.getLogger(__name__)


def _insert(question: str, vector: list[float], result: dict) -> None:
    with get_session() as session:
        session.add(SemanticCacheEntry(
            question=question,
            embedding=vector,
            answer=result["answer"],
            model=result["model"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
        ))
        session.commit()


def find_closest(vector: list[float], model: str) -> dict | None:
    """Closest saved question for this model, or None if the table has no row for it.

    Sync (the DB driver is sync): callers run it with asyncio.to_thread.
    """
    # <=> is pgvector's cosine distance; ordering by it lets the HNSW index do the search.
    distance = SemanticCacheEntry.embedding.cosine_distance(vector)
    query = (
        select(SemanticCacheEntry, distance.label("distance"))
        .where(SemanticCacheEntry.model == model)
        .order_by(distance)
        .limit(1)
    )
    def run():
        with get_session() as session:
            # AUTOCOMMIT: a plain read needs no transaction, which saves extra network round trips.
            session.connection(execution_options={"isolation_level": "AUTOCOMMIT"})
            return session.execute(query).first()

    row = retry_once_if_dropped(run)
    if row is None:
        return None
    entry, dist = row
    return {"answer": entry.answer, "similarity": 1 - float(dist)}


async def save_entry(question: str, result: dict, vector: list[float] | None = None) -> None:
    """Store a question with its answer. Pass the vector if we already have it. Never raises."""
    try:
        if vector is None:
            vector = await embeddings.embed(question)
        await asyncio.to_thread(retry_once_if_dropped, lambda: _insert(question, vector, result))  # DB call off the main thread
    except Exception:
        logger.exception("Could not save to semantic cache (the user still got their answer)")
