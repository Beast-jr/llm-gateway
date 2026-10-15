import asyncio
import logging

from models.semantic_cache import SemanticCacheEntry
from services import embeddings
from services.db import get_session

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


async def save_entry(question: str, result: dict) -> None:
    """Embed the question and store it with its answer. Never raises."""
    try:
        vector = await embeddings.embed(question)
        await asyncio.to_thread(_insert, question, vector, result)  # DB call off the main thread
    except Exception:
        logger.exception("Could not save to semantic cache (the user still got their answer)")
