import logging

from config import PRICES_PER_MILLION
from models.call import Call
from services.db import get_session

logger = logging.getLogger(__name__)


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn token counts into US dollars using the price table in config.py."""
    price = PRICES_PER_MILLION.get(model)
    if price is None:
        logger.warning("No price listed for model %r in config.py; saving cost as 0", model)
        return 0.0
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


def save_call(question: str, result: dict) -> None:
    """Save one /chat call to the database.

    Never raises: if the database is down, we log the error and move on,
    because logging must never break /chat.
    """
    try:
        with get_session() as session:
            session.add(Call(
                question=question,
                answer=result["answer"],
                model=result["model"],
                provider=result["provider"],
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                latency_ms=result["latency_ms"],
                cost_usd=calculate_cost(result["model"], result["prompt_tokens"], result["completion_tokens"]),
                cache_hit=result["cache_hit"],  # on a hit, tokens are 0 so cost is 0 too
                cache_type=result.get("cache_type"),
                similarity=result.get("similarity"),
            ))
            session.commit()
    except Exception:
        logger.exception("Could not save call to the database (the user still got their answer)")
