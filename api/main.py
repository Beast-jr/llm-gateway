import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query

import config
from models.schemas import ChatRequest, ChatResponse, StatsResponse
from services import cache, db, embeddings, gemini
from services.auth import AuthUser, current_user
from services.budget import BudgetExceeded
from services.call_log import save_call
from services.stats import compute_stats
from services.gateway import answer_question, remember_answer
from services.llm import LLMError

logging.basicConfig(level=logging.INFO)  # so our timing lines show up in the server terminal
logger = logging.getLogger(__name__)


async def _warm_up(name: str, work) -> None:
    """Run one start-up step with a time limit. Never raises."""
    started = time.perf_counter()
    try:
        await asyncio.wait_for(work, config.STARTUP_WARMUP_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.error("Warm-up of %s did not finish within %ss; starting anyway", name, config.STARTUP_WARMUP_TIMEOUT_SECONDS)
    except Exception as e:
        logger.error("Could not warm up %s: %r", name, e)
    else:
        logger.info("startup: %s ready in %.1fs", name, time.perf_counter() - started)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: load the model and open the DB + Redis connections
    # (all three are slow the first time, so we do them now, in parallel, before the first user arrives).
    # If one fails or takes too long, the server still starts and that part is just skipped.
    started = time.perf_counter()
    await asyncio.gather(
        _warm_up("embedding model", embeddings.warm_up()),
        _warm_up("database", asyncio.to_thread(db.warm_up)),
        _warm_up("redis", cache.warm_up()),
    )
    logger.info("startup: ready to accept requests after %.1fs", time.perf_counter() - started)
    # Not needed until Groq fails, so load Google's SDK in the background instead of delaying the first user.
    background = asyncio.create_task(asyncio.to_thread(gemini.preload))
    yield
    background.cancel()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks, user: AuthUser = Depends(current_user)):
    # Blank questions never reach here — Pydantic already returned 422.
    try:
        result = await answer_question(req.question, use_semantic=req.semantic_cache, user_id=user.id,
                                      daily_budget=user.daily_token_budget)
    except BudgetExceeded as e:
        # Over the daily limit: tell the caller when it resets. (Cached answers never reach here.)
        raise HTTPException(status_code=429, detail=str(e), headers={"Retry-After": str(e.retry_after)})
    except LLMError as e:
        # The upstream provider failed: tell the caller clearly instead of crashing.
        raise HTTPException(status_code=502, detail=str(e))
    # These run AFTER the answer is sent, so they never slow down /chat (and never raise).
    background_tasks.add_task(save_call, req.question, result, user.id)
    background_tasks.add_task(remember_answer, req.question, result, user.id)
    return result


@app.get("/stats", response_model=StatsResponse)
def stats(
    days: int = Query(7, ge=1, le=365, description="Look back this many days (rolling window, UTC)."),
    mine: bool = Query(False, description="true = only your own requests; false = everyone's, combined."),
    user: AuthUser = Depends(current_user),
):
    """Numbers only (hit rates, speed, cost, money saved). Never any question or answer text."""
    try:
        return compute_stats(days, user.id if mine else None)
    except Exception:
        logger.exception("Could not compute /stats")
        raise HTTPException(status_code=503, detail="Stats are temporarily unavailable. Please try again.")
