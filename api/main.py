import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException

from models.schemas import ChatRequest, ChatResponse
from services import cache, db, embeddings
from services.call_log import save_call
from services.gateway import answer_question, remember_answer
from services.llm import LLMError

logging.basicConfig(level=logging.INFO)  # so our timing lines show up in the server terminal
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: load the model and open the DB + Redis connections
    # (all three are slow the first time). If one fails, the server still starts and that
    # part is just skipped, like any other failure.
    steps = {
        "embedding model": embeddings.warm_up(),
        "database": asyncio.to_thread(db.warm_up),
        "redis": cache.warm_up(),
    }
    results = await asyncio.gather(*steps.values(), return_exceptions=True)
    for name, result in zip(steps, results):
        if isinstance(result, Exception):
            logger.error("Could not warm up %s: %r", name, result)
    yield


app = FastAPI(title="LLM Gateway", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    # Blank questions never reach here — Pydantic already returned 422.
    try:
        result = await answer_question(req.question, use_semantic=req.semantic_cache)
    except LLMError as e:
        # The upstream provider failed: tell the caller clearly instead of crashing.
        raise HTTPException(status_code=502, detail=str(e))
    # These run AFTER the answer is sent, so they never slow down /chat (and never raise).
    background_tasks.add_task(save_call, req.question, result)
    background_tasks.add_task(remember_answer, req.question, result)
    return result
