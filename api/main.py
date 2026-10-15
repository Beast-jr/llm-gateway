from fastapi import BackgroundTasks, FastAPI, HTTPException

from models.schemas import ChatRequest, ChatResponse
from services.call_log import save_call
from services.gateway import answer_question, remember_answer
from services.llm import LLMError

app = FastAPI(title="LLM Gateway")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    # Blank questions never reach here — Pydantic already returned 422.
    try:
        result = await answer_question(req.question)
    except LLMError as e:
        # The upstream provider failed: tell the caller clearly instead of crashing.
        raise HTTPException(status_code=502, detail=str(e))
    # These run AFTER the answer is sent, so they never slow down /chat (and never raise).
    background_tasks.add_task(save_call, req.question, result)
    background_tasks.add_task(remember_answer, req.question, result)
    return result
