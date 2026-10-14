from fastapi import BackgroundTasks, FastAPI, HTTPException

from models.schemas import ChatRequest, ChatResponse
from services.call_log import save_call
from services.llm import LLMError, ask_llm

app = FastAPI(title="LLM Gateway")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    # Blank questions never reach here — Pydantic already returned 422.
    try:
        result = await ask_llm(req.question)
    except LLMError as e:
        # The upstream provider failed: tell the caller clearly instead of crashing.
        raise HTTPException(status_code=502, detail=str(e))
    # Save to the database AFTER the answer is sent, so logging never slows down /chat.
    background_tasks.add_task(save_call, req.question, result)
    return result
