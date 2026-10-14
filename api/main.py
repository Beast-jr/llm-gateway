from fastapi import FastAPI, HTTPException

from models.schemas import ChatRequest, ChatResponse
from services.llm import LLMError, ask_llm

app = FastAPI(title="LLM Gateway")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Blank questions never reach here — Pydantic already returned 422.
    try:
        result = await ask_llm(req.question)
    except LLMError as e:
        # The upstream provider failed: tell the caller clearly instead of crashing.
        raise HTTPException(status_code=502, detail=str(e))
    return result
