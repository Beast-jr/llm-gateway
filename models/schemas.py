from typing import Annotated

from pydantic import BaseModel, StringConstraints

# A question must have at least 1 real character after trimming spaces.
# So "" and "   " are both rejected with a 422 error automatically.
NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChatRequest(BaseModel):
    question: NonBlankStr


class ChatResponse(BaseModel):
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cache_hit: bool
    cache_type: str | None = None  # "exact", "semantic", or null on a miss
    similarity: float | None = None  # how close the matched question was (semantic hits only)
