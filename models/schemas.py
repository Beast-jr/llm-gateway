from typing import Annotated

from pydantic import BaseModel, StringConstraints

# A question must have at least 1 real character after trimming spaces.
# So "" and "   " are both rejected with a 422 error automatically.
NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChatRequest(BaseModel):
    question: NonBlankStr
    semantic_cache: bool = True  # false = never reuse a similar question's answer (exact cache still applies)


class ChatResponse(BaseModel):
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    provider: str  # who produced the answer: "groq", "gemini", or "cache"
    fallback_reason: str | None = None  # why Gemini was used instead of Groq (null = Groq worked / cache hit)
    cache_hit: bool
    truncated: bool = False  # true = the answer hit the length limit and is cut off (never cached)
    cache_type: str | None = None  # "exact", "semantic", or null on a miss
    matched_question: str | None = None  # semantic hits only: the saved question whose answer we reused
    tokens_used_today: int | None = None  # this user's tokens used today (UTC); null if the counter is unavailable
    daily_budget: int | None = None  # this user's daily token limit
    similarity: float | None = None  # score of the nearest saved question (hit, or near-miss on a miss)
