from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, false, func
from sqlalchemy.orm import Mapped, mapped_column

from services.db import Base


class Call(Base):
    """One row per /chat request."""

    __tablename__ = "calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(100))
    provider: Mapped[str] = mapped_column(String(50))  # "groq" now, "gemini" from Day 10
    prompt_tokens: Mapped[int]
    completion_tokens: Mapped[int]
    latency_ms: Mapped[float]
    cost_usd: Mapped[float]
    cache_hit: Mapped[bool] = mapped_column(default=False, server_default=false())
    cache_type: Mapped[str | None] = mapped_column(String(20))  # "exact", "semantic" or None (miss)
    similarity: Mapped[float | None]  # nearest saved question's score (semantic hit, or near-miss); empty if none
    saved_cost_usd: Mapped[float] = mapped_column(default=0.0, server_default="0")  # cache hits: what the reused answer originally cost
    truncated: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())  # answer hit the length limit
    fallback_reason: Mapped[str | None] = mapped_column(String(50))  # why Groq was skipped, e.g. "groq_timeout"; empty if not
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)  # who asked (empty on old rows)
    # Later days add new columns here (+ a new Alembic migration).
