from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from services.db import Base


class Call(Base):
    """One row per /chat request."""

    __tablename__ = "calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(100))
    provider: Mapped[str] = mapped_column(String(50))  # "groq" now, "gemini" from Day 10
    prompt_tokens: Mapped[int]
    completion_tokens: Mapped[int]
    latency_ms: Mapped[float]
    cost_usd: Mapped[float]
    # Later days add new columns here (+ a new Alembic migration), e.g. cache_hit on Day 4.
