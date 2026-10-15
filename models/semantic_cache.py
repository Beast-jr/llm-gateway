from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

import config
from services.db import Base


class SemanticCacheEntry(Base):
    """One answered question + its embedding, so similar questions can reuse the answer (Day 6)."""

    __tablename__ = "semantic_cache"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    question: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(config.EMBEDDING_DIM), nullable=False)
    answer: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(100))
    prompt_tokens: Mapped[int]
    completion_tokens: Mapped[int]

    __table_args__ = (
        # HNSW index: lets Postgres find the closest vectors fast, without checking every row.
        # vector_cosine_ops = compare by cosine distance (what our similarity uses).
        Index(
            "ix_semantic_cache_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
