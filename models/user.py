from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, func, true
from sqlalchemy.orm import Mapped, mapped_column

from services.db import Base


class User(Base):
    """One row per API-key holder. The plain key is never stored, only its SHA-256 hash."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # sha256 hex of the key
    key_prefix: Mapped[str] = mapped_column(String(16))  # first few characters, only for showing in lists
    daily_token_budget: Mapped[int] = mapped_column(default=10000, server_default="10000")  # used from Day 9
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
