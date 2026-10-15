"""add semantic_cache table

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-15 03:38:45.072067

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '0003'
down_revision: Union[str, Sequence[str], None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Turn on pgvector, then create semantic_cache with a fast cosine-similarity index."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "semantic_cache",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        # 384 is written out on purpose: a migration is a snapshot and shouldn't change if
        # config.py changes later (a different-size model would need a new migration).
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
    )
    op.create_index(
        "ix_semantic_cache_embedding_hnsw",
        "semantic_cache",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Undo upgrade(). The vector extension stays installed (harmless; other tables may use it)."""
    op.drop_index("ix_semantic_cache_embedding_hnsw", table_name="semantic_cache")
    op.drop_table("semantic_cache")
