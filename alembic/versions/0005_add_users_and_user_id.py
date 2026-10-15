"""add users table, and user_id to calls and semantic_cache

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0005'
down_revision: Union[str, Sequence[str], None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("api_key_hash", sa.String(length=64), nullable=False),
        sa.Column("key_prefix", sa.String(length=16), nullable=False),
        sa.Column("daily_token_budget", sa.Integer(), server_default="10000", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_users_api_key_hash", "users", ["api_key_hash"], unique=True)

    # Nullable: old rows have no owner. With per-user cache scope they are simply never matched.
    for table in ("calls", "semantic_cache"):
        op.add_column(table, sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True))
        op.create_index(f"ix_{table}_user_id", table, ["user_id"])


def downgrade() -> None:
    for table in ("semantic_cache", "calls"):
        op.drop_index(f"ix_{table}_user_id", table_name=table)
        op.drop_column(table, "user_id")
    op.drop_index("ix_users_api_key_hash", table_name="users")
    op.drop_table("users")
