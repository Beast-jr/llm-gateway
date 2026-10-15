"""add cache_hit to calls

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-15 03:29:49.687831

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0002'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add cache_hit. Existing rows (all real Groq calls) become false automatically."""
    op.add_column(
        "calls",
        sa.Column("cache_hit", sa.Boolean(), server_default=sa.false(), nullable=False),
    )


def downgrade() -> None:
    """Undo upgrade(): remove the cache_hit column."""
    op.drop_column("calls", "cache_hit")
