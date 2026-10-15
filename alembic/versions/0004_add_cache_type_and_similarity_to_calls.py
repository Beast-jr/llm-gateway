"""add cache_type and similarity to calls

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0004'
down_revision: Union[str, Sequence[str], None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Both columns are nullable: old rows and cache misses simply have no value."""
    op.add_column("calls", sa.Column("cache_type", sa.String(length=20), nullable=True))
    op.add_column("calls", sa.Column("similarity", sa.Float(), nullable=True))


def downgrade() -> None:
    """Undo upgrade()."""
    op.drop_column("calls", "similarity")
    op.drop_column("calls", "cache_type")
