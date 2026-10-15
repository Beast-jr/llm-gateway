"""add fallback_reason to calls

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0006'
down_revision: Union[str, Sequence[str], None] = '0005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Nullable: empty means Groq answered normally (or it was a cache hit)."""
    op.add_column("calls", sa.Column("fallback_reason", sa.String(length=50), nullable=True))


def downgrade() -> None:
    op.drop_column("calls", "fallback_reason")
