"""add saved_cost_usd and truncated to calls, and an index on created_at

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0007'
down_revision: Union[str, Sequence[str], None] = '0006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Old rows get 0 / false: we did not record these before, so they simply add nothing to /stats.
    op.add_column("calls", sa.Column("saved_cost_usd", sa.Float(), server_default="0", nullable=False))
    op.add_column("calls", sa.Column("truncated", sa.Boolean(), server_default=sa.false(), nullable=False))
    op.create_index("ix_calls_created_at", "calls", ["created_at"])  # /stats filters by date


def downgrade() -> None:
    op.drop_index("ix_calls_created_at", table_name="calls")
    op.drop_column("calls", "truncated")
    op.drop_column("calls", "saved_cost_usd")
