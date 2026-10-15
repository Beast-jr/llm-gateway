from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context

import models.call  # noqa: F401  (makes the tables known to Base.metadata)
import models.semantic_cache  # noqa: F401
from services.db import Base, get_database_url

config = context.config

# Set up logging from alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Lets `alembic revision --autogenerate` compare our models with the real database.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Print the SQL instead of running it (`alembic upgrade head --sql`)."""
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database from .env (DATABASE_URL) and run the migrations."""
    engine = create_engine(get_database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
