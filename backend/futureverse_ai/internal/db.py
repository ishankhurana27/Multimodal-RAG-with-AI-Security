import os
import json
import logging
from contextlib import contextmanager
from typing import Any, Optional

from futureverse_ai.internal.wrappers import register_connection
from futureverse_ai.env import (
    futureverse_ai_DIR,
    DATABASE_URL,
    DATABASE_SCHEMA,
    DATABASE_POOL_MAX_OVERFLOW,
    DATABASE_POOL_RECYCLE,
    DATABASE_POOL_SIZE,
    DATABASE_POOL_TIMEOUT,
    DATABASE_ENABLE_SQLITE_WAL,
    DATABASE_ENABLE_SESSION_SHARING,
    ENABLE_DB_MIGRATIONS,
)
from peewee_migrate import Router
from sqlalchemy import Dialect, create_engine, MetaData, event, types, String, Text
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.type_api import _T
from typing_extensions import Self

log = logging.getLogger(__name__)


# ── MySQL compatibility shims ────────────────────────────────────────────────
# The migrations and models target SQLite/PostgreSQL, where VARCHAR may be
# unbounded and TEXT columns can be indexed freely. MySQL is stricter:
#   1. Every VARCHAR must declare a length.
#   2. A TEXT/BLOB column cannot be a PRIMARY KEY, UNIQUE, or INDEX without a
#      prefix length.
# These two compiler hooks adapt the schema to MySQL automatically — applying
# to both Alembic migrations and runtime table creation — without editing the
# ~25 migration files or the model definitions.

# (1) Any length-less String compiles to VARCHAR(255) on MySQL.
@compiles(String, "mysql")
def _compile_unbounded_string_for_mysql(type_, compiler, **kw):
    if type_.length is None:
        return "VARCHAR(255)"
    return compiler.visit_VARCHAR(type_, **kw)


# (2) On CREATE TABLE, bound any keyed (PK / unique / indexed) Text or
#     unbounded-String column to VARCHAR(255) so MySQL can index it. Long
#     free-text columns (content, etc.) are left as TEXT.
@compiles(CreateTable, "mysql")
def _bound_keyed_text_columns_for_mysql(element, compiler, **kw):
    table = element.element

    keyed = set()
    for col in table.primary_key.columns:
        keyed.add(col.name)
    for constraint in table.constraints:
        if constraint.__class__.__name__ == "UniqueConstraint":
            for col in constraint.columns:
                keyed.add(col.name)
    for idx in table.indexes:
        for col in idx.columns:
            keyed.add(col.name)
    for col in table.columns:
        if col.primary_key or col.unique or col.index:
            keyed.add(col.name)

    for col in table.columns:
        if col.name in keyed and isinstance(col.type, (String, Text)):
            if getattr(col.type, "length", None) is None:
                col.type = String(255)

    return compiler.visit_create_table(element, **kw)


class JSONField(types.TypeDecorator):
    impl = types.Text
    cache_ok = True

    def process_bind_param(self, value: Optional[_T], dialect: Dialect) -> Any:
        return json.dumps(value)

    def process_result_value(self, value: Optional[_T], dialect: Dialect) -> Any:
        if value is not None:
            return json.loads(value)

    def copy(self, **kw: Any) -> Self:
        return JSONField(self.impl.length)

    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        if value is not None:
            return json.loads(value)


# Workaround to handle the peewee migration
# This is required to ensure the peewee migration is handled before the alembic migration
def handle_peewee_migration(DATABASE_URL):
    # db = None
    try:
        # Replace the postgresql:// with postgres:// to handle the peewee migration
        db = register_connection(DATABASE_URL.replace("postgresql://", "postgres://"))
        migrate_dir = futureverse_ai_DIR / "internal" / "migrations"
        router = Router(db, logger=log, migrate_dir=migrate_dir)
        router.run()
        db.close()

    except Exception as e:
        log.error(f"Failed to initialize the database connection: {e}")
        log.warning(
            "Hint: If your database password contains special characters, you may need to URL-encode it."
        )
        raise
    finally:
        # Properly closing the database connection
        if db and not db.is_closed():
            db.close()

        # Assert if db connection has been closed
        assert db.is_closed(), "Database connection is still open."


if ENABLE_DB_MIGRATIONS and "sqlite" in DATABASE_URL:
    # Peewee migrations are a SQLite-only legacy system.
    # MySQL and PostgreSQL use Alembic exclusively (runs in config.py).
    handle_peewee_migration(DATABASE_URL)


SQLALCHEMY_DATABASE_URL = DATABASE_URL

# Handle SQLCipher URLs
if SQLALCHEMY_DATABASE_URL.startswith("sqlite+sqlcipher://"):
    database_password = os.environ.get("DATABASE_PASSWORD")
    if not database_password or database_password.strip() == "":
        raise ValueError(
            "DATABASE_PASSWORD is required when using sqlite+sqlcipher:// URLs"
        )

    # Extract database path from SQLCipher URL
    db_path = SQLALCHEMY_DATABASE_URL.replace("sqlite+sqlcipher://", "")

    # Create a custom creator function that uses sqlcipher3
    def create_sqlcipher_connection():
        import sqlcipher3

        conn = sqlcipher3.connect(db_path, check_same_thread=False)
        conn.execute(f"PRAGMA key = '{database_password}'")
        return conn

    engine = create_engine(
        "sqlite://",  # Dummy URL since we're using creator
        creator=create_sqlcipher_connection,
        echo=False,
    )

    log.info("Connected to encrypted SQLite database using SQLCipher")

elif "sqlite" in SQLALCHEMY_DATABASE_URL:
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )

    def on_connect(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        if DATABASE_ENABLE_SQLITE_WAL:
            cursor.execute("PRAGMA journal_mode=WAL")
        else:
            cursor.execute("PRAGMA journal_mode=DELETE")
        cursor.close()

    event.listen(engine, "connect", on_connect)
else:
    if isinstance(DATABASE_POOL_SIZE, int):
        if DATABASE_POOL_SIZE > 0:
            engine = create_engine(
                SQLALCHEMY_DATABASE_URL,
                pool_size=DATABASE_POOL_SIZE,
                max_overflow=DATABASE_POOL_MAX_OVERFLOW,
                pool_timeout=DATABASE_POOL_TIMEOUT,
                pool_recycle=DATABASE_POOL_RECYCLE,
                pool_pre_ping=True,
                poolclass=QueuePool,
            )
        else:
            engine = create_engine(
                SQLALCHEMY_DATABASE_URL, pool_pre_ping=True, poolclass=NullPool
            )
    else:
        engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)


SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, expire_on_commit=False
)
metadata_obj = MetaData(schema=DATABASE_SCHEMA)
Base = declarative_base(metadata=metadata_obj)
ScopedSession = scoped_session(SessionLocal)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


get_db = contextmanager(get_session)


@contextmanager
def get_db_context(db: Optional[Session] = None):
    if isinstance(db, Session) and DATABASE_ENABLE_SESSION_SHARING:
        yield db
    else:
        with get_db() as session:
            yield session
