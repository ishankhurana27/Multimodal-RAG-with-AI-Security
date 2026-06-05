import logging
import os
from contextvars import ContextVar

from peewee import *
from peewee import InterfaceError as PeeWeeInterfaceError
from peewee import MySQLDatabase, PostgresqlDatabase
from playhouse.db_url import connect, parse
from playhouse.shortcuts import ReconnectMixin

log = logging.getLogger(__name__)

db_state_default = {"closed": None, "conn": None, "ctx": None, "transactions": None}
db_state = ContextVar("db_state", default=db_state_default.copy())


class PeeweeConnectionState(object):
    def __init__(self, **kwargs):
        super().__setattr__("_state", db_state)
        super().__init__(**kwargs)

    def __setattr__(self, name, value):
        self._state.get()[name] = value

    def __getattr__(self, name):
        value = self._state.get()[name]
        return value


class CustomReconnectMixin(ReconnectMixin):
    reconnect_errors = (
        # psycopg2 / MySQL
        (OperationalError, "termin"),
        (OperationalError, "Lost connection"),
        (OperationalError, "MySQL server has gone away"),
        (InterfaceError, "closed"),
        # peewee
        (PeeWeeInterfaceError, "closed"),
    )


class ReconnectingPostgresqlDatabase(CustomReconnectMixin, PostgresqlDatabase):
    pass


class ReconnectingMySQLDatabase(CustomReconnectMixin, MySQLDatabase):
    pass


def register_connection(db_url):
    # Check if using SQLCipher protocol
    if db_url.startswith("sqlite+sqlcipher://"):
        database_password = os.environ.get("DATABASE_PASSWORD")
        if not database_password or database_password.strip() == "":
            raise ValueError(
                "DATABASE_PASSWORD is required when using sqlite+sqlcipher:// URLs"
            )
        from playhouse.sqlcipher_ext import SqlCipherDatabase

        db_path = db_url.replace("sqlite+sqlcipher://", "")
        db = SqlCipherDatabase(db_path, passphrase=database_password)
        db.autoconnect = True
        db.reuse_if_open = True
        log.info("Connected to encrypted SQLite database using SQLCipher")

    else:
        # Strip SQLAlchemy driver suffixes so Peewee's URL parser works
        # e.g. mysql+pymysql://  ->  mysql://
        peewee_url = db_url
        if db_url.startswith("mysql+"):
            peewee_url = "mysql" + db_url[db_url.index("://"):]

        db = connect(peewee_url, unquote_user=True, unquote_password=True)

        if isinstance(db, PostgresqlDatabase):
            db.autoconnect = True
            db.reuse_if_open = True
            log.info("Connected to PostgreSQL database")
            connection = parse(peewee_url, unquote_user=True, unquote_password=True)
            db = ReconnectingPostgresqlDatabase(**connection)
            db.connect(reuse_if_open=True)

        elif isinstance(db, MySQLDatabase):
            db.autoconnect = True
            db.reuse_if_open = True
            log.info("Connected to MySQL database")
            connection = parse(peewee_url, unquote_user=True, unquote_password=True)
            # Ensure utf8mb4 charset for full Unicode + emoji support
            connection.setdefault("charset", "utf8mb4")
            db = ReconnectingMySQLDatabase(**connection)
            db.connect(reuse_if_open=True)

        elif isinstance(db, SqliteDatabase):
            db.autoconnect = True
            db.reuse_if_open = True
            log.info("Connected to SQLite database")

        else:
            raise ValueError(f"Unsupported database connection: {db_url}")

    return db
