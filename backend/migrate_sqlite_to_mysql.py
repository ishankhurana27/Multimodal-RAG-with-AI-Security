"""
SQLite -> MySQL Data Migration Script
=====================================
Run AFTER:
  1. MySQL database is created
  2. Backend has started once (so Alembic creates all MySQL tables)
  3. PyMySQL is installed: pip install PyMySQL==1.1.1

Usage:
  cd backend
  # Run with: MYSQL_USER=... MYSQL_PASSWORD=... python migrate_sqlite_to_mysql.py
  $env:MYSQL_USER="futureverseai"; $env:MYSQL_PASSWORD="yourpassword"; py -3.11 migrate_sqlite_to_mysql.py
"""

import sqlite3
import pymysql
import json
import sys
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
SQLITE_PATH = os.path.join(os.path.dirname(__file__), "data", "webui.db")

MYSQL_HOST     = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT     = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER     = os.environ.get("MYSQL_USER", "futureverseai")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DB       = os.environ.get("MYSQL_DB", "futureverseai")

# Tables that should never be copied (migration metadata or MySQL-only)
SKIP_TABLES = {
    "alembic_version",   # migration bookkeeping — must NOT be overwritten
    "migratehistory",    # old peewee migration log, not present in MySQL
}

# Table migration order — respects foreign key dependencies
# Parent tables must come before child tables
TABLE_ORDER = [
    "auth",
    "user",
    "group",
    "folder",
    "file",
    "tag",
    "model",
    "tool",
    "function",
    "skill",
    "prompt",
    "memory",
    "knowledge",
    "chat",
    "channel",
    "note",
    "oauth_session",
    "feedback",
    "access_grant",
    "group_member",
    "chat_file",
    "chat_message",
    "message",
    "knowledge_file",
    "channel_member",
    "channel_file",
    "channel_webhook",
    "message_reaction",
    "prompt_history",
    "api_key",
]


def get_sqlite_table_names(sqlite_conn):
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    return {row[0] for row in cursor.fetchall()}


def get_sqlite_columns(sqlite_conn, table):
    cursor = sqlite_conn.cursor()
    cursor.execute(f"PRAGMA table_info(`{table}`)")
    return [row[1] for row in cursor.fetchall()]


def get_mysql_columns(mysql_conn, table):
    cursor = mysql_conn.cursor()
    cursor.execute(f"SHOW COLUMNS FROM `{table}`")
    return {row[0] for row in cursor.fetchall()}


def migrate_table(sqlite_conn, mysql_conn, table):
    sqlite_cur = sqlite_conn.cursor()
    mysql_cur  = mysql_conn.cursor()

    # Count rows
    sqlite_cur.execute(f"SELECT COUNT(*) FROM `{table}`")
    total = sqlite_cur.fetchone()[0]

    if total == 0:
        print(f"  [{table}] skipped — 0 rows")
        return 0

    # Only use columns that exist in BOTH databases.
    # SQLite may have old dropped columns (api_key, oauth_sub on user, etc.)
    # MySQL may have new columns (profile_banner_image_url, timezone, etc.) —
    # those will be left NULL, which is fine.
    sqlite_cols = get_sqlite_columns(sqlite_conn, table)
    mysql_cols  = get_mysql_columns(mysql_conn, table)
    columns     = [c for c in sqlite_cols if c in mysql_cols]

    if not columns:
        print(f"  [{table}] skipped — no overlapping columns")
        return 0

    dropped = set(sqlite_cols) - mysql_cols
    if dropped:
        print(f"  [{table}] dropping SQLite-only columns: {sorted(dropped)}")

    col_list     = ", ".join(f"`{c}`" for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))

    # Read all rows from SQLite (only common columns)
    sqlite_cur.execute(f"SELECT {col_list} FROM `{table}`")
    rows = sqlite_cur.fetchall()

    batch_size = 500
    inserted = 0
    skipped  = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        clean_batch = []

        for row in batch:
            clean_row = []
            for val in row:
                if isinstance(val, bytes):
                    try:
                        val = val.decode("utf-8")
                    except Exception:
                        val = val.decode("latin-1")
                clean_row.append(val)
            clean_batch.append(tuple(clean_row))

        try:
            sql = (
                f"INSERT IGNORE INTO `{table}` ({col_list}) "
                f"VALUES ({placeholders})"
            )
            mysql_cur.executemany(sql, clean_batch)
            mysql_conn.commit()
            inserted += mysql_cur.rowcount
        except Exception as e:
            mysql_conn.rollback()
            print(f"  [{table}] ERROR on batch {i//batch_size + 1}: {e}")
            skipped += len(batch)

    print(f"  [{table}] {total} rows -> inserted: {inserted}, skipped/duplicate: {skipped}")
    return inserted


def main():
    # Verify SQLite file exists
    if not os.path.exists(SQLITE_PATH):
        print(f"ERROR: SQLite file not found at {SQLITE_PATH}")
        sys.exit(1)

    print(f"Source (SQLite): {SQLITE_PATH}")
    print(f"Target (MySQL):  {MYSQL_USER}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}")
    print()

    # Connect to SQLite
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = None

    # Connect to MySQL
    try:
        mysql_conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            charset="utf8mb4",
            autocommit=False,
        )
    except Exception as e:
        print(f"ERROR: Cannot connect to MySQL: {e}")
        print("Make sure MySQL is running and credentials in this script are correct.")
        sys.exit(1)

    mysql_cur = mysql_conn.cursor()

    # Disable FK checks during migration so order doesn't block inserts
    mysql_cur.execute("SET FOREIGN_KEY_CHECKS = 0")
    mysql_conn.commit()

    # Get tables that actually exist in SQLite
    sqlite_tables = get_sqlite_table_names(sqlite_conn)
    print(f"Tables found in SQLite: {sorted(sqlite_tables)}\n")

    total_inserted = 0
    for table in TABLE_ORDER:
        if table not in sqlite_tables:
            print(f"  [{table}] not in SQLite — skipping")
            continue
        total_inserted += migrate_table(sqlite_conn, mysql_conn, table)

    # Migrate any SQLite tables not in our ordered list that also exist in MySQL
    mysql_cur2 = mysql_conn.cursor()
    mysql_cur2.execute("SHOW TABLES")
    mysql_all_tables = {row[0] for row in mysql_cur2.fetchall()}

    remaining = sqlite_tables - set(TABLE_ORDER) - SKIP_TABLES
    for table in sorted(remaining):
        if table not in mysql_all_tables:
            print(f"  [{table}] (extra table, not in MySQL — skipping)")
            continue
        print(f"  [{table}] (extra table)")
        total_inserted += migrate_table(sqlite_conn, mysql_conn, table)

    # Re-enable FK checks
    mysql_cur.execute("SET FOREIGN_KEY_CHECKS = 1")
    mysql_conn.commit()

    sqlite_conn.close()
    mysql_conn.close()

    print(f"\nDone. Total rows migrated: {total_inserted}")
    print("You can now start the backend — it will use MySQL with your existing data.")


if __name__ == "__main__":
    main()
