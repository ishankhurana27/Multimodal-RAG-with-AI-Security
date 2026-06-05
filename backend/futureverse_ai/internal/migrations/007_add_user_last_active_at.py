"""Peewee migrations -- 007_add_user_last_active_at.py."""

import peewee as pw
from peewee_migrate import Migrator


def _get_columns(database, table):
    """Return set of existing column names (lowercase) for a table."""
    cursor = database.execute_sql(
        "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s",
        (table,),
    )
    return {row[0].lower() for row in cursor.fetchall()}


def migrate(migrator: Migrator, database: pw.Database, *, fake=False):
    """Idempotent — safe to re-run after a partial failure."""

    existing = _get_columns(database, "user")

    # Add only columns that don't already exist (handles partial-run resume)
    fields_to_add = {}
    if "created_at" not in existing:
        fields_to_add["created_at"] = pw.BigIntegerField(null=True)
    if "updated_at" not in existing:
        fields_to_add["updated_at"] = pw.BigIntegerField(null=True)
    if "last_active_at" not in existing:
        fields_to_add["last_active_at"] = pw.BigIntegerField(null=True)

    if fields_to_add:
        migrator.add_fields("user", **fields_to_add)

    # Copy data from timestamp only if it still exists
    if "timestamp" in existing:
        migrator.sql(
            "UPDATE `user` SET created_at = timestamp, updated_at = timestamp, "
            "last_active_at = timestamp WHERE timestamp IS NOT NULL"
        )
        migrator.remove_fields("user", "timestamp")

    # Make the new columns NOT NULL
    migrator.change_fields(
        "user",
        created_at=pw.BigIntegerField(null=False),
        updated_at=pw.BigIntegerField(null=False),
        last_active_at=pw.BigIntegerField(null=False),
    )


def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    existing = _get_columns(database, "user")

    if "timestamp" not in existing:
        migrator.add_fields("user", timestamp=pw.BigIntegerField(null=True))

    if "created_at" in existing:
        migrator.sql("UPDATE `user` SET timestamp = created_at")
        migrator.remove_fields("user", "created_at", "updated_at", "last_active_at")

    migrator.change_fields("user", timestamp=pw.BigIntegerField(null=False))
