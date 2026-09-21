"""Database configuration and connection pooling helpers for marketlake."""
import sqlite3
from typing import Optional

def configure_sqlite_connection(conn: sqlite3.Connection, busy_timeout_ms: int = 5000) -> None:
    """
    Apply high-concurrency PRAGMA settings to SQLite connection.
    Enables WAL mode and sets busy timeout to prevent locking errors.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute(f"PRAGMA busy_timeout = {busy_timeout_ms};")
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.close()
