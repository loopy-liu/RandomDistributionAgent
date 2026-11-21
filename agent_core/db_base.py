from __future__ import annotations
import os
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor


class DatabaseAdapter:
    """统一数据库适配器：本地默认SQLite，配置SUPABASE_DB_URI环境变量自动切换PostgreSQL"""

    def __init__(self, database_path: str | Path | None = None):
        self.pg_uri = os.environ.get("SUPABASE_DB_URI", "")
        self.is_postgres = bool(self.pg_uri.strip())

        if self.is_postgres:
            self.sqlite_path = None
        else:
            self.sqlite_path = Path(database_path) if database_path else Path("data/agent.sqlite3")
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self):
        if self.is_postgres:
            conn = psycopg2.connect(self.pg_uri, sslmode='require')
            conn.cursor_factory = RealDictCursor
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            try:
                with conn:
                    yield conn
            finally:
                conn.close()
