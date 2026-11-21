from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from .db_base import DatabaseAdapter


@dataclass(frozen=True)
class AuditLogEntry:
    log_id: str
    operator: str
    action: str
    target: str
    detail: str
    created_at: str


class AuditLogRepository:
    """操作审计日志仓库，兼容SQLite/PostgreSQL"""

    def __init__(self, db) -> None:
        if isinstance(db, DatabaseAdapter):
            self.db = db
        else:
            self.db = DatabaseAdapter(database_path=db)
        self._initialize()

    def _adapt_sql(self, sql: str) -> str:
        if self.db.is_postgres:
            return sql
        return sql.replace("%s", "?")

    def _initialize(self) -> None:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            sql = """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id TEXT PRIMARY KEY,
                    operator TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """
            cursor.execute(self._adapt_sql(sql))
            index_sql = "CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at DESC)"
            cursor.execute(self._adapt_sql(index_sql))

    def log(self, operator: str, action: str, target: str, detail: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log_id = f"log_{int(datetime.now().timestamp())}_{abs(hash(f'{operator}{action}{target}{now}')) % 100000}"
        with self.db.connect() as conn:
            cursor = conn.cursor()
            sql = """
                INSERT INTO audit_logs (log_id, operator, action, target, detail, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(self._adapt_sql(sql), (log_id, operator, action, target, detail, now))

    def list_recent(self, limit: int = 100) -> list[AuditLogEntry]:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            sql = "SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT %s"
            cursor.execute(self._adapt_sql(sql), (limit,))
            rows = cursor.fetchall()
        return [
            AuditLogEntry(
                log_id=row["log_id"],
                operator=row["operator"],
                action=row["action"],
                target=row["target"],
                detail=row["detail"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
