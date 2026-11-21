from __future__ import annotations
import bcrypt
from dataclasses import dataclass
from datetime import datetime, timezone
from .db_base import DatabaseAdapter


@dataclass(frozen=True)
class UserInfo:
    user_id: str
    username: str
    role: str
    created_at: str


class UserRepository:
    """用户账号仓库：密码bcrypt哈希存储，兼容SQLite/PostgreSQL"""

    DEFAULT_ADMIN_USERNAME = "admin"
    DEFAULT_ADMIN_PASSWORD = "123456"

    def __init__(self, db) -> None:
        if isinstance(db, DatabaseAdapter):
            self.db = db
        else:
            self.db = DatabaseAdapter(database_path=db)
        self._initialize()
        self._init_default_admin()

    def _adapt_sql(self, sql: str) -> str:
        if self.db.is_postgres:
            return sql
        return sql.replace("%s", "?")

    def _initialize(self) -> None:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            sql = """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'member',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            cursor.execute(self._adapt_sql(sql))

    def _init_default_admin(self) -> None:
        if self.get_by_username(self.DEFAULT_ADMIN_USERNAME) is not None:
            return
        try:
            self.register(
                username=self.DEFAULT_ADMIN_USERNAME,
                password=self.DEFAULT_ADMIN_PASSWORD,
                role="admin",
            )
        except Exception:
            pass

    @staticmethod
    def _hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def _verify_password(password: str, password_hash: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except ValueError:
            return False

    def register(self, username: str, password: str, role: str = "member") -> UserInfo:
        if not isinstance(username, str) or not username.strip() or len(username.strip()) > 32:
            raise ValueError("用户名不能为空，且长度不超过32个字符")
        if not isinstance(password, str) or len(password) < 6:
            raise ValueError("密码长度不能少于6位")
        if role not in ("member", "admin"):
            raise ValueError("角色只能是 member 或 admin")

        username = username.strip()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        user_id = f"user_{int(datetime.now().timestamp())}"
        password_hash = self._hash_password(password)

        try:
            with self.db.connect() as conn:
                cursor = conn.cursor()
                sql = """
                    INSERT INTO users (user_id, username, password_hash, role, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(self._adapt_sql(sql), (user_id, username, password_hash, role, now, now))
        except Exception as exc:
            if "unique" in str(exc).lower():
                raise ValueError(f"用户名「{username}」已存在，请更换用户名") from exc
            raise

        return UserInfo(user_id=user_id, username=username, role=role, created_at=now)

    def verify(self, username: str, password: str) -> UserInfo:
        if not isinstance(username, str) or not username.strip():
            raise ValueError("请输入用户名")
        if not isinstance(password, str) or not password:
            raise ValueError("请输入密码")

        username = username.strip()
        with self.db.connect() as conn:
            cursor = conn.cursor()
            sql = "SELECT * FROM users WHERE username = %s"
            cursor.execute(self._adapt_sql(sql), (username,))
            row = cursor.fetchone()

        if row is None:
            raise ValueError("用户名或密码错误")

        if not self._verify_password(password, row["password_hash"]):
            raise ValueError("用户名或密码错误")

        return UserInfo(
            user_id=row["user_id"],
            username=row["username"],
            role=row["role"],
            created_at=row["created_at"],
        )

    def get_by_username(self, username: str) -> UserInfo | None:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            sql = "SELECT user_id, username, role, created_at FROM users WHERE username = %s"
            cursor.execute(self._adapt_sql(sql), (username,))
            row = cursor.fetchone()
        if row is None:
            return None
        return UserInfo(
            user_id=row["user_id"],
            username=row["username"],
            role=row["role"],
            created_at=row["created_at"],
        )

    def list_all_users(self) -> list[UserInfo]:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            sql = "SELECT user_id, username, role, created_at FROM users ORDER BY created_at"
            cursor.execute(self._adapt_sql(sql))
            rows = cursor.fetchall()
        return [
            UserInfo(
                user_id=row["user_id"],
                username=row["username"],
                role=row["role"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_user(self, username: str, operator: str) -> None:
        if username == self.DEFAULT_ADMIN_USERNAME:
            raise ValueError("不能删除默认管理员账号")
        if operator == username:
            raise ValueError("不能删除当前登录的账号")
        
        with self.db.connect() as conn:
            cursor = conn.cursor()
            sql = "DELETE FROM users WHERE username = %s"
            cursor.execute(self._adapt_sql(sql), (username,))
            if cursor.rowcount == 0:
                raise ValueError(f"找不到用户：{username}")
