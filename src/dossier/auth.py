from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from fastapi import Header, HTTPException, status

from .config import get_settings


@dataclass
class User:
    api_key: str
    name: str
    role: str = "analyst"
    departments: list[str] = field(default_factory=list)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def allowed_departments(self) -> list[str] | None:
        """None means unrestricted (admin)."""
        return None if self.is_admin else [d.lower() for d in self.departments]

    def can_access(self, department: str) -> bool:
        allowed = self.allowed_departments()
        return allowed is None or department.lower() in allowed

    def scope_key(self) -> str:
        allowed = self.allowed_departments()
        return "*" if allowed is None else ",".join(sorted(allowed))


class UserStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or get_settings().users_file)
        self._users: dict[str, User] = {}
        self.reload()

    def reload(self) -> None:
        self._users = {}
        if not self.path.exists():
            return
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        for raw in data.get("users", []):
            user = User(
                api_key=str(raw["api_key"]),
                name=raw.get("name", "user"),
                role=raw.get("role", "analyst"),
                departments=[str(d) for d in raw.get("departments", [])],
            )
            self._users[user.api_key] = user

    def get(self, api_key: str) -> User | None:
        return self._users.get(api_key)

    def add(self, user: User) -> None:
        self._users[user.api_key] = user


_store: UserStore | None = None


def get_user_store() -> UserStore:
    global _store
    if _store is None:
        _store = UserStore()
    return _store


def set_user_store(store: UserStore) -> None:
    global _store
    _store = store


def current_user(x_api_key: str | None = Header(default=None)) -> User:
    if not x_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-API-Key header")
    user = get_user_store().get(x_api_key)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
    return user
