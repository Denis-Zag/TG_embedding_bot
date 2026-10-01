"""Модели данных базы знаний и результатов поиска."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class KnowledgeRecord:
    """Одна фраза, которая преобразуется в вектор."""

    record_id: str
    project: str
    category: str
    text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeRecord:
        """Создать запись из словаря и проверить обязательные поля."""

        required = ("id", "project", "category", "text")
        missing = [field for field in required if not str(data.get(field, "")).strip()]
        if missing:
            raise ValueError(f"В записи отсутствуют поля: {', '.join(missing)}")
        return cls(
            record_id=str(data["id"]).strip(),
            project=str(data["project"]).strip(),
            category=str(data["category"]).strip(),
            text=str(data["text"]).strip(),
        )


@dataclass(frozen=True, slots=True)
class SearchMatch:
    """Результат семантического поиска."""

    record_id: str
    score: float
    project: str
    category: str
    text: str
