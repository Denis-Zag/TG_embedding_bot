"""Чтение учебного набора фактов и проверочных запросов."""

from __future__ import annotations

import json
from pathlib import Path

from app.models import KnowledgeRecord


def load_records(path: Path) -> list[KnowledgeRecord]:
    """Прочитать записи из JSON и проверить уникальность ID."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Файл с фактами должен содержать JSON-массив")

    records = [KnowledgeRecord.from_dict(item) for item in payload]
    ids = [record.record_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("ID записей в наборе данных должны быть уникальными")
    return records


def load_queries(path: Path) -> list[str]:
    """Прочитать список запросов для демонстрации домашнего задания."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(item, str) and item.strip() for item in payload
    ):
        raise ValueError("Файл запросов должен содержать массив непустых строк")
    return [item.strip() for item in payload]
