"""Загрузка учебных фактов в Pinecone."""

import sys
from pathlib import Path

from app.config import ConfigurationError, load_settings
from app.data_loader import load_records
from app.factory import create_knowledge_base

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Создать embeddings и загрузить все факты."""

    settings = load_settings(require_telegram_token=False)
    records = load_records(ROOT / "data" / "projects.json")
    uploaded = create_knowledge_base(settings).upload(records)
    print(f"Загружено векторов: {uploaded}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except ConfigurationError as error:
        print(f"Ошибка конфигурации: {error}")
