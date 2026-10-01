"""Проверка создания одного embedding-вектора."""

import sys

from app.config import ConfigurationError, load_settings
from app.factory import create_knowledge_base


def main() -> None:
    """Создать тестовый вектор и вывести его размерность."""

    settings = load_settings(require_telegram_token=False)
    vector = create_knowledge_base(settings).check_embedding(
        "Telegram-бот использует семантический поиск по учебным проектам."
    )
    print(f"Embedding создан. Размерность: {len(vector)}")
    print(f"Первые пять значений: {vector[:5]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except ConfigurationError as error:
        print(f"Ошибка конфигурации: {error}")
