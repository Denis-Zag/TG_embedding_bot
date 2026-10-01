"""Безопасная проверка внешних сервисов без запуска polling."""

from __future__ import annotations

import asyncio
import sys

from aiogram import Bot

from app.config import ConfigurationError, load_settings
from app.factory import create_assistant, create_knowledge_base


async def check_telegram(token: str) -> str:
    """Проверить токен Telegram и вернуть имя бота."""

    bot = Bot(token=token)
    try:
        info = await bot.get_me()
        return f"@{info.username}" if info.username else str(info.id)
    finally:
        await bot.session.close()


def main() -> None:
    """Последовательно проверить Telegram, Pinecone и ProxyAPI."""

    settings = load_settings()
    bot_name = asyncio.run(check_telegram(settings.telegram_bot_token))
    print(f"Telegram: подключение успешно, бот {bot_name}")

    knowledge_base = create_knowledge_base(settings)
    count = knowledge_base.vector_count()
    print(f"Pinecone: подключение успешно, векторов в namespace: {count}")

    question = "Какой проект использует семантический поиск?"
    matches = knowledge_base.search(question)
    if not matches:
        raise RuntimeError("Pinecone не вернул результаты поиска")
    print(
        "Семантический поиск: "
        f"первый результат {matches[0].record_id}, score={matches[0].score:.4f}"
    )

    answer = create_assistant(settings).answer(question, matches)
    print(f"Языковая модель: ответ получен — {answer}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except ConfigurationError as error:
        print(f"Ошибка конфигурации: {error}")
