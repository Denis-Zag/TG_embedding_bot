"""Точка запуска Telegram-бота."""

from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message

from app.assistant_service import AssistantService, format_search_results
from app.config import ConfigurationError, load_settings
from app.factory import create_assistant, create_knowledge_base
from app.knowledge_base import KnowledgeBase

router = Router(name=__name__)
logger = logging.getLogger(__name__)

HELP_TEXT = (
    "Я ищу информацию по смыслу в базе учебных проектов.\n\n"
    "Команды:\n"
    "/search <запрос> — показать ID, тексты и score из Pinecone\n"
    "/status — показать количество загруженных векторов\n"
    "/help — показать эту справку\n\n"
    "Можно также просто задать вопрос обычным сообщением."
)


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    """Поздороваться и объяснить назначение бота."""

    await message.answer(
        "Привет! Я помощник по учебным проектам Дениса.\n\n" + HELP_TEXT
    )


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    """Показать справку."""

    await message.answer(HELP_TEXT)


@router.message(Command("status"))
async def status_handler(message: Message, knowledge_base: KnowledgeBase) -> None:
    """Показать число векторов в учебном пространстве."""

    try:
        count = await asyncio.to_thread(knowledge_base.vector_count)
        await message.answer(f"В Pinecone загружено векторов: {count}")
    except Exception:
        logger.exception("Не удалось получить статистику Pinecone")
        await message.answer("Не удалось получить статистику. Проверь настройки сервисов.")


@router.message(Command("search"))
async def search_handler(message: Message, knowledge_base: KnowledgeBase) -> None:
    """Показать необработанные результаты семантического поиска."""

    query = (message.text or "").partition(" ")[2].strip()
    if not query:
        await message.answer("Добавь запрос после команды: /search как найти проект с Docker")
        return
    try:
        raw_matches = await asyncio.to_thread(knowledge_base.search_raw, query)
        relevant = knowledge_base.filter_relevant(raw_matches)
        result = format_search_results(raw_matches)
        if not relevant:
            result += (
                "\n\nНадёжных совпадений нет: лучший score ниже порога "
                f"{knowledge_base.score_threshold:.2f}."
            )
        await message.answer(result)
    except Exception:
        logger.exception("Ошибка семантического поиска")
        await message.answer("Поиск не выполнен. Проверь ключи и доступность сервисов.")


@router.message(F.text)
async def question_handler(
    message: Message,
    knowledge_base: KnowledgeBase,
    assistant: AssistantService,
) -> None:
    """Найти контекст и сформировать ответ на обычный вопрос."""

    question = (message.text or "").strip()
    try:
        matches = await asyncio.to_thread(knowledge_base.search, question)
        answer = await asyncio.to_thread(assistant.answer, question, matches)
        await message.answer(answer)
    except Exception:
        logger.exception("Ошибка обработки сообщения")
        await message.answer("Не удалось подготовить ответ. Проверь настройки сервисов.")


async def main() -> None:
    """Запустить бота в режиме long polling."""

    settings = load_settings()
    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    await bot.set_my_commands(
        [
            BotCommand(command="search", description="Показать результаты Pinecone"),
            BotCommand(command="status", description="Показать число векторов"),
            BotCommand(command="help", description="Показать справку"),
        ]
    )
    await dispatcher.start_polling(
        bot,
        knowledge_base=create_knowledge_base(settings),
        assistant=create_assistant(settings),
        allowed_updates=dispatcher.resolve_used_update_types(),
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    try:
        asyncio.run(main())
    except ConfigurationError as error:
        logger.error("Ошибка конфигурации: %s", error)
