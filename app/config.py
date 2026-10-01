"""Загрузка и проверка настроек приложения."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Ошибка отсутствующей или некорректной настройки."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Настройки внешних сервисов и поиска."""

    telegram_bot_token: str
    proxyapi_key: str
    proxyapi_base_url: str
    embedding_model: str
    embedding_dimension: int
    chat_model: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_namespace: str
    search_top_k: int
    search_score_threshold: float


def _positive_int(name: str, default: str) -> int:
    raw_value = os.getenv(name, default)
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} должен быть целым числом") from error
    if value <= 0:
        raise ConfigurationError(f"{name} должен быть больше нуля")
    return value


def _score_threshold(name: str, default: str) -> float:
    raw_value = os.getenv(name, default)
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} должен быть числом") from error
    if not 0 <= value <= 1:
        raise ConfigurationError(f"{name} должен находиться в диапазоне от 0 до 1")
    return value


def load_settings(*, require_telegram_token: bool = True) -> Settings:
    """Загрузить настройки из переменных окружения и файла .env."""

    load_dotenv()
    values = {
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        "proxyapi_key": os.getenv("PROXYAPI_KEY", "").strip(),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY", "").strip(),
    }
    required = ["proxyapi_key", "pinecone_api_key"]
    if require_telegram_token:
        required.append("telegram_bot_token")

    missing = [name for name in required if not values[name]]
    if missing:
        env_names = {
            "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
            "proxyapi_key": "PROXYAPI_KEY",
            "pinecone_api_key": "PINECONE_API_KEY",
        }
        missing_text = ", ".join(env_names[name] for name in missing)
        raise ConfigurationError(f"Заполните переменные в .env: {missing_text}")

    return Settings(
        telegram_bot_token=values["telegram_bot_token"],
        proxyapi_key=values["proxyapi_key"],
        proxyapi_base_url=os.getenv(
            "PROXYAPI_BASE_URL", "https://api.proxyapi.ru/openai/v1"
        ).strip(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        ).strip(),
        embedding_dimension=_positive_int("EMBEDDING_DIMENSION", "1536"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini").strip(),
        pinecone_api_key=values["pinecone_api_key"],
        pinecone_index_name=os.getenv(
            "PINECONE_INDEX_NAME", "tg-embedding-projects"
        ).strip(),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
        pinecone_namespace=os.getenv(
            "PINECONE_NAMESPACE", "educational-projects"
        ).strip(),
        search_top_k=_positive_int("SEARCH_TOP_K", "5"),
        search_score_threshold=_score_threshold("SEARCH_SCORE_THRESHOLD", "0.40"),
    )
