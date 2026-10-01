"""Создание настроенных компонентов приложения."""

from app.assistant_service import AssistantService
from app.config import Settings
from app.embedding_client import EmbeddingClient
from app.knowledge_base import KnowledgeBase
from app.pinecone_store import PineconeStore


def create_knowledge_base(settings: Settings) -> KnowledgeBase:
    """Создать сервис базы знаний из настроек."""

    embedder = EmbeddingClient(
        api_key=settings.proxyapi_key,
        base_url=settings.proxyapi_base_url,
        model=settings.embedding_model,
    )
    store = PineconeStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        dimension=settings.embedding_dimension,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        namespace=settings.pinecone_namespace,
    )
    return KnowledgeBase(
        embedder=embedder,
        store=store,
        top_k=settings.search_top_k,
        expected_dimension=settings.embedding_dimension,
        score_threshold=settings.search_score_threshold,
    )


def create_assistant(settings: Settings) -> AssistantService:
    """Создать сервис генерации ответов."""

    return AssistantService(
        api_key=settings.proxyapi_key,
        base_url=settings.proxyapi_base_url,
        model=settings.chat_model,
    )
