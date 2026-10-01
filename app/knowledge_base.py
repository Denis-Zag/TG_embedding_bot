"""Сценарии загрузки и семантического поиска."""

from __future__ import annotations

from app.embedding_client import EmbeddingClient
from app.models import KnowledgeRecord, SearchMatch
from app.pinecone_store import PineconeStore


class KnowledgeBase:
    """Связывает embedding-модель с векторным хранилищем."""

    def __init__(
        self,
        *,
        embedder: EmbeddingClient,
        store: PineconeStore,
        top_k: int,
        expected_dimension: int,
        score_threshold: float,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._expected_dimension = expected_dimension
        self._score_threshold = score_threshold

    def check_embedding(self, text: str) -> list[float]:
        """Создать тестовый вектор и проверить его размерность."""

        vector = self._embedder.create([text])[0]
        if len(vector) != self._expected_dimension:
            raise ValueError(
                "Размерность embedding не совпадает с настройкой: "
                f"получено {len(vector)}, ожидалось {self._expected_dimension}"
            )
        return vector

    def upload(self, records: list[KnowledgeRecord]) -> int:
        """Преобразовать факты в векторы и загрузить их в Pinecone."""

        embeddings = self._embedder.create([record.text for record in records])
        for embedding in embeddings:
            if len(embedding) != self._expected_dimension:
                raise ValueError("Обнаружен embedding с неверной размерностью")
        return self._store.upsert(records, embeddings)

    @property
    def score_threshold(self) -> float:
        """Вернуть минимальную допустимую оценку релевантности."""

        return self._score_threshold

    def search_raw(self, query: str) -> list[SearchMatch]:
        """Вернуть исходные результаты Pinecone без фильтрации по score."""

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Поисковый запрос не должен быть пустым")
        vector = self.check_embedding(clean_query)
        return self._store.search(vector, top_k=self._top_k)

    def filter_relevant(self, matches: list[SearchMatch]) -> list[SearchMatch]:
        """Оставить результаты, прошедшие порог релевантности."""

        return [match for match in matches if match.score >= self._score_threshold]

    def search(self, query: str) -> list[SearchMatch]:
        """Выполнить поиск и исключить недостаточно релевантные результаты."""

        return self.filter_relevant(self.search_raw(query))

    def vector_count(self) -> int:
        """Вернуть число загруженных векторов."""

        return self._store.vector_count()
