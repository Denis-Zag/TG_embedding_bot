"""Работа с индексом Pinecone."""

from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec

from app.models import KnowledgeRecord, SearchMatch


class PineconeStore:
    """Создаёт индекс, записывает векторы и выполняет поиск."""

    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        dimension: int,
        cloud: str,
        region: str,
        namespace: str,
    ) -> None:
        self._client = Pinecone(api_key=api_key)
        self._index_name = index_name
        self._dimension = dimension
        self._cloud = cloud
        self._region = region
        self._namespace = namespace

    def ensure_index(self) -> None:
        """Создать serverless-индекс, если он ещё не существует."""

        index_names = set(self._client.list_indexes().names())
        if self._index_name not in index_names:
            self._client.create_index(
                name=self._index_name,
                dimension=self._dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
                timeout=120,
            )

    def upsert(
        self,
        records: list[KnowledgeRecord],
        embeddings: list[list[float]],
    ) -> int:
        """Записать векторы и исходные тексты в Pinecone."""

        if len(records) != len(embeddings):
            raise ValueError("Число записей и embeddings должно совпадать")
        self.ensure_index()
        vectors = [
            {
                "id": record.record_id,
                "values": embedding,
                "metadata": {
                    "text": record.text,
                    "project": record.project,
                    "category": record.category,
                },
            }
            for record, embedding in zip(records, embeddings, strict=True)
        ]
        response = self._client.Index(self._index_name).upsert(
            vectors=vectors,
            namespace=self._namespace,
        )
        return int(response.upserted_count)

    def search(self, vector: list[float], *, top_k: int) -> list[SearchMatch]:
        """Найти ближайшие векторы и вернуть их метаданные."""

        self.ensure_index()
        response = self._client.Index(self._index_name).query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self._namespace,
        )
        matches: list[SearchMatch] = []
        for item in response.matches:
            metadata = item.metadata or {}
            matches.append(
                SearchMatch(
                    record_id=str(item.id),
                    score=float(item.score),
                    project=str(metadata.get("project", "Без названия")),
                    category=str(metadata.get("category", "Без категории")),
                    text=str(metadata.get("text", "Текст отсутствует")),
                )
            )
        return matches

    def vector_count(self) -> int:
        """Вернуть число векторов в учебном пространстве имён."""

        self.ensure_index()
        stats = self._client.Index(self._index_name).describe_index_stats()
        namespace = stats.namespaces.get(self._namespace)
        return int(namespace.vector_count) if namespace else 0
