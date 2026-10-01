"""Клиент создания векторных представлений через ProxyAPI."""

from __future__ import annotations

from openai import OpenAI


class EmbeddingClient:
    """Создаёт embeddings для текстов и поисковых запросов."""

    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        self._model = model

    def create(self, texts: list[str]) -> list[list[float]]:
        """Преобразовать список непустых строк в список векторов."""

        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Для создания embeddings нужны непустые строки")
        response = self._client.embeddings.create(input=texts, model=self._model)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
