"""Формирование ответа модели на основе результатов Pinecone."""

from __future__ import annotations

from openai import OpenAI

from app.models import SearchMatch


def build_context(matches: list[SearchMatch]) -> str:
    """Собрать найденные факты в читаемый контекст."""

    if not matches:
        return "Подходящих фактов не найдено."
    return "\n".join(
        f"{number}. [{match.record_id}] {match.project}: {match.text}"
        for number, match in enumerate(matches, start=1)
    )


def format_search_results(matches: list[SearchMatch]) -> str:
    """Подготовить ID, score и тексты для команды /search."""

    if not matches:
        return "Совпадений не найдено."
    blocks = [
        (
            f"{number}. ID: {match.record_id}\n"
            f"Проект: {match.project}\n"
            f"Score: {match.score:.4f}\n"
            f"Текст: {match.text}"
        )
        for number, match in enumerate(matches, start=1)
    ]
    return "\n\n".join(blocks)


class AssistantService:
    """Генерирует ответ, используя только найденный учебный контекст."""

    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=45.0)
        self._model = model

    def answer(self, question: str, matches: list[SearchMatch]) -> str:
        """Ответить на вопрос по релевантным фактам."""

        if not matches:
            return "В базе знаний нет достаточно релевантных данных для ответа."
        context = build_context(matches)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты помощник по учебным проектам Дениса. Отвечай по-русски, "
                        "кратко и только на основе переданного контекста. Если данных "
                        "недостаточно, честно скажи об этом. Не придумывай факты."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Контекст:\n{context}\n\nВопрос: {question}",
                },
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else "Модель вернула пустой ответ."
