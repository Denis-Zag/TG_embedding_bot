"""Офлайн-проверки данных и основной логики."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.assistant_service import AssistantService, build_context, format_search_results
from app.data_loader import load_queries, load_records
from app.knowledge_base import KnowledgeBase
from app.models import KnowledgeRecord, SearchMatch

ROOT = Path(__file__).resolve().parents[1]


class FakeEmbedder:
    """Предсказуемая замена внешнего Embedding API."""

    def create(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.5, 1.0] for text in texts]


class FakeStore:
    """Замена Pinecone без сетевых запросов."""

    def __init__(self) -> None:
        self.records: list[KnowledgeRecord] = []
        self.top_k = 0

    def upsert(
        self,
        records: list[KnowledgeRecord],
        embeddings: list[list[float]],
    ) -> int:
        self.records = records
        return len(embeddings)

    def search(self, vector: list[float], *, top_k: int) -> list[SearchMatch]:
        self.top_k = top_k
        return [
            SearchMatch(
                record_id="proj-024",
                score=0.97,
                project="TG embedding bot",
                category="vector-search",
                text="Бот ищет проекты по смысловой близости.",
            )
        ]

    def vector_count(self) -> int:
        return len(self.records)


class DatasetTests(unittest.TestCase):
    def test_homework_dataset_has_at_least_twenty_unique_records(self) -> None:
        records = load_records(ROOT / "data" / "projects.json")
        self.assertGreaterEqual(len(records), 20)
        self.assertEqual(len({record.record_id for record in records}), len(records))

    def test_homework_has_five_unique_queries(self) -> None:
        queries = load_queries(ROOT / "data" / "homework_queries.json")
        self.assertEqual(len(queries), 5)
        self.assertEqual(len(set(queries)), 5)

    def test_negative_queries_are_unique(self) -> None:
        queries = load_queries(ROOT / "data" / "negative_queries.json")
        self.assertEqual(len(queries), 5)
        self.assertEqual(len(set(queries)), 5)

    def test_duplicate_ids_are_rejected(self) -> None:
        source = (
            '[{"id":"same","project":"A","category":"x","text":"one"},'
            '{"id":"same","project":"B","category":"y","text":"two"}]'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "уникальными"):
                load_records(path)


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeStore()
        self.knowledge_base = KnowledgeBase(
            embedder=FakeEmbedder(),
            store=self.store,
            top_k=5,
            expected_dimension=3,
            score_threshold=0.4,
        )

    def test_upload_and_search(self) -> None:
        records = [KnowledgeRecord("id-1", "Project", "test", "Полезный факт")]
        self.assertEqual(self.knowledge_base.upload(records), 1)
        matches = self.knowledge_base.search("Что найти?")
        self.assertEqual(matches[0].record_id, "proj-024")
        self.assertEqual(self.store.top_k, 5)

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "пустым"):
            self.knowledge_base.search("   ")

    def test_low_score_matches_are_rejected(self) -> None:
        matches = [SearchMatch("id-1", 0.2, "Project", "test", "Факт")]
        self.assertEqual(self.knowledge_base.filter_relevant(matches), [])


class FormattingTests(unittest.TestCase):
    def test_search_output_contains_id_score_and_text(self) -> None:
        match = SearchMatch("id-7", 0.91234, "Project", "api", "Описание")
        output = format_search_results([match])
        self.assertIn("ID: id-7", output)
        self.assertIn("Score: 0.9123", output)
        self.assertIn("Текст: Описание", output)

    def test_context_contains_source_id(self) -> None:
        match = SearchMatch("id-7", 0.9, "Project", "api", "Описание")
        self.assertIn("[id-7]", build_context([match]))

    def test_assistant_does_not_call_model_without_relevant_matches(self) -> None:
        assistant = AssistantService(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="test-model",
        )
        answer = assistant.answer("Вопрос вне базы", [])
        self.assertIn("нет достаточно релевантных данных", answer)


if __name__ == "__main__":
    unittest.main()
