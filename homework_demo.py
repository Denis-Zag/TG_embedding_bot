"""Полный сценарий проверки домашнего задания."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from app.config import ConfigurationError, load_settings
from app.data_loader import load_queries, load_records
from app.factory import create_knowledge_base

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Загрузить факты, выполнить пять запросов и сохранить результат."""

    settings = load_settings(require_telegram_token=False)
    knowledge_base = create_knowledge_base(settings)
    records = load_records(ROOT / "data" / "projects.json")
    queries = load_queries(ROOT / "data" / "homework_queries.json")
    negative_queries = load_queries(ROOT / "data" / "negative_queries.json")

    uploaded = knowledge_base.upload(records)
    report: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "index_name": settings.pinecone_index_name,
        "namespace": settings.pinecone_namespace,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "uploaded_vectors": uploaded,
        "queries": [],
        "negative_queries": [],
    }

    query_results: list[dict[str, object]] = []
    for query in queries:
        matches = knowledge_base.search_raw(query)
        relevant = knowledge_base.filter_relevant(matches)
        query_results.append(
            {
                "query": query,
                "answer_allowed": bool(relevant),
                "matches": [
                    {
                        "id": match.record_id,
                        "score": round(match.score, 6),
                        "project": match.project,
                        "text": match.text,
                    }
                    for match in matches
                ],
            }
        )
        print(f"\nЗапрос: {query}")
        for match in matches:
            print(f"  {match.record_id} | {match.score:.4f} | {match.text}")

    report["queries"] = query_results

    negative_results: list[dict[str, object]] = []
    for query in negative_queries:
        matches = knowledge_base.search_raw(query)
        relevant = knowledge_base.filter_relevant(matches)
        negative_results.append(
            {
                "query": query,
                "answer_allowed": bool(relevant),
                "matches": [
                    {
                        "id": match.record_id,
                        "score": round(match.score, 6),
                        "project": match.project,
                        "text": match.text,
                    }
                    for match in matches
                ],
            }
        )
        print(f"\nНегативный запрос: {query}")
        for match in matches:
            print(f"  {match.record_id} | {match.score:.4f} | {match.text}")
        print(
            "  Решение: "
            + ("ответ разрешён" if relevant else "ответ запрещён — данных недостаточно")
        )

    report["negative_queries"] = negative_results
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "search_results.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nРезультат сохранён: {output_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except ConfigurationError as error:
        print(f"Ошибка конфигурации: {error}")
