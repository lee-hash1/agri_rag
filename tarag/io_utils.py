from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import KnowledgeDoc


def load_raw_records(path: str | Path) -> list[Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if isinstance(data.get("data"), list):
                return data["data"]
            return [data]
        raise ValueError("Unsupported JSON format: top-level must be list or dict.")

    if suffix == ".jsonl":
        records: list[Any] = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append(line)
        return records

    if suffix in {".txt", ".md"}:
        return [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    raise ValueError(
        f"Unsupported input format: {suffix}. "
        "Please use .json, .jsonl, .txt, or .md."
    )


def save_clean_docs(path: str | Path, docs: list[KnowledgeDoc]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [doc.to_dict() for doc in docs]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_clean_docs(path: str | Path) -> list[KnowledgeDoc]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Knowledge base file does not exist: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Knowledge base file must be a JSON array.")

    docs: list[KnowledgeDoc] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        doc = KnowledgeDoc.from_dict(item)
        if doc.is_valid():
            docs.append(doc)

    if not docs:
        raise ValueError("No valid docs found in knowledge base file.")
    return docs


def save_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_query_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Query file does not exist: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Query file must be a JSON array.")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Query record at index {index} must be a JSON object.")

        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Query record at index {index} must contain non-empty string field `query`.")

        records.append(
            {
                "query": query,
                "golden_answer": item.get("golden_answer"),
                "answer": item.get("answer"),
            }
        )
    return records
