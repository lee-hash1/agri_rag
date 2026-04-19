from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class KnowledgeDoc:
    disease: str
    stage: str
    treatment: str
    source: str | None = None

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "KnowledgeDoc":
        return cls(
            disease=str(item.get("disease", "")).strip(),
            stage=str(item.get("stage", "")).strip(),
            treatment=str(item.get("treatment", "")).strip(),
            source=(str(item.get("source")).strip() if item.get("source") is not None else None),
        )

    def is_valid(self) -> bool:
        return bool(self.disease and self.stage and self.treatment)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["source"] is None:
            payload.pop("source")
        return payload


@dataclass
class RetrievalHit:
    rank: int
    doc: KnowledgeDoc
    bm25_score: float
    stage_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "bm25_score": self.bm25_score,
            "stage_score": self.stage_score,
            "doc": self.doc.to_dict(),
        }


@dataclass
class PipelineResult:
    question: str
    parsed_query: dict[str, str]
    retrieved: list[RetrievalHit]
    answer: str
    verification: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "parsed_query": self.parsed_query,
            "retrieved": [hit.to_dict() for hit in self.retrieved],
            "answer": self.answer,
            "verification": self.verification,
        }
