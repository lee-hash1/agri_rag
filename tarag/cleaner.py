from __future__ import annotations

import json
from typing import Any

from .local_llm import LocalLLM
from .schemas import KnowledgeDoc


class KnowledgeBaseCleaner:
    SYSTEM_PROMPT = (
        "你是农业知识清洗助手。"
        "请把输入中的病虫害防治知识整理成标准 JSON 数组，每条仅包含字段："
        "disease, stage, treatment。"
        "不要输出任何额外说明。"
        "如果信息不完整，请尽力保留可确定内容，无法确定的字段用空字符串。"
    )

    def __init__(
        self,
        llm: LocalLLM,
        batch_size: int = 4,
        passthrough_structured: bool = False,
    ) -> None:
        self.llm = llm
        self.batch_size = max(1, batch_size)
        self.passthrough_structured = passthrough_structured

    @staticmethod
    def _already_structured(record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        required = {"disease", "stage", "treatment"}
        return required.issubset(record.keys())

    @staticmethod
    def _render_record(record: Any) -> str:
        if isinstance(record, str):
            return record
        if isinstance(record, dict):
            return json.dumps(record, ensure_ascii=False)
        return str(record)

    def clean_records(self, raw_records: list[Any]) -> list[KnowledgeDoc]:
        cleaned_docs: list[KnowledgeDoc] = []
        llm_buffer: list[Any] = []

        for record in raw_records:
            if self.passthrough_structured and self._already_structured(record):
                doc = KnowledgeDoc.from_dict(record)
                if doc.is_valid():
                    cleaned_docs.append(doc)
                continue
            llm_buffer.append(record)

        for start in range(0, len(llm_buffer), self.batch_size):
            chunk = llm_buffer[start : start + self.batch_size]
            numbered_lines = [
                f"{idx + 1}. {self._render_record(record)}" for idx, record in enumerate(chunk)
            ]
            user_prompt = (
                "请清洗下面的农业知识记录，并返回 JSON 数组：\n"
                + "\n".join(numbered_lines)
            )

            parsed = self.llm.ask_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                fallback=[],
                max_new_tokens=1024,
            )

            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                parsed = []

            for item in parsed:
                if not isinstance(item, dict):
                    continue
                doc = KnowledgeDoc.from_dict(item)
                if doc.is_valid():
                    cleaned_docs.append(doc)
        return cleaned_docs
