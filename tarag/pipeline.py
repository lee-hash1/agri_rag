from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import load_clean_docs
from .local_llm import LocalLLM
from .retriever import TimeAwareRetriever
from .schemas import KnowledgeDoc, PipelineResult, RetrievalHit


class QueryParser:
    SYSTEM_PROMPT = (
        "你是问题解析器。请从用户农业问题中提取病虫害名称和时间信息。"
        "只输出 JSON 对象，格式为："
        '{"disease":"", "time":""}。'
        "如果某项缺失，返回空字符串。"
    )

    def __init__(self, llm: LocalLLM) -> None:
        self.llm = llm

    def parse(self, question: str) -> dict[str, str]:
        parsed = self.llm.ask_json(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=f"问题：{question}",
            fallback={"disease": "", "time": ""},
            max_new_tokens=256,
        )
        if not isinstance(parsed, dict):
            parsed = {"disease": "", "time": ""}

        disease = str(parsed.get("disease", "")).strip()
        time_info = str(parsed.get("time", "")).strip()
        return {"disease": disease, "time": time_info}


class AnswerGenerator:
    SYSTEM_PROMPT = (
        "你是农业病虫害防治助手。"
        "你必须只根据给定知识库片段作答，不允许捏造。"
        "回答尽量简洁，优先给出与时间阶段匹配的处理建议。"
    )

    def __init__(self, llm: LocalLLM) -> None:
        self.llm = llm

    @staticmethod
    def _format_context(docs: list[KnowledgeDoc]) -> str:
        lines = []
        for i, doc in enumerate(docs, start=1):
            lines.append(
                f"{i}. disease={doc.disease}; stage={doc.stage}; treatment={doc.treatment}"
            )
        return "\n".join(lines)

    def generate(
        self,
        question: str,
        parsed_query: dict[str, str],
        hits: list[RetrievalHit],
        generation_top_k: int = 5,
    ) -> str:
        if not hits:
            return "知识库中未检索到相关信息，暂时无法给出可靠建议。"

        docs = [hit.doc for hit in hits[:generation_top_k]]
        context = self._format_context(docs)
        user_prompt = (
            f"用户问题：{question}\n"
            f"解析结果：{json.dumps(parsed_query, ensure_ascii=False)}\n\n"
            f"检索文档：\n{context}\n\n"
            "请给出最终建议。如果时间与病害信息不足，请明确说明不确定性。"
        )
        return self.llm.chat(self.SYSTEM_PROMPT, user_prompt, max_new_tokens=512, temperature=0.2)


class AnswerVerifier:
    SYSTEM_PROMPT = (
        "你是回答校验器。请判断回答是否被检索文档支持。"
        "只输出 JSON："
        '{"is_correct": true, "reason": "...", "unsupported_claims": []}。'
    )

    def __init__(self, llm: LocalLLM) -> None:
        self.llm = llm

    @staticmethod
    def _format_context(docs: list[KnowledgeDoc]) -> str:
        lines = []
        for i, doc in enumerate(docs, start=1):
            lines.append(
                f"{i}. disease={doc.disease}; stage={doc.stage}; treatment={doc.treatment}"
            )
        return "\n".join(lines)

    def verify(self, question: str, answer: str, hits: list[RetrievalHit]) -> dict[str, Any]:
        docs = [hit.doc for hit in hits]
        context = self._format_context(docs)
        user_prompt = (
            f"问题：{question}\n"
            f"回答：{answer}\n\n"
            f"证据文档：\n{context}\n\n"
            "请判断回答是否由证据支持。"
        )
        verdict = self.llm.ask_json(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback={
                "is_correct": False,
                "reason": "模型未返回可解析的校验结果。",
                "unsupported_claims": [],
            },
            max_new_tokens=256,
        )
        if not isinstance(verdict, dict):
            return {
                "is_correct": False,
                "reason": "模型未返回可解析的校验结果。",
                "unsupported_claims": [],
            }

        return {
            "is_correct": bool(verdict.get("is_correct", False)),
            "reason": str(verdict.get("reason", "")).strip(),
            "unsupported_claims": verdict.get("unsupported_claims", []),
        }


class TARAGPipeline:
    def __init__(
        self,
        kb_path: str | Path,
        model_dir: str | Path = "./models",
        embedding_model_dir: str | Path = "./embedding_models",
        bm25_top_k: int = 100,
        rerank_top_k: int = 10,
        generation_top_k: int = 5,
    ) -> None:
        self.docs = load_clean_docs(kb_path)
        self.llm = LocalLLM(model_dir=model_dir)
        self.retriever = TimeAwareRetriever(self.docs, embedding_model_dir=embedding_model_dir)
        self.query_parser = QueryParser(self.llm)
        self.answer_generator = AnswerGenerator(self.llm)
        self.answer_verifier = AnswerVerifier(self.llm)
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k
        self.generation_top_k = generation_top_k

    def run(self, question: str) -> PipelineResult:
        parsed_query = self.query_parser.parse(question)

        disease_query = parsed_query["disease"] or question
        time_query = parsed_query["time"]

        hits = self.retriever.retrieve(
            query_disease=disease_query,
            query_time=time_query,
            bm25_top_k=self.bm25_top_k,
            final_top_k=self.rerank_top_k,
        )

        answer = self.answer_generator.generate(
            question=question,
            parsed_query=parsed_query,
            hits=hits,
            generation_top_k=self.generation_top_k,
        )
        verification = self.answer_verifier.verify(question=question, answer=answer, hits=hits)

        return PipelineResult(
            question=question,
            parsed_query=parsed_query,
            retrieved=hits,
            answer=answer,
            verification=verification,
        )
