from __future__ import annotations

import re
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .schemas import KnowledgeDoc, RetrievalHit


def tokenize_for_bm25(text: str) -> list[str]:
    cleaned = text.strip().lower()
    if not cleaned:
        return []
    zh_tokens = jieba.lcut(cleaned, cut_all=False)
    en_tokens = re.findall(r"[a-z0-9]+", cleaned)
    tokens = [token.strip() for token in zh_tokens + en_tokens if token.strip()]
    return tokens or [cleaned]


class StageEmbedder:
    def __init__(self, model_dir: str | Path) -> None:
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Embedding model path not found: {model_path}. "
                "Please put your embedding model under ./embedding_models first."
            )
        self.model = SentenceTransformer(str(model_path), local_files_only=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class TimeAwareRetriever:
    def __init__(self, docs: list[KnowledgeDoc], embedding_model_dir: str | Path) -> None:
        if not docs:
            raise ValueError("Knowledge docs are empty.")
        self.docs = docs
        self.corpus_tokens = [tokenize_for_bm25(doc.disease) for doc in docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.embedder = StageEmbedder(embedding_model_dir)
        self.stage_embeddings = self.embedder.encode([doc.stage for doc in docs])

    def retrieve(
        self,
        query_disease: str,
        query_time: str,
        bm25_top_k: int = 100,
        final_top_k: int = 10,
    ) -> list[RetrievalHit]:
        tokenized_query = tokenize_for_bm25(query_disease)
        bm25_scores = (
            self.bm25.get_scores(tokenized_query)
            if tokenized_query
            else np.zeros(len(self.docs), dtype=float)
        )

        candidate_size = min(max(1, bm25_top_k), len(self.docs))
        candidate_indices = np.argsort(bm25_scores)[::-1][:candidate_size]

        has_time = bool(query_time.strip())
        if has_time:
            query_time_embedding = self.embedder.encode([query_time.strip()])[0]
            candidate_stage_scores = np.dot(self.stage_embeddings[candidate_indices], query_time_embedding)
            rerank_order = np.argsort(candidate_stage_scores)[::-1]
        else:
            query_time_embedding = None
            rerank_order = np.argsort(bm25_scores[candidate_indices])[::-1]

        output_size = min(max(1, final_top_k), len(candidate_indices))
        selected_indices = candidate_indices[rerank_order][:output_size]

        hits: list[RetrievalHit] = []
        for rank, doc_index in enumerate(selected_indices, start=1):
            if query_time_embedding is not None:
                stage_score = float(np.dot(self.stage_embeddings[doc_index], query_time_embedding))
            else:
                stage_score = 0.0
            hits.append(
                RetrievalHit(
                    rank=rank,
                    doc=self.docs[doc_index],
                    bm25_score=float(bm25_scores[doc_index]),
                    stage_score=stage_score,
                )
            )
        return hits
