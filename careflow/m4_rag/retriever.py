"""M4 — Retrieval with selectable variants (so each change can be measured).

    dense          Qdrant cosine search only                         (baseline)
    hybrid         dense + BM25 fused with Reciprocal Rank Fusion
    hybrid_rerank  hybrid candidates re-ordered by a cross-encoder    (shipped default)

The reranker's order is the order passed to the LLM.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from careflow import llm
from careflow.config import settings

log = logging.getLogger(__name__)
Variant = Literal["dense", "hybrid", "hybrid_rerank"]
_TOKEN = re.compile(r"[A-Za-z0-9\-\$%\.]+")


@dataclass
class Passage:
    chunk_id: str          # citation id, e.g. "preauthorization_policy#1"
    source: str
    title: str
    section: str
    text: str
    score: float           # score that decided the final order
    rerank_score: float | None = None
    flags: list[str] = field(default_factory=list)


def _tok(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class HybridRetriever:
    def __init__(self) -> None:
        self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self._bm25: BM25Okapi | None = None
        self._payloads: list[dict] = []
        self._reranker = None
        self.last_latency_ms: dict[str, float] = {}

    # BM25 index is built from the same Qdrant collection (index == corpus)
    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        pts, offset = [], None
        while True:
            batch, offset = self._client.scroll(settings.qdrant_collection, limit=256, offset=offset,
                                                with_payload=True, with_vectors=False)
            pts += batch
            if offset is None:
                break
        self._payloads = [p.payload for p in pts]
        self._bm25 = BM25Okapi([_tok(p["text"]) for p in self._payloads])
        log.info("BM25 index over %d chunks", len(self._payloads))

    def _ensure_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(settings.rag_rerank_model)
        return self._reranker

    @staticmethod
    def _passage(payload: dict, score: float) -> Passage:
        return Passage(payload["chunk_id"], payload["source"], payload["title"], payload["section"],
                       payload["text"], score)

    async def _dense(self, query: str, n: int) -> list[Passage]:
        [qv] = await llm.aembed([query])
        res = self._client.query_points(settings.qdrant_collection, query=qv, limit=n, with_payload=True)
        return [self._passage(p.payload, p.score) for p in res.points]

    def _sparse(self, query: str, n: int) -> list[Passage]:
        self._ensure_bm25()
        scores = self._bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [self._passage(self._payloads[i], float(scores[i])) for i in order if scores[i] > 0]

    @staticmethod
    def _rrf(lists: list[list[Passage]]) -> list[Passage]:
        fused: dict[str, float] = {}
        keep: dict[str, Passage] = {}
        for lst in lists:
            for rank, p in enumerate(lst):
                fused[p.chunk_id] = fused.get(p.chunk_id, 0.0) + 1.0 / (settings.rag_rrf_k + rank + 1)
                keep[p.chunk_id] = p
        out = []
        for cid in sorted(fused, key=lambda c: -fused[c]):
            keep[cid].score = fused[cid]
            out.append(keep[cid])
        return out

    async def retrieve(self, query: str, k: int | None = None, variant: Variant | None = None) -> list[Passage]:
        k = k or settings.rag_k_final
        variant = variant or settings.rag_variant
        n = settings.rag_k_candidates
        t0 = time.perf_counter()
        dense = await self._dense(query, n)
        if variant == "dense":
            self.last_latency_ms = {"total": (time.perf_counter() - t0) * 1000}
            return dense[:k]
        fused = self._rrf([dense, self._sparse(query, n)])
        if variant == "hybrid":
            self.last_latency_ms = {"total": (time.perf_counter() - t0) * 1000}
            return fused[:k]
        cands = fused[:n]
        scores = self._ensure_reranker().predict([(query, p.text) for p in cands])
        for p, s in zip(cands, scores):
            p.rerank_score = float(s)
            p.score = float(s)
        cands.sort(key=lambda p: -p.score)  # reranker order is the final order
        self.last_latency_ms = {"total": (time.perf_counter() - t0) * 1000}
        return cands[:k]
