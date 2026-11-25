"""M4 — Ingest the policy corpus into Qdrant.   python -m careflow.m4_rag.ingest"""
from __future__ import annotations

import asyncio
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from careflow import llm
from careflow.config import settings
from careflow.m4_rag.chunking import chunk_document, iter_corpus

log = logging.getLogger(__name__)
_NS = uuid.UUID("6f1d4a52-3c1e-4f0a-9d6b-0c3a5e7b9a11")


async def ingest() -> int:
    files = iter_corpus()
    if not files:
        raise FileNotFoundError(f"No corpus files in {settings.corpus_dir}. Run: python generate_data.py")
    client = QdrantClient(url=settings.qdrant_url)
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)  # full rebuild keeps index == corpus
    client.create_collection(
        settings.qdrant_collection,
        vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE))
    total = 0
    for path in files:
        chunks = chunk_document(path)
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), 32):
            vectors += await llm.aembed([c.text for c in chunks[i:i + 32]])
        client.upsert(settings.qdrant_collection, points=[
            PointStruct(
                id=str(uuid.uuid5(_NS, c.chunk_id)), vector=v,
                payload={"chunk_id": c.chunk_id, "source": c.source, "title": c.title,
                         "section": c.section, "chunk_index": c.index, "text": c.text,
                         "filename": path.name})
            for c, v in zip(chunks, vectors)])
        total += len(chunks)
        log.info("%s -> %d chunks", path.name, len(chunks))
    log.info("ingested %d chunks from %d documents", total, len(files))
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(ingest())
