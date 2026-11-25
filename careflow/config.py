"""Global settings — one place for models, thresholds, paths and timeouts (H2/H3/H4).

Everything is overridable through environment variables or `.env`.
Paths are derived from this file, never from the working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # ── LLM (provider-agnostic via LiteLLM; change the string to switch provider)
    gemini_api_key: str = ""
    openai_api_key: str = ""
    careflow_llm_model: str = "gemini/gemini-3.5-flash-lite"
    careflow_fallback_llm_model: str = "gemini/gemini-3.5-flash"  # secondary model; empty = no fallback
    careflow_embed_model: str = "gemini/gemini-embedding-001"
    embed_dim: int = 768
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    llm_backoff_base_s: float = 1.0
    llm_json_repair_attempts: int = 2

    # ── Vector store / memory stores
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""  # required by Qdrant Cloud; leave empty for a local or private Qdrant
    qdrant_collection: str = "careflow_corpus"
    redis_url: str = "redis://localhost:6379/0"
    checkpoint_db: str = "runtime/checkpoints.sqlite"
    ticket_db: str = "runtime/tickets.sqlite"

    # ── Observability
    langfuse_public_key: str = ""   # set in .env (never in code)
    langfuse_secret_key: str = ""   # set in .env (never in code)
    langfuse_host: str = "https://cloud.langfuse.com"
    kpi_log: str = "runtime/kpi.jsonl"

    # ── Data
    careflow_data_dir: str = "data"

    # ── Circuit breaker
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 30

    # ── Memory budgets
    working_memory_max_turns: int = 12
    working_memory_summary_keep: int = 6
    memory_recall_max_items: int = 5

    # ── RAG
    rag_k_final: int = 4
    rag_k_candidates: int = 12
    rag_rrf_k: int = 60
    rag_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # dense | hybrid | hybrid_rerank. Default is `hybrid` so the service fits in a 512 MB free instance:
    # the cross-encoder reranker (torch) is only loaded for hybrid_rerank. Set RAG_VARIANT=hybrid_rerank to restore it.
    rag_variant: str = "hybrid"
    # below this the corpus is treated as not answering. Tuned on the retrieval golden set:
    # answerable top scores were all >= +5.3; unanswerable ones were -3.7 and -10.1. 0.0 sits in the gap.
    rag_min_rerank_score: float = 0.0
    rag_max_attempts: int = 2  # answer generations allowed when the citation check fails

    # ── Guardrail thresholds (tuned on the golden set; defined once here)
    injection_block_score: float = 0.5
    groundedness_min_score: float = 0.7
    max_message_chars: int = 4000

    # ── Workflow bounds
    max_agent_iterations: int = 6
    max_team_hops: int = 8
    max_review_loops: int = 1
    graph_recursion_limit: int = 30

    # ── Failure injection (M7.C6) — off by default
    fault_rate: float = 0.0
    fault_seed: int = 7

    # ── Resolved paths
    @property
    def data_dir(self) -> Path:
        p = Path(self.careflow_data_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def tables_dir(self) -> Path:
        return self.data_dir / "tables"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    def runtime_path(self, rel: str) -> Path:
        p = Path(rel)
        p = p if p.is_absolute() else PROJECT_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()

# LiteLLM / LangFuse read credentials from os.environ.
for _k, _v in (
    ("GEMINI_API_KEY", settings.gemini_api_key),
    ("GOOGLE_API_KEY", settings.gemini_api_key),
    ("OPENAI_API_KEY", settings.openai_api_key),
    ("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key),
    ("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key),
    ("LANGFUSE_HOST", settings.langfuse_host),
):
    if _v and not os.environ.get(_k):
        os.environ[_k] = _v
