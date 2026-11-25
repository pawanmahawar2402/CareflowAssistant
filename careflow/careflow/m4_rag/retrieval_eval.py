"""M4 — Retrieval evaluation on the golden set: baseline vs. each retrieval variant.

    python -m careflow.m4_rag.retrieval_eval

Metrics per variant (answerable questions only): precision@k, recall@k, MRR over the top-k list,
and mean retrieval latency. Unanswerable questions are scored on whether the top passage falls
below the rerank refusal threshold (only meaningful for `hybrid_rerank`). Results are written to
`eval_results/retrieval_results.json` so the comparison is committed, not just printed.
"""
from __future__ import annotations

import asyncio
import json
import statistics
from datetime import datetime, timezone

from careflow.config import PROJECT_ROOT, settings
from careflow.m4_rag.retriever import HybridRetriever

VARIANTS = ["dense", "hybrid", "hybrid_rerank"]
K = settings.rag_k_final


def _load_golden() -> list[dict]:
    path = settings.eval_dir / "retrieval_golden.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _score(passages, expected: set[str]) -> dict[str, float]:
    sources = [p.source for p in passages[:K]]
    hits = [s in expected for s in sources]
    recall = len({s for s in sources if s in expected}) / len(expected)
    precision = sum(hits) / K
    rr = next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    return {"precision": precision, "recall": recall, "mrr": rr}


async def main() -> None:
    golden = _load_golden()
    retriever = HybridRetriever()
    per_question: dict[str, dict] = {}
    summary: dict[str, dict] = {}

    for variant in VARIANTS:
        scores, latencies = [], []
        for row in golden:
            passages = await retriever.retrieve(row["question"], k=K, variant=variant)
            latencies.append(retriever.last_latency_ms["total"])
            if row["kind"] == "unanswerable":
                top = passages[0].rerank_score if passages and passages[0].rerank_score is not None else None
                per_question.setdefault(row["id"], {})[variant] = {
                    "top_rerank_score": top, "refused": top is not None and top < settings.rag_min_rerank_score}
                continue
            s = _score(passages, set(row["expected_sources"]))
            top = passages[0].rerank_score if passages and passages[0].rerank_score is not None else None
            per_question.setdefault(row["id"], {})[variant] = {
                **s, "top_sources": [p.source for p in passages[:K]], "top_rerank_score": top}
            scores.append(s)
        summary[variant] = {
            "precision_at_k": round(statistics.mean(s["precision"] for s in scores), 4),
            "recall_at_k": round(statistics.mean(s["recall"] for s in scores), 4),
            "mrr": round(statistics.mean(s["mrr"] for s in scores), 4),
            "mean_latency_ms": round(statistics.mean(latencies), 1),
            "answerable_questions": len(scores),
        }

    refused = [per_question[r["id"]]["hybrid_rerank"]["refused"]
               for r in golden if r["kind"] == "unanswerable"]
    summary["unanswerable_refused_by_hybrid_rerank"] = f"{sum(refused)}/{len(refused)}"
    answerable_scores = [per_question[r["id"]]["hybrid_rerank"]["top_rerank_score"]
                         for r in golden if r["kind"] != "unanswerable"]
    wrongly_refused = [r["id"] for r in golden if r["kind"] != "unanswerable"
                       and per_question[r["id"]]["hybrid_rerank"]["top_rerank_score"] < settings.rag_min_rerank_score]
    summary["answerable_top_rerank_score_min"] = round(min(answerable_scores), 3)
    summary["answerable_wrongly_refused_by_hybrid_rerank"] = wrongly_refused

    out = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "k": K,
        "baseline": "dense",
        "embed_model": settings.careflow_embed_model,
        "rerank_model": settings.rag_rerank_model,
        "rerank_refusal_threshold": settings.rag_min_rerank_score,
        "golden_questions": len(golden),
        "summary": summary,
        "per_question": per_question,
    }
    path = PROJECT_ROOT / "eval_results" / "retrieval_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'variant':16}{'P@'+str(K):>8}{'R@'+str(K):>8}{'MRR':>8}{'ms':>10}")
    for v in VARIANTS:
        s = summary[v]
        print(f"{v:16}{s['precision_at_k']:>8.3f}{s['recall_at_k']:>8.3f}{s['mrr']:>8.3f}{s['mean_latency_ms']:>10.1f}")
    print(summary["unanswerable_refused_by_hybrid_rerank"], "unanswerable refused by hybrid_rerank")
    print("lowest answerable top rerank score:", summary["answerable_top_rerank_score_min"],
          "| answerable wrongly refused:", summary["answerable_wrongly_refused_by_hybrid_rerank"])
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
