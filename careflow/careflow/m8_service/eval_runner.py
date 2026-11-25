"""M8 — Golden-set evaluation. Run: python -m careflow.m8_service.eval_runner"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone

from careflow.config import PROJECT_ROOT, settings
from careflow.m5_workflow import graph as wf


# golden expected_outcome -> the graph outcome it must produce
_ESCALATED = {"refuse", "human_review"}


def _route_ok(expected: str, st: dict) -> bool:
    if expected == "escalated":
        return st.get("route") in _ESCALATED or bool(st.get("pending_human"))
    return st.get("route") == "auto"


async def main() -> None:
    rows = [json.loads(l) for l in (settings.eval_dir / "e2e_golden.jsonl").read_text().splitlines() if l]
    wanted = {i for i in os.environ.get("EVAL_IDS", "").split(",") if i}  # e.g. EVAL_IDS=e04,e11 to rerun cases
    if wanted:
        rows = [r for r in rows if r["id"] in wanted]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results, passed = [], 0
    for r in rows:
        # a unique thread per run: reusing a thread would carry state from an earlier run
        st = await wf.run(r["message"], f"eval-{run_id}-{r['id']}")
        text = st.get("response", "")
        route_ok = _route_ok(r["expected_outcome"], st)
        contains_ok = all(s.lower() in text.lower() for s in r["expected_contains"])
        ok = route_ok and contains_ok
        passed += ok
        results.append({"id": r["id"], "category": r["category"], "adversarial": r.get("adversarial", False),
                        "expected_outcome": r["expected_outcome"], "route": st.get("route"),
                        "route_ok": route_ok, "contains_ok": contains_ok, "pass": ok,
                        "response": text, "review_by": (st.get("review") or {}).get("by"),
                        "policy_flags": (st.get("policy_result") or {}).get("flags"),
                        "policy_citations": (st.get("policy_result") or {}).get("citations")})
        print(f"{'PASS' if ok else 'FAIL'} {r['id']} route={st.get('route')} :: {text[:80]!r}")
    total = len(rows)
    print(f"\n{passed}/{total} passed ({passed / total:.0%})")
    out = {"run_id": run_id, "passed": passed, "total": total, "results": results}
    path = PROJECT_ROOT / "eval_results" / f"e2e_results_{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
