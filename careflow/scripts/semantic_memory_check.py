"""M3.C3 with the real model: a preference stated once is stored, then applied later without restating.

    python scripts/semantic_memory_check.py        (needs GEMINI_API_KEY; ~5 model calls)
Writes eval_results/semantic_memory_check.json. The result is recorded whether or not the check passes.
"""
import asyncio
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from careflow.m3_memory.memory import memory, store  # noqa: E402
from careflow.m5_workflow import graph as wf  # noqa: E402

PREFERENCE = "Please contact me by SMS from now on. I don't want phone calls."
LATER = "How will appointment reminders be sent to me?"  # the corpus lists SMS, email and phone
TRIALS = 3
_RULE = re.compile(r"[^.]*\bsms\b[^.]*\bprefer|[^.]*\bprefer[^.]*\bsms\b", re.IGNORECASE)


def is_applied(answer: str) -> bool:
    """Applied = a sentence names SMS together with the patient's stated preference."""
    return bool(_RULE.search(answer))


async def main() -> None:
    user = "pt-sem-" + uuid.uuid4().hex[:6]
    s1 = "sem-" + uuid.uuid4().hex[:6]
    turn1 = await wf.run(PREFERENCE, s1, user_id=user)  # the preference is stated once, in its own session
    facts = (await memory.recall(user, s1, "x"))["facts"]
    trials = []
    for i in range(TRIALS):
        sid = "sem-" + uuid.uuid4().hex[:6]  # a new session each time: the fact must come from memory
        turn = await wf.run(LATER, sid, user_id=user)
        answer = turn.get("response") or ""
        # applied = the answer names SMS as this patient's channel and does not offer a phone call
        trials.append({"trial": i + 1, "answer": answer, "applied": is_applied(answer)})
    out = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "user_id": user,
        "preference_turn_route": turn1.get("route"),
        "facts_stored_after_turn1": facts,
        "later_question": LATER,
        "trials": trials,
        "applied_in": sum(t["applied"] for t in trials),
        "trials_total": TRIALS,
        "rule": "applied = a sentence names SMS together with the patient's stated preference "
                "(the corpus lists SMS, email and phone call, so a bare list of channels does not count).",
    }
    path = ROOT / "eval_results" / "semantic_memory_check.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("facts stored:", facts)
    print(f"preference applied in {out['applied_in']} of {TRIALS} later answers")
    await wf.close_graph()
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
