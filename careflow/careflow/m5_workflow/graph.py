"""M5/M6 — CareFlow workflow: a deterministic supervisor over the M6 specialist team.

    guard_input ─┬─► refuse ─────────────────────────────────────────────┐
                 ├─► human_review (HIL interrupt) ─► finalize ──────────┤
                 └─► recall (M3) ─► intake ─┬─► refuse                  │
                                            ├─► human_review            │
                                            ├─► policy_agent ─┐         │
                                            ├─► referral_agent┼─► safety_reviewer
                                            └─► history_agent ┘     │   │
                                 ┌───────────────── revise ◄── llm rejection (capped)
                                 │                                  └─► review_failed ─► refuse
                                 └─► finalize ─► remember (M3) ─► END

Routing is a pure function of state: no LLM chooses the next node. Loops are capped by
`review_loops` and `settings.max_review_loops`; the run is also bounded by the recursion limit.
The graph is compiled with a durable checkpointer and `interrupt_before=["human_review"]`,
so emergent cases pause until staff call `resume()` with a decision.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from careflow.config import settings
from careflow.m2_agent import tools
from careflow.m3_memory.memory import memory
from careflow.m5_workflow.state import CareFlowState
from careflow.m6_agents import team
from careflow.m7_observability.resilience import trace_span
from careflow.m8_service.guardrails import EMERGENCY, REFUSAL, check_input

log = logging.getLogger(__name__)

UNAVAILABLE = "The assistant is temporarily unavailable; please contact the clinic."
_REF_RE = re.compile(r"\bREF-[A-Z0-9]{4,8}\b", re.I)
_HISTORY_RE = re.compile(r"\b(last time|last visit|what did (we|i)|previous (visit|request|appointment))\b", re.I)


def _ticket(state: CareFlowState, node: str, reason: str, priority: str,
            patient_id: str | None, approved_by: str | None = None) -> dict[str, Any]:
    """Staff ticket keyed by session + node + message, so a retried node cannot duplicate it."""
    digest = hashlib.sha1(str(state.get("raw_message", "")).encode()).hexdigest()[:12]
    key = f"{state.get('session_id', 'no-session')}:{node}:{digest}"
    return tools.create_escalation_ticket(reason, key, patient_id, priority,
                                          source_agent=node, approved_by=approved_by)


# ── Guard, memory, intake ────────────────────────────────────────────────────

async def guard_input(state: CareFlowState) -> dict[str, Any]:
    res = check_input(state["raw_message"])
    if res.allowed:
        return {"route": "auto"}
    route = "human_review" if res.category == "emergent" else "refuse"
    return {"route": route, "response": res.message, "refusal_reason": f"Input guardrail: {res.category}",
            "guardrail_flags": list(res.flags)}


async def recall(state: CareFlowState) -> dict[str, Any]:
    """Recall working, semantic, episodic and procedural memory for this request (M3)."""
    try:
        ctx = await memory.recall(state.get("user_id"), state["session_id"], state["raw_message"])
    except Exception as exc:  # memory is best-effort; a Redis outage must not stop care
        log.warning("memory recall unavailable: %s", type(exc).__name__)
        ctx = {}
    return {"memory": ctx}


async def intake(state: CareFlowState) -> dict[str, Any]:
    with trace_span("intake"):
        out = await team.intake_agent({"raw_message": state["raw_message"],
                                       "channel": state.get("channel", "unknown")})
    if "patient_request" not in out:  # LLM unavailable: fail closed
        return {**out, "route": "refuse", "response": UNAVAILABLE, "refusal_reason": "Intake unavailable"}
    pr = out["patient_request"]
    if pr["clinical_advice_sought"]:
        return {**out, "route": "refuse", "response": REFUSAL, "refusal_reason": "Clinical advice sought",
                "guardrail_flags": ["clinical_advice_request"]}
    if pr["urgency"] == "emergent":
        return {**out, "route": "human_review", "response": EMERGENCY}
    return {**out, "route": "auto"}


# ── Specialists (wrapped team agents from m6_agents.team) ────────────────────

async def policy_agent(state: CareFlowState) -> dict[str, Any]:
    return await team.policy_agent(dict(state))


async def referral_agent(state: CareFlowState) -> dict[str, Any]:
    return await team.referral_agent(dict(state))


async def history_agent(state: CareFlowState) -> dict[str, Any]:
    return await team.history_agent(dict(state))


async def safety_reviewer(state: CareFlowState) -> dict[str, Any]:
    return await team.safety_reviewer(dict(state))


async def revise(state: CareFlowState) -> dict[str, Any]:
    """One bounded revision: the drafter gets the reviewer's feedback on its next pass."""
    return {"review_loops": (state.get("review_loops") or 0) + 1}


async def review_failed(state: CareFlowState) -> dict[str, Any]:
    """The draft was rejected and no revision budget is left: never release it."""
    return {"response": REFUSAL, "route": "refuse", "refusal_reason": "Draft failed safety review",
            "guardrail_flags": ["output_safety_review"]}


# ── Terminal nodes ───────────────────────────────────────────────────────────

async def refuse(state: CareFlowState) -> dict[str, Any]:
    req = state.get("patient_request") or {}
    tk = _ticket(state, "refuse", state.get("refusal_reason") or "Refused by guardrail", "normal",
                 req.get("patient_id"))
    return {"escalation_ticket": tk, "response": state.get("response") or REFUSAL}


async def human_review(state: CareFlowState) -> dict[str, Any]:
    """Side effect (staff ticket) — runs only after resume(), i.e. after the interrupt."""
    req = state.get("patient_request") or {}
    approved = state.get("hil_approved")
    note = state.get("hil_note") or ""
    tk = _ticket(state, "human_review", note or "Emergent / escalated case", "high",
                 req.get("patient_id"), approved_by="staff" if approved is not None else None)
    if approved is False:
        resp = f"A staff member reviewed your request and will contact you directly. {note}".strip()
    else:
        resp = state.get("response") or EMERGENCY
    return {"escalation_ticket": tk, "response": resp}


async def finalize(state: CareFlowState) -> dict[str, Any]:
    """Release an approved draft. Policy questions the corpus cannot answer also reach staff."""
    out: dict[str, Any] = {"response": state.get("draft_response") or state.get("response", "")}
    if state.get("draft_kind") == "not_in_corpus":
        req = state.get("patient_request") or {}
        out["escalation_ticket"] = _ticket(state, "finalize", "Policy question not in corpus", "normal",
                                           req.get("patient_id"))
    return out


async def remember(state: CareFlowState) -> dict[str, Any]:
    """Write memory for this turn (M3): working turns always; episodic and semantic for patients."""
    sid, uid = state["session_id"], state.get("user_id")
    try:
        await memory.working.add(sid, "user", state["raw_message"])
        await memory.working.add(sid, "assistant", state.get("response", ""))
        if uid:
            await memory.episodic.record(
                uid, sid, "chat", summary=(state.get("response") or "")[:160],
                details={"route": state.get("route"), "last_agent": state.get("last_agent")})
            if state.get("route") == "auto":  # never extract facts from refused or injected text
                await memory.semantic.extract_and_store(uid, sid, state["raw_message"])
    except Exception as exc:  # best-effort, as in recall
        log.warning("memory write unavailable: %s", type(exc).__name__)
    return {}


# ── Routing (pure functions of state) ────────────────────────────────────────

def _after_guard(state: CareFlowState) -> str:
    return {"auto": "recall", "refuse": "refuse", "human_review": "human_review"}[state["route"]]


def _specialist(state: CareFlowState) -> str:
    msg = state["raw_message"]
    if _HISTORY_RE.search(msg):  # answered from this patient's episodes, or "no record yet"
        return "history_agent"
    pr = state.get("patient_request") or {}
    if pr.get("intent") in ("scheduling", "referral_status") or _REF_RE.search(msg):
        return "referral_agent"
    return "policy_agent"


def _after_intake(state: CareFlowState) -> str:
    route = state["route"]
    return _specialist(state) if route == "auto" else route  # refuse | human_review


def _after_review(state: CareFlowState) -> str:
    rv = state.get("review") or {}
    if rv.get("approved"):
        return "finalize"
    if rv.get("by") == "llm" and (state.get("review_loops") or 0) < settings.max_review_loops:
        return "revise"
    return "review_failed"


def _after_revise(state: CareFlowState) -> str:
    return state.get("last_agent") or "policy_agent"


# ── Graph ────────────────────────────────────────────────────────────────────

def build_graph(checkpointer: Any | None = None):
    g = StateGraph(CareFlowState)
    for name, fn in [
        ("guard_input", guard_input), ("recall", recall), ("intake", intake),
        ("policy_agent", policy_agent), ("referral_agent", referral_agent), ("history_agent", history_agent),
        ("safety_reviewer", safety_reviewer), ("revise", revise), ("review_failed", review_failed),
        ("refuse", refuse), ("human_review", human_review), ("finalize", finalize), ("remember", remember),
    ]:
        g.add_node(name, fn)
    g.set_entry_point("guard_input")
    g.add_conditional_edges("guard_input", _after_guard,
                            {"recall": "recall", "refuse": "refuse", "human_review": "human_review"})
    g.add_edge("recall", "intake")
    g.add_conditional_edges("intake", _after_intake, {
        "policy_agent": "policy_agent", "referral_agent": "referral_agent", "history_agent": "history_agent",
        "refuse": "refuse", "human_review": "human_review"})
    for drafter in ("policy_agent", "referral_agent", "history_agent"):
        g.add_edge(drafter, "safety_reviewer")
    g.add_conditional_edges("safety_reviewer", _after_review,
                            {"finalize": "finalize", "revise": "revise", "review_failed": "review_failed"})
    g.add_conditional_edges("revise", _after_revise,
                            {"policy_agent": "policy_agent", "referral_agent": "referral_agent",
                             "history_agent": "history_agent"})
    g.add_edge("review_failed", "refuse")
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", "remember")
    g.add_edge("refuse", "remember")
    g.add_edge("remember", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=["human_review"])


_graph: Any = None


async def get_graph():
    """Compiled graph with a durable SQLite checkpointer (M5.C4).

    Paused runs live in `settings.checkpoint_db`, so a staff resume still works after the
    process restarts. The connection is opened lazily because it must be created inside
    the running event loop.
    """
    global _graph
    if _graph is None:
        conn = await aiosqlite.connect(settings.runtime_path(settings.checkpoint_db))
        _graph = build_graph(AsyncSqliteSaver(conn))
    return _graph


async def close_graph() -> None:
    """Close the checkpoint connection (its worker thread would otherwise keep the process alive)."""
    global _graph
    if _graph is not None:
        await _graph.checkpointer.conn.close()
        _graph = None


def _cfg(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id}, "recursion_limit": settings.graph_recursion_limit}


async def run(message: str, session_id: str, channel: str = "unknown",
              user_id: str | None = None) -> dict[str, Any]:
    """Run a message; returns state plus `pending_human` when paused for review.

    `user_id` scopes episodic and semantic memory to one patient (no `user_id` = no patient memory).
    """
    graph = await get_graph()
    cfg = _cfg(session_id)
    state = await graph.ainvoke(
        {"raw_message": message, "channel": channel, "session_id": session_id, "user_id": user_id}, cfg
    )
    snap = await graph.aget_state(cfg)
    state = dict(state)
    state["pending_human"] = bool(snap.next)
    return state


async def resume(session_id: str, approved: bool, note: str = "") -> dict[str, Any]:
    """Staff decision on a paused case."""
    graph = await get_graph()
    cfg = _cfg(session_id)
    await graph.aupdate_state(cfg, {"hil_approved": approved, "hil_note": note})
    state = dict(await graph.ainvoke(None, cfg))
    state["pending_human"] = False
    return state
