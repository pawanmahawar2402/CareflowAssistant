"""M6 — The specialist agents of the CareFlow team.

Roster: Intake · Insurance/Policy RAG · Referral Tracker · History (episodic recall) ·
Clinical Safety Reviewer. The supervisor that routes between them lives in
`careflow.m5_workflow.graph`; the routing there is deterministic (no LLM picks the next agent).

Boundaries (enforced, not just prompted):
  * each agent sees a scoped slice of state (`reads`) and may write only `writes`
  * each agent reaches the EHR only through its own ScopedMCP allow-list
Domain systems are reached exclusively over MCP.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from careflow import llm
from careflow.m1_intake.models import PatientRequest
from careflow.m1_intake.parser import parse_message
from careflow.m4_rag.rag_agent import InsurancePolicyAgent
from careflow.m6_agents.mcp_client import ScopedMCP
from careflow.m7_observability import tracing
from careflow.m8_service.guardrails import check_output

log = logging.getLogger(__name__)

AgentFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

# ── Per-agent tool scopes (the only tools each agent can reach) ──────────────
POLICY_MCP = ScopedMCP("policy_agent", {"check_eligibility", "calculate_copay"}, ("careflow://plans/",))
REFERRAL_MCP = ScopedMCP("referral_agent", {"get_referral_status", "lookup_patient", "book_appointment"},
                         ("careflow://policy/",))

_REF_RE = re.compile(r"\bREF-[A-Za-z0-9]{4,8}\b", re.I)
_rag: InsurancePolicyAgent | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def agent_node(name: str, reads: set[str], writes: set[str]):
    """Wrap an agent: scoped state in, scoped keys out, traced, audited."""

    def deco(fn: AgentFn):
        @wraps(fn)
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            scoped = {k: state[k] for k in reads if k in state}
            with tracing.span(f"agent:{name}", as_type="agent", agent=name):
                out = await fn(scoped)
            illegal = set(out) - writes
            if illegal:
                raise RuntimeError(f"agent '{name}' tried to write keys outside its scope: {sorted(illegal)}")
            out["audit"] = [{"agent": name, "ts": _now(), "event": out.get("draft_kind") or "ran"}]
            return out

        node.__agent__ = name  # type: ignore[attr-defined]
        return node

    return deco


# ── Intake Agent ─────────────────────────────────────────────────────────────

@agent_node("intake_agent", reads={"raw_message", "channel"}, writes={"patient_request", "intake_error", "audit"})
async def intake_agent(s: dict[str, Any]) -> dict[str, Any]:
    try:
        req = await parse_message(s["raw_message"], s.get("channel", "unknown"))
    except (llm.LLMUnavailable, ValueError) as exc:
        log.error("intake failed: %s", type(exc).__name__)
        return {"intake_error": type(exc).__name__}
    return {"patient_request": req.model_dump(mode="json")}


# ── Insurance / Policy RAG Agent ─────────────────────────────────────────────

def _patient_context(mem: dict[str, Any]) -> dict[str, str]:
    """Recalled semantic facts (preferences) plus the rolling conversation summary, as plain text."""
    ctx = {k: v for k, v in (mem.get("facts") or {}).items()}
    summary = (mem.get("working") or {}).get("summary")
    if summary:
        ctx["conversation_summary"] = summary
    return ctx


@agent_node(
    "policy_agent",
    reads={"patient_request", "raw_message", "memory", "review", "draft_version"},
    writes={"policy_result", "draft_response", "draft_version", "draft_kind", "last_agent", "audit"},
)
async def policy_agent(s: dict[str, Any]) -> dict[str, Any]:
    global _rag
    req = PatientRequest.model_validate(s["patient_request"])
    version = s.get("draft_version", 0) + 1
    base = {"draft_version": version, "last_agent": "policy_agent"}

    if req.intent == "eligibility_check":
        if not req.insurance_id:
            return {**base, "draft_kind": "answered", "policy_result": {"tool": "check_eligibility"},
                    "draft_response": "Please share your insurance member ID (for example MRD-123456) and I will check it."}
        res = await POLICY_MCP.call("check_eligibility", insurance_id=req.insurance_id)
        if not res.get("ok"):
            text = (f"I couldn't verify insurance ID {req.insurance_id}: "
                    f"{res.get('error', {}).get('message', 'lookup failed')}.")
        elif res["eligible"]:
            text = (f"Insurance ID {res['insurance_id']} is active on {res['plan_name']}: deductible "
                    f"${res['deductible_usd']:.0f}, coinsurance {res['coinsurance_pct'] * 100:.0f}%, "
                    f"out-of-pocket maximum ${res['out_of_pocket_max_usd']:.0f}.")
        else:
            text = f"Insurance ID {res['insurance_id']} is not currently active (status: {res['eligibility_status']})."
        return {**base, "draft_kind": "answered", "policy_result": {"tool": "check_eligibility", "ok": res.get("ok")},
                "draft_response": text}

    _rag = _rag or InsurancePolicyAgent()
    mem = s.get("memory") or {}
    procedure = mem.get("procedure")
    feedback = (s.get("review") or {}).get("feedback")
    question = s["raw_message"] + (f"\n(Reviewer feedback to address: {feedback})" if feedback else "")
    rag = await _rag.answer(question, procedure=procedure, context=_patient_context(mem) or None)
    return {**base, "draft_kind": "answered" if rag.answerable else "not_in_corpus",
            "policy_result": {"citations": rag.citations, "answerable": rag.answerable,
                              "flags": rag.flags, "used_fallback": rag.used_fallback,
                              "sources": [p.chunk_id for p in rag.passages]},
            "draft_response": rag.answer}


# ── Referral Tracking Agent ──────────────────────────────────────────────────

@agent_node(
    "referral_agent",
    reads={"patient_request", "raw_message", "draft_version"},
    writes={"referral_result", "draft_response", "draft_version", "draft_kind", "last_agent", "audit"},
)
async def referral_agent(s: dict[str, Any]) -> dict[str, Any]:
    req = PatientRequest.model_validate(s["patient_request"])
    version = s.get("draft_version", 0) + 1
    base = {"draft_version": version, "last_agent": "referral_agent", "draft_kind": "answered"}

    if req.intent == "scheduling":
        if not req.patient_id or not req.requested_specialty:
            return {**base, "referral_result": {"tool": "book_appointment", "ok": False},
                    "draft_response": "To book, please share your patient ID (like P0004) and the specialty you need."}
        res = await REFERRAL_MCP.call("book_appointment", patient_id=req.patient_id,
                                      specialty=req.requested_specialty.title(),
                                      urgency="urgent" if req.urgency == "urgent" else "routine")
        text = (f"Booked {res['specialty']} with {res['provider_name']} on {res['date']} at {res['time']} "
                f"({res['location']}). Appointment ID {res['appointment_id']}."
                if res.get("ok") else f"I couldn't book that: {res['error']['message']}.")
        return {**base, "referral_result": {"tool": "book_appointment", "ok": res.get("ok")}, "draft_response": text}

    m = _REF_RE.search(s["raw_message"])
    if not m:
        return {**base, "referral_result": {"tool": "get_referral_status", "ok": False},
                "draft_response": "Please share your referral ID (for example REF-0003) and I'll check its status."}
    ref_id = m.group(0).upper()
    res = await REFERRAL_MCP.call("get_referral_status", referral_id=ref_id)
    if not res.get("ok"):
        return {**base, "referral_result": {"tool": "get_referral_status", "ok": False},
                "draft_response": f"I couldn't find referral {ref_id}. Please double-check the ID or contact the front desk."}
    r = res["referral"]
    text = f"Referral {r['referral_id']} ({r['specialty']}) is currently {r['status']} (urgency: {r['urgency']})."
    if r["status"] == "denied":
        policy = await REFERRAL_MCP.read("careflow://policy/referral_policy")
        appeal = next((ln for ln in policy.splitlines() if "appeal" in ln.lower()), "")
        text += f" {appeal}".rstrip()
    return {**base, "referral_result": {"tool": "get_referral_status", "ok": True, "status": r["status"]},
            "draft_response": text}


# ── History Agent (episodic recall, M3.C2 / T4) ──────────────────────────────

@agent_node(
    "history_agent",
    reads={"memory"},
    writes={"draft_response", "draft_version", "draft_kind", "last_agent", "audit"},
)
async def history_agent(s: dict[str, Any]) -> dict[str, Any]:
    """Answers 'what did we do last time?' from the patient's own episodic memory only."""
    episodes = (s.get("memory") or {}).get("episodes") or []
    version = (s.get("draft_version") or 0) + 1
    base = {"draft_version": version, "last_agent": "history_agent", "draft_kind": "answered"}
    if not episodes:
        return {**base, "draft_response": "I don't have a record of an earlier request from you yet."}
    last = episodes[0]  # newest first
    when = str(last.get("ts", ""))[:10]
    summary = str(last.get("summary", "we handled a request")).rstrip(". ")
    return {**base, "draft_response": f"Last time ({when}), {summary}."}


# ── Clinical Safety Reviewer ─────────────────────────────────────────────────

class _Critic(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)


_CRITIC_SYSTEM = """\
You are the Clinical Safety Reviewer for a clinic's care-coordination assistant. Review the DRAFT reply
to the patient. Reject it if it contains diagnosis, symptom interpretation, medication or treatment advice,
a clinical opinion, or a policy figure not supported by the CITED SOURCES. Otherwise approve.
Return JSON {"approved": bool, "issues": [short strings]}."""


@agent_node(
    "safety_reviewer",
    reads={"draft_response", "draft_version", "policy_result"},
    writes={"review", "audit"},
)
async def safety_reviewer(s: dict[str, Any]) -> dict[str, Any]:
    draft = s["draft_response"]
    version = s.get("draft_version", 1)
    det = check_output(draft)  # deterministic first: cannot be talked out of it
    if not det.allowed:
        return {"review": {"approved": False, "issues": det.flags, "reviewed_version": version,
                           "feedback": "Remove any clinical advice or sensitive identifiers.", "by": "rules"}}
    if (s.get("policy_result") or {}).get("citations"):  # LLM critic only for generated (RAG) text
        try:
            verdict, _ = await llm.astructured(
                [{"role": "system", "content": _CRITIC_SYSTEM},
                 {"role": "user", "content": f"DRAFT:\n{draft}"}], _Critic, name="safety_critic")
        except (llm.LLMUnavailable, ValueError):
            # Fail closed: an unreviewed generated answer is never released.
            return {"review": {"approved": False, "issues": ["reviewer_unavailable"], "reviewed_version": version,
                               "feedback": "", "by": "fail_closed"}}
        return {"review": {"approved": verdict.approved, "issues": verdict.issues, "reviewed_version": version,
                           "feedback": "; ".join(verdict.issues), "by": "llm"}}
    return {"review": {"approved": True, "issues": [], "reviewed_version": version, "feedback": "", "by": "rules"}}
