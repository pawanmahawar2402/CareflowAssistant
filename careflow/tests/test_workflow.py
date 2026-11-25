"""Workflow tests: the live graph end to end, with the LLM stubbed out (no quota used).

Real components: guardrails, the graph, the MCP server (subprocess), SQLite checkpoints,
Redis memory (skipped when Redis is not running). Stubbed: intake parsing, the RAG retriever,
and the LLM critic, so the routing and review logic can be asserted deterministically.
"""
import uuid

import pytest

from careflow import llm
from careflow.m1_intake.models import PatientRequest
from careflow.m4_rag.rag_agent import RAGResult
from careflow.m5_workflow import graph as wf
from careflow.m6_agents import team
from careflow.m8_service.guardrails import REFUSAL

from .conftest import requires_redis

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _sid() -> str:
    return "t-" + uuid.uuid4().hex[:10]


@pytest.fixture(autouse=True)
def fake_intake(monkeypatch):
    """Intake parsing without an LLM: intent is chosen from the message text."""

    async def parse(raw, channel="unknown"):
        if "MRD-100037" in raw:
            return PatientRequest(chief_complaint="eligibility", raw_message=raw, channel=channel,
                                  intent="eligibility_check", insurance_id="MRD-100037")
        if "REF-0003" in raw:
            return PatientRequest(chief_complaint="referral", raw_message=raw, channel=channel,
                                  intent="referral_status")
        return PatientRequest(chief_complaint="general", raw_message=raw, channel=channel, intent="other")

    monkeypatch.setattr(team, "parse_message", parse)


class _FakeRAG:
    """Stands in for the retriever-backed policy agent; the first draft is deliberately vague."""

    async def answer(self, question, procedure=None, context=None, **_):
        text = ("Pre-authorization takes 3 business days [preauthorization_policy#0]."
                if "Reviewer feedback" in question else
                "Pre-authorization usually takes a while [preauthorization_policy#0].")
        return RAGResult(text, True, citations=["preauthorization_policy#0"], grounded=True,
                         groundedness_score=1.0, deterministic_ok=True)


def _critic(verdicts):
    """Scripted LLM critic: returns the next verdict in `verdicts` on each call."""
    calls = iter(verdicts)

    async def astructured(messages, schema, *, name="structured", model=None):
        ok = next(calls)
        return schema(approved=ok, issues=[] if ok else ["figure not supported"]), None

    return astructured


async def test_clinical_request_is_refused_and_ticketed():
    st = await wf.run("What dose of ibuprofen should I take?", _sid())
    assert st["route"] == "refuse"
    assert st["response"] == REFUSAL
    assert st["escalation_ticket"]["ticket_id"].startswith("TKT-")
    assert st["pending_human"] is False


async def test_emergency_pauses_and_resume_creates_exactly_one_ticket():
    sid = _sid()
    st = await wf.run("I have crushing chest pain", sid)
    assert st["pending_human"] is True
    first = await wf.resume(sid, True, "staff confirmed")
    again = await wf.resume(sid, True, "staff confirmed")  # a second resume must not duplicate
    assert first["escalation_ticket"]["ticket_id"] == again["escalation_ticket"]["ticket_id"]
    assert first["pending_human"] is False


async def test_eligibility_is_answered_through_mcp():
    st = await wf.run("Is insurance ID MRD-100037 active?", _sid())
    assert st["route"] == "auto"
    assert "MRD-100037" in st["response"] and "active" in st["response"]


async def test_referral_status_is_answered_through_mcp():
    st = await wf.run("Where is my referral REF-0003?", _sid())
    assert st["route"] == "auto"
    assert "REF-0003" in st["response"]


async def test_policy_answer_is_rejected_once_then_revised(monkeypatch):
    monkeypatch.setattr(team, "_rag", _FakeRAG())
    monkeypatch.setattr(llm, "astructured", _critic([False, True]))
    st = await wf.run("How long does pre-authorization take for an MRI?", _sid())
    assert st["review_loops"] == 1
    assert "3 business days" in st["response"]


async def test_draft_rejected_twice_is_never_released(monkeypatch):
    monkeypatch.setattr(team, "_rag", _FakeRAG())
    monkeypatch.setattr(llm, "astructured", _critic([False, False]))
    st = await wf.run("How long does pre-authorization take for an MRI?", _sid())
    assert st["route"] == "refuse"
    assert st["response"] == REFUSAL
    assert "usually takes a while" not in st["response"]


@requires_redis
async def test_episodic_memory_answers_last_time_in_a_new_session():
    user = "pt-" + uuid.uuid4().hex[:8]
    await wf.run("Where is my referral REF-0003?", _sid(), user_id=user)
    st = await wf.run("What did we do last time?", _sid(), user_id=user)
    assert "REF-0003" in st["response"]


@requires_redis
async def test_memory_is_not_shared_between_users():
    owner = "pt-" + uuid.uuid4().hex[:8]
    other = "pt-" + uuid.uuid4().hex[:8]
    await wf.run("Where is my referral REF-0003?", _sid(), user_id=owner)
    st = await wf.run("What did we do last time?", _sid(), user_id=other)
    assert "REF-0003" not in st["response"]
    assert "don't have a record" in st["response"]
