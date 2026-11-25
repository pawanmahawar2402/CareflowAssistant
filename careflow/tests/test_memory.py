"""Memory tests against real Redis (M3). The LLM used for fact extraction and summaries is stubbed."""
import json
import uuid

import pytest

from careflow import llm
from careflow.m3_memory.memory import MemoryManager, SemanticMemory, _Facts, _Fact

from .conftest import requires_redis

pytestmark = [pytest.mark.asyncio(loop_scope="module"), requires_redis]


def _user() -> str:
    return "pt-" + uuid.uuid4().hex[:8]


def _sid() -> str:
    return "s-" + uuid.uuid4().hex[:8]


def _fact_llm(monkeypatch, facts):
    """Stub the extraction call so each test controls exactly which facts the 'model' found."""

    async def astructured(messages, schema, *, name="structured", model=None):
        return _Facts(facts=[_Fact(key=k, value=v) for k, v in facts]), None

    monkeypatch.setattr(llm, "astructured", astructured)


async def test_semantic_preference_is_stored_and_recalled_in_a_new_session(monkeypatch):
    m, user = MemoryManager(), _user()
    _fact_llm(monkeypatch, [("preferred_channel", "sms")])
    await m.semantic.extract_and_store(user, _sid(), "Please text me, I prefer SMS")
    ctx = await m.recall(user, _sid(), "any appointments?")  # brand-new session
    assert ctx["facts"]["preferred_channel"] == "sms"


async def test_changed_fact_replaces_the_old_value(monkeypatch):
    m, user = MemoryManager(), _user()
    _fact_llm(monkeypatch, [("preferred_channel", "sms")])
    await m.semantic.extract_and_store(user, _sid(), "SMS please")
    _fact_llm(monkeypatch, [("preferred_channel", "phone")])
    written = await m.semantic.extract_and_store(user, _sid(), "Actually, call me instead")
    assert written[0]["replaced"] == "sms"
    assert (await m.recall(user, _sid(), "x"))["facts"]["preferred_channel"] == "phone"


async def test_episodes_are_recalled_newest_first_and_scoped_to_the_user():
    m, owner, other = MemoryManager(), _user(), _user()
    await m.episodic.record(owner, _sid(), "chat", "booked Cardiology")
    await m.episodic.record(owner, _sid(), "chat", "checked REF-0003")
    recent = await m.episodic.recent(owner)
    assert recent[0]["summary"] == "checked REF-0003"
    assert await m.episodic.recent(other) == []


async def test_procedure_is_selected_by_task_keywords():
    m = MemoryManager()
    sel = await m.procedural.select("I need an MRI, what pre-auth steps apply?")
    assert sel is not None and sel[0] == "preauth_imaging"
    assert len(sel[1]["steps"]) >= 3


async def test_policy_question_mentioning_referral_gets_no_task_procedure():
    m = MemoryManager()
    # a general policy question must not pull in the referral follow-up steps (found in the e2e run)
    assert await m.procedural.select("Can Gold members see a specialist without a referral?") is None
    sel = await m.procedural.select("Where is my referral REF-0003?")
    assert sel is not None and sel[0] == "referral_followup"


async def test_changed_seed_procedure_replaces_the_stored_copy():
    m = MemoryManager()
    await m.procedural.seed()
    stored = await m.procedural.get("referral_followup")
    # simulate an older stored copy with the previous, too-broad keyword list
    from careflow.m3_memory.memory import store
    old = {**stored, "keywords": ["referral"], "version": 1}
    await (await store.r()).hset(m.procedural._K, "referral_followup", json.dumps(old))
    await m.procedural.seed()
    fresh = await m.procedural.get("referral_followup")
    assert fresh["keywords"] != ["referral"] and fresh["version"] == 2


async def test_privacy_delete_removes_one_users_memory_only(monkeypatch):
    m, keep, drop = MemoryManager(), _user(), _user()
    _fact_llm(monkeypatch, [("preferred_language", "telugu")])
    await m.semantic.extract_and_store(keep, _sid(), "Telugu please")
    await m.semantic.extract_and_store(drop, _sid(), "Telugu please")
    await m.episodic.record(drop, _sid(), "chat", "something")
    assert await m.delete_user_data(drop) >= 1
    assert (await m.recall(drop, _sid(), "x"))["facts"] == {}
    assert (await m.recall(keep, _sid(), "x"))["facts"]["preferred_language"] == "telugu"


async def test_invalid_user_id_is_rejected_before_any_write():
    m = MemoryManager()
    with pytest.raises(ValueError):
        await m.episodic.record("bad user id!", _sid(), "chat", "x")
