"""M5 — Typed LangGraph state with deliberate reducers.

overwrite (default)  control keys the router reads: next_agent, hops, review_loops, ...
append (operator.add) accumulating keys: `audit` is an append-only decision trail
Each agent writes only its own keys (enforced in `m6_agents.team.agent_node`).
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class CareFlowState(TypedDict, total=False):
    # ── input (set once by the service layer) ───────────────────────────────
    raw_message: str
    channel: str
    session_id: str
    user_id: str | None
    memory: dict[str, Any]            # recalled working/semantic/episodic/procedural context

    # ── deterministic input guard ───────────────────────────────────────────
    guard: dict[str, Any]             # {allowed, category, message, flags}

    # ── intake agent ────────────────────────────────────────────────────────
    patient_request: dict[str, Any]   # PatientRequest.model_dump(mode="json")
    intake_error: str

    # ── supervisor control (overwrite) ──────────────────────────────────────
    route: str                        # auto | refuse | human_review (read by the routers)
    response: str                     # reply text the patient would see
    next_agent: str
    hops: int
    review_loops: int
    escalation: dict[str, Any]        # {category, reason, priority, patient_message}

    # ── specialist agents ───────────────────────────────────────────────────
    policy_result: dict[str, Any]
    referral_result: dict[str, Any]
    draft_response: str
    draft_version: int
    draft_kind: str                   # answered | not_in_corpus
    last_agent: str

    # ── clinical safety reviewer ────────────────────────────────────────────
    review: dict[str, Any]            # {approved, issues, reviewed_version, feedback}

    # ── human-in-the-loop ───────────────────────────────────────────────────
    approval: dict[str, Any]          # {decision, approver, reason, decided_at}
    ticket: dict[str, Any]
    hil_approved: bool | None         # written by resume(): staff decision
    hil_note: str                     # written by resume(): staff note
    escalation_ticket: dict[str, Any]  # ticket created after the interrupt

    # ── result ──────────────────────────────────────────────────────────────
    final_response: str
    outcome: str                      # answered | not_in_corpus | escalated | error
    guardrail_flags: list[str]
    refusal_reason: str               # why the case was refused (goes on the staff ticket)

    # ── audit trail (append) ────────────────────────────────────────────────
    audit: Annotated[list[dict[str, Any]], operator.add]
