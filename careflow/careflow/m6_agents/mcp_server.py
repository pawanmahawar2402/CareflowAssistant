"""M6 — MCP server exposing the mock EHR / scheduling / insurance system.

Run (stdio):  python -m careflow.m6_agents.mcp_server

* tools      typed signatures + docstrings; arguments re-validated server-side
* resources  read-only context: policy documents and plan terms by URI
* prompts    reusable escalation-summary template
* hardening  closed set of readable documents (no path traversal), no dangerous
             capabilities, logging to STDERR only (stdout is the JSON-RPC stream)
"""
from __future__ import annotations

import logging
import sys
from typing import Literal

from mcp.server.fastmcp import FastMCP

from careflow.config import settings
from careflow.m2_agent import specs
from careflow.m2_agent import tools as impl

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="mcp-server %(levelname)s %(message)s")
log = logging.getLogger("careflow.mcp")

mcp = FastMCP("careflow-ehr")


@mcp.tool()
def lookup_patient(patient_id: str) -> dict:
    """Look up a synthetic patient record by ID (e.g. P0004). Returns {ok, patient} or {ok:false, error}."""
    return specs.invoke("lookup_patient", {"patient_id": patient_id})


@mcp.tool()
def check_eligibility(insurance_id: str) -> dict:
    """Check if an insurance member ID (e.g. MRD-100037) is active; returns plan terms or an error."""
    return specs.invoke("check_eligibility", {"insurance_id": insurance_id})


@mcp.tool()
def calculate_copay(plan_code: str, procedure_code: str, deductible_met_usd: float = 0.0) -> dict:
    """Estimate patient cost for a procedure on a plan. Not a coverage guarantee."""
    return specs.invoke("calculate_copay", {
        "plan_code": plan_code, "procedure_code": procedure_code, "deductible_met_usd": deductible_met_usd})


@mcp.tool()
def book_appointment(patient_id: str, specialty: str, urgency: Literal["routine", "priority", "urgent"] = "routine") -> dict:
    """Book the earliest open slot for a specialty. Enforces the per-patient booking cap."""
    return specs.invoke("book_appointment", {"patient_id": patient_id, "specialty": specialty, "urgency": urgency})


@mcp.tool()
def get_referral_status(referral_id: str) -> dict:
    """Get a referral's status by ID (e.g. REF-0003)."""
    return specs.invoke("get_referral_status", {"referral_id": referral_id})


@mcp.tool()
def create_escalation_ticket(
    reason: str,
    idempotency_key: str,
    priority: Literal["low", "normal", "high"] = "normal",
    patient_id: str | None = None,
    source_agent: str = "team",
    approved_by: str | None = None,
) -> dict:
    """Create a staff ticket exactly once per idempotency_key. Call ONLY after a human approved the escalation."""
    return impl.create_escalation_ticket(reason, idempotency_key, patient_id, priority, source_agent, approved_by)


# ── Resources ────────────────────────────────────────────────────────────────

def _policy_docs() -> dict[str, str]:
    return {p.stem: p.name for p in settings.corpus_dir.iterdir() if p.is_file()}


@mcp.resource("careflow://policy/{doc_id}")
def policy_document(doc_id: str) -> str:
    """Read one policy document by id (file stem). Only documents in the corpus can be read."""
    docs = _policy_docs()
    if doc_id not in docs:
        raise ValueError(f"unknown policy document '{doc_id}'")
    return (settings.corpus_dir / docs[doc_id]).read_text(encoding="utf-8")


@mcp.resource("careflow://plans/{plan_code}")
def plan_terms(plan_code: str) -> str:
    """Plan cost-share terms (JSON text) for a plan code."""
    import json

    for row in impl._load_table("plans"):
        if row["plan_code"] == plan_code:
            return json.dumps(row)
    raise ValueError(f"unknown plan '{plan_code}'")


# ── Prompts ──────────────────────────────────────────────────────────────────

@mcp.prompt()
def escalation_summary(category: str, reason: str) -> str:
    """Template for the summary a clinician sees when a case is escalated."""
    return (
        f"Escalation category: {category}\nReason: {reason}\n"
        "Write a 2-sentence neutral summary for a licensed clinician. Do not diagnose or advise."
    )


if __name__ == "__main__":
    log.info("starting CareFlow MCP server (stdio)")
    mcp.run()
