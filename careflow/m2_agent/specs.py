"""M2 — Model-readable tool specs + a single safe invocation path.

The same registry feeds the single-agent loop (m2), the team graph (m5/m6) and
the MCP server (m6), so a tool's contract is defined exactly once.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from careflow.m2_agent import tools as impl
from careflow.m7_observability import tracing

log = logging.getLogger(__name__)

Specialty = Literal["Cardiology", "Orthopedics", "Dermatology", "Neurology", "Endocrinology", "Gastroenterology"]
Urgency = Literal["routine", "priority", "urgent"]


class LookupPatientArgs(BaseModel):
    patient_id: str = Field(..., pattern=r"^P\d{4}$", description="EHR patient ID, e.g. P0004")


class CheckEligibilityArgs(BaseModel):
    insurance_id: str = Field(..., pattern=r"^MRD-\d{5,8}$", description="Member ID, e.g. MRD-100037")


class CalculateCopayArgs(BaseModel):
    plan_code: Literal["MERIDIAN-BRONZE", "MERIDIAN-SILVER", "MERIDIAN-GOLD"] = Field(
        ..., description="Plan code returned by check_eligibility")
    procedure_code: Literal["CONSULT-PCP", "CONSULT-SPEC", "TELEHEALTH", "MRI-KNEE", "CT-CHEST", "LAB-PANEL"] = Field(
        ..., description="Procedure to price")
    deductible_met_usd: float = Field(0.0, ge=0, description="Deductible already met this year, in USD (>= 0)")


class BookAppointmentArgs(BaseModel):
    patient_id: str = Field(..., pattern=r"^P\d{4}$", description="EHR patient ID")
    specialty: Specialty = Field(..., description="Clinic specialty to book")
    urgency: Urgency = Field("routine", description="Scheduling urgency; never emergent (emergencies go to staff/911)")
    preferred_date: str | None = Field(None, description="Optional ISO date prefix, e.g. 2026-12-03")


class GetReferralStatusArgs(BaseModel):
    referral_id: str = Field(..., pattern=r"^REF-[A-Za-z0-9]{4,8}$", description="Referral ID, e.g. REF-0003")


class CreateReferralArgs(BaseModel):
    patient_id: str = Field(..., pattern=r"^P\d{4}$", description="EHR patient ID")
    specialty: Specialty = Field(..., description="Specialty being referred to")
    urgency: Urgency = Field("routine", description="Referral urgency")
    reason: str = Field(..., min_length=3, max_length=impl.MAX_REFERRAL_REASON_CHARS,
                        description="Administrative reason for the referral (no clinical interpretation)")


class RequestHumanHandoffArgs(BaseModel):
    reason: str = Field(..., min_length=3, description="Why a human must take over")
    idempotency_key: str = Field(..., min_length=3, description="Stable key (e.g. session id) so retries do not duplicate tickets")
    patient_id: str | None = Field(None, pattern=r"^P\d{4}$", description="Patient ID if known")
    priority: Literal["low", "normal", "high"] = Field("normal", description="Staff queue priority")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[..., dict[str, Any]]


TOOL_SPECS: dict[str, ToolSpec] = {
    s.name: s
    for s in [
        ToolSpec(
            "lookup_patient",
            "Look up a synthetic patient record by EHR patient ID. Use when the patient gives an ID "
            "and you need their plan code or eligibility. Do NOT use to answer anything clinical. "
            "Returns {ok, found, patient{patient_id, plan_code, eligibility_status, ...}} or "
            "{ok:false, error{code,message}}.",
            LookupPatientArgs, impl.lookup_patient),
        ToolSpec(
            "check_eligibility",
            "Check whether an insurance member ID is active and return plan cost-share terms. "
            "Use for 'is my insurance active?' or before estimating cost. Do NOT guess IDs. "
            "Returns {ok, eligible, plan_code, plan_name, deductible_usd, coinsurance_pct, ...} "
            "or {ok:false, error}.",
            CheckEligibilityArgs, impl.check_eligibility),
        ToolSpec(
            "calculate_copay",
            "Estimate what the patient owes for a procedure on a plan (flat co-pay for consultations, "
            "deductible + coinsurance for imaging/labs). Use after check_eligibility provides plan_code. "
            "Not a guarantee of coverage. Returns {ok, patient_owes_usd, copay_type, ...} or {ok:false, error}.",
            CalculateCopayArgs, impl.calculate_copay),
        ToolSpec(
            "book_appointment",
            "Book the earliest open slot for a specialty for a known patient. Use only after the patient "
            "asked to book and gave an ID. Enforces a per-patient booking cap. Never for emergencies. "
            "Returns {ok, appointment_id, provider_name, date, time, location} or {ok:false, error}.",
            BookAppointmentArgs, impl.book_appointment),
        ToolSpec(
            "get_referral_status",
            "Get the current status (pending/authorized/scheduled/completed/denied) of a referral by ID. "
            "Use for 'where is my referral?'. Returns {ok, found, referral{...}} or {ok:false, error}.",
            GetReferralStatusArgs, impl.get_referral_status),
        ToolSpec(
            "create_referral",
            "Create a new administrative specialist referral record. Use only when staff-approved intake "
            "asks for it; the reason must be administrative, not a diagnosis. "
            "Returns {ok, referral_id, status:'pending'} or {ok:false, error}.",
            CreateReferralArgs, impl.create_referral),
        ToolSpec(
            "request_human_handoff",
            "Hand the case to human clinic staff by opening a ticket. Use for ANY clinical question "
            "(diagnosis, symptoms, dosing, treatment), emergencies, or when unsure. Idempotent per key. "
            "Returns {ok, ticket_id, created, message} or {ok:false, error}.",
            RequestHumanHandoffArgs,
            lambda reason, idempotency_key, patient_id=None, priority="normal": impl.create_escalation_ticket(
                reason, idempotency_key, patient_id, priority, source_agent="single_agent")),
    ]
}


def openai_schemas(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Tool schemas in the OpenAI function-calling format (accepted by LiteLLM for all providers)."""
    return [
        {"type": "function", "function": {
            "name": s.name, "description": s.description, "parameters": s.args.model_json_schema()}}
        for n, s in TOOL_SPECS.items() if names is None or n in names
    ]


def invoke(name: str, arguments: dict[str, Any], allowed: list[str] | None = None) -> dict[str, Any]:
    """Validate arguments, run the tool, and ALWAYS return a structured dict.

    A failing or misused tool becomes {"ok": False, "error": ...} so the agent
    loop keeps running and the model can recover.
    """
    spec = TOOL_SPECS.get(name)
    if spec is None or (allowed is not None and name not in allowed):
        return impl.err("unknown_tool", f"Tool '{name}' is not available to this agent")
    start = time.perf_counter()
    with tracing.span(f"tool:{name}", as_type="tool", tool=name):
        try:
            parsed = spec.args.model_validate(arguments or {})
            result = spec.fn(**parsed.model_dump())
        except ValidationError as exc:
            result = impl.err("invalid_argument", "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()))
        except Exception:  # tool bug must not crash the loop
            log.exception("tool %s crashed", name)
            result = impl.err("tool_failure", f"Tool {name} failed unexpectedly")
    tracing.kpi({"kind": "tool", "name": name, "ok": bool(result.get("ok")),
                 "latency_ms": (time.perf_counter() - start) * 1000})
    return result
