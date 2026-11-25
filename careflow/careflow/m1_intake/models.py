"""M1 — PatientRequest Pydantic model.

Parses raw intake messages (phone transcript, web form, email, SMS) into a
validated, structured object that the downstream agents can rely on.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class UrgencyLevel(str, Enum):
    routine = "routine"
    priority = "priority"
    urgent = "urgent"
    emergent = "emergent"


class Intent(str, Enum):
    policy_question = "policy_question"
    referral_status = "referral_status"
    eligibility_check = "eligibility_check"
    scheduling = "scheduling"
    clinical_advice = "clinical_advice"
    other = "other"


INSURANCE_ID_RE = re.compile(r"^MRD-\d{5,8}$")
PATIENT_ID_RE = re.compile(r"^P\d{4}$")


class Channel(str, Enum):
    phone = "phone"
    web_form = "web_form"
    email = "email"
    sms = "sms"
    unknown = "unknown"


class PatientRequest(BaseModel):
    """Validated patient intake object produced by the Intake Agent (M1)."""

    # Identity
    patient_id: str | None = Field(None, description="EHR patient ID, if provided")
    full_name: str | None = Field(None, description="Patient's full name")

    # Insurance
    insurance_id: str | None = Field(None, description="Insurance member / policy ID")
    plan_code: str | None = Field(
        None, description="Normalised plan code, e.g. MERIDIAN-GOLD"
    )
    insurance_missing: bool = Field(
        False, description="True when the message contains no usable insurance ID"
    )

    # Chief complaint
    chief_complaint: str = Field(
        ..., description="One-sentence summary of why the patient is contacting the clinic"
    )
    symptoms: list[str] = Field(
        default_factory=list,
        description="Distinct symptom strings extracted from the message",
    )

    # Urgency
    urgency: UrgencyLevel = Field(
        UrgencyLevel.routine,
        description="Triage level assigned by the parser",
    )
    urgency_reason: str | None = Field(
        None, description="Why this urgency level was assigned"
    )

    # Routing
    requested_specialty: str | None = Field(
        None, description="Specialty the patient asked for, if any"
    )
    specialty_missing: bool = Field(
        False, description="True when no specialty could be inferred"
    )
    clinical_advice_sought: bool = Field(
        False,
        description=(
            "True when the message is fishing for a diagnosis, treatment plan, "
            "dosing advice, or symptom interpretation — anything that must be "
            "refused and escalated to a licensed clinician."
        ),
    )

    intent: Intent = Field(
        Intent.other,
        description="Primary purpose of the message (drives routing).",
    )

    # Metadata
    channel: Channel = Field(Channel.unknown, description="Message channel")
    raw_message: str = Field(..., description="Original unmodified patient message")
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional fields the LLM extracted that don't fit above",
    )

    @field_validator("symptoms", mode="before")
    @classmethod
    def _coerce_symptoms(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v or []

    @field_validator("urgency", mode="before")
    @classmethod
    def _normalise_urgency(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.lower().strip()
        return v

    @field_validator("insurance_id")
    @classmethod
    def _insurance_pattern(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip().upper()
        return v if INSURANCE_ID_RE.match(v) else None  # malformed IDs are treated as missing

    @field_validator("patient_id")
    @classmethod
    def _patient_pattern(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip().upper()
        return v if PATIENT_ID_RE.match(v) else None

    @model_validator(mode="after")
    def _cross_field(self) -> "PatientRequest":
        # insurance_missing must agree with insurance_id
        self.insurance_missing = self.insurance_id is None
        # clinical-advice requests are always routed as such
        if self.clinical_advice_sought:
            self.intent = Intent.clinical_advice.value
        if self.intent in (Intent.clinical_advice, Intent.clinical_advice.value):
            self.clinical_advice_sought = True
        # emergent triage must carry a reason
        if self.urgency in (UrgencyLevel.emergent, UrgencyLevel.emergent.value) and not self.urgency_reason:
            self.urgency_reason = "Symptoms described may be life-threatening"
        return self

    model_config = ConfigDict(use_enum_values=True)
