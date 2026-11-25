"""M8 — Guardrails enforced in code, independent of model behaviour.

* injection scan on user input AND on retrieved content (weighted pattern score vs one threshold)
* domain policy guardrail: block clinical / diagnostic advice requests (incl. self-declared clinicians)
* emergency detection → human escalation
* output validation: schema + unsafe-advice patterns + sensitive identifiers

Every threshold lives in `careflow.config.settings`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from careflow.config import settings


class ChatResult(BaseModel):
    """Output schema every response must satisfy before leaving the system."""

    response: str = Field(..., min_length=1, max_length=4000)
    route: str
    guardrail_flags: list[str] = Field(default_factory=list)


# (pattern, weight) — a message is blocked when the summed weight >= settings.injection_block_score
_INJECTION: list[tuple[str, float]] = [
    (r"\bignore (all |any |your |the )?(previous |prior |above )?(instructions|rules|guidelines|policies)\b", 0.6),
    (r"\bdisregard (all |any |your |the )?(previous |prior )?(instructions|rules|policies)\b", 0.6),
    (r"\b(you are|act as|pretend to be) (now )?(dan|an? unrestricted|an? unfiltered|the system|admin)", 0.5),
    (r"\b(system prompt|developer message|reveal your (prompt|instructions))\b", 0.4),
    (r"\bapprove (this|it|everything) (immediately|now|without)\b", 0.5),
    (r"\bwithout (human|staff|clinician) (review|approval|sign[- ]?off)\b", 0.4),
    (r"\b(override|bypass|skip) (the )?(guardrails?|safety|review|approval|checks?)\b", 0.6),
    (r"<\s*/?\s*(system|assistant|instructions?)\s*>", 0.5),
    (r"\[\s*(system|inst)\s*\]|\bBEGIN (SYSTEM|HIDDEN) ", 0.5),
    (r"\bnew instructions?:", 0.4),
]

_CLINICAL_REQUEST = [
    r"\bdiagnos\w*",
    r"\bwhat (dose|dosage)\b|\bhow (much|many)\b.*\b(mg|milligrams?|tablets?|pills?|units)\b|\b\d+\s?mg\b",
    r"\b(dose|dosage|dosing)\b",
    r"\bshould i (take|stop|start|use|increase|reduce)\b",
    r"\bwhat (cream|medicine|medication|drug|treatment|antibiotic)\b",
    r"\bis (it|this|my \w+) (cancer|serious|dangerous|infected|a tumou?r|normal)\b",
    r"\bwhat (is|could|might) (it|this|my \w+) (be|mean)\b",
    r"\b(interpret|explain) (my )?(test|lab|scan|mri|blood) results?\b",
    r"\b(prescribe|prescription for)\b",
    r"\b(is|are) .* (safe|ok|okay) to (take|combine|mix)\b",
    r"\b(tell|decide) .*\b(what'?s wrong|what is wrong)\b",
]
_EMERGENT = [
    r"\bchest (pain|pressure|tightness)\b", r"\bcan'?t breathe\b", r"\bcannot breathe\b",
    r"\b(difficulty|trouble) breathing\b", r"\bstroke\b", r"\bsuicid\w*", r"\bunconscious\b",
    r"\bsevere bleeding\b", r"\blips are blue\b", r"\boverdos\w*", r"\bface (is )?droop\w*",
]
_UNSAFE_OUTPUT = [
    r"\byou (likely|probably|may) have\b", r"\bthis (sounds|looks) like (a|an)\b",
    r"\byou should (take|stop taking|start taking|increase|reduce)\b",
    r"\b(take|use) \d+\s?(mg|ml|pills?|tablets?)\b", r"\bI (diagnose|recommend taking)\b",
    r"\bit is (safe|fine) to (take|combine|mix)\b",
]
_PII = [r"\b\d{3}-\d{2}-\d{4}\b", r"\b\d{16}\b"]

REFUSAL = (
    "I'm a care coordination assistant, so I can't give diagnosis, symptom interpretation, "
    "medication or treatment advice. A licensed clinician on our nurse line will follow up "
    "with you. If this is an emergency, call 911 or go to the nearest ER."
)
EMERGENCY = (
    "Your message describes symptoms that may be an emergency. Please call 911 or go to the "
    "nearest emergency room right now. Our clinical staff have been alerted."
)
INJECTION_REFUSAL = (
    "I can't act on that request. I can only help with scheduling, referrals and insurance "
    "questions, and I never bypass staff review. Your message was sent to our staff."
)


def _hit(patterns: list[str], text: str) -> list[str]:
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


@dataclass
class InjectionScan:
    score: float
    flags: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.score >= settings.injection_block_score


def scan_injection(text: str) -> InjectionScan:
    """Weighted pattern score; used for user input and for retrieved documents."""
    score, flags = 0.0, []
    for pat, w in _INJECTION:
        if re.search(pat, text, re.IGNORECASE):
            score += w
            flags.append(pat[:40])
    return InjectionScan(min(score, 1.0), flags)


@dataclass
class GuardResult:
    allowed: bool
    category: str  # ok | injection | clinical | emergent | invalid | unsafe_output | pii
    message: str = ""
    flags: list[str] = field(default_factory=list)


def check_input(text: str) -> GuardResult:
    """Order matters: emergencies first (never delay a 911 message), then injection, then clinical."""
    if not text or not text.strip():
        return GuardResult(False, "invalid", "Message is empty.")
    if len(text) > settings.max_message_chars:
        return GuardResult(False, "invalid", f"Message exceeds {settings.max_message_chars} characters.")
    if _hit(_EMERGENT, text):
        return GuardResult(False, "emergent", EMERGENCY, ["emergent_symptoms"])
    inj = scan_injection(text)
    if inj.blocked:
        return GuardResult(False, "injection", INJECTION_REFUSAL, ["prompt_injection", *inj.flags])
    if _hit(_CLINICAL_REQUEST, text):
        return GuardResult(False, "clinical", REFUSAL, ["clinical_advice_request"])
    return GuardResult(True, "ok")


def check_retrieved(text: str) -> GuardResult:
    """Retrieved documents are data; instruction-like content is dropped before prompting."""
    inj = scan_injection(text)
    if inj.blocked:
        return GuardResult(False, "injection", "", ["indirect_injection", *inj.flags])
    return GuardResult(True, "ok")


def check_output(text: str) -> GuardResult:
    if _hit(_UNSAFE_OUTPUT, text):
        return GuardResult(False, "unsafe_output", REFUSAL, ["clinical_advice_in_output"])
    if _hit(_PII, text):
        return GuardResult(False, "pii", "[Response withheld: it contained sensitive identifiers.]", ["pii_in_output"])
    return GuardResult(True, "ok")
