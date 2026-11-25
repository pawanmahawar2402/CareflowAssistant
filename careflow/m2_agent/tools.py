"""M2 — Mock EHR / scheduling / insurance tools (pure functions, no LLM).

Contract for every tool:
  * returns a dict:  {"ok": True, ...data}   or   {"ok": False, "error": {"code", "message"}}
  * never raises on bad input (the model must see the error and recover)
  * business limits are enforced HERE, not in prompts.

Read-only data comes from CSV tables (data/tables); writes (tickets, bookings,
referrals) go to a SQLite file under runtime/ so side effects are inspectable
and idempotent.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from careflow.config import settings

log = logging.getLogger(__name__)

# ── Business limits / constants (single definition; read by tools and guardrails)
SPECIALTIES = ("Cardiology", "Orthopedics", "Dermatology", "Neurology", "Endocrinology", "Gastroenterology")
URGENCY_LEVELS = ("routine", "priority", "urgent", "emergent")
PRIORITIES = ("low", "normal", "high")
MAX_ACTIVE_BOOKINGS_PER_PATIENT = 3
MAX_REFERRAL_REASON_CHARS = 500

_ID_PATIENT = re.compile(r"^P\d{4}$")
_ID_INSURANCE = re.compile(r"^MRD-\d{5,8}$")
_ID_REFERRAL = re.compile(r"^REF-[A-Z0-9]{4,8}$")


def err(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_table(name: str) -> list[dict[str, Any]]:
    path = settings.tables_dir / f"{name}.csv"
    if not path.exists():
        log.warning("table %s not found at %s", name, path)
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── Persistent write store ────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(settings.runtime_path(settings.ticket_db))
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS tickets(
            idempotency_key TEXT PRIMARY KEY, ticket_id TEXT, patient_id TEXT,
            reason TEXT, priority TEXT, source_agent TEXT, approved_by TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS bookings(
            appointment_id TEXT PRIMARY KEY, patient_id TEXT, specialty TEXT, slot_id TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS referrals(
            referral_id TEXT PRIMARY KEY, patient_id TEXT, specialty TEXT, urgency TEXT,
            reason TEXT, status TEXT, created_at TEXT);
        """
    )
    return con


# ── EHR: patient lookup ───────────────────────────────────────────────────────

def lookup_patient(patient_id: str) -> dict[str, Any]:
    if not _ID_PATIENT.match(patient_id or ""):
        return err("invalid_argument", "patient_id must look like P0001")
    for row in _load_table("patients"):
        if row.get("patient_id") == patient_id:
            return {"ok": True, "found": True, "patient": row}
    return err("not_found", f"No patient {patient_id}")


# ── Insurance ─────────────────────────────────────────────────────────────────

def check_eligibility(insurance_id: str) -> dict[str, Any]:
    insurance_id = (insurance_id or "").strip().upper()
    if not _ID_INSURANCE.match(insurance_id):
        return err("invalid_argument", "insurance_id must look like MRD-123456")
    plans = {r["plan_code"]: r for r in _load_table("plans")}
    for row in _load_table("patients"):
        if row.get("insurance_id") == insurance_id:
            plan = plans.get(row.get("plan_code", ""), {})
            return {
                "ok": True,
                "eligible": row.get("eligibility_status") == "active",
                "insurance_id": insurance_id,
                "patient_id": row.get("patient_id"),
                "plan_code": row.get("plan_code"),
                "plan_name": plan.get("plan_name"),
                "deductible_usd": float(plan.get("annual_deductible_usd", 0) or 0),
                "deductible_met_usd": float(row.get("deductible_met_usd", 0) or 0),
                "coinsurance_pct": float(plan.get("coinsurance_pct", 0) or 0),
                "out_of_pocket_max_usd": float(plan.get("out_of_pocket_max_usd", 0) or 0),
                "eligibility_status": row.get("eligibility_status"),
            }
    return err("not_found", f"Insurance ID {insurance_id} not found")


def calculate_copay(plan_code: str, procedure_code: str, deductible_met_usd: float = 0.0) -> dict[str, Any]:
    if deductible_met_usd is None or deductible_met_usd < 0:
        return err("invalid_argument", "deductible_met_usd must be >= 0")
    plan = next((r for r in _load_table("plans") if r["plan_code"] == plan_code), None)
    proc = next((r for r in _load_table("procedures") if r["procedure_code"] == procedure_code), None)
    if not plan:
        return err("not_found", f"Unknown plan_code {plan_code}")
    if not proc:
        return err("not_found", f"Unknown procedure_code {procedure_code}")
    copay = next((r for r in _load_table("copays")
                  if r["plan_code"] == plan_code and r["procedure_code"] == procedure_code), None)
    list_price = float(proc["list_price_usd"])
    base = {"ok": True, "plan_code": plan_code, "procedure_code": procedure_code,
            "category": proc["category"], "list_price_usd": list_price}
    if copay:
        return {**base, "patient_owes_usd": float(copay["patient_copay_usd"]), "copay_type": "flat"}
    deductible = float(plan["annual_deductible_usd"])
    coins = float(plan["coinsurance_pct"])
    unmet = max(deductible - deductible_met_usd, 0.0)
    coinsurance_amount = round(max(list_price - unmet, 0.0) * coins, 2)
    return {**base, "deductible_usd": deductible, "unmet_deductible_usd": round(unmet, 2),
            "coinsurance_pct": coins, "coinsurance_amount_usd": coinsurance_amount,
            "patient_owes_usd": round(min(unmet, list_price) + coinsurance_amount, 2),
            "copay_type": "deductible+coinsurance"}


# ── Scheduling ────────────────────────────────────────────────────────────────

def book_appointment(patient_id: str, specialty: str, urgency: str = "routine",
                     preferred_date: str | None = None) -> dict[str, Any]:
    if not _ID_PATIENT.match(patient_id or ""):
        return err("invalid_argument", "patient_id must look like P0001")
    if specialty not in SPECIALTIES:
        return err("invalid_argument", f"specialty must be one of {', '.join(SPECIALTIES)}")
    if urgency not in URGENCY_LEVELS[:3]:
        return err("invalid_argument", "urgency must be routine, priority or urgent")
    if not lookup_patient(patient_id).get("ok"):
        return err("not_found", f"No patient {patient_id}")
    con = _db()
    try:
        active = con.execute("SELECT COUNT(*) FROM bookings WHERE patient_id=?", (patient_id,)).fetchone()[0]
        if active >= MAX_ACTIVE_BOOKINGS_PER_PATIENT:
            return err("limit_exceeded", f"Patient already has {active} active bookings (max "
                       f"{MAX_ACTIVE_BOOKINGS_PER_PATIENT}); ask staff to review")
        taken = {r[0] for r in con.execute("SELECT slot_id FROM bookings")}
        slots = sorted((s for s in _load_table("open_slots")
                        if s["specialty"] == specialty and s["status"] == "open" and s["slot_id"] not in taken),
                       key=lambda s: (s["date"], s["start_time"]))
        if not slots:
            return err("no_availability", f"No open {specialty} slots")
        slot = next((s for s in slots if preferred_date and s["date"].startswith(preferred_date)), slots[0])
        appt = f"APT-{patient_id}-{slot['slot_id']}"
        con.execute("INSERT INTO bookings VALUES(?,?,?,?,?)", (appt, patient_id, specialty, slot["slot_id"], _now()))
        con.commit()
    finally:
        con.close()
    return {"ok": True, "appointment_id": appt, "patient_id": patient_id, "specialty": specialty,
            "provider_name": slot["provider_name"], "date": slot["date"], "time": slot["start_time"],
            "location": slot["site"], "urgency": urgency}


# ── Referrals ─────────────────────────────────────────────────────────────────

def get_referral_status(referral_id: str) -> dict[str, Any]:
    referral_id = (referral_id or "").strip().upper()
    if not _ID_REFERRAL.match(referral_id):
        return err("invalid_argument", "referral_id must look like REF-0003")
    con = _db()
    try:
        row = con.execute("SELECT * FROM referrals WHERE referral_id=?", (referral_id,)).fetchone()
    finally:
        con.close()
    if row:
        return {"ok": True, "found": True, "referral": dict(row)}
    for r in _load_table("referrals"):
        if r["referral_id"] == referral_id:
            return {"ok": True, "found": True, "referral": r}
    return err("not_found", f"Referral {referral_id} not found")


def create_referral(patient_id: str, specialty: str, urgency: str, reason: str) -> dict[str, Any]:
    if not _ID_PATIENT.match(patient_id or ""):
        return err("invalid_argument", "patient_id must look like P0001")
    if specialty not in SPECIALTIES:
        return err("invalid_argument", f"specialty must be one of {', '.join(SPECIALTIES)}")
    if urgency not in URGENCY_LEVELS[:3]:
        return err("invalid_argument", "urgency must be routine, priority or urgent")
    if not reason or len(reason) > MAX_REFERRAL_REASON_CHARS:
        return err("invalid_argument", f"reason is required (<= {MAX_REFERRAL_REASON_CHARS} chars)")
    if not lookup_patient(patient_id).get("ok"):
        return err("not_found", f"No patient {patient_id}")
    ref_id = "REF-" + hashlib.sha1(f"{patient_id}:{specialty}:{_now()}".encode()).hexdigest()[:6].upper()
    con = _db()
    try:
        con.execute("INSERT INTO referrals VALUES(?,?,?,?,?,?,?)",
                    (ref_id, patient_id, specialty, urgency, reason, "pending", _now()))
        con.commit()
    finally:
        con.close()
    return {"ok": True, "referral_id": ref_id, "patient_id": patient_id, "specialty": specialty,
            "urgency": urgency, "status": "pending"}


# ── Escalation (idempotent, written only AFTER human approval in the workflow) ──

def create_escalation_ticket(reason: str, idempotency_key: str, patient_id: str | None = None,
                             priority: str = "normal", source_agent: str = "system",
                             approved_by: str | None = None) -> dict[str, Any]:
    """Create (once) a staff ticket. Same idempotency_key → same ticket, no duplicate row."""
    if priority not in PRIORITIES:
        return err("invalid_argument", f"priority must be one of {', '.join(PRIORITIES)}")
    if not reason or not idempotency_key:
        return err("invalid_argument", "reason and idempotency_key are required")
    ticket_id = "TKT-" + hashlib.sha1(idempotency_key.encode()).hexdigest()[:6].upper()
    con = _db()
    try:
        cur = con.execute("INSERT OR IGNORE INTO tickets VALUES(?,?,?,?,?,?,?,?)",
                          (idempotency_key, ticket_id, patient_id, reason, priority, source_agent,
                           approved_by, _now()))
        con.commit()
        created = cur.rowcount == 1
    finally:
        con.close()
    log.warning("escalation ticket=%s created=%s priority=%s source=%s", ticket_id, created, priority, source_agent)
    return {"ok": True, "ticket_id": ticket_id, "created": created, "priority": priority,
            "message": f"A staff member will contact you. Reference {ticket_id}."}


def count_tickets(idempotency_key: str | None = None) -> int:
    """Test/verification helper: number of ticket rows (optionally for one key)."""
    con = _db()
    try:
        if idempotency_key:
            return con.execute("SELECT COUNT(*) FROM tickets WHERE idempotency_key=?", (idempotency_key,)).fetchone()[0]
        return con.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    finally:
        con.close()
