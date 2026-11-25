"""Generate synthetic, de-identified CareFlow data.

    python generate_data.py

Writes:
    data/tables/*.csv            mock EHR / scheduling / insurance tables
    data/corpus/*                policy & insurance documents for RAG
    data/eval/retrieval_golden.jsonl   M4 retrieval golden set
    data/eval/e2e_golden.jsonl         M8 end-to-end golden set
"""
from __future__ import annotations

import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)
ROOT = Path(__file__).parent / "data"
TABLES, CORPUS, EVAL = ROOT / "tables", ROOT / "corpus", ROOT / "eval"
for d in (TABLES, CORPUS, EVAL):
    d.mkdir(parents=True, exist_ok=True)


def write_csv(name: str, rows: list[dict]) -> None:
    with (TABLES / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


PLANS = [
    {"plan_code": "MERIDIAN-BRONZE", "plan_name": "Meridian Bronze", "annual_deductible_usd": 4000, "coinsurance_pct": 0.30, "out_of_pocket_max_usd": 8000},
    {"plan_code": "MERIDIAN-SILVER", "plan_name": "Meridian Silver", "annual_deductible_usd": 2000, "coinsurance_pct": 0.20, "out_of_pocket_max_usd": 5500},
    {"plan_code": "MERIDIAN-GOLD", "plan_name": "Meridian Gold", "annual_deductible_usd": 500, "coinsurance_pct": 0.10, "out_of_pocket_max_usd": 3000},
]
PROCEDURES = [
    {"procedure_code": "CONSULT-PCP", "category": "consultation", "list_price_usd": 180},
    {"procedure_code": "CONSULT-SPEC", "category": "consultation", "list_price_usd": 320},
    {"procedure_code": "TELEHEALTH", "category": "telehealth", "list_price_usd": 90},
    {"procedure_code": "MRI-KNEE", "category": "imaging", "list_price_usd": 1400},
    {"procedure_code": "CT-CHEST", "category": "imaging", "list_price_usd": 1100},
    {"procedure_code": "LAB-PANEL", "category": "lab", "list_price_usd": 150},
]
COPAY_BASE = {"MERIDIAN-BRONZE": (60, 100, 30), "MERIDIAN-SILVER": (40, 70, 20), "MERIDIAN-GOLD": (20, 40, 10)}
COPAYS = []
for plan, (pcp, spec, tele) in COPAY_BASE.items():
    COPAYS += [
        {"plan_code": plan, "procedure_code": "CONSULT-PCP", "patient_copay_usd": pcp},
        {"plan_code": plan, "procedure_code": "CONSULT-SPEC", "patient_copay_usd": spec},
        {"plan_code": plan, "procedure_code": "TELEHEALTH", "patient_copay_usd": tele},
    ]

SPECIALTIES = ["Cardiology", "Orthopedics", "Dermatology", "Neurology", "Endocrinology", "Gastroenterology"]
patients = []
for i in range(1, 41):
    plan = random.choice(PLANS)
    patients.append({
        "patient_id": f"P{i:04d}",
        "full_name": f"Patient {i:04d}",  # synthetic, de-identified
        "insurance_id": f"MRD-{100000 + i * 37}",
        "plan_code": plan["plan_code"],
        "eligibility_status": "active" if i % 9 else "lapsed",
        "deductible_met_usd": random.choice([0, 250, 800, 1500, 2500]),
    })
slots, start = [], date.today() + timedelta(days=2)
for n in range(60):
    spec = SPECIALTIES[n % len(SPECIALTIES)]
    slots.append({
        "slot_id": f"S{n:03d}", "specialty": spec, "status": "open",
        "date": (start + timedelta(days=n // 3)).isoformat(),
        "start_time": random.choice(["09:00", "10:30", "13:00", "15:30"]),
        "provider_id": f"DR{(n % 12) + 1:02d}", "provider_name": f"Dr. Synthetic {(n % 12) + 1:02d}",
        "site": random.choice(["Meridian Downtown", "Meridian Westside"]),
    })
referrals = []
for i in range(1, 21):
    referrals.append({
        "referral_id": f"REF-{i:04d}", "patient_id": f"P{i:04d}",
        "specialty": random.choice(SPECIALTIES),
        "status": random.choice(["pending", "authorized", "scheduled", "completed", "denied"]),
        "urgency": random.choice(["routine", "priority", "urgent"]),
        "created_at": (date.today() - timedelta(days=random.randint(1, 30))).isoformat(),
    })
for name, rows in [("plans", PLANS), ("procedures", PROCEDURES), ("copays", COPAYS),
                   ("patients", patients), ("open_slots", slots), ("referrals", referrals)]:
    write_csv(name, rows)

# ── Corpus + golden sets (content lives in data_content.py) ───────────────────
from data_content import DOCS, E2E_GOLDEN, RETRIEVAL_GOLDEN  # noqa: E402

for old in CORPUS.glob("*"):
    old.unlink()
for fname, body in DOCS.items():
    (CORPUS / fname).write_text(body, encoding="utf-8")
for name, rows in (("retrieval_golden", RETRIEVAL_GOLDEN), ("e2e_golden", E2E_GOLDEN)):
    with (EVAL / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
(EVAL / "golden.jsonl").unlink(missing_ok=True)

print(f"Wrote tables -> {TABLES}\nWrote {len(DOCS)} corpus docs -> {CORPUS}\nWrote golden sets -> {EVAL}")
