"""M7.C6 — Seeded failure injection: LLM-call success rate before and after hardening.

    python -m careflow.m7_observability.fault_harness

The model is a stub that always answers, so every failure seen here is an injected fault from the
seeded harness (`llm.configure_faults`). Two configurations are measured on the same seeded fault
sequence:

  baseline   no retries, no fallback, no breaker effect (hardened=False)
  hardened   retries with exponential backoff + jitter on transient errors, circuit breaker,
             and the secondary-model fallback (hardened=True)

Backoff and the breaker's recovery window are shortened so the run finishes quickly; the
retry counts and fault sequence are unchanged. Results go to `eval_results/fault_injection.json`.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import litellm

from careflow import llm
from careflow.config import PROJECT_ROOT, settings
from careflow.m7_observability.breaker import llm_breaker

CALLS = 500
FAULT_RATE = 0.3
SEED = settings.fault_seed


async def _stub_completion(model, messages, timeout=None, **_):
    msg = SimpleNamespace(content="ok", tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


async def _measure(hardened: bool) -> dict:
    llm.configure_faults(FAULT_RATE, SEED, hardened=hardened)
    llm_breaker._failures, llm_breaker._opened_at = 0, None  # each configuration starts with a closed breaker
    ok, latencies, fallbacks, retries = 0, [], 0, 0
    for _ in range(CALLS):
        t0 = time.perf_counter()
        try:
            res = await llm.acomplete([{"role": "user", "content": "ping"}], name="fault_harness")
            ok += 1
            fallbacks += res.used_fallback
            retries += res.retries
        except llm.LLMUnavailable:
            pass
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "calls": CALLS,
        "successes": ok,
        "success_rate": round(ok / CALLS, 4),
        "retries_total": retries,
        "fallback_calls": fallbacks,
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 2),
    }


async def main() -> None:
    litellm.acompletion = _stub_completion
    litellm.completion_cost = lambda completion_response: 0.0
    settings.llm_backoff_base_s = 0.01  # shorter backoff for the harness only
    llm_breaker.recovery_timeout = 0.5

    baseline = await _measure(hardened=False)
    hardened = await _measure(hardened=True)
    llm.configure_faults(0.0, SEED, hardened=True)  # switch injection off again

    out = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "fault_rate": FAULT_RATE,
        "seed": SEED,
        "calls_per_configuration": CALLS,
        "note": "stub model; faults injected at the LLM layer; backoff shortened to 0.01s and breaker "
                "recovery to 0.5s for the run (production: 1.0s base backoff, 30s recovery)",
        "baseline": baseline,
        "hardened": hardened,
    }
    path = PROJECT_ROOT / "eval_results" / "fault_injection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"fault rate {FAULT_RATE}, {CALLS} calls each, seed {SEED}")
    print(f"baseline success {baseline['success_rate']:.1%}  hardened success {hardened['success_rate']:.1%}")
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
