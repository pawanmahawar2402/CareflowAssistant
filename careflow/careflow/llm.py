"""Single provider-agnostic LLM layer (M1.C1/C2, M7.C5).

Every LLM and embedding call in CareFlow goes through this module; no provider
SDK is imported anywhere else. The model is chosen by configuration only.

Hardening applied to every call:
  * per-call timeout
  * retries with exponential backoff + jitter on *transient* errors only
  * circuit breaker (closed / open / half-open)
  * optional secondary model; the result is flagged with `used_fallback`
  * tracing as a Langfuse *generation* with token usage and cost, plus a KPI log line
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from careflow.config import settings
from careflow.m7_observability import tracing
from careflow.m7_observability.breaker import CircuitOpenError, llm_breaker

log = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)

litellm.num_retries = 0  # retries are governed here, not inside LiteLLM
litellm.drop_params = True
litellm.suppress_debug_info = True

_TRANSIENT = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.InternalServerError,
)


class LLMUnavailable(RuntimeError):
    """LLM could not be reached after retries / fallback."""


class InjectedFault(litellm.ServiceUnavailableError):
    """Seeded fault used by the failure-injection harness."""

    def __init__(self) -> None:
        super().__init__(message="injected fault", llm_provider="fault", model="fault")


class _Runtime:
    """Process-wide switches the failure-injection harness flips."""

    hardened = True
    fault_rate = settings.fault_rate
    rng = random.Random(settings.fault_seed)


runtime = _Runtime()


def configure_faults(rate: float, seed: int, hardened: bool) -> None:
    runtime.fault_rate = rate
    runtime.rng = random.Random(seed)
    runtime.hardened = hardened


@dataclass
class LLMResult:
    content: str
    message: Any = None
    tool_calls: list[Any] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float | None = None
    used_fallback: bool = False
    retries: int = 0
    latency_ms: float = 0.0


def _maybe_inject_fault() -> None:
    if runtime.fault_rate > 0 and runtime.rng.random() < runtime.fault_rate:
        raise InjectedFault()


async def _call_once(model: str, kwargs: dict[str, Any]) -> Any:
    _maybe_inject_fault()
    return await litellm.acompletion(model=model, timeout=settings.llm_timeout_s, **kwargs)


async def _call_with_retries(model: str, kwargs: dict[str, Any]) -> tuple[Any, int]:
    attempts = settings.llm_max_retries + 1 if runtime.hardened else 1
    retries = 0
    for attempt in range(attempts):
        try:
            if runtime.hardened:
                return await llm_breaker.call(_call_once, model, kwargs), retries
            return await _call_once(model, kwargs), retries
        except _TRANSIENT as exc:
            if attempt == attempts - 1:
                raise
            retries += 1
            delay = settings.llm_backoff_base_s * (2**attempt) * (0.5 + random.random() / 2)
            log.warning("transient LLM error (%s); retry %d in %.1fs", type(exc).__name__, retries, delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def acomplete(
    messages: list[dict[str, Any]],
    *,
    name: str = "llm",
    tools: list[dict[str, Any]] | None = None,
    json_mode: bool = False,
    model: str | None = None,
    temperature: float | None = None,
) -> LLMResult:
    """One chat completion through LiteLLM with the full hardening stack."""
    primary = model or settings.careflow_llm_model
    kwargs: dict[str, Any] = {
        "messages": messages,
        "temperature": settings.llm_temperature if temperature is None else temperature,
    }
    if tools:
        kwargs.update(tools=tools, tool_choice="auto")
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    used_fallback = False
    start = time.perf_counter()
    chosen = primary
    with tracing.generation(name, primary, agent=tracing.current_agent()) as obs:
        try:
            try:
                resp, retries = await _call_with_retries(primary, kwargs)
            except (*_TRANSIENT, CircuitOpenError) as exc:
                fb = settings.careflow_fallback_llm_model
                if not (fb and runtime.hardened):
                    raise LLMUnavailable(f"LLM unavailable: {type(exc).__name__}") from exc
                log.warning("primary model failed (%s); using fallback %s", type(exc).__name__, fb)
                chosen, used_fallback = fb, True
                resp, retries = await _call_with_retries(fb, kwargs)
        except Exception as exc:
            tracing.kpi({"kind": "llm", "name": name, "model": chosen, "ok": False,
                         "error": type(exc).__name__, "latency_ms": (time.perf_counter() - start) * 1000})
            if isinstance(exc, (LLMUnavailable,)):
                raise
            if isinstance(exc, (*_TRANSIENT, CircuitOpenError)):
                raise LLMUnavailable(f"LLM unavailable: {type(exc).__name__}") from exc
            raise

        latency = (time.perf_counter() - start) * 1000
        msg = resp.choices[0].message
        u = getattr(resp, "usage", None)
        usage = {
            "input": getattr(u, "prompt_tokens", 0) or 0,
            "output": getattr(u, "completion_tokens", 0) or 0,
            "total": getattr(u, "total_tokens", 0) or 0,
        }
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = None
        tracing.record_generation(obs, chosen, usage, cost, used_fallback=used_fallback, retries=retries)
        tracing.kpi({"kind": "llm", "name": name, "model": chosen, "ok": True, "tokens": usage["total"],
                     "cost_usd": cost, "latency_ms": latency, "used_fallback": used_fallback,
                     "retries": retries})
        return LLMResult(
            content=msg.content or "", message=msg, tool_calls=list(msg.tool_calls or []),
            model=chosen, usage=usage, cost_usd=cost, used_fallback=used_fallback,
            retries=retries, latency_ms=latency,
        )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    return json.loads(text)


async def astructured(
    messages: list[dict[str, Any]],
    schema: type[M],
    *,
    name: str = "structured",
    model: str | None = None,
) -> tuple[M, LLMResult]:
    """Structured output validated against a Pydantic model.

    Invalid output goes through a bounded repair loop (the validation error is
    sent back to the model); after `llm_json_repair_attempts` a ValueError is raised.
    Validation errors are never retried as transient errors.
    """
    msgs = list(messages)
    msgs.append({"role": "system", "content": "Respond with one JSON object matching this JSON Schema:\n"
                 + json.dumps(schema.model_json_schema())})
    last_err: Exception | None = None
    for attempt in range(settings.llm_json_repair_attempts + 1):
        res = await acomplete(msgs, name=name, json_mode=True, model=model)
        try:
            return schema.model_validate(_extract_json(res.content)), res
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_err = exc
            log.warning("structured output invalid (attempt %d): %s", attempt + 1, str(exc)[:200])
            msgs.append({"role": "assistant", "content": res.content})
            msgs.append({"role": "user", "content": f"That was invalid: {str(exc)[:500]}. Return corrected JSON only."})
    raise ValueError(f"structured output failed validation after repairs: {last_err}")


async def aembed(texts: list[str]) -> list[list[float]]:
    """Embeddings through LiteLLM with the same retry policy."""
    attempts = settings.llm_max_retries + 1
    for attempt in range(attempts):
        try:
            _maybe_inject_fault()
            r = await litellm.aembedding(model=settings.careflow_embed_model, input=texts,
                                         dimensions=settings.embed_dim, timeout=settings.llm_timeout_s)
            return [d["embedding"] for d in r.data]
        except _TRANSIENT as exc:
            if attempt == attempts - 1:
                raise LLMUnavailable(f"embedding unavailable: {type(exc).__name__}") from exc
            await asyncio.sleep(settings.llm_backoff_base_s * (2**attempt))
    raise AssertionError("unreachable")
