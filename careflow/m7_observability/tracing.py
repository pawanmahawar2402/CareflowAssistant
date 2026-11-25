"""M7 — Tracing (Langfuse v3+/OTel) and KPI log.

* One trace per request, nested spans per graph node / agent / tool call.
* LLM calls are recorded as *generations* with model, token usage and cost.
* Session and user IDs are attached to the trace.
* A local JSONL KPI log (tokens, cost, latency, fallback/error) is always written,
  so KPIs exist even when Langfuse is not configured.
No prompts containing personal data are sent as span inputs: only metadata.
"""
from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from contextvars import ContextVar
from typing import Any, Iterator

from careflow.config import settings

log = logging.getLogger(__name__)

_lock = threading.Lock()
_current_agent: ContextVar[str] = ContextVar("current_agent", default="unknown")
_lf: Any = None
_lf_init = False


def langfuse_client() -> Any | None:
    """Langfuse client, or None when keys are not configured."""
    global _lf, _lf_init
    if _lf_init:
        return _lf
    _lf_init = True
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        log.info("Langfuse keys not set; tracing limited to the local KPI log")
        return None
    try:
        from langfuse import get_client

        _lf = get_client()
    except Exception as exc:  # pragma: no cover
        log.warning("Langfuse init failed: %s", exc)
    return _lf


def flush() -> None:
    lf = langfuse_client()
    if lf is not None:
        try:
            lf.flush()
        except Exception:  # pragma: no cover
            log.warning("Langfuse flush failed")


def current_agent() -> str:
    return _current_agent.get()


@contextlib.contextmanager
def span(name: str, as_type: str = "span", agent: str | None = None, **metadata: Any) -> Iterator[Any]:
    """Nested span. `as_type` may be span | agent | tool | retriever | guardrail."""
    token = _current_agent.set(agent) if agent else None
    lf = langfuse_client()
    cm = None
    obs = None
    start = time.perf_counter()
    try:
        if lf is not None:
            cm = lf.start_as_current_observation(name=name, as_type=as_type, metadata=metadata or None)
            obs = cm.__enter__()
    except Exception as exc:  # pragma: no cover
        log.warning("span start failed: %s", exc)
        cm = None
    try:
        yield obs
    except Exception as exc:
        if obs is not None:
            with contextlib.suppress(Exception):
                obs.update(level="ERROR", status_message=type(exc).__name__)
        raise
    finally:
        log.debug("span=%s ms=%.1f", name, (time.perf_counter() - start) * 1000)
        if cm is not None:
            with contextlib.suppress(Exception):
                cm.__exit__(None, None, None)
        if token is not None:
            _current_agent.reset(token)


@contextlib.contextmanager
def request_trace(name: str, session_id: str | None, user_id: str | None, **metadata: Any) -> Iterator[Any]:
    """Root span for one request, carrying session_id / user_id."""
    lf = langfuse_client()
    if lf is None:
        yield None
        return
    try:
        from langfuse import propagate_attributes

        with propagate_attributes(session_id=session_id, user_id=user_id):
            with span(name, as_type="span", **metadata) as root:
                yield root
    except Exception as exc:  # pragma: no cover
        log.warning("request_trace degraded: %s", exc)
        yield None
    finally:
        flush()


@contextlib.contextmanager
def generation(name: str, model: str, **metadata: Any) -> Iterator[Any]:
    """Generation observation; caller sets usage via `record_generation`."""
    lf = langfuse_client()
    cm = None
    obs = None
    try:
        if lf is not None:
            cm = lf.start_as_current_observation(
                name=name, as_type="generation", model=model, metadata=metadata or None
            )
            obs = cm.__enter__()
    except Exception as exc:  # pragma: no cover
        log.warning("generation start failed: %s", exc)
        cm = None
    try:
        yield obs
    finally:
        if cm is not None:
            with contextlib.suppress(Exception):
                cm.__exit__(None, None, None)


def record_generation(obs: Any, model: str, usage: dict[str, int], cost_usd: float | None, **meta: Any) -> None:
    if obs is None:
        return
    with contextlib.suppress(Exception):
        obs.update(
            model=model,
            usage_details=usage,
            cost_details={"total": cost_usd} if cost_usd is not None else None,
            metadata=meta or None,
        )


def kpi(event: dict[str, Any]) -> None:
    """Append one KPI event (LLM call or tool call) to the JSONL log."""
    event = {"ts": time.time(), "agent": current_agent(), **event}
    path = settings.runtime_path(settings.kpi_log)
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")
