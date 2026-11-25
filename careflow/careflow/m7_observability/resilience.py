"""M7 — Circuit breaker + LangFuse tracing (graceful no-op when keys are absent)."""
from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, TypeVar

from careflow.config import settings

log = logging.getLogger(__name__)
T = TypeVar("T")


# ── Circuit breaker ──────────────────────────────────────────────────────────

class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
    """Classic closed → open → half-open breaker for flaky downstream calls."""

    def __init__(
        self,
        name: str,
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold or settings.circuit_breaker_failure_threshold
        self.recovery_timeout = recovery_timeout or settings.circuit_breaker_recovery_timeout
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.recovery_timeout:
            return "half_open"
        return "open"

    def _on_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def _on_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold or self.state == "half_open":
            self._opened_at = time.monotonic()
            log.error("Circuit '%s' OPEN after %d failures", self.name, self._failures)

    async def call(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        if self.state == "open":
            raise CircuitOpenError(f"Circuit '{self.name}' is open")
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result


def with_breaker(breaker: CircuitBreaker):
    """Decorator form for async functions."""
    def deco(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*a: Any, **kw: Any) -> T:
            return await breaker.call(fn, *a, **kw)
        return wrapper
    return deco


llm_breaker = CircuitBreaker("llm")


# ── LangFuse tracing ─────────────────────────────────────────────────────────

_client: Any = None
_init_done = False


def get_langfuse() -> Any | None:
    """Return a LangFuse client, or None when keys are not configured."""
    global _client, _init_done
    if _init_done:
        return _client
    _init_done = True
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        log.info("LangFuse keys not set — tracing disabled")
        return None
    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:  # pragma: no cover
        log.warning("LangFuse init failed: %s", exc)
    return _client


def enable_litellm_callbacks() -> None:
    """Send every LiteLLM call (tokens, latency, cost) to LangFuse."""
    if get_langfuse() is None:
        return
    import litellm
    if "langfuse" not in litellm.success_callback:
        litellm.success_callback.append("langfuse")
        litellm.failure_callback.append("langfuse")


@contextmanager
def trace_span(name: str, **metadata: Any):
    """Span: always logs duration; records to LangFuse (v3 or v2) when enabled."""
    start = time.perf_counter()
    lf = get_langfuse()
    cm = None
    try:
        if lf is not None and hasattr(lf, "start_as_current_span"):  # v3
            cm = lf.start_as_current_span(name=name, metadata=metadata)
            cm.__enter__()
        elif lf is not None and hasattr(lf, "trace"):  # v2
            lf.trace(name=name, metadata=metadata)
    except Exception as exc:  # pragma: no cover
        log.warning("LangFuse span start failed: %s", exc)
        cm = None
    try:
        yield None
    finally:
        ms = (time.perf_counter() - start) * 1000
        log.info("span=%s duration_ms=%.1f %s", name, ms, metadata)
        try:
            if cm is not None:
                cm.__exit__(None, None, None)
            if lf is not None:
                lf.flush()
        except Exception:  # pragma: no cover
            pass

