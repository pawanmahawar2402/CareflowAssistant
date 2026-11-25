"""M7 — Resilience and tracing entry points used by the API and the workflow.

The circuit breaker lives in `careflow.m7_observability.breaker` (one instance, `llm_breaker`, shared by
the LLM layer and `/health`). Tracing lives in `careflow.m7_observability.tracing`; this module only
exposes the names the API and the workflow import.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from careflow.m7_observability import tracing
from careflow.m7_observability.breaker import CircuitBreaker, CircuitOpenError, llm_breaker

__all__ = ["CircuitBreaker", "CircuitOpenError", "llm_breaker", "enable_litellm_callbacks", "trace_span"]


def enable_litellm_callbacks() -> None:
    """Send every LiteLLM call (tokens, latency, cost) to LangFuse when keys are configured."""
    if tracing.langfuse_client() is None:
        return
    import litellm

    if "langfuse" not in litellm.success_callback:
        litellm.success_callback.append("langfuse")
        litellm.failure_callback.append("langfuse")


@contextmanager
def trace_span(name: str, **metadata: Any) -> Iterator[Any]:
    """Nested span for a request-level step (delegates to the shared tracing module)."""
    with tracing.span(name, as_type="span", **metadata) as obs:
        yield obs
