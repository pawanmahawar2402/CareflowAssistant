"""M7 — Circuit breaker with closed / open / half-open states."""
from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, TypeVar

from careflow.config import settings

log = logging.getLogger(__name__)
T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
    """closed → (N consecutive failures) → open → (timeout) → half_open → probe.

    In half-open exactly one probe call is let through: success closes the
    circuit, failure re-opens it.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold or settings.circuit_breaker_failure_threshold
        self.recovery_timeout = (
            recovery_timeout if recovery_timeout is not None else settings.circuit_breaker_recovery_timeout
        )
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.recovery_timeout:
            return "half_open"
        return "open"

    def _on_success(self) -> None:
        if self._opened_at is not None:
            log.info("circuit '%s' closed after successful probe", self.name)
        self._failures = 0
        self._opened_at = None
        self._probe_in_flight = False

    def _on_failure(self) -> None:
        self._failures += 1
        if self._probe_in_flight or self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()
            log.error("circuit '%s' OPEN (failures=%d)", self.name, self._failures)
        self._probe_in_flight = False

    async def call(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        state = self.state
        if state == "open" or (state == "half_open" and self._probe_in_flight):
            raise CircuitOpenError(f"circuit '{self.name}' is {state}")
        if state == "half_open":
            self._probe_in_flight = True
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result


llm_breaker = CircuitBreaker("llm")
