"""Circuit breaker implementation for API fetchers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int
    reset_timeout: timedelta


class CircuitBreakerOpen(Exception):
    """Raised when downstream calls are blocked."""


class CircuitBreaker:
    """Simple failure counter with half-open reset."""

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._state = "closed"
        self._failures = 0
        self._opened_at: datetime | None = None

    def allow_attempt(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open" and self._opened_at:
            if datetime.now(timezone.utc) - self._opened_at > self._config.reset_timeout:
                self._state = "half-open"
                return True
        return self._state == "half-open"

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._config.failure_threshold:
            self._state = "open"
            self._opened_at = datetime.now(timezone.utc)

    def guard(self) -> None:
        if not self.allow_attempt():
            raise CircuitBreakerOpen("circuit breaker open; refusing upstream call")
