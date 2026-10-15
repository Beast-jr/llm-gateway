"""A tiny circuit breaker for Groq, kept in memory.

After BREAKER_FAILURES failures in a row we stop trying Groq for BREAKER_COOLDOWN_SECONDS (so users
don't wait for a provider that is clearly down). When the cooldown ends, the next request tries Groq
again: one success resets everything, one more failure closes the door again straight away.
"""
import time

import config

_now = time.monotonic  # tests replace this to move the clock without waiting


class CircuitBreaker:
    def __init__(self) -> None:
        self.failures = 0
        self.open_until = 0.0

    def is_open(self) -> bool:
        """True while we should skip the provider."""
        return _now() < self.open_until

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= config.BREAKER_FAILURES:
            self.open_until = _now() + config.BREAKER_COOLDOWN_SECONDS

    def record_success(self) -> None:
        self.failures = 0
        self.open_until = 0.0


groq_breaker = CircuitBreaker()
