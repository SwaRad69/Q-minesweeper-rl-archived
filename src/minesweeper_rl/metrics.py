"""Lightweight, opt-in latency telemetry for the browser environment.

The environment only collects timings when :attr:`MinesweeperEnvironment.metrics`
is set, so normal use pays no cost.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Latency:
    """Running latency statistics plus raw samples for percentiles."""

    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0
    samples: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.count += 1
        self.total_s += seconds
        if seconds > self.max_s:
            self.max_s = seconds
        self.samples.append(seconds)

    @property
    def mean_ms(self) -> float:
        return self.total_s / self.count * 1000.0 if self.count else 0.0

    @property
    def max_ms(self) -> float:
        return self.max_s * 1000.0

    @property
    def median_ms(self) -> float:
        return self._percentile(0.5)

    @property
    def p95_ms(self) -> float:
        return self._percentile(0.95)

    def _percentile(self, fraction: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return ordered[index] * 1000.0


@dataclass
class EnvironmentMetrics:
    """Timings for the environment's observable operations."""

    reset: Latency = field(default_factory=Latency)
    step: Latency = field(default_factory=Latency)
    observe: Latency = field(default_factory=Latency)
    dom_read: Latency = field(default_factory=Latency)
    parse: Latency = field(default_factory=Latency)
    click: Latency = field(default_factory=Latency)
    clicks: int = 0
    event_fallbacks: int = 0


@contextmanager
def timed(recorder: Latency | None) -> Iterator[None]:
    """Time a block into ``recorder`` if it is not ``None``."""

    if recorder is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        recorder.add(time.perf_counter() - started)
