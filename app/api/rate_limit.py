"""Per-caller sliding-window limits (rate and upload quota). In-memory and per process.

Callers are identified only by authenticated API-key names, so the number of tracked keys is bounded by the
configuration. A deployment with several worker processes multiplies these limits by the worker count; enforce the
real limit at a shared gateway or in a shared store when that matters.
"""

from collections import deque
import threading
import time
from typing import Callable


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float, clock: Callable[[], float] = time.monotonic):
        if limit < 1 or window_seconds <= 0:
            raise ValueError("limit and window must be positive")
        self.limit, self.window, self.clock = limit, float(window_seconds), clock
        self._events: dict[str, deque] = {}
        self._lock = threading.Lock()

    def acquire(self, caller: str) -> float:
        """Record one event and return 0.0, or return the seconds to wait if the caller is over the limit."""
        now = self.clock()
        with self._lock:
            events = self._events.setdefault(caller, deque())
            while events and now - events[0] >= self.window:
                events.popleft()
            if len(events) >= self.limit:
                return max(self.window - (now - events[0]), 0.001)
            events.append(now)
            return 0.0
