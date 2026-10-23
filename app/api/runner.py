"""Bounded background execution and thread-safe lazy services.

Replaces "global lock + the framework's 40-thread pool". Each runner owns its OWN small executor, so waiting jobs
can never exhaust the pool that FastAPI uses for its own synchronous work, and the number of queued plus running
jobs is capped: when the cap is reached new work is refused immediately (load shedding) instead of piling up.
The cap counts a job until its thread actually finishes, even if the caller already gave up on it.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading


class ServerBusyError(Exception):
    """The runner is at its pending-job cap."""


class JobTimeoutError(Exception):
    """The caller's wait ended before the job finished. A job already running cannot be interrupted."""


class JobRunner:
    def __init__(self, name: str, workers: int, max_pending: int, timeout_seconds: float):
        if workers < 1 or max_pending < workers or timeout_seconds <= 0:
            raise ValueError("invalid runner limits")
        self.name, self.max_pending, self.timeout = name, max_pending, float(timeout_seconds)
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"api-{name}")
        self._pending, self._lock = 0, threading.Lock()

    @property
    def pending(self) -> int:
        return self._pending

    def _finished(self, _future) -> None:
        with self._lock:
            self._pending -= 1

    async def run(self, function, *arguments, timeout: float | None = None):
        with self._lock:
            if self._pending >= self.max_pending:
                raise ServerBusyError(self.name)
            self._pending += 1
        try:
            future = self._executor.submit(function, *arguments)
        except BaseException:
            self._finished(None)
            raise
        future.add_done_callback(self._finished)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout or self.timeout)
        except asyncio.TimeoutError:
            future.cancel()                                  # only effective while the job is still queued
            raise JobTimeoutError(self.name) from None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


_UNSET = object()


class LazyService:
    """Built on first use, exactly once, under a lock. A failed build is not cached, so the next request retries."""

    def __init__(self, builder):
        self._builder, self._value, self._lock = builder, _UNSET, threading.Lock()

    @property
    def built(self) -> bool:
        return self._value is not _UNSET

    def get_blocking(self):
        if self._value is not _UNSET:
            return self._value
        with self._lock:
            if self._value is _UNSET:
                self._value = self._builder()
            return self._value

    async def get(self, runner: JobRunner):
        if self._value is not _UNSET:
            return self._value
        return await runner.run(self.get_blocking)           # the (slow) build never runs on the event loop


async def resolve(service, runner: JobRunner):
    """A LazyService is built on demand; any other object (for example a test fake) is returned as it is."""
    return await service.get(runner) if isinstance(service, LazyService) else service
