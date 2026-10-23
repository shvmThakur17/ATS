"""Explicit readiness for Pinecone data (H3). Reads only; nothing is written or deleted.

Pinecone reads are eventually consistent, and the vector count N1 sizes retrieval from comes from index statistics,
which can lag. A zero or too-small count used to produce a successful but different ranking. Now:

* Upload waits (bounded) until the namespace count reaches the number of chunks just written; if it does not, the
  response says PENDING_INDEXING and the server remembers the expected count for that document.
* Match makes ONE statistics call for all candidates, then refuses to rank any candidate that is not provably ready:
  count 0, count below the remembered expected count, count above the retrieval cap, or a probe query that returns
  more vectors than the statistics claim. The caller gets a retryable error naming the candidate positions.

The expected-count memory is in-process only. After a restart a document is judged by the count and probe checks
alone, which cannot detect an index that is consistently behind by the same amount; that residual risk is reported.
"""

from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Callable, Sequence

READY, EMPTY, STALE, TOO_LARGE = "READY", "EMPTY", "STALE", "TOO_LARGE"
PROBE_LIMIT = 1000                       # Pinecone's top_k ceiling for queries that return metadata


class ReadinessUnavailableError(Exception):
    """The index could not be read, so readiness is neither confirmed nor shown to be pending. Nothing is assumed:
    callers must not rank, and must not report the data as ready or as "still indexing". Only the exception TYPE of
    the underlying provider failure is kept (`reason`); provider messages and payloads are discarded."""

    def __init__(self, reason: str = "unknown"):
        super().__init__("readiness unavailable")
        self.reason = reason


def provider_call(function, *arguments):
    """Run one read against the index; any failure becomes ReadinessUnavailableError (fail closed, safe detail)."""
    try:
        return function(*arguments)
    except Exception as error:
        raise ReadinessUnavailableError(type(error).__name__) from None


class CandidatesNotReadyError(Exception):
    def __init__(self, statuses: Sequence["NamespaceStatus"]):
        super().__init__("candidates not ready")
        self.statuses = list(statuses)

    def indexes(self, state: str | None = None) -> list[int]:
        return [i for i, s in enumerate(self.statuses) if s.state != READY and (state is None or s.state == state)]


@dataclass(frozen=True)
class NamespaceStatus:
    document_id: str
    namespace: str
    vector_count: int
    state: str


class ReadinessTracker:
    """Bounded LRU map document_id -> number of chunks the last successful ingestion wrote."""

    def __init__(self, max_entries: int = 10_000):
        self._entries: OrderedDict[str, int] = OrderedDict()
        self._max, self._lock = max_entries, threading.Lock()

    def record(self, document_id: str, chunk_count: int) -> None:
        with self._lock:
            self._entries[document_id] = chunk_count
            self._entries.move_to_end(document_id)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def expected(self, document_id: str) -> int | None:
        with self._lock:
            return self._entries.get(document_id)


def read_counts(store, namespaces: Sequence[str]) -> dict[str, int]:
    """One describe_index_stats call for every namespace asked about (missing namespaces count as 0)."""
    stats = store.index.describe_index_stats()
    found = stats.namespaces
    return {ns: int(found[ns].vector_count) if ns in found else 0 for ns in namespaces}


def probe_vector(store) -> list[float]:
    dimension = store.dimension if store.dimension is not None else store.describe_index().dimension
    return [1.0] + [0.0] * (dimension - 1)


class ReadinessGate:
    def __init__(self, tracker: ReadinessTracker, max_retrieval_depth: int):
        self.tracker, self.max_depth = tracker, max_retrieval_depth

    def verify(self, store, refs) -> dict[str, int]:
        """Blocking. Returns the verified {namespace: count} map or raises CandidatesNotReadyError."""
        counts = provider_call(read_counts, store, [r.namespace for r in refs])    # a failed read never ranks anyone
        statuses = [self.assess(store, r, counts[r.namespace]) for r in refs]
        if any(s.state != READY for s in statuses):
            raise CandidatesNotReadyError(statuses)
        return counts

    def assess(self, store, ref, count: int) -> NamespaceStatus:
        def status(state):
            return NamespaceStatus(ref.document_id, ref.namespace, count, state)
        if count <= 0:
            return status(EMPTY)
        if count > self.max_depth:
            return status(TOO_LARGE)
        expected = self.tracker.expected(ref.document_id)
        if expected is not None and count < expected:
            return status(STALE)
        if count < PROBE_LIMIT:                                      # does the index hold more than the stats claim?
            visible = provider_call(lambda: len(store.query(probe_vector(store), count + 1, ref.namespace)))
            if visible > count:
                return status(STALE)
        return status(READY)

    async def wait_until_indexed(self, runner, store, namespace: str, expected: int, timeout: float, interval: float,
                                 sleep: Callable, clock: Callable[[], float]) -> bool:
        """Poll the namespace count (off the event loop) until it reaches `expected` or the bound passes.

        True  -> the count reached `expected`: ready.
        False -> at least one read succeeded and showed fewer vectors than expected for the whole wait: pending.
        Raises ReadinessUnavailableError when no read ever succeeded: nothing was observed, so "pending" would be a
        guess. A read that fails is retried until the deadline, so a brief blip does not decide the outcome."""
        deadline = clock() + timeout
        observed, last_error = False, "unknown"
        while True:
            try:
                count = await runner.run(lambda: read_counts(store, [namespace])[namespace])
                observed = True
                if count >= expected:
                    return True
            except Exception as error:                              # provider failure, or our own runner refusing
                last_error = type(error).__name__
            if clock() >= deadline:
                if observed:
                    return False
                raise ReadinessUnavailableError(last_error)
            await sleep(interval)
