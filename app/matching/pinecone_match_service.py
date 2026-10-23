"""Skills-only matching over explicitly supplied Pinecone candidates (N1).

A thin wrapper around the frozen M13 `JDMatcher`: it sizes retrieval to each candidate's namespace, rebuilds the
minimal canonical resume from Pinecone metadata, and calls M13 with an explicit namespace list. Ranking is M13's
and is not touched here. Read-only: it never writes to or deletes from Pinecone and never discovers namespaces.
"""

from dataclasses import dataclass, replace
import os
from typing import Mapping, Sequence

from ..parsing.vector_store import PineconeStore
from .jd_matcher import JDMatcher
from .models import FrozenModel, InvalidConfigError, JobDescription, MatchResult, MatchingConfig, MatchingError
from .pinecone_canonical_adapter import PineconeCanonicalAdapter

DEFAULT_MAX_RETRIEVAL_DEPTH = 200
PINECONE_MAX_TOP_K_WITH_METADATA = 1000    # Pinecone's documented limit for queries that return metadata


class NamespaceCountUnavailableError(MatchingError):
    """The namespace's vector count could not be verified; no depth is guessed."""


class RetrievalDepthExceededError(MatchingError):
    """A namespace has more vectors than the configured cap, so its evidence cannot be retrieved completely."""


class DuplicateCandidateError(MatchingError):
    """The same document id or namespace was supplied twice."""


@dataclass(frozen=True)
class PineconeMatchLimits:
    """Wrapper-level settings (kept out of `MatchingConfig`, which is frozen M13)."""
    max_retrieval_depth: int = DEFAULT_MAX_RETRIEVAL_DEPTH   # most vectors a single candidate namespace may hold

    def __post_init__(self):
        if isinstance(self.max_retrieval_depth, bool) or not isinstance(self.max_retrieval_depth, int) \
                or not 1 <= self.max_retrieval_depth <= PINECONE_MAX_TOP_K_WITH_METADATA:
            raise InvalidConfigError(f"max_retrieval_depth must be an integer within "
                                     f"[1, {PINECONE_MAX_TOP_K_WITH_METADATA}]")

    @classmethod
    def from_env(cls) -> "PineconeMatchLimits":
        raw = os.environ.get("M13_MAX_RETRIEVAL_DEPTH")
        try:
            return cls() if raw in (None, "") else cls(max_retrieval_depth=int(raw))
        except ValueError:
            raise InvalidConfigError(f"M13_MAX_RETRIEVAL_DEPTH must be an integer, got {raw!r}") from None


@dataclass(frozen=True)
class CandidateRef:
    """One resume to rank: its document id and the namespace that holds its vectors."""
    document_id: str
    namespace: str


class NamespaceDepth(FrozenModel):
    document_id: str
    namespace: str
    vector_count: int


class PineconeMatchOutcome(FrozenModel):
    result: MatchResult                    # M13's result, unchanged
    effective_top_k: int                   # retrieval depth used for this call only
    max_retrieval_depth: int
    namespaces: tuple[NamespaceDepth, ...]  # verified vector count per candidate, in the order supplied


class PineconeMatchService:
    def __init__(self, store: PineconeStore, embedder, config: MatchingConfig | None = None,
                 limits: PineconeMatchLimits | None = None):
        self.store, self.embedder = store, embedder
        self.config = config or MatchingConfig()          # never mutated: each call derives its own copy
        self.limits = limits or PineconeMatchLimits()
        self.adapter = PineconeCanonicalAdapter(store)

    def match(self, jd: JobDescription, candidates: Sequence[CandidateRef],
              vector_counts: Mapping[str, int] | None = None) -> PineconeMatchOutcome:
        """`vector_counts` lets a caller that already read the index statistics (one call for all candidates) pass
        them in; a namespace missing from it is looked up individually as before."""
        refs = self.validate(candidates)
        known = vector_counts or {}
        depths = tuple(NamespaceDepth(document_id=r.document_id, namespace=r.namespace,
                                      vector_count=self.checked_count(r.namespace, known[r.namespace])
                                      if r.namespace in known else self.vector_count(r.namespace)) for r in refs)
        for depth in depths:                               # every count is checked before anything is retrieved
            if depth.vector_count > self.limits.max_retrieval_depth:
                raise RetrievalDepthExceededError(
                    f"namespace {depth.namespace!r} holds {depth.vector_count} vectors, over the cap of "
                    f"{self.limits.max_retrieval_depth}; matching would silently omit evidence")
        canonical = {d.document_id: self.adapter.rebuild(d.document_id, d.namespace, d.vector_count) for d in depths}
        effective = max([d.vector_count for d in depths], default=0) or 1       # top_k must be >= 1
        request_config = replace(self.config, top_k=effective)                  # request-scoped copy
        result = JDMatcher(self.store, self.embedder, request_config).match(
            jd, canonical, [d.namespace for d in depths])                       # explicit list: no discovery
        return PineconeMatchOutcome(result=result, effective_top_k=effective,
                                    max_retrieval_depth=self.limits.max_retrieval_depth, namespaces=depths)

    def validate(self, candidates: Sequence[CandidateRef]) -> list[CandidateRef]:
        refs = list(candidates)
        for ref in refs:
            self.adapter.check_namespace(ref.document_id, ref.namespace)
        for label, values in (("document id", [r.document_id for r in refs]), ("namespace", [r.namespace for r in refs])):
            if len(set(values)) != len(values):
                raise DuplicateCandidateError(f"duplicate {label} in the candidate list")
        return refs

    def vector_count(self, namespace: str) -> int:
        try:
            count = self.store.namespace_vector_count(namespace)
        except Exception as error:     # store errors and anything the stats call raises: never guess a depth
            raise NamespaceCountUnavailableError(
                f"could not verify the vector count of {namespace!r}: {type(error).__name__}: {error}") from error
        return self.checked_count(namespace, count)

    @staticmethod
    def checked_count(namespace: str, count) -> int:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise NamespaceCountUnavailableError(f"vector count of {namespace!r} is not a usable number: {count!r}")
        return count
