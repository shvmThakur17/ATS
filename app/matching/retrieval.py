"""JD query units and Pinecone retrieval across resume namespaces.

Each resume lives in its own namespace (M12) and Pinecone queries one namespace at a time, so the
retriever hides that: it embeds each query unit ONCE, then queries every resume namespace with it.
"""

from math import isfinite

from ..parsing.embeddings.models import EmbeddingError
from ..parsing.vector_store import InvalidVectorError, PineconeStore, VectorStoreError
from .models import (EmbeddingDimensionError, JobDescription, MalformedMatchResultError, MatchEvidence, MatchingConfig,
                     NamespaceDiscoveryError, QueryEmbeddingError, QueryType, QueryUnit, RetrievalError)
from .scoring import strength

REQUIRED_METADATA = ("document_id", "chunk_type", "canonical_path", "text")


def build_query_units(jd: JobDescription, config: MatchingConfig) -> list[QueryUnit]:
    """One unit per JD aspect that actually has content; the JD is not embedded as a single blob."""
    def unit(query_type, text):
        return QueryUnit(query_type=query_type, text=text,
                         chunk_types=tuple(config.query_chunk_types.get(query_type.value, ())))

    units = []
    if jd.required_skills:
        units.append(unit(QueryType.REQUIRED_SKILLS,
                          "Required skills: " + ", ".join(s.normalized for s in jd.required_skills)))
    if jd.preferred_skills:
        units.append(unit(QueryType.PREFERRED_SKILLS,
                          "Preferred skills: " + ", ".join(s.normalized for s in jd.preferred_skills)))
    if jd.responsibilities:
        units.append(unit(QueryType.RESPONSIBILITIES, "Responsibilities: " + "; ".join(jd.responsibilities)))
    if jd.experience_requirements:
        units.append(unit(QueryType.EXPERIENCE,
                          "Experience: " + "; ".join(r.text for r in jd.experience_requirements)))
    if jd.education_requirements:
        units.append(unit(QueryType.EDUCATION, "Education: " + "; ".join(jd.education_requirements)))
    return units


class ResumeRetriever:
    """Reuses one embedder and one Pinecone store for the whole matching operation."""

    def __init__(self, store: PineconeStore, embedder, config: MatchingConfig):
        self.store, self.embedder, self.config = store, embedder, config

    def discover_namespaces(self) -> list[str]:
        """Every namespace under the store's resume prefix (`resume:` by default), sorted."""
        prefix = self.store.config.namespace_prefix + ":"
        try:
            stats = self.store.index.describe_index_stats()
        except Exception as error:
            raise NamespaceDiscoveryError(f"could not list namespaces: {type(error).__name__}: {error}") from error
        return sorted(name for name in stats.namespaces if name.startswith(prefix))

    def retrieve(self, units: list[QueryUnit], namespaces: list[str] | None = None):
        """Return (namespaces searched, {document_id: [MatchEvidence...]}). Nothing is filtered by score."""
        namespaces = sorted(namespaces) if namespaces is not None else self.discover_namespaces()
        vectors: dict[str, list[float]] = {}  # query embeddings are reused within one operation
        for unit in units:
            if unit.text not in vectors:
                try:
                    vectors[unit.text] = self.embedder.embed_query(unit.text)
                except EmbeddingError as error:
                    raise QueryEmbeddingError(f"{unit.query_type.value}: {type(error).__name__}: {error}") from error
        evidence: dict[str, list[MatchEvidence]] = {}
        for namespace in namespaces:
            for unit in units:
                for match in self.query(vectors[unit.text], unit, namespace):
                    evidence.setdefault(match.document_id, []).append(match)
        return namespaces, evidence

    def query(self, vector: list[float], unit: QueryUnit, namespace: str) -> list[MatchEvidence]:
        flt = {"chunk_type": {"$in": list(unit.chunk_types)}} if unit.chunk_types else None
        try:
            matches = self.store.query(vector, self.config.top_k, namespace, filter=flt)
        except InvalidVectorError as error:
            raise EmbeddingDimensionError(f"{unit.query_type.value} query vector does not fit the index: {error}") from error
        except VectorStoreError as error:
            raise RetrievalError(f"{unit.query_type.value} query failed in {namespace!r}: {error}") from error
        results = [self.to_evidence(match, unit, namespace) for match in matches]
        return sorted(results, key=lambda e: (-e.similarity_score, e.chunk_id))

    def to_evidence(self, match, unit: QueryUnit, namespace: str) -> MatchEvidence:
        metadata = match.metadata or {}
        missing = [key for key in REQUIRED_METADATA if not isinstance(metadata.get(key), str) or not metadata[key]]
        if not match.id or missing or not isinstance(match.score, float) or not isfinite(match.score):
            raise MalformedMatchResultError(f"match {match.id!r} in {namespace!r} is missing/invalid: "
                                            f"{missing or 'id/score'}")
        if not namespace.endswith(":" + metadata["document_id"]):
            raise MalformedMatchResultError(f"match {match.id!r}: document_id {metadata['document_id']!r} does not "
                                            f"belong to namespace {namespace!r}")
        return MatchEvidence(
            document_id=metadata["document_id"], namespace=namespace, chunk_id=match.id,
            chunk_type=metadata["chunk_type"], canonical_path=metadata["canonical_path"],
            similarity_score=match.score, query_type=unit.query_type, matched_text=metadata["text"],
            evidence_strength=strength(match.score, self.config),
            logical_block_ids=tuple(metadata.get("logical_block_ids", ())),
            page_numbers=tuple(metadata.get("page_numbers", ())), entity_ids=tuple(metadata.get("entity_ids", ())),
            metadata=metadata)
