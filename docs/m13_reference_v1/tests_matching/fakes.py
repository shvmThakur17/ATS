"""Deterministic fakes for M13 unit tests: a hashed bag-of-words embedder and an in-memory Pinecone."""

import hashlib
import math
import re
from types import SimpleNamespace

from app.parsing.chunks import compile_chunks
from app.parsing.embeddings import EmbeddingConfig, NomicEmbedder
from app.parsing.entities.models import EntityType as E
from app.parsing.sections import SectionType as S
from app.parsing.vector_store import PineconeStore, VectorStoreConfig
from tests.parsing.canonical.test_canonical import canon
from tests.parsing.vector_store.test_pinecone_store import FakeClient, FakeIndex

DIM = 768
PREFIXES = ("search_document: ", "search_query: ")
DASH = "–"


class HashBackend:
    """Token-overlap embedding: texts sharing words get higher cosine similarity. No model needed."""
    dimension = DIM

    def encode(self, texts):
        vectors = []
        for text in texts:
            for prefix in PREFIXES:
                text = text.removeprefix(prefix)
            vector = [0.0] * DIM
            for token in re.findall(r"[a-z0-9#.+]+", text.lower()):
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % DIM] += 1.0
            vectors.append(vector)
        return vectors


class CountingEmbedder(NomicEmbedder):
    def __init__(self):
        super().__init__(EmbeddingConfig(), HashBackend())
        self.query_texts = []

    def embed_query(self, text):
        self.query_texts.append(text)
        return super().embed_query(text)


class InIndex(FakeIndex):
    """FakeIndex with $in / $eq filters."""

    def query(self, vector, top_k, namespace="", include_metadata=True, filter=None):
        def cosine(a, b):
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
            return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0

        def allowed(metadata):
            for key, want in (filter or {}).items():
                if isinstance(want, dict) and "$in" in want:
                    if metadata.get(key) not in want["$in"]:
                        return False
                elif metadata.get(key) != (want["$eq"] if isinstance(want, dict) else want):
                    return False
            return True

        scored = [SimpleNamespace(id=i, score=cosine(vector, v["values"]), metadata=v["metadata"])
                  for i, v in self.namespaces.get(namespace, {}).items() if allowed(v["metadata"])]
        return SimpleNamespace(matches=sorted(scored, key=lambda m: (-m.score, m.id))[:top_k])


def make_store(prefix="resume"):
    client = FakeClient()
    client.fake_index = InIndex()
    store = PineconeStore(VectorStoreConfig(api_key="k", index_name="idx", namespace_prefix=prefix), client)
    store.dimension = DIM
    return store


X, C, ED, P = S.EXPERIENCE, S.CONTACT, S.EDUCATION, S.PROJECTS

CANDIDATE_A = (  # .NET developer, Kukoo-like: SQL Server only appears in text, REST APIs/Docker/Azure absent
    (E.NAME, "Alice Dotnet", S.UNKNOWN, 0), (E.EMAIL, "alice@example.com", C, 1), (E.PHONE, "8683077463", C, 1),
    (E.DESCRIPTION, ".NET Developer with 2+ years of experience building web applications using C#, ASP.NET Core "
                    "and SQL Server.", S.SUMMARY, 2),
    (E.ROLE, "Software Engineer", X, 3), (E.DATE_RANGE, f"July 2022 {DASH} Jan 2025", X, 3),
    (E.COMPANY, "Ginlytics", X, 4),
    (E.ROLE, "Software Engineer", X, 5), (E.DATE_RANGE, f"Jan 2025{DASH} Present", X, 5), (E.COMPANY, "OffBeat", X, 6),
    (E.PROJECT_NAME, "Shop Findy (API)", P, 7),
    (E.PROJECT_DESCRIPTION, "Developed Web APIs in a layered architecture, using SQL Server for data operations.", P, 8),
    (E.DEGREE, "DIPLOMA", ED, 9), (E.FIELD_OF_STUDY, "CS", ED, 9), (E.INSTITUTION, "Govt. Polytechnic", ED, 10),
    (E.SKILL, "HTML", S.SKILLS, 11), (E.SKILL, "ASP.Net Core", S.SKILLS, 11), (E.SKILL, "C#", S.SKILLS, 11),
    (E.SKILL, "SQL", S.SKILLS, 12), (E.SKILL, "Web APIs", S.SKILLS, 12),
)
CANDIDATE_B = (  # deliberately different: Python/cloud engineer
    (E.NAME, "Bob Python", S.UNKNOWN, 0), (E.EMAIL, "bob@example.com", C, 1),
    (E.DESCRIPTION, "Backend engineer building Python services, REST APIs and containerized deployments on Azure.",
     S.SUMMARY, 2),
    (E.ROLE, "Backend Engineer", X, 3), (E.DATE_RANGE, "Jan 2019 - Jun 2023", X, 3), (E.COMPANY, "CloudCo", X, 4),
    (E.DESCRIPTION, "Designed and maintained REST APIs and Docker based deployments.", X, 4),
    (E.PROJECT_NAME, "Pipeline", P, 5), (E.PROJECT_DESCRIPTION, "Built data pipelines with Python and Docker.", P, 6),
    (E.TECHNOLOGY, "Kubernetes", P, 6),
    (E.DEGREE, "B.Tech", ED, 7), (E.INSTITUTION, "State University", ED, 8),
    (E.SKILL, "Python", S.SKILLS, 9), (E.SKILL, "Docker", S.SKILLS, 9), (E.SKILL, "Azure", S.SKILLS, 9),
    (E.SKILL, "REST APIs", S.SKILLS, 9),
)


def build_candidate(document_id, source_file, specs):
    """specs -> (canonical resume with its own document_id, chunks, embedding records)."""
    resume = canon(*specs)
    resume = resume.model_copy(update={"metadata": resume.metadata.model_copy(
        update={"document_id": document_id, "source_file_name": source_file})})
    chunks = compile_chunks(resume)
    records = NomicEmbedder(EmbeddingConfig(), HashBackend()).embed_chunks(chunks)
    return resume, chunks, records


def ingest(store, records):
    return store.upsert_embeddings(records).namespace


def two_candidates(store=None):
    store = store or make_store()
    a, _, a_records = build_candidate("doc-a", "a.pdf", CANDIDATE_A)
    b, _, b_records = build_candidate("doc-b", "b.pdf", CANDIDATE_B)
    ns_a, ns_b = ingest(store, a_records), ingest(store, b_records)
    return store, {"doc-a": a, "doc-b": b}, (ns_a, ns_b)
