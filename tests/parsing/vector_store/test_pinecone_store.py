"""Unit tests use a fake Pinecone client (no network, no credentials). The integration test at the bottom
talks to the real index and is opt-in: set RUN_PINECONE_INTEGRATION=1. It only touches the namespace
`m12-test:...` and deletes it afterwards."""

import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from app.parsing.chunks import compile_chunks
from app.parsing.embeddings import EmbeddingConfig, EmbeddingRecord, NomicEmbedder
from app.parsing.entities.models import EntityType as E
from app.parsing.sections import SectionType as S
from app.parsing.vector_store import (IndexDimensionMismatchError, IndexMetricMismatchError, InvalidVectorError,
                                      MissingConfigurationError, NamespaceError, PineconeStore, QueryError,
                                      UpsertError, VectorStoreConfig, build_metadata, wait_until)
from tests.parsing.canonical.test_canonical import canon
from tests.parsing.embeddings.test_nomic_embeddings import FakeBackend

DIM = 768
X, C, P = S.EXPERIENCE, S.CONTACT, S.PROJECTS


def make_records(include_contact=False, dimension=DIM):
    resume = canon((E.NAME, "KUKOO RATHEE", S.UNKNOWN, 0), (E.EMAIL, "kukoo@gmail.com", C, 1),
                   (E.PHONE, "8683077463", C, 1), (E.LOCATION, "Mohali", C, 2),
                   (E.DESCRIPTION, "A .NET developer.", S.SUMMARY, 3),
                   (E.ROLE, "Software Engineer", X, 4), (E.COMPANY, "Acme Pvt. Ltd.", X, 5),
                   (E.DATE_RANGE, "Jan 2025 - Present", X, 4),
                   (E.PROJECT_NAME, "Shop Findy (API)", P, 6), (E.PROJECT_DESCRIPTION, "Built APIs.", P, 7),
                   (E.SKILL, "ASP.Net Core", S.SKILLS, 8), (E.LANGUAGE, "English", S.LANGUAGES, 9))
    chunks = compile_chunks(resume, include_contact=include_contact)
    return NomicEmbedder(EmbeddingConfig(), FakeBackend(dimension=dimension)).embed_chunks(chunks)


class FakeIndex:
    def __init__(self):
        self.namespaces = {}
        self.upsert_calls = []

    def upsert(self, vectors, namespace=""):
        self.upsert_calls.append((namespace, [v["id"] for v in vectors]))
        store = self.namespaces.setdefault(namespace, {})
        for v in vectors:
            store[v["id"]] = {"values": v["values"], "metadata": v["metadata"]}

    def fetch(self, ids, namespace=""):
        found = self.namespaces.get(namespace, {})
        return SimpleNamespace(vectors={i: SimpleNamespace(values=found[i]["values"], metadata=found[i]["metadata"])
                                        for i in ids if i in found})

    def query(self, vector, top_k, namespace="", include_metadata=True, filter=None):
        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

        def allowed(metadata):
            for key, want in (filter or {}).items():
                want = want["$eq"] if isinstance(want, dict) else want
                if metadata.get(key) != want:
                    return False
            return True

        scored = [SimpleNamespace(id=i, score=cosine(vector, v["values"]), metadata=v["metadata"])
                  for i, v in self.namespaces.get(namespace, {}).items() if allowed(v["metadata"])]
        return SimpleNamespace(matches=sorted(scored, key=lambda m: -m.score)[:top_k])

    def delete(self, ids=None, delete_all=False, namespace=""):
        if delete_all:
            self.namespaces.pop(namespace, None)
        else:
            for i in ids:
                self.namespaces.get(namespace, {}).pop(i, None)

    def describe_index_stats(self):
        return SimpleNamespace(namespaces={k: SimpleNamespace(vector_count=len(v)) for k, v in self.namespaces.items()})


class FakeClient:
    def __init__(self, exists=True, dimension=DIM, metric="cosine"):
        self.exists, self.dimension, self.metric = exists, dimension, metric
        self.created = []
        self.fake_index = FakeIndex()

    def has_index(self, name):
        return self.exists

    def describe_index(self, name):
        return SimpleNamespace(dimension=self.dimension, metric=self.metric, status={"ready": True})

    def create_index(self, name, dimension, metric, spec):
        self.created.append((name, dimension, metric, spec))
        self.exists, self.dimension, self.metric = True, dimension, metric

    def Index(self, name):
        return self.fake_index


def store(client=None, **config):
    config = VectorStoreConfig(api_key="SECRET-KEY", index_name="test-index", **config)
    s = PineconeStore(config, client or FakeClient())
    s.dimension = DIM
    return s


class StoreTests(unittest.TestCase):
    def test_vector_ids_are_chunk_ids_and_namespace_is_deterministic(self):
        records = make_records()
        s = store()
        result = s.upsert_embeddings(records)
        self.assertEqual(list(result.vector_ids), [r.chunk_id for r in records])
        doc = records[0].metadata["document_id"]
        self.assertEqual(result.namespace, f"resume:{doc}")
        self.assertEqual(s.namespace_for(doc), result.namespace)
        self.assertEqual(store(namespace_prefix="m12-test").namespace_for("abc"), "m12-test:abc")

    def test_namespace_validation(self):
        for bad in ("", " x", "a b", "x" * 200):
            with self.assertRaises(NamespaceError):
                store().fetch(["id"], bad)
        with self.assertRaises(NamespaceError):
            store().namespace_for("")

    def test_metadata_construction_and_text_exact(self):
        records = make_records()
        for record in records:
            m = build_metadata(record)
            self.assertEqual(m["text"], record.text)
            self.assertEqual((m["chunk_id"], m["chunk_type"], m["canonical_path"], m["model_name"], m["embedding_dimension"],
                              m["schema_version"], m["document_id"], m["source_file"]),
                             (record.chunk_id, record.chunk_type, record.canonical_path, record.model_name, DIM,
                              "1.0", record.metadata["document_id"], "fixture.pdf"))
            self.assertEqual(m["page_numbers"], ["0"])
            self.assertEqual(m["logical_block_ids"], list(record.provenance.logical_block_ids))
            self.assertTrue(m["entity_ids"])

    def test_metadata_is_pinecone_compatible(self):
        for record in make_records():
            for key, value in build_metadata(record).items():
                self.assertTrue(isinstance(value, (str, int, float, bool)) or
                                (isinstance(value, list) and all(isinstance(x, str) for x in value)), key)

    def test_pii_excluded_from_metadata(self):
        records = make_records(include_contact=True)
        contact = next(r for r in records if r.chunk_type == "CONTACT")
        self.assertEqual(contact.metadata["email"], "kukoo@gmail.com")  # M10 keeps it ...
        for record in records:  # ... M12 never stores it
            m = build_metadata(record)
            self.assertNotIn("email", m)
            self.assertNotIn("phone", m)
            self.assertNotIn("kukoo@gmail.com", str(m))
            self.assertNotIn("8683077463", str(m))

    def test_dimension_validation(self):
        s = store()
        with self.assertRaises(InvalidVectorError) as context:
            s.upsert_embeddings(make_records(dimension=8))
        self.assertIn("index dimension is 768", str(context.exception))
        self.assertEqual(s.index.upsert_calls, [])

    def test_invalid_vectors_rejected_and_nothing_partially_upserted(self):
        for bad in ([], [float("nan")] * DIM, [float("inf")] * DIM, ["a"] * DIM, [True] * DIM):
            records = make_records()
            records[2] = records[2].model_copy(update={"embedding": bad})
            s = store()
            with self.assertRaises(InvalidVectorError):
                s.upsert_embeddings(records)
            self.assertEqual(s.index.upsert_calls, [])  # validated first, so no partial write

    def test_missing_id_and_duplicates_rejected(self):
        records = make_records()
        with self.assertRaises(InvalidVectorError):
            store().upsert_embeddings([records[0].model_copy(update={"chunk_id": ""})] + records[1:])
        with self.assertRaises(InvalidVectorError):
            store().upsert_embeddings([records[0], records[0]])
        with self.assertRaises(InvalidVectorError):
            store().upsert_embeddings([])

    def test_mixed_documents_rejected(self):
        records = make_records()
        other = records[1].model_copy(update={"metadata": {**records[1].metadata, "document_id": "other"}})
        with self.assertRaises(InvalidVectorError):
            store().upsert_embeddings([records[0], other])

    def test_batch_construction(self):
        records = make_records()
        s = store()
        result = s.upsert_embeddings(records, batch_size=2)
        sizes = [len(ids) for _, ids in s.index.upsert_calls]
        self.assertEqual(len(records), 5)
        self.assertEqual(sizes, [2, 2, 1])
        self.assertEqual(sum(sizes), len(records))
        self.assertEqual((result.upserted, result.failed), (len(records), 0))
        self.assertTrue(all(ns == result.namespace for ns, _ in s.index.upsert_calls))

    def test_idempotent_upsert_replaces(self):
        records = make_records()
        s = store()
        first = s.upsert_embeddings(records)
        s.upsert_embeddings(records)
        self.assertEqual(s.namespace_vector_count(first.namespace), len(records))
        changed = records[0].model_copy(update={"embedding": [1.0] * DIM})
        s.upsert_embeddings([changed] + records[1:])
        self.assertEqual(s.namespace_vector_count(first.namespace), len(records))
        self.assertEqual(s.fetch([records[0].chunk_id], first.namespace)[records[0].chunk_id].values, [1.0] * DIM)

    def test_fetch_and_query(self):
        records = make_records()
        s = store()
        ns = s.upsert_embeddings(records).namespace
        fetched = s.fetch([records[0].chunk_id], ns)
        self.assertEqual(list(fetched), [records[0].chunk_id])
        self.assertEqual(len(fetched[records[0].chunk_id].values), DIM)
        matches = s.query(records[1].embedding, 3, ns)
        self.assertEqual(matches[0].id, records[1].chunk_id)
        self.assertEqual((matches[0].chunk_type, matches[0].canonical_path, matches[0].document_id),
                         (records[1].chunk_type, records[1].canonical_path, records[1].metadata["document_id"]))
        self.assertAlmostEqual(matches[0].score, 1.0, places=5)

    def test_metadata_filter(self):
        records = make_records()
        s = store()
        ns = s.upsert_embeddings(records).namespace
        got = s.query(records[0].embedding, 10, ns, filter={"chunk_type": {"$eq": "WORK_EXPERIENCE"}})
        self.assertTrue(got)
        self.assertTrue(all(m.chunk_type == "WORK_EXPERIENCE" for m in got))
        got = s.query(records[0].embedding, 10, ns, filter={"chunk_type": "SKILLS"})
        self.assertEqual({m.chunk_type for m in got}, {"SKILLS"})

    def test_query_validation(self):
        s = store()
        with self.assertRaises(InvalidVectorError):
            s.query([0.1] * 5, 3, "ns")
        with self.assertRaises(QueryError):
            s.query([0.1] * DIM, 0, "ns")

    def test_namespace_isolation_and_delete(self):
        records = make_records()
        s = store()
        a = s.upsert_embeddings(records).namespace
        other = [r.model_copy(update={"metadata": {**r.metadata, "document_id": "doc-b"}}) for r in records]
        b = s.upsert_embeddings(other).namespace
        self.assertNotEqual(a, b)
        self.assertEqual(s.namespace_vector_count(a), len(records))
        s.delete_namespace(a)
        self.assertEqual(s.fetch([records[0].chunk_id], a), {})
        self.assertEqual(s.namespace_vector_count(b), len(records))  # other resume untouched
        s.delete([records[0].chunk_id], b)
        self.assertEqual(s.namespace_vector_count(b), len(records) - 1)

    def test_deleting_missing_namespace_is_a_noop_but_other_errors_raise(self):
        class NotFoundError(Exception):
            status = 404
        client = FakeClient()
        client.fake_index.delete = lambda **kw: (_ for _ in ()).throw(NotFoundError("Namespace not found"))
        self.assertFalse(store(client).delete_namespace("resume:none"))
        client.fake_index.delete = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        with self.assertRaises(NamespaceError):
            store(client).delete_namespace("resume:none")
        self.assertTrue(store().delete_namespace("resume:fine"))

    def test_default_namespace_is_never_deleted(self):
        with self.assertRaises(NamespaceError):
            store().delete_namespace("")

    def test_upsert_failure_is_explicit(self):
        client = FakeClient()
        client.fake_index.upsert = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        with self.assertRaises(UpsertError):
            store(client).upsert_embeddings(make_records())


class IndexTests(unittest.TestCase):
    def ensure(self, client, **config):
        s = PineconeStore(VectorStoreConfig(api_key="k", index_name="idx", **config), client)
        s.ensure_index(DIM)
        return s

    def test_existing_index_verified_not_recreated(self):
        client = FakeClient()
        s = self.ensure(client)
        self.assertEqual((s.dimension, client.created), (DIM, []))

    def test_dimension_and_metric_mismatch_fail(self):
        with self.assertRaises(IndexDimensionMismatchError):
            self.ensure(FakeClient(dimension=384))
        with self.assertRaises(IndexMetricMismatchError):
            self.ensure(FakeClient(metric="euclidean"))

    def test_missing_index_needs_explicit_cloud_region(self):
        with self.assertRaises(MissingConfigurationError):
            self.ensure(FakeClient(exists=False))
        client = FakeClient(exists=False)
        self.ensure(client, cloud="aws", region="us-east-1")
        self.assertEqual([(n, d, m) for n, d, m, _ in client.created], [("idx", DIM, "cosine")])

    def test_config_from_env(self):
        keys = ("PINECONE_API_KEY", "PINECONE_INDEX_NAME", "PINECONE_NAMESPACE")
        old = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ.pop(k, None)
            with self.assertRaises(MissingConfigurationError):
                VectorStoreConfig.from_env()
            os.environ["PINECONE_API_KEY"] = "SECRET-KEY"
            with self.assertRaises(MissingConfigurationError):
                VectorStoreConfig.from_env()
            os.environ["PINECONE_INDEX_NAME"] = "idx"
            config = VectorStoreConfig.from_env()
            self.assertEqual((config.index_name, config.namespace_prefix, config.metric), ("idx", "resume", "cosine"))
            self.assertNotIn("SECRET-KEY", repr(config))
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_wait_until(self):
        self.assertTrue(wait_until(lambda: True, timeout=0.1))
        self.assertFalse(wait_until(lambda: False, timeout=0.05, interval=0.01))


@unittest.skipUnless(os.environ.get("RUN_PINECONE_INTEGRATION") == "1",
                     "real Pinecone integration test: set RUN_PINECONE_INTEGRATION=1 (uses .env credentials)")
class RealPineconeIntegrationTests(unittest.TestCase):
    PREFIX = "m12-test"

    def test_real_roundtrip(self):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
        config = VectorStoreConfig.from_env()
        config = VectorStoreConfig(api_key=config.api_key, index_name=config.index_name,
                                   namespace_prefix=self.PREFIX, cloud=config.cloud, region=config.region)
        s = PineconeStore(config)
        s.ensure_index(DIM)
        base = make_records()[:3]
        records = []
        for n, record in enumerate(base):  # known one-hot vectors: cosine(e_n, e_n) == 1
            vector = [0.0] * DIM
            vector[n] = 1.0
            records.append(record.model_copy(update={"embedding": vector, "metadata": {
                **record.metadata, "document_id": "integration"}}))
        namespace = s.namespace_for("integration")
        self.assertTrue(namespace.startswith(f"{self.PREFIX}:"))
        try:
            s.delete_namespace(namespace)
            result = s.upsert_embeddings(records)
            ids = [r.chunk_id for r in records]
            fetched = wait_until(lambda: (f := s.fetch(ids, namespace)) and len(f) == len(ids) and f, 60)
            self.assertEqual(sorted(fetched), sorted(ids))
            self.assertEqual(fetched[ids[1]].metadata["text"], records[1].text)
            self.assertEqual(len(fetched[ids[1]].values), DIM)
            top = wait_until(lambda: (m := s.query(records[1].embedding, 3, namespace)) and m[0].id == ids[1] and m, 60)
            self.assertEqual(top[0].id, ids[1])
            self.assertEqual(top[0].document_id, "integration")
            self.assertEqual(result.upserted, 3)
            self.assertEqual(s.query(records[1].embedding, 3, namespace + "-other"), [])  # isolation
        finally:
            s.delete_namespace(namespace)


if __name__ == "__main__":
    unittest.main()
