"""Unit tests use a fake backend (no model, no network). The real-model test at the bottom is an
opt-in integration test: set RUN_NOMIC_INTEGRATION=1 to run it. It is never faked."""

import math
import os
import unittest

from app.parsing.chunks import compile_chunks
from app.parsing.embeddings import (EmbeddingBatchError, EmbeddingConfig, EmptyEmbeddingError, EmptyTextError,
                                    IncompatibleModelError, InconsistentDimensionError, InvalidEmbeddingError,
                                    NomicEmbedder)
from app.parsing.embeddings.nomic_embedder import resolve_device
from app.parsing.entities.models import EntityType as E
from app.parsing.sections import SectionType as S
from tests.parsing.canonical.test_canonical import canon

DOC, QUERY = "search_document: ", "search_query: "


class FakeBackend:
    """Deterministic stand-in: the vector depends only on the text it receives."""

    def __init__(self, dimension=4, documented=None, output=None, error=None):
        self.dimension = documented
        self.size = dimension
        self.calls = []
        self.output = output
        self.error = error

    def encode(self, texts):
        self.calls.append(list(texts))
        if self.error:
            raise self.error
        if self.output is not None:
            return self.output
        return [[float(sum(map(ord, t)) % 97 + i) for i in range(self.size)] for t in texts]


def embedder(backend=None, **config):
    return NomicEmbedder(EmbeddingConfig(**config), backend or FakeBackend())


def chunks():
    return compile_chunks(canon((E.PROJECT_NAME, "Shop Findy (API)", S.PROJECTS, 0),
                                (E.PROJECT_DESCRIPTION, "Developed RESTful APIs.", S.PROJECTS, 1),
                                (E.PROJECT_NAME, "Baskin Robbins – Online Ordering", S.PROJECTS, 2),
                                (E.SKILL, "ASP.Net Core", S.SKILLS, 3), (E.LANGUAGE, "English", S.LANGUAGES, 4)))


class EmbedderTests(unittest.TestCase):
    def test_initialization(self):
        e = embedder(FakeBackend(documented=4))
        self.assertEqual((e.model_name, e.documented_dimension, e.dimension, e.device),
                         ("nomic-ai/nomic-embed-text-v1.5", 4, None, "cpu"))

    def test_single_text_and_dimension_detected_from_output(self):
        e = embedder()
        (vector,) = e.embed_documents(["hello"])
        self.assertEqual((len(vector), e.dimension), (4, 4))
        self.assertTrue(all(isinstance(v, float) for v in vector))

    def test_batch_consistent_finite(self):
        e = embedder(batch_size=2)
        vectors = e.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])
        self.assertEqual(len(vectors), 5)
        self.assertEqual({len(v) for v in vectors}, {4})
        self.assertTrue(all(math.isfinite(x) for v in vectors for x in v))

    def test_empty_input(self):
        backend = FakeBackend()
        self.assertEqual(embedder(backend).embed_documents([]), [])
        self.assertEqual(backend.calls, [])
        for text in ("", "   \n"):
            with self.assertRaises(EmptyTextError):
                embedder().embed_documents([text])

    def test_chunk_records_preserve_identity_order_metadata_provenance(self):
        source = chunks()
        records = embedder().embed_chunks(source)
        self.assertEqual(len(records), len(source))
        for chunk, record in zip(source, records):
            self.assertEqual((record.chunk_id, record.chunk_type, record.canonical_path, record.source_record_id),
                             (chunk.chunk_id, chunk.chunk_type.value, chunk.canonical_path, chunk.source_record_id))
            self.assertEqual(record.metadata, chunk.metadata)
            self.assertEqual(record.provenance, chunk.provenance)
            self.assertEqual((record.model_name, record.dimension, len(record.embedding)),
                             ("nomic-ai/nomic-embed-text-v1.5", 4, 4))

    def test_chunk_text_is_unchanged_and_only_the_model_prefix_is_added(self):
        source = chunks()
        backend = FakeBackend()
        records = embedder(backend, batch_size=100).embed_chunks(source)
        sent = [t for call in backend.calls for t in call]
        self.assertEqual(sent, [DOC + c.text for c in source])
        self.assertEqual([r.text for r in records], [c.text for c in source])
        self.assertIn("–", records[1].text)  # not normalized

    def test_no_random_preprocessing_same_input_same_vectors(self):
        e = embedder()
        source = chunks()
        self.assertEqual([r.embedding for r in e.embed_chunks(source)], [r.embedding for r in e.embed_chunks(source)])

    def test_query_uses_query_prefix(self):
        backend = FakeBackend()
        embedder(backend).embed_query("find .NET devs")
        self.assertEqual(backend.calls, [[QUERY + "find .NET devs"]])

    def test_model_reuse_and_batching(self):
        backend = FakeBackend()
        e = embedder(backend, batch_size=2)
        e.embed_chunks(chunks())
        self.assertEqual([len(c) for c in backend.calls], [2, 2])  # one backend, batches of 2
        self.assertIs(e.backend, backend)

    def test_cpu_and_device_resolution(self):
        self.assertEqual(embedder().device, "cpu")
        self.assertEqual(resolve_device("cpu"), "cpu")
        self.assertIn(resolve_device("auto"), ("cpu", "cuda"))

    def test_config_from_env(self):
        old = {k: os.environ.get(k) for k in ("NOMIC_EMBEDDING_MODEL", "NOMIC_EMBEDDING_DEVICE", "NOMIC_EMBEDDING_BATCH_SIZE")}
        try:
            os.environ.update(NOMIC_EMBEDDING_MODEL="m", NOMIC_EMBEDDING_DEVICE="auto", NOMIC_EMBEDDING_BATCH_SIZE="3")
            config = EmbeddingConfig.from_env()
            self.assertEqual((config.model_name, config.device, config.batch_size), ("m", "auto", 3))
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_invalid_vectors_are_rejected_with_chunk_id(self):
        source = chunks()[:1]
        cases = ((EmptyEmbeddingError, [[]]), (InvalidEmbeddingError, [[1.0, float("nan")]]),
                 (InvalidEmbeddingError, [[1.0, float("inf")]]), (InvalidEmbeddingError, [["a", "b"]]),
                 (InvalidEmbeddingError, ["abc"]))
        for error, output in cases:
            with self.assertRaises(error) as context:
                embedder(FakeBackend(output=output)).embed_chunks(source)
            self.assertIn(source[0].chunk_id, str(context.exception))

    def test_inconsistent_dimensions(self):
        source = chunks()[:2]
        with self.assertRaises(InconsistentDimensionError) as context:
            embedder(FakeBackend(output=[[1.0, 2.0], [1.0, 2.0, 3.0]])).embed_chunks(source)
        self.assertIn(source[1].chunk_id, str(context.exception))
        with self.assertRaises(InconsistentDimensionError):
            embedder(FakeBackend(dimension=4, documented=8)).embed_documents(["x"])

    def test_wrong_vector_count(self):
        with self.assertRaises(EmbeddingBatchError):
            embedder(FakeBackend(output=[[1.0]])).embed_documents(["a", "b"])

    def test_backend_failures_are_not_swallowed(self):
        source = chunks()[:2]
        with self.assertRaises(EmbeddingBatchError) as context:
            embedder(FakeBackend(error=RuntimeError("boom"))).embed_chunks(source)
        self.assertIn(source[0].chunk_id, str(context.exception))
        with self.assertRaises(IncompatibleModelError) as context:
            embedder(FakeBackend(error=AttributeError("no get_extended_attention_mask"))).embed_chunks(source)
        self.assertIn("get_extended_attention_mask", str(context.exception))
        self.assertIn("transformers==", str(context.exception))


@unittest.skipUnless(os.environ.get("RUN_NOMIC_INTEGRATION") == "1",
                     "real Nomic integration test: set RUN_NOMIC_INTEGRATION=1 (model must be cached locally)")
class RealNomicIntegrationTests(unittest.TestCase):
    def test_real_model_smoke(self):
        e = NomicEmbedder(EmbeddingConfig.from_env())
        text = "The candidate has experience building REST APIs with ASP.NET Core and SQL Server."
        (vector,) = e.embed_documents([text])
        self.assertGreater(len(vector), 0)
        self.assertTrue(all(math.isfinite(x) for x in vector))
        self.assertEqual(e.dimension, len(vector))
        other = e.embed_documents([text, "Languages: English, Hindi"])
        self.assertEqual({len(v) for v in other}, {len(vector)})
        self.assertEqual(e.model_name, "nomic-ai/nomic-embed-text-v1.5")


if __name__ == "__main__":
    unittest.main()
