"""N4.1 regression tests: D1 (upload lifecycle after a successful write) and D2 (provider failures while matching).

Offline: fake pipeline/embedder and the in-memory Pinecone only. No credentials, no model, no network."""

from types import SimpleNamespace
import unittest
from unittest import mock

from app.api.readiness import (CandidatesNotReadyError, ReadinessGate, ReadinessTracker, ReadinessUnavailableError)
from app.api.runner import JobRunner
from app.api.runtime import ApiRuntime
from app.api import dependencies
from app.ingestion.pdf_resume_ingestion_service import PdfProcessingError, PdfResumeIngestionService
from app.matching import parse_job_description
from app.matching.models import RetrievalError
from app.matching.pinecone_match_service import CandidateRef
from app.parsing.embeddings import EmbeddingBatchError
from app.parsing.vector_store import QueryError
from tests.api.test_n4_hardening import Clock, build_app
from tests.api.test_pipeline_api import (JD, PDF, SECRET, SIGNER, TEST_SETTINGS, candidate, hid, match, real_matching,
                                         upload)
from tests.ingestion.test_pdf_resume_ingestion_service import FakePipeline
from tests.parsing.matching.fakes import CountingEmbedder, make_store
from tests.parsing.matching.test_skill_ranking import spec

PERSON = "person@example.com"
JOB = "Role\nRequired Skills\n- C#\n- SQL\n- Docker\nPreferred Skills\n- Azure\nResponsibilities\n- Build\n"


class FlakyStats:
    """describe_index_stats stand-in: normal, failing, or reporting too few vectors, switchable mid-test."""

    def __init__(self, store):
        self.store, self.mode, self.calls, self.low = store, "ok", 0, 1

    def __call__(self):
        self.calls += 1
        if self.mode == "fail":
            raise RuntimeError(f"{SECRET} {PERSON} provider payload")
        counts = {ns: len(v) for ns, v in self.store.index.namespaces.items()}
        if self.mode == "low":
            counts = {ns: min(n, self.low) for ns, n in counts.items()}
        if self.mode == "malformed":
            return SimpleNamespace()
        return SimpleNamespace(namespaces={ns: SimpleNamespace(vector_count=n) for ns, n in counts.items()})


def real_ingestion(store, embedder=None, pipeline=None):
    return PdfResumeIngestionService(parser_pipeline=pipeline or FakePipeline(), embedder=embedder or CountingEmbedder(),
                                     pinecone_store=store)


def upload_app(store, on_sleep=None, **kwargs):
    clock = Clock()

    async def sleep(seconds):
        clock.now += seconds
        if on_sleep:
            on_sleep()
    runtime = ApiRuntime(settings_provider=lambda: TEST_SETTINGS, clock=clock, sleep=sleep)
    client, runtime, _ = build_app(ingestion=real_ingestion(store, **kwargs), runtime=runtime)
    return client, runtime, clock


def vectors(store):
    return {ns: sorted(v) for ns, v in store.index.namespaces.items()}


# ======================================================================== D1
class UploadLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        self.stats = FlakyStats(self.store)
        patcher = mock.patch.object(self.store.index, "describe_index_stats", side_effect=self.stats)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.upsert = mock.Mock(wraps=self.store.index.upsert)
        self.store.index.upsert = self.upsert

    def test_successful_ready_upload(self):
        client, runtime, _ = upload_app(self.store)
        response = upload(client)
        body = response.json()
        self.assertEqual((response.status_code, body["status"], body["ready"]), (201, "INGESTED", True))
        self.assertTrue(SIGNER.verify(body["document_id"], body["access_token"]))
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.upsert.call_count, 1)
        self.assertEqual(self.store.namespace_vector_count(body["namespace"]), body["chunk_count"])

    def test_readiness_stats_failure_after_a_successful_upsert_is_not_reported_as_a_plain_failure(self):
        client, runtime, _ = upload_app(self.store, on_sleep=None)
        self.stats.mode = "fail"
        with self.assertLogs("app.api", level="WARNING") as logged:
            response = upload(client)
        error = response.json()["error"]
        self.assertEqual((response.status_code, error["code"]), (502, "READINESS_UNCONFIRMED"))
        self.assertEqual(error["details"], {"stored": True, "ready": False, "retryable": True})
        self.assertIn("was stored", error["message"])
        self.assertIn("not confirmed ready", error["message"])
        self.assertNotIn("access_token", response.text)                 # no token for an unconfirmed upload
        self.assertEqual(response.headers["retry-after"], "5")
        self.assertEqual(response.headers["cache-control"], "no-store")
        for text in (response.text, " ".join(logged.output)):
            for secret in (SECRET, PERSON, "provider payload"):
                self.assertNotIn(secret, text)
        self.assertEqual(self.upsert.call_count, 1)                     # storage WAS attempted and succeeded
        (namespace, ids), = vectors(self.store).items()
        self.assertTrue(ids)
        self.assertEqual(runtime.tracker.expected(sha256_of(PDF)), len(ids))   # matching will insist on this count
        self.assertGreater(self.stats.calls, 2)                         # the failing read was retried, not given up on

    def test_a_failure_is_never_presented_as_pending_or_ready(self):
        client, _, _ = upload_app(self.store)
        self.stats.mode = "fail"
        response = upload(client)
        self.assertNotEqual(response.status_code, 202)
        self.assertNotIn("PENDING_INDEXING", response.text)
        self.assertNotIn('"ready":true', response.text.replace(" ", ""))

    def test_transient_stats_failure_is_retried_and_can_still_be_ready(self):
        calls = {"sleeps": 0}

        def on_sleep():
            calls["sleeps"] += 1
            if calls["sleeps"] == 2:
                self.stats.mode = "ok"
        client, _, _ = upload_app(self.store, on_sleep=on_sleep)
        self.stats.mode = "fail"
        response = upload(client)
        self.assertEqual((response.status_code, response.json()["ready"]), (201, True))
        self.assertGreaterEqual(calls["sleeps"], 2)

    def test_delayed_visibility_is_pending_with_a_token_and_retry_after(self):
        client, runtime, clock = upload_app(self.store)
        self.stats.mode = "low"                                         # the reads succeed but show too few vectors
        response = upload(client)
        body = response.json()
        self.assertEqual((response.status_code, body["status"], body["ready"]), (202, "PENDING_INDEXING", False))
        self.assertTrue(SIGNER.verify(body["document_id"], body["access_token"]))
        self.assertEqual(response.headers["retry-after"], "5")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertGreaterEqual(clock.now, 1000.0 + TEST_SETTINGS.readiness_timeout_seconds)       # bounded wait
        self.assertEqual(self.upsert.call_count, 1)

    def test_delayed_visibility_that_resolves_during_the_wait_is_ready(self):
        sleeps = []

        def on_sleep():
            sleeps.append(1)
            if len(sleeps) == 3:
                self.stats.mode = "ok"
        client, _, _ = upload_app(self.store, on_sleep=on_sleep)
        self.stats.mode = "low"
        response = upload(client)
        self.assertEqual((response.status_code, response.json()["status"], response.json()["ready"]),
                         (201, "INGESTED", True))

    def test_failures_after_an_observed_low_count_are_still_pending_not_unconfirmed(self):
        sleeps = []

        def on_sleep():
            sleeps.append(1)
            if len(sleeps) == 1:
                self.stats.mode = "fail"                                # one read showed "too few", later reads fail
        client, _, _ = upload_app(self.store, on_sleep=on_sleep)
        self.stats.mode = "low"
        response = upload(client)
        self.assertEqual((response.status_code, response.json()["status"]), (202, "PENDING_INDEXING"))

    def test_failure_before_any_write_has_no_token_and_never_touches_readiness(self):
        broken = CountingEmbedder()
        broken.embed_chunks = mock.Mock(side_effect=EmbeddingBatchError(f"{SECRET} {PERSON}"))
        client, _, _ = upload_app(self.store, embedder=broken)
        response = upload(client)
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (503, "EMBEDDING_UNAVAILABLE"))
        self.assertNotIn("access_token", response.text)
        self.assertNotIn(SECRET, response.text)
        self.assertEqual(self.upsert.call_count, 0)                     # confirmed pre-write failure
        self.assertEqual(self.stats.calls, 0)
        self.assertEqual(vectors(self.store), {})

        class Unreadable(FakePipeline):
            def build(self, pdf_bytes, source_filename):
                raise PdfProcessingError("M1 failed: PDFExtractionError", "M1")
        client, _, _ = upload_app(self.store, pipeline=Unreadable())
        response = upload(client)
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (422, "PDF_UNPROCESSABLE"))
        self.assertEqual((self.upsert.call_count, self.stats.calls, vectors(self.store)), (0, 0, {}))

    def test_partial_write_is_reported_as_a_storage_failure_without_readiness_polling(self):
        original = self.store.upsert_embeddings
        self.store.upsert_embeddings = lambda records: original(records, batch_size=2)       # force several batches
        writes = []

        def fail_on_second_batch(vectors, namespace=""):
            writes.append(len(vectors))
            if len(writes) == 2:
                raise RuntimeError(f"{SECRET} batch rejected")
            return real_upsert(vectors=vectors, namespace=namespace)
        real_upsert = self.upsert
        self.store.index.upsert = fail_on_second_batch
        client, _, _ = upload_app(self.store)
        response = upload(client)
        error = response.json()["error"]
        self.assertEqual((response.status_code, error["code"]), (502, "STORAGE_UNAVAILABLE"))
        self.assertEqual(error["details"], {"partial_write_possible": True, "retryable": True})
        self.assertNotIn("access_token", response.text)
        self.assertNotIn(SECRET, response.text)
        (namespace, ids), = vectors(self.store).items()
        self.assertEqual(len(ids), 2)                                   # exactly the first batch was written
        self.assertEqual(self.stats.calls, 1)                           # only the pre-write count; no readiness polling

    def test_reupload_after_an_uncertain_result_repairs_it_idempotently(self):
        client, runtime, _ = upload_app(self.store)
        self.stats.mode = "fail"
        first = upload(client)
        self.assertEqual(first.json()["error"]["code"], "READINESS_UNCONFIRMED")
        stored = vectors(self.store)
        self.stats.mode = "ok"
        second = upload(client)
        body = second.json()
        self.assertEqual((second.status_code, body["status"], body["ready"], body["replaced_existing"]),
                         (200, "INGESTED", True, True))
        self.assertTrue(SIGNER.verify(body["document_id"], body["access_token"]))
        self.assertEqual(vectors(self.store), stored)                   # same ids: replaced, nothing duplicated
        self.assertEqual(self.upsert.call_count, 2)

    def test_any_unexpected_readiness_error_after_a_write_keeps_the_stored_state(self):
        class Exploding:
            async def wait_until_indexed(self, *args, **kwargs):
                raise ValueError(f"{SECRET} {PERSON}")
        client, _, _ = upload_app(self.store)
        client.app.dependency_overrides[dependencies.get_readiness] = lambda: Exploding()
        with self.assertLogs("app.api", level="WARNING") as logged:
            response = upload(client)
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (502, "READINESS_UNCONFIRMED"))
        self.assertEqual(response.json()["error"]["details"]["stored"], True)
        for text in (response.text, " ".join(logged.output)):
            self.assertNotIn(SECRET, text)
            self.assertNotIn(PERSON, text)


def sha256_of(data):
    from hashlib import sha256
    return sha256(data).hexdigest()


class WaitUntilIndexedTests(unittest.TestCase):
    """The tri-state contract of ReadinessGate.wait_until_indexed in isolation."""

    def run_wait(self, outcomes, timeout=3.0):
        clock, calls = Clock(), []

        async def sleep(seconds):
            clock.now += seconds
        store = SimpleNamespace(index=SimpleNamespace(describe_index_stats=lambda: next_outcome()))

        def next_outcome():
            calls.append(1)
            outcome = outcomes[min(len(calls), len(outcomes)) - 1]
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(namespaces={"ns": SimpleNamespace(vector_count=outcome)})

        async def main():
            runner = JobRunner("light", 2, 8, 5)
            return await ReadinessGate(ReadinessTracker(), 200).wait_until_indexed(
                runner, store, "ns", 4, timeout, 1.0, sleep, clock)
        import asyncio
        return asyncio.run(main()), len(calls)

    def test_ready_pending_and_unavailable_are_distinct(self):
        self.assertEqual(self.run_wait([4]), (True, 1))
        self.assertEqual(self.run_wait([1, 2, 4]), (True, 3))
        self.assertEqual(self.run_wait([1])[0], False)                           # observed, never reached: pending
        self.assertEqual(self.run_wait([RuntimeError("x"), 1, RuntimeError("x")])[0], False)
        self.assertEqual(self.run_wait([RuntimeError("x"), 4])[0], True)         # a blip does not decide the outcome
        with self.assertRaises(ReadinessUnavailableError) as raised:
            self.run_wait([RuntimeError(f"{SECRET}")])
        self.assertEqual(raised.exception.reason, "RuntimeError")
        self.assertNotIn(SECRET, str(raised.exception) + raised.exception.reason)

    def test_zero_timeout_still_reads_once(self):
        self.assertEqual(self.run_wait([1], timeout=0)[0], False)
        with self.assertRaises(ReadinessUnavailableError):
            self.run_wait([RuntimeError("x")], timeout=0)


# ======================================================================== D2
class MatchProviderFailureTests(unittest.TestCase):
    def setUp(self):
        self.service = real_matching({"a": spec("A", ["C#", "SQL", "Docker"]), "b": spec("B", ["C#"])})
        self.store = self.service.store
        self.stats = FlakyStats(self.store)
        patcher = mock.patch.object(self.store.index, "describe_index_stats", side_effect=self.stats)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ranked = mock.Mock(wraps=self.service.match)
        self.service.match = self.ranked
        self.client, self.runtime, _ = build_app(matching=self.service)
        self.ns = {n: self.store.namespace_for(hid(n)) for n in ("a", "b")}

    def assert_retryable_provider_error(self, response):
        error = response.json()["error"]
        self.assertEqual((response.status_code, error["code"]), (502, "RETRIEVAL_UNAVAILABLE"))
        self.assertEqual(error["details"], {"retryable": True})
        self.assertEqual(response.headers["retry-after"], "5")
        self.assertEqual(set(response.json()), {"error"})                        # no ranking of any kind
        for secret in (SECRET, PERSON, "provider payload", "Traceback"):
            self.assertNotIn(secret, response.text)
        self.ranked.assert_not_called()                                          # fail closed: nobody was ranked

    def test_stats_exception_is_a_retryable_502_and_nobody_is_ranked(self):
        self.stats.mode = "fail"
        with self.assertLogs("app.api", level="WARNING") as logged:
            response = match(self.client, [candidate("a"), candidate("b")], JOB)
        self.assert_retryable_provider_error(response)
        for secret in (SECRET, PERSON, "provider payload"):
            self.assertNotIn(secret, " ".join(logged.output))
        self.assertIn("RuntimeError", " ".join(logged.output))                   # the TYPE is logged, nothing else

    def test_malformed_stats_payload_is_a_provider_error(self):
        self.stats.mode = "malformed"
        self.assert_retryable_provider_error(match(self.client, [candidate("a")], JOB))

    def test_probe_query_exception_is_a_provider_error(self):
        for error in (QueryError(f"{SECRET} {PERSON} provider payload"), RuntimeError(f"{SECRET} provider payload")):
            with mock.patch.object(self.store, "query", side_effect=error):
                self.assert_retryable_provider_error(match(self.client, [candidate("a"), candidate("b")], JOB))

    def test_failure_on_a_later_candidate_probe_ranks_nobody(self):
        real_query, seen = self.store.query, []

        def second_probe_fails(vector, top_k, namespace, filter=None):
            seen.append(namespace)
            if len(seen) == 2:
                raise QueryError(f"{SECRET} provider payload")
            return real_query(vector, top_k, namespace, filter)
        self.store.query = second_probe_fails
        self.assert_retryable_provider_error(match(self.client, [candidate("a"), candidate("b")], JOB))

    def test_index_description_failure_while_probing_is_a_provider_error(self):
        self.store.dimension = None
        with mock.patch.object(self.store, "describe_index", side_effect=RuntimeError(f"{SECRET} provider payload")):
            self.assert_retryable_provider_error(match(self.client, [candidate("a")], JOB))

    def test_matcher_provider_failures_use_the_same_contract(self):
        self.ranked.side_effect = RetrievalError(f"{SECRET} provider payload")
        response = match(self.client, [candidate("a")], JOB)
        error = response.json()["error"]
        self.assertEqual((response.status_code, error["code"], error["details"]), (502, "RETRIEVAL_UNAVAILABLE",
                                                                                   {"retryable": True}))
        self.assertEqual(response.headers["retry-after"], "5")
        self.assertNotIn(SECRET, response.text)

    def test_confirmed_not_ready_candidates_stay_409_not_provider_errors(self):
        self.stats.mode = "low"                                                  # reads succeed; counts are too low
        self.stats.low = 1
        response = match(self.client, [candidate("a"), candidate("b")], JOB)
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (409, "CANDIDATE_NOT_READY"))
        self.assertEqual(response.json()["error"]["details"]["retryable"], True)
        self.assertEqual(response.headers["retry-after"], "5")
        self.ranked.assert_not_called()
        self.runtime.tracker.record(hid("a"), 99)
        self.stats.mode = "ok"
        mixed = match(self.client, [candidate("a"), candidate("b")], JOB)
        self.assertEqual((mixed.status_code, mixed.json()["error"]["details"]["candidate_indexes"]), (409, [0]))

    def test_empty_namespace_is_not_ready_not_a_provider_error(self):
        response = match(self.client, [candidate("a"), candidate("never-uploaded")], JOB)
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (409, "CANDIDATE_NOT_READY"))
        self.assertEqual(response.json()["error"]["details"]["candidate_indexes"], [1])

    def test_successful_readiness_ranks_exactly_like_the_direct_service(self):
        response = match(self.client, [candidate("b"), candidate("a")], JOB)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.ranked.call_count, 1)
        refs = [CandidateRef(hid(n), self.ns[n]) for n in ("a", "b")]
        self.ranked.reset_mock()
        baseline = self.service.match(parse_job_description(JOB), refs).result.candidates
        self.assertEqual([(c.document_id, c.rank, list(c.skill_score.key)) for c in baseline],
                         [(c["document_id"], c["rank"], c["ranking_key"]) for c in response.json()["candidates"]])

    def test_client_errors_stay_client_errors_and_never_reach_the_provider(self):
        self.stats.mode = "fail"
        duplicates = match(self.client, [candidate("a"), candidate("a")], JOB)
        forged = match(self.client, [{"document_id": hid("a"), "access_token": "0" * 64}], JOB)
        bad_jd = match(self.client, [candidate("a")], "   ")
        self.assertEqual([r.status_code for r in (duplicates, forged, bad_jd)], [422, 403, 422])
        self.assertEqual(self.stats.calls, 0)
        self.ranked.assert_not_called()


class GateUnitTests(unittest.TestCase):
    def test_gate_wraps_provider_failures_but_not_confirmed_states(self):
        service = real_matching({"a": spec("A", ["C#"])})
        store, ns = service.store, service.store.namespace_for(hid("a"))
        refs = [CandidateRef(hid("a"), ns)]
        gate = ReadinessGate(ReadinessTracker(), 200)
        self.assertEqual(set(gate.verify(store, refs)), {ns})
        with mock.patch.object(store.index, "describe_index_stats", side_effect=KeyError(SECRET)):
            with self.assertRaises(ReadinessUnavailableError) as raised:
                gate.verify(store, refs)
        self.assertEqual(raised.exception.reason, "KeyError")
        self.assertNotIn(SECRET, repr(raised.exception) + raised.exception.reason)
        self.assertIsNone(raised.exception.__cause__)                            # the provider exception is not chained
        gate_tracked = ReadinessTracker()
        gate_tracked.record(hid("a"), 50)
        with self.assertRaises(CandidatesNotReadyError):                         # a confirmed state is not "unavailable"
            ReadinessGate(gate_tracked, 200).verify(store, refs)


if __name__ == "__main__":
    unittest.main()
