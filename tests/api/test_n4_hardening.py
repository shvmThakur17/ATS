"""N4 hardening tests. Offline: fakes and the in-memory Pinecone only. No credentials, no model, no network.

The only processes started are local child processes for the PDF-isolation tests."""

import asyncio
from dataclasses import replace
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import evidence, settings as settings_module
from app.api.auth import AuthGateMiddleware
from app.api.candidate_access import CandidateTokenSigner
from app.api.app_factory import configure_app
from app.api.dependencies import ApiLimits, get_ingestion_service, get_matching_service, get_token_signer
from app.api.evidence import redact_emails, redact_phones, safe_excerpt
from app.api.rate_limit import SlidingWindowLimiter
from app.api.readiness import CandidatesNotReadyError, ReadinessGate, ReadinessTracker
from app.api.runner import JobRunner, JobTimeoutError, LazyService, ServerBusyError
from app.api.runtime import ApiRuntime
from app.api.settings import ApiKeyAuthenticator, ApiSettings, SettingsError, parse_api_keys, parse_origins
from app.api.v1 import jd_matching
from app.ingestion.isolated_pipeline import (IsolatedResumePipeline, PdfProcessingTimeoutError, PdfResourceError)
from app.ingestion.pdf_resume_ingestion_service import (
    DeterministicResumePipeline, IngestionLimits, IngestionResult, InvalidIngestionConfigError, PdfProcessingError,
    PdfResumeIngestionService, StorageStageError, TooManyChunksError)
from app.matching import parse_job_description
from app.matching.pinecone_match_service import (CandidateRef, NamespaceCountUnavailableError, PineconeMatchLimits,
                                                  PineconeMatchService)
from tests.api.test_pipeline_api import (API_KEY, AUTH, CONFIG, FakeIngestion, FakeMatching, JD, PDF, SECRET, SIGNER,
                                         TEST_SETTINGS, AlwaysReady, candidate, hid, ingestion_result, match,
                                         real_matching, upload)
from tests.ingestion.test_pdf_resume_ingestion_service import RESUME_ROWS, real_pdf
from tests.parsing.matching.fakes import CANDIDATE_A, CANDIDATE_B, CountingEmbedder, make_store
from tests.parsing.matching.test_pinecone_canonical_adapter import world
from tests.parsing.matching.test_skill_ranking import spec

ROOT = Path(__file__).resolve().parents[2]
JOB = "Role\nRequired Skills\n- C#\n- SQL\n- Docker\nPreferred Skills\n- Azure\nResponsibilities\n- Build\n"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds


def build_app(settings=TEST_SETTINGS, ingestion=None, matching=None, clock=None, gate="real", limits=None,
              with_headers=True, runtime=None):
    clock = clock or Clock()
    runtime = runtime or ApiRuntime(settings_provider=lambda: settings, clock=clock, sleep=clock.sleep)
    app = FastAPI()
    configure_app(app, (lambda: limits) if limits else None, runtime)
    app.dependency_overrides[get_token_signer] = lambda: SIGNER
    if ingestion is not None:
        app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    if matching is not None:
        app.dependency_overrides[get_matching_service] = lambda: matching
    if gate != "real":
        app.dependency_overrides[__import__("app.api.dependencies", fromlist=["get_readiness"]).get_readiness] = lambda: gate
    client = TestClient(app, raise_server_exceptions=False, headers=AUTH if with_headers else None)
    return client, runtime, clock


def spy_runtime(settings=TEST_SETTINGS, clock=None):
    """A runtime whose heavy builders fail the test if anything tries to construct a model, client or service."""
    clock = clock or Clock()
    runtime = ApiRuntime(settings_provider=lambda: settings, clock=clock, sleep=clock.sleep)
    built = []

    def forbidden(name):
        def builder():
            built.append(name)
            raise AssertionError(f"{name} must not be built")
        return builder
    runtime.embedder, runtime.store = LazyService(forbidden("embedder")), LazyService(forbidden("store"))
    runtime.ingestion, runtime.matching = LazyService(forbidden("ingestion")), LazyService(forbidden("matching"))
    return runtime, built


def fill(store, namespace, count):
    store.index.namespaces[namespace] = {f"id-{i}": {"values": [0.0], "metadata": {}} for i in range(count)}


def stats_for(store, override=None):
    """describe_index_stats result built from the fake index, optionally overriding some namespaces' counts."""
    def stats():
        counts = {ns: len(v) for ns, v in store.index.namespaces.items()}
        counts.update(override or {})
        return SimpleNamespace(namespaces={ns: SimpleNamespace(vector_count=n) for ns, n in counts.items() if n is not None})
    return stats


# ======================================================================== H1: authentication, rate limit, quota
class AuthenticationTests(unittest.TestCase):
    def test_unauthenticated_requests_are_rejected_before_anything_is_built(self):
        runtime, built = spy_runtime()
        client, _, _ = build_app(runtime=runtime, with_headers=False)
        bad_headers = ({}, {"Authorization": "Bearer wrong-key"}, {"Authorization": API_KEY},
                       {"Authorization": f"Basic {API_KEY}"}, {"Authorization": "Bearer "},
                       {"Authorization": "Bearer " + "k" * 300}, {"Authorization": f"Bearer {API_KEY}x"})
        for headers in bad_headers:
            for response in (client.post("/api/v1/resume/upload", files={"file": ("a.pdf", PDF, "application/pdf")},
                                         headers=headers),
                             client.post("/api/v1/jd/match", json={"job_description": JD, "candidates": [candidate("a")]},
                                         headers=headers)):
                self.assertEqual((response.status_code, response.json()["error"]["code"]), (401, "UNAUTHENTICATED"), headers)
                self.assertEqual(response.headers["www-authenticate"], "Bearer")
                self.assertNotIn(API_KEY, response.text)
        self.assertEqual(built, [])                            # no model, store or service was ever constructed

    def test_gate_does_not_read_the_body_of_an_unauthenticated_request(self):
        reads, called = [], []

        async def inner(scope, receive, send):
            called.append(1)

        async def receive():
            reads.append(1)
            return {"type": "http.request", "body": b"x" * 100, "more_body": False}

        sent = []

        async def send(message):
            sent.append(message)
        gate = AuthGateMiddleware(inner, lambda: TEST_SETTINGS, lambda: SlidingWindowLimiter(5, 60))
        scope = {"type": "http", "method": "POST", "path": "/svc/api/v1/resume/upload", "headers": []}
        asyncio.run(gate(scope, receive, send))
        self.assertEqual((sent[0]["status"], reads, called), (401, [], []))

    def test_unconfigured_authentication_fails_closed(self):
        for settings in (ApiSettings(token_secret=TEST_SETTINGS.token_secret), None):
            provider = (lambda: settings) if settings else mock.Mock(side_effect=SettingsError("bad"))
            runtime = ApiRuntime(settings_provider=provider)
            client, _, _ = build_app(runtime=runtime)
            response = upload(client)
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (503, "SERVICE_UNAVAILABLE"))

    def test_valid_key_passes_and_identity_headers_are_never_trusted(self):
        ingestion = FakeIngestion()
        client, runtime, _ = build_app(ingestion=ingestion, with_headers=False)
        spoof = {"X-User": "admin", "X-Caller": "tester", "X-Forwarded-For": "10.0.0.1", "X-Authenticated-User": "admin"}
        self.assertEqual(upload(client, headers=spoof) if False else client.post(
            "/api/v1/resume/upload", files={"file": ("a.pdf", PDF, "application/pdf")}, headers=spoof).status_code, 401)
        response = client.post("/api/v1/resume/upload", files={"file": ("a.pdf", PDF, "application/pdf")},
                               headers={**AUTH, "X-User": "someone-else"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(set(runtime.rate_limiter()._events), {"tester"})        # the caller is the key's owner only

    def test_authenticator_handles_malformed_input_without_raising(self):
        auth = ApiKeyAuthenticator(TEST_SETTINGS.api_key_hashes)
        self.assertEqual(auth.authenticate(f"Bearer {API_KEY}"), "tester")
        self.assertEqual(auth.authenticate(f"bearer   {API_KEY}  "), "tester")
        for header in (None, "", "Bearer", "Bearer éé", "Bearer " + "\ud800", b"Bearer x", 5):
            self.assertIsNone(auth.authenticate(header))

    def test_keys_are_configured_as_hashes_and_never_appear_in_repr(self):
        self.assertNotIn(API_KEY, repr(TEST_SETTINGS))
        self.assertNotIn(TEST_SETTINGS.token_secret, repr(TEST_SETTINGS))
        for bad in ("name", "name=abc", "na me=" + "a" * 64, "name=" + "A" * 64, f"a={'a' * 64},a={'b' * 64}"):
            with self.assertRaises(SettingsError, msg=bad):
                parse_api_keys(bad)
        self.assertEqual(list(parse_api_keys(f"alice={'a' * 64}, bob={'b' * 64}")), ["alice", "bob"])

    def test_secrets_are_never_logged(self):
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        root = logging.getLogger()
        root.addHandler(handler)
        previous = root.level
        root.setLevel(logging.DEBUG)
        try:
            client, _, _ = build_app(ingestion=FakeIngestion(error=RuntimeError(f"{SECRET} {API_KEY}")),
                                     matching=FakeMatching(error=RuntimeError(API_KEY)))
            upload(client)
            match(client, [candidate("a")])
            TestClient(client.app, raise_server_exceptions=False).post("/api/v1/jd/match", json={}, headers={
                "Authorization": "Bearer nope"})
        finally:
            root.removeHandler(handler)
            root.setLevel(previous)
        text = " ".join(r.getMessage() for r in records)
        for secret in (API_KEY, SECRET, SIGNER.issue(hid("a")), TEST_SETTINGS.token_secret):
            self.assertNotIn(secret, text)


class RateLimitAndQuotaTests(unittest.TestCase):
    def test_rate_limit_is_per_caller_and_recovers(self):
        other = "second-key-" + "z" * 24
        settings = replace(TEST_SETTINGS, rate_limit_per_minute=3,
                           api_key_hashes={**TEST_SETTINGS.api_key_hashes, "other": sha256(other.encode()).hexdigest()})
        client, runtime, clock = build_app(settings=settings, ingestion=FakeIngestion())
        statuses = [upload(client).status_code for _ in range(5)]
        self.assertEqual(statuses, [201, 201, 201, 429, 429])
        limited = upload(client)
        self.assertEqual(limited.json()["error"]["code"], "RATE_LIMITED")
        self.assertGreaterEqual(int(limited.headers["retry-after"]), 1)
        self.assertEqual(client.post("/api/v1/resume/upload", files={"file": ("a.pdf", PDF, "application/pdf")},
                                     headers={"Authorization": f"Bearer {other}"}).status_code, 201)   # separate budget
        clock.now += 61
        self.assertEqual(upload(client).status_code, 201)

    def test_upload_quota_is_enforced_before_ingestion_and_invalid_uploads_do_not_consume_it(self):
        ingestion = FakeIngestion()
        client, runtime, clock = build_app(settings=replace(TEST_SETTINGS, upload_quota_per_day=2), ingestion=ingestion)
        self.assertEqual(upload(client, PDF, "notes.txt").status_code, 415)                  # rejected, no quota used
        self.assertEqual(upload(client, b"", "empty.pdf").status_code, 400)
        self.assertEqual([upload(client).status_code for _ in range(3)], [201, 201, 429])
        denied = upload(client)
        self.assertEqual((denied.status_code, denied.json()["error"]["code"]), (429, "QUOTA_EXCEEDED"))
        self.assertIn("retry-after", denied.headers)
        self.assertEqual(len(ingestion.calls), 2)                                             # the third never ran
        clock.now += 86_401
        self.assertEqual(upload(client).status_code, 201)

    def test_limiter_is_bounded_per_caller(self):
        limiter = SlidingWindowLimiter(3, 10, Clock())
        for _ in range(100):
            limiter.acquire("a")
        self.assertLessEqual(len(limiter._events["a"]), 3)
        with self.assertRaises(ValueError):
            SlidingWindowLimiter(0, 10)


class SettingsTests(unittest.TestCase):
    def test_from_env_parsing_and_validation(self):
        env = {"API_KEY_HASHES": f"ops={'c' * 64}", "CANDIDATE_TOKEN_SECRET": "s" * 40, "API_RATE_LIMIT_PER_MINUTE": "10",
               "API_UPLOAD_QUOTA_PER_DAY": "5", "API_CORS_ORIGINS": "https://app.example.com, http://localhost:4200/",
               "INGEST_MAX_CHUNKS": "50", "M13_MAX_RETRIEVAL_DEPTH": "100"}
        settings = ApiSettings.from_env(env)
        self.assertEqual((settings.rate_limit_per_minute, settings.upload_quota_per_day, settings.max_chunks),
                         (10, 5, 50))
        self.assertEqual(settings.cors_origins, ("https://app.example.com", "http://localhost:4200"))
        self.assertIsNotNone(settings.signer)
        for bad in ({"API_RATE_LIMIT_PER_MINUTE": "many"}, {"API_RATE_LIMIT_PER_MINUTE": "0"},
                    {"CANDIDATE_TOKEN_SECRET": "short"}, {"API_KEY_HASHES": "x"}, {"API_CORS_ORIGINS": "*"},
                    {"INGEST_MAX_CHUNKS": "300", "M13_MAX_RETRIEVAL_DEPTH": "200"}):
            with self.assertRaises(SettingsError, msg=bad):
                ApiSettings.from_env(bad)
        self.assertIsNone(ApiSettings.from_env({}).authenticator)
        self.assertIsNone(ApiSettings.from_env({}).signer)

    def test_configuration_is_loaded_once_not_per_request(self):
        counts = {"env": 0, "dotenv": 0}

        def from_env(*_):
            counts["env"] += 1
            return TEST_SETTINGS

        previous = settings_module._loaded
        settings_module._loaded = None
        try:
            with mock.patch.object(ApiSettings, "from_env", side_effect=from_env), \
                    mock.patch("dotenv.load_dotenv", side_effect=lambda *a, **k: counts.__setitem__("dotenv", counts["dotenv"] + 1)):
                for _ in range(5):
                    settings_module.load_settings()
        finally:
            settings_module._loaded = previous
        self.assertEqual(counts, {"env": 1, "dotenv": 1})

    def test_requests_never_reload_the_environment(self):
        with mock.patch("dotenv.load_dotenv", side_effect=AssertionError("env file re-read")):
            client, _, _ = build_app(ingestion=FakeIngestion())
            self.assertEqual([upload(client).status_code for _ in range(3)], [201, 201, 201])


# ======================================================================== H2: bounded evidence processing
class EvidenceTests(unittest.TestCase):
    def test_redaction_still_works(self):
        self.assertEqual(redact_emails("write to john.doe@example.com now"), "write to [email] now")
        self.assertEqual(redact_emails("mail: a+b@sub.example.co.uk."), "mail: [email].")
        self.assertEqual(redact_emails("cut off tail@exa"), "cut off [email]")
        self.assertEqual(redact_phones("call +91 98765 43210 today"), "call [phone] today")
        self.assertEqual(redact_phones("call (555) 123-4567."), "call [phone].")
        self.assertEqual(redact_phones("9876543210"), "[phone]")
        for untouched in ("C#, ASP.NET Core and SQL Server", "version 3.5.1 released", "ISO 27001 audit",
                          "abc1234567def", "12345 items", "meet @ noon"):
            self.assertEqual(redact_emails(redact_phones(untouched)), untouched, untouched)

    def test_excerpt_is_short_and_centered_on_the_matched_term(self):
        text = "x " * 5000 + "Built services with Python and SQL. Contact jane@example.com +1 202 555 0143."
        excerpt = safe_excerpt(text, "Python")
        self.assertIn("Python", excerpt)
        self.assertTrue(excerpt.startswith("..."))
        self.assertLessEqual(len(excerpt), 300)
        both = safe_excerpt("Python " + "word " * 40 + "jane@example.com +1 202 555 0143", "Python")
        self.assertNotIn("jane@", both)
        self.assertNotIn("555", both)
        self.assertIsNone(safe_excerpt(None))
        self.assertIsNone(safe_excerpt(""))

    def test_adversarial_text_is_processed_in_bounded_time(self):
        hostile = {"long word then at": "a" * 100_000 + "@", "dots": "1." * 50_000, "digits then letter": "1-" * 49_999 + "a",
                   "many ats": "@" * 100_000, "spaces": " " * 100_000, "digit runs": "9 " * 50_000,
                   "parens": "(1" * 50_000, "email-ish": "a.b-" * 25_000 + "@x", "plus signs": "+" * 100_000,
                   "term everywhere": "python " * 14_000}
        for name, text in hostile.items():
            for term in (None, "python"):
                started = time.perf_counter()
                excerpt = safe_excerpt(text, term)
                elapsed = time.perf_counter() - started
                self.assertLess(elapsed, 0.25, f"{name}: {elapsed:.3f}s")
                self.assertTrue(excerpt is None or len(excerpt) <= 300, name)

    def test_the_matching_text_window_is_bounded_before_any_scan(self):
        self.assertLessEqual(evidence.WINDOW_CHARS, 1200)
        with mock.patch.object(evidence, "redact_phones", wraps=evidence.redact_phones) as phones:
            safe_excerpt("word " * 200_000, "word")
        self.assertLessEqual(len(phones.call_args.args[0]), evidence.WINDOW_CHARS)

    def test_match_response_is_built_off_the_event_loop_and_stays_fast_on_a_hostile_resume(self):
        hostile = "Python and C# " + "1." * 8_000 + " " + "a" * 6_000 + "@"
        service = real_matching({"evil": spec("E", ["C#", "SQL", "Docker"], description=hostile)})
        client, runtime, _ = build_app(matching=service)
        names = []
        real = jd_matching.to_response
        with mock.patch.object(jd_matching, "to_response", side_effect=lambda *a: (
                names.append(threading.current_thread().name), real(*a))[1]):
            started = time.perf_counter()
            response = match(client, [candidate("evil")], JOB)
            elapsed = time.perf_counter() - started
        self.assertEqual(response.status_code, 200)
        self.assertLess(elapsed, 3.0)
        self.assertTrue(names and names[0].startswith("api-light"))
        for skill in response.json()["candidates"][0]["required_skills"]:
            if skill["evidence"] and skill["evidence"]["excerpt"]:
                self.assertLessEqual(len(skill["evidence"]["excerpt"]), 300)


# ======================================================================== H3: explicit readiness
class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.service = real_matching({"a": spec("A", ["C#", "SQL", "Docker"]), "b": spec("B", ["C#"])})
        self.store = self.service.store
        self.ns = {n: self.store.namespace_for(hid(n)) for n in ("a", "b")}
        self.client, self.runtime, _ = build_app(matching=self.service)

    def test_ready_candidates_are_ranked_exactly_like_the_direct_service(self):
        body = match(self.client, [candidate("b"), candidate("a")], JOB)
        self.assertEqual(body.status_code, 200)
        refs = [CandidateRef(hid(n), self.ns[n]) for n in ("a", "b")]
        expected = self.service.match(parse_job_description(JOB), refs).result.candidates
        self.assertEqual([(c.document_id, c.rank, list(c.skill_score.key)) for c in expected],
                         [(c["document_id"], c["rank"], c["ranking_key"]) for c in body.json()["candidates"]])

    def test_zero_count_is_not_ranked(self):
        with mock.patch.object(self.store.index, "describe_index_stats", stats_for(self.store, {self.ns["a"]: None})), \
                mock.patch.object(self.service, "match", wraps=self.service.match) as ranked:
            response = match(self.client, [candidate("a"), candidate("b")], JOB)
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (409, "CANDIDATE_NOT_READY"))
        self.assertEqual(response.json()["error"]["details"], {"candidate_indexes": [0], "retryable": True})
        self.assertEqual(response.headers["retry-after"], "5")
        ranked.assert_not_called()                          # nothing was ranked, not even the ready candidate
        self.assertEqual(set(response.json()), {"error"})                  # no ranking of any kind in the body

    def test_stale_low_count_is_detected_by_the_probe(self):
        real = self.store.namespace_vector_count(self.ns["a"])
        with mock.patch.object(self.store.index, "describe_index_stats", stats_for(self.store, {self.ns["a"]: real - 2})):
            response = match(self.client, [candidate("a")], JOB)
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (409, "CANDIDATE_NOT_READY"))

    def test_count_behind_the_ingestion_that_just_happened_is_not_ready(self):
        real = self.store.namespace_vector_count(self.ns["a"])
        self.runtime.tracker.record(hid("a"), real + 3)       # the last upload wrote more chunks than are visible
        response = match(self.client, [candidate("a")], JOB)
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (409, "CANDIDATE_NOT_READY"))
        self.runtime.tracker.record(hid("a"), real)
        self.assertEqual(match(self.client, [candidate("a")], JOB).status_code, 200)

    def test_oversized_candidates_are_reported_with_their_position(self):
        counts = [self.store.namespace_vector_count(self.ns[n]) for n in ("b", "a")]
        cap = min(counts) - 1
        client, _, _ = build_app(matching=self.service, settings=replace(TEST_SETTINGS, max_chunks=cap, max_retrieval_depth=cap))
        response = match(client, [candidate("b"), candidate("a")], JOB)
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (422, "CANDIDATE_TOO_LARGE"))
        self.assertEqual(response.json()["error"]["details"]["candidate_indexes"], [0, 1])

    def test_one_statistics_call_serves_every_candidate(self):
        calls = []
        base = stats_for(self.store)
        with mock.patch.object(self.store.index, "describe_index_stats", lambda: (calls.append(1), base())[1]), \
                mock.patch.object(self.store, "namespace_vector_count", side_effect=AssertionError("re-counted")):
            self.assertEqual(match(self.client, [candidate("a"), candidate("b")], JOB).status_code, 200)
        self.assertEqual(len(calls), 1)

    def test_namespace_isolation_only_requested_resumes_are_read(self):
        seen = []
        real = self.store.query
        self.store.query = lambda v, k, ns, filter=None: (seen.append(ns), real(v, k, ns, filter))[1]
        self.assertEqual(match(self.client, [candidate("a")], JOB).status_code, 200)
        self.assertEqual(set(seen), {self.ns["a"]})

    def test_n1_service_accepts_precomputed_counts(self):
        jd = parse_job_description(JOB)
        refs = [CandidateRef(hid(n), self.ns[n]) for n in ("a", "b")]
        counts = {self.ns[n]: self.store.namespace_vector_count(self.ns[n]) for n in ("a", "b")}
        with mock.patch.object(self.store, "namespace_vector_count", side_effect=AssertionError("re-counted")):
            outcome = self.service.match(jd, refs, counts)
        self.assertEqual(outcome.effective_top_k, max(counts.values()))
        partial = self.service.match(jd, refs, {self.ns["a"]: counts[self.ns["a"]]})     # missing key: looked up
        self.assertEqual(len(partial.result.candidates), 2)
        with self.assertRaises(NamespaceCountUnavailableError):
            self.service.match(jd, refs, {self.ns["a"]: -1})

    def test_gate_states_directly(self):
        gate = ReadinessGate(ReadinessTracker(), 200)
        refs = [CandidateRef(hid(n), self.ns[n]) for n in ("a", "b")]
        self.assertEqual(set(gate.verify(self.store, refs)), set(self.ns.values()))
        with mock.patch.object(self.store.index, "describe_index_stats", stats_for(self.store, {self.ns["b"]: 0})):
            with self.assertRaises(CandidatesNotReadyError) as raised:
                gate.verify(self.store, refs)
        self.assertEqual(raised.exception.indexes(), [1])

    def test_tracker_is_bounded(self):
        tracker = ReadinessTracker(max_entries=3)
        for i in range(10):
            tracker.record(str(i), i)
        self.assertIsNone(tracker.expected("0"))
        self.assertEqual(tracker.expected("9"), 9)


class UploadReadinessTests(unittest.TestCase):
    def fake(self, store):
        class Ingestion(FakeIngestion):
            pinecone_store = store
        return Ingestion()

    def test_upload_waits_for_visibility_then_reports_ready(self):
        store = make_store()
        namespace = ingestion_result().namespace
        clock = Clock()
        sleeps = []

        async def sleep(seconds):
            sleeps.append(seconds)
            clock.now += seconds
            if len(sleeps) == 3:
                fill(store, namespace, 4)                          # the vectors become visible on the third poll
        runtime = ApiRuntime(settings_provider=lambda: TEST_SETTINGS, clock=clock, sleep=sleep)
        client, _, _ = build_app(ingestion=self.fake(store), runtime=runtime)
        response = upload(client)
        self.assertEqual((response.status_code, response.json()["status"], response.json()["ready"]), (201, "INGESTED", True))
        self.assertEqual(len(sleeps), 3)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_upload_reports_pending_indexing_when_visibility_is_not_reached(self):
        store = make_store()
        client, runtime, clock = build_app(ingestion=self.fake(store))
        response = upload(client)
        body = response.json()
        self.assertEqual((response.status_code, body["status"], body["ready"]), (202, "PENDING_INDEXING", False))
        self.assertEqual(response.headers["retry-after"], "5")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertTrue(SIGNER.verify(body["document_id"], body["access_token"]))
        self.assertGreaterEqual(clock.now, 1000.0 + TEST_SETTINGS.readiness_timeout_seconds)   # bounded wait
        self.assertEqual(runtime.tracker.expected(body["document_id"]), 4)                     # matching will insist on it

    def test_match_refuses_a_document_that_upload_reported_pending(self):
        service = real_matching({"a": spec("A", ["C#"])})
        client, runtime, _ = build_app(matching=service)
        runtime.tracker.record(hid("a"), 50)
        self.assertEqual(match(client, [candidate("a")], JOB).status_code, 409)


# ======================================================================== M1 / L1: body caps
class BodyCapTests(unittest.TestCase):
    def mounted(self, prefix, limits=ApiLimits(max_upload_bytes=1000)):
        ingestion, reached = FakeIngestion(), []
        inner_client, runtime, _ = build_app(ingestion=ingestion, limits=limits)
        inner = inner_client.app
        inner.dependency_overrides[get_token_signer] = lambda: (reached.append(1), SIGNER)[1]
        outer = FastAPI()
        outer.mount(prefix, inner)
        return TestClient(outer, raise_server_exceptions=False, headers=AUTH), ingestion, reached

    def test_cap_holds_when_the_app_is_mounted_under_a_prefix(self):
        for prefix in ("/svc", "/a/b/c"):
            client, ingestion, reached = self.mounted(prefix)
            response = client.post(f"{prefix}/api/v1/resume/upload",
                                   files={"file": ("a.pdf", PDF + b"x" * 200_000, "application/pdf")})
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (413, "PAYLOAD_TOO_LARGE"), prefix)
            self.assertEqual((ingestion.calls, reached), ([], []), prefix)           # the body was never parsed

    def test_chunked_upload_without_content_length_is_stopped_under_a_prefix(self):
        client, ingestion, reached = self.mounted("/svc")
        boundary = "b0undary"
        head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.pdf\"\r\n"
                "Content-Type: application/pdf\r\n\r\n").encode()

        def chunks():
            yield head
            for _ in range(20):
                yield b"x" * 10_000
            yield f"\r\n--{boundary}--\r\n".encode()
        response = client.post("/svc/api/v1/resume/upload", content=chunks(),
                               headers={"content-type": f"multipart/form-data; boundary={boundary}"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual((ingestion.calls, reached), ([], []))

    def test_every_body_carrying_route_has_a_default_cap(self):
        client, _, _ = build_app(limits=ApiLimits())
        response = client.post("/api/v1/jd/search", content=b"x" * 400_000)           # a route with no explicit cap
        self.assertEqual(response.status_code, 413)
        self.assertEqual(client.post("/api/v1/jd/search", json={}).status_code, 410)    # small bodies still fine
        self.assertEqual(client.get("/nope").status_code, 404)                          # GET carries no body: not capped

    def test_valid_unicode_job_descriptions_are_not_rejected(self):
        client, _, _ = build_app(matching=FakeMatching())
        cases = {"devanagari": "न", "emoji (surrogate pairs)": "\U0001F600", "ascii": "a"}
        for name, character in cases.items():
            text = "Required Skills\n- C#\n" + character * (20_000 - 22)
            for ensure_ascii in (True, False):
                body = json.dumps({"job_description": text, "candidates": [
                    {"document_id": hid(str(i)), "access_token": "0" * 64} for i in range(100)]}, ensure_ascii=ensure_ascii)
                response = client.post("/api/v1/jd/match", content=body.encode(), headers={"content-type": "application/json"})
                self.assertEqual(response.status_code, 403, (name, ensure_ascii, len(body)))   # got past the size cap

    def test_the_hard_memory_bound_remains(self):
        limits = ApiLimits()
        client, _, _ = build_app(matching=FakeMatching())
        response = client.post("/api/v1/jd/match", content=b" " * (limits.max_json_body_bytes + 1),
                               headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 413)
        self.assertLess(limits.max_json_body_bytes, 400_000)

    def test_upload_body_allowance_still_applies(self):
        ingestion = FakeIngestion()
        client, _, _ = build_app(ingestion=ingestion, limits=ApiLimits(max_upload_bytes=1000))
        self.assertEqual(upload(client, PDF + b"x" * 500).status_code, 201)
        self.assertEqual(upload(client, PDF + b"x" * 2000).status_code, 413)             # over the file cap, inside the body cap
        self.assertEqual(len(ingestion.calls), 1)


# ======================================================================== M3 / M4 / M5: concurrency and initialization
class ConcurrencyTests(unittest.TestCase):
    def test_lazy_service_builds_exactly_once_under_concurrent_first_use(self):
        built = []

        def slow():
            built.append(threading.get_ident())
            time.sleep(0.2)
            return object()
        service = LazyService(slow)
        results = []
        threads = [threading.Thread(target=lambda: results.append(service.get_blocking())) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(built), 1)
        self.assertEqual(len({id(r) for r in results}), 1)

    def test_lazy_service_failures_are_not_cached(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("down")
            return "ok"
        service = LazyService(flaky)
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                service.get_blocking()
        self.assertEqual((service.get_blocking(), service.get_blocking(), len(attempts)), ("ok", "ok", 3))

    def test_async_first_use_by_many_requests_builds_once(self):
        built = []
        runner = JobRunner("heavy", 1, 20, 10)

        def builder():
            built.append(1)
            time.sleep(0.1)
            return "svc"
        service = LazyService(builder)

        async def main():
            return await asyncio.gather(*(service.get(runner) for _ in range(10)))
        self.assertEqual(set(asyncio.run(main())), {"svc"})
        self.assertEqual(len(built), 1)

    def test_runner_sheds_load_beyond_its_pending_cap_and_releases_slots(self):
        runner = JobRunner("heavy", 1, 2, 10)
        release = threading.Event()

        async def main():
            first = asyncio.ensure_future(runner.run(release.wait, 5))
            second = asyncio.ensure_future(runner.run(release.wait, 5))
            await asyncio.sleep(0.05)
            with self.assertRaises(ServerBusyError):
                await runner.run(lambda: None)                       # refused immediately, nothing queued
            self.assertEqual(runner.pending, 2)
            release.set()
            await asyncio.gather(first, second)
            await asyncio.sleep(0.05)
            self.assertEqual(runner.pending, 0)
            return await runner.run(lambda: "again")
        self.assertEqual(asyncio.run(main()), "again")

    def test_runner_timeout_keeps_counting_the_running_job(self):
        runner = JobRunner("heavy", 1, 1, 0.1)
        release = threading.Event()

        async def main():
            with self.assertRaises(JobTimeoutError):
                await runner.run(release.wait, 5)
            self.assertEqual(runner.pending, 1)                      # the thread is still busy: the cap stays honest
            with self.assertRaises(ServerBusyError):
                await runner.run(lambda: None)
            release.set()
            await asyncio.sleep(0.1)
            self.assertEqual(runner.pending, 0)
        asyncio.run(main())

    def test_api_answers_busy_and_timeout_without_hanging(self):
        release = threading.Event()

        class Slow(FakeIngestion):
            def ingest_pdf(self, data, name):
                release.wait(5)
                return super().ingest_pdf(data, name)
        client, runtime, _ = build_app(ingestion=Slow())
        runtime._parts["heavy"] = JobRunner("heavy", 1, 1, 0.2)
        timed_out = upload(client)
        self.assertEqual((timed_out.status_code, timed_out.json()["error"]["code"]), (504, "PROCESSING_TIMEOUT"))
        busy = upload(client)                                        # the first job is still running and holds the slot
        self.assertEqual((busy.status_code, busy.json()["error"]["code"]), (503, "SERVER_BUSY"))
        self.assertEqual(busy.headers["retry-after"], "5")
        release.set()

    def test_cheap_routes_stay_responsive_while_the_heavy_worker_is_saturated(self):
        release = threading.Event()

        class Slow(FakeIngestion):
            def ingest_pdf(self, data, name):
                release.wait(5)
                return super().ingest_pdf(data, name)
        client, runtime, _ = build_app(ingestion=Slow())
        runtime._parts["heavy"] = JobRunner("heavy", 1, 3, 5)
        threads = [threading.Thread(target=lambda: upload(client)) for _ in range(3)]
        [t.start() for t in threads]
        time.sleep(0.3)
        started = time.perf_counter()
        self.assertEqual(client.get("/nope").status_code, 404)
        self.assertEqual(client.post("/api/v1/jd/search", json={}).status_code, 410)
        self.assertLess(time.perf_counter() - started, 1.0)
        release.set()
        [t.join() for t in threads]


class ValidationBeforeInitializationTests(unittest.TestCase):
    def test_cheap_validation_failures_never_build_services(self):
        runtime, built = spy_runtime()
        client, _, _ = build_app(runtime=runtime, limits=ApiLimits(max_upload_bytes=1000))
        for response in (upload(client, PDF, "resume.docx"), upload(client, b"", "empty.pdf"),
                         upload(client, PDF + b"x" * 2000, "big.pdf"),
                         match(client, [candidate("a"), candidate("a")], JOB),
                         match(client, [{"document_id": hid("a"), "access_token": "0" * 64}], JOB),
                         match(client, [candidate("a")], "   "),
                         match(client, [candidate("a")], "Role\nResponsibilities\n- Build\n"),
                         match(client, [], JOB)):
            self.assertIn(response.status_code, (400, 403, 413, 415, 422))
        self.assertEqual(built, [])

    def test_quota_and_rate_limit_rejections_never_build_services(self):
        runtime, built = spy_runtime(replace(TEST_SETTINGS, upload_quota_per_day=1, rate_limit_per_minute=100))
        runtime.upload_quota().acquire("tester")                    # the quota is already spent
        client, _, _ = build_app(runtime=runtime)
        self.assertEqual(upload(client).status_code, 429)
        self.assertEqual(built, [])

    def test_valid_requests_build_exactly_what_they_need(self):
        runtime = ApiRuntime(settings_provider=lambda: TEST_SETTINGS)
        order = []
        runtime.ingestion = LazyService(lambda: (order.append("ingestion"), FakeIngestion())[1])
        runtime.matching = LazyService(lambda: (order.append("matching"), FakeMatching())[1])
        app = FastAPI()
        configure_app(app, None, runtime)
        app.dependency_overrides[get_token_signer] = lambda: SIGNER
        from app.api.dependencies import get_readiness
        app.dependency_overrides[get_readiness] = lambda: AlwaysReady()
        client = TestClient(app, raise_server_exceptions=False, headers=AUTH)
        upload(client)
        upload(client)
        self.assertEqual(order, ["ingestion"])                       # built once, and the matching service not at all


# ======================================================================== M6 and partial writes
class ChunkCapTests(unittest.TestCase):
    def service(self, max_chunks, embedder=None, store=None):
        from tests.api.test_pipeline_api import FakeIngestion as _  # noqa: F401  (keeps import order explicit)
        from tests.ingestion.test_pdf_resume_ingestion_service import FakePipeline
        return PdfResumeIngestionService(parser_pipeline=FakePipeline(), embedder=embedder or CountingEmbedder(),
                                         pinecone_store=store or make_store(), limits=IngestionLimits(max_chunks=max_chunks))

    def test_resume_over_the_cap_is_rejected_before_embedding_or_storage(self):
        embedder, store = CountingEmbedder(), make_store()
        embedder.embed_chunks = mock.Mock(side_effect=AssertionError("embedded"))
        service = self.service(2, embedder, store)
        with mock.patch.object(store.index, "upsert") as upsert:
            with self.assertRaises(TooManyChunksError):
                service.ingest_pdf(PDF, "cv.pdf")
        upsert.assert_not_called()

    def test_resume_exactly_at_the_cap_is_accepted(self):
        store = make_store()
        probe = self.service(200, store=store).ingest_pdf(PDF, "cv.pdf").chunk_count
        self.assertEqual(self.service(probe).ingest_pdf(PDF, "cv.pdf").chunk_count, probe)
        with self.assertRaises(TooManyChunksError):
            self.service(probe - 1).ingest_pdf(PDF, "cv.pdf")

    def test_cap_is_validated_and_compatible_with_matching(self):
        for bad in (0, -1, True, 1001, "5"):
            with self.assertRaises(InvalidIngestionConfigError, msg=repr(bad)):
                IngestionLimits(max_chunks=bad)
        with mock.patch.dict(os.environ, {"INGEST_MAX_CHUNKS": "25"}):
            self.assertEqual(IngestionLimits.from_env().max_chunks, 25)
        self.assertEqual(IngestionLimits().max_chunks, PineconeMatchLimits().max_retrieval_depth)   # defaults agree
        with self.assertRaises(SettingsError):
            replace(TEST_SETTINGS, max_chunks=201, max_retrieval_depth=200)

    def test_too_many_chunks_maps_to_422_and_never_reaches_storage(self):
        client, _, _ = build_app(ingestion=FakeIngestion(error=TooManyChunksError("x", "M10")))
        response = upload(client)
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (422, "RESUME_TOO_LARGE"))


class PartialWriteTests(unittest.TestCase):
    def test_storage_failure_is_reported_as_possible_partial_write_not_as_atomic(self):
        client, _, _ = build_app(ingestion=FakeIngestion(error=StorageStageError(f"{SECRET} person@example.com", "M12")))
        response = upload(client)
        error = response.json()["error"]
        self.assertEqual((response.status_code, error["code"]), (502, "STORAGE_UNAVAILABLE"))
        self.assertEqual(error["details"], {"partial_write_possible": True, "retryable": True})
        self.assertIn("may have been written", error["message"])
        self.assertIn("same file again", error["message"])
        self.assertNotIn("atomic", response.text.lower())
        self.assertNotIn(SECRET, response.text)
        self.assertNotIn("access_token", response.text)             # no token is issued for a failed ingestion


# ======================================================================== M2: PDF process isolation
class IsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf = real_pdf(RESUME_ROWS)

    def test_child_process_result_equals_the_in_process_result(self):
        isolated = IsolatedResumePipeline(60, 2 * 1024 ** 3).build(self.pdf, "jane.pdf")
        direct = DeterministicResumePipeline().build(self.pdf, "jane.pdf")
        self.assertEqual(isolated.model_dump(), direct.model_dump())

    def test_corrupt_pdf_is_a_sanitized_processing_error(self):
        with self.assertRaises(PdfProcessingError) as raised:
            IsolatedResumePipeline(60, 2 * 1024 ** 3).build(b"%PDF-1.4\nnot a pdf", "bad.pdf")
        self.assertEqual(str(raised.exception), "M1 failed: PDF object extraction failed")

    def test_a_hung_parser_is_killed_at_the_deadline(self):
        from tests.api import worker_targets
        started = time.perf_counter()
        with self.assertRaises(PdfProcessingTimeoutError):
            IsolatedResumePipeline(1.5, 1024 ** 3, target=worker_targets.sleep_forever).build(b"%PDF-", "x.pdf")
        self.assertLess(time.perf_counter() - started, 15)
        import multiprocessing
        self.assertEqual([p for p in multiprocessing.active_children() if p.is_alive()], [])

    def test_a_crashing_parser_cannot_take_the_server_down(self):
        from tests.api import worker_targets
        with self.assertRaises(PdfResourceError):
            IsolatedResumePipeline(30, 1024 ** 3, target=worker_targets.die_abruptly).build(b"%PDF-", "x.pdf")

    def test_memory_exhaustion_is_contained(self):
        from tests.api import worker_targets
        with self.assertRaises(PdfResourceError):
            IsolatedResumePipeline(60, 256 * 1024 ** 2, target=worker_targets.exhaust_memory).build(b"%PDF-", "x.pdf")

    def test_child_exception_text_never_reaches_the_parent(self):
        from tests.api import worker_targets
        with self.assertRaises(PdfProcessingError) as raised:
            IsolatedResumePipeline(30, 1024 ** 3, target=worker_targets.raise_with_secret).build(b"%PDF-", "x.pdf")
        self.assertNotIn(SECRET, str(raised.exception))
        self.assertNotIn("person@example.com", str(raised.exception))

    def test_full_ingestion_through_the_isolated_pipeline(self):
        store = make_store()
        service = PdfResumeIngestionService(parser_pipeline=IsolatedResumePipeline(60, 2 * 1024 ** 3),
                                            embedder=CountingEmbedder(), pinecone_store=store)
        result = service.ingest_pdf(self.pdf, "jane.pdf")
        self.assertEqual(result.status, "INGESTED")
        self.assertEqual(store.namespace_vector_count(result.namespace), result.chunk_count)

    def test_timeouts_and_resource_errors_map_to_422_responses(self):
        client, _, _ = build_app(ingestion=FakeIngestion(error=PdfProcessingTimeoutError("x", "M1")))
        self.assertEqual(upload(client).json()["error"]["code"], "PDF_PROCESSING_TIMEOUT")
        client, _, _ = build_app(ingestion=FakeIngestion(error=PdfResourceError("x", "M1")))
        response = upload(client)
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (422, "PDF_RESOURCE_LIMIT"))


# ======================================================================== low-severity items
class LowSeverityTests(unittest.TestCase):
    def test_token_responses_are_not_cacheable(self):
        client, _, _ = build_app(ingestion=FakeIngestion())
        response = upload(client)
        self.assertEqual((response.status_code, response.headers["cache-control"], response.headers["pragma"]),
                         (201, "no-store", "no-cache"))
        self.assertIn("access_token", response.json())

    def test_405_keeps_its_allow_header(self):
        client, _, _ = build_app()
        response = client.get("/api/v1/jd/match")
        self.assertEqual((response.status_code, response.headers["allow"]), (405, "POST"))
        self.assertEqual(response.json()["error"]["code"], "METHOD_NOT_ALLOWED")
        self.assertEqual(client.post("/nope").status_code, 404)
        upload_get = client.get("/api/v1/resume/upload")
        self.assertEqual((upload_get.status_code, upload_get.headers["allow"]), (405, "POST"))

    def test_malformed_and_non_ascii_tokens_never_raise(self):
        good = SIGNER.issue(hid("a"))
        for token in ("é" * 64, "١" * 64, "", good[:-1], good + "0", None, 5, b"x", good.upper()):
            self.assertFalse(SIGNER.verify(hid("a"), token), repr(token))
        self.assertFalse(SIGNER.verify(None, good))
        self.assertFalse(SIGNER.verify("é", good))
        self.assertTrue(SIGNER.verify(hid("a"), good))
        client, _, _ = build_app(matching=FakeMatching())
        for token in ("é" * 64, "x" * 64, "0" * 63):
            response = match(client, [{"document_id": hid("a"), "access_token": token}])
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (422, "INVALID_REQUEST"))

    def test_validation_errors_do_not_echo_caller_chosen_field_names_or_values(self):
        client, _, _ = build_app(matching=FakeMatching())
        response = client.post("/api/v1/jd/match", json={
            "job_description": JD, "candidates": [{**candidate("a"), "<script>alert(1)</script>": SECRET}],
            "namespace": f"resume:{SECRET}", SECRET: 1})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(SECRET, response.text)
        self.assertNotIn("<script>", response.text)
        fields = {f["field"] for f in response.json()["error"]["details"]["fields"]}
        self.assertTrue(all(part in ("candidates", "0", "document_id", "access_token", "job_description", "(unknown)", "")
                            for field in fields for part in field.split(".")), fields)
        self.assertIn("(unknown)", fields)
        known = client.post("/api/v1/jd/match", json={"job_description": JD, "candidates": [
            {"document_id": "bad", "access_token": "bad"}]})
        self.assertEqual({f["field"] for f in known.json()["error"]["details"]["fields"]},
                         {"candidates.0.document_id", "candidates.0.access_token"})


class CorsTests(unittest.TestCase):
    ORIGIN = "https://app.example.com"

    def client(self, origins=(ORIGIN,)):
        client, _, _ = build_app(settings=replace(TEST_SETTINGS, cors_origins=origins), ingestion=FakeIngestion(),
                                 with_headers=False)
        return client

    def test_allowlisted_origin_gets_exact_headers_and_no_credentials(self):
        client = self.client()
        preflight = client.options("/api/v1/jd/match", headers={
            "Origin": self.ORIGIN, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type"})
        self.assertEqual(preflight.status_code, 204)
        self.assertEqual(preflight.headers["access-control-allow-origin"], self.ORIGIN)
        self.assertNotIn("access-control-allow-credentials", preflight.headers)
        self.assertEqual(preflight.headers["vary"], "Origin")
        self.assertIn("Authorization", preflight.headers["access-control-allow-headers"])
        actual = client.post("/api/v1/resume/upload", files={"file": ("a.pdf", PDF, "application/pdf")},
                             headers={**AUTH, "Origin": self.ORIGIN})
        self.assertEqual(actual.status_code, 201)
        self.assertEqual(actual.headers["access-control-allow-origin"], self.ORIGIN)
        denied = client.post("/api/v1/resume/upload", headers={"Origin": self.ORIGIN})     # errors carry CORS too
        self.assertEqual((denied.status_code, denied.headers["access-control-allow-origin"]), (401, self.ORIGIN))

    def test_other_origins_get_no_cors_headers(self):
        client = self.client()
        for origin in ("https://evil.example.com", "http://app.example.com", "https://app.example.com.evil.com", "null",
                       "https://app.example.com:8443"):
            for response in (client.options("/api/v1/jd/match", headers={"Origin": origin,
                                                                         "Access-Control-Request-Method": "POST"}),
                             client.post("/api/v1/jd/match", headers={"Origin": origin, **AUTH}, json={})):
                self.assertNotIn("access-control-allow-origin", response.headers, origin)
        self.assertNotIn("access-control-allow-origin", self.client(()).get("/", headers={"Origin": self.ORIGIN}).headers)

    def test_wildcards_are_rejected_at_configuration_time(self):
        for bad in ("*", "https://*.example.com", "https://app.example.com/path", "ftp://x", "app.example.com"):
            with self.assertRaises(SettingsError, msg=bad):
                parse_origins(bad)
        self.assertEqual(parse_origins(""), ())


class RegistrationAndDocsTests(unittest.TestCase):
    def test_routes_registered_once_and_import_stays_clean(self):
        code = ("import sys, collections, app.main as m\n"
                "c = collections.Counter((x, r.path) for r in m.app.routes for x in (getattr(r, 'methods', None) or ()))\n"
                "dup = [k for k, v in c.items() if v > 1]\n"
                "bad = [n for n in ('sentence_transformers', 'torch', 'pinecone', 'requests', 'fitz', 'dotenv', "
                "'app.services.embedding_service', 'app.services.ai.ollama_service', 'app.api.v1.jd', 'app.api.v1.resume', "
                "'app.core.config', 'multiprocessing.spawn') if n in sys.modules]\n"
                "print(dup, bad)\n")
        done = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "[] []"), done.stderr[-400:])

    def test_docs_are_on_in_development_and_can_be_switched_off(self):
        code = "import app.main as m\nprint(m.app.openapi_url, m.app.docs_url, m.app.redoc_url)\n"
        run = lambda extra: subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True,
                                           timeout=120, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **extra})
        self.assertEqual(run({"API_DOCS_ENABLED": "true"}).stdout.strip(), "/openapi.json /docs /redoc")
        self.assertEqual(run({"API_DOCS_ENABLED": "false"}).stdout.strip(), "None None None")


class RuntimeWiringTests(unittest.TestCase):
    """The production builders, run with the model and Pinecone client replaced by in-memory fakes."""

    def test_builders_apply_the_settings_and_build_each_part_once(self):
        settings = replace(TEST_SETTINGS, max_chunks=50, max_retrieval_depth=80, parse_timeout_seconds=7, parse_memory_mb=300)
        runtime = ApiRuntime(settings_provider=lambda: settings)
        fake_store, fake_embedder, created = make_store(), CountingEmbedder(), []
        with mock.patch("app.parsing.embeddings.NomicEmbedder", side_effect=lambda: (created.append("embedder"), fake_embedder)[1]),                 mock.patch("app.parsing.vector_store.PineconeStore", side_effect=lambda config: (created.append("store"), fake_store)[1]),                 mock.patch("app.parsing.vector_store.VectorStoreConfig.from_env", return_value=object()):
            ingestion, matching = runtime.ingestion.get_blocking(), runtime.matching.get_blocking()
            self.assertIs(runtime.ingestion.get_blocking(), ingestion)
        self.assertEqual(sorted(created), ["embedder", "store"])               # shared by both services, built once each
        self.assertIs(ingestion.embedder, matching.embedder)
        self.assertIs(ingestion.pinecone_store, matching.store)
        self.assertEqual((ingestion.limits.max_chunks, ingestion.limits.max_pdf_bytes), (50, settings.max_upload_bytes))
        self.assertIsInstance(ingestion.parser_pipeline, IsolatedResumePipeline)
        self.assertEqual((ingestion.parser_pipeline.timeout, ingestion.parser_pipeline.memory_bytes), (7.0, 300 * 1024 ** 2))
        self.assertEqual(matching.limits.max_retrieval_depth, 80)

    def test_runners_are_sized_from_the_settings(self):
        runtime = ApiRuntime(settings_provider=lambda: replace(TEST_SETTINGS, max_pending_jobs=5, job_timeout_seconds=9))
        heavy = runtime.heavy()
        self.assertEqual((heavy.max_pending, heavy.timeout, runtime.heavy() is heavy), (5, 9.0, True))
        self.assertEqual(runtime.light().name, "light")


class FrozenManifestTests(unittest.TestCase):
    def test_m13_frozen_manifest_still_verifies(self):
        lines = (ROOT / "docs/m13_frozen_manifest.sha256").read_text(encoding="utf-8").splitlines()
        entries = [line.split(" *", 1) for line in lines if line.strip()]
        self.assertGreaterEqual(len(entries), 17)
        self.assertEqual([p for d, p in entries if sha256((ROOT / p).read_bytes()).hexdigest() != d], [])


if __name__ == "__main__":
    unittest.main()
