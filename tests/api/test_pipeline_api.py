"""N3 API tests. Offline: fake ingestion/matching services or the in-memory Pinecone fake; no credentials, no model."""

import asyncio
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import dependencies
from app.api.runtime import ApiRuntime
from app.api.runner import LazyService
from app.api.settings import ApiSettings
from app.api.app_factory import configure_app
from app.api.body_limit import BodySizeLimitMiddleware
from app.api.candidate_access import CandidateTokenSigner
from app.api.dependencies import (ApiLimits, get_ingestion_service, get_matching_service, get_readiness,
                                  get_token_signer)
from app.api.v1.jd_matching import retired_search, safe_excerpt
from app.ingestion.pdf_resume_ingestion_service import (
    EmbeddingStageError, EmptyPdfInputError, EmptyResumeContentError, IngestionError, IngestionResult,
    InconsistentPipelineOutputError, InvalidPdfInputError, NotAPdfError, PdfProcessingError, PdfTooLargeError,
    StorageStageError)
from app.matching import MatchingConfig
from app.matching.models import MatchingError, QueryEmbeddingError, RetrievalError
from app.matching.pinecone_canonical_adapter import CanonicalReconstructionError, MetadataRetrievalError
from app.matching.pinecone_match_service import (NamespaceCountUnavailableError, PineconeMatchLimits,
                                                  PineconeMatchService, RetrievalDepthExceededError)
from app.matching.retrieval import ResumeRetriever
from tests.parsing.matching.fakes import CountingEmbedder, make_store
from tests.parsing.matching.test_pinecone_canonical_adapter import world
from tests.parsing.matching.test_skill_ranking import spec

ROOT = Path(__file__).resolve().parents[2]
SIGNER = CandidateTokenSigner("unit-test-secret-" + "x" * 32)
SECRET = "SECRET-TOKEN-123"
EMAIL = "person@example.com"
PDF = b"%PDF-1.4\n% synthetic upload\n"
JD = ("Backend Engineer\nRequired Skills\n- C#\n- SQL\n- Docker\nPreferred Skills\n- Azure\n"
      "Responsibilities\n- Build backend services\n")
CONFIG = MatchingConfig(top_k=10, semantic_match_threshold=0.05, strong_evidence_threshold=0.5,
                        required_skill_match_threshold=0.05, preferred_skill_match_threshold=0.05, as_of="2026-10")


def hid(name):
    return sha256(name.encode()).hexdigest()


def candidate(name):
    return {"document_id": hid(name), "access_token": SIGNER.issue(hid(name))}


def ingestion_result(replaced=False):
    digest = sha256(PDF).hexdigest()
    return IngestionResult(status="INGESTED", document_id=digest, namespace=f"resume:{digest}", chunk_count=4,
                           vectors_upserted=4, page_count=1, source_filename="jane.pdf", embedding_model="nomic",
                           embedding_dimension=768, replaced_existing=replaced)


class FakeIngestion:
    def __init__(self, error=None, replaced=False):
        self.error, self.replaced, self.calls = error, replaced, []

    def ingest_pdf(self, data, name):
        self.calls.append((data, name, threading.current_thread().name))
        if self.error:
            raise self.error
        return ingestion_result(self.replaced)


class FakeMatching:
    def __init__(self, error=None):
        self.error, self.calls, self.store = error, [], make_store()

    def match(self, jd, refs):
        self.calls.append(refs)
        raise self.error


API_KEY = "n3-test-api-key-" + "k" * 24
AUTH = {"Authorization": f"Bearer {API_KEY}"}
TEST_SETTINGS = ApiSettings(api_key_hashes={"tester": sha256(API_KEY.encode()).hexdigest()},
                            token_secret="unit-test-secret-" + "x" * 32)


class AlwaysReady:
    """These tests are about request handling, not index freshness (that is covered in test_n4_hardening)."""

    def verify(self, store, refs):
        return {}

    async def wait_until_indexed(self, *args, **kwargs):
        return True


def make_client(ingestion=None, matching=None, signer=SIGNER, limits=None, settings=TEST_SETTINGS):
    app = FastAPI()
    runtime = ApiRuntime(settings_provider=lambda: settings)
    configure_app(app, (lambda: limits) if limits else None, runtime)
    app.dependency_overrides[get_token_signer] = lambda: signer
    app.dependency_overrides[get_readiness] = lambda: AlwaysReady()
    if ingestion is not None:
        app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    if matching is not None:
        app.dependency_overrides[get_matching_service] = lambda: matching
    return TestClient(app, raise_server_exceptions=False, headers=AUTH)


def upload(client, data=PDF, name="jane.pdf"):
    return client.post("/api/v1/resume/upload", files={"file": (name, data, "application/pdf")})


def real_matching(docs, cap=200):
    """docs: {name: specs}. The real N1 service over the in-memory Pinecone fake, keyed by sha256 document ids."""
    store, _, _ = world({hid(name): specs for name, specs in docs.items()})
    return PineconeMatchService(store, CountingEmbedder(), CONFIG, PineconeMatchLimits(cap))


def match(client, candidates, text=JD, **extra):
    return client.post("/api/v1/jd/match", json={"job_description": text, "candidates": candidates, **extra})


class UploadTests(unittest.TestCase):
    def test_successful_upload(self):
        ingestion = FakeIngestion()
        response = upload(make_client(ingestion))
        self.assertEqual(response.status_code, 201)
        body = response.json()
        digest = sha256(PDF).hexdigest()
        self.assertEqual((body["status"], body["document_id"], body["namespace"], body["chunk_count"]),
                         ("INGESTED", digest, f"resume:{digest}", 4))
        self.assertTrue(SIGNER.verify(digest, body["access_token"]))
        self.assertNotIn("source_filename", body)
        self.assertNotIn("jane", response.text)
        self.assertEqual([(d, n) for d, n, _ in ingestion.calls], [(PDF, "jane.pdf")])

    def test_repeat_upload_returns_200(self):
        self.assertEqual(upload(make_client(FakeIngestion(replaced=True))).status_code, 200)

    def test_ingestion_runs_on_the_bounded_heavy_worker_not_the_event_loop(self):
        ingestion = FakeIngestion()
        upload(make_client(ingestion))
        self.assertTrue(ingestion.calls[0][2].startswith("api-heavy"))

    def test_unsupported_file_types_are_rejected_before_ingestion(self):
        ingestion = FakeIngestion()
        client = make_client(ingestion)
        for name in ("resume.docx", "resume.txt", "resume", "resume.pdf.exe"):
            response = upload(client, PDF, name)
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (415, "UNSUPPORTED_MEDIA_TYPE"), name)
        self.assertEqual(ingestion.calls, [])

    def test_empty_upload_is_rejected(self):
        ingestion = FakeIngestion()
        response = upload(make_client(ingestion), b"")
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (400, "EMPTY_FILE"))
        self.assertEqual(ingestion.calls, [])

    def test_missing_file_field_is_rejected_without_echo(self):
        response = make_client(FakeIngestion()).post("/api/v1/resume/upload", data={"note": SECRET})
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (422, "INVALID_REQUEST"))
        self.assertNotIn(SECRET, response.text)

    def test_oversized_upload_is_rejected_before_ingestion(self):
        ingestion = FakeIngestion()
        client = make_client(ingestion, limits=ApiLimits(max_upload_bytes=1000))
        response = upload(client, PDF + b"x" * 70_000)                     # over the cap plus multipart allowance
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (413, "PAYLOAD_TOO_LARGE"))
        self.assertEqual(ingestion.calls, [])

    def test_oversized_upload_without_content_length_is_rejected(self):
        ingestion = FakeIngestion()
        client = make_client(ingestion, limits=ApiLimits(max_upload_bytes=1000))
        boundary = "b0undary"
        head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.pdf\"\r\n"
                "Content-Type: application/pdf\r\n\r\n").encode()

        def chunks():
            yield head
            for _ in range(10):
                yield b"x" * 10_000
            yield f"\r\n--{boundary}--\r\n".encode()

        response = client.post("/api/v1/resume/upload", content=chunks(),
                               headers={"content-type": f"multipart/form-data; boundary={boundary}"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(ingestion.calls, [])

    def test_file_between_the_cap_and_the_multipart_allowance_is_still_rejected(self):
        ingestion = FakeIngestion()
        client = make_client(ingestion, limits=ApiLimits(max_upload_bytes=1000))
        response = upload(client, PDF + b"x" * 2000)                       # inside the body cap, over the file cap
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (413, "PAYLOAD_TOO_LARGE"))
        self.assertEqual(ingestion.calls, [])

    def test_ingestion_errors_map_to_sanitized_responses(self):
        cases = ((EmptyPdfInputError, 400, "EMPTY_FILE"), (PdfTooLargeError, 413, "PAYLOAD_TOO_LARGE"),
                 (NotAPdfError, 415, "UNSUPPORTED_MEDIA_TYPE"), (InvalidPdfInputError, 400, "INVALID_FILE"),
                 (PdfProcessingError, 422, "PDF_UNPROCESSABLE"), (EmptyResumeContentError, 422, "NO_TEXT_CONTENT"),
                 (EmbeddingStageError, 503, "EMBEDDING_UNAVAILABLE"), (StorageStageError, 502, "STORAGE_UNAVAILABLE"),
                 (InconsistentPipelineOutputError, 500, "INGESTION_FAILED"), (IngestionError, 500, "INGESTION_FAILED"))
        for kind, status, code in cases:
            error = kind(f"{SECRET} {EMAIL} namespace resume:abc /home/user/cv.pdf", "M1")
            response = upload(make_client(FakeIngestion(error=error)))
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (status, code), kind.__name__)
            for leaked in (SECRET, EMAIL, "resume:abc", "/home/user", "Traceback"):
                self.assertNotIn(leaked, response.text)

    def test_unexpected_errors_are_sanitized_in_the_response_and_the_log(self):
        client = make_client(FakeIngestion(error=RuntimeError(f"{SECRET} {EMAIL}")))
        with self.assertLogs("app.api", level="ERROR") as captured:
            response = upload(client)
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (500, "INGESTION_FAILED"))
        for text in (response.text, " ".join(captured.output)):
            self.assertNotIn(SECRET, text)
            self.assertNotIn(EMAIL, text)

    def test_unhandled_exceptions_use_the_same_envelope(self):
        app = FastAPI()
        configure_app(app)

        @app.get("/boom")
        async def boom():
            raise RuntimeError(f"{SECRET}")
        response = TestClient(app, raise_server_exceptions=False).get("/boom")
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (500, "INTERNAL_ERROR"))
        self.assertNotIn(SECRET, response.text)

    def test_missing_or_short_signing_secret_fails_closed_with_503(self):
        for secret in (None, "short"):
            client = make_client(FakeIngestion(), signer=None, settings=ApiSettings(
                api_key_hashes=TEST_SETTINGS.api_key_hashes, token_secret=secret))
            client.app.dependency_overrides.pop(get_token_signer)
            response = upload(client)
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (503, "SERVICE_UNAVAILABLE"))


class MatchTests(unittest.TestCase):
    def test_valid_request_returns_skills_only_ranking(self):
        service = real_matching({"a": spec("A", ["C#", "SQL", "Docker"]), "b": spec("B", ["C#"])})
        response = match(make_client(matching=service), [candidate("b"), candidate("a")])
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["ranking_basis"], "skills-only")
        self.assertEqual(body["jd"]["required_skills"], ["C#", "SQL", "Docker"])
        self.assertEqual(body["jd"]["preferred_skills"], ["Azure"])
        self.assertEqual([(c["rank"], c["document_id"]) for c in body["candidates"]], [(1, hid("a")), (2, hid("b"))])
        top = body["candidates"][0]
        self.assertEqual((top["required_skill_coverage"], top["preferred_skill_coverage"]), (1.0, 0.0))
        self.assertEqual([s["status"] for s in top["required_skills"]], ["EXPLICIT_SKILL"] * 3)
        self.assertEqual(top["missing_skills"], {"required": [], "preferred": ["Azure"]})
        self.assertEqual(top["ranking_key"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(top["required_skills"][0]["evidence"]["canonical_path"], "skills[0]")
        second = body["candidates"][1]
        self.assertEqual(second["missing_skills"]["required"], ["SQL", "Docker"])
        self.assertIn(body["retrieval"]["effective_top_k"], range(1, 20))
        for forbidden in ("experience", "education", "similarity", "namespace"):
            self.assertNotIn(forbidden, response.text.lower())

    def test_required_coverage_outranks_preferred_coverage(self):
        job = "Role\nRequired Skills\n- C#\n- SQL\nPreferred Skills\n- Docker\n- Azure\n- Angular\nResponsibilities\n- Build\n"
        service = real_matching({"required_only": spec("A", ["C#", "SQL"]),
                                 "all_preferred": spec("B", ["C#", "Docker", "Azure", "Angular"])})
        body = match(make_client(matching=service), [candidate("all_preferred"), candidate("required_only")], job).json()
        self.assertEqual([c["document_id"] for c in body["candidates"]], [hid("required_only"), hid("all_preferred")])
        self.assertEqual([c["rank"] for c in body["candidates"]], [1, 2])
        self.assertGreater(body["candidates"][1]["preferred_skill_coverage"], body["candidates"][0]["preferred_skill_coverage"])

    def test_preferred_skills_separate_candidates_tied_on_required(self):
        job = "Role\nRequired Skills\n- C#\n- SQL\nPreferred Skills\n- Docker\n- Azure\nResponsibilities\n- Build\n"
        service = real_matching({"more": spec("A", ["C#", "SQL", "Docker", "Azure"]), "none": spec("B", ["C#", "SQL"])})
        body = match(make_client(matching=service), [candidate("none"), candidate("more")], job).json()
        self.assertEqual([c["document_id"] for c in body["candidates"]], [hid("more"), hid("none")])

    def test_semantic_only_evidence_is_not_coverage(self):
        job = "Role\nRequired Skills\n- SQL\n- Java\n- ASP.NET Core\nResponsibilities\n- Build\n"
        service = real_matching({"c": spec("C", ["SQL Server", "JavaScript", "Entity Framework Core"])})
        body = match(make_client(matching=service), [candidate("c")], job).json()
        result = body["candidates"][0]
        self.assertTrue(all(not s["covered"] for s in result["required_skills"]))
        self.assertEqual(result["required_skill_coverage"], 0.0)
        self.assertEqual(result["missing_skills"]["required"], ["SQL", "Java", "ASP.NET Core"])

    def test_response_matches_the_service_ranking_exactly(self):
        service = real_matching({"a": spec("A", ["C#", "SQL"]), "b": spec("B", ["C#", "SQL", "Docker", "Azure"])})
        from app.matching import parse_job_description
        from app.matching.pinecone_match_service import CandidateRef
        refs = [CandidateRef(hid(n), service.store.namespace_for(hid(n))) for n in ("a", "b")]
        expected = service.match(parse_job_description(JD), refs).result.candidates
        body = match(make_client(matching=service), [candidate("a"), candidate("b")]).json()
        self.assertEqual([(c.document_id, c.rank, list(c.skill_score.key)) for c in expected],
                         [(c["document_id"], c["rank"], c["ranking_key"]) for c in body["candidates"]])

    def test_invalid_and_empty_candidate_lists(self):
        client = make_client(matching=FakeMatching())
        self.assertEqual(match(client, []).status_code, 422)
        self.assertEqual(client.post("/api/v1/jd/match", json={"job_description": JD}).status_code, 422)
        self.assertEqual(client.post("/api/v1/jd/match", json={"candidates": [candidate("a")]}).status_code, 422)
        too_many = [candidate(str(i)) for i in range(101)]
        self.assertEqual(match(client, too_many).status_code, 422)
        for bad in ({"document_id": "resume:abc", "access_token": "x" * 64},
                    {"document_id": hid("a").upper(), "access_token": hid("a")},
                    {"document_id": hid("a"), "access_token": "short"}, {"document_id": hid("a")}, "string", None):
            response = match(client, [bad])
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (422, "INVALID_REQUEST"), bad)

    def test_invalid_job_descriptions(self):
        matching = FakeMatching()
        client = make_client(matching=matching)
        for text, code in (("   ", "INVALID_JOB_DESCRIPTION"), ("Just some prose with no sections.", "INVALID_JOB_DESCRIPTION"),
                           ("Role\nResponsibilities\n- Build things\n", "NO_SKILLS_IN_JOB_DESCRIPTION")):
            response = match(client, [candidate("a")], text)
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (422, code), text)
        self.assertEqual(matching.calls, [])
        self.assertEqual(match(client, [candidate("a")], "").status_code, 422)
        self.assertEqual(match(client, [candidate("a")], "x" * 20_001).status_code, 422)

    def test_duplicate_candidates_are_rejected_before_matching(self):
        matching = FakeMatching()
        response = match(make_client(matching=matching), [candidate("a"), candidate("b"), candidate("a")])
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (422, "DUPLICATE_CANDIDATE"))
        self.assertEqual(response.json()["error"]["details"], {"candidate_indexes": [2]})
        self.assertEqual(matching.calls, [])

    def test_unauthorized_candidates_never_reach_pinecone(self):
        matching = FakeMatching()
        client = make_client(matching=matching)
        forged = {"document_id": hid("a"), "access_token": "0" * 64}
        swapped = {"document_id": hid("c"), "access_token": candidate("b")["access_token"]}    # another resume's token
        for refs, indexes in (([forged], [0]), ([candidate("b"), swapped], [1]), ([forged, swapped], [0, 1])):
            response = match(client, refs)
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (403, "UNAUTHORIZED_CANDIDATE"))
            self.assertEqual(response.json()["error"]["details"], {"candidate_indexes": indexes})
        self.assertEqual(matching.calls, [])
        other = CandidateTokenSigner("another-server-secret-" + "y" * 32)       # tokens from another deployment
        response = match(client, [{"document_id": hid("a"), "access_token": other.issue(hid("a"))}])
        self.assertEqual(response.status_code, 403)

    def test_caller_supplied_namespaces_are_never_accepted(self):
        matching = FakeMatching()
        client = make_client(matching=matching)
        smuggled = {**candidate("a"), "namespace": "resume:someone-else"}
        for payload in ({"job_description": JD, "candidates": [smuggled]},
                        {"job_description": JD, "candidates": [candidate("a")], "namespace": "resume:x"},
                        {"job_description": JD, "candidates": [candidate("a")], "namespaces": ["resume:x"]}):
            response = client.post("/api/v1/jd/match", json=payload)
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (422, "INVALID_REQUEST"))
        self.assertEqual(matching.calls, [])

    def test_namespace_isolation_only_requested_resumes_are_queried(self):
        service = real_matching({"a": spec("A", ["C#", "SQL", "Docker"]), "b": spec("B", ["C#"]),
                                 "c": spec("C", ["Azure"])})
        seen = []
        real_query = service.store.query
        service.store.query = lambda vector, top_k, namespace, filter=None: (seen.append(namespace),
                                                                              real_query(vector, top_k, namespace, filter))[1]
        with mock.patch.object(ResumeRetriever, "discover_namespaces", side_effect=AssertionError("discovery")):
            body = match(make_client(matching=service), [candidate("a")]).json()
        self.assertEqual([c["document_id"] for c in body["candidates"]], [hid("a")])
        self.assertEqual(set(seen), {f"resume:{hid('a')}"})
        both = match(make_client(matching=service), [candidate("a"), candidate("b")]).json()
        a_alone = body["candidates"][0]
        a_with_b = next(c for c in both["candidates"] if c["document_id"] == hid("a"))
        self.assertEqual((a_alone["ranking_key"], a_alone["required_skills"]), (a_with_b["ranking_key"], a_with_b["required_skills"]))

    def test_unknown_but_authorized_resume_is_reported_not_hidden(self):
        service = real_matching({"a": spec("A", ["C#"])})
        body = match(make_client(matching=service), [candidate("a"), candidate("never-uploaded")]).json()
        ghost = next(c for c in body["candidates"] if c["document_id"] == hid("never-uploaded"))
        self.assertEqual((ghost["warnings"], ghost["required_skill_coverage"]), (["NO_RESUME_VECTORS"], 0.0))
        self.assertEqual(body["candidates"][-1]["document_id"], hid("never-uploaded"))

    def test_matching_errors_map_to_sanitized_responses(self):
        cases = ((RetrievalDepthExceededError, 422, "CANDIDATE_TOO_LARGE"), (NamespaceCountUnavailableError, 502, "RETRIEVAL_UNAVAILABLE"),
                 (MetadataRetrievalError, 502, "RETRIEVAL_UNAVAILABLE"), (RetrievalError, 502, "RETRIEVAL_UNAVAILABLE"),
                 (QueryEmbeddingError, 503, "EMBEDDING_UNAVAILABLE"), (CanonicalReconstructionError, 500, "MATCHING_FAILED"),
                 (MatchingError, 500, "MATCHING_FAILED"), (RuntimeError, 500, "MATCHING_FAILED"))
        for kind, status, code in cases:
            matching = FakeMatching(error=kind(f"{SECRET} {EMAIL} resume:abc"))
            response = match(make_client(matching=matching), [candidate("a")])
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (status, code), kind.__name__)
            for leaked in (SECRET, EMAIL, "resume:abc"):
                self.assertNotIn(leaked, response.text)
            self.assertEqual(len(matching.calls), 1)

    def test_validation_errors_never_echo_the_request(self):
        response = make_client(matching=FakeMatching()).post("/api/v1/jd/match", json={
            "job_description": f"{SECRET} {EMAIL}", "candidates": [{"document_id": SECRET, "access_token": EMAIL}]})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(SECRET, response.text)
        self.assertNotIn(EMAIL, response.text)

    def test_oversized_json_body_is_rejected(self):
        client = make_client(matching=FakeMatching(), limits=ApiLimits(max_jd_chars=100))
        response = client.post("/api/v1/jd/match", content=b"{" + b" " * 100_000 + b"}",
                               headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 413)

    def test_excerpts_redact_contact_details_and_are_short(self):
        text = f"Reach me at {EMAIL} or +91 98765 43210 about   C#\n and SQL. " + "x" * 400
        excerpt = safe_excerpt(text)
        self.assertNotIn(EMAIL, excerpt)
        self.assertNotIn("98765", excerpt)
        self.assertIn("[email]", excerpt)
        self.assertIn("[phone]", excerpt)
        self.assertLessEqual(len(excerpt), 300)
        self.assertIsNone(safe_excerpt(None))


class LegacyAndRegistrationTests(unittest.TestCase):
    def test_legacy_search_is_retired_not_delegated(self):
        client = make_client(matching=FakeMatching(), ingestion=FakeIngestion())
        for body in ({"job_description": JD}, {}, None):
            response = client.post("/api/v1/jd/search", json=body)
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (410, "ENDPOINT_RETIRED"))
            self.assertIn("/api/v1/jd/match", response.json()["error"]["message"])
        self.assertEqual(client.get("/api/v1/jd/search").status_code, 405)

    def test_application_import_loads_no_legacy_ollama_or_model_modules(self):
        code = ("import sys, app.main\n"
                "bad = [n for n in ('sentence_transformers', 'torch', 'pinecone', 'requests', 'fitz', "
                "'app.services.embedding_service', 'app.services.ai.ollama_service', 'app.services.job_search_service', "
                "'app.services.pinecone_service', 'app.api.v1.jd', 'app.api.v1.resume', 'app.core.config') if n in sys.modules]\n"
                "print(','.join(bad))\n")
        done = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-500:])
        self.assertEqual(done.stdout.strip(), "")

    def test_routes_are_registered_on_the_real_app(self):
        from app.main import app
        routes = {(method, route.path): route for route in app.routes for method in getattr(route, "methods", ())}
        self.assertIn(("POST", "/api/v1/resume/upload"), routes)
        self.assertIn(("POST", "/api/v1/jd/match"), routes)
        self.assertIn(("GET", "/"), routes)
        self.assertIs(routes[("POST", "/api/v1/jd/search")].endpoint, retired_search)
        self.assertEqual(routes[("POST", "/api/v1/resume/upload")].endpoint.__module__, "app.api.v1.resume_upload")

    def test_runtime_builds_nothing_until_used(self):
        runtime = ApiRuntime(settings_provider=lambda: TEST_SETTINGS)
        self.assertFalse(any(part.built for part in (runtime.embedder, runtime.store, runtime.ingestion, runtime.matching)))

    def test_service_build_failures_are_sanitized_and_not_cached(self):
        runtime = ApiRuntime(settings_provider=lambda: TEST_SETTINGS)
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError(SECRET)
            return FakeIngestion()
        runtime.ingestion = LazyService(flaky)
        app = FastAPI()
        configure_app(app, None, runtime)
        app.dependency_overrides[get_token_signer] = lambda: SIGNER
        app.dependency_overrides[get_readiness] = lambda: AlwaysReady()
        client = TestClient(app, raise_server_exceptions=False, headers=AUTH)
        first = upload(client)
        self.assertEqual((first.status_code, first.json()["error"]["code"]), (503, "SERVICE_UNAVAILABLE"))
        self.assertNotIn(SECRET, first.text)
        self.assertEqual(upload(client).status_code, 201)            # the failed build was not cached
        self.assertEqual(len(attempts), 2)

    def test_unknown_routes_use_the_error_envelope(self):
        response = make_client().get("/nope")
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (404, "NOT_FOUND"))


class BodyLimitMiddlewareTests(unittest.TestCase):
    def run_request(self, limit, chunks, headers=()):
        sent, consumed = [], []

        async def inner(scope, receive, send):
            while True:
                message = await receive()
                consumed.append(len(message.get("body", b"")))
                if not message.get("more_body"):
                    break
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        queue = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1} for i, c in enumerate(chunks)]

        async def receive():
            return queue.pop(0)

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "method": "POST", "path": "/p", "headers": list(headers)}
        asyncio.run(BodySizeLimitMiddleware(inner, lambda method, path: limit)(scope, receive, send))
        return sent, consumed, queue

    def test_streamed_body_is_aborted_as_soon_as_it_passes_the_cap(self):
        sent, consumed, remaining = self.run_request(100, [b"x" * 60] * 10)
        self.assertEqual(sent[0]["status"], 413)
        self.assertEqual(consumed, [60])                           # the second chunk crossed the cap and was refused
        self.assertEqual(len(remaining), 8)                        # the other eight chunks were never read

    def test_declared_length_over_the_cap_is_rejected_without_reading(self):
        sent, consumed, _ = self.run_request(100, [b"x"], [(b"content-length", b"101")])
        self.assertEqual((sent[0]["status"], consumed), (413, []))
        sent, consumed, _ = self.run_request(100, [b"x"], [(b"content-length", b"abc")])
        self.assertEqual(sent[0]["status"], 400)

    def test_body_within_the_cap_and_unlimited_paths_pass_through(self):
        self.assertEqual(self.run_request(100, [b"x" * 50, b"y" * 50])[0][0]["status"], 200)
        self.assertEqual(self.run_request(None, [b"x" * 5000])[0][0]["status"], 200)


class FrozenManifestTests(unittest.TestCase):
    def test_m13_frozen_manifest_still_verifies(self):
        lines = (ROOT / "docs/m13_frozen_manifest.sha256").read_text(encoding="utf-8").splitlines()
        entries = [line.split(" *", 1) for line in lines if line.strip()]
        self.assertGreaterEqual(len(entries), 17)
        self.assertEqual([p for d, p in entries if sha256((ROOT / p).read_bytes()).hexdigest() != d], [])


if __name__ == "__main__":
    unittest.main()
