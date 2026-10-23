"""POST /api/v1/jd/rank: server-side discovery over the shared pool. Offline: in-memory Pinecone fake, no credentials."""

import unittest

from app.api.dependencies import get_readiness
from app.api.readiness import ReadinessGate, ReadinessTracker
from app.matching.pinecone_match_service import PineconeMatchLimits, PineconeMatchService
from tests.api.test_pipeline_api import AUTH, CONFIG, JD, hid, make_client
from tests.parsing.matching.fakes import CountingEmbedder, build_candidate, ingest
from tests.parsing.matching.test_pinecone_canonical_adapter import world
from tests.parsing.matching.test_skill_ranking import spec


def pool_service(docs, cap=200):
    store, _, _ = world({hid(name): specs for name, specs in docs.items()})
    return PineconeMatchService(store, CountingEmbedder(), CONFIG, PineconeMatchLimits(cap))


def client_for(service, cap=200):
    client = make_client(matching=service)
    client.app.dependency_overrides[get_readiness] = lambda: ReadinessGate(ReadinessTracker(), cap)
    return client


def rank(client, text=JD, **extra):
    return client.post("/api/v1/jd/rank", json={"job_description": text, **extra})


class PoolRankTests(unittest.TestCase):
    def test_ranks_the_whole_pool_without_candidate_ids_or_tokens(self):
        service = pool_service({"a": spec("A", ["C#", "SQL", "Docker"]), "b": spec("B", ["C#"]),
                                "c": spec("C", ["Python"])})
        response = rank(client_for(service))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["ranking_basis"], "skills-only")
        self.assertEqual([c["document_id"] for c in body["candidates"]], [hid("a"), hid("b"), hid("c")])
        self.assertEqual([c["rank"] for c in body["candidates"]], [1, 2, 3])
        self.assertEqual(body["pool"], {"discovered": 3, "considered": 3, "ranked": 3, "skipped": 0,
                                        "truncated": False, "max_considered": 100})
        self.assertEqual(body["skipped"], [])
        self.assertEqual(body["candidates"][0]["required_skills"][0]["evidence"]["canonical_path"], "skills[0]")
        self.assertEqual(body["candidates"][0]["source_file"], f"{hid('a')}.pdf")
        for forbidden in ("namespace", "access_token"):
            self.assertNotIn(forbidden, response.text.lower())

    def test_matches_the_explicit_candidate_endpoint(self):
        service = pool_service({"a": spec("A", ["C#", "SQL"]), "b": spec("B", ["C#"])})
        client = client_for(service)
        pool = rank(client).json()["candidates"]
        from tests.api.test_pipeline_api import candidate, match
        explicit = match(make_client(matching=service), [candidate("a"), candidate("b")]).json()["candidates"]
        # strip = lambda items: [{k: v for k, v in c.items() if k != "source_file"} for c in items]
        m14_fields = {
            "required_skill_points",
            "preferred_skill_points",
            "overall_match_score",
            "reasoning",
        }

        strip = lambda items: [
            {
                key: value
                for key, value in candidate.items()
                if key != "source_file" and key not in m14_fields
            }
            for candidate in items
        ]
        self.assertEqual(strip(pool), explicit)

    def test_empty_pool_returns_an_empty_ranking(self):
        service = pool_service({})
        response = rank(client_for(service))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["candidates"], [])
        self.assertEqual(body["pool"]["discovered"], 0)

    def test_unready_resumes_are_skipped_and_reported_not_ranked(self):
        service = pool_service({"a": spec("A", ["C#", "SQL", "Docker"]), "big": spec("B", ["C#", "SQL"])})
        response = rank(client_for(service, cap=1))          # every resume holds more than 1 vector
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["candidates"], [])
        self.assertEqual({s["reason"] for s in body["skipped"]}, {"TOO_LARGE"})
        self.assertEqual(body["pool"]["skipped"], 2)

    def test_stale_resume_is_skipped_while_others_are_ranked(self):
        service = pool_service({"a": spec("A", ["C#", "SQL", "Docker"]), "b": spec("B", ["C#"])})
        tracker = ReadinessTracker()
        tracker.record(hid("b"), 10_000)                     # claims more chunks were written than the index shows
        client = make_client(matching=service)
        client.app.dependency_overrides[get_readiness] = lambda: ReadinessGate(tracker, 200)
        body = rank(client).json()
        self.assertEqual([c["document_id"] for c in body["candidates"]], [hid("a")])
        self.assertEqual(body["skipped"], [{"document_id": hid("b"), "reason": "STALE"}])

    def test_non_resume_namespaces_are_ignored(self):
        service = pool_service({"a": spec("A", ["C#"])})
        _, _, records = build_candidate("not-a-hash", "x.pdf", spec("X", ["C#"]))
        service.store.upsert_embeddings(records)             # namespace "resume:not-a-hash": not an N2 resume
        body = rank(client_for(service)).json()
        self.assertEqual(body["pool"]["discovered"], 1)

    def test_pool_is_capped_and_flagged_truncated(self):
        from app.api.v1 import jd_pool
        service = pool_service({"a": spec("A", ["C#"]), "b": spec("B", ["C#"]), "c": spec("C", ["C#"])})
        original = jd_pool.MAX_POOL_CONSIDERED
        jd_pool.MAX_POOL_CONSIDERED = 2
        try:
            body = rank(client_for(service)).json()
        finally:
            jd_pool.MAX_POOL_CONSIDERED = original
        self.assertTrue(body["pool"]["truncated"])
        self.assertEqual((body["pool"]["discovered"], body["pool"]["considered"]), (3, 2))
        self.assertEqual(len(body["candidates"]), 2)

    def test_requires_the_api_key(self):
        client = client_for(pool_service({}))
        response = client.post("/api/v1/jd/rank", json={"job_description": JD}, headers={"Authorization": ""})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "UNAUTHENTICATED")

    def test_rejects_extra_fields_and_bad_job_descriptions(self):
        client = client_for(pool_service({"a": spec("A", ["C#"])}))
        self.assertEqual(rank(client, namespace="resume:x").status_code, 422)
        self.assertEqual(client.post("/api/v1/jd/rank", json={}).status_code, 422)
        self.assertEqual(rank(client, text="hello").status_code, 422)

    def test_store_failure_is_a_retryable_502(self):
        service = pool_service({"a": spec("A", ["C#"])})

        def boom():
            raise RuntimeError("secret provider detail")
        service.store.index.describe_index_stats = boom
        response = rank(client_for(service))
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["code"], "RETRIEVAL_UNAVAILABLE")
        self.assertNotIn("secret", response.text)
        
    def test_returns_deterministic_m14_score_fields(self):
        service = pool_service({
            "a": spec("A", ["C#", "SQL", "Docker"]),
            "b": spec("B", ["C#"]),
        })

        response = rank(client_for(service))
        self.assertEqual(response.status_code, 200)

        candidates = response.json()["candidates"]
        self.assertEqual(len(candidates), 2)

        for candidate in candidates:
            self.assertIn("required_skill_points", candidate)
            self.assertIn("preferred_skill_points", candidate)
            self.assertIn("overall_match_score", candidate)

            score = candidate["overall_match_score"]
            if score is not None:
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 100)

    def test_m14_score_does_not_change_candidate_order(self):
        service = pool_service({
            "a": spec("A", ["C#", "SQL", "Docker"]),
            "b": spec("B", ["C#"]),
            "c": spec("C", ["Python"]),
        })

        response = rank(client_for(service))
        self.assertEqual(response.status_code, 200)

        candidates = response.json()["candidates"]

        # M14 adds scores but preserves the existing M13 candidate order.
        self.assertEqual(
            [candidate["document_id"] for candidate in candidates],
            [hid("a"), hid("b"), hid("c")],
        )
        self.assertEqual(
            [candidate["rank"] for candidate in candidates],
            [1, 2, 3],
        )

if __name__ == "__main__":
    unittest.main()


class AuthDisabledTests(unittest.TestCase):
    """API_AUTH_DISABLED is an explicit opt-in. Default is still fail-closed with a key."""

    def client(self, **flags):
        from dataclasses import replace
        from tests.api.test_pipeline_api import TEST_SETTINGS
        client = make_client(matching=pool_service({"a": spec("A", ["C#"])}), settings=replace(TEST_SETTINGS, **flags))
        client.app.dependency_overrides[get_readiness] = lambda: ReadinessGate(ReadinessTracker(), 200)
        client.headers.pop("Authorization", None)
        return client

    def test_no_key_works_when_auth_is_disabled(self):
        response = self.client(auth_disabled=True).post("/api/v1/jd/rank", json={"job_description": JD})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["pool"]["ranked"], 1)

    def test_no_key_is_still_refused_by_default(self):
        response = self.client().post("/api/v1/jd/rank", json={"job_description": JD})
        self.assertEqual(response.status_code, 401)

    def test_disabled_mode_needs_no_key_even_with_no_keys_configured(self):
        response = self.client(auth_disabled=True, api_key_hashes={}).post("/api/v1/jd/rank", json={"job_description": JD})
        self.assertEqual(response.status_code, 200)

    def test_env_flag_parsing(self):
        from app.api.settings import ApiSettings
        self.assertFalse(ApiSettings.from_env({}).auth_disabled)
        self.assertFalse(ApiSettings.from_env({"API_AUTH_DISABLED": "false"}).auth_disabled)
        self.assertTrue(ApiSettings.from_env({"API_AUTH_DISABLED": "True"}).auth_disabled)
