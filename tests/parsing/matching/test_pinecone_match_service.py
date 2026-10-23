"""PineconeMatchService: offline tests against the in-memory Pinecone fake. No live Pinecone, no model."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import unittest
from unittest import mock

from app.matching import InvalidConfigError, JDMatcher, MatchingConfig, SkillStatus as St, parse_job_description
from app.matching.pinecone_canonical_adapter import NamespaceMismatchError
from app.matching.pinecone_match_service import (CandidateRef, DuplicateCandidateError, NamespaceCountUnavailableError,
                                                  PineconeMatchLimits, PineconeMatchService, RetrievalDepthExceededError)
from app.matching.ranking import skill_score
from tests.parsing.matching.fakes import CANDIDATE_A, CANDIDATE_B, CountingEmbedder, make_store
from tests.parsing.matching.test_pinecone_canonical_adapter import (CONFIG, SPRING_BOOT, SPRING_JD, dotnet_jd, keys,
                                                                    match, world)
from tests.parsing.matching.test_skill_ranking import jd as skills_jd, spec

ROOT = Path(__file__).resolve().parents[3]
LOW = replace(CONFIG, top_k=1, semantic_match_threshold=0.05, required_skill_match_threshold=0.05,
              preferred_skill_match_threshold=0.05, strong_evidence_threshold=0.5)


def refs(namespaces, order=None):
    return [CandidateRef(d, namespaces[d]) for d in (order or namespaces)]


def service(store, config=LOW, cap=200, embedder=None):
    return PineconeMatchService(store, embedder or CountingEmbedder(), config, PineconeMatchLimits(cap))


def baseline_keys(store, jd, full, namespaces):
    return keys(match(store, jd, full, namespaces))


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.store, self.full, self.ns = world({"doc-a": CANDIDATE_A, "doc-b": CANDIDATE_B})
        self.count_a, self.count_b = (self.store.namespace_vector_count(self.ns[d]) for d in ("doc-a", "doc-b"))

    def test_11_top_k_is_raised_to_cover_the_namespaces(self):
        jd = dotnet_jd()
        outcome = service(self.store, replace(CONFIG, top_k=1)).match(jd, refs(self.ns))
        self.assertEqual(outcome.effective_top_k, max(self.count_a, self.count_b))
        self.assertEqual(keys(outcome.result), baseline_keys(self.store, jd, self.full, self.ns))
        self.assertEqual({(n.document_id, n.vector_count) for n in outcome.namespaces},
                         {("doc-a", self.count_a), ("doc-b", self.count_b)})

    def test_12_a_namespace_exactly_at_the_cap_is_accepted(self):
        outcome = service(self.store, cap=max(self.count_a, self.count_b)).match(dotnet_jd(), refs(self.ns))
        self.assertEqual(outcome.effective_top_k, outcome.max_retrieval_depth)
        self.assertEqual(len(outcome.result.candidates), 2)

    def test_13_and_27_a_namespace_over_the_cap_fails_before_any_retrieval(self):
        embedder = CountingEmbedder()
        with mock.patch.object(self.store, "query", wraps=self.store.query) as query:
            with self.assertRaises(RetrievalDepthExceededError) as raised:
                service(self.store, cap=max(self.count_a, self.count_b) - 1, embedder=embedder).match(dotnet_jd(), refs(self.ns))
        query.assert_not_called()                      # neither the adapter nor M13 retrieved anything
        self.assertEqual(embedder.query_texts, [])
        self.assertIn("cap", str(raised.exception))

    def test_14_unknown_vector_count_fails_instead_of_guessing(self):
        with mock.patch.object(self.store.index, "describe_index_stats", side_effect=RuntimeError("stats down")):
            with self.assertRaises(NamespaceCountUnavailableError):
                service(self.store).match(dotnet_jd(), refs(self.ns))
        for bad in (None, "6", -1, 2.5):
            with mock.patch.object(self.store, "namespace_vector_count", return_value=bad):
                with self.assertRaises(NamespaceCountUnavailableError, msg=repr(bad)):
                    service(self.store).match(dotnet_jd(), refs(self.ns))

    def test_15_empty_namespaces_are_reported_deterministically(self):
        candidates = refs(self.ns) + [CandidateRef("ghost", "resume:ghost")]
        first = service(self.store).match(dotnet_jd(), candidates)
        second = service(self.store).match(dotnet_jd(), candidates)
        self.assertEqual(first, second)
        ghost = next(c for c in first.result.candidates if c.document_id == "ghost")
        self.assertEqual((ghost.rank, ghost.retrieval_count), (3, 0))
        self.assertTrue(all(m.status == St.UNKNOWN for m in ghost.skill_matches))
        self.assertIn("no chunks were retrieved", " | ".join(ghost.warnings))
        self.assertEqual(ghost.skill_score.key, (0.0, 0.0, 0.0, 0.0))
        only = service(self.store).match(dotnet_jd(), [CandidateRef("ghost", "resume:ghost")])
        self.assertEqual((only.effective_top_k, len(only.result.candidates)), (1, 1))     # top_k never below 1
        self.assertEqual(service(self.store).match(dotnet_jd(), []).result.candidates, ())  # nothing supplied, nothing discovered

    def test_16_effective_depth_applies_to_one_call_only(self):
        base = replace(CONFIG, top_k=3)
        svc = service(self.store, base)
        first = svc.match(dotnet_jd(), refs(self.ns))
        second = svc.match(dotnet_jd(), refs(self.ns, ["doc-b"]))
        self.assertEqual((first.effective_top_k, second.effective_top_k),
                         (max(self.count_a, self.count_b), self.count_b))
        self.assertEqual((svc.config.top_k, base.top_k), (3, 3))
        self.assertEqual(MatchingConfig().top_k, 10)

    def test_26_low_depth_cannot_silently_omit_required_evidence(self):
        jd = parse_job_description("Backend Engineer\nRequired Skills\n- C#\n- ASP.NET Core\n- SQL Server\n- REST APIs\n"
                                   "- Docker\nPreferred Skills\n- Azure\n- Kubernetes\nResponsibilities\n- Build backend services\n")
        baseline = baseline_keys(self.store, jd, self.full, self.ns)
        raw_shallow = keys(JDMatcher(self.store, CountingEmbedder(), LOW).match(jd, {}, list(self.ns.values())))
        self.assertNotEqual(raw_shallow, baseline)                   # what the wrapper exists to prevent
        self.assertEqual(keys(service(self.store, LOW).match(jd, refs(self.ns)).result), baseline)


class LimitsTests(unittest.TestCase):
    def test_limits_validation_and_env(self):
        self.assertEqual(PineconeMatchLimits().max_retrieval_depth, 200)
        for bad in (0, -1, 1001, True, "5", 2.5):
            with self.assertRaises(InvalidConfigError, msg=repr(bad)):
                PineconeMatchLimits(bad)
        with mock.patch.dict("os.environ", {"M13_MAX_RETRIEVAL_DEPTH": "40"}):
            self.assertEqual(PineconeMatchLimits.from_env().max_retrieval_depth, 40)
        with mock.patch.dict("os.environ", {"M13_MAX_RETRIEVAL_DEPTH": "many"}):
            with self.assertRaises(InvalidConfigError):
                PineconeMatchLimits.from_env()
        with mock.patch.dict("os.environ", {}, clear=False):
            __import__("os").environ.pop("M13_MAX_RETRIEVAL_DEPTH", None)
            self.assertEqual(PineconeMatchLimits.from_env().max_retrieval_depth, 200)


class IsolationTests(unittest.TestCase):
    def test_validation_of_the_candidate_list(self):
        store, _, ns = world({"doc-a": CANDIDATE_A})
        with self.assertRaises(NamespaceMismatchError):
            service(store).match(dotnet_jd(), [CandidateRef("doc-a", "resume:doc-z")])
        with self.assertRaises(NamespaceMismatchError):
            service(store).match(dotnet_jd(), [CandidateRef("doc-a", "m13-test:doc-a")])
        with self.assertRaises(DuplicateCandidateError):
            service(store).match(dotnet_jd(), refs(ns) * 2)

    def test_the_wrapper_never_discovers_namespaces(self):
        store, _, ns = world({"doc-a": CANDIDATE_A, "doc-b": CANDIDATE_B})
        test_store = make_store("m13-test")
        test_store.client = store.client                       # same index, other prefix
        outcome = service(store).match(dotnet_jd(), refs(ns, ["doc-a"]))
        self.assertEqual([c.document_id for c in outcome.result.candidates], ["doc-a"])
        self.assertEqual(outcome.result.namespaces, (ns["doc-a"],))
        self.assertTrue(all(e.namespace == ns["doc-a"] for c in outcome.result.candidates for e in c.evidence))

    def test_17_and_18_candidates_do_not_influence_each_other(self):
        jd = dotnet_jd()
        store, full, ns = world({"doc-a": CANDIDATE_A, "doc-b": CANDIDATE_B, "doc-c": SPRING_BOOT})
        alone = {d: service(store).match(jd, refs(ns, [d])).result.candidates[0].skill_score.key for d in ns}
        together = keys(service(store).match(jd, refs(ns)).result)
        self.assertEqual({d: together[d][1] for d in ns}, alone)
        # adding another candidate's vectors afterwards leaves an existing candidate's key unchanged
        before = service(store).match(jd, refs(ns, ["doc-a"])).result.candidates[0]
        world({"doc-d": CANDIDATE_B, "doc-e": CANDIDATE_A}, store)
        after = service(store).match(jd, refs(ns, ["doc-a"])).result.candidates[0]
        self.assertEqual((after.skill_score, after.rank, after.evidence), (before.skill_score, before.rank, before.evidence))
        self.assertEqual(together, baseline_keys(store, jd, full, ns))

    def test_19_identical_profiles_tie_independent_of_ingestion_order(self):
        jd = dotnet_jd()
        results = []
        for order in (("doc-a", "doc-a2", "doc-b"), ("doc-b", "doc-a2", "doc-a")):
            store, _, ns = world({d: CANDIDATE_B if d == "doc-b" else CANDIDATE_A for d in order})
            results.append(service(store).match(jd, refs(ns, list(order))).result)
        self.assertEqual(keys(results[0]), keys(results[1]))
        self.assertEqual([c.document_id for c in results[0].candidates], [c.document_id for c in results[1].candidates])
        tied = keys(results[0])
        self.assertEqual(tied["doc-a"], tied["doc-a2"])          # identical key and identical (shared) rank
        self.assertGreater(tied["doc-b"][0], tied["doc-a"][0])


class RankingRegressionTests(unittest.TestCase):
    def ranked(self, job, docs):
        store, full, ns = world(docs)
        outcome = service(store).match(job, refs(ns))
        self.assertEqual(keys(outcome.result), baseline_keys(store, job, full, ns))
        return {c.document_id: c for c in outcome.result.candidates}, [c.document_id for c in outcome.result.candidates]

    def test_21_required_coverage_takes_priority_over_preferred(self):
        job = skills_jd(["C#", "SQL"], ["Docker", "Azure", "Angular"])
        by, order = self.ranked(job, {"required_only": spec("A", ["C#", "SQL"]),
                                      "all_preferred": spec("B", ["C#", "Docker", "Azure", "Angular"])})
        self.assertEqual(order, ["required_only", "all_preferred"])

    def test_22_preferred_skills_are_secondary(self):
        job = skills_jd(["C#", "SQL"], ["Docker", "Azure"])
        by, order = self.ranked(job, {"more": spec("A", ["C#", "SQL", "Docker", "Azure"]),
                                      "less": spec("B", ["C#", "SQL", "Docker"]), "none": spec("C", ["C#", "SQL"])})
        self.assertEqual(order, ["more", "less", "none"])
        self.assertEqual({by[d].skill_score.required_coverage for d in order}, {1.0})

    def test_23_semantic_only_evidence_is_never_coverage(self):
        job = skills_jd(["SQL", "Java", "ASP.NET Core"])
        by, _ = self.ranked(job, {"c": spec("C", ["SQL Server", "JavaScript", "Entity Framework Core"])})
        statuses = {m.skill: m.status for m in by["c"].skill_matches}
        self.assertFalse(any(s in (St.EXPLICIT_SKILL, St.EXPLICIT_TEXT_EVIDENCE) for s in statuses.values()), statuses)
        self.assertEqual(by["c"].skill_score.key[0], 0.0)

    def test_24_dotnet_jd_reproduces_baseline_keys_and_order(self):
        store, full, ns = world({"doc-a": CANDIDATE_A, "doc-b": CANDIDATE_B})
        outcome = service(store).match(dotnet_jd(), refs(ns))
        baseline = match(store, dotnet_jd(), full, ns)
        self.assertEqual(keys(outcome.result), keys(baseline))
        self.assertEqual([c.document_id for c in outcome.result.candidates], [c.document_id for c in baseline.candidates])
        self.assertEqual([c.rank for c in outcome.result.candidates], [c.rank for c in baseline.candidates])

    def test_25_spring_regression_matches_the_canonical_baseline(self):
        store, full, ns = world({"c": SPRING_BOOT})
        job = parse_job_description(SPRING_JD)
        ours = service(store).match(job, refs(ns)).result.candidates[0]
        self.assertEqual(ours.skill_score.key, match(store, job, full, ns).candidates[0].skill_score.key)
        self.assertEqual({m.skill: m.status for m in ours.skill_matches}["Spring"], St.NOT_FOUND)

    def test_20_ranking_uses_the_unchanged_m13_formula(self):
        store, _, ns = world({"doc-a": CANDIDATE_A, "doc-b": CANDIDATE_B})
        for candidate in service(store).match(dotnet_jd(), refs(ns)).result.candidates:
            self.assertEqual(candidate.skill_score, skill_score(list(candidate.skill_matches)))
        text = (ROOT / "app/matching/pinecone_match_service.py").read_text(encoding="utf-8")
        for forbidden in ("skill_score(", "assign_ranks", "ollama", "Ollama", "openai"):
            self.assertNotIn(forbidden, text)


class FrozenManifestTests(unittest.TestCase):
    def test_28_m13_frozen_manifest_still_verifies(self):
        lines = (ROOT / "docs/m13_frozen_manifest.sha256").read_text(encoding="utf-8").splitlines()
        entries = [line.split(" *", 1) for line in lines if line.strip()]
        self.assertGreaterEqual(len(entries), 17)
        mismatched = [path for digest, path in entries if sha256((ROOT / path).read_bytes()).hexdigest() != digest]
        self.assertEqual(mismatched, [])


if __name__ == "__main__":
    unittest.main()
