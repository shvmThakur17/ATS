from pathlib import Path
from types import SimpleNamespace
import unittest

from app.matching import (MatchingConfig, JDMatcher, MalformedMatchResultError,
                          QueryEmbeddingError, QueryType, RetrievalError, EducationStatus as Ed, SkillStatus as St, parse_job_description)
from app.matching.models import EmbeddingDimensionError, NamespaceDiscoveryError
from app.parsing.embeddings import EmbeddingBatchError
from tests.parsing.matching.fakes import (CANDIDATE_A, CANDIDATE_B, CountingEmbedder, HashBackend, build_candidate,
                                          ingest, make_store, two_candidates)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "jds" / "dotnet_backend_jd.txt"
# Low thresholds: the hashed fake embedder gives modest cosines. The logic under test is unchanged.
CONFIG = MatchingConfig(top_k=10, semantic_match_threshold=0.15, strong_evidence_threshold=0.5,
                        required_skill_match_threshold=0.15, preferred_skill_match_threshold=0.15, as_of="2026-10")


def jd():
    return parse_job_description(FIXTURE.read_text(encoding="utf-8"))


def run(config=CONFIG, namespaces=None, canonical=True):
    store, resumes, (ns_a, ns_b) = two_candidates()
    embedder = CountingEmbedder()
    matcher = JDMatcher(store, embedder, config)
    result = matcher.match(jd(), resumes if canonical else None, namespaces)
    return store, embedder, result, {c.document_id: c for c in result.candidates}


class MatchingTests(unittest.TestCase):
    def test_candidates_grouped_by_document_id(self):
        _, _, result, by_doc = run()
        self.assertEqual(sorted(by_doc), ["doc-a", "doc-b"])
        self.assertEqual(len(result.candidates), 2)
        for doc, candidate in by_doc.items():
            self.assertTrue(candidate.evidence)
            self.assertTrue(all(e.document_id == doc and e.namespace == f"resume:{doc}" for e in candidate.evidence))

    def test_multiple_chunks_per_candidate_grouped_by_type(self):
        _, _, _, by_doc = run()
        grouped = by_doc["doc-a"].grouped_evidence()
        self.assertGreater(len({e.chunk_id for e in by_doc["doc-a"].evidence}), 3)
        self.assertEqual(list(grouped), [t for t in ("SUMMARY", "WORK_EXPERIENCE", "PROJECT", "EDUCATION", "SKILLS") if t in grouped])
        self.assertIn("SKILLS", grouped)
        self.assertTrue(all(e.chunk_type == t for t, items in grouped.items() for e in items))

    def test_required_skill_results_for_candidate_a(self):
        _, _, _, by_doc = run()
        status = {m.skill: m.status for m in by_doc["doc-a"].required_skill_matches}
        self.assertEqual(status, {"C#": St.EXPLICIT_SKILL, "ASP.NET Core": St.EXPLICIT_SKILL,
                                  ".NET": St.EXPLICIT_TEXT_EVIDENCE,        # only in the summary text
                                  "SQL Server": St.EXPLICIT_TEXT_EVIDENCE,  # resume lists "SQL", text says SQL Server
                                  "REST APIs": St.SEMANTIC_MATCH})  # resume says "Web APIs": related, never explicit
        rest = next(m for m in by_doc["doc-a"].required_skill_matches if m.skill == "REST APIs")
        self.assertEqual(rest.matched_term, "APIs")
        net = next(m for m in by_doc["doc-a"].required_skill_matches if m.skill == ".NET")
        self.assertEqual((net.canonical_path, net.page_numbers), ("summary", ("0",)))
        self.assertIn(".NET Developer", net.supporting_text)
        self.assertTrue(net.logical_block_ids and net.entity_ids)
        self.assertEqual(by_doc["doc-a"].required_skill_coverage, 0.8)    # C#, ASP.NET Core, .NET, SQL Server
        self.assertEqual({m.skill: m.status for m in by_doc["doc-a"].preferred_skill_matches},
                         {"Azure": St.NOT_FOUND, "Docker": St.NOT_FOUND, "Angular": St.NOT_FOUND})
        self.assertEqual(by_doc["doc-a"].preferred_skill_coverage, 0.0)

    def test_evidence_index_formula_and_missing_component_policy(self):
        from dataclasses import replace
        _, _, _, by_doc = run()
        a = by_doc["doc-a"]
        weights = CONFIG.weights
        self.assertIsNone(a.components["experience_evidence"])               # UNKNOWN (conflicting verdicts)
        self.assertEqual(a.unavailable_components, ("experience_evidence",))
        self.assertEqual(a.not_applicable_components, ())
        expected = sum(w * (a.components[n] or 0.0) for n, w in weights.items()) / sum(weights.values())
        self.assertAlmostEqual(a.evidence_index, expected)                   # "zero": the missing 0.10 stays in
        self.assertAlmostEqual(a.evidence_completeness, 0.9)
        _, _, _, lenient = run(replace(CONFIG, missing_component_policy="renormalize"))
        self.assertGreater(lenient["doc-a"].evidence_index, a.evidence_index)   # omission would have raised it
        self.assertEqual(lenient["doc-a"].unavailable_components, a.unavailable_components)

    def test_not_applicable_components_and_completeness(self):
        store, resumes, _ = two_candidates()
        minimal = parse_job_description("Required Skills\n- C#\n\nResponsibilities\n- Build web applications")
        result = JDMatcher(store, CountingEmbedder(), CONFIG).match(minimal, resumes)
        for candidate in result.candidates:
            self.assertEqual(set(candidate.not_applicable_components),
                             {"preferred_skill_coverage", "experience_evidence", "education_evidence"})
            self.assertEqual(candidate.unavailable_components, ())
            self.assertEqual(candidate.evidence_completeness, 1.0)       # N/A components do not reduce it
            weights = CONFIG.weights
            names = ("required_skill_coverage", "semantic_relevance", "responsibility_relevance")
            expected = sum(weights[n] * candidate.components[n] for n in names) / sum(weights[n] for n in names)
            self.assertAlmostEqual(candidate.evidence_index, expected)

    def test_no_resume_means_unavailable_not_free_points(self):
        _, _, _, without = run(canonical=False)
        a = without["doc-a"]
        self.assertIn("education_evidence", a.unavailable_components)
        self.assertLess(a.evidence_completeness, 1.0)
        weights = CONFIG.weights
        expected = sum(weights[n] * (a.components[n] or 0.0) for n in weights) / sum(weights.values())
        self.assertAlmostEqual(a.evidence_index, expected)          # unavailable components contribute 0, not nothing

    def test_candidates_are_matched_independently(self):
        _, _, _, by_doc = run()
        b = {m.skill: m.status for m in by_doc["doc-b"].skill_matches}
        self.assertEqual(b["REST APIs"], St.EXPLICIT_SKILL)
        self.assertEqual((b["Azure"], b["Docker"]), (St.EXPLICIT_SKILL, St.EXPLICIT_SKILL))
        self.assertEqual((b["C#"], b["ASP.NET Core"], b["Angular"]), (St.NOT_FOUND,) * 3)
        self.assertEqual(by_doc["doc-b"].required_skill_coverage, 0.2)
        self.assertAlmostEqual(by_doc["doc-b"].preferred_skill_coverage, 2 / 3)
        self.assertNotEqual(by_doc["doc-a"].required_skill_coverage, by_doc["doc-b"].required_skill_coverage)

    def test_experience_and_education_evidence(self):
        _, _, _, by_doc = run()
        a, b = by_doc["doc-a"], by_doc["doc-b"]
        self.assertEqual((a.experience.candidate_months, a.experience.meets), (51, True))
        self.assertEqual((b.experience.candidate_months, b.experience.meets), (53, True))
        self.assertEqual(a.education.status, Ed.NOT_FOUND)           # diploma, not bachelor
        self.assertEqual(b.education.status, Ed.EXPLICIT_MATCH)      # B.Tech

    def test_provenance_chain_is_preserved(self):
        _, _, result, by_doc = run()
        queries = {q.query_type for q in result.queries}
        for evidence in by_doc["doc-a"].evidence:
            self.assertIn(evidence.query_type, queries)                # JD query
            self.assertRegex(evidence.chunk_id, r"^resume-[a-z]+-\d{3}$")  # vector id == chunk id
            self.assertTrue(evidence.canonical_path)
            self.assertTrue(evidence.logical_block_ids)                # M10 provenance -> logical blocks
            self.assertEqual(evidence.page_numbers, ("0",))
            self.assertTrue(evidence.entity_ids)
            self.assertTrue(evidence.matched_text)
            self.assertEqual(evidence.metadata["chunk_id"], evidence.chunk_id)
        skills_chunk = next(e for e in by_doc["doc-a"].evidence if e.chunk_type == "SKILLS")
        self.assertEqual(skills_chunk.matched_text, "HTML, ASP.NET Core, C#, SQL, Web APIs")

    def test_similarity_scores_are_preserved_unfiltered(self):
        config = MatchingConfig(top_k=50, semantic_match_threshold=0.99, strong_evidence_threshold=0.99, as_of="2026-10")
        store, _, result, by_doc = run(config)
        evidence = by_doc["doc-a"].evidence
        self.assertTrue(any(e.similarity_score < 0.99 for e in evidence))   # low scores are not discarded
        text = next(q.text for q in result.queries if q.query_type == QueryType.REQUIRED_SKILLS)
        direct = {m.id: m.score for m in store.query(CountingEmbedder().embed_query(text), 50, "resume:doc-a")}
        required = [e for e in evidence if e.query_type == QueryType.REQUIRED_SKILLS]
        self.assertEqual(len(required), len(direct))
        for e in required:
            self.assertEqual(e.similarity_score, direct[e.chunk_id])      # exactly what Pinecone returned

    def test_query_units_embedded_once_not_per_namespace_or_skill(self):
        _, embedder, result, _ = run()
        self.assertEqual(len(embedder.query_texts), len(result.queries))
        self.assertEqual(len(set(embedder.query_texts)), len(embedder.query_texts))
        self.assertEqual(len(result.queries), 5)

    def test_chunk_type_filters_are_applied(self):
        _, _, _, by_doc = run()
        by_query = {}
        for e in by_doc["doc-a"].evidence:
            by_query.setdefault(e.query_type, set()).add(e.chunk_type)
        self.assertTrue(by_query[QueryType.RESPONSIBILITIES] <= {"WORK_EXPERIENCE", "PROJECT", "SUMMARY"})
        self.assertTrue(by_query[QueryType.EDUCATION] <= {"EDUCATION", "CERTIFICATION"})
        self.assertTrue(len(by_query[QueryType.REQUIRED_SKILLS]) > 2)

    def test_top_k_is_configurable(self):
        _, _, _, by_doc = run(MatchingConfig(top_k=2, semantic_match_threshold=0.15, strong_evidence_threshold=0.5,
                                              as_of="2026-10"))
        for query_type in (QueryType.REQUIRED_SKILLS, QueryType.PREFERRED_SKILLS):
            self.assertLessEqual(sum(e.query_type == query_type for e in by_doc["doc-a"].evidence), 2)

    def test_scoring_components_and_not_a_fit_percentage(self):
        _, _, _, by_doc = run()
        a = by_doc["doc-a"]
        self.assertEqual(set(a.components), set(CONFIG.weights))
        self.assertEqual(a.components["required_skill_coverage"], 0.8)
        self.assertIsNone(a.components["experience_evidence"])   # stated 2+ years vs 51 months: no silent winner
        self.assertEqual(a.components["education_evidence"], 0.0)
        self.assertTrue(0.0 <= a.evidence_index <= 1.0)
        self.assertEqual(a.retrieval_count, len(a.evidence))
        self.assertGreaterEqual(a.retrieval.top_score, a.retrieval.average_top_k_score)

    def test_warnings_report_missing_requirements(self):
        _, _, _, by_doc = run()
        text = " | ".join(by_doc["doc-a"].warnings)
        self.assertIn("REST APIs (required) has only semantic evidence", text)
        self.assertIn("Azure (preferred) has no explicit resume evidence", text)
        self.assertIn(".NET (required) is only mentioned in text", text)
        self.assertIn("education requirement not explicitly met", text)
        self.assertIn('resume states "2+ years"', text)
        self.assertIn("neither is treated as authoritative", text)
        self.assertIn("experience requirement is not confirmed (UNKNOWN)", text)
        self.assertEqual(by_doc["doc-a"].experience.status.value, "UNKNOWN")
        a = by_doc["doc-a"]
        self.assertEqual((a.experience.stated_months, a.experience.candidate_months), (24, 51))
        self.assertEqual(a.education.evidence[0].degree, "Diploma")       # M8-normalized
        self.assertIn("DIPLOMA", a.education.evidence[0].raw_text)        # as written
        self.assertTrue(a.education.evidence[0].logical_block_ids)

    def test_without_canonical_resumes_skills_are_unknown_not_invented(self):
        _, _, _, by_doc = run(canonical=False)
        a = by_doc["doc-a"]
        self.assertTrue(all(m.status in (St.UNKNOWN, St.EXPLICIT_TEXT_EVIDENCE, St.SEMANTIC_MATCH) for m in a.skill_matches))
        self.assertTrue(any(m.status == St.UNKNOWN for m in a.skill_matches))
        self.assertFalse(any(m.status == St.EXPLICIT_SKILL for m in a.skill_matches))   # nothing structured is invented
        self.assertEqual(a.education.status, Ed.UNKNOWN)
        self.assertIsNone(a.components["education_evidence"])
        self.assertIn("canonical resume not provided", " | ".join(a.warnings))

    def test_deterministic_output(self):
        first = run()[2]
        second = run()[2]
        self.assertEqual(first.model_dump_json(), second.model_dump_json())

    def test_result_order_does_not_depend_on_ingest_order(self):
        store = make_store()
        a, _, a_records = build_candidate("doc-a", "a.pdf", CANDIDATE_A)
        b, _, b_records = build_candidate("doc-b", "b.pdf", CANDIDATE_B)
        ingest(store, b_records)
        ingest(store, a_records)
        reordered = JDMatcher(store, CountingEmbedder(), CONFIG).match(jd(), {"doc-b": b, "doc-a": a})
        original = run()[2]
        self.assertEqual([c.document_id for c in reordered.candidates], [c.document_id for c in original.candidates])
        self.assertEqual(reordered.model_dump_json(), original.model_dump_json())

    def test_no_pii_in_evidence(self):
        _, _, _, by_doc = run()
        blob = " ".join(str(e) for c in by_doc.values() for e in c.evidence)
        for secret in ("alice@example.com", "bob@example.com", "8683077463"):
            self.assertNotIn(secret, blob)


class NamespaceTests(unittest.TestCase):
    def test_discovery_finds_resume_namespaces_only(self):
        store, resumes, (ns_a, ns_b) = two_candidates()
        _, _, records = build_candidate("other", "o.pdf", CANDIDATE_B)
        from app.parsing.vector_store import PineconeStore
        scratch = PineconeStore(store.config.__class__(api_key="k", index_name="idx", namespace_prefix="m12-test"),
                                store.client)
        scratch.dimension = 768
        ingest(scratch, records)
        matcher = JDMatcher(store, CountingEmbedder(), CONFIG)
        self.assertEqual(matcher.retriever.discover_namespaces(), sorted([ns_a, ns_b]))
        result = matcher.match(jd(), resumes)
        self.assertEqual(set(result.namespaces), {ns_a, ns_b})
        self.assertEqual({c.document_id for c in result.candidates}, {"doc-a", "doc-b"})

    def test_explicit_namespace_limits_the_search(self):
        store, resumes, (ns_a, ns_b) = two_candidates()
        result = JDMatcher(store, CountingEmbedder(), CONFIG).match(jd(), {"doc-a": resumes["doc-a"]}, [ns_a])
        self.assertEqual(result.namespaces, (ns_a,))
        self.assertEqual([c.document_id for c in result.candidates], ["doc-a"])
        self.assertTrue(all(e.namespace == ns_a for e in result.candidates[0].evidence))

    def test_candidate_without_vectors_is_reported(self):
        store, resumes, (ns_a, _) = two_candidates()
        result = JDMatcher(store, CountingEmbedder(), CONFIG).match(jd(), resumes, [ns_a])
        b = next(c for c in result.candidates if c.document_id == "doc-b")
        self.assertEqual((b.retrieval_count, b.retrieval.top_score), (0, None))
        self.assertIn("no chunks were retrieved", " | ".join(b.warnings))
        self.assertEqual({m.skill: m.status for m in b.required_skill_matches}["REST APIs"], St.EXPLICIT_SKILL)

    def test_one_resume_per_namespace_keeps_chunk_ids_isolated(self):
        _, _, result, by_doc = run()
        ids_a = {e.chunk_id for e in by_doc["doc-a"].evidence}
        ids_b = {e.chunk_id for e in by_doc["doc-b"].evidence}
        self.assertTrue(ids_a & ids_b)                       # same chunk ids exist in both resumes ...
        self.assertTrue(all(e.namespace == "resume:doc-a" for e in by_doc["doc-a"].evidence))  # ... but never mix


class ErrorTests(unittest.TestCase):
    def test_embedding_failure(self):
        store, resumes, (ns_a, _) = two_candidates()

        class Broken(CountingEmbedder):
            def embed_query(self, text):
                raise EmbeddingBatchError("boom")
        with self.assertRaises(QueryEmbeddingError):
            JDMatcher(store, Broken(), CONFIG).match(jd(), resumes, [ns_a])

    def test_inconsistent_dimension(self):
        store, resumes, (ns_a, _) = two_candidates()

        class Short(CountingEmbedder):
            def embed_query(self, text):
                return [0.1] * 8
        with self.assertRaises(EmbeddingDimensionError):
            JDMatcher(store, Short(), CONFIG).match(jd(), resumes, [ns_a])

    def test_pinecone_failure_and_namespace_failure(self):
        store, resumes, (ns_a, _) = two_candidates()
        store.index.query = lambda **kw: (_ for _ in ()).throw(RuntimeError("down"))
        with self.assertRaises(RetrievalError):
            JDMatcher(store, CountingEmbedder(), CONFIG).match(jd(), resumes, [ns_a])
        store2, resumes2, _ = two_candidates()
        store2.index.describe_index_stats = lambda: (_ for _ in ()).throw(RuntimeError("down"))
        with self.assertRaises(NamespaceDiscoveryError):
            JDMatcher(store2, CountingEmbedder(), CONFIG).match(jd(), resumes2)

    def test_malformed_match_result(self):
        store, resumes, (ns_a, _) = two_candidates()
        for metadata in ({}, {"document_id": "doc-a", "chunk_type": "SKILLS", "canonical_path": "skills"},
                         {"document_id": "wrong", "chunk_type": "SKILLS", "canonical_path": "skills", "text": "t"}):
            matcher = JDMatcher(store, CountingEmbedder(), CONFIG)
            matcher.retriever.store = SimpleNamespace(
                query=lambda *a, **k: [SimpleNamespace(id="resume-skills-001", score=0.5, metadata=metadata)],
                config=store.config, index=store.index)
            with self.assertRaises(MalformedMatchResultError):
                matcher.match(jd(), resumes, [ns_a])


if __name__ == "__main__":
    unittest.main()
