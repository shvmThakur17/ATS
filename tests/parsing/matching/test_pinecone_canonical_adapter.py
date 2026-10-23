"""PineconeCanonicalAdapter: offline tests against the in-memory Pinecone fake. No live Pinecone, no model."""

import builtins
import copy
import io
import os
from pathlib import Path
import unittest
from unittest import mock

from app.matching import JDMatcher, MatchingConfig, SkillStatus as St, parse_job_description
from app.matching.pinecone_canonical_adapter import (CanonicalReconstructionError, MalformedSkillMetadataError,
                                                      NamespaceMismatchError, PineconeCanonicalAdapter)
from app.parsing.canonical.models import CanonicalResume
from app.parsing.entities.models import EntityType as E
from app.parsing.sections import SectionType as S
from tests.parsing.matching.fakes import (CANDIDATE_A, CANDIDATE_B, CountingEmbedder, build_candidate, ingest,
                                          make_store)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "jds" / "dotnet_backend_jd.txt"
# Same low thresholds as test_jd_matching (the hashed fake embedder gives modest cosines). Depth is large so the
# canonical baseline sees every chunk too.
CONFIG = MatchingConfig(top_k=100, semantic_match_threshold=0.15, strong_evidence_threshold=0.5,
                        required_skill_match_threshold=0.15, preferred_skill_match_threshold=0.15, as_of="2026-10")
X = S.EXPERIENCE
SPRING_BOOT = ((E.NAME, "Carl", S.UNKNOWN, 0), (E.ROLE, "Dev", X, 1), (E.DATE_RANGE, "Jan 2020 - Jan 2024", X, 1),
               (E.COMPANY, "Co", X, 2), (E.DESCRIPTION, "Built services with Spring Boot.", X, 2),
               (E.SKILL, "Java", S.SKILLS, 3), (E.SKILL, "Spring Boot", S.SKILLS, 3))
SUMMARY_ONLY = ((E.NAME, "Dana", S.UNKNOWN, 0), (E.DESCRIPTION, "Backend developer.", S.SUMMARY, 1))
SPRING_JD = "Backend\nRequired Skills\n- Spring\n- Java\nResponsibilities\n- Build\n"


def dotnet_jd():
    return parse_job_description(FIXTURE.read_text(encoding="utf-8"))


def world(docs, store=None):
    """docs: {document_id: specs}. Returns (store, {document_id: full canonical}, {document_id: namespace})."""
    store = store or make_store()
    resumes, namespaces = {}, {}
    for document_id, specs in docs.items():
        resume, _, records = build_candidate(document_id, f"{document_id}.pdf", specs)
        resumes[document_id], namespaces[document_id] = resume, ingest(store, records)
    return store, resumes, namespaces


def count(store, namespace):
    return store.namespace_vector_count(namespace)


def rebuilt(store, namespaces):
    adapter = PineconeCanonicalAdapter(store)
    return {d: adapter.rebuild(d, ns, count(store, ns)) for d, ns in namespaces.items()}


def keys(result):
    return {c.document_id: (c.rank, c.skill_score.key) for c in result.candidates}


def match(store, jd, canonical, namespaces, config=CONFIG):
    return JDMatcher(store, CountingEmbedder(), config).match(jd, canonical, list(namespaces.values()))


def vector(store, namespace, chunk_id):
    return store.index.namespaces[namespace][chunk_id]


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.store, self.full, self.ns = world({"doc-a": CANDIDATE_A, "doc-b": CANDIDATE_B})
        self.adapter = PineconeCanonicalAdapter(self.store)

    def rebuild(self, document_id):
        return self.adapter.rebuild(document_id, self.ns[document_id], count(self.store, self.ns[document_id]))

    def test_1_skills_keep_the_stored_order(self):
        for document_id in ("doc-a", "doc-b"):
            rebuilt_resume = self.rebuild(document_id)
            self.assertIsInstance(rebuilt_resume, CanonicalResume)
            self.assertEqual([s.value for s in rebuilt_resume.skills], [s.value for s in self.full[document_id].skills])
            self.assertEqual([s.raw_value for s in rebuilt_resume.skills], [s.raw_value for s in self.full[document_id].skills])

    def test_2_project_technologies(self):
        b = self.rebuild("doc-b")
        self.assertEqual([[t.value for t in p.technologies] for p in b.projects],
                         [[t.value for t in p.technologies] for p in self.full["doc-b"].projects])
        self.assertEqual([t.value for t in b.projects[0].technologies], ["Kubernetes"])
        self.assertEqual([[t.value for t in p.technologies] for p in self.rebuild("doc-a").projects], [[]])

    def test_3_duplicate_skills_follow_the_canonical_convention(self):
        store, full, ns = world({"dup": CANDIDATE_A + ((E.SKILL, "C#", S.SKILLS, 12), (E.SKILL, "html", S.SKILLS, 12))})
        rebuilt_resume = PineconeCanonicalAdapter(store).rebuild("dup", ns["dup"], count(store, ns["dup"]))
        self.assertEqual([s.value for s in rebuilt_resume.skills], [s.value for s in full["dup"].skills])  # no dedupe, no reorder
        jd = dotnet_jd()
        self.assertEqual(keys(match(store, jd, {"dup": rebuilt_resume}, ns)), keys(match(store, jd, full, ns)))

    def test_4_empty_skills_and_projects(self):
        store, full, ns = world({"empty": SUMMARY_ONLY})
        adapter_resume = PineconeCanonicalAdapter(store).rebuild("empty", ns["empty"], count(store, ns["empty"]))
        self.assertEqual((adapter_resume.skills, adapter_resume.projects), ([], []))
        jd = dotnet_jd()
        result = match(store, jd, {"empty": adapter_resume}, ns).candidates[0]
        self.assertEqual(result.skill_score.key, match(store, jd, full, ns).candidates[0].skill_score.key)
        self.assertTrue(all(m.status in (St.UNKNOWN, St.NOT_FOUND) for m in result.skill_matches))
        self.assertTrue(any(m.status == St.UNKNOWN for m in result.skill_matches))   # M13's own rule: no content -> UNKNOWN

    def test_4b_empty_namespace_needs_no_query(self):
        store = make_store()
        with mock.patch.object(store, "query") as query:
            empty = PineconeCanonicalAdapter(store).rebuild("nobody", "resume:nobody", 0)
        query.assert_not_called()
        self.assertEqual((empty.skills, empty.projects, empty.metadata.document_id), ([], [], "nobody"))

    def test_5_malformed_metadata_is_rejected(self):
        def case(mutate, message):
            store, _, ns = world({"doc-b": CANDIDATE_B})
            mutate(store, ns["doc-b"])
            with self.assertRaises(MalformedSkillMetadataError, msg=message):
                PineconeCanonicalAdapter(store).rebuild("doc-b", ns["doc-b"], count(store, ns["doc-b"]))

        def skills_meta(store, ns):
            return vector(store, ns, "resume-skills-001")["metadata"]

        def project_meta(store, ns):
            return vector(store, ns, "resume-project-001")["metadata"]

        case(lambda s, n: skills_meta(s, n).update(skills="Python, Docker"), "skills is a string")
        case(lambda s, n: skills_meta(s, n).pop("skills"), "skills missing")
        case(lambda s, n: skills_meta(s, n).update(skills=["Python", " "]), "blank skill")
        case(lambda s, n: skills_meta(s, n).update(skills=["Python"]), "skills disagree with the stored text")
        case(lambda s, n: skills_meta(s, n).update(raw_skills=["x"]), "raw_skills length differs")
        case(lambda s, n: skills_meta(s, n).update(canonical_path="skill"), "wrong skills path")
        case(lambda s, n: n and s.index.namespaces[n].update({"resume-skills-002": copy.deepcopy(vector(s, n, "resume-skills-001"))}),
             "two SKILLS chunks")
        case(lambda s, n: project_meta(s, n).update(canonical_path="project"), "bad project path")
        case(lambda s, n: project_meta(s, n).update(record_index=3), "record_index disagrees with the path")
        case(lambda s, n: project_meta(s, n).update(record_index=True), "record_index not a number")
        case(lambda s, n: project_meta(s, n).update(technologies="Kubernetes"), "technologies is a string")
        case(lambda s, n: project_meta(s, n).update(technologies=["Helm"]), "technologies disagree with the stored text")

    def test_5b_live_style_numbers_and_missing_optional_fields_are_accepted(self):
        meta = vector(self.store, self.ns["doc-b"], "resume-project-001")["metadata"]
        meta["record_index"] = 0.0                        # Pinecone returns numbers as floats
        self.assertEqual([t.value for t in self.rebuild("doc-b").projects[0].technologies], ["Kubernetes"])
        vector(self.store, self.ns["doc-b"], "resume-skills-001")["metadata"].pop("raw_skills")
        self.assertEqual([s.value for s in self.rebuild("doc-b").skills], [s.value for s in self.full["doc-b"].skills])

    def test_6_namespace_must_belong_to_the_document(self):
        for document_id, namespace in (("doc-a", self.ns["doc-b"]), ("doc-a", "m13-test:doc-a"), ("doc-a", "resume:"),
                                       ("", "resume:"), ("doc-a", "")):
            with self.assertRaises(NamespaceMismatchError, msg=(document_id, namespace)):
                self.adapter.rebuild(document_id, namespace, 5)

    def test_7_other_candidates_never_leak_in(self):
        a = self.rebuild("doc-a")
        self.assertEqual([s.value for s in a.skills], [s.value for s in self.full["doc-a"].skills])
        self.assertFalse({"Python", "Docker", "Azure", "Kubernetes"} & {s.value for s in a.skills}
                         | {t.value for p in a.projects for t in p.technologies})
        # a vector of another resume planted in this namespace is refused, not merged
        planted = copy.deepcopy(vector(self.store, self.ns["doc-b"], "resume-skills-001"))
        self.store.index.namespaces[self.ns["doc-a"]]["resume-skills-001"] = planted
        with self.assertRaises(NamespaceMismatchError):
            self.rebuild("doc-a")

    def test_8_prefix_regression_spring_vs_spring_boot(self):
        store, full, ns = world({"c": SPRING_BOOT})
        canonical = rebuilt(store, ns)
        self.assertEqual([s.value for s in canonical["c"].skills], ["Java", "Spring Boot"])
        jd = parse_job_description(SPRING_JD)
        baseline, ours = match(store, jd, full, ns).candidates[0], match(store, jd, canonical, ns).candidates[0]
        by_skill = {m.skill: m.status for m in ours.skill_matches}
        self.assertNotIn(by_skill["Spring"], (St.EXPLICIT_SKILL, St.EXPLICIT_TEXT_EVIDENCE))
        self.assertEqual(by_skill["Java"], St.EXPLICIT_SKILL)
        self.assertEqual(ours.skill_score.key, baseline.skill_score.key)
        self.assertEqual(ours.skill_score.key[0], 0.5)
        # without the adapter M13 cannot see "Spring Boot" as a longer term and over-credits "Spring"
        raw = match(store, jd, {}, ns).candidates[0]
        self.assertEqual(raw.skill_score.key[0], 1.0)

    def test_9_same_ranking_key_as_the_full_canonical_baseline(self):
        jd = dotnet_jd()
        baseline = match(self.store, jd, self.full, self.ns)
        ours = match(self.store, jd, rebuilt(self.store, self.ns), self.ns)
        self.assertEqual(keys(ours), keys(baseline))
        self.assertEqual([c.document_id for c in ours.candidates], [c.document_id for c in baseline.candidates])

    def test_10_nothing_is_written(self):
        before = (sorted(os.listdir(".")), sum(1 for _ in Path("app").rglob("*")))
        real_open = builtins.open

        def guarded(file, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"):
                raise AssertionError(f"adapter tried to write {file!r}")
            return real_open(file, mode, *args, **kwargs)

        with mock.patch.object(builtins, "open", guarded), mock.patch.object(io, "open", guarded):
            self.rebuild("doc-a")
            self.rebuild("doc-b")
        self.assertEqual((sorted(os.listdir(".")), sum(1 for _ in Path("app").rglob("*"))), before)

    def test_store_failure_is_a_typed_error(self):
        with mock.patch.object(self.store, "query", side_effect=__import__("app.parsing.vector_store",
                                                                           fromlist=["QueryError"]).QueryError("boom")):
            with self.assertRaises(CanonicalReconstructionError):
                self.rebuild("doc-a")

    def test_gaps_in_project_indexes_keep_original_paths(self):
        store, full, ns = world({"doc-b": CANDIDATE_B})
        record = vector(store, ns["doc-b"], "resume-project-001")
        moved = copy.deepcopy(record)
        moved["metadata"].update(canonical_path="projects[2]", record_index=2)
        store.index.namespaces[ns["doc-b"]]["resume-project-001"] = moved
        projects = PineconeCanonicalAdapter(store).rebuild("doc-b", ns["doc-b"], count(store, ns["doc-b"])).projects
        self.assertEqual([[t.value for t in p.technologies] for p in projects], [[], [], ["Kubernetes"]])

    def test_selects_by_chunk_type_not_by_similarity(self):
        # at full namespace depth the SKILLS chunk is returned however the probe vector ranks it
        store, _, ns = world({"doc-b": CANDIDATE_B})
        full_depth = count(store, ns["doc-b"])
        self.assertEqual(len(PineconeCanonicalAdapter(store).rebuild("doc-b", ns["doc-b"], full_depth).skills), 4)


if __name__ == "__main__":
    unittest.main()
