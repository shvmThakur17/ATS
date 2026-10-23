"""Run M1-M13 for one JD against real Nomic embeddings and real Pinecone vectors.

Candidate A is uploads/Kukoo.pdf (its namespace resume:<document_id> is created/refreshed idempotently and
left in place). Candidate B is a synthetic resume with deliberately different skills, stored in a temporary
`m13-test:` namespace that is deleted at the end. No LLM and no Ollama: this prints evidence only.
"""

from dataclasses import replace
from datetime import date
from pathlib import Path
import sys

from dotenv import load_dotenv

# Allow the documented script command to import app from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.matching import JDMatcher, MatchingConfig, SkillStatus, parse_job_description
from app.parsing.pdf_objects import extract_pdf
from app.parsing.layout import analyze_layout
from app.parsing.reading_order import analyze_reading_order
from app.parsing.logical_blocks import analyze_logical_blocks
from app.parsing.sections import detect_sections, resolve_logical_text
from app.parsing.entities import extract_entities
from app.parsing.records import build_records
from app.parsing.normalization import normalize_records
from app.parsing.canonical import build_canonical_resume
from app.parsing.chunks import compile_chunks
from app.parsing.embeddings import NomicEmbedder
from app.parsing.vector_store import PineconeStore, VectorStoreConfig, wait_until
from tests.parsing.canonical.test_canonical import canon
from tests.parsing.matching.fakes import CANDIDATE_B

JD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "jds" / "dotnet_backend_jd.txt"
TEST_PREFIX = "m13-test"
B_DOCUMENT = "synthetic-candidate-b"


def run_stage(name, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        raise RuntimeError(f"{name} failed: {error}") from error


def heading(title, char="="):
    print(f"\n{title}\n{char * len(title)}")


def pct(value):
    return "n/a" if value is None else f"{value:.2f}"


def short(text, width=110):
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 3] + "..."


def build_kukoo(embedder):
    pdf_path = PROJECT_ROOT / "uploads" / "Kukoo.pdf"
    document = run_stage("M1", extract_pdf, pdf_path)
    layout = run_stage("M2", analyze_layout, document)
    order = run_stage("M3", analyze_reading_order, layout)
    logical = run_stage("M4", analyze_logical_blocks, document, layout, order)
    text = run_stage("M5 text", resolve_logical_text, logical, document)
    sections = run_stage("M5", detect_sections, logical, text)
    entities = run_stage("M6", extract_entities, sections, logical, text)
    normalized = run_stage("M7/M8", lambda: normalize_records(build_records(entities)))
    resume = run_stage("M9", build_canonical_resume, normalized, pdf_path.name, len(document.pages),
                       sum(len(p.logical_blocks) for p in logical.pages), len(entities.entities))
    chunks = run_stage("M10", compile_chunks, resume)
    return resume, run_stage("M11", embedder.embed_chunks, chunks)


def build_synthetic_b(embedder):
    resume = canon(*CANDIDATE_B)
    resume = resume.model_copy(update={"metadata": resume.metadata.model_copy(
        update={"document_id": B_DOCUMENT, "source_file_name": "synthetic_b.pdf"})})
    return resume, embedder.embed_chunks(compile_chunks(resume))


def print_candidate(candidate, result):
    heading("CANDIDATE")
    print(f"Document: {candidate.document_id}")
    for title, matches in (("REQUIRED SKILLS", candidate.required_skill_matches),
                           ("PREFERRED SKILLS", candidate.preferred_skill_matches)):
        heading(title)
        print(f"{'Skill':<16}{'Status':<26}{'Evidence':<42}{'Similarity'}")
        for m in matches:
            where = f"{m.evidence_kind.value if m.evidence_kind else '-'} {m.canonical_path or ''}".strip()
            print(f"{m.skill:<16}{m.status.value:<26}{where:<42}{pct(m.similarity_score)}")

    heading("COVERAGE")
    print(f"Required Skill Coverage:  {pct(candidate.required_skill_coverage)}   (required skills with explicit evidence / required skills)")
    print(f"Preferred Skill Coverage: {pct(candidate.preferred_skill_coverage)}   (preferred skills with explicit evidence / preferred skills)")

    heading("RETRIEVAL (informational, not used for ranking)")
    r = candidate.retrieval
    print(f"Queries: {', '.join(q.query_type.value for q in result.queries)}")
    print(f"Retrieved chunks: {r.retrieval_count} ({r.unique_chunk_count} unique)")
    print(f"Top semantic score: {pct(r.top_score)}\nAverage top-k score: {pct(r.average_top_k_score)}")
    print(f"Chunks above the relevance threshold: {r.relevant_chunk_count}")

    heading("EXPERIENCE / EDUCATION")
    e = candidate.experience
    if e:
        months = "n/a" if e.candidate_months is None else f"{e.candidate_months} months"
        print(f"Experience: JD asks {e.required_years}+ years; overall employment {months}; meets={e.meets}\n  ({e.note})")
        if e.stated_text:
            print(f"  Resume-stated: \"{e.stated_text}\" ({e.stated_months} months, meets={e.stated_meets}) "
                  f"from {e.stated_source_path}, blocks {', '.join(e.logical_block_ids)}, pages {', '.join(e.page_numbers)}")
        if e.discrepancy:
            print(f"  DISCREPANCY: {e.discrepancy_note}")
    d = candidate.education
    if d:
        print(f"Education: JD levels {list(d.required_levels)}; resume levels {list(d.candidate_levels)}; "
              f"{d.status.value}\n  ({d.note})")
        for item in d.evidence:
            print(f"  {item.canonical_path}: degree={item.degree!r}, field={item.field_of_study!r}, "
                  f"institution={item.institution!r}, levels={list(item.levels)}, "
                  f"blocks {', '.join(item.logical_block_ids)}, pages {', '.join(item.page_numbers)}")

    heading("SKILL RANKING (the only ranking input)")
    k = candidate.skill_score
    print(f"  rank {candidate.rank}")
    print(f"  required:  {k.required_explicit}/{k.required_total} explicit, {k.required_semantic} semantic-only")
    print(f"  preferred: {k.preferred_explicit}/{k.preferred_total} explicit, {k.preferred_semantic} semantic-only")
    print("  key = (required explicit share, preferred explicit share, required semantic share, preferred semantic "
          f"share) = {tuple(round(v, 3) for v in k.key)}")
    print("  compared lexicographically; semantic-only never counts as covered; experience, education and "
          "retrieval similarity are shown for information and do not rank")

    heading("EVIDENCE")
    shown = sorted(candidate.evidence, key=lambda x: (-x.similarity_score, x.chunk_id))[:6]
    for item in shown:
        print(f"\nQuery Type: {item.query_type.value}\nChunk: {item.chunk_id}\nChunk Type: {item.chunk_type}")
        print(f"Canonical Path: {item.canonical_path}\nScore: {item.similarity_score:.4f} ({item.evidence_strength.value})")
        print(f"Logical Blocks: {', '.join(item.logical_block_ids)}\nEvidence: {short(item.matched_text)}")
    for m in candidate.skill_matches:
        if m.status in (SkillStatus.EXPLICIT_TEXT_EVIDENCE, SkillStatus.SEMANTIC_MATCH):
            print(f"\nSkill {m.skill}: {m.status.value} at {m.canonical_path} (blocks {', '.join(m.logical_block_ids)}; "
                  f"pages {', '.join(m.page_numbers)}; chunk {m.supporting_chunk_id}; score {pct(m.similarity_score)})"
                  f"\n  term: {m.matched_term}\n  text: {short(m.supporting_text or '')}")

    heading("WARNINGS")
    for warning in candidate.warnings:
        print(f"- {warning}")


def main():
    jd = run_stage("JD parse", parse_job_description, JD_PATH.read_text(encoding="utf-8"))
    config = replace(MatchingConfig.from_env(), as_of=date.today().strftime("%Y-%m"))
    embedder = run_stage("Nomic model load", NomicEmbedder)
    kukoo, kukoo_vectors = build_kukoo(embedder)
    b, b_vectors = build_synthetic_b(embedder)

    store = PineconeStore(VectorStoreConfig.from_env())
    store.ensure_index(embedder.dimension)
    test_store = PineconeStore(replace(store.config, namespace_prefix=TEST_PREFIX), store.client)
    test_store.dimension = store.dimension
    a_namespace = store.upsert_embeddings(kukoo_vectors).namespace      # idempotent refresh of the M12 data
    b_namespace = test_store.namespace_for(B_DOCUMENT)
    try:
        test_store.delete_namespace(b_namespace)
        test_store.upsert_embeddings(b_vectors)
        for namespace, count in ((a_namespace, len(kukoo_vectors)), (b_namespace, len(b_vectors))):
            ready = wait_until(lambda: store.namespace_vector_count(namespace) == count, 90)
            if not ready:
                raise RuntimeError(f"vectors in {namespace} were not visible in time")
        matcher = JDMatcher(store, embedder, config)
        result = matcher.match(jd, {kukoo.metadata.document_id: kukoo, B_DOCUMENT: b},
                               [a_namespace, b_namespace])
    finally:
        test_store.delete_namespace(b_namespace)

    print("JD MATCHING\n===========")
    print(f"JD:\n{result.title}  ({result.jd_id})\n\nCandidates:\n{len(result.candidates)}")
    print(f"Required skills parsed: {', '.join(s.normalized for s in jd.required_skills)}")
    print(f"Preferred skills parsed: {', '.join(s.normalized for s in jd.preferred_skills)}")
    print(f"Responsibilities: {len(jd.responsibilities)}; experience requirements: {len(jd.experience_requirements)}; "
          f"education requirements: {len(jd.education_requirements)}; unclassified lines: {len(jd.unclassified_text)}")
    print(f"As-of month for current roles: {config.as_of}; top_k={config.top_k}")
    print("Candidates are ranked by skills only (required coverage, then preferred coverage, then semantic evidence).")
    for candidate in result.candidates:
        print_candidate(candidate, result)
    print(f"\nTemporary namespace {b_namespace} deleted; {a_namespace} left in place.")

    heading("FINAL RESULT")
    print("Structured matching evidence generated successfully.")
    print("This is evidence about the resumes, not a hiring recommendation. M14 performs the final reasoning.")


if __name__ == "__main__":
    main()
