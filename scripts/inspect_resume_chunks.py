"""Run M1-M10 on a local PDF and print the semantic chunks. No embeddings, no network."""

from collections import Counter
from pathlib import Path
import sys

# Allow the documented script command to import app from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.pdf_objects import extract_pdf
from app.parsing.layout import analyze_layout
from app.parsing.reading_order import analyze_reading_order
from app.parsing.logical_blocks import analyze_logical_blocks
from app.parsing.sections import detect_sections, resolve_logical_text
from app.parsing.entities import extract_entities
from app.parsing.records import build_records
from app.parsing.normalization import normalize_records
from app.parsing.canonical import build_canonical_resume
from app.parsing.chunks import ChunkType, compile_chunks

LIST_CHUNKS = ("SKILLS", "LANGUAGE", "INTEREST")


def run_stage(name, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        raise RuntimeError(f"{name} failed: {error}") from error


def leaves(value, out):
    """Every string in the canonical resume, excluding the provenance blocks."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in ("provenance", "validation", "metadata"):
                leaves(item, out)
    elif isinstance(value, list):
        for item in value:
            leaves(item, out)
    elif isinstance(value, str):
        out.add(value)
    return out


def check_embedding_ready(resume, chunks):
    """Return a list of problems; empty means ready."""
    problems = []
    known = leaves(resume.model_dump(mode="json"), set()) | {"Present"}
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        problems.append("duplicate chunk ids")
    for c in chunks:
        if not c.text.strip():
            problems.append(f"{c.chunk_id}: empty text")
        if not (c.provenance.fields and c.provenance.logical_block_ids and c.provenance.page_numbers):
            problems.append(f"{c.chunk_id}: lost provenance")
        pieces = []
        for line in c.text.split("\n"):
            label, sep, rest = line.partition(": ")
            value = rest if sep and label in ("Role", "Company", "Employment", "Description", "Project", "Technologies",
                                              "Degree", "Field of Study", "Institution", "Certification", "Issuer",
                                              "Name", "Location") else line
            if label == "Employment":
                pieces += value.split(" to ")
            elif label == "Technologies" or c.chunk_type.value in LIST_CHUNKS:
                pieces += value.split(", ")
            else:
                pieces.append(value)
        missing = [p for p in pieces if p not in known]
        if missing:
            problems.append(f"{c.chunk_id}: text not found in M9 values: {missing}")
    return problems


def main():
    pdf_path = PROJECT_ROOT / "uploads" / "Kukoo.pdf"
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Input PDF does not exist: {pdf_path}")
    document = run_stage("M1 PDF Object Extraction", extract_pdf, pdf_path)
    layout = run_stage("M2 Layout Analysis", analyze_layout, document)
    order = run_stage("M3 Reading Order", analyze_reading_order, layout)
    logical = run_stage("M4 Logical Blocks", analyze_logical_blocks, document, layout, order)
    text = run_stage("M5 text reference resolution", resolve_logical_text, logical, document)
    sections = run_stage("M5 Section Detection", detect_sections, logical, text)
    entities = run_stage("M6 Entity Extraction", extract_entities, sections, logical, text)
    records = run_stage("M7 Record Construction", build_records, entities)
    normalized = run_stage("M8 Normalization + Validation", normalize_records, records)
    resume = run_stage("M9 Canonical Resume", build_canonical_resume, normalized, pdf_path.name,
                       len(document.pages), sum(len(p.logical_blocks) for p in logical.pages), len(entities.entities))
    include_contact = "--contact" in sys.argv
    chunks = run_stage("M10 Semantic Chunks", compile_chunks, resume, include_contact=include_contact)

    print("SEMANTIC CHUNKS\n===============")
    for number, chunk in enumerate(chunks, 1):
        print(f"\nChunk {number}\n--------")
        print(f"ID: {chunk.chunk_id}")
        print(f"Type: {chunk.chunk_type.value}")
        print(f"Canonical Path: {chunk.canonical_path}")
        print("Text:")
        print("    " + chunk.text.replace("\n", "\n    "))
        print(f"Source Fields: {', '.join(chunk.source_fields)}")
        print(f"Logical Blocks: {', '.join(chunk.provenance.logical_block_ids)}")
        print(f"Pages: {list(chunk.provenance.page_numbers)}")

    counts = Counter(chunk.chunk_type for chunk in chunks)
    print("\nSUMMARY\n=======")
    print(f"Total chunks: {len(chunks)}")
    for kind, label in ((ChunkType.SUMMARY, "Summary"), (ChunkType.WORK_EXPERIENCE, "Work experience"),
                        (ChunkType.PROJECT, "Project"), (ChunkType.EDUCATION, "Education"),
                        (ChunkType.SKILLS, "Skills"), (ChunkType.CERTIFICATION, "Certification"),
                        (ChunkType.AWARD, "Award"), (ChunkType.LANGUAGE, "Language"),
                        (ChunkType.INTEREST, "Interest"), (ChunkType.CONTACT, "Contact")):
        print(f"{label} chunks: {counts[kind]}")
    print(f"(contact chunk {'enabled' if include_contact else 'disabled; pass --contact to enable'})")

    print("\nDETERMINISM CHECK\n=================")
    again = compile_chunks(resume, include_contact=include_contact)
    checks = {"same chunk IDs": [c.chunk_id for c in chunks] == [c.chunk_id for c in again],
              "same order": [c.chunk_type for c in chunks] == [c.chunk_type for c in again],
              "same text": [c.text for c in chunks] == [c.text for c in again],
              "same metadata": [c.metadata for c in chunks] == [c.metadata for c in again],
              "same provenance": [c.provenance for c in chunks] == [c.provenance for c in again]}
    for name, ok in checks.items():
        print(f"{name}: {'yes' if ok else 'NO'}")
    print(f"Deterministic: {'PASS' if all(checks.values()) else 'FAIL'}")

    problems = check_embedding_ready(resume, chunks)
    print("\nEMBEDDING READINESS\n===================")
    for problem in problems:
        print(f"- {problem}")
    print(f"Embedding Ready: {'YES' if not problems and all(checks.values()) else 'NO'}")


if __name__ == "__main__":
    main()
