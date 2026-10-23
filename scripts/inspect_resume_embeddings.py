"""Run M1-M11 on a local PDF and print the embedding summary. Local model only; no Pinecone, no network."""

from math import isfinite
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
from app.parsing.chunks import compile_chunks
from app.parsing.embeddings import EmbeddingError, NomicEmbedder

SMOKE_TEXTS = ["The candidate has experience building REST APIs with ASP.NET Core and SQL Server.",
               "Languages: English, Hindi"]


def run_stage(name, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        raise RuntimeError(f"{name} failed: {error}") from error


def verdict(ok):
    return "PASS" if ok else "FAIL"


def main():
    print("SMOKE TEST\n==========")
    try:
        embedder = NomicEmbedder()
        vectors = embedder.embed_documents(SMOKE_TEXTS)
    except EmbeddingError as error:
        print(f"{type(error).__name__}: {error}")
        print("\nEmbedding Ready: NO (model could not produce embeddings; see error above)")
        sys.exit(1)
    print(f"Model: {embedder.model_name}\nVectors: {len(vectors)}\nDimension: {embedder.dimension}")
    print(f"All finite: {verdict(all(isfinite(x) for v in vectors for x in v))}")
    print(f"Same dimension: {verdict(len({len(v) for v in vectors}) == 1)}")

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
    chunks = run_stage("M10 Semantic Chunks", compile_chunks, resume)
    try:
        embedded = embedder.embed_chunks(chunks)
        failed = 0
    except EmbeddingError as error:
        print(f"\n{type(error).__name__}: {error}")
        embedded, failed = [], len(chunks)

    print("\nEMBEDDING SUMMARY\n=================")
    print(f"Model: {embedder.model_name}\nDimension: {embedder.dimension}\nDevice: {embedder.device}")
    print(f"\nTotal chunks: {len(chunks)}\nEmbedded chunks: {len(embedded)}\nFailed chunks: {failed}")
    for record in embedded:
        preview = ", ".join(f"{x:.4f}" for x in record.embedding[:4])
        print(f"\nChunk ID: {record.chunk_id}\nType: {record.chunk_type}\nCanonical Path: {record.canonical_path}"
              f"\nDimension: {record.dimension}\nVector Preview: [{preview}, ...]")

    checks = {
        "All vectors same dimension": len({r.dimension for r in embedded}) == 1 and all(
            len(r.embedding) == r.dimension for r in embedded),
        "All vectors finite": all(isfinite(x) for r in embedded for x in r.embedding),
        "Chunk count matches vector count": len(embedded) == len(chunks),
        "Chunk IDs preserved": [r.chunk_id for r in embedded] == [c.chunk_id for c in chunks],
        "Order preserved": [r.canonical_path for r in embedded] == [c.canonical_path for c in chunks],
        "Chunk text unchanged": [r.text for r in embedded] == [c.text for c in chunks],
        "Provenance preserved": [r.provenance for r in embedded] == [c.provenance for c in chunks],
    }
    print("\nVECTOR VALIDATION\n=================")
    for name, ok in checks.items():
        print(f"{name}: {verdict(ok)}")
    print(f"\nEmbedding Ready: {'YES' if embedded and all(checks.values()) else 'NO'}")


if __name__ == "__main__":
    main()
