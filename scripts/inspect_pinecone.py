"""Run M1-M12 on a local PDF: embed the chunks locally and store them in Pinecone.

Credentials come from .env / the environment and are never printed. The resume is stored in its own
namespace (resume:<document_id>) and left there; pass --cleanup to delete that namespace afterwards.
"""

from dataclasses import replace
from pathlib import Path
import sys

from dotenv import load_dotenv

# Allow the documented script command to import app from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

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
from app.parsing.vector_store import PineconeStore, VectorStoreConfig, VectorStoreError, wait_until

TEST_PREFIX = "m12-test"
SUMMARY_ID = "resume-summary-001"


def run_stage(name, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        raise RuntimeError(f"{name} failed: {error}") from error


def heading(title):
    print(f"\n{title}\n{'=' * len(title)}")


def verdict(ok):
    return "PASS" if ok else "FAIL"


def namespace_isolation_test(config, embedding_dimension, template, client):
    """Known test vector in a temporary namespace: fetchable there, invisible elsewhere, then deleted."""
    store = PineconeStore(replace(config, namespace_prefix=TEST_PREFIX), client)
    store.dimension = embedding_dimension
    vector = [0.0] * embedding_dimension
    vector[0] = 1.0
    record = template.model_copy(update={"embedding": vector,
                                         "metadata": {**template.metadata, "document_id": "isolation"}})
    namespace, other = store.namespace_for("isolation"), store.namespace_for("isolation-other")
    try:
        store.delete_namespace(namespace)
        store.upsert_embeddings([record])
        fetched = wait_until(lambda: store.fetch([record.chunk_id], namespace), 60)
        visible_elsewhere = store.fetch([record.chunk_id], other) or store.query(vector, 3, other)
        return bool(fetched), not visible_elsewhere, namespace
    finally:
        store.delete_namespace(namespace)


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
    chunks = run_stage("M10 Semantic Chunks", compile_chunks, resume)
    embedder = run_stage("M11 model load", NomicEmbedder)
    vectors = run_stage("M11 Nomic Embeddings", embedder.embed_chunks, chunks)

    try:
        config = VectorStoreConfig.from_env()
        store = PineconeStore(config)
        store.ensure_index(embedder.dimension)
        description = store.describe_index()
        result = store.upsert_embeddings(vectors)
    except VectorStoreError as error:
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)

    namespace = result.namespace
    print("PINECONE SUMMARY\n================")
    print(f"Index: {config.index_name}\nNamespace: {namespace}\nDimension: {description.dimension}")
    print(f"Metric: {description.metric}\nChunks: {len(chunks)}\nVectors embedded: {len(vectors)}")
    print(f"Vectors upserted: {result.upserted}\nUpsert failures: {result.failed}")

    heading("FETCH TEST")
    fetched = wait_until(lambda: store.fetch([SUMMARY_ID], namespace), 60) or {}
    print(f"Requested: 1 vector\nReturned: {len(fetched)} vector")
    if fetched:
        item = fetched[SUMMARY_ID]
        print(f"ID: {item.id}\nDimension: {len(item.values)}")
        print(f"Metadata keys: {', '.join(sorted(item.metadata))}")
        print(f"Text stored exactly: {verdict(item.metadata.get('text') == chunks[0].text)}")
        print(f"PII in metadata: {'FAIL' if {'email', 'phone'} & set(item.metadata) else 'none'}")

    heading("QUERY TEST")
    summary_vector = next(v.embedding for v in vectors if v.chunk_id == SUMMARY_ID)
    matches = wait_until(lambda: (m := store.query(summary_vector, 3, namespace)) and m[0].id == SUMMARY_ID and m, 60) or []
    print(f"Query: {SUMMARY_ID} vector\nTop K: 3")
    for rank, match in enumerate(matches, 1):
        print(f"  {rank}. {match.id}  type={match.chunk_type}  path={match.canonical_path}  score={match.score:.4f}")
    print(f"Top result: {matches[0].id if matches else '-'}\nScore: {f'{matches[0].score:.4f}' if matches else '-'}")
    filtered = store.query(summary_vector, 10, namespace, filter={"chunk_type": {"$eq": "PROJECT"}})
    print(f"Filter chunk_type == PROJECT -> {len(filtered)} results, "
          f"all PROJECT: {verdict(bool(filtered) and all(m.chunk_type == 'PROJECT' for m in filtered))}")

    heading("IDEMPOTENCY TEST")
    count = wait_until(lambda: store.namespace_vector_count(namespace) == len(vectors), 60)
    store.upsert_embeddings(vectors)
    after = wait_until(lambda: store.namespace_vector_count(namespace) == len(vectors), 30)
    print(f"Vectors in namespace after first upsert: {store.namespace_vector_count(namespace)}")
    print(f"Re-upserted the same {len(vectors)} vectors -> still {store.namespace_vector_count(namespace)}: "
          f"{verdict(bool(count) and bool(after) and store.namespace_vector_count(namespace) == len(vectors))}")

    heading("NAMESPACE ISOLATION TEST")
    stored, isolated, test_namespace = namespace_isolation_test(config, embedder.dimension, vectors[0], store.client)
    print(f"Temporary namespace: {test_namespace}")
    print(f"Fetched from its own namespace: {verdict(stored)}")
    print(f"Not visible from another namespace: {verdict(isolated)}")
    print("Temporary namespace deleted: yes")

    if "--cleanup" in sys.argv:
        store.delete_namespace(namespace)
        print(f"\nDeleted namespace {namespace}")

    ok = (result.upserted == len(chunks) and result.failed == 0 and len(fetched) == 1 and bool(matches)
          and matches[0].id == SUMMARY_ID and stored and isolated)
    print(f"\nPinecone Ready: {'YES' if ok else 'NO'}")


if __name__ == "__main__":
    main()
