"""N2: bytes of one PDF resume -> M1-M9 canonical resume -> M10 chunks -> M11 embeddings -> M12 Pinecone.

Orchestration only: every stage is the existing frozen implementation, called in the order the project's own
scripts use. Nothing here parses, normalizes, chunks, embeds or matches, and nothing is persisted besides the
Pinecone vectors M12 writes. Failures raise typed errors; a result object means the vectors were upserted.

Storage contract (what M12 actually guarantees, no more):
  * same PDF bytes -> same document id -> same namespace ``<prefix>:<sha256>`` and same chunk ids;
  * upserting an existing namespace + id REPLACES that vector, so re-ingesting identical bytes does not duplicate
    vectors. Vectors that an older chunking run wrote under ids the new run no longer produces are NOT removed
    (this service never deletes);
  * M12 validates every vector before the first write, but writes in batches of 100, so a resume with more than
    100 chunks can be partially written if Pinecone fails mid-way. Typical resumes fit one call.
"""

from dataclasses import dataclass
from hashlib import sha256
import logging
import os
from pathlib import Path
import re
from typing import Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from ..parsing.canonical import CanonicalResume, build_canonical_resume
from ..parsing.chunks import compile_chunks
from ..parsing.embeddings import EmbeddingRecord
from ..parsing.entities import extract_entities
from ..parsing.layout import analyze_layout
from ..parsing.logical_blocks import analyze_logical_blocks
from ..parsing.normalization import normalize_records
from ..parsing.pdf_objects import extract_pdf
from ..parsing.reading_order import analyze_reading_order
from ..parsing.records import build_records
from ..parsing.sections import detect_sections, resolve_logical_text
from ..parsing.vector_store import PineconeStore

logger = logging.getLogger("app.ingestion")

DEFAULT_MAX_PDF_BYTES = 10 * 1024 * 1024
PDF_HEADER = b"%PDF-"
PDF_HEADER_WINDOW = 1024            # the PDF spec lets the header sit anywhere in the first 1024 bytes
MAX_FILENAME_LENGTH = 255
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


# ---- errors: messages name the stage and the error TYPE only, never resume text, paths or credentials ------------
class IngestionError(Exception):
    """Base class. `code` is stable and safe to show to a user; `stage` says where it failed."""
    code = "INGESTION_FAILED"

    def __init__(self, message: str, stage: str | None = None):
        super().__init__(message)
        self.stage = stage


class InvalidPdfInputError(IngestionError):
    code = "INVALID_PDF_INPUT"


class EmptyPdfInputError(InvalidPdfInputError):
    code = "EMPTY_PDF_INPUT"


class PdfTooLargeError(InvalidPdfInputError):
    code = "PDF_TOO_LARGE"


class NotAPdfError(InvalidPdfInputError):
    code = "NOT_A_PDF"


class InvalidIngestionConfigError(IngestionError):
    code = "INVALID_INGESTION_CONFIG"


class PdfProcessingError(IngestionError):
    """An M1-M9 stage rejected the PDF (corrupt, encrypted, over the M1 limits, ...)."""
    code = "PDF_PROCESSING_FAILED"


class EmptyResumeContentError(IngestionError):
    """The PDF parsed but produced no chunks (for example a scanned, image-only resume). Nothing is stored."""
    code = "NO_TEXT_CONTENT"


class TooManyChunksError(IngestionError):
    """The resume compiles to more chunks than the configured cap (which matching must also be able to retrieve)."""
    code = "TOO_MANY_CHUNKS"


class InconsistentPipelineOutputError(IngestionError):
    code = "INCONSISTENT_PIPELINE_OUTPUT"


class EmbeddingStageError(IngestionError):
    code = "EMBEDDING_FAILED"


class StorageStageError(IngestionError):
    """Pinecone refused or failed the write. The namespace may be partially written; nothing was cleaned up."""
    code = "STORAGE_FAILED"


# ---- configuration and result ------------------------------------------------------------------------------------
DEFAULT_MAX_CHUNKS = 200            # equals PineconeMatchLimits' default retrieval-depth cap: what is stored can be matched
MAX_CHUNKS_CEILING = 1000           # Pinecone's top_k limit for queries that return metadata


@dataclass(frozen=True)
class IngestionLimits:
    max_pdf_bytes: int = DEFAULT_MAX_PDF_BYTES      # M1 enforces its own, separate 25 MB ceiling as well
    max_chunks: int = DEFAULT_MAX_CHUNKS            # keep <= M13_MAX_RETRIEVAL_DEPTH (the API checks this at startup)

    def __post_init__(self):
        for name, high in (("max_pdf_bytes", None), ("max_chunks", MAX_CHUNKS_CEILING)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (high and value > high):
                raise InvalidIngestionConfigError(f"{name} must be a positive integer"
                                                  + (f" of at most {high}" if high else ""))

    @classmethod
    def from_env(cls) -> "IngestionLimits":
        values = {}
        for name, variable in (("max_pdf_bytes", "INGEST_MAX_PDF_BYTES"), ("max_chunks", "INGEST_MAX_CHUNKS")):
            raw = os.environ.get(variable)
            if raw not in (None, ""):
                try:
                    values[name] = int(raw)
                except ValueError:
                    raise InvalidIngestionConfigError(f"{variable} must be an integer") from None
        return cls(**values)


class IngestionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["INGESTED"]
    document_id: str                  # SHA-256 of the original bytes
    namespace: str                    # M12 convention: <prefix>:<document_id>
    chunk_count: int
    vectors_upserted: int
    page_count: int
    source_filename: str              # sanitized; stored in vector metadata as `source_file`
    embedding_model: str
    embedding_dimension: int
    replaced_existing: bool | None    # True if the namespace already held vectors; None if that could not be read


# ---- the M1 -> M9 pipeline ---------------------------------------------------------------------------------------
class ResumePipeline(Protocol):
    def build(self, pdf_bytes: bytes, source_filename: str) -> CanonicalResume: ...


@dataclass(frozen=True)
class PipelineStages:
    """The frozen stage functions, injectable so tests can observe order. Defaults are the real ones."""
    extract: Callable = extract_pdf
    layout: Callable = analyze_layout
    reading_order: Callable = analyze_reading_order
    logical_blocks: Callable = analyze_logical_blocks
    resolve_text: Callable = resolve_logical_text
    sections: Callable = detect_sections
    entities: Callable = extract_entities
    records: Callable = build_records
    normalize: Callable = normalize_records
    canonical: Callable = build_canonical_resume


class DeterministicResumePipeline:
    """M1-M9 exactly as scripts/inspect_resume_canonical.py chains them."""

    def __init__(self, stages: PipelineStages | None = None):
        self.stages = stages or PipelineStages()

    def build(self, pdf_bytes: bytes, source_filename: str) -> CanonicalResume:
        s = self.stages
        document = self.run("M1", s.extract, pdf_bytes)
        layout = self.run("M2", s.layout, document)
        order = self.run("M3", s.reading_order, layout)
        logical = self.run("M4", s.logical_blocks, document, layout, order)
        text = self.run("M5", s.resolve_text, logical, document)
        sections = self.run("M5", s.sections, logical, text)
        entities = self.run("M6", s.entities, sections, logical, text)
        records = self.run("M7", s.records, entities)
        normalized = self.run("M8", s.normalize, records)
        return self.run("M9", s.canonical, normalized, source_filename, len(document.pages),
                        sum(len(page.logical_blocks) for page in logical.pages), len(entities.entities))

    @staticmethod
    def run(stage: str, function, *arguments):
        try:
            return function(*arguments)
        except Exception as error:
            # M1's messages are sanitized by design; for every other stage only the error type is reported.
            detail = str(error) if stage == "M1" and type(error).__name__ == "PDFExtractionError" \
                else type(error).__name__
            raise PdfProcessingError(f"{stage} failed: {detail}", stage) from error


# ---- the service -------------------------------------------------------------------------------------------------
def sanitize_filename(name: str | None) -> str:
    """Base name only, no control characters, bounded length. Never used to open a file."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = CONTROL_CHARACTERS.sub("", base).strip()
    return base[:MAX_FILENAME_LENGTH] or "resume.pdf"


class PdfResumeIngestionService:
    def __init__(self, parser_pipeline: ResumePipeline | None = None, embedder=None,
                 pinecone_store: PineconeStore | None = None, limits: IngestionLimits | None = None,
                 chunk_compiler: Callable = compile_chunks):
        if embedder is None or pinecone_store is None:
            raise InvalidIngestionConfigError("an embedder and a Pinecone store are required")
        self.parser_pipeline = parser_pipeline or DeterministicResumePipeline()
        self.embedder, self.pinecone_store = embedder, pinecone_store
        self.limits = limits or IngestionLimits()
        self.chunk_compiler = chunk_compiler

    # --- public API ---------------------------------------------------------------------------------------------
    def ingest_pdf(self, pdf_bytes: bytes, source_filename: str | None = None) -> IngestionResult:
        data = self.validate(pdf_bytes)
        filename = sanitize_filename(source_filename)
        document_id = sha256(data).hexdigest()
        try:
            namespace = self.pinecone_store.namespace_for(document_id)     # M12 convention, before any work
        except Exception as error:
            raise StorageStageError(f"namespace could not be derived: {type(error).__name__}", "M12") from error

        resume = self.parser_pipeline.build(data, filename)
        if resume.metadata.document_id != document_id:
            raise InconsistentPipelineOutputError("canonical resume document id does not match the PDF bytes", "M9")

        chunks = self.chunk_compiler(resume)
        if not chunks:
            raise EmptyResumeContentError("the PDF produced no text chunks (scanned or empty resume?); nothing stored",
                                          "M10")
        if len(chunks) > self.limits.max_chunks:       # before any embedding or write
            raise TooManyChunksError(f"the resume compiles to more than {self.limits.max_chunks} chunks; nothing stored",
                                     "M10")
        records = self.embed(chunks)
        replaced = self.existing_vectors(namespace)
        self.store(records, namespace)
        logger.info("ingested document=%s chunks=%d pages=%d", document_id[:12], len(records),
                    resume.metadata.page_count)
        return IngestionResult(
            status="INGESTED", document_id=document_id, namespace=namespace, chunk_count=len(chunks),
            vectors_upserted=len(records), page_count=resume.metadata.page_count, source_filename=filename,
            embedding_model=records[0].model_name, embedding_dimension=records[0].dimension,
            replaced_existing=replaced)

    def ingest_pdf_file(self, path: str | Path) -> IngestionResult:
        """Read a local file (bounded by the size limit) and ingest it. The file name is the source file name."""
        path = Path(path)
        try:
            with path.open("rb") as stream:
                data = stream.read(self.limits.max_pdf_bytes + 1)
        except OSError:
            raise InvalidPdfInputError("PDF file could not be read", "input") from None
        return self.ingest_pdf(data, path.name)

    # --- stages -------------------------------------------------------------------------------------------------
    def validate(self, pdf_bytes) -> bytes:
        """Cheap checks only; the real PDF validation is M1's (`extract_pdf` opens and bounds the document)."""
        if not isinstance(pdf_bytes, (bytes, bytearray)):
            raise InvalidPdfInputError("PDF input must be bytes", "input")
        if not pdf_bytes:
            raise EmptyPdfInputError("PDF input is empty", "input")
        if len(pdf_bytes) > self.limits.max_pdf_bytes:
            raise PdfTooLargeError(f"PDF exceeds the {self.limits.max_pdf_bytes}-byte limit", "input")
        data = bytes(pdf_bytes)
        if PDF_HEADER not in data[:PDF_HEADER_WINDOW]:
            raise NotAPdfError("input does not look like a PDF (no %PDF- header)", "input")
        return data

    def embed(self, chunks) -> list[EmbeddingRecord]:
        try:
            records = self.embedder.embed_chunks(chunks)
        except Exception as error:
            raise EmbeddingStageError(f"M11 failed: {type(error).__name__}", "M11") from error
        if len(records) != len(chunks) or [r.chunk_id for r in records] != [c.chunk_id for c in chunks]:
            raise InconsistentPipelineOutputError("embeddings do not correspond one-to-one to the chunks", "M11")
        return records

    def existing_vectors(self, namespace: str) -> bool | None:
        """Read-only note of whether this namespace already held vectors; failure here never blocks ingestion."""
        try:
            return self.pinecone_store.namespace_vector_count(namespace) > 0
        except Exception:
            return None

    def store(self, records: list[EmbeddingRecord], namespace: str) -> None:
        try:
            result = self.pinecone_store.upsert_embeddings(records)
        except Exception as error:
            raise StorageStageError(f"M12 upsert failed for namespace {namespace!r}: {type(error).__name__}; the "
                                    f"namespace may be partially written and was not cleaned up", "M12") from error
        if result.namespace != namespace or result.failed or result.upserted != len(records):
            raise StorageStageError(f"M12 reported an incomplete upsert for namespace {namespace!r}", "M12")
