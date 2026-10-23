"""POST /api/v1/resume/upload: PDF in, N2 ingestion service, sanitized result out.

Order matters: authentication and rate limiting happen in middleware before the body is read; here the cheap
checks (extension, empty, size, per-caller quota) come first, and only then is the heavy service built and used.
"""

import logging

from fastapi import APIRouter, Depends, File, Response, UploadFile

from app.api.candidate_access import CandidateTokenSigner
from app.api.dependencies import (ApiLimits, get_api_limits, get_caller, get_ingestion_service, get_readiness,
                                  get_runtime, get_token_signer)
from app.api.errors import ApiError
from app.api.readiness import ReadinessUnavailableError
from app.api.runner import JobTimeoutError, ServerBusyError, resolve
from app.ingestion.isolated_pipeline import PdfProcessingTimeoutError, PdfResourceError
from app.ingestion.pdf_resume_ingestion_service import (
    EmbeddingStageError, EmptyPdfInputError, EmptyResumeContentError, IngestionError, InvalidPdfInputError,
    NotAPdfError, PdfProcessingError, PdfTooLargeError, StorageStageError, TooManyChunksError)
from app.schemas.pipeline_api import ResumeUploadResponse

logger = logging.getLogger("app.api")
router = APIRouter()
RETRY_SECONDS = 5

# (exception class, HTTP status, code, fixed message, details). Most specific first; messages never echo exception text.
INGESTION_ERRORS = (
    (EmptyPdfInputError, 400, "EMPTY_FILE", "The uploaded file is empty.", None),
    (PdfTooLargeError, 413, "PAYLOAD_TOO_LARGE", "The uploaded file is too large.", None),
    (NotAPdfError, 415, "UNSUPPORTED_MEDIA_TYPE", "The uploaded file is not a PDF.", None),
    (InvalidPdfInputError, 400, "INVALID_FILE", "The uploaded file could not be read.", None),
    (PdfProcessingTimeoutError, 422, "PDF_PROCESSING_TIMEOUT", "The PDF took too long to process.", None),
    (PdfResourceError, 422, "PDF_RESOURCE_LIMIT", "The PDF needs more resources than are allowed.", None),
    (PdfProcessingError, 422, "PDF_UNPROCESSABLE", "The PDF is damaged, protected or could not be processed.", None),
    (EmptyResumeContentError, 422, "NO_TEXT_CONTENT", "No readable resume text was found (is this a scanned PDF?).", None),
    (TooManyChunksError, 422, "RESUME_TOO_LARGE", "The resume has more sections than can be stored and matched.", None),
    (EmbeddingStageError, 503, "EMBEDDING_UNAVAILABLE", "The embedding service is not available right now.", None),
    (StorageStageError, 502, "STORAGE_UNAVAILABLE",
     "The vector store failed. Some of this resume's data may have been written. Upload the same file again to repair it.",
     {"partial_write_possible": True, "retryable": True}),
)


def to_api_error(error: IngestionError) -> ApiError:
    for kind, status, code, message, details in INGESTION_ERRORS:
        if isinstance(error, kind):
            return ApiError(status, code, message, details)
    return ApiError(500, "INGESTION_FAILED", "The resume could not be ingested.")


def readiness_unconfirmed() -> ApiError:
    """Vectors were upserted, but the index could not be read to confirm they are searchable. Not a pre-write failure
    (so not STORAGE_UNAVAILABLE), not "still indexing" (so not 202): the state is unknown and the caller is told so."""
    return ApiError(
        502, "READINESS_UNCONFIRMED",
        "The resume was stored, but the index could not be checked, so it is not confirmed ready and no access token "
        "was issued. Upload the same file again to get a token, or try again shortly.",
        {"stored": True, "ready": False, "retryable": True},
        {"Retry-After": RETRY_SECONDS, "Cache-Control": "no-store"})


def busy() -> ApiError:
    return ApiError(503, "SERVER_BUSY", "The server is busy. Try again shortly.", headers={"Retry-After": RETRY_SECONDS})


@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(response: Response, file: UploadFile = File(...), limits: ApiLimits = Depends(get_api_limits),
                        caller: str = Depends(get_caller), signer: CandidateTokenSigner = Depends(get_token_signer),
                        runtime=Depends(get_runtime), ingestion=Depends(get_ingestion_service),
                        gate=Depends(get_readiness)):
    name = file.filename or ""
    if not name.lower().endswith(".pdf"):
        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Only PDF files are accepted.")   # real check is the content's
    data = await file.read(limits.max_upload_bytes + 1)     # the body cap already stopped oversized uploads upstream
    await file.close()
    if not data:
        raise ApiError(400, "EMPTY_FILE", "The uploaded file is empty.")
    if len(data) > limits.max_upload_bytes:
        raise ApiError(413, "PAYLOAD_TOO_LARGE", "The uploaded file is too large.")
    wait = runtime.upload_quota().acquire(caller)
    if wait > 0:
        raise ApiError(429, "QUOTA_EXCEEDED", "The upload quota for this API key has been used up.",
                       headers={"Retry-After": max(1, int(wait) + 1)})
    heavy = runtime.heavy()
    try:
        service = await resolve(ingestion, heavy)           # first use builds the model/clients; later calls are free
        result = await heavy.run(service.ingest_pdf, data, name)
    except ServerBusyError:
        raise busy() from None
    except JobTimeoutError:
        raise ApiError(504, "PROCESSING_TIMEOUT", "Processing took too long. Try again.") from None
    except IngestionError as error:
        logger.warning("ingestion rejected code=%s stage=%s", error.code, error.stage)
        raise to_api_error(error) from None
    except Exception as error:
        logger.error("ingestion failed type=%s", type(error).__name__)
        status = ApiError(503, "SERVICE_UNAVAILABLE", "The service is not available right now.") \
            if not getattr(ingestion, "built", True) else ApiError(500, "INGESTION_FAILED", "The resume could not be ingested.")
        raise status from None
    runtime.tracker.record(result.document_id, result.chunk_count)
    store = getattr(service, "pinecone_store", None)
    settings = runtime.settings()
    try:
        indexed = True if store is None else await gate.wait_until_indexed(
            runtime.light(), store, result.namespace, result.chunk_count, settings.readiness_timeout_seconds,
            settings.readiness_poll_seconds, runtime.sleep, runtime.clock)
    except Exception as error:                                # the write SUCCEEDED; only the readiness check failed
        reason = error.reason if isinstance(error, ReadinessUnavailableError) else type(error).__name__
        logger.warning("readiness unconfirmed after a successful write reason=%s", reason)
        raise readiness_unconfirmed() from None
    response.headers["Cache-Control"] = "no-store"            # the body carries a bearer access token
    response.headers["Pragma"] = "no-cache"
    if indexed:
        response.status_code = 200 if result.replaced_existing else 201
    else:
        response.status_code = 202
        response.headers["Retry-After"] = str(RETRY_SECONDS)
    return ResumeUploadResponse(
        status="INGESTED" if indexed else "PENDING_INDEXING", ready=indexed, document_id=result.document_id,
        namespace=result.namespace, chunk_count=result.chunk_count, vectors_upserted=result.vectors_upserted,
        page_count=result.page_count, replaced_existing=result.replaced_existing,
        embedding_model=result.embedding_model, embedding_dimension=result.embedding_dimension,
        access_token=signer.issue(result.document_id))
