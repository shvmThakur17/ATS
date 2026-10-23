"""Per-application runtime: settings, limiters, job runners, lazy services and the readiness gate.

Everything is created lazily and at most once per application object (thread-safe), so importing the API builds
nothing, a request that is rejected early never triggers a model load or a Pinecone client, and tests get an
isolated runtime for every app they build.
"""

import asyncio
import threading
import time
from typing import Callable

from ..ingestion.pdf_resume_ingestion_service import IngestionLimits, PdfResumeIngestionService
from ..matching.models import MatchingConfig
from ..matching.pinecone_match_service import PineconeMatchLimits, PineconeMatchService
from .rate_limit import SlidingWindowLimiter
from .readiness import ReadinessGate, ReadinessTracker
from .runner import JobRunner, LazyService
from .settings import ApiSettings, load_settings

MEBIBYTE = 1024 * 1024
LIGHT_WORKERS, LIGHT_MAX_PENDING, LIGHT_TIMEOUT = 4, 64, 15.0


class ApiRuntime:
    def __init__(self, settings_provider: Callable[[], ApiSettings] = load_settings,
                 clock: Callable[[], float] = time.monotonic, sleep=asyncio.sleep):
        self.settings = settings_provider
        self.clock, self.sleep = clock, sleep
        self.tracker = ReadinessTracker()
        self.embedder = LazyService(self.build_embedder)
        self.store = LazyService(self.build_store)
        self.ingestion = LazyService(self.build_ingestion)
        self.matching = LazyService(self.build_matching)
        self._lock = threading.Lock()
        self._parts: dict = {}

    def once(self, name: str, factory):
        with self._lock:
            if name not in self._parts:
                self._parts[name] = factory()
            return self._parts[name]

    # --- cheap, settings-driven parts ---------------------------------------------------------------------------
    def rate_limiter(self) -> SlidingWindowLimiter:
        return self.once("rate", lambda: SlidingWindowLimiter(self.settings().rate_limit_per_minute, 60, self.clock))

    def upload_quota(self) -> SlidingWindowLimiter:
        return self.once("quota", lambda: SlidingWindowLimiter(self.settings().upload_quota_per_day, 86_400, self.clock))

    def heavy(self) -> JobRunner:
        """One worker: the shared embedder and Pinecone client are used by one job at a time, and the queue is capped."""
        settings = self.settings()
        return self.once("heavy", lambda: JobRunner("heavy", 1, settings.max_pending_jobs, settings.job_timeout_seconds))

    def light(self) -> JobRunner:
        return self.once("light", lambda: JobRunner("light", LIGHT_WORKERS, LIGHT_MAX_PENDING, LIGHT_TIMEOUT))

    def gate(self) -> ReadinessGate:
        return self.once("gate", lambda: ReadinessGate(self.tracker, self.settings().max_retrieval_depth))

    # --- expensive services: built on first use, exactly once ---------------------------------------------------
    def build_embedder(self):
        self.settings()                                   # makes sure .env was read before the model/client are built
        from ..parsing.embeddings import NomicEmbedder
        return NomicEmbedder()

    def build_store(self):
        self.settings()
        from ..parsing.vector_store import PineconeStore, VectorStoreConfig
        return PineconeStore(VectorStoreConfig.from_env())

    def build_ingestion(self) -> PdfResumeIngestionService:
        from ..ingestion.isolated_pipeline import IsolatedResumePipeline
        settings = self.settings()
        return PdfResumeIngestionService(
            parser_pipeline=IsolatedResumePipeline(settings.parse_timeout_seconds, settings.parse_memory_mb * MEBIBYTE),
            embedder=self.embedder.get_blocking(), pinecone_store=self.store.get_blocking(),
            limits=IngestionLimits(settings.max_upload_bytes, settings.max_chunks))

    def build_matching(self) -> PineconeMatchService:
        settings = self.settings()
        return PineconeMatchService(self.store.get_blocking(), self.embedder.get_blocking(), MatchingConfig.from_env(),
                                    PineconeMatchLimits(settings.max_retrieval_depth))

    def shutdown(self) -> None:
        for name in ("heavy", "light"):
            runner = self._parts.get(name)
            if runner:
                runner.shutdown()
