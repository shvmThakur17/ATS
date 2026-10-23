# Architecture and file map

Single-company MVP: Angular frontend, FastAPI backend, local embedding model, Pinecone vector index.
See `mvp-single-company.md` for scope limits.

```
Browser (Angular, :4200)
   |  /api/* via dev proxy (frontend/proxy.conf.json)
   v
FastAPI (:8000)   middleware -> routes -> services
   |                                   |            |
   |                         Ingestion (M1-M12)   Matching (M13 + N1)
   |                                   |            |
   +-----------------------------> Embedder (local Nomic) 
                                       |            |
                                       v            v
                                   Pinecone index, one namespace per resume
```

Request flows

- Upload: select PDFs -> upload queue (one at a time) -> `POST /api/v1/resume/upload` -> ingestion
  (M1-M9 parse, M10 chunk, M11 embed, M12 store) -> wait until vectors are searchable -> `INGESTED` or `PENDING_INDEXING`.
- Search: paste JD -> `POST /api/v1/jd/rank` -> list resume namespaces -> readiness check per resume (unready ones are
  skipped and reported) -> M13 skills-only ranking -> ranked list with evidence.

Legend: **ACTIVE** = on the current request path. **FROZEN** = M1-M13, hash-checked by
`docs/m13_frozen_manifest.sha256` (M13) and not to be edited. **LEGACY** = old code, not reachable from `app.main`.

## 1. Frontend (`frontend/`, Angular 20)

| File | Responsibility |
|---|---|
| `package.json`, `package-lock.json` | Dependencies and npm scripts (`npm start`, build, test) |
| `angular.json` | Build/serve/test config; serve uses `proxy.conf.json` |
| `proxy.conf.json` | Dev proxy: `/api` -> `http://localhost:8000` (no CORS needed) |
| `tsconfig*.json` | TypeScript settings for app and tests |
| `src/index.html`, `src/main.ts` | Page shell and bootstrap |
| `src/styles.css` | Global responsive styles (nav, cards, badges, tables) |
| `src/app/app.ts`, `app.html`, `app.css` | Shell: header, nav links ("Resume Upload", "JD Candidate Search"), router outlet |
| `src/app/app.config.ts` | Providers: HttpClient, router |
| `src/app/app.routes.ts` | Routes: `/` -> `/upload`, `/upload`, `/search` |
| `src/app/models.ts` | TypeScript types mirroring `app/schemas/pipeline_api.py` and `pool_api.py` |
| `src/app/resume-api.service.ts` | The only HTTP layer: `upload()` (with progress), `rank()`, error normalization |
| `src/app/upload-queue.service.ts` | Upload queue state: queued/uploading/indexed/pending/failed, sequential upload, retry, remove |
| `src/app/upload.page.ts/.html` | Resume Upload page: file picker, per-file status, progress, retry |
| `src/app/search.page.ts/.html` | JD Candidate Search page: JD input, ranked results, evidence, skipped/truncated notices |
| `src/app/testing.ts` | Test fixtures (mock responses, helpers) |
| `src/app/*.spec.ts` | Unit tests with mocked HTTP: routing, upload queue, search page |

## 2. Backend entry and API layer (`app/`)

| File | Status | Responsibility |
|---|---|---|
| `app/main.py` | ACTIVE | Creates the FastAPI app, health route `/`, calls `configure_app` |
| `app/api/app_factory.py` | ACTIVE | Registers routers, error handlers, middleware (CORS, auth gate, body cap) |
| `app/api/auth.py` | ACTIVE | Auth gate: API key, or anonymous when `API_AUTH_DISABLED=true`; rate limit; protected paths |
| `app/api/cors.py` | ACTIVE | Exact-origin CORS allow-list (`API_CORS_ORIGINS`) |
| `app/api/body_limit.py` | ACTIVE | Request-size caps, enforced while streaming |
| `app/api/rate_limit.py` | ACTIVE | Sliding-window limiter (requests/minute, uploads/day) |
| `app/api/settings.py` | ACTIVE | All env configuration, loaded once from `.env` |
| `app/api/runtime.py` | ACTIVE | Lazily builds the embedder, Pinecone store, ingestion and matching services; job runners |
| `app/api/runner.py` | ACTIVE | Bounded worker pools, timeouts, thread-safe lazy services |
| `app/api/dependencies.py` | ACTIVE | FastAPI dependencies and request limits |
| `app/api/errors.py` | ACTIVE | One sanitized error envelope `{error:{code,message}}` |
| `app/api/candidate_access.py` | ACTIVE | HMAC access tokens issued on upload (used by `/jd/match`) |
| `app/api/readiness.py` | ACTIVE | Decides whether a resume is fully searchable (READY/EMPTY/STALE/TOO_LARGE) |
| `app/api/evidence.py` | ACTIVE | Short, redacted evidence excerpts (no emails/phones) |
| `app/api/v1/resume_upload.py` | ACTIVE | `POST /api/v1/resume/upload` |
| `app/api/v1/jd_matching.py` | ACTIVE | `POST /api/v1/jd/match` (explicit candidates + tokens); retired `/jd/search` returns 410 |
| `app/api/v1/jd_pool.py` | ACTIVE | `POST /api/v1/jd/rank` (server-side discovery of the whole pool) |
| `app/api/v1/jd.py`, `app/api/v1/resume.py` | LEGACY | Old routes, no longer imported |
| `app/schemas/pipeline_api.py` | ACTIVE | Upload and match request/response models |
| `app/schemas/pool_api.py` | ACTIVE | `/jd/rank` request/response models (pool summary, skipped, source file) |

## 3. Ingestion pipeline (FROZEN M1-M12, PDF -> Pinecone)

| File | Stage | Responsibility |
|---|---|---|
| `app/ingestion/pdf_resume_ingestion_service.py` | N2 | Orchestrates one PDF: M1-M9 -> M10 -> M11 -> M12; typed errors |
| `app/ingestion/isolated_pipeline.py` | N2 | Runs the parser in a child process with time and memory limits |
| `app/parsing/models.py`, `pdf_objects.py` | M1 | PDF bytes -> geometry (text spans, images, vectors), no reading order |
| `app/parsing/layout/*` (`layout_analyzer`, `region_detector`, `relationship_detector`, `table_detector`, `block_features`, `geometry`, `models`) | M2 | Physical layout: regions, columns, table-like signals |
| `app/parsing/reading_order/*` | M3 | Deterministic reading order over layout regions |
| `app/parsing/logical_blocks/*` | M4 | Groups lines into logical blocks |
| `app/parsing/sections/*` (`section_detector`, `section_rules`, `models`) | M5 | Detects sections (skills, projects, experience, ...) |
| `app/parsing/entities/*` (`entity_extractor`, `entity_rules`, `models`) | M6 | Atomic entities (name, email, skill, dates, ...) |
| `app/parsing/records/*` | M7 | Groups entities into records (jobs, projects, education) |
| `app/parsing/normalization/*` (`text`, `date`, `contact`, `skill`, `record` normalizers, `validators`, `models`) | M8 | Safe normalization; raw values are kept |
| `app/parsing/canonical/*` (`builder`, `models`) | M9 | Stable `CanonicalResume` contract |
| `app/parsing/chunks/*` (`compiler`, `renderers`, `models`) | M10 | Semantic chunks (skills chunk, one chunk per project, ...) |
| `app/parsing/embeddings/*` (`nomic_embedder`, `models`) | M11 | Local Nomic embeddings for chunks and queries |
| `app/parsing/vector_store/*` (`pinecone_store`, `metadata`, `models`) | M12 | Pinecone upsert/query/stats; namespace `resume:<document_id>`; flat metadata |

## 4. Matching (FROZEN M13 + N1 wrapper)

| File | Status | Responsibility |
|---|---|---|
| `app/matching/jd_parser.py` | FROZEN M13 | Deterministic JD parse into required and preferred skills (no LLM) |
| `app/matching/skill_matching.py` | FROZEN M13 | Classifies evidence per skill (explicit skill, text evidence, semantic, not found) |
| `app/matching/scoring.py` | FROZEN M13 | Informational evidence helpers |
| `app/matching/ranking.py` | FROZEN M13 | Skills-only ranking key: required coverage, preferred, then semantic tie-breaks |
| `app/matching/retrieval.py` | FROZEN M13 | Query embedding and Pinecone retrieval per namespace |
| `app/matching/jd_matcher.py` | FROZEN M13 | One JD against many resumes, explainable result |
| `app/matching/models.py`, `__init__.py` | FROZEN M13 | Models, config, errors, public exports |
| `app/matching/pinecone_canonical_adapter.py` | N1 | Rebuilds the minimal resume model from Pinecone metadata |
| `app/matching/pinecone_match_service.py` | N1 | Sizes retrieval per candidate and calls M13 with an explicit namespace list |

## 5. Legacy code (not reachable from `app.main`)

Only referenced by other legacy files: `app/services/*` (old parsing/chunking/Pinecone/Ollama search),
`app/repositories/*`, `app/utils/*` (pdf/docx/text helpers), `app/core/*` (old config/logger/constants),
`app/schemas/candidate_response.py`, `chroma_repository.py`, `job_search_request.py`, `job_search_response.py`,
`normalized_resume_response.py`. Kept in place; safe to remove in a separate cleanup task.

## 6. Scripts, docs, tests

| Path | Responsibility |
|---|---|
| `scripts/inspect_*.py` | Developer tools to print each stage's output for one resume (`inspect_jd_matching.py` is M13-frozen) |
| `docs/deterministic-resume-pipeline.md`, `m2-layout-analysis.md`, `m5-section-detection.md` | Pipeline design notes |
| `docs/m13-skills-only-ranking.md`, `m13_frozen_manifest.sha256`, `m13_reference_v1/` | M13 spec, frozen file hashes, reference copy |
| `docs/mvp-single-company.md` | MVP scope, `/jd/rank` contract, auth-disabled mode, limits |
| `docs/architecture.md` | This file |
| `tests/api/` | API tests: upload, match, rank (`test_pool_rank.py`), hardening, lifecycle (offline, in-memory Pinecone fake) |
| `tests/ingestion/`, `tests/parsing/**` | Per-stage tests M1-M13 and N1/N2 |
| `tests/test_pdf_objects.py`, `test_resume_section_service.py` | Stage-one and legacy section tests |
| `tests/ollama_test.py`, `pinecone_test.py`, `pinecone_search_test.py`, `similarity_test.py` | Manual scripts that call real Ollama/Pinecone; excluded from offline runs |
| `tests/fixtures/` | Synthetic sample JDs |

## 7. Root and configuration

| Path | Responsibility |
|---|---|
| `.env` | Pinecone/Ollama settings, `API_AUTH_DISABLED`, `CANDIDATE_TOKEN_SECRET` (never commit) |
| `requirements.txt` | Python dependencies |
| `uploads/` | Sample PDFs from earlier manual runs (not used by the API) |
| `outputs/` | Saved inspection output from `scripts/` |
| `.vscode/launch.json` | Debug launch for uvicorn |
| `AUDIT_REPORT.md` | Earlier audit notes |
| `.venv/`, `venv/`, `__azurite_db_blob*.json` | Local Python environments and Azurite state |

## Commands

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000     # backend
cd frontend; npm start                                              # frontend at http://localhost:4200
.\.venv\Scripts\python.exe -m pytest tests/api -q                   # backend API tests
cd frontend; npx ng test --watch=false --browsers=ChromeHeadless   # frontend tests
```
