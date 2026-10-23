# MVP scope: single company, one shared resume pool

**This MVP is single-company. It is NOT ready for multi-company (multi-tenant) exposure.**

## What the MVP does

- One company, one shared pool of resumes (all `resume:<document_id>` namespaces in the Pinecone index).
- `POST /api/v1/resume/upload` adds a PDF to the pool.
- `POST /api/v1/jd/rank` takes only `{"job_description": "..."}`. The server discovers the resumes in the pool,
  checks that each is fully indexed, and ranks them with the unchanged M13 skills-only matcher. HR never handles
  document ids or access tokens.
- `POST /api/v1/jd/match` (explicit `document_id` + `access_token` list) is unchanged and still available.
- The API-key contract is unchanged: `Authorization: Bearer <key>`, only SHA-256 hashes configured server-side
  (`API_KEY_HASHES`). The Pinecone key stays on the server; Angular only ever holds the API key typed in by the user.

## `/jd/rank` response contract

`ranking_basis`, `jd`, `retrieval` as in `/jd/match`, plus:

- `pool`: `discovered`, `considered`, `ranked`, `skipped`, `truncated`, `max_considered` (100).
  If `truncated` is true the pool is larger than the cap and the remainder was **not** ranked.
- `skipped[]`: `{document_id, reason}` for resumes not provably fully indexed (`EMPTY`, `STALE`) or too large
  (`TOO_LARGE`). They are reported, never ranked from partial data.
- `candidates[]`: as in `/jd/match`, plus `source_file` (original PDF name when the index holds it).

## Deliberately NOT implemented (deferred)

- No user login, roles, permissions, company membership or tenant isolation.
- Any holder of a valid API key can rank, and see excerpts from, **every** resume in the pool.
- API keys are static and shared per caller (see `app/api/auth.py`); `access_token` is not checked by `/jd/rank`.
- The pool cap is 100 resumes per request; a larger pool needs pagination or a pre-filter.
- Readiness memory (`ReadinessTracker`) is in-process only.

## Before serving a second company

Per-company namespaces or metadata filters, an identity provider, per-user authorization on `/jd/rank` and
`/resume/upload`, audit logging, and a pool-size strategy are all required first.

## Running without an API key (local / internal MVP)

Setting `API_AUTH_DISABLED=true` (in `.env` or the environment) makes the backend accept requests without
`Authorization`. Every request is treated as one shared `anonymous` caller, still rate limited. The default
(variable unset) is unchanged: an API key is required and the API fails closed. The Angular app sends no key.
**Anyone who can reach the backend port can upload and rank resumes while this is on**, so keep it on localhost or a
trusted internal network only, and turn it off (and use `API_KEY_HASHES`) before any wider exposure.
`CANDIDATE_TOKEN_SECRET` is still required for uploads and is set in `.env`.
