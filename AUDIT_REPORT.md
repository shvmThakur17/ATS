# AI Resume Screening System — project and environment audit

Audit date: 2026-09-23. Scope: `C:\Users\ShivamThakur\source\repos\resume`.

This is an audit of the supplied folder, not a certification that the application runs. All 38 application/test Python files, the VS Code launch configuration, environment-variable names/presence, both virtual-environment configurations, installed distribution metadata, and the project file inventory were inspected. No secret values are reproduced here. No application files, environments, packages, indexes, or Git history were changed during this audit.

Python execution returned Access is denied in the sandbox. The request to retry outside the sandbox was declined. Consequently interpreter health, imports, tests, API startup, model execution, and authenticated Pinecone access remain unverified. File presence and package metadata are explicitly distinguished from successful execution below.

## 1. MACHINE STATUS

- [OK] Windows/PowerShell; Git, Node, npm, VS Code, Azure CLI, and Azure Functions Core Tools resolve to installed command paths.
- [BROKEN] This session cannot resolve `python`, `pip`, or `py` on PATH. This does not prove Python is absent from the machine or another terminal.
- [WARNING] The base Python executable exists at `C:\Users\ShivamThakur\AppData\Local\Python\pythoncore-3.14-64\python.exe`, but its version could not be executed/verified.
- [WARNING] Port availability for 8000, 4200, and 11434 was not established: the listener inspection produced no usable evidence. Do not interpret that as proof that the ports are free.
- [MISSING] No Ollama command was discovered on PATH or executable in the inspected conventional per-user installation folder. Other installation locations remain possible.

## 2. PYTHON STATUS

- [OK] No project Python-version declaration exists. The newer `.venv/pyvenv.cfg` records 3.14.7; the older `venv/pyvenv.cfg` records 3.14.3. These are creation metadata, not measured runtime versions.
- [WARNING] Both reference the same existing base installation. There is no evidence supporting the initial hypothesis that its path no longer exists.
- [OK] Inspected dependency `Requires-Python` metadata does not exclude 3.14. The installed Torch 2.12.0 wheel is explicitly `cp314-cp314-win_amd64`; PyMuPDF and Pinecone use compatible-looking ABI3 Windows wheel tags. A blanket Python downgrade is not justified by these findings.
- [WARNING] Metadata cannot establish native DLL health, model compatibility, or a consistent dependency graph. Run imports and `pip check` before retaining this stack or selecting another interpreter.

## 3. VIRTUAL ENVIRONMENT STATUS

| Check | `.venv` | `venv` |
|---|---|---|
| Python executable | Present | Present |
| pip executable | Present | Present |
| PowerShell activation script | Present | Present |
| Creation Python | 3.14.7 | 3.14.3 |
| Installed distribution metadata | pip 26.2.1 only | Application stack and legacy packages |
| Recorded creation location | Current project | Old `resume-screening-system` folder |
| Executable validation | Blocked | Blocked |

- [MISSING] `.venv` has no application dependencies. Selecting it in VS Code cannot currently provide FastAPI/Uvicorn.
- [BROKEN] `venv/Scripts/activate.bat` hardcodes the old project directory. This proves stale batch activation configuration; it does not prove that the PowerShell activation script behaves the same way.
- [OK] Effective listed PowerShell policy is LocalMachine `RemoteSigned`, with higher scopes undefined. No observed policy error explains the missing prompt.
- [WARNING] Prompt appearance is not a reliable environment test. Check `$env:VIRTUAL_ENV` and `sys.executable`. Prefer explicit interpreter paths over activation and standalone pip launchers.
- [WARNING] Preserve both environments for now. Do not delete/recreate either until execution and package compatibility are checked.

## 4. PYTHON DEPENDENCY STATUS

[MISSING] No `requirements.txt`, `pyproject.toml`, dependency lockfile, or Python setup metadata exists in the supplied application folder. There are no duplicate project manifests because there are no manifests at all.

Observed direct dependencies and versions recorded under the older `venv`:

| Distribution | Why needed | Recorded version |
|---|---|---|
| fastapi | API and upload routing | 0.136.3 |
| uvicorn | Configured ASGI startup/debug module | 0.49.0 |
| pydantic | Request/response schemas | 2.13.4 |
| PyMuPDF | `fitz` PDF extraction | 1.27.2.3 |
| python-dotenv | Existing settings loader | 1.2.2 |
| python-multipart | FastAPI `File` upload support | 0.0.32 |
| sentence-transformers | Local Nomic embeddings | 5.6.0 |
| pinecone | Vector search/storage | 9.1.0 |
| requests | Ollama HTTP client | 2.34.2 |

- [WARNING] Relevant transitive/model stack: transformers 5.12.1, torch 2.12.0, einops 0.8.2, numpy 2.4.6, scikit-learn 1.9.0. Version presence is not proof of successful import or model loading.
- [OK] Unit tests use standard-library `unittest`; pytest is not required for the six existing unit tests. Neither environment contains pytest distribution metadata.
- [WARNING] `python-docx` 1.2.0 is present only in the older environment but is not imported by the application. It becomes a direct dependency when DOCX support is implemented.
- [OBSOLETE] ChromaDB 1.5.9, psycopg2-binary 2.9.12, SQLAlchemy 2.0.50, and OpenAI 2.41.0 are installed in the old environment but have no current application imports. Do not copy its entire freeze into a new production manifest.
- [WARNING] `tests/similarity_test.py` directly uses scikit-learn and a different MiniLM model. It is an experiment, not a Nomic regression test.
- [WARNING] Pin direct requirements and a resolved transitive lock only after validation. The recorded versions above are recovery candidates, not an approved working lock.

## 5. FASTAPI STATUS

- [OK] Entry point: `app.main:app`, launched from the project root. Routes: `GET /`, `POST /api/v1/resume/upload`, `POST /api/v1/jd/search`; standard FastAPI docs are enabled.
- [OK] Actual development command after dependencies are restored:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- [BROKEN] The proposed `.venv` currently lacks Uvicorn and all app dependencies.
- [WARNING] Importing `app.main` imports JD search, which imports `EmbeddingService` and constructs a SentenceTransformer immediately. Model download/loading can prevent even the health route from starting. Lazy initialization is a small, justified startup fix.
- [BROKEN] Uploads normalize and return a resume but never invoke chunking, embeddings, or Pinecone upsert. A newly uploaded resume cannot enter search through the current API.
- [BROKEN] The upload path uses the client-supplied filename directly. Paths can escape `uploads`, and repeated names overwrite files. The directory is relative to the process working directory and is not created by the endpoint.
- [WARNING] No upload size/type controls, graceful corrupt-file handling, authentication, or CORS configuration. Blocking PDF/model/network work runs inside async handlers, limiting concurrency. These are concrete production gaps, not reasons to redesign the architecture during environment repair.
- [WARNING] No API startup or endpoint request succeeded during this audit; execution was blocked before those checks.

## 6. DEBUGGER STATUS

- [OK] `.vscode/launch.json` already uses `type: debugpy`, `module: uvicorn`, and the correct `app.main:app` target. Its module path is not the underlying mistake.
- [OK] Installed extension folders include Python 2026.4.0, Python Debugger 2026.6.0, Pylance, and Python Environments. The CLI extension listing was blocked; folder presence does not prove extension activation.
- [MISSING] No workspace settings, tasks, or explicit interpreter selection. Existing launch config lacks explicit `python`, `cwd`, and `envFile`.
- [WARNING] Current `--reload` uses a child process; subprocess debugging is not explicitly configured. A single-process F5 launch removes that uncertainty.
- [WARNING] The selected VS Code interpreter could not be observed. Therefore it is not proven that interpreter selection or reload is the sole debugger failure.
- [OK] The VS Code Python Debugger extension supplies debugpy; absence of a project debugpy distribution alone is not a reason to install it. A CLI `python -m debugpy` workflow would require that package separately.

Recommended single backend launch configuration, after `.venv` is validated:

```json
{
  "version": "0.2.0",
  "configurations": [{
    "name": "FastAPI",
    "type": "debugpy",
    "request": "launch",
    "python": "${workspaceFolder}/.venv/Scripts/python.exe",
    "module": "uvicorn",
    "args": ["app.main:app", "--host", "127.0.0.1", "--port", "8000"],
    "cwd": "${workspaceFolder}",
    "envFile": "${workspaceFolder}/.env",
    "console": "integratedTerminal",
    "justMyCode": true
  }]
}
```

Set `python.defaultInterpreterPath` to `${workspaceFolder}/.venv/Scripts/python.exe` and explicitly select that interpreter for an already-open workspace. Set a breakpoint in `health`, press F5, then request `/`. See [VS Code's debugger documentation](https://code.visualstudio.com/docs/python/debugging).

## 7. ANGULAR STATUS

- [MISSING] No Angular source, `package.json`, `package-lock.json`, `angular.json`, TypeScript configuration, Angular environment files, or frontend debug configuration exists anywhere in this project folder.
- [WARNING] Angular version, CLI version, supported Node range, npm scripts, API URL, frontend authentication, and build state cannot be inferred. This folder alone cannot satisfy the full-stack recovery goal.
- [WARNING] Restore or identify the original Angular checkout before running npm commands. Do not create a replacement Angular project or automatically upgrade it.

## 8. NODE/NPM STATUS

- [OK] Executed Node version: 22.17.1. Executed npm version: 10.9.2.
- [WARNING] Compatibility with this project's Angular version is unknown because its manifests are absent.
- [WARNING] No project `node_modules` is present, but installing it in this backend folder would be incorrect. Once the original frontend and lockfile are available, use its documented scripts and normally `npm ci` for reproducibility.

## 9. OLLAMA STATUS

- [OK] Existing settings match the intended localhost port 11434 and `qwen3:4b` configuration. Actual `.env` values are not copied into this report.
- [MISSING] Ollama was not found in the inspected command/install locations. No qwen3 manifest was found at the conventional per-user model location.
- [WARNING] Service reachability and installed model inventory remain unverified; no inference request was made.
- [OK] Ollama is used only after ranking, on the top five candidates and their top three matching chunks. It is not used for extraction.
- [WARNING] `num_predict=150` can truncate structured reasoning output. The service falls back to parsing `thinking` as JSON and does not validate output with its defined `AIAnalysisResponse` schema. Validate a synthetic response before treating this flow as reliable.

## 10. NOMIC STATUS

- [OK] Uses `sentence_transformers.SentenceTransformer`, model `nomic-ai/nomic-embed-text-v1.5`, locally. It does not use the Nomic API client or require `NOMIC_API_KEY`.
- [OK] Expected full output dimension is 768. The code does not truncate embeddings. Actual output length is unverified.
- [BROKEN] `generate` encodes raw text without retrieval task prefixes. The model documentation specifies `search_query:` for queries and `search_document:` for indexed documents. Fixing this requires consistent document/query handling and a plan for existing vectors, not a silent change to only queries. See the [official model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).
- [WARNING] No Nomic cache folder was found in the conventional Hugging Face hub cache. An alternate cache location is possible. Initial model load may require a download.
- [WARNING] Model code is loaded with `trust_remote_code=True` and no revision pin. Verify the existing stack and select a reviewed revision for reproducibility; do not replace the model merely because this option exists.

## 11. PINECONE STATUS

- [OK] Existing code reads `PINECONE_API_KEY` and `PINECONE_INDEX_NAME`. Both are populated in `.env`; credential validity is unverified. The index name is intentionally not reproduced.
- [OK] Uses `Pinecone(api_key=...)` and `pc.Index(...)`. No index-creation path, cloud/region settings, or explicit namespace exists. Operations use the SDK's default namespace.
- [WARNING] Actual index dimension, metric, region, host, and vector count require an authenticated read-only description. No index was created, modified, or queried during the audit. Compare its dimension to a measured 768-element embedding before upsert.
- [BROKEN] `upsert_test_vector` writes `test-1`, then references undefined `chunks`, `embeddings`, and `file_name`. Do not run this helper: it can mutate the index and then fail.
- [WARNING] The separate `upsert_chunks` implementation exists but is disconnected from the upload route. It uses `zip`, which silently truncates unequal input counts, and filename-based vector IDs, which can collide and leave stale chunks on replacement.
- [WARNING] Upsert and search code print vector metadata/resume text. Remove these prints before using real candidate data in production logs.

## 12. ENVIRONMENT VARIABLES / TOKEN STATUS

| Variable | Local `.env` | Process environment | Required by current code |
|---|---|---|---|
| PINECONE_API_KEY | SET | UNSET | Yes, vector operations |
| PINECONE_INDEX_NAME | SET | UNSET | Yes, vector operations |
| OLLAMA_BASE_URL | SET | UNSET | Yes, reasoning |
| OLLAMA_MODEL | SET | UNSET | Yes, reasoning |
| NOMIC_API_KEY | Not configured | UNSET | No |

- [OK] Existing `python-dotenv` settings approach loads `.env`; process-level unset variables are not automatically missing application settings.
- [WARNING] Settings are captured at import time, with no required-value validation. Missing values fail later in SDK/HTTP calls. Pin `.env` loading to the repository path and retain process-environment precedence.
- [MISSING] No root `.gitignore` or `.env.example`. Existing per-venv ignore files do not protect the root `.env`, uploads, or app bytecode.
- [WARNING] Git tracking/history cannot be checked because this folder is not a Git repository. There is no basis to claim that secrets were or were not committed previously.
- [OK] No literal application API-key assignment was identified in the inspected Python source; the Pinecone key comes from settings. No secret value was logged by the audit.

Proposed `.env.example` contains only required names and public defaults:

```dotenv
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
```

Preserve the existing `.env`; do not overwrite it with this example. Add root ignore rules for `.env`, `.env.*` with `!.env.example`, `.venv/`, `venv/`, `node_modules/`, `__pycache__/`, `*.py[cod]`, and `uploads/`. Review tracking after restoring the original Git checkout before making any index changes.

## 13. GIT STATUS

- [OK] Git 2.55.0.windows.5 runs.
- [MISSING] `git status`, `git branch`, and `git remote -v` report that this folder is not a Git repository. There is no `.git` metadata in the supplied folder or a recognized parent.
- [WARNING] This may be an exported/copied working tree. Branch, remote, uncommitted changes, and secret history cannot be recovered from these files alone.
- [WARNING] Recover the original checkout or obtain the original remote separately and compare files. Do not initialize a replacement history, overwrite this folder, or delete branches as an environment repair.

## 14. AZURE STATUS

- [OK] Azure CLI and Functions Core Tools command paths exist. `func --version` returned 4.14.0.
- [WARNING] Azure CLI runtime version was not executed/verified.
- [OK] No Functions host configuration, function app, Azure SDK imports, Azure deployment configuration, or Docker setup is present in the supplied project. Azure tools are not prerequisites for its current FastAPI startup. No installation is warranted.

## 15. MISSING COMPONENTS

- [MISSING] Reproducible dependency manifest/lock, README, root ignore rules, and environment example.
- [MISSING] Original Angular project and Git metadata.
- [MISSING] DOCX extraction: `app/utils/docx_reader.py` is empty; the active parser unconditionally uses PyMuPDF.
- [MISSING] Layout processing: `resume_layout_service.py` is empty. PDF parsing uses plain `get_text("text")`, with no coordinates, blocks, font/bold information, explicit reading-order reconstruction, or two-column algorithm. `_recover_column_order` is a no-op.
- [MISSING] Upload-to-chunk-to-embedding-to-index orchestration.
- [MISSING] Structured date fields, extraction quality validation, and a preserved output field for unassigned content. Pydantic currently validates response shape only.
- [MISSING] Focused PDF/DOCX, normalization, full skills/projects/work experience, chunking, embedding-contract, mocked Pinecone/Ollama, API, and end-to-end tests.

## 16. BROKEN COMPONENTS

- [BROKEN] The new `.venv` is dependency-empty; it cannot run the API as supplied.
- [BROKEN] Old batch activation uses a stale absolute path.
- [BROKEN] Uploads cannot become searchable candidates through the current API.
- [BROKEN] The Pinecone test-vector helper contains undefined-variable code after a write.
- [BROKEN] Untrusted upload names are used as filesystem paths.
- [BROKEN] Work-experience extraction assigns a detected date period to `role`, rather than a job title. It can split employer/role information incorrectly. Fix only with regression examples that define the expected grouping.
- [WARNING] Summary extraction stops at title-like/date lines; project descriptions are assigned heuristically after collecting titles; unassigned content is not returned. These need fixtures before changes, not unsupported claims of parsing accuracy.
- [WARNING] No circular import or unresolved application module path was identified statically. Missing `__init__.py` files alone are not an error: Python supports namespace packages when launched from the correct root.

## 17. OBSOLETE COMPONENTS

- [OBSOLETE] Empty repository-layer files for embedding, JD parsing, matching, OpenAI, and resume parsing; empty Chroma repository schema. They are placeholders, not working alternate backends.
- [WARNING] Empty constants/logger/request/response and PDF utility files may be unfinished scaffolding. Preserve them during recovery.
- [OBSOLETE] `app/services/pdf_reader.py` duplicates basic PDF extraction and has no current caller. The active parser lives in `app/services/resume_parser.py`.
- [OBSOLETE] Installed Chroma/PostgreSQL/OpenAI stacks and the MiniLM similarity experiment do not implement the intended Nomic/Pinecone pipeline. Do not reinstall them as direct app requirements.

## 18. RECOMMENDED FIXES

Use the requested phase order. Verify each phase before proceeding; no broad rewrite is needed.

1. **Machine prerequisites:** resolve the Python execution restriction and identify the original Angular/Git locations. Restart terminals after any PATH repair. Do not reinstall installed tools speculatively.
2. **Python interpreter:** execute the existing base/venv interpreters and inspect `sys.executable`, `sys.prefix`, and `sys.base_prefix`. Keep 3.14 if imports/model tests succeed.
3. **Virtual environment:** use `.venv` if its interpreter is healthy. Recreate only if proven invalid, preserving the old environment as recovery evidence.
4. **Dependencies:** validate the old environment with `pip check` and imports, then build a minimal pinned manifest from actual imports. Resolve in dry-run mode before installation into `.venv`; do not bulk-freeze legacy packages.
5. **Environment:** add ignore rules and `.env.example`, retain the existing dotenv mechanism with an explicit path, and validate required settings without printing values.
6. **FastAPI:** lazy-load embeddings, prove health/docs and a synthetic PDF upload, then separately fix upload containment and file handling. Keep the existing module structure.
7. **Debugger:** apply the single configuration in section 6, select `.venv`, and prove an actual breakpoint hit in the health endpoint.
8. **Ollama:** locate/install the runtime only if absent; verify service and model list, fetch qwen3:4b only if missing, test synthetic structured reasoning.
9. **Nomic:** load the existing model, measure dimension, and test query/document prefixes. Plan any required re-embedding of existing data before changing retrieval behavior.
10. **Pinecone:** describe the existing index read-only, check dimensions/metric, and test SDK response compatibility with synthetic data in an isolated test namespace. Repair the broken helper and validate input lengths. Do not create a new index by default.
11. **Angular:** recover the original source, inspect its lockfile/version range, install with its supported Node version, run its build/start scripts, and verify API connectivity.
12. **Azure:** no action for this supplied code; reassess only if original deployment files are recovered.
13. **End-to-end:** wire the missing ingestion steps without replacing parser/model architecture; address DOCX/layout gaps with focused fixtures, then verify upload, retrieval, ranking, reasoning, and frontend interaction.

The first runtime gate is currently blocked by denied execution. Dependency installation and debugger validation cannot honestly be marked complete before that is resolved.

## 19. EXACT COMMANDS TO RESTORE THE MACHINE

These are staged recovery commands, not commands already executed successfully. Start in the current project root. Stop on a failing phase rather than running everything blindly.

```powershell
Set-Location 'C:\Users\ShivamThakur\source\repos\resume'

# Phase 1–3: diagnose existing interpreters without activation or recreation.
& .\.venv\Scripts\python.exe --version
& .\.venv\Scripts\python.exe -c "import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix)"
& .\.venv\Scripts\python.exe -m pip --version
& .\venv\Scripts\python.exe --version
& .\venv\Scripts\python.exe -m pip --version
& .\venv\Scripts\python.exe -m pip check

# Phase 4: inspect actual core imports in the older environment.
# Model loading is deliberately a later check.
& .\venv\Scripts\python.exe -c "import fastapi, uvicorn, fitz, dotenv, multipart, pydantic, requests, pinecone; print('Core imports OK')"
& .\venv\Scripts\python.exe -c "import torch, sentence_transformers; print('Embedding imports OK')"

# After creating and reviewing a validated requirements.txt (currently absent):
& .\.venv\Scripts\python.exe -m pip install --dry-run -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip check

# Existing offline unit tests: exactly six tests, no external-service scripts.
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_resume_section_service.py' -v

# Phase 6: start API without reload for the first smoke test.
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, while the API is running:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/
Invoke-WebRequest http://127.0.0.1:8000/openapi.json | Select-Object StatusCode
```

Once Ollama is available on PATH:

```powershell
ollama --version
ollama list
# Run only if qwen3:4b is absent:
ollama pull qwen3:4b
# Run only if the local service is not already running:
ollama serve
```

In another terminal:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags |
    Select-Object -ExpandProperty models |
    Select-Object name
```

Optional PowerShell activation after interpreter validation:

```powershell
. .\.venv\Scripts\Activate.ps1
$env:VIRTUAL_ENV
python -c "import sys; print(sys.executable)"
```

No honest exact Angular installation/start command or Python reinstallation command can be selected yet: the frontend manifest is absent and the existing interpreter has not been proven invalid. Do not run `pip freeze` as the final manifest, broad pytest discovery, the Pinecone write helper, or sample scripts that print real resumes as restoration checks.

## 20. FINAL VALIDATION CHECKLIST

- [x] [OK] Supplied source/configuration inventory inspected and actual startup module identified.
- [x] [OK] Both environments inspected without deletion/recreation.
- [x] [OK] Required setting names/presence checked without exposing secret values.
- [x] [OK] Node, npm, Git, and Functions Core Tools versions measured.
- [ ] [WARNING] Execute Python and pip; establish actual runtime and native-library health.
- [ ] [MISSING] Restore minimal pinned dependency manifest and populate `.venv`.
- [ ] [MISSING] Add root ignore rules and environment example; verify tracking in recovered Git checkout.
- [ ] [WARNING] Run the six existing section-detection tests. They cover heading aliases, numbering, inline headings, uppercase skills, ignored sections, and repeated sections—not full parser accuracy.
- [ ] [WARNING] Start API, receive health/OpenAPI responses, and hit an F5 breakpoint.
- [ ] [MISSING] Add critical synthetic PDF/DOCX and API upload fixtures, including path containment and invalid-file cases.
- [ ] [WARNING] Verify local Nomic embedding dimension and retrieval prefixes.
- [ ] [WARNING] Describe existing Pinecone index and verify dimension/metric compatibility without writes.
- [ ] [WARNING] Verify qwen3:4b availability and valid schema-conforming reasoning.
- [ ] [MISSING] Connect ingestion to index storage and test retrieval of an uploaded synthetic candidate.
- [ ] [MISSING] Recover Angular, validate Node compatibility, build, and connect to the API.
- [ ] [WARNING] Separate the four manual integration/experiment scripts from offline test discovery. They execute model/network operations at import time and have no automated assertions.
- [ ] [WARNING] Production readiness: add bounded file handling, appropriate access control, concurrency/error handling, and remove candidate-content logging before deployment.

Current conclusion: the folder contains a partially implemented backend and recoverable dependency evidence. It is not merely a formatted-machine setup problem, and it cannot yet be certified fully runnable or production-ready. Runtime verification and missing original components are the next gates; this report preserves the existing architecture and business logic.
