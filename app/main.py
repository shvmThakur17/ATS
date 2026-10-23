import os
from fastapi import FastAPI
from .api.app_factory import configure_app
# Interactive docs are useful in development and are on by default. In a deployment, set API_DOCS_ENABLED=false in the
# process environment to stop publishing the API schema (read here, at construction, from the real environment only).
DOCS_ENABLED = os.environ.get("API_DOCS_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")
app = FastAPI(
    title="Resume Screening API",
    version="1.0.0",
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
)
# N3: skills-only upload/match routers, error handlers and body-size caps. The legacy modules
# (app/api/v1/resume.py, app/api/v1/jd.py) are no longer imported here: jd.py pulled in a model loaded at
# import time and an Ollama-based, similarity-ranked search.
configure_app(app)

@app.get("/")
async def health():
    return {"status": "running"}