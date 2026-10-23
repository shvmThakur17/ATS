# Deterministic resume pipeline architecture — v1

## Scope and rollout

The existing upload route and response contract remain unchanged until this pipeline passes its acceptance gates. New code lives in `app/parsing`, independently of embeddings, Pinecone, Ollama and API transport. No stage imports or invokes an LLM, embedding model, network service or probabilistic section classifier.

The first increment implements **PDF object extraction → geometric document IR**, plus shared downstream contracts. **M2 physical layout analysis is now implemented separately**; see [M2 design, usage and limitations](m2-layout-analysis.md). Stages after M2 below remain designs, not implemented capabilities. Supporting 1,000+ unknown layouts is a benchmark objective, not a property that can be asserted from a few fixtures. Unknown layouts must be accepted without forcing a confident parse.

Scanned PDFs require OCR: geometry cannot recover characters that exist only as pixels. Stage one preserves image placement and reports pages with no usable native text. An OCR adapter is a later, explicitly configured dependency. Under a strict prohibition on all AI, neural OCR is not acceptable; select an approved conventional OCR engine or return `NEEDS_OCR`. Do not claim that extraction alone reads scans. OCR-derived fonts/styles remain unknown, never inferred as native PDF facts.

## Shared contracts and invariants

- Immutable stage values with explicit schema/stage versions. Pin the PDF engine, OCR executable/language packs/configuration, rule bundles, taxonomies and chunk compiler. Document hash + all dependency/configuration versions form the reproducibility key. No timestamps, random IDs, environment-dependent sorting or network lookups in output.
- Coordinates are points in PyMuPDF's **unrotated page coordinate space**, top-left origin. Keep crop/media boxes, page rotation, page transformation and rotation matrices to map back to rendered/source coordinates. Use quads for rotated text, with bounding boxes for spatial indexing. Do not treat `page.rect` on a rotated page as the raw-text coordinate frame.
- Preserve source block/line/span indices, exact Unicode characters, character boxes/origins, span quads, font name/size/flags/color, line direction and source image/drawing references. PDF extraction indices are source identity, **not reading order**. Native PDF logical blocks are hints, not semantic paragraphs.
- Source refs include document SHA-256, page, object ID, bbox and optional span-relative character interval. The PDF plus engine version is required to reproduce native object IDs; xrefs alone are insufficient for text.
- Every semantic field contains raw text, optional normalized text/taxonomy identifier, status, provenance and rule evidence. `UNKNOWN` has no normalized value and a reason. Empty raw text is reserved for a genuinely absent field; a rejected ambiguous value retains its raw evidence.
- Confidence is a deterministic **rule-strength score**, not a probability. Keep rule IDs and reasons; calibrate scores against held-out labeled data before using acceptance thresholds. A high extraction-fidelity score must not imply a high entity-association score.
- Derived blocks/records/chunks retain parent IDs and source unions. Do not discard unknown sections or unassigned blocks. Validators reject dangling source refs, inverted/out-of-frame geometry, cycles, impossible dates and conflicting associations.
- Missing names/titles/degrees/certifications stay unknown. A skill mentioned in one job must not be silently assigned to all jobs. Taxonomy similarity and neighboring text alone are not proof of a credential or employment association.

## Stages and contracts

| Stage/version | Input → output | Algorithm and acceptance gate |
|---|---|---|
| pdf-objects/1 | PDF bytes → DocumentIR | PyMuPDF rawdict without sorting; chars, quads, image bounds, vector geometry, page transforms. Hash original bytes, preserve object order, release handles, enforce byte/page/object limits. Gate: no flattening and stable IDs, including rotation/image-only fixtures. |
| ocr-adapter/1 (conditional) | Image regions + page transforms → additional evidenced tokens | Approved pinned engine, fixed raster DPI/language/deskew settings. Map pixels through inverse render/deskew transforms into page coordinates. Keep OCR engine score separate; reconcile overlaps with native text, retaining rejected alternatives. Gate: scan, mixed scan/native, rotated scan and duplicate invisible text fixtures. |
| layout/1 | DocumentIR → LayoutIR | Build spatial index; estimate line height/glyph widths; detect sustained whitespace gutters, separator lines and alignment bands. Recursively segment page bands so full-width headings can bridge columns. Detect repeated headers/footers across pages. Keep competing regions UNKNOWN if thresholds tie. Gate: two/three columns, narrow sidebar, spanners, overlapping/rotated elements. |
| reading-order/1 | LayoutIR → OrderGraph | Nodes are logical region/line candidates. Add labeled edges from same-column baselines, region containment, spanner boundaries, table cells and cross-page continuation rules. Never connect all objects by global XY sort. Resolve contradictory edges by fixed rule precedence; retain rejected edges as issues. Stable topological order only after cycle checks; tie-break by source ID for reproducibility, flag incomparable semantic paths. |
| logical-blocks/1 | Geometry + graph → LogicalBlocks | Join lines only within compatible region, direction, font/indent/spacing context. Preserve bullets and paragraph boundaries. Detect ruled tables from intersecting strokes, unruled tables from repeated row/column alignments; keep cells, spans and row groups. Sidebars remain separate scopes. Gate: table records cannot interleave with main-column paragraphs. |
| sections/1 | LogicalBlocks → SectionIR | Versioned heading aliases plus font prominence, separator, whitespace, page position and repeated structural context. Exclude contact labels and sentence-internal aliases. Propagate boundaries within region streams; attach cross-page continuation only with explicit evidence. UNKNOWN sections retain content. |
| records/1 | Sections + graph → raw CanonicalResume | Regex/grammars for contact/dates; section-specific finite-state record assemblers. Attach title/company/date anchors by alignment, intervening boundaries and shared table row/block; use bullets as responsibilities. Multi-role employers and concurrent jobs remain distinct. Preserve competing associations as issues instead of choosing by nearest distance alone. |
| taxonomy/1 | Raw fields + versioned catalogs → normalized fields | Exact normalized alias matching with locale/context constraints. Each taxonomy has immutable version, stable IDs and explicit ambiguous aliases. Preserve input spelling and catalog version. Avoid substring guesses such as matching Java inside JavaScript; ambiguous degree/title abbreviations remain UNKNOWN. |
| validation/1 | CanonicalResume → ValidationReport | Validate source integrity, field/status consistency, dates/precision, record anchors, duplicate identities and evidence ownership. Unknowns are valid partial results. Errors block affected records from chunk compilation; warnings remain visible. |
| chunks/1 | Validated CanonicalResume only → SemanticChunks | One coherent job/project/education/skill group per chunk; split long responsibility lists at item boundaries under pinned token budgets. Repeat minimal evidenced parent context and retain field paths/source refs. Stable hash IDs include document/record/compiler versions. No raw PDF fallback; unknown facts are omitted or explicitly labeled. |

Stage functions should accept immutable input + explicit configuration and return a typed result with issues. Do not modify earlier artifacts in place. Store artifacts per document/version so a failed stage can be inspected/replayed without re-running others. Native-text extraction is deterministic only within a pinned engine/platform configuration; do not promise byte-identical output across engine upgrades.

## Canonical model

`CanonicalResume` contains contact fields, summary fields, experience records, education records, skills, projects, certifications, unassigned evidenced values, and stage/taxonomy versions. Shared `FieldValue` wraps **each** value; dates are evidence-bearing raw/normalized fields with explicit precision in the later date grammar (never invent a day/month). `Record` uses a typed kind with named fields and named lists of evidenced items so each responsibility and associated skill has its own provenance. Kind-specific validators will enforce the following schemas before compilation:

| Kind | Scalar fields | Repeated fields |
|---|---|---|
| experience | title, company, location, start_date, end_date | responsibilities, skills |
| education | institution, degree, field_of_study, start_date, end_date | honors |
| project | name, role, start_date, end_date, url | descriptions, skills |
| certification | name, issuer, issue_date, expiry_date, credential_id, url | none |

Contact fields: name, emails, phones, links, location. Prefer deterministic syntax validation over guessing identity from a filename. Phone normalization requires a known country context; otherwise preserve raw text. Missing dates stay unknown. Taxonomy normalization must not change the observed seniority, employment duration or degree level.

## Validation and coverage plan

Use synthetic PDFs for public regression fixtures; never commit uploaded real resumes. Add stage-local tests before each implementation and retain golden IR/canonical artifacts with expected provenance. The eventual corpus should cover 1,000+ layouts with author/template-family holdouts to prevent template memorization, languages/fonts, page sizes, extraction order permutations, partial scans, tables, sidebars, headers and page continuations. Separately measure OCR character error, region accuracy, order-edge accuracy, section boundaries, entity precision/recall, record association accuracy, provenance coverage, abstention rate, confidence calibration, latency and memory.

Release gates must be set from actual business tolerances and labeled corpus results; no unsupported accuracy percentage is specified here. Require 100% schema-valid output and provenance coverage for known values, no cross-record leakage in the critical fixtures, and explicit unresolved output for unsupported cases. Test chunk boundaries and source coverage separately from parser quality.

## Increment one: usage and limits

```python
from app.parsing.pdf_objects import extract_pdf

document_ir = extract_pdf("resume.pdf")
# Inspect document_ir.pages and document_ir.issues; no text flattening occurs.
```

The extractor accepts PDF bytes or a local path, does not send data anywhere, does not run OCR, and does not change the upload endpoint. It preserves native text spans/characters and image/vector placement. Outlined text, broken character maps, image-only pages, mixed raster content and rotated lines surface issues for later stages. It does not claim PDF tagging, arbitrary graphics-path reconstruction, or semantic table detection yet. Vector paths are represented by their bounding region and separator strokes; original PDF references preserve access to richer graphics for a later table detector.

In-process limits reduce accidental overuse but cannot make hostile PDFs safe: production decoding belongs in a resource-limited worker with CPU/memory/time quotas. A cancellation/timeout boundary must be added at API integration. Do not auto-process candidate data through unfinished stages.

M2 layout segmentation is available through `app.parsing.layout.analyze_layout`. The next milestone is M3 reading order; it is not implemented by M2. Keep the legacy parser until canonical/validation/chunk stages and migration tests are complete.

## Increment-one verification

Executed with the existing Python 3.12 environment outside the sandbox:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

All **17 tests passed**: 11 new contract/PDF tests and six existing section-detection tests. New fixtures cover columns with intentionally reversed source insertion order, style and block/line preservation, byte-identical replay for identical input, rotated multi-page documents, scans requiring OCR, blank pages, byte/page/character/object limits, corrupt and encrypted inputs, and cropped-page/table-rule geometry. This establishes the first increment's tested behavior, not semantic extraction accuracy or broad-layout production readiness.
