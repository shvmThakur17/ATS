# M2 — deterministic physical layout analysis

Implemented boundary: **M1 DocumentIR → M2 LayoutIR**. No M3 reading order, logical record construction, section labels, entities, normalization, embeddings or external services are implemented or called by M2. No dependencies were added.

## Usage

```python
from app.parsing.pdf_objects import extract_pdf
from app.parsing.layout import analyze_layout, format_diagnostics, LayoutConfig

source = extract_pdf("resume.pdf")
layout = analyze_layout(source, LayoutConfig())
serialized = layout.to_json()
print(format_diagnostics(layout))  # IDs, coordinates, counts and issue codes only
```

M2 itself performs no I/O. Keep `source` alongside `layout`: blocks, lines, spans, fonts, characters, image transforms and drawing segments resolve through original M1 IDs. M2 does not copy resume text. M1's schema/stage/engine/config versions and source issues are retained in LayoutIR. M2's schema, stage version, and configuration fingerprint are explicit.

## Model and interpretation

- `LayoutIR.pages`: independently analyzed pages, in page-number storage order.
- `LayoutPage.block_features`: original block source refs plus extents, center, area and page-relative dimensions.
- `LayoutPage.regions`: physical candidate regions with bounds, original block/line IDs, evidence, and original `SourceRef` instances. Region bounds describe observed content, not the entire empty column envelope.
- `LayoutPage.assignments`: each block has one or more region IDs. `spans_regions` means its native lines participate in multiple regions. Full-width content in a dedicated region also exposes its physical kind.
- `LayoutPage.relationships`: local measured signals. Directions describe `first_id` relative to `second_id`; IDs are canonically sorted, not ordered for reading. Signals include overlap, containment, relative position, proximity, horizontal/vertical overlap, edge/center alignment and similar widths.
- `LayoutPage.table_candidates`: weak grid/bordered-panel or repeated-row-alignment signals; `possible_table=True` is not proof of a table. Regions are not assigned resume semantics or converted to cells.
- Image/drawing IDs survive even when there are no text regions. M1 OCR-required issues remain visible through `source_issues`.

Kinds are constrained by `RegionKind`: `UNKNOWN`, `MAIN`, `SIDEBAR`, `HEADER`, `FOOTER`, `FULL_WIDTH`. MAIN denotes an unsplit or column-shaped physical region; it does not mean employment or any other section. HEADER and FOOTER are isolated page-margin candidates, not confirmed repeating document headers/footers. Confidence is an explicit configurable rule-strength score, not a calibrated probability.

## Algorithm

1. Validate source schema, IDs, page ownership and span references; apply M1's page/object/character limits. Custom M1 limits may be passed as `source_limits` and participate in M2's fingerprint. Byte limits remain M1's responsibility because M2 does not read PDF bytes.
2. Calculate block features. Use native line geometry as segmentation atoms so a large native block spanning several columns does not hide its internal layout. A block with no lines uses its own box. Retain rotated/vertical, degenerate or out-of-page atoms as UNKNOWN with diagnostics.
3. Isolate top/bottom candidates only when both page-relative margin position and separation from body content support the decision. Do not compare resume meaning or assume that every top line is a header.
4. Find connected vertical row bands and stable local horizontal profiles. Repeated changes in profile establish independent local zones, allowing unequal columns, different column counts, and full-width interruptions on the same page.
5. Sweep repeated whitespace intervals to find supported gutters; supplement these with globally empty gutters between repeated objects with overlapping vertical extents for staggered baselines. No fixed X coordinate, fixed column count or equal-width assumption is used.
6. Preserve gutter-crossing bands as FULL_WIDTH only when observed width justifies it; otherwise emit UNKNOWN with a diagnostic. Re-analyze the surrounding zones independently. Narrow columns are SIDEBAR candidates only relative to their peers and content envelope.
7. Detect local block relationships with a uniform spatial grid. Only geometrically nearby pairs are emitted; alignment across arbitrarily distant blocks is deliberately not exhaustively enumerated. Overlapping/contained pairs remain explicit and no source block is removed.
8. Detect weak table signals from connected orthogonal drawing boundaries and repeated aligned rows. Leave both content and region assignments intact.
9. Serialize immutable results deterministically. Source-ID/hash sorting is a storage convention; no final reading-order list or graph is produced.

All heuristic thresholds, scores and resource ceilings are centralized in `LayoutConfig`. Region/table IDs use SHA-256 of canonical source membership, scoped to the LayoutIR document. Work, grid entry, spatial comparison, candidate-pair and recursion limits fail explicitly with `LayoutLimitError`; output is never silently truncated. Sorting/sweeps and the local grid avoid mandatory all-block pair comparisons. Dense/adversarial overlap and many drawing components can still approach quadratic work, bounded by the configured budgets. The M1 recommendation to decode untrusted PDFs in a resource-limited worker still applies.

## Files

Created application files:

- `app/parsing/layout/__init__.py` — public API.
- `app/parsing/layout/models.py` — immutable LayoutIR, configuration, physical kinds and limits.
- `app/parsing/layout/geometry.py` — reusable rectangle helpers.
- `app/parsing/layout/block_features.py` — source-preserving feature calculation.
- `app/parsing/layout/relationship_detector.py` — bounded spatial index and physical relations.
- `app/parsing/layout/region_detector.py` — local physical segmentation.
- `app/parsing/layout/table_detector.py` — possible-table signals.
- `app/parsing/layout/layout_analyzer.py` — validation, composition and private diagnostics.

Created test files: `tests/parsing/__init__.py`, `tests/parsing/layout/__init__.py`, `fixtures.py`, `test_geometry.py`, `test_layout.py`, and `test_pdf_layout.py` (last four under `tests/parsing/layout/`). Added this document and updated the original architecture document's milestone status. M1, shared/canonical models, legacy services and the upload endpoint were not changed.

## Verification

Run from the repository root:

```powershell
# Complete discoverable offline suite: existing tests and M2.
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v

# M2 only, with package context for fixture imports.
& .\.venv\Scripts\python.exe -m unittest discover -s tests/parsing/layout -t tests -p 'test_*.py' -v
```

Full suite: **48 tests passed** (17 existing + 31 M2). The separately executed M2-only suite also passed **31 tests**. Coverage includes single/two/four columns, unequal widths, sidebar on either side, headers/footers, mixed layouts with and without a separator, staggered baselines, overlaps/containment/alignment, differing sizes, scale invariance, multi-page independence, source collection permutation, stable serialization, provenance preservation, spanning native blocks, table hints, blank/rotated-page handling, invalid configuration/limits and duplicate IDs. Synthetic PDF integration exercises the actual M1→M2 boundary, including cropped/rotated pages, no-content diagnostics and scan preservation.

The four existing `*_test.py` files are manual model/service experiments with import-time external effects, not discoverable offline unit tests. They were inspected but not executed: M2 has no Ollama/Pinecone/model-network dependency. No real candidate resumes or credentials were used.

## Known limitations and M3 handoff

- Region inference is geometric and heuristic. Sparse, overlapping, staggered, irregular or decorative layouts can remain UNKNOWN or need threshold tuning; 1,000-layout accuracy is not established by these fixtures.
- Rotated page metadata is supported through M1's unrotated coordinate frame. Non-horizontal individual text lines are preserved as UNKNOWN rather than forcibly segmented on horizontal assumptions.
- No OCR: image-only pages retain image references and OCR-needed issues without fabricated text regions.
- Column alignment can resemble a table, and a bordered panel can resemble a grid. Table candidates intentionally remain tentative; there is no cell reconstruction.
- Header/footer detection uses isolated page-margin geometry only. Repetition across pages is not yet used, and layouts are analyzed independently.
- Stable source ID ordering is not reading order. M3 must consume regions, memberships, original line geometry and local relationship signals to build and validate its reading-order graph, resolve ambiguous regions, and handle cross-page continuation. That work is not part of M2.
