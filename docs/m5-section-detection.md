# M5: section detection

M5 consumes `LogicalDocumentIR` and resolved text keyed by logical block ID.
M4 deliberately stores references rather than text, so `resolve_logical_text`
resolves existing M1 span references. It preserves line breaks and does not run
extraction, layout, grouping, or reading-order analysis again.

```python
from app.parsing.sections import detect_sections, resolve_logical_text

text = resolve_logical_text(logical_document, source_document)
sections = detect_sections(logical_document, text)
```

The output contains one `SectionBlock` for every M4 logical block, in M4 order.
Each classification references the logical block and its active heading. Retain
the M4 input to resolve original source IDs, geometry, provenance, and upstream
issues. Output contains no resume text.

## Rules

- Whole-heading aliases live in `section_rules.py`; substring matches are never
  used. Numbering and a trailing colon are permitted.
- An alias needs M4 heading-like evidence or following content in the same
  vertical flow. Short following item groups provide additional evidence for
  skills, languages, and interests.
- Content inherits a recognized section while progressing downward in the same
  horizontal extent. A new heading replaces that context.
- An unknown heading, column transition, blank block, or empty page clears the
  context. Unsupported content remains `UNKNOWN`.
- The next consecutive page can inherit the section when its first content block
  aligns with the previous block. This carries a lower evidence score.

Scores are heuristic strengths, not calibrated probabilities. M5 makes no network
calls and uses no model, embedding, or external service.

## Limits

M5 does not split M4 blocks. If an upstream block combines a heading and unrelated
body text, exact heading recognition may fail; the result remains conservative.
Cross-page continuation uses geometry and prior heading context, so it cannot
prove semantic continuation. Complex indentation or unusually narrow headings may
also reset context. The small alias list is English-only; unfamiliar headings
remain unknown until explicitly added. M6 and later stages are not implemented.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/parsing/sections -v
.\.venv\Scripts\python.exe -m pytest tests/parsing/logical_blocks tests/parsing/reading_order tests/parsing/layout tests/test_pdf_objects.py -v
```
