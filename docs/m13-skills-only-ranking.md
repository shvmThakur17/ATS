# M13 — skills-only ranking

Business rule: candidates are ranked **only** by how well their resume evidences the skills in the JD.

## Reference snapshot
`docs/m13_reference_v1/` is a copy of M13 as it was before this change (package, tests, inspect script), including the
blended `evidence_index`. It is not imported by anything.

## Per-skill status (unchanged; every match keeps its source and provenance)
| Status | Meaning | Counts as covered |
|---|---|---|
| EXPLICIT_SKILL | in `skills` or a project's technologies | yes |
| EXPLICIT_TEXT_EVIDENCE | the term is stated in summary / roles / descriptions / a retrieved chunk | yes |
| SEMANTIC_MATCH | related wording only (e.g. "Web APIs" for "REST APIs"); the term is not stated | no (tie-break only) |
| NOT_FOUND / UNKNOWN | no evidence / nothing to judge by | no |

A shorter skill is not confirmed by a longer one (`SQL` by `SQL Server`, `.NET` by `.NET Core`).

## Formula
```
key = ( |covered required| / |required|,
        |covered preferred| / |preferred|,
        |semantic required| / |required|,
        |semantic preferred| / |preferred| )
```
Candidates are ordered by `key`, compared lexicographically, larger first. A level absent from the JD contributes 0.0
for every candidate. There are no weights: required coverage always dominates, preferred coverage only separates
candidates with identical required coverage, and semantic evidence only separates candidates identical on both explicit
coverages. Equal keys share a rank (1, 1, 3); `document_id` only orders the listing.

Experience, education, responsibilities, retrieval similarity and completeness are reported for information
(`experience`, `education`, `retrieval`, `warnings`) and never enter the key.
