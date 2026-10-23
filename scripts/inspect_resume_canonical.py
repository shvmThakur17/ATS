"""Run M1-M9 on a local PDF and print the canonical resume."""

from pathlib import Path
import sys

# Allow the documented script command to import app from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.pdf_objects import extract_pdf
from app.parsing.layout import analyze_layout
from app.parsing.reading_order import analyze_reading_order
from app.parsing.logical_blocks import analyze_logical_blocks
from app.parsing.sections import detect_sections, resolve_logical_text
from app.parsing.entities import extract_entities
from app.parsing.records import build_records
from app.parsing.normalization import normalize_records
from app.parsing.canonical import build_canonical_resume


def run_stage(name, function, *args):
    try:
        return function(*args)
    except Exception as error:
        raise RuntimeError(f"{name} failed: {error}") from error


def heading(title):
    print(f"\n{title}\n{'-' * len(title)}")


def show(label, value):
    print(f"{label}: {value if value is not None else '-'}")


def describe(provenance):
    """One compact line: raw value, rules, and the M6 entities it came from."""
    refs = "; ".join(f"{r.entity_id} block={r.logical_block_id} page={r.page_number} "
                     f"pdf={','.join(r.source_block_ids)}" for r in provenance.entities)
    rules = ", ".join(provenance.normalization_rules) or "-"
    return (f"raw={provenance.raw_value!r} status={provenance.normalization_status} rules=[{rules}]\n"
            f"      M6: {refs or '-'}")


def print_canonical(resume):
    print("CANONICAL RESUME\n================")
    show("\nSchema Version", resume.schema_version)

    heading("CANDIDATE")
    for field in ("name", "email", "phone", "location", "linkedin", "github", "website"):
        show(field.title(), getattr(resume.candidate, field))

    heading("SUMMARY")
    print(resume.summary.value if resume.summary else "-")

    heading("WORK EXPERIENCE")
    for i, w in enumerate(resume.work_experience, 1):
        print(f"\n{i}.")
        for label, value in (("Company", w.company), ("Role", w.role), ("Start", w.start_date),
                             ("End", w.end_date), ("Current", w.current), ("Description", w.description)):
            show(label, value)

    heading("EDUCATION")
    for i, e in enumerate(resume.education, 1):
        print(f"\n{i}.")
        for label, value in (("Institution", e.institution), ("Degree", e.degree),
                             ("Field of Study", e.field_of_study), ("Start", e.start_date), ("End", e.end_date)):
            show(label, value)

    heading("PROJECTS")
    for i, p in enumerate(resume.projects, 1):
        print(f"\n{i}.")
        show("Name", p.name)
        show("Description", p.description)
        show("Technologies", [t.value for t in p.technologies])

    heading("CERTIFICATIONS")
    for i, c in enumerate(resume.certifications, 1):
        print(f"\n{i}.")
        for label, value in (("Name", c.name), ("Issuer", c.issuing_organization),
                             ("Credential ID", c.credential_id), ("Credential URL", c.credential_url)):
            show(label, value)

    heading("AWARDS")
    for i, a in enumerate(resume.awards, 1):
        print(f"{i}. {a.value}")

    heading("SKILLS")
    for i, s in enumerate(resume.skills, 1):
        print(f"{i}. {s.value}" + (f"   (raw: {s.raw_value})" if s.raw_value != s.value else ""))

    heading("LANGUAGES")
    for i, x in enumerate(resume.languages, 1):
        print(f"{i}. {x.value}")

    heading("INTERESTS")
    for i, x in enumerate(resume.interests, 1):
        print(f"{i}. {x.value}")

    heading("METADATA")
    m = resume.metadata
    show("Source File", m.source_file_name)
    show("Pages", m.page_count)
    show("Logical Blocks", m.logical_block_count)
    show("M6 Entities", m.extracted_entity_count)
    show("Validation Status", m.validation_status.value)

    heading("VALIDATION")
    show("Status", resume.validation.status.value)
    for issue in resume.validation.issues:
        print(f"  {issue.severity.value} {issue.code}: {issue.message}")
    for ref in resume.validation.unresolved:
        print(f"  Unresolved: {ref.entity_type.value} {ref.entity_id} {ref.raw_value!r}")

    print("\nPROVENANCE INSPECTION\n=====================")
    print(f"\ncandidate.name\n    {describe(resume.candidate.provenance['name'])}")
    if resume.summary:
        print(f"\nsummary\n    {describe(resume.summary.provenance)}")
    if resume.skills:
        print(f"\nskills[{len(resume.skills) - 1}]\n    {describe(resume.skills[-1].provenance)}")
    if resume.work_experience:
        w = resume.work_experience[-1]
        print(f"\nwork_experience[{len(resume.work_experience) - 1}] (blocks {', '.join(w.logical_block_ids)})")
        for field, provenance in w.provenance.items():
            print(f"    {field}: {describe(provenance)}")
    if resume.projects:
        p = resume.projects[0]
        print(f"\nprojects[0] (blocks {', '.join(p.logical_block_ids)})")
        for field, provenance in p.provenance.items():
            print(f"    {field}: {describe(provenance)}")


def main():
    pdf_path = PROJECT_ROOT / "uploads" / "Kukoo.pdf"
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Input PDF does not exist: {pdf_path}")
    document = run_stage("M1 PDF Object Extraction", extract_pdf, pdf_path)
    layout = run_stage("M2 Layout Analysis", analyze_layout, document)
    order = run_stage("M3 Reading Order", analyze_reading_order, layout)
    logical = run_stage("M4 Logical Blocks", analyze_logical_blocks, document, layout, order)
    text = run_stage("M5 text reference resolution", resolve_logical_text, logical, document)
    sections = run_stage("M5 Section Detection", detect_sections, logical, text)
    entities = run_stage("M6 Entity Extraction", extract_entities, sections, logical, text)
    records = run_stage("M7 Record Construction", build_records, entities)
    normalized = run_stage("M8 Normalization + Validation", normalize_records, records)
    resume = run_stage("M9 Canonical Resume", build_canonical_resume, normalized, pdf_path.name,
                       len(document.pages), sum(len(page.logical_blocks) for page in logical.pages),
                       len(entities.entities))
    print_canonical(resume)
    if "--json" in sys.argv:
        print("\nJSON\n====")
        print(resume.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
