"""Run M1-M8 on a local PDF and print only the M8 normalization result."""

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


def run_stage(name, function, *args):
    try:
        return function(*args)
    except Exception as error:
        raise RuntimeError(f"{name} failed: {error}") from error


def heading(title):
    print(f"\n{title}\n{'-' * len(title)}")


def issues_for(result, record_type, index, field):
    return [i for i in result.issues
            if (i.record_type, i.record_index, i.field) == (record_type, index, field)]


def show_value(result, label, value, record_type, index, field):
    if value is None:
        print(f"{label}: -")
        return
    print(f"{label}:")
    print(f"  Raw: {value.raw_value!r}")
    print(f"  Normalized: {value.normalized_value!r}")
    print(f"  Status: {value.status.value}")
    print(f"  Rules: {', '.join(value.rules) or '-'}")
    found = [f"{i.severity.value} {i.code}" for i in issues_for(result, record_type, index, field)]
    print(f"  Validation: {'; '.join(found) or 'ok'}")
    print(f"  Source: {', '.join(f'{r.entity_id} ({r.logical_block_id}, page {r.page_number})' for r in value.sources) or '-'}")


def show_date(result, label, date, record_type, index):
    if date is None:
        print(f"{label}: -")
        return
    print(f"{label}:")
    print(f"  Raw: {date.raw_value!r}")
    print(f"  Start Year: {date.start_year}")
    print(f"  Start Month: {date.start_month}")
    print(f"  End Year: {date.end_year}")
    print(f"  End Month: {date.end_month}")
    print(f"  Current: {date.is_current}")
    print(f"  Normalized: {date.normalized_value!r}")
    print(f"  Status: {date.status.value}")
    print(f"  Rules: {', '.join(date.rules) or '-'}")
    found = [f"{i.severity.value} {i.code}" for i in issues_for(result, record_type, index, "date_range")]
    print(f"  Validation: {'; '.join(found) or 'ok'}")
    print(f"  Source: {', '.join(f'{r.entity_id} ({r.logical_block_id}, page {r.page_number})' for r in date.sources) or '-'}")


def show_list(result, title, record_type, record, values):
    heading(title)
    for value in values if record else ():
        show_value(result, "Item", value, record_type, None, record_type)
    if record:
        print(f"Logical Blocks: {', '.join(record.logical_block_ids)}")


def print_resume(result):
    resume = result.resume
    heading("SUMMARY")
    if resume.summary:
        show_value(result, "Text", resume.summary.text, "summary", None, "text")

    heading("CONTACT")
    for field in ("name", "email", "phone", "location", "linkedin", "github", "website"):
        show_value(result, field.title(), getattr(resume.contact, field), "contact", None, field)

    heading("WORK EXPERIENCE")
    for i, record in enumerate(resume.work_experience):
        print(f"\nRecord {i + 1} (blocks: {', '.join(record.logical_block_ids)}):")
        show_value(result, "Company", record.company, "work_experience", i, "company")
        show_value(result, "Role", record.role, "work_experience", i, "role")
        show_date(result, "Date Range", record.date_range, "work_experience", i)
        show_value(result, "Description", record.description, "work_experience", i, "description")

    heading("EDUCATION")
    for i, record in enumerate(resume.education):
        print(f"\nRecord {i + 1} (blocks: {', '.join(record.logical_block_ids)}):")
        show_value(result, "Institution", record.institution, "education", i, "institution")
        show_value(result, "Degree", record.degree, "education", i, "degree")
        show_value(result, "Field", record.field_of_study, "education", i, "field_of_study")
        show_date(result, "Date Range", record.date_range, "education", i)

    heading("PROJECTS")
    for i, record in enumerate(resume.projects):
        print(f"\nProject {i + 1} (blocks: {', '.join(record.logical_block_ids)}):")
        show_value(result, "Name", record.name, "projects", i, "name")
        show_value(result, "Description", record.description, "projects", i, "description")
        for tech in record.technology:
            show_value(result, "Technology", tech, "projects", i, "technology")

    heading("CERTIFICATIONS")
    for i, record in enumerate(resume.certifications):
        print(f"\nCertification {i + 1}:")
        for field in ("certification", "issuing_organization", "credential_id", "credential_url"):
            show_value(result, field, getattr(record, field), "certifications", i, field)

    heading("AWARDS")
    for i, record in enumerate(resume.awards):
        print(f"\nAward {i + 1}:")
        show_value(result, "Text", record.text, "awards", i, "text")

    show_list(result, "SKILLS", "skills", resume.skills, resume.skills.skills if resume.skills else ())
    show_list(result, "LANGUAGES", "languages", resume.languages,
              resume.languages.languages if resume.languages else ())
    show_list(result, "INTERESTS", "interests", resume.interests,
              resume.interests.interests if resume.interests else ())

    heading("UNRESOLVED M6 ENTITIES (passed through unchanged)")
    for ref in resume.unassigned:
        print(f"- {ref.entity_type.value} [{ref.section_type.value}] {ref.logical_block_id}: {ref.raw_value!r}")

    print("\nVALIDATION ISSUES\n=================")
    for issue in result.issues:
        where = f"{issue.record_type}[{issue.record_index}]" if issue.record_index is not None else issue.record_type
        print(f"\nSeverity: {issue.severity.value}\nCode: {issue.code}\nField: {where}.{issue.field}"
              f"\nMessage: {issue.message}")
    print(f"\nOverall status: {result.status.value}")


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
    result = run_stage("M8 Normalization + Validation", normalize_records, records)
    print_resume(result)
    counts = {}
    for issue in result.issues:
        counts[issue.severity.value] = counts.get(issue.severity.value, 0) + 1
    print("Issue counts: " + ", ".join(f"{k}={counts.get(k, 0)}" for k in ("ERROR", "WARNING", "INFO")))


if __name__ == "__main__":
    main()
