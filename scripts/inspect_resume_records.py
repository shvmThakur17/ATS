"""Run M1-M7 on a local PDF and print only the M7 records."""

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


def run_stage(name, function, *args):
    try:
        return function(*args)
    except Exception as error:
        raise RuntimeError(f"{name} failed: {error}") from error


def heading(title):
    print(f"\n{title}\n{'-' * len(title)}")


def show(label, value):
    print(f"{label}: {value if value is not None else '-'}")


def show_record(prefix, index, fields, record):
    print(f"\n{prefix} {index}:")
    for label, value in fields:
        show(label, value)
    print(f"Raw Text: {record.raw_text!r}")
    print(f"Logical Blocks: {', '.join(record.logical_block_ids)}")


def print_records(records):
    contact = records.contact
    heading("CONTACT")
    for label, value in (("Name", contact.name), ("Email", contact.email), ("Phone", contact.phone),
                         ("Location", contact.location), ("LinkedIn", contact.linkedin),
                         ("GitHub", contact.github), ("Website", contact.website)):
        show(label, value)

    heading("SUMMARY")
    if records.summary:
        print(f"Text:\n{records.summary.text}")
        print(f"\nRaw Text:\n{records.summary.raw_text!r}")
        print(f"\nLogical Blocks: {', '.join(records.summary.logical_block_ids)}")
        print("\nProvenance:")
        for ref in records.summary.entities:
            print(f"  Entity: {ref.entity_id} ({ref.entity_type.value}) section={ref.section_type.value} "
                  f"page={ref.page_number} blocks={', '.join(ref.source_block_ids)} "
                  f"offsets=[{ref.text_start}, {ref.text_end}) score={ref.evidence.score} "
                  f"rules={', '.join(ref.evidence.rules)}")
            for source in ref.sources:
                print(f"    Source: {source.object_id} page={source.page} method={source.method}")
    else:
        print("-")

    heading("WORK EXPERIENCE")
    for i, r in enumerate(records.work_experience, 1):
        show_record("Record", i, (("Company", r.company), ("Role", r.role),
                                  ("Date Range", r.date_range), ("Description", r.description)), r)

    heading("EDUCATION")
    for i, r in enumerate(records.education, 1):
        show_record("Record", i, (("Institution", r.institution), ("Degree", r.degree),
                                  ("Field", r.field_of_study), ("Date Range", r.date_range)), r)

    heading("PROJECTS")
    for i, r in enumerate(records.projects, 1):
        show_record("Project", i, (("Name", r.name), ("Description", r.description),
                                   ("Technology", ", ".join(r.technology) or None)), r)

    heading("CERTIFICATIONS")
    for i, r in enumerate(records.certifications, 1):
        show_record("Certification", i, (("Name", r.certification), ("Issuer", r.issuing_organization),
                                         ("Credential ID", r.credential_id),
                                         ("Credential URL", r.credential_url)), r)

    heading("AWARDS")
    for i, r in enumerate(records.awards, 1):
        show_record("Award", i, (("Text", r.text),), r)

    for title, record, values in (("SKILLS", records.skills, records.skills and records.skills.skills),
                                  ("LANGUAGES", records.languages, records.languages and records.languages.languages),
                                  ("INTERESTS", records.interests, records.interests and records.interests.interests)):
        heading(title)
        for value in values or ():
            print(f"- {value}")
        if record:
            print(f"Logical Blocks: {', '.join(record.logical_block_ids)}")

    heading("UNASSIGNED M6 ENTITIES")
    for ref in records.unassigned:
        print(f"- {ref.entity_type.value} [{ref.section_type.value}] {ref.logical_block_id}: {ref.raw_value!r}")


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

    print_records(records)
    print("\nSUMMARY\n=======")
    show("Pages", len(document.pages))
    show("Logical blocks", sum(len(page.logical_blocks) for page in logical.pages))
    show("M6 entities", len(entities.entities))
    show("Work experience count", len(records.work_experience))
    show("Education count", len(records.education))
    show("Project count", len(records.projects))
    show("Certification count", len(records.certifications))
    show("Award count", len(records.awards))
    show("Skill count", len(records.skills.skills) if records.skills else 0)
    show("Language count", len(records.languages.languages) if records.languages else 0)
    show("Interest count", len(records.interests.interests) if records.interests else 0)
    show("Unassigned M6 entities", len(records.unassigned))


if __name__ == "__main__":
    main()
