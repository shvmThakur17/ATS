"""Print the existing M1–M6 pipeline output for a local PDF."""

from collections import Counter
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
from app.parsing.entities import EntityType, extract_entities


def run_stage(name, function, *args):
    try:
        return function(*args)
    except Exception as error:
        raise RuntimeError(f"{name} failed: {error}") from error


def print_heading(title):
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


def print_sections(sections, text):
    print_heading("M5 SECTIONS")
    for index, section in enumerate(sections.blocks, 1):
        print(f"\nSection: {index}")
        print(f"Section Type: {section.section_type.value}")
        print(f"Logical Block ID: {section.logical_block_id}")
        print("Text:")
        print(text[section.logical_block_id])


def print_entity(entity):
    print(f"\nEntity Type: {entity.entity_type.value}")
    print(f"Value: {entity.raw_value}")
    print(f"Section: {entity.section_type.value}")
    print(f"Logical Block ID: {entity.logical_block_id}")
    print(f"Page: {entity.page_number + 1} (source index: {entity.page_number})")
    print(f"Confidence: {entity.evidence.score}")
    print(f"Evidence rules: {', '.join(entity.evidence.rules)}")
    print("Evidence:")
    for reason in entity.evidence.reasons:
        print(f"    {reason}")
    print(f"Source block IDs: {', '.join(entity.source_block_ids)}")
    print(f"Text offsets: [{entity.text_start}, {entity.text_end}) in resolved logical-block text")
    print("Source references:")
    for source in entity.sources:
        print(f"    Object: {source.object_id}; page index: {source.page}; method: {source.method}")
        print(f"    Document: {source.document_id}")
        print(f"    Box: {source.box}")
        if source.char_start is not None:
            print(f"    Character offsets: [{source.char_start}, {source.char_end})")


def print_entities(result):
    print_heading("M6 ENTITIES")
    groups = {}
    for entity in result.entities:
        # Unclassified entities remain visible even inside a known section.
        name = "UNKNOWN" if entity.entity_type == EntityType.UNKNOWN else entity.section_type.value
        groups.setdefault(name, []).append(entity)
    for name, entities in groups.items():
        if name == "UNKNOWN":
            continue
        print(f"\n[{name}]")
        for entity in entities:
            print_entity(entity)
    if "UNKNOWN" in groups:
        print("\n[UNKNOWN]")
        for entity in groups["UNKNOWN"]:
            print_entity(entity)


def main():
    relative_path = Path("uploads/Kukoo.pdf")
    pdf_path = PROJECT_ROOT / relative_path
    print("=" * 50)
    print("RESUME ENTITY INSPECTION")
    print(f"File: {relative_path.as_posix()}")
    print("=" * 50)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Input PDF does not exist: {pdf_path}")

    document = run_stage("M1 PDF Object Extraction", extract_pdf, pdf_path)
    layout = run_stage("M2 Layout Analysis", analyze_layout, document)
    order = run_stage("M3 Reading Order", analyze_reading_order, layout)
    logical = run_stage("M4 Logical Blocks", analyze_logical_blocks, document, layout, order)
    text = run_stage("M5 text reference resolution", resolve_logical_text, logical, document)
    sections = run_stage("M5 Section Detection", detect_sections, logical, text)
    entities = run_stage("M6 Entity Extraction", extract_entities, sections, logical, text)

    print_sections(sections, text)
    print_entities(entities)
    print_heading("SUMMARY")
    print(f"Pages: {len(document.pages)}")
    print(f"Logical blocks: {sum(len(page.logical_blocks) for page in logical.pages)}")
    print(f"Sections: {len(sections.blocks)} (M5 block assignments)")
    print(f"Distinct section types: {len({block.section_type for block in sections.blocks})}")
    print(f"Entities: {len(entities.entities)}")
    print("\nEntity counts by type:")
    counts = Counter(entity.entity_type for entity in entities.entities)
    for kind in EntityType:
        print(f"    {kind.value}: {counts[kind]}")


if __name__ == "__main__":
    main()
