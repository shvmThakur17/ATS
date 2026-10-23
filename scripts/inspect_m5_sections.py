"""Inspect M5 section detection for a real resume."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from app.parsing.pdf_objects import extract_pdf
from app.parsing.layout import analyze_layout
from app.parsing.reading_order import analyze_reading_order
from app.parsing.logical_blocks import analyze_logical_blocks
from app.parsing.sections import (
    detect_sections,
    resolve_logical_text,
)


def get_section_for_block(sections, logical_block_id):
    """Return the M5 section assignment for a logical block."""
    for section in sections.blocks:
        if section.logical_block_id == logical_block_id:
            return section

    return None


def print_section_evidence(section):
    """Print M5 evidence information safely."""
    if not section or not section.evidence:
        return

    print(f"Evidence score: {section.evidence.score}")

    if section.evidence.rules:
        print(
            "Evidence rules: "
            + ", ".join(section.evidence.rules)
        )

    if section.evidence.reasons:
        print("Evidence reasons:")

        for reason in section.evidence.reasons:
            print(f"  - {reason}")


def main():
    pdf_path = PROJECT_ROOT / "uploads" / "Kukoo.pdf"

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    print("=" * 80)
    print("M5 SECTION DETECTION - REAL RESUME")
    print("=" * 80)
    print(f"File: {pdf_path}")
    print()

    # ------------------------------------------------------------------
    # M1 - PDF Object Extraction
    # ------------------------------------------------------------------
    print("Running M1...")
    document = extract_pdf(pdf_path)

    # ------------------------------------------------------------------
    # M2 - Layout Analysis
    # ------------------------------------------------------------------
    print("Running M2...")
    layout = analyze_layout(document)

    # ------------------------------------------------------------------
    # M3 - Reading Order
    # ------------------------------------------------------------------
    print("Running M3...")
    order = analyze_reading_order(layout)

    # ------------------------------------------------------------------
    # M4 - Logical Blocks
    # ------------------------------------------------------------------
    print("Running M4...")
    logical = analyze_logical_blocks(
        document,
        layout,
        order,
    )

    # ------------------------------------------------------------------
    # M5 - Resolve logical text
    # ------------------------------------------------------------------
    print("Running M5 text resolution...")
    text = resolve_logical_text(
        logical,
        document,
    )

    # ------------------------------------------------------------------
    # M5 - Section Detection
    # ------------------------------------------------------------------
    print("Running M5 section detection...")
    sections = detect_sections(
        logical,
        text,
    )

    print()
    print("=" * 80)
    print("M5 SECTION ASSIGNMENTS")
    print("=" * 80)

    total_blocks = 0

    for page in logical.pages:
        print()
        print(f"PAGE {page.page_number + 1}")
        print("-" * 80)

        for block in page.logical_blocks:
            total_blocks += 1

            logical_block_id = block.logical_block_id

            section = get_section_for_block(
                sections,
                logical_block_id,
            )

            if section:
                section_name = section.section_type.value
                confidence = (
                    section.evidence.score
                    if section.evidence
                    else 0
                )
            else:
                section_name = "NO_SECTION_ASSIGNMENT"
                confidence = 0

            block_text = text.get(
                logical_block_id,
                "",
            ).strip()

            print()
            print(f"Block: {logical_block_id}")
            print(f"Section: {section_name}")
            print(f"Confidence: {confidence}")
            print(f"Text: {block_text}")

            print_section_evidence(section)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("M5 SUMMARY")
    print("=" * 80)

    print(f"Pages: {len(logical.pages)}")
    print(f"Logical blocks: {total_blocks}")
    print(f"M5 assignments: {len(sections.blocks)}")

    section_counts = {}

    for section in sections.blocks:
        section_name = section.section_type.value

        section_counts[section_name] = (
            section_counts.get(section_name, 0) + 1
        )

    print()
    print("Blocks by section:")

    for section_name, count in sorted(
        section_counts.items()
    ):
        print(f"  {section_name}: {count}")

    # ------------------------------------------------------------------
    # Important blocks for debugging
    # ------------------------------------------------------------------
    important_block_ids = {
        "p0/lb0",
        "p0/lb1",
        "p0/lb4",
        "p0/lb12",
        "p0/lb13",
        "p0/lb16",
        "p0/lb17",
        "p0/lb18",
        "p0/lb19",
        "p0/lb20",
        "p0/lb21",
        "p0/lb23",
        "p0/lb30",
        "p0/lb31",
        "p0/lb32",
        "p0/lb33",
        "p0/lb34",
        "p0/lb35",
        "p0/lb36",
        "p0/lb37",
        "p0/lb38",
    }

    print()
    print("=" * 80)
    print("IMPORTANT BLOCKS")
    print("=" * 80)

    for page in logical.pages:
        for block in page.logical_blocks:
            if block.logical_block_id not in important_block_ids:
                continue

            section = get_section_for_block(
                sections,
                block.logical_block_id,
            )

            if section:
                section_name = section.section_type.value

                confidence = (
                    section.evidence.score
                    if section.evidence
                    else 0
                )
            else:
                section_name = "NO_SECTION_ASSIGNMENT"
                confidence = 0

            block_text = text.get(
                block.logical_block_id,
                "",
            ).strip()

            print()
            print(
                f"{block.logical_block_id}"
            )
            print(
                f"  Section: {section_name}"
            )
            print(
                f"  Confidence: {confidence}"
            )
            print(
                f"  Text: {block_text}"
            )

    print()
    print("=" * 80)
    print("M5 INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()