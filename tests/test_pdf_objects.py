"""Synthetic fixtures: no candidate files, models, network or credentials."""

from dataclasses import asdict
import json
import unittest

try:
    import fitz
except ImportError:
    fitz = None

from app.parsing.models import Box, Evidence, FieldValue, SourceRef, ValueStatus
from app.parsing.pdf_objects import ExtractionConfig, PDFExtractionError, extract_pdf


class ContractTests(unittest.TestCase):
    def test_known_values_require_provenance(self):
        with self.assertRaises(ValueError):
            FieldValue("Python", "Python", ValueStatus.KNOWN, (), Evidence(1, ("alias/1",), ("exact",)))

    def test_unknown_retains_raw_evidence_but_cannot_claim_normalization(self):
        source = SourceRef("document", 0, "span", Box(0, 0, 10, 10))
        evidence = Evidence(0, ("ambiguous/1",), ("Multiple interpretations",))
        value = FieldValue("MS", None, ValueStatus.UNKNOWN, (source,), evidence)
        self.assertEqual(value.raw, "MS")
        with self.assertRaises(ValueError):
            FieldValue("MS", "Master of Science", ValueStatus.UNKNOWN, (source,), evidence)

    def test_invalid_geometry_and_confidence_rejected(self):
        for coords in ((2, 0, 1, 1), (0, float("nan"), 1, 1)):
            with self.assertRaises(ValueError):
                Box(*coords)
        with self.assertRaises(ValueError):
            Evidence(1.1, ("rule",), ("reason",))
        with self.assertRaises(ValueError):
            SourceRef("doc", 0, "span", Box(0, 0, 1, 1), 2, 1)


@unittest.skipIf(fitz is None, "PyMuPDF required for PDF integration fixtures")
class PDFObjectTests(unittest.TestCase):
    def pdf(self, build):
        with fitz.open() as document:
            build(document)
            return document.tobytes()

    def test_columns_keep_source_order_geometry_styles_and_character_evidence(self):
        def build(document):
            page = document.new_page()
            # Deliberately draw the right column first: extraction must NOT sort it.
            page.insert_text((330, 90), "Right column", fontsize=12, fontname="hebo")
            page.insert_text((40, 60), "Left column", fontsize=10)
            page.draw_line((300, 30), (300, 400))
        result = extract_pdf(self.pdf(build))
        page = result.pages[0]
        self.assertEqual([span.text for span in page.spans], ["Right column", "Left column"])
        self.assertGreater(page.spans[0].source.box.x0, page.spans[1].source.box.x0)
        self.assertNotEqual(page.spans[0].style.font, page.spans[1].style.font)
        self.assertEqual(page.spans[0].characters[0].text, "R")
        self.assertEqual(page.spans[0].source.document_id, result.document_id)
        self.assertEqual(page.blocks[0].lines[0].span_ids, (page.spans[0].source.object_id,))
        self.assertIsNotNone(page.spans[0].style.ascender)
        self.assertEqual(len(page.spans[0].quad), 4)
        self.assertTrue(page.drawings[0].segments)

    def test_same_bytes_produce_identical_serialized_ir(self):
        def build(document):
            document.new_page().insert_text((40, 60), "Sample")
        data = self.pdf(build)
        first = json.dumps(asdict(extract_pdf(data)), sort_keys=True)
        second = json.dumps(asdict(extract_pdf(data)), sort_keys=True)
        self.assertEqual(first, second)

    def test_rotated_multipage_pdf_keeps_transforms_and_unique_sources(self):
        def build(document):
            page = document.new_page(width=400, height=600)
            page.insert_text((50, 70), "First")
            page.set_rotation(90)
            document.new_page().insert_text((50, 70), "Second")
        pages = extract_pdf(self.pdf(build)).pages
        self.assertEqual(pages[0].rotation, 90)
        self.assertEqual(pages[0].bounds, Box(0, 0, 400, 600))
        self.assertNotEqual(pages[0].rotation_matrix, pages[1].rotation_matrix)
        self.assertNotEqual(pages[0].spans[0].source.object_id, pages[1].spans[0].source.object_id)

    def test_image_only_page_requests_ocr_without_inventing_text(self):
        def build(document):
            page = document.new_page()
            # Tiny deterministic PPM image, no image-generation dependencies.
            image = b"P6\n2 2\n255\n" + b"\x00\x00\x00" * 4
            page.insert_image(fitz.Rect(20, 20, 400, 600), stream=image)
        result = extract_pdf(self.pdf(build))
        self.assertEqual(result.pages[0].text_status, "NEEDS_OCR")
        self.assertEqual(result.pages[0].spans, ())
        self.assertEqual(len(result.pages[0].images), 1)
        self.assertIn("NEEDS_OCR", [issue.code for issue in result.issues])

    def test_blank_page_is_not_claimed_as_a_successful_parse(self):
        result = extract_pdf(self.pdf(lambda document: document.new_page()))
        self.assertEqual(result.pages[0].text_status, "EMPTY_OR_OUTLINED")

    def test_limits_and_invalid_pdf_fail_explicitly(self):
        def build(document):
            document.new_page().insert_text((40, 60), "Long text")
            document.new_page()
        data = self.pdf(build)
        for config in (ExtractionConfig(max_bytes=10), ExtractionConfig(max_pages=1),
                       ExtractionConfig(max_characters=1), ExtractionConfig(max_objects=1)):
            with self.assertRaises(PDFExtractionError):
                extract_pdf(data, config)
        with self.assertRaises(PDFExtractionError):
            extract_pdf(b"not a pdf")

    def test_table_rules_and_cropped_page_geometry_are_preserved(self):
        def build(document):
            page = document.new_page(width=600, height=800)
            page.insert_text((100, 150), "Cell")
            page.draw_rect(fitz.Rect(80, 120, 300, 200))
            page.set_cropbox(fitz.Rect(50, 100, 550, 700))
        page = extract_pdf(self.pdf(build)).pages[0]
        self.assertEqual(page.bounds, Box(0, 0, 500, 600))
        self.assertEqual(page.crop_box, Box(50, 100, 550, 700))
        self.assertAlmostEqual(page.spans[0].characters[0].origin.x, 50)
        self.assertAlmostEqual(page.spans[0].characters[0].origin.y, 50)
        self.assertEqual(len(page.drawings[0].segments), 4)

    def test_encrypted_input_is_rejected_without_requesting_credentials(self):
        with fitz.open() as document:
            document.new_page()
            data = document.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256,
                                    owner_pw="synthetic-owner", user_pw="synthetic-user")
        with self.assertRaisesRegex(PDFExtractionError, "unencrypted"):
            extract_pdf(data)


if __name__ == "__main__":
    unittest.main()
