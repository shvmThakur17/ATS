import unittest

import fitz

from app.parsing.pdf_objects import extract_pdf
from app.parsing.layout import analyze_layout, format_diagnostics


class SyntheticPDFLayoutTests(unittest.TestCase):
    def test_real_m1_to_m2_two_columns_replay_and_private_text(self):
        with fitz.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            for x in (340, 50):
                for y in (140, 210, 280):
                    page.insert_text((x, y), "PRIVATE SYNTHETIC TEXT", fontsize=10)
            data = pdf.tobytes()
        source = extract_pdf(data)
        layout = analyze_layout(source)
        self.assertEqual(len(layout.pages[0].regions), 2)
        self.assertEqual(layout.to_json(), analyze_layout(extract_pdf(data)).to_json())
        self.assertNotIn("PRIVATE SYNTHETIC TEXT", layout.to_json())
        self.assertNotIn("PRIVATE SYNTHETIC TEXT", format_diagnostics(layout))

    def test_rotated_cropped_pdf_stays_in_m1_coordinate_space(self):
        with fitz.open() as pdf:
            page = pdf.new_page(width=700, height=900)
            for x in (90, 400):
                for y in (240, 310, 380):
                    page.insert_text((x, y), "Synthetic line", fontsize=10)
            page.set_cropbox(fitz.Rect(50, 100, 650, 900))
            page.set_rotation(90)
            data = pdf.tobytes()
        source = extract_pdf(data)
        result = analyze_layout(source)
        self.assertEqual(result.pages[0].bounds, source.pages[0].bounds)
        self.assertEqual(len(result.pages[0].regions), 2)

    def test_scanned_pdf_preserves_image_refs_without_inventing_regions(self):
        with fitz.open() as pdf:
            page = pdf.new_page()
            image = b"P6\n2 2\n255\n" + b"\x00\x00\x00" * 4
            page.insert_image(fitz.Rect(20, 20, 400, 600), stream=image)
            data = pdf.tobytes()
        source = extract_pdf(data)
        result = analyze_layout(source)
        self.assertEqual(result.pages[0].regions, ())
        self.assertEqual(result.pages[0].image_ids, (source.pages[0].images[0].source.object_id,))
        self.assertEqual(result.source_issues, source.issues)
