"""Stage one: preserve PDF geometry without assigning semantic reading order."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from .models import (
    SCHEMA_VERSION, Box, Character, DocumentIR, DrawingObject, ImageObject,
    Issue, PageIR, Point, SourceRef, TextBlock, TextLine, TextSpan, TextStyle,
)


STAGE_VERSION = "pdf-objects/1"


class PDFExtractionError(ValueError):
    """Sanitized failure without PDF content or filesystem details."""


@dataclass(frozen=True)
class ExtractionConfig:
    max_bytes: int = 25 * 1024 * 1024
    max_pages: int = 100
    max_objects: int = 200_000
    max_characters: int = 2_000_000

    def __post_init__(self):
        if min(self.max_bytes, self.max_pages, self.max_objects, self.max_characters) < 1:
            raise ValueError("Extraction limits must be positive")


def _box(value) -> Box:
    return Box(*(float(v) for v in value))


def _point(value) -> Point:
    return Point(float(value[0]), float(value[1]))


def extract_pdf(source: str | Path | bytes, config: ExtractionConfig | None = None) -> DocumentIR:
    """Extract immutable source objects. No OCR, normalization, sorting or flattening.

    Call in a bounded worker for untrusted input: library decoding itself cannot
    be CPU/memory-limited by the post-decode object checks here.
    """
    config = config or ExtractionConfig()
    if isinstance(source, bytes):
        data = source
    else:
        try:
            with Path(source).open("rb") as stream:
                data = stream.read(config.max_bytes + 1)
        except OSError:
            raise PDFExtractionError("PDF input could not be read") from None
    if not data or len(data) > config.max_bytes:
        raise PDFExtractionError("PDF is empty or exceeds byte limit")

    # Lazy dependency import keeps shared contracts usable without PDF libraries.
    import fitz

    document_id = sha256(data).hexdigest()
    fingerprint = sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    pages, issues = [], []
    object_count = character_count = 0
    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            if not document.is_pdf or document.needs_pass:
                raise PDFExtractionError("Input must be an unencrypted PDF")
            if not 0 < len(document) <= config.max_pages:
                raise PDFExtractionError("PDF page count is outside configured limits")
            for page_number, page in enumerate(document):
                # Exclude image payload decoding; get_image_info supplies geometry.
                flags = fitz.TEXTFLAGS_RAWDICT & ~fitz.TEXT_PRESERVE_IMAGES
                raw = page.get_text("rawdict", flags=flags, sort=False)
                bounds = _box((0, 0, raw["width"], raw["height"]))
                blocks, spans, images, drawings = [], [], [], []

                def ref(object_id, box):
                    return SourceRef(document_id, page_number, object_id, box)

                def count_object():
                    nonlocal object_count
                    object_count += 1
                    if object_count > config.max_objects:
                        raise PDFExtractionError("PDF exceeds object limit")

                for block_index, block in enumerate(raw["blocks"]):
                    if block["type"] != 0:
                        continue
                    count_object()
                    lines = []
                    for line_index, line in enumerate(block["lines"]):
                        count_object()
                        direction = _point(line["dir"])
                        span_ids = []
                        for span_index, span in enumerate(line["spans"]):
                            count_object()
                            character_count += len(span["chars"])
                            if character_count > config.max_characters:
                                raise PDFExtractionError("PDF exceeds character limit")
                            object_id = f"p{page_number}/b{block_index}/l{line_index}/s{span_index}"
                            span_ids.append(object_id)
                            source_ref = ref(object_id, _box(span["bbox"]))
                            characters = tuple(
                                Character(char["c"], _box(char["bbox"]), _point(char["origin"]))
                                for char in span["chars"]
                            )
                            quad = fitz.recover_quad(line["dir"], span)
                            spans.append(TextSpan(
                                source_ref, block_index, line_index, span_index, characters,
                                tuple(_point(p) for p in (quad.ul, quad.ur, quad.ll, quad.lr)),
                                TextStyle(span["font"], float(span["size"]), span["flags"],
                                          span["color"], span.get("alpha"),
                                          span.get("ascender"), span.get("descender")),
                                direction, line.get("wmode", 0),
                            ))
                            if any("\ufffd" in char.text for char in characters):
                                issues.append(Issue("UNMAPPED_GLYPH", "warning",
                                                    "Native text contains replacement characters", (source_ref,)))
                            if direction.x < 0.99 or abs(direction.y) > 0.01:
                                issues.append(Issue("NON_HORIZONTAL_TEXT", "info",
                                                    "Layout stage must use text direction and quads", (source_ref,)))
                            if (source_ref.box.x0 < bounds.x0 or source_ref.box.y0 < bounds.y0
                                    or source_ref.box.x1 > bounds.x1 or source_ref.box.y1 > bounds.y1):
                                issues.append(Issue("OUT_OF_PAGE_BOUNDS", "warning",
                                                    "Preserved source geometry extends beyond page bounds", (source_ref,)))
                        lines.append(TextLine(
                            ref(f"p{page_number}/b{block_index}/l{line_index}", _box(line["bbox"])),
                            tuple(span_ids), direction, line.get("wmode", 0),
                        ))
                    blocks.append(TextBlock(
                        ref(f"p{page_number}/b{block_index}", _box(block["bbox"])),
                        block.get("number", block_index), tuple(lines),
                    ))

                for index, item in enumerate(page.get_image_info()):
                    count_object()
                    images.append(ImageObject(
                        ref(f"p{page_number}/image{index}", _box(item["bbox"])),
                        item["width"], item["height"], tuple(float(v) for v in item["transform"]),
                    ))
                for index, drawing in enumerate(page.get_drawings()):
                    count_object()
                    segments = []
                    for item in drawing["items"]:
                        if item[0] == "l":
                            segments.append((_point(item[1]), _point(item[2])))
                        elif item[0] == "re":
                            rect = item[1]
                            corners = (rect.tl, rect.tr, rect.br, rect.bl)
                            segments.extend((_point(corners[i]), _point(corners[(i + 1) % 4])) for i in range(4))
                    drawings.append(DrawingObject(
                        ref(f"p{page_number}/drawing{index}", _box(drawing["rect"])),
                        drawing["type"], float(drawing.get("width") or 0), tuple(segments),
                    ))
                usable_text = any(span.text.strip().replace("\ufffd", "") for span in spans)
                status = "NATIVE_TEXT" if usable_text else "NEEDS_OCR" if images else "EMPTY_OR_OUTLINED"
                page_ref = ref(f"p{page_number}", bounds)
                if status != "NATIVE_TEXT":
                    issues.append(Issue(status, "warning",
                                        "Page has no usable native text; do not infer resume fields", (page_ref,)))
                elif images:
                    issues.append(Issue("RASTER_CONTENT_UNCLASSIFIED", "info",
                                        "Images may contain additional text; OCR eligibility is unresolved", (page_ref,)))
                pages.append(PageIR(
                    page_number, bounds, _box(page.mediabox), _box(page.cropbox), page.rotation,
                    tuple(page.transformation_matrix), tuple(page.rotation_matrix),
                    tuple(blocks), tuple(spans), tuple(images), tuple(drawings), status,
                ))
    except PDFExtractionError:
        raise
    except Exception:
        raise PDFExtractionError("PDF object extraction failed") from None
    return DocumentIR(document_id, SCHEMA_VERSION, STAGE_VERSION,
                      f"PyMuPDF/{fitz.VersionBind}", fingerprint, tuple(pages), tuple(issues))
