"""M4: conservative logical grouping over M1/M2/M3; no resume semantics."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from statistics import median

from ..layout.geometry import union
from ..layout.models import LayoutIR
from ..models import DocumentIR, Evidence, Issue
from ..reading_order.models import ReadingOrderIR
from .models import (LOGICAL_BLOCKS_VERSION, LogicalBlock, LogicalDocumentIR,
                     LogicalPage, LogicalRelationship, RelationshipKind, StructuralType)

# Base PDF fonts can encode a bullet as a middle dot (U+00B7).
_BULLET = re.compile(r"^\s*[•·●▪◦\-*]\s+")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+")
_LABEL = re.compile(r"^\s*[^:\n]{1,60}:\s*$")
_DATE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}\b|\b(?:19|20)\d{2}\b|\bpresent\b", re.I)


@dataclass(frozen=True)
class LogicalBlockConfig:
    max_vertical_gap_page_ratio: float = 0.035
    heading_size_ratio: float = 1.15

    def __post_init__(self):
        if not 0 < self.max_vertical_gap_page_ratio <= 1 or self.heading_size_ratio < 1:
            raise ValueError("Invalid logical block configuration")


def _text_and_style(page):
    spans = {span.source.object_id: span for span in page.spans}
    result = {}
    for block in page.blocks:
        block_spans = [spans[span_id] for line in block.lines for span_id in line.span_ids if span_id in spans]
        text = "".join(span.text for span in block_spans).strip()
        sizes = [span.style.size for span in block_spans]
        result[block.source.object_id] = (text, sum(sizes) / len(sizes) if sizes else None,
                                          any(span.style.flags & 16 for span in block_spans))
    return result


def _kind(text):
    if _BULLET.match(text):
        return StructuralType.BULLET_LIST
    if _NUMBERED.match(text):
        return StructuralType.NUMBERED_LIST
    return StructuralType.UNKNOWN


def analyze_logical_blocks(document: DocumentIR, layout: LayoutIR, order: ReadingOrderIR,
                           config: LogicalBlockConfig | None = None) -> LogicalDocumentIR:
    """Group only adjacent M3 blocks when multiple structural signals agree."""
    if document.document_id != layout.document_id or document.document_id != order.document_id:
        raise ValueError("M1, M2 and M3 document identities must match")
    if layout.stage_version != "layout/1" or order.stage_version != "reading-order/1":
        raise ValueError("Unsupported M2 or M3 contract")
    config = config or LogicalBlockConfig()
    fingerprint = sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    m1_pages = {page.number: page for page in document.pages}
    m2_pages = {page.page_number: page for page in layout.pages}
    pages = []
    for order_page in sorted(order.pages, key=lambda page: page.page_number):
        page = m1_pages.get(order_page.page_number)
        layout_page = m2_pages.get(order_page.page_number)
        if page is None or layout_page is None:
            raise ValueError("M3 page must have corresponding M1 and M2 pages")
        data = _text_and_style(page)
        features = {feature.block_id: feature for feature in layout_page.block_features}
        membership = {assignment.block_id: assignment.region_ids for assignment in layout_page.assignments}
        ids = order_page.ordered_block_ids
        if set(ids) != set(features) or len(ids) != len(set(ids)):
            raise ValueError("M3 must contain each M2 block exactly once")
        # Native blocks may contain several body lines. Count their spans rather
        # than giving a short title the same weight as a whole body block.
        sizes = [span.style.size for span in page.spans if span.text.strip()]
        median_size = median(sizes) if sizes else None
        max_gap = (page.bounds.y1 - page.bounds.y0) * config.max_vertical_gap_page_ratio
        def is_heading(block_id):
            text, size, bold = data[block_id]
            return bool(text) and len(text.split()) <= 10 and (
                bold or median_size is not None and size is not None
                and size >= median_size * config.heading_size_ratio)

        grouped, relationships = [], []
        index = 0
        while index < len(ids):
            current = [ids[index]]
            current_kind = _kind(data[ids[index]][0])
            while index + 1 < len(ids):
                first, second = ids[index], ids[index + 1]
                first_text, first_size, _ = data[first]
                second_text, second_size, _ = data[second]
                same_region = bool(set(membership[first]) & set(membership[second]))
                gap = max(0.0, features[second].y0 - features[first].y1)
                close = gap <= max_gap
                # Region membership may be ambiguous. Require physical alignment
                # and forward movement before joining native blocks.
                a, b = features[first], features[second]
                alignment_tolerance = min(a.height, b.height)
                same_flow = (b.y0 >= a.y1 and min(a.x1, b.x1) > max(a.x0, b.x0)
                             and abs(a.x0 - b.x0) <= alignment_tolerance)
                safe_pair = same_flow and not is_heading(first) and not is_heading(second)
                second_kind = _kind(second_text)
                same_style = (first_size is None or second_size is None or abs(first_size - second_size) <= 0.5)
                label_value = bool(_LABEL.match(first_text)) and bool(second_text) and same_region and close
                list_pair = current_kind != StructuralType.UNKNOWN and current_kind == second_kind and same_region and close
                continuation = (bool(first_text and second_text) and same_region and close and same_style
                                and current_kind == StructuralType.UNKNOWN and second_kind == StructuralType.UNKNOWN
                                and not first_text.rstrip().endswith((".", "!", "?", ":")))
                if not safe_pair or not (label_value or list_pair or continuation):
                    break
                relationship_kind = (RelationshipKind.LABEL_VALUE if label_value else
                                     RelationshipKind.LIST_CONTINUATION if list_pair else
                                     RelationshipKind.PARAGRAPH_CONTINUATION)
                relationships.append(LogicalRelationship(first, second, relationship_kind,
                    Evidence(0.85 if label_value or list_pair else 0.70,
                             ("m4/adjacent-structural-grouping/1",),
                             ("Adjacent M3 blocks share region and satisfy conservative spacing/style cues",))))
                current.append(second)
                current_kind = current_kind if list_pair else (StructuralType.LABEL_VALUE if label_value else StructuralType.PARAGRAPH)
                index += 1
            grouped.append((tuple(current), current_kind))
            index += 1
        logical = []
        for logical_index, (source_ids, structural_type) in enumerate(grouped):
            text, size, bold = data[source_ids[0]]
            heading_like = (len(source_ids) == 1 and bool(text) and len(text.split()) <= 10
                            and ((median_size is not None and size is not None and size >= median_size * config.heading_size_ratio) or bold))
            if heading_like:
                structural_type = StructuralType.HEADING_LIKE
            elif structural_type == StructuralType.UNKNOWN:
                structural_type = StructuralType.TEXT_GROUP if len(source_ids) > 1 else StructuralType.UNKNOWN
            sources = tuple(features[block_id].source for block_id in source_ids)
            positions = tuple(ids.index(block_id) for block_id in source_ids)
            logical.append(LogicalBlock(f"p{page.number}/lb{logical_index}", page.number, source_ids, positions,
                                        union(source.box for source in sources), structural_type, heading_like, sources,
                                        Evidence(0.80, ("m4/logical-block/1",),
                                                 ("Adjacent structural references retained without semantic classification",))))
        for first, second in zip(ids, ids[1:]):
            if _DATE.search(data[first][0]) or _DATE.search(data[second][0]):
                if bool(set(membership[first]) & set(membership[second])):
                    relationships.append(LogicalRelationship(first, second, RelationshipKind.DATE_RELATIONSHIP,
                        Evidence(0.55, ("m4/date-like-adjacency/1",),
                                 ("Date-like text is adjacent within the same physical region",))))
        pages.append(LogicalPage(page.number, tuple(logical), tuple(relationships), ()))
    return LogicalDocumentIR(document.document_id, "logical-document-ir/1", LOGICAL_BLOCKS_VERSION,
                             layout.stage_version, order.stage_version, fingerprint, tuple(pages), document.issues)
