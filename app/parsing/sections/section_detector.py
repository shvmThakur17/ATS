"""Conservative section boundaries in the existing M4 traversal order."""

from ..models import DocumentIR, Evidence
from dataclasses import replace
import re
from ..logical_blocks.models import LogicalDocumentIR
from .models import SectionBlock, SectionDocumentIR, SectionType
from .section_rules import heading_type

ALIGNMENT_HEIGHT_FACTOR = 1.0
MAX_CONTEXT_WORDS = 12


class ResolvedLogicalText(dict):
    """Ordinary text mapping with optional, unchanged native line provenance."""

    def __init__(self):
        super().__init__()
        self.lines_by_block = {}


def is_record_title(section, block, text, following, following_text):
    """A title followed by body text can belong inside an established section."""
    if section == SectionType.EXPERIENCE and block.heading_like:
        return bool(re.search(r"\b(?:Engineer|Developer|Manager|Analyst|Architect|Consultant|Ltd|Limited|Inc|LLC)\b", text, re.I))
    return (section in (SectionType.PROJECTS, SectionType.EXPERIENCE, SectionType.EDUCATION)
            and block.heading_like and following is not None
            and not following.heading_like and heading_type(following_text) == SectionType.UNKNOWN
            and same_flow(block, following))


def resolve_logical_text(logical: LogicalDocumentIR, source: DocumentIR):
    """Resolve M4 references because M4 deliberately contains no text payload.

    Preserve native line boundaries; perform no extraction, layout or reordering.
    """
    if logical.document_id != source.document_id:
        raise ValueError("M1 and M4 document identities must match")
    native = {}
    for page in source.pages:
        spans = {span.source.object_id: span.text for span in page.spans}
        for block in page.blocks:
            lines = ["".join(spans[key] for key in line.span_ids) for line in block.lines]
            native[block.source.object_id] = (page.number, "\n".join(lines),
                                             tuple(zip(lines, block.lines)))
    text = ResolvedLogicalText()
    for page in logical.pages:
        for block in page.logical_blocks:
            parts = []
            resolved_lines = []
            for source_id in block.source_block_ids:
                if source_id not in native or native[source_id][0] != page.page_number:
                    raise ValueError("M4 contains a missing or wrong-page source reference")
                parts.append(native[source_id][1])
                resolved_lines.extend(native[source_id][2])
            text[block.logical_block_id] = "\n".join(parts)
            text.lines_by_block[block.logical_block_id] = tuple(resolved_lines)
    return text


def embedded_heading(block, text, resolved):
    """Only an exact alias on the final standalone line is a candidate."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    section = heading_type(lines[-1])
    if section == SectionType.UNKNOWN:
        return None
    native_lines = getattr(resolved, "lines_by_block", {}).get(block.logical_block_id, ())
    native_lines = [(value, line) for value, line in native_lines if value.strip()]
    if native_lines and native_lines[-1][0].strip() == lines[-1].strip():
        source = native_lines[-1][1].source
        anchor = replace(block, bounding_box=source.box)
        reason = f"Trailing whole-line alias references native line {source.object_id}"
    else:
        # Plain mappings retain the existing API, but only block geometry is known.
        anchor = block
        reason = "Trailing whole-line alias uses logical-block geometry"
    return section, anchor, reason


def apply_embedded_headings(logical, resolved, assignments):
    """Recover boundaries without splitting, relabeling or reordering mixed blocks.

    Native geometry also handles a heading whose column is visited earlier in M3.
    Require aligned content below it; a nearer valid heading ends its influence.
    """
    by_id = {item.logical_block_id: item for item in assignments}
    for page in logical.pages:
        blocks = page.logical_blocks
        candidates = []
        for block in blocks:
            candidate = embedded_heading(block, resolved[block.logical_block_id], resolved)
            if candidate is not None:
                section, anchor, reason = candidate
                if any(other.logical_block_id != block.logical_block_id
                       and same_flow(anchor, other) and aligned(anchor, other)
                       and resolved[other.logical_block_id].strip() for other in blocks):
                    candidates.append((section, anchor, reason))
        for block in blocks:
            item = by_id[block.logical_block_id]
            if item.is_heading or not resolved[block.logical_block_id].strip():
                continue
            eligible = []
            for section, anchor, reason in candidates:
                if (block.logical_block_id == anchor.logical_block_id
                        or not same_flow(anchor, block) or not aligned(anchor, block)):
                    continue
                barriers = []
                for index, other in enumerate(blocks):
                    following = blocks[index + 1] if index + 1 < len(blocks) else None
                    following_text = resolved[following.logical_block_id] if following else ""
                    record_title = is_record_title(section, other, resolved[other.logical_block_id],
                                                   following, following_text)
                    boundary = (by_id[other.logical_block_id].is_heading
                                or (other.heading_like and not record_title)
                                or not resolved[other.logical_block_id].strip())
                    if boundary and same_flow(anchor, other) and aligned(anchor, other):
                        barriers.append(other)
                intervening = any(other.logical_block_id == block.logical_block_id
                                  or same_flow(other, block) for other in barriers)
                if not intervening:
                    eligible.append((section, anchor, reason))
            if not eligible:
                continue
            nearest_y = max(anchor.bounding_box.y1 for _, anchor, _ in eligible)
            nearest = [candidate for candidate in eligible if candidate[1].bounding_box.y1 == nearest_y]
            if len(nearest) != 1:
                continue  # Ambiguous competing headings must not create certainty.
            section, anchor, reason = nearest[0]
            evidence = Evidence(.7, ("m5/trailing-heading-and-flow/1",),
                                (reason, "Content is below and aligned with the embedded heading"))
            by_id[block.logical_block_id] = replace(item, section_type=section,
                heading_block_id=anchor.logical_block_id, evidence=evidence)
        for _, anchor, reason in candidates:
            item = by_id[anchor.logical_block_id]
            evidence = replace(item.evidence,
                rules=item.evidence.rules + ("m5/trailing-heading-and-flow/1",),
                reasons=item.evidence.reasons + (reason, "Mixed block retains its previous section assignment"))
            by_id[anchor.logical_block_id] = replace(item, evidence=evidence)
    return tuple(by_id[item.logical_block_id] for item in assignments)


def aligned(first, second):
    a, b = first.bounding_box, second.bounding_box
    tolerance = min(a.y1 - a.y0, b.y1 - b.y0) * ALIGNMENT_HEIGHT_FACTOR
    return abs(a.x0 - b.x0) <= tolerance and min(a.x1, b.x1) > max(a.x0, b.x0)


def same_flow(first, second):
    a, b = first.bounding_box, second.bounding_box
    return b.y0 >= a.y1 and min(a.x1, b.x1) > max(a.x0, b.x0)


def heading_evidence(block, text, following, following_text):
    section = heading_type(text)
    if section == SectionType.UNKNOWN:
        return section, None
    reasons = ["Whole heading matches a section alias"]
    if block.heading_like:
        reasons.append("M4 identifies heading-like typography")
    if following is not None and same_flow(block, following) and following_text.strip():
        if heading_type(following_text) == SectionType.UNKNOWN and not following.heading_like:
            reasons.append("Following content occupies the same vertical flow")
            if section in (SectionType.SKILLS, SectionType.LANGUAGES, SectionType.INTERESTS):
                if len(following_text.split()) <= MAX_CONTEXT_WORDS:
                    reasons.append("Following content is a short item or item group")
    if len(reasons) == 1:
        return SectionType.UNKNOWN, None
    return section, Evidence(0.9 if block.heading_like else 0.7,
                             ("m5/heading-and-context/1",), tuple(reasons))


def detect_sections(logical: LogicalDocumentIR, text_by_block: dict[str, str]):
    """Text is supplied by resolve_logical_text or an equivalent reference resolver.

    The caller retains M4 for provenance and source issues. M5 emits one reference
    per logical block and never splits or reorders an upstream block.
    """
    if logical.schema_version != "logical-document-ir/1" or logical.stage_version != "logical-blocks/1":
        raise ValueError("Unsupported M4 contract")
    seen = set()
    page_numbers = [page.page_number for page in logical.pages]
    if page_numbers != sorted(set(page_numbers)):
        raise ValueError("M4 pages must be unique and ordered")
    output = []
    current = SectionType.UNKNOWN
    heading_id = None
    previous = None
    for page in logical.pages:
        if not page.logical_blocks:
            current, heading_id, previous = SectionType.UNKNOWN, None, None
        for index, block in enumerate(page.logical_blocks):
            key = block.logical_block_id
            if key in seen or block.page_number != page.page_number:
                raise ValueError("Duplicate or wrong-page M4 logical block")
            if key not in text_by_block or not isinstance(text_by_block[key], str):
                raise ValueError("Every M4 block requires resolved text")
            seen.add(key)
            following = page.logical_blocks[index + 1] if index + 1 < len(page.logical_blocks) else None
            following_text = text_by_block.get(following.logical_block_id, "") if following else ""
            text = text_by_block[key]
            section, evidence = heading_evidence(block, text, following, following_text)
            is_heading = evidence is not None
            if is_heading:
                current, heading_id = section, key
            else:
                new_page = previous is not None and previous.page_number != block.page_number
                continues = previous is not None and same_flow(previous, block)
                if new_page:
                    continues = block.page_number == previous.page_number + 1 and aligned(previous, block)
                record_title = is_record_title(current, block, text, following, following_text)
                mixed_boundary = embedded_heading(block, text, text_by_block) is not None
                # A composite block's heading typography must not relabel its prefix.
                record_title = record_title or (mixed_boundary and current != SectionType.UNKNOWN and continues)
                if (block.heading_like and not record_title) or not text.strip() or not continues:
                    current, heading_id = SectionType.UNKNOWN, None
                if current == SectionType.UNKNOWN:
                    evidence = Evidence(0.0, ("m5/unknown/1",), ("Insufficient heading or continuation evidence",))
                else:
                    reason = "Aligned continuation on the next page" if new_page else "Content follows the active heading in the same flow"
                    evidence = Evidence(0.55 if new_page else 0.75, ("m5/section-continuation/1",), (reason,))
            output.append(SectionBlock(key, page.page_number, current, heading_id, is_heading, evidence))
            previous = block
    return SectionDocumentIR(logical.document_id, "sections-ir/1", logical.stage_version,
                             logical.config_fingerprint, "sections/1",
                             apply_embedded_headings(logical, text_by_block, output))
