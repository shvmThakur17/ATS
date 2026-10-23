"""M5 -> M6 entities. All text rules are kept in entity_rules.py."""

from ..logical_blocks.models import LogicalDocumentIR
from ..models import Evidence
from ..sections.models import SectionDocumentIR
from ..sections.models import SectionType
from ..sections.section_rules import heading_type
from .entity_rules import LABELS, ROLE, PROSE_START, name_shaped, line_matches
from .models import Entity, EntityDocumentIR, EntityType


def text_lines(text):
    """Keep offsets in the caller's exact resolved text."""
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        value = raw_line.strip()
        if value:
            start = offset + len(raw_line) - len(raw_line.lstrip())
            yield value, start
        offset += len(raw_line)


def narrative_matches(section, lines):
    """Keep wrapped narratives as contiguous source slices, stopping at labels.

    M5 remains authoritative. Only a standalone heading at the end of a mixed
    project block is excluded; heading words inside prose remain untouched.
    """
    if section == SectionType.PROJECTS:
        lines = list(lines)
        while len(lines) > 1 and heading_type(lines[-1][0]) != SectionType.UNKNOWN:
            lines.pop()
    if not lines:
        return [], lines
    if section == SectionType.SUMMARY:
        start = lines[0][1]
        end = lines[-1][1] + len(lines[-1][0])
        return [(start, end, EntityType.DESCRIPTION, "summary-block")], []
    if section not in (SectionType.PROJECTS, SectionType.AWARDS):
        return [], lines
    matches, remaining = [], []
    run = []

    def flush():
        if not run:
            return
        first = run[0][0]
        combined = " ".join(value for value, _ in run)
        kind = None
        if section == SectionType.AWARDS:
            if first.lower().startswith(("awarded ", "received ", "won ")):
                kind = EntityType.AWARD
            elif len(run) > 1 and any(m[0] == EntityType.AWARD for m in line_matches(section, first)):
                kind = EntityType.AWARD
        elif PROSE_START.match(first) or combined.endswith((".", "!")):
            kind = EntityType.PROJECT_DESCRIPTION
        if kind:
            matches.append((run[0][1], run[-1][1] + len(run[-1][0]), kind,
                            "section-narrative-block"))
        else:
            remaining.extend(run)
        run.clear()

    for index, (value, start) in enumerate(lines):
        following = lines[index + 1][0] if index + 1 < len(lines) else ""
        label, colon, _ = value.partition(":")
        labeled = bool(colon and (label.strip().lower() in LABELS.get(section, {})
                                 or section == SectionType.PROJECTS and label.strip().lower()
                                 in ("technologies", "technology", "tech stack")))
        title = (section == SectionType.PROJECTS and not run
                 and any(m[0] == EntityType.PROJECT_NAME for m in line_matches(section, value, following)))
        if title:
            flush()
            matches.append((start, start + len(value), EntityType.PROJECT_NAME,
                            "title-followed-by-description"))
        elif labeled:
            flush()
            remaining.append((value, start))
            # A label alone owns the next line; preserve the existing label/value path.
        elif remaining and remaining[-1][0].endswith(":"):
            remaining.append((value, start))
        else:
            run.append((value, start))
    flush()
    return matches, remaining


def validate_inputs(sections, logical, text_by_block):
    if sections.schema_version != "sections-ir/1" or sections.stage_version != "sections/1":
        raise ValueError("Unsupported M5 contract")
    if logical.schema_version != "logical-document-ir/1" or logical.stage_version != "logical-blocks/1":
        raise ValueError("Unsupported M4 contract")
    if sections.document_id != logical.document_id or sections.source_config_fingerprint != logical.config_fingerprint:
        raise ValueError("M5 must refer to the supplied M4 document")
    blocks = [block for page in logical.pages for block in page.logical_blocks]
    ids = [block.logical_block_id for block in blocks]
    assigned_ids = [block.logical_block_id for block in sections.blocks]
    if len(ids) != len(set(ids)) or ids != assigned_ids:
        raise ValueError("M5 assignments must cover M4 blocks once in the existing order")
    for assignment, block in zip(sections.blocks, blocks):
        if assignment.page_number != block.page_number or not block.sources:
            raise ValueError("Missing provenance or wrong-page assignment")
        if block.logical_block_id not in text_by_block or not isinstance(text_by_block[block.logical_block_id], str):
            raise ValueError("Every logical block needs resolved text")
    return blocks


def extract_entities(sections: SectionDocumentIR, logical: LogicalDocumentIR,
                     text_by_block: dict[str, str]) -> EntityDocumentIR:
    """Trust M5 section assignments; emit raw atomic values in document order.

    SourceRefs retain M4 block-level provenance. text_start/text_end are exact
    offsets into text_by_block, not invented native span character offsets.
    """
    blocks = validate_inputs(sections, logical, text_by_block)
    entities = []
    for block_index, (assignment, block) in enumerate(zip(sections.blocks, blocks)):
        if assignment.is_heading:
            continue
        text = text_by_block[block.logical_block_id]
        lines = list(text_lines(text))
        next_text = ""
        if block_index + 1 < len(blocks):
            next_assignment = sections.blocks[block_index + 1]
            if (next_assignment.page_number == assignment.page_number
                    and next_assignment.section_type == assignment.section_type
                    and not next_assignment.is_heading):
                next_text = text_by_block[blocks[block_index + 1].logical_block_id].strip()
        matches, lines = narrative_matches(assignment.section_type, lines)
        # Restrict unlabeled names to the leading styled block and an adjacent
        # job title on the same page. No coordinates or resume-specific names.
        if (block_index == 0 and assignment.section_type == SectionType.UNKNOWN
                and block.heading_like and len(lines) == 1 and name_shaped(lines[0][0])
                and heading_type(lines[0][0]) == SectionType.UNKNOWN
                and next_text and len(next_text.split()) <= 6 and ROLE.search(next_text)
                and not PROSE_START.match(next_text)):
            value, start = lines[0]
            matches.append((start, start + len(value), EntityType.NAME, "leading-name-with-role-context"))
            lines = []
        if (assignment.section_type == SectionType.PROJECTS and block.heading_like
                and len(lines) == 1 and next_text):
            value, start = lines[0]
            matches.append((start, start + len(value), EntityType.PROJECT_NAME, "project-title-with-body-context"))
            lines = []
        consumed_lines = set()
        for index, (line, line_start) in enumerate(lines):
            if index in consumed_lines:
                continue
            following = lines[index + 1][0] if index + 1 < len(lines) else next_text
            # M4 label/value groups may retain the label on its own native line.
            # Resolve only within that logical block; never cross a record boundary.
            if line.endswith(":") and index + 1 < len(lines):
                label = line[:-1].strip().lower()
                if label in LABELS.get(assignment.section_type, {}):
                    value, value_start = lines[index + 1]
                    line = text[line_start:value_start + len(value)]
                    consumed_lines.add(index + 1)
            found = list(line_matches(assignment.section_type, line, following))
            if not found:
                found = [(EntityType.UNKNOWN, 0, len(line), "unclassified-text")]
            for kind, start, end, rule in found:
                matches.append((line_start + start, line_start + end, kind, rule))
        seen = set()
        for start, end, kind, rule in sorted(matches, key=lambda match: (match[0], match[1], match[2].value)):
            identity = (kind, start, end)
            if identity in seen:
                continue
            seen.add(identity)
            strength = 0.0 if kind == EntityType.UNKNOWN else min(assignment.evidence.score, 0.85)
            entities.append(Entity(f"{block.logical_block_id}/e{len(seen) - 1}", kind, text[start:end],
                                   block.logical_block_id, block.page_number, assignment.section_type,
                                   block.source_block_ids, block.sources, start, end,
                                   Evidence(strength, (f"m6/{rule}/1",),
                                            ("Rule applied within the M5-assigned section; raw text retained",))))
    return EntityDocumentIR(sections.document_id, "entities-ir/1", "entities/1",
                             sections.stage_version, sections.source_config_fingerprint, tuple(entities))
