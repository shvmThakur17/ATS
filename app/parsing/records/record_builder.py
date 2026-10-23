"""M6 -> M7. Group the flat entity list into records using document order only.

Grouping rule (no coordinates, no inference):
  * Each record family has single-value "slots" (e.g. ROLE, COMPANY, DATE_RANGE) and
    optional multi-value entities (e.g. DESCRIPTION). Slot order is never assumed.
  * An entity whose slot is already filled in the open record starts a new record.
  * A change of M5 section closes every open record, so records never span sections.
Entities no record claims are returned in `unassigned`, unchanged.
"""

from ..entities.models import Entity, EntityDocumentIR, EntityType as E
from ..sections.models import SectionType as S
from .models import (AwardRecord, CertificationRecord, ContactRecord, EducationRecord, EntityRef,
                     InterestRecord, LanguageRecord, ProjectRecord, ResumeRecords, SkillsRecord,
                     SummaryRecord, WorkExperienceRecord)

CONTACT_FIELDS = {E.NAME: "name", E.EMAIL: "email", E.PHONE: "phone", E.LOCATION: "location",
                  E.LINKEDIN: "linkedin", E.GITHUB: "github", E.WEBSITE: "website"}

# family -> (slot types, multi-value types). Slots are filled at most once per record.
GROUPED = {
    "work": ((E.ROLE, E.COMPANY, E.DATE_RANGE), (E.DESCRIPTION,)),
    "education": ((E.INSTITUTION, E.DEGREE, E.FIELD_OF_STUDY, E.DATE_RANGE), ()),
    "project": ((E.PROJECT_NAME,), (E.PROJECT_DESCRIPTION, E.TECHNOLOGY)),
    "certification": ((E.CERTIFICATION, E.ISSUING_ORGANIZATION, E.CREDENTIAL_ID, E.CREDENTIAL_URL), ()),
}
# Types that mean different things in different sections are routed by section.
SECTION_ROUTED = {(E.DATE_RANGE, S.EXPERIENCE): "work", (E.DESCRIPTION, S.EXPERIENCE): "work",
                  (E.DATE_RANGE, S.EDUCATION): "education",
                  (E.DESCRIPTION, S.SUMMARY): "summary"}
TYPE_ROUTED = {kind: family for family, (slots, multi) in GROUPED.items()
               for kind in slots + multi if kind not in (E.DATE_RANGE, E.DESCRIPTION)}
LIST_FAMILIES = {E.SKILL: "skills", E.LANGUAGE: "languages", E.INTEREST: "interests", E.AWARD: "awards"}


def family_of(entity: Entity) -> str | None:
    kind = entity.entity_type
    if kind in CONTACT_FIELDS:
        return "contact"
    return (LIST_FAMILIES.get(kind) or TYPE_ROUTED.get(kind)
            or SECTION_ROUTED.get((kind, entity.section_type)))


def ordered_unique(values):
    return tuple(dict.fromkeys(values))


def common(refs: list[EntityRef]) -> dict:
    return {"raw_text": "\n".join(ref.raw_value for ref in refs),
            "logical_block_ids": ordered_unique(ref.logical_block_id for ref in refs),
            "entities": tuple(refs)}


def joined(refs: list[EntityRef], kind: E) -> str | None:
    values = [ref.raw_value for ref in refs if ref.entity_type == kind]
    return "\n".join(values) if values else None


def first(refs: list[EntityRef], kind: E) -> str | None:
    return next((ref.raw_value for ref in refs if ref.entity_type == kind), None)


def make_work(refs):
    return WorkExperienceRecord(company=first(refs, E.COMPANY), role=first(refs, E.ROLE),
                                date_range=first(refs, E.DATE_RANGE),
                                description=joined(refs, E.DESCRIPTION), **common(refs))


def make_education(refs):
    return EducationRecord(institution=first(refs, E.INSTITUTION), degree=first(refs, E.DEGREE),
                           field_of_study=first(refs, E.FIELD_OF_STUDY),
                           date_range=first(refs, E.DATE_RANGE), **common(refs))


def make_project(refs):
    return ProjectRecord(name=first(refs, E.PROJECT_NAME),
                         description=joined(refs, E.PROJECT_DESCRIPTION),
                         technology=tuple(r.raw_value for r in refs if r.entity_type == E.TECHNOLOGY),
                         **common(refs))


def make_certification(refs):
    return CertificationRecord(certification=first(refs, E.CERTIFICATION),
                               issuing_organization=first(refs, E.ISSUING_ORGANIZATION),
                               credential_id=first(refs, E.CREDENTIAL_ID),
                               credential_url=first(refs, E.CREDENTIAL_URL), **common(refs))


def group_records(entities: list[Entity]):
    """Return {family: [member EntityRef lists]} in document order."""
    records = {family: [] for family in GROUPED}
    open_refs = {}  # family -> member refs of the open record (also stored in `records`)
    last_section = None
    for entity in entities:
        family = family_of(entity)
        if family not in GROUPED:
            continue
        if entity.section_type != last_section:
            open_refs.clear()
            last_section = entity.section_type
        slots = GROUPED[family][0]
        ref = EntityRef.from_entity(entity)
        current = open_refs.get(family)
        if current is None or (entity.entity_type in slots
                               and any(r.entity_type == entity.entity_type for r in current)):
            current = []
            open_refs[family] = current
            records[family].append(current)
        current.append(ref)
    return records


def build_records(entity_ir: EntityDocumentIR) -> ResumeRecords:
    entities = list(entity_ir.entities)
    contact, contact_refs, claimed = {}, [], set()
    lists = {"skills": [], "languages": [], "interests": [], "awards": [],
             "summary": []}
    for entity in entities:
        family = family_of(entity)
        if family == "contact":
            field = CONTACT_FIELDS[entity.entity_type]
            if field not in contact:  # first value wins; surplus stays unassigned
                contact[field] = entity.raw_value
                contact_refs.append(EntityRef.from_entity(entity))
                claimed.add(entity.entity_id)
        elif family in lists:
            lists[family].append(EntityRef.from_entity(entity))
            claimed.add(entity.entity_id)

    grouped = group_records(entities)
    for members in grouped.values():
        claimed.update(ref.entity_id for refs in members for ref in refs)

    def collect(family, record_type, field):
        refs = lists[family]
        if not refs:
            return None
        # Exact duplicates only; provenance of every occurrence is kept in `entities`.
        values = ordered_unique(ref.raw_value for ref in refs)
        return record_type(**{field: values}, **common(refs))

    return ResumeRecords(
        document_id=entity_ir.document_id,
        contact=ContactRecord(**contact, **common(contact_refs)),
        summary=(SummaryRecord(text="\n".join(ref.raw_value for ref in lists["summary"]),
                               **common(lists["summary"])) if lists["summary"] else None),
        work_experience=tuple(make_work(refs) for refs in grouped["work"]),
        education=tuple(make_education(refs) for refs in grouped["education"]),
        projects=tuple(make_project(refs) for refs in grouped["project"]),
        certifications=tuple(make_certification(refs) for refs in grouped["certification"]),
        awards=tuple(AwardRecord(text=ref.raw_value, **common([ref])) for ref in lists["awards"]),
        skills=collect("skills", SkillsRecord, "skills"),
        languages=collect("languages", LanguageRecord, "languages"),
        interests=collect("interests", InterestRecord, "interests"),
        unassigned=tuple(EntityRef.from_entity(e) for e in entities if e.entity_id not in claimed),
    )
