"""M7 records -> M8 normalized records. M7 objects are never modified."""

import re

from ..entities.models import EntityType as E
from ..records.models import ResumeRecords
from .contact_normalizer import normalize_email, normalize_phone, normalize_url
from .date_normalizer import normalize_date_range
from .models import (NormalizationResult, NormalizedAward, NormalizedCertification, NormalizedContact,
                     NormalizedEducation, NormalizedInterests, NormalizedLanguages, NormalizedProject,
                     NormalizedResume, NormalizedSkills, NormalizedSummary, NormalizedWorkExperience,
                     NormalizedValue)
from .skill_normalizer import normalize_skill
from .text_normalizer import clean_text, make_value, normalize_text, suspicious_warnings
from .validators import overall_status, validate

STAGE_VERSION = "normalization/1"
# Degrees: only explicit formatting entries. "CS" is deliberately not expanded.
DEGREE_FORMATTING = {"diploma": "Diploma"}
ORDINAL = re.compile(r"(\d{1,2})(st|nd|rd|th)", re.IGNORECASE)


def refs_for(record, kind: E, value: str | None):
    """The M6 entity references a record field came from (exact value match, else all of that type)."""
    matching = [ref for ref in record.entities if ref.entity_type == kind]
    exact = [ref for ref in matching if ref.raw_value == value]
    return tuple(exact[:1] or matching)


def text_field(record, kind: E, value: str | None):
    return None if value is None else normalize_text(value, refs_for(record, kind, value))


def date_field(record, value: str | None):
    if value is None:
        return None
    refs = refs_for(record, E.DATE_RANGE, value)
    result = normalize_date_range(value, refs)
    distinct = {ref.raw_value for ref in record.entities if ref.entity_type == E.DATE_RANGE}
    if len(distinct) > 1:
        result = result.model_copy(update={"warnings": result.warnings + (
            f"conflicting-date-values: record holds {len(distinct)} different date ranges",)})
    return result


def normalize_degree(raw: str, sources) -> NormalizedValue:
    text, rules = clean_text(raw)
    if text.casefold() in DEGREE_FORMATTING and DEGREE_FORMATTING[text.casefold()] != text:
        text = DEGREE_FORMATTING[text.casefold()]
        rules.append("degree-mapping/" + text.casefold())
    elif (match := ORDINAL.fullmatch(text)) and match[2] != match[2].lower():
        text = match[1] + match[2].lower()
        rules.append("degree/ordinal-suffix-case")
    return make_value(raw, text, rules, suspicious_warnings(raw), sources)


def common(record) -> dict:
    return {"raw_text": record.raw_text, "logical_block_ids": record.logical_block_ids}


def normalize_contact(record) -> NormalizedContact:
    def field(kind, value, normalizer):
        return None if value is None else normalizer(value, refs_for(record, kind, value))
    return NormalizedContact(
        name=text_field(record, E.NAME, record.name),
        email=field(E.EMAIL, record.email, normalize_email),
        phone=field(E.PHONE, record.phone, normalize_phone),
        location=text_field(record, E.LOCATION, record.location),
        linkedin=field(E.LINKEDIN, record.linkedin, normalize_url),
        github=field(E.GITHUB, record.github, normalize_url),
        website=field(E.WEBSITE, record.website, normalize_url), **common(record))


def normalize_work(record) -> NormalizedWorkExperience:
    return NormalizedWorkExperience(
        company=text_field(record, E.COMPANY, record.company), role=text_field(record, E.ROLE, record.role),
        date_range=date_field(record, record.date_range),
        description=text_field(record, E.DESCRIPTION, record.description), **common(record))


def normalize_education(record) -> NormalizedEducation:
    degree = (None if record.degree is None
              else normalize_degree(record.degree, refs_for(record, E.DEGREE, record.degree)))
    return NormalizedEducation(
        institution=text_field(record, E.INSTITUTION, record.institution), degree=degree,
        field_of_study=text_field(record, E.FIELD_OF_STUDY, record.field_of_study),
        date_range=date_field(record, record.date_range), **common(record))


def normalize_project(record) -> NormalizedProject:
    return NormalizedProject(
        name=text_field(record, E.PROJECT_NAME, record.name),
        description=text_field(record, E.PROJECT_DESCRIPTION, record.description),
        technology=tuple(normalize_skill(v, refs_for(record, E.TECHNOLOGY, v)) for v in record.technology),
        **common(record))


def normalize_certification(record) -> NormalizedCertification:
    url = record.credential_url
    return NormalizedCertification(
        certification=text_field(record, E.CERTIFICATION, record.certification),
        issuing_organization=text_field(record, E.ISSUING_ORGANIZATION, record.issuing_organization),
        credential_id=text_field(record, E.CREDENTIAL_ID, record.credential_id),
        credential_url=None if url is None else normalize_url(url, refs_for(record, E.CREDENTIAL_URL, url)),
        **common(record))


def normalize_resume(records: ResumeRecords) -> NormalizedResume:
    skills, languages, interests = records.skills, records.languages, records.interests
    return NormalizedResume(
        document_id=records.document_id,
        contact=normalize_contact(records.contact),
        summary=(NormalizedSummary(text=normalize_text(records.summary.text, records.summary.entities),
                                   **common(records.summary)) if records.summary else None),
        work_experience=tuple(normalize_work(r) for r in records.work_experience),
        education=tuple(normalize_education(r) for r in records.education),
        projects=tuple(normalize_project(r) for r in records.projects),
        certifications=tuple(normalize_certification(r) for r in records.certifications),
        awards=tuple(NormalizedAward(text=normalize_text(r.text, r.entities), **common(r)) for r in records.awards),
        skills=(NormalizedSkills(skills=tuple(normalize_skill(v, refs_for(skills, E.SKILL, v)) for v in skills.skills),
                                 **common(skills)) if skills else None),
        languages=(NormalizedLanguages(
            languages=tuple(normalize_text(v, refs_for(languages, E.LANGUAGE, v)) for v in languages.languages),
            **common(languages)) if languages else None),
        interests=(NormalizedInterests(
            interests=tuple(normalize_text(v, refs_for(interests, E.INTEREST, v)) for v in interests.interests),
            **common(interests)) if interests else None),
        unassigned=records.unassigned)


def normalize_records(records: ResumeRecords) -> NormalizationResult:
    resume = normalize_resume(records)
    issues = validate(resume)
    status = overall_status(issues)
    return NormalizationResult(stage_version=STAGE_VERSION, resume=resume, issues=issues, status=status)
