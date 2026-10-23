"""M8 -> M9. A pure mapping: no parsing, no normalization, no inference."""

from ..normalization.date_normalizer import fmt
from ..normalization.models import NormalizationResult, NormalizationStatus as N, NormalizedDateRange, NormalizedValue
from .models import (CanonicalAward, CanonicalCandidate, CanonicalCertification, CanonicalEducation,
                     CanonicalInterest, CanonicalLanguage, CanonicalMetadata, CanonicalProject,
                     CanonicalResume, CanonicalSkill, CanonicalSummary, CanonicalValidation,
                     CanonicalWorkExperience, Provenance)

RANGE_RULE_PREFIX = "date-range/separator"


def value_of(value: NormalizedValue | None) -> str | None:
    return None if value is None else value.normalized_value


def provenance_of(value: NormalizedValue) -> Provenance:
    return Provenance(raw_value=value.raw_value, normalization_status=value.status.value,
                      normalization_rules=value.rules, entities=value.sources)


def field_provenance(record, fields) -> dict[str, Provenance]:
    """Provenance for each named M8 field that exists, in the given (stable) order."""
    return {name: provenance_of(getattr(record, name)) for name in fields
            if getattr(record, name) is not None}


def date_fields(date: NormalizedDateRange | None) -> tuple[str | None, str | None, bool]:
    """(start, end, current) from M8's structured date. Only successfully parsed ranges are mapped.

    A lone date such as "2018" is not mapped: M8 does not say whether it is a start or an end,
    and M9 must not guess. Its raw value stays available in provenance.
    """
    if date is None or date.status not in (N.NORMALIZED, N.UNCHANGED):
        return None, None, False
    if not any(rule.startswith(RANGE_RULE_PREFIX) for rule in date.rules):
        return None, None, False
    end = None if date.end_year is None else fmt(date.end_year, date.end_month)
    return fmt(date.start_year, date.start_month), end, date.is_current


def items(values, cls):
    """Map list values; an item M8 could not normalize (value None) has no canonical value and is skipped
    (M8 already reports it as a validation issue)."""
    return [cls(value=v.normalized_value, raw_value=v.raw_value, provenance=provenance_of(v))
            for v in values if v.normalized_value is not None]


def build_canonical_resume(result: NormalizationResult, source_file_name: str, page_count: int,
                           logical_block_count: int, extracted_entity_count: int) -> CanonicalResume:
    resume = result.resume
    contact = resume.contact
    contact_fields = ("name", "email", "phone", "location", "linkedin", "github", "website")
    candidate = CanonicalCandidate(**{name: value_of(getattr(contact, name)) for name in contact_fields},
                                   provenance=field_provenance(contact, contact_fields))
    summary = None
    if resume.summary:
        text = resume.summary.text
        summary = CanonicalSummary(value=text.normalized_value, raw_value=text.raw_value,
                                   provenance=provenance_of(text))

    work = []
    for record in resume.work_experience:
        start, end, current = date_fields(record.date_range)
        work.append(CanonicalWorkExperience(
            company=value_of(record.company), role=value_of(record.role), start_date=start, end_date=end,
            current=current, description=value_of(record.description), raw_text=record.raw_text,
            logical_block_ids=record.logical_block_ids,
            provenance=field_provenance(record, ("company", "role", "date_range", "description"))))

    education = []
    for record in resume.education:
        start, end, _ = date_fields(record.date_range)
        education.append(CanonicalEducation(
            institution=value_of(record.institution), degree=value_of(record.degree),
            field_of_study=value_of(record.field_of_study), start_date=start, end_date=end,
            raw_text=record.raw_text, logical_block_ids=record.logical_block_ids,
            provenance=field_provenance(record, ("institution", "degree", "field_of_study", "date_range"))))

    projects = [CanonicalProject(
        name=value_of(record.name), description=value_of(record.description),
        technologies=items(record.technology, CanonicalSkill), raw_text=record.raw_text,
        logical_block_ids=record.logical_block_ids,
        provenance=field_provenance(record, ("name", "description"))) for record in resume.projects]

    certifications = [CanonicalCertification(
        name=value_of(record.certification), issuing_organization=value_of(record.issuing_organization),
        credential_id=value_of(record.credential_id), credential_url=value_of(record.credential_url),
        raw_text=record.raw_text, logical_block_ids=record.logical_block_ids,
        provenance=field_provenance(record, ("certification", "issuing_organization", "credential_id",
                                             "credential_url"))) for record in resume.certifications]

    awards = [CanonicalAward(value=value_of(record.text), raw_text=record.raw_text,
                             logical_block_ids=record.logical_block_ids, provenance=provenance_of(record.text))
              for record in resume.awards]

    return CanonicalResume(
        candidate=candidate, summary=summary,
        skills=items(resume.skills.skills, CanonicalSkill) if resume.skills else [],
        work_experience=work, education=education, projects=projects, certifications=certifications,
        awards=awards,
        languages=items(resume.languages.languages, CanonicalLanguage) if resume.languages else [],
        interests=items(resume.interests.interests, CanonicalInterest) if resume.interests else [],
        metadata=CanonicalMetadata(document_id=resume.document_id, source_file_name=source_file_name,
                                   page_count=page_count, logical_block_count=logical_block_count,
                                   extracted_entity_count=extracted_entity_count,
                                   validation_status=result.status),
        validation=CanonicalValidation(status=result.status, issues=list(result.issues),
                                       unresolved=list(resume.unassigned)))
