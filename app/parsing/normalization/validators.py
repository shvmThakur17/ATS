"""Deterministic validation of normalized records. Never edits or repairs a value."""

from collections import Counter

from ..entities.models import EntityType
from .models import (NormalizationStatus as Status, NormalizedValue, NormalizedResume, Severity,
                     ValidationIssue, ValidationStatus)

RECORD_TYPES = ("work_experience", "education", "projects", "certifications", "awards")
# (record_type, field) -> code reported when the value is missing or empty (ERROR).
REQUIRED = {("projects", "name"): "PROJECT_NAME_MISSING", ("awards", "text"): "AWARD_TEXT_MISSING",
            ("skills", "skills"): "SKILL_EMPTY", ("languages", "languages"): "LANGUAGE_EMPTY",
            ("interests", "interests"): "INTEREST_EMPTY"}


def records_of(resume: NormalizedResume):
    """Yield (record_type, index, record) in a stable order."""
    yield "contact", None, resume.contact
    if resume.summary:
        yield "summary", None, resume.summary
    for record_type in RECORD_TYPES:
        for index, record in enumerate(getattr(resume, record_type)):
            yield record_type, index, record
    for record_type in ("skills", "languages", "interests"):
        if getattr(resume, record_type):
            yield record_type, None, getattr(resume, record_type)


def values_of(record):
    """Yield (field, value) for every NormalizedValue held by a record."""
    for field in type(record).model_fields:
        item = getattr(record, field)
        if isinstance(item, NormalizedValue):
            yield field, item
        elif isinstance(item, tuple):
            for value in item:
                if isinstance(value, NormalizedValue):
                    yield field, value


def split_warning(warning: str) -> tuple[str, str]:
    code, _, message = warning.partition(": ")
    return code.upper().replace("-", "_"), message or warning


def validate(resume: NormalizedResume) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []

    def add(severity, code, message, record_type, index, field=None, sources=()):
        issues.append(ValidationIssue(severity=severity, code=code, message=message, field=field,
                                      record_type=record_type, record_index=index,
                                      source_entity_ids=tuple(ref.entity_id for ref in sources)))

    for record_type, index, record in records_of(resume):
        for field, value in values_of(record):
            check_value(add, record_type, index, field, value)
        CHECKS.get(record_type, lambda *_: None)(add, record_type, index, record)

    for ref in resume.unassigned:
        add(Severity.WARNING, "UNRESOLVED_ENTITY",
            f"{ref.entity_type.value} entity left unresolved by M7: {ref.raw_value!r}",
            "unassigned", None, ref.entity_type.value, (ref,))
    return tuple(issues)


def check_value(add, record_type, index, field, value):
    """Rules that apply to every value: raw preservation, traceability, normalizer warnings."""
    sources = value.sources
    if sources and value.raw_value != "\n".join(ref.raw_value for ref in sources):
        add(Severity.ERROR, "RAW_VALUE_MISMATCH", "raw_value does not match its M6 source entity",
            record_type, index, field, sources)
    if value.normalized_value is not None and value.normalized_value != value.raw_value and (
            value.status != Status.NORMALIZED or not value.rules):
        add(Severity.ERROR, "UNDOCUMENTED_NORMALIZATION",
            "normalized_value differs from raw_value without status NORMALIZED and a rule",
            record_type, index, field, sources)
    required = REQUIRED.get((record_type, field))
    for warning in value.warnings:
        code, message = split_warning(warning)
        if code == "EMPTY_VALUE":
            if required:
                continue  # reported below as a required-field error
            code = "EMPTY_NORMALIZED_FIELD"
        severity = Severity.ERROR if value.status == Status.INVALID else Severity.WARNING
        add(severity, code, message, record_type, index, field, sources)
    if required and value.normalized_value is None:
        add(Severity.ERROR, required, f"{field} is empty", record_type, index, field, sources)
    if value.status in (Status.UNKNOWN, Status.INVALID) and not value.warnings:
        add(Severity.WARNING, f"NORMALIZATION_{value.status.value}", "value could not be normalized",
            record_type, index, field, sources)


def check_contact(add, record_type, index, contact):
    for field in ("email", "phone"):
        if getattr(contact, field) is None:
            add(Severity.INFO, f"CONTACT_{field.upper()}_MISSING", f"no {field} was extracted", record_type, index, field)


def check_dates(add, record_type, index, date):
    if date is None:
        return
    if date.is_current:
        add(Severity.INFO, "CURRENT_EMPLOYMENT", "date range ends with Present", record_type, index, "date_range",
            date.sources)
        if date.end_year is not None:
            add(Severity.ERROR, "CURRENT_WITH_END_DATE", "a current range also has an end date",
                record_type, index, "date_range", date.sources)
    elif date.status in (Status.NORMALIZED, Status.UNCHANGED) and "present" in date.raw_value.lower():
        add(Severity.WARNING, "PRESENT_NOT_CURRENT", "raw date mentions Present but range is not current",
            record_type, index, "date_range", date.sources)


def check_work(add, record_type, index, record):
    for field in ("company", "role"):
        if getattr(record, field) is None:
            add(Severity.WARNING, f"WORK_{field.upper()}_MISSING", f"work record has no {field}",
                record_type, index, field)
    check_dates(add, record_type, index, record.date_range)


def check_education(add, record_type, index, record):
    if record.institution is None and record.degree is None:
        add(Severity.ERROR, "EDUCATION_EMPTY", "education record has neither institution nor degree",
            record_type, index)
    else:
        for field in ("institution", "degree"):
            if getattr(record, field) is None:
                add(Severity.WARNING, f"EDUCATION_{field.upper()}_MISSING", f"education record has no {field}",
                    record_type, index, field)
    check_dates(add, record_type, index, record.date_range)


def check_project(add, record_type, index, record):
    if record.name is None:
        add(Severity.ERROR, "PROJECT_NAME_MISSING", "project has no name", record_type, index, "name")


def check_certification(add, record_type, index, record):
    if record.certification is None:
        add(Severity.WARNING, "CERTIFICATION_NAME_MISSING", "certification record has no name", record_type, index)


def check_skills(add, record_type, index, record):
    counts = Counter(v.normalized_value for v in record.skills if v.normalized_value)
    for value, count in counts.items():
        if count > 1:
            add(Severity.WARNING, "SKILL_DUPLICATE", f"skill {value!r} appears {count} times",
                record_type, index, "skills")


CHECKS = {"contact": check_contact, "work_experience": check_work, "education": check_education,
          "projects": check_project, "certifications": check_certification, "skills": check_skills}


def overall_status(issues) -> ValidationStatus:
    severities = {issue.severity for issue in issues}
    if Severity.ERROR in severities:
        return ValidationStatus.INVALID
    return ValidationStatus.VALID_WITH_WARNINGS if Severity.WARNING in severities else ValidationStatus.VALID
