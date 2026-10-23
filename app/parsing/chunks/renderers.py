"""Chunk text rendering. Each renderer returns (text, source_fields) from M9 values only.

Lines for missing values are omitted, never filled in. Values are used exactly as M9 has them.
"""


def labeled(pairs) -> tuple[str, tuple[str, ...]]:
    """pairs: (label, value, field) -> "Label: value" lines and the M9 fields that were used."""
    used = [(label, value, field) for label, value, field in pairs if value]
    return "\n".join(f"{label}: {value}" for label, value, _ in used), tuple(f for _, _, f in used)


def render_summary(summary):
    return summary.value, ("summary",)


def render_work(work):
    if work.start_date and work.end_date:
        employment, dates = f"{work.start_date} to {work.end_date}", ("start_date", "end_date")
    elif work.start_date and work.current:
        employment, dates = f"{work.start_date} to Present", ("start_date", "current")
    else:
        employment, dates = None, ()  # M9 mapped no usable range; do not invent one
    text, fields = labeled((("Role", work.role, "role"), ("Company", work.company, "company"),
                            ("Employment", employment, "employment"),
                            ("Description", work.description, "description")))
    return text, tuple(item for field in fields for item in (dates if field == "employment" else (field,)))


def render_project(project):
    technologies = ", ".join(t.value for t in project.technologies)
    return labeled((("Project", project.name, "name"), ("Description", project.description, "description"),
                    ("Technologies", technologies, "technologies")))


def render_education(education):
    return labeled((("Degree", education.degree, "degree"),
                    ("Field of Study", education.field_of_study, "field_of_study"),
                    ("Institution", education.institution, "institution")))


def render_certification(certification):
    # Credential ID / URL are identifiers, not semantics: they go to metadata only.
    return labeled((("Certification", certification.name, "name"),
                    ("Issuer", certification.issuing_organization, "issuing_organization")))


def render_award(award):
    return award.value, ("value",)


def render_list(items):
    return ", ".join(item.value for item in items), ("value",)


def render_contact(candidate):
    # Email and phone are deliberately excluded from the embedded text.
    return labeled((("Name", candidate.name, "name"), ("Location", candidate.location, "location")))
