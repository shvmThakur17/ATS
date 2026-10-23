"""M6 context and contiguous-text regressions; fixtures are not resume-specific."""

from dataclasses import asdict, replace

import pytest

from app.parsing.entities import EntityType as E, extract_entities
from app.parsing.models import Evidence
from app.parsing.sections import SectionBlock, SectionDocumentIR, SectionType as S
from tests.parsing.sections.test_sections import fixture
from tests.parsing.entities.test_entities import inputs


def values(section, text, kind):
    return [e.raw_value for e in extract_entities(*inputs(section, text)).entities
            if e.entity_type == kind]


def header_inputs(name, *, styled=True, later=False, second_page=False):
    entries = [(name, styled, 330, 250), ("Senior Developer", True, 330, 280)]
    if later:
        entries.insert(0, ("Other content", False, 330, 200))
    logical, text = fixture(entries[:1], entries[1:]) if second_page else fixture(entries)
    assignments = tuple(SectionBlock(b.logical_block_id, b.page_number, S.UNKNOWN, None,
                                    False, Evidence(.7, ("fixture",), ("Original evidence",)))
                        for page in logical.pages for b in page.logical_blocks)
    sections = SectionDocumentIR(logical.document_id, "sections-ir/1", logical.stage_version,
                                 logical.config_fingerprint, "sections/1", assignments)
    return sections, logical, text


@pytest.mark.parametrize("name", ["ANITA SHARMA", "Mary O'Connor", "Élodie Martin"])
def test_leading_styled_name_with_role_context(name):
    result = extract_entities(*header_inputs(name))
    assert [(e.entity_type, e.raw_value) for e in result.entities] == [
        (E.NAME, name), (E.UNKNOWN, "Senior Developer")]


@pytest.mark.parametrize("name,options", [
    ("Senior Engineer", {}), ("ABC Technologies", {}), ("CURRICULUM VITAE", {}),
    ("Anita Sharma", {"styled": False}), ("Anita Sharma", {"later": True}),
    ("Anita Sharma", {"second_page": True}), ("Technical Skills", {}),
])
def test_name_requires_conservative_document_context(name, options):
    assert not any(e.entity_type == E.NAME for e in extract_entities(*header_inputs(name, **options)).entities)


@pytest.mark.parametrize("address", ["Chandigarh Sector 22", "12 Oak Street", "Pune, India"])
def test_unlabeled_contact_address(address):
    assert values(S.CONTACT, address, E.LOCATION) == [address]
    assert values(S.UNKNOWN, address, E.LOCATION) == []


@pytest.mark.parametrize("text", ["Anita Sharma", "Senior Developer", "Call me tomorrow",
                                  "9876543210", "a@example.com", "Street performance improved by 12"])
def test_contact_location_negatives(text):
    assert values(S.CONTACT, text, E.LOCATION) == []


@pytest.mark.parametrize("text,degree,field", [
    ("10th", "10th", None), ("12th", "12th", None),
    ("DIPLOMA IN ELECTRONICS", "DIPLOMA", "ELECTRONICS"),
    ("Diploma", "Diploma", None),
])
def test_education_degree_variants(text, degree, field):
    assert values(S.EDUCATION, text, E.DEGREE) == [degree]
    assert values(S.EDUCATION, text, E.FIELD_OF_STUDY) == ([field] if field else [])
    assert values(S.UNKNOWN, text, E.DEGREE) == []


@pytest.mark.parametrize("section,text,kind", [
    (S.SUMMARY, "  Developer with experience\r\n  building reliable services.  ", E.DESCRIPTION),
    (S.AWARDS, "Awarded Employee of\nthe Year at Example\nCompany — March 2024", E.AWARD),
    (S.PROJECTS, "Maintained service performance and\nresolved customer issues.", E.PROJECT_DESCRIPTION),
    (S.PROJECTS, "Built a service with\nmultiple components", E.PROJECT_DESCRIPTION),
])
def test_full_block_narrative_and_provenance(section, text, kind):
    args = inputs(section, text)
    before = (asdict(args[0]), asdict(args[1]), dict(args[2]))
    result = extract_entities(*args)
    assert [(e.entity_type, e.raw_value) for e in result.entities] == [(kind, text.strip())]
    entity = result.entities[0]
    block = args[1].pages[0].logical_blocks[0]
    assert entity.raw_value == text[entity.text_start:entity.text_end]
    assert entity.sources is block.sources
    assert entity.source_block_ids == block.source_block_ids
    assert entity.logical_block_id == block.logical_block_id
    assert entity.page_number == block.page_number
    assert entity.section_type == section
    assert entity.evidence.score == .8
    assert entity.evidence.rules and entity.evidence.reasons
    assert result.to_json() == extract_entities(*args).to_json()
    assert before == (asdict(args[0]), asdict(args[1]), args[2])


@pytest.mark.parametrize("heading", ["EDUCATION", "Work Experience:", "02. TECHNICAL SKILLS"])
def test_trailing_heading_removed_from_mixed_project_only(heading):
    body = "Maintained an application and\r\nprovided technical support."
    text = "  " + body + "\r\n  " + heading + "  "
    result = extract_entities(*inputs(S.PROJECTS, text))
    assert [(e.entity_type, e.raw_value) for e in result.entities] == [(E.PROJECT_DESCRIPTION, body)]
    e = result.entities[0]
    assert text[e.text_start:e.text_end] == body
    assert values(S.SUMMARY, text, E.DESCRIPTION) == [text.strip()]


def test_heading_words_inside_prose_are_retained():
    text = "Built an EDUCATION portal.\nSupported work experience tracking."
    assert values(S.PROJECTS, text, E.PROJECT_DESCRIPTION) == [text]


def test_project_title_body_and_technologies_remain_distinct():
    text = "Customer Portal\nMaintained the portal and\nresolved customer issues.\nTechnologies: Python, SQL"
    result = extract_entities(*inputs(S.PROJECTS, text))
    assert [(e.entity_type, e.raw_value) for e in result.entities] == [
        (E.PROJECT_NAME, "Customer Portal"),
        (E.PROJECT_DESCRIPTION, "Maintained the portal and\nresolved customer issues."),
        (E.TECHNOLOGY, "Python"), (E.TECHNOLOGY, "SQL")]


def test_award_narrative_stops_at_metadata():
    text = "Won a service award\nfor customer support\nIssuer: Example Foundation\nDate: March 2024"
    result = extract_entities(*inputs(S.AWARDS, text))
    assert [(e.entity_type, e.raw_value) for e in result.entities] == [
        (E.AWARD, "Won a service award\nfor customer support"),
        (E.ISSUING_ORGANIZATION, "Example Foundation"), (E.DATE, "March 2024")]


def test_name_keeps_unknown_section_and_confidence():
    args = header_inputs("ANITA SHARMA")
    e = extract_entities(*args).entities[0]
    assert e.section_type == S.UNKNOWN
    assert e.evidence.score == .7
    assert e.sources is args[1].pages[0].logical_blocks[0].sources


def test_m5_headings_still_skipped():
    args = inputs(S.SUMMARY, "PROFILE")
    sections = replace(args[0], blocks=(replace(args[0].blocks[0], is_heading=True),))
    assert extract_entities(sections, args[1], args[2]).entities == ()
