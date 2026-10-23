"""M9 canonical resume -> deterministic semantic chunks, one per resume concept.

Boundaries are semantic (one record = one chunk). Text is never truncated or split by length.
Order: summary, work, projects, education, skills, certifications, awards, languages,
interests, contact; records keep M9 document order.
"""

from ..canonical.models import CanonicalResume
from . import renderers
from .models import ChunkProvenance, ChunkType as T, SemanticChunk

# chunk type -> (id slug, source-record slug)
SLUGS = {T.SUMMARY: ("summary", "summary"), T.WORK_EXPERIENCE: ("work", "work-experience"),
         T.PROJECT: ("project", "project"), T.EDUCATION: ("education", "education"),
         T.SKILLS: ("skills", "skills"), T.CERTIFICATION: ("certification", "certification"),
         T.AWARD: ("award", "award"), T.LANGUAGE: ("language", "language"),
         T.INTEREST: ("interest", "interest"), T.CONTACT: ("contact", "contact")}


def unique(values) -> tuple:
    return tuple(dict.fromkeys(values))


def make_provenance(fields: dict, logical_block_ids=None) -> ChunkProvenance:
    entities = [e for provenance in fields.values() for e in provenance.entities]
    block_ids = (tuple(logical_block_ids) if logical_block_ids is not None
                 else unique(e.logical_block_id for e in entities))
    # Document order: by position of the logical block, then offset within it.
    entities.sort(key=lambda e: (block_ids.index(e.logical_block_id) if e.logical_block_id in block_ids
                                 else len(block_ids), e.text_start))
    return ChunkProvenance(
        logical_block_ids=block_ids,
        page_numbers=tuple(sorted({e.page_number for e in entities})),
        source_block_ids=unique(b for e in entities for b in e.source_block_ids),
        raw_values={key: provenance.raw_value for key, provenance in fields.items()},
        fields=dict(fields))


def compile_chunks(resume: CanonicalResume, include_contact: bool = False) -> list[SemanticChunk]:
    chunks: list[SemanticChunk] = []
    counters: dict = {}

    def add(chunk_type, rendered, canonical_path, index, fields, extra, logical_block_ids=None):
        text, source_fields = rendered
        if not text:
            return  # never emit an empty chunk
        number = counters[chunk_type] = counters.get(chunk_type, 0) + 1
        id_slug, record_slug = SLUGS[chunk_type]
        provenance = make_provenance(fields, logical_block_ids)
        metadata = {"chunk_type": chunk_type.value, "canonical_path": canonical_path,
                    "document_id": resume.metadata.document_id,
                    "source_file": resume.metadata.source_file_name,
                    "page_numbers": list(provenance.page_numbers),
                    "logical_block_ids": list(provenance.logical_block_ids)}
        if index is not None:
            metadata["record_index"] = index
        metadata.update({k: v for k, v in extra.items() if v not in (None, [], "")})
        chunks.append(SemanticChunk(
            chunk_id=f"resume-{id_slug}-{number:03d}", chunk_type=chunk_type, text=text,
            canonical_path=canonical_path, source_record_id=f"{record_slug}-{number:03d}",
            source_fields=source_fields, provenance=provenance, metadata=metadata))

    if resume.summary:
        add(T.SUMMARY, renderers.render_summary(resume.summary), "summary", None,
            {"summary": resume.summary.provenance}, {"raw_summary": resume.summary.raw_value})

    for i, work in enumerate(resume.work_experience):
        raw = {k: v.raw_value for k, v in work.provenance.items()}
        add(T.WORK_EXPERIENCE, renderers.render_work(work), f"work_experience[{i}]", i, work.provenance,
            {"role": work.role, "company": work.company, "start_date": work.start_date,
             "end_date": work.end_date, "current": work.current, "raw_role": raw.get("role"),
             "raw_company": raw.get("company"), "raw_date_range": raw.get("date_range")},
            work.logical_block_ids)

    for i, project in enumerate(resume.projects):
        add(T.PROJECT, renderers.render_project(project), f"projects[{i}]", i, project.provenance,
            {"name": project.name, "technologies": [t.value for t in project.technologies]},
            project.logical_block_ids)

    for i, education in enumerate(resume.education):
        add(T.EDUCATION, renderers.render_education(education), f"education[{i}]", i, education.provenance,
            {"institution": education.institution, "degree": education.degree,
             "field_of_study": education.field_of_study}, education.logical_block_ids)

    if resume.skills:
        add(T.SKILLS, renderers.render_list(resume.skills), "skills", None,
            {f"skills[{i}]": s.provenance for i, s in enumerate(resume.skills)},
            {"skills": [s.value for s in resume.skills], "raw_skills": [s.raw_value for s in resume.skills]})

    for i, cert in enumerate(resume.certifications):
        add(T.CERTIFICATION, renderers.render_certification(cert), f"certifications[{i}]", i, cert.provenance,
            {"name": cert.name, "issuing_organization": cert.issuing_organization,
             "credential_id": cert.credential_id, "credential_url": cert.credential_url},
            cert.logical_block_ids)

    for i, award in enumerate(resume.awards):
        add(T.AWARD, renderers.render_award(award), f"awards[{i}]", i, {"value": award.provenance},
            {"raw_award": award.provenance.raw_value}, award.logical_block_ids)

    if resume.languages:
        add(T.LANGUAGE, renderers.render_list(resume.languages), "languages", None,
            {f"languages[{i}]": x.provenance for i, x in enumerate(resume.languages)},
            {"languages": [x.value for x in resume.languages]})

    if resume.interests:
        add(T.INTEREST, renderers.render_list(resume.interests), "interests", None,
            {f"interests[{i}]": x.provenance for i, x in enumerate(resume.interests)},
            {"interests": [x.value for x in resume.interests]})

    if include_contact:
        candidate = resume.candidate
        add(T.CONTACT, renderers.render_contact(candidate), "candidate", None, candidate.provenance,
            {"email": candidate.email, "phone": candidate.phone})
    return chunks
