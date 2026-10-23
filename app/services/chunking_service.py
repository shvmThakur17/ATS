from app.services.resume_section_service import ResumeSectionService


class ResumeChunkingService:

    @staticmethod
    def create_chunks(text: str):

        sections = ResumeSectionService.extract_sections(text)

        chunks = []

        for section, content in sections.items():

            if not content:
                continue

            chunks.append({
                "section": section,
                "subsection": None,
                "text": content
            })

        return chunks