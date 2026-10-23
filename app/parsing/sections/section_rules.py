"""Small heading vocabulary; matches whole headings, never substrings."""

import re

from .models import SectionType

ALIASES = {
    SectionType.SUMMARY: ("summary", "profile", "professional summary", "about me"),
    SectionType.SKILLS: ("skills", "technical skills", "core skills", "technologies", "technical expertise"),
    SectionType.EXPERIENCE: ("experience", "work experience", "professional experience", "employment history", "career history"),
    SectionType.PROJECTS: ("projects", "selected projects", "personal projects", "key projects"),
    SectionType.EDUCATION: ("education", "academic background", "educational qualifications"),
    SectionType.CERTIFICATIONS: ("certifications", "certificates"),
    SectionType.AWARDS: ("awards", "award", "honors", "achievements", "achievement", "achivement"),
    SectionType.LANGUAGES: ("languages", "spoken languages"),
    SectionType.INTERESTS: ("interests", "hobbies"),
    SectionType.CONTACT: ("contact", "contact information", "contact details"),
    SectionType.OTHER: ("additional information", "other information"),
}


def heading_type(text):
    normalized = re.sub(r"^\s*\d{1,2}[.)]\s*", "", text)
    normalized = " ".join(normalized.strip().rstrip(":").lower().split())
    for section, aliases in ALIASES.items():
        if normalized in aliases:
            return section
    return SectionType.UNKNOWN
