"""Deterministic JD parsing: headings and keywords only, no LLM.

Every line of the JD ends up somewhere (a structured field or `unclassified_text`); the raw text is kept.
Skill terminology is normalized with the M8 skill normalizer so JD and resume use the same spelling.
There is no technology dictionary here: skills come from (a) list-like items, (b) phrases after
"experience with / knowledge of ...", and (c) the small M8 vocabulary for sentence-style items.
"""

import hashlib
import re

from ..parsing.normalization.skill_normalizer import SKILL_FORMATTING, normalize_skill
from .models import (EmptyJobDescriptionError, ExperienceRequirement, JDSkill, JobDescription, MissingJDSectionsError,
                     RequirementLevel as Level)

PREFERRED, REQUIRED, RESPONSIBILITIES, QUALIFICATIONS, EXPERIENCE, EDUCATION, OTHER = (
    "preferred", "required", "responsibilities", "qualifications", "experience", "education", "other")

# Checked in order; the first full-line match wins.
HEADINGS = (
    (PREFERRED, r"(?:preferred|desired|desirable|nice to have|good to have|bonus)(?: (?:skills|qualifications|requirements|experience|points))?"
                r"|nice[- ]to[- ]haves?|pluses|bonus points"),
    (REQUIRED, r"(?:required|mandatory|minimum|essential|key)(?: (?:skills|qualifications|requirements|experience))?"
               r"|requirements|must[- ]haves?|must have|what you(?:'|’)?ll need|what we(?:'|’)?re looking for"
               r"|what you bring|skills(?: required| needed)?|technical skills|tech stack"),
    (RESPONSIBILITIES, r"(?:key |core |primary |job )?responsibilities|what you(?:'|’)?ll do|duties|your role|the role"
                       r"|role (?:and|&) responsibilities"),
    (QUALIFICATIONS, r"qualifications"),
    (EXPERIENCE, r"(?:work |professional )?experience(?: required| requirements)?"),
    (EDUCATION, r"education(?:al)?(?: requirements| qualifications| background)?|academic (?:requirements|qualifications)"),
)
HEADING_PATTERNS = [(kind, re.compile(rf"(?:{pattern})", re.IGNORECASE)) for kind, pattern in HEADINGS]
BULLET = re.compile(r"^\s*(?:[-*•·▪●◦–—]|\d+[.)])\s+")
YEARS = re.compile(r"(\d{1,2})(?:\s*(?:\+|plus))?(?:\s*(?:-|–|to)\s*\d{1,2})?\s*(?:years?|yrs?)\b", re.IGNORECASE)
EDUCATION_WORDS = re.compile(r"\b(?:degree|bachelor'?s?|master'?s?|b\.?\s?tech|m\.?\s?tech|b\.?sc|m\.?sc|diploma|ph\.?d|"
                             r"doctorate|graduate|university|b\.?e\.?|bca|mca|mba)\b", re.IGNORECASE)
TRIGGER = re.compile(r"(?:hands[- ]on )?(?:experience (?:with|in|using)|proficien\w* (?:in|with)|knowledge of|"
                     r"familiarity with|expertise in|skilled in|working with)\s+(.+?)(?:[.;]|$)", re.IGNORECASE)
LEADING_FILLER = re.compile(r"^(?:the|a|an|strong|good|solid|working|basic|deep|and|or)\s+", re.IGNORECASE)
MAX_SKILL_WORDS = 4
SOFT_SKILL_HINTS = ("communication", "teamwork", "team player", "problem solving", "problem-solving", "leadership",
                    "collaborat", "self-motivated", "attention to detail", "willing", "passion")


def is_heading(line: str):
    """Return (kind, inline_content) when the line is a heading, else None. Unknown headings are OTHER."""
    text = re.sub(r"^#+\s*", "", line.strip()).strip("*_ ")
    head, colon, rest = text.partition(":")
    candidate = (head if colon else text).strip()
    if not candidate or len(candidate.split()) > 6 or BULLET.match(line):
        return None
    for kind, pattern in HEADING_PATTERNS:
        if pattern.fullmatch(candidate):
            return kind, rest.strip()
    if colon and not rest.strip():  # "Some other heading:" -> keep its lines, as unclassified
        return OTHER, ""
    return None


def looks_like_unknown_heading(lines: list[str], index: int) -> bool:
    """A short un-bulleted line after a blank line, directly followed by a bullet, starts a new section
    (e.g. "Benefits"). Anything less clear-cut is left as ordinary content."""
    line = lines[index].strip()
    following = next((l for l in lines[index + 1:] if l.strip()), "")
    return (bool(line) and not BULLET.match(line) and len(line.split()) <= 4 and line[0].isupper()
            and not line.endswith((".", ",", ";", "!")) and (index == 0 or not lines[index - 1].strip())
            and bool(BULLET.match(following)))


def clean_phrase(phrase: str) -> str:
    phrase = re.sub(r"\(.*?\)", "", phrase)
    phrase = re.sub(r"\b(?:etc\.?|and more|or similar|or equivalent)\b", "", phrase, flags=re.IGNORECASE)
    phrase = LEADING_FILLER.sub("", phrase.strip(" ,;:-–—*•"))
    return LEADING_FILLER.sub("", phrase).strip(" ,;:-–—").rstrip(".")  # keep a leading dot: ".NET"


def split_list(text: str) -> list[str]:
    return [clean_phrase(part) for part in re.split(r",|;|/|\band\b|\bor\b|&", text) if clean_phrase(part)]


def plausible_skill(phrase: str) -> bool:
    return (0 < len(phrase.split()) <= MAX_SKILL_WORDS and not YEARS.search(phrase)
            and not phrase.lower().startswith(("able to", "ability", "strong", "excellent", "good "))
            and not any(hint in phrase.lower() for hint in SOFT_SKILL_HINTS))


def vocabulary_matches(text: str) -> list[str]:
    """M8-vocabulary terms in the text (longest first, non-overlapping, whole-term only)."""
    spans, found = [], []
    for key in sorted(SKILL_FORMATTING, key=len, reverse=True):
        for match in re.finditer(rf"(?<![\w.#+]){re.escape(key)}(?![\w#+])", text, re.IGNORECASE):
            if not any(match.start() < end and start < match.end() for start, end in spans):
                spans.append((match.start(), match.end()))
                found.append((match.start(), text[match.start():match.end()]))
    return [value for _, value in sorted(found)]


def extract_skill_phrases(item: str) -> list[str]:
    """Raw skill phrases in one JD item, in order of appearance, without duplicates."""
    phrases = []
    sentence_like = len(item.split()) > 6 or item.rstrip().endswith((".", "!"))
    if not sentence_like:  # a list-like line such as "C#, ASP.NET Core" or "Azure and Docker"
        phrases += [p for p in split_list(item) if plausible_skill(p)]
    else:
        for match in TRIGGER.finditer(item):
            phrases += [p for p in split_list(match.group(1)) if plausible_skill(p)]
    phrases += vocabulary_matches(item)
    seen, unique = set(), []
    for phrase in phrases:
        key = normalize_skill(phrase).normalized_value
        if key and key.casefold() not in seen:
            seen.add(key.casefold())
            unique.append(phrase)
    return unique


def logical_items(lines: list[str]) -> list[str]:
    """Bullets become items; an un-bulleted lowercase line continues the previous item."""
    items = []
    for line in lines:
        text = BULLET.sub("", line).strip()
        if not text:
            continue
        if items and not BULLET.match(line) and text[0].islower():
            items[-1] += " " + text
        else:
            items.append(text)
    return items


def parse_job_description(raw_text: str, job_id: str | None = None, title: str | None = None) -> JobDescription:
    if raw_text is None or not raw_text.strip():
        raise EmptyJobDescriptionError("job description is empty")
    lines = [line.rstrip() for line in raw_text.splitlines()]
    sections: list[tuple[str, list[str]]] = [(OTHER, [])]
    detected_title = None
    first_line = next((i for i, line in enumerate(lines) if line.strip()), None)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        heading = is_heading(line) or ((OTHER, "") if looks_like_unknown_heading(lines, index) and index != first_line
                                       else None)
        if heading:
            kind, inline = heading
            # an unrecognized heading is itself kept (as unclassified text)
            sections.append((kind, [line.strip()] if kind == OTHER else ([inline] if inline else [])))
        elif detected_title is None and index == next(i for i, l in enumerate(lines) if l.strip()) \
                and len(sections) == 1 and not BULLET.match(line) and len(line.split()) <= 12:
            detected_title = re.sub(r"^(?:job )?title\s*:\s*", "", line.strip(), flags=re.IGNORECASE)
        else:
            sections[-1][1].append(line)

    required, preferred, responsibilities, qualifications, unclassified = [], [], [], [], []
    experience, education = [], []

    def add_skills(phrases, level):
        bucket = required if level == Level.REQUIRED else preferred
        for phrase in phrases:
            normalized = normalize_skill(phrase).normalized_value
            if normalized and all(s.normalized.casefold() != normalized.casefold() for s in required + preferred):
                bucket.append(JDSkill(raw=phrase, normalized=normalized, requirement=level))

    for kind, section_lines in sections:
        for item in logical_items(section_lines):
            if kind == RESPONSIBILITIES:
                responsibilities.append(item)
            elif kind == OTHER:
                unclassified.append(item)
            else:
                level = Level.PREFERRED if kind == PREFERRED else Level.REQUIRED
                skills = extract_skill_phrases(item)
                if kind == EXPERIENCE or YEARS.search(item):
                    experience.append((item, level))
                    add_skills(skills, level)
                elif kind == EDUCATION or EDUCATION_WORDS.search(item):
                    education.append(item)
                else:
                    add_skills(skills, level)
                    if kind == QUALIFICATIONS or not skills:
                        qualifications.append(item)

    jd_skills = required + preferred
    requirements = []
    for item, _ in experience:
        years = YEARS.search(item)
        mentioned = tuple(s.normalized for s in jd_skills if re.search(
            rf"(?<![\w.#+]){re.escape(s.normalized)}(?![\w#+])", item, re.IGNORECASE))
        requirements.append(ExperienceRequirement(text=item, min_years=int(years.group(1)) if years else None,
                                                  mentions_skills=mentioned))
    if not required and not responsibilities:
        raise MissingJDSectionsError("job description has no recognizable required skills or responsibilities")
    warnings = []
    if not required:
        warnings.append("no required skills were recognized in the job description")
    return JobDescription(
        job_id=job_id or "jd-" + hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:12],
        title=title or detected_title, raw_text=raw_text, required_skills=tuple(required),
        preferred_skills=tuple(preferred), responsibilities=tuple(responsibilities),
        qualifications=tuple(qualifications), experience_requirements=tuple(requirements),
        education_requirements=tuple(education), unclassified_text=tuple(unclassified), warnings=tuple(warnings))
