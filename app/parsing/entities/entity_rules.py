"""Visible section-specific rules. Values and dates are never normalized."""

import re

from ..sections.models import SectionType as S
from .models import EntityType as E

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b")
PHONE = re.compile(r"(?<!\w)\+?\d[\d ().-]{6,}\d(?!\w)")
URL = re.compile(r"(?:https?://|www\.)[^\s<>]+|(?:linkedin\.com|github\.com)/[^\s<>]+", re.I)
MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_PART = rf"(?:{MONTH}\s+(?:19|20)\d{{2}}|(?:0?[1-9]|1[0-2])/(?:19|20)\d{{2}}|(?:19|20)\d{{2}}|Present)"
DATE_RANGE = re.compile(rf"{DATE_PART}\s*(?:[-–—]|to)\s*{DATE_PART}", re.I)
DATE = re.compile(DATE_PART, re.I)
BULLET = re.compile(r"^\s*(?:[•·●▪◦*-]|\d+[.)])\s+")
DEGREE = re.compile(r"(?P<degree>(?:Bachelor|Master|Doctor) of [A-Za-z]+|B\.?Tech\.?|M\.?Tech\.?|B\.?Sc\.?|M\.?Sc\.?|MBA|Ph\.?D\.?|Diploma|10th|12th)(?:\s+in\s+(?P<field>.+))?", re.I)
ROLE = re.compile(r"\b(?:Engineer|Developer|Manager|Analyst|Architect|Consultant|Designer|Intern|Director)\b", re.I)
COMPANY = re.compile(r"\b(?:Technologies|Solutions|Ltd\.?|Limited|Inc\.?|LLC|Corporation|Corp\.?)$", re.I)
INSTITUTION = re.compile(r"\b(?:University|College|Institute|School|Polytechnic)\b", re.I)
SPOKEN_LANGUAGES = {"english", "hindi", "punjabi", "french", "spanish", "german", "arabic", "mandarin", "tamil", "telugu", "bengali", "marathi", "urdu", "japanese", "korean", "portuguese", "italian"}
PROSE_START = re.compile(r"^(?:Strong|Experienced|Skilled|Worked|Built|Developed|Created|Maintained|Handled|Managed|Implemented|Delivered|Supported|Projects included|Technology enthusiast|I\b|We\b)", re.I)
MAX_ITEM_WORDS = 6
MAX_TITLE_WORDS = 10


def name_shaped(text):
    """A name candidate still needs first-block typography and role context."""
    words = text.split()
    return (2 <= len(words) <= 4 and not ROLE.search(text)
            and not COMPANY.search(text) and not INSTITUTION.search(text)
            and not PROSE_START.match(text)
            and not {word.lower() for word in words} & {"resume", "curriculum", "vitae", "software", "profile"}
            and all(word[0].isupper() and all(c.isalpha() or c in "'-’" for c in word)
                    for word in words))


def contact_location(text):
    """Require address syntax rather than treating arbitrary contact text as a city."""
    return (len(text.split()) <= MAX_TITLE_WORDS and not PROSE_START.match(text)
            and not ROLE.search(text) and not contact_matches(text)
            and bool(re.fullmatch(r"[A-Z][A-Za-z .'-]+,\s*[A-Z][A-Za-z .'-]+", text)
                     or re.fullmatch(r"[A-Za-z0-9 .,'#/-]+", text)
                     and re.search(r"\b(?:sector|street|road|avenue|lane|block|apartment)\b", text, re.I)
                     and (re.match(r"\d+\s+[A-Za-z]", text)
                          or re.search(r"\S\s+(?:sector|block|apartment)\s+\d+[A-Za-z]?\b", text, re.I))))

LABELS = {
    S.CONTACT: {"name": E.NAME, "email": E.EMAIL, "phone": E.PHONE, "mobile": E.PHONE,
                "location": E.LOCATION, "address": E.LOCATION, "linkedin": E.LINKEDIN,
                "github": E.GITHUB, "website": E.WEBSITE},
    S.EXPERIENCE: {"company": E.COMPANY, "employer": E.COMPANY, "role": E.ROLE,
                   "job title": E.ROLE, "location": E.LOCATION, "description": E.DESCRIPTION},
    S.EDUCATION: {"institution": E.INSTITUTION, "degree": E.DEGREE,
                  "field of study": E.FIELD_OF_STUDY, "location": E.LOCATION},
    S.PROJECTS: {"project": E.PROJECT_NAME, "project name": E.PROJECT_NAME,
                 "description": E.PROJECT_DESCRIPTION},
    S.CERTIFICATIONS: {"certification": E.CERTIFICATION, "issuer": E.ISSUING_ORGANIZATION,
                       "issuing organization": E.ISSUING_ORGANIZATION,
                       "credential id": E.CREDENTIAL_ID, "credential url": E.CREDENTIAL_URL},
    S.AWARDS: {"award": E.AWARD, "issuer": E.ISSUING_ORGANIZATION,
               "issuing organization": E.ISSUING_ORGANIZATION, "description": E.DESCRIPTION},
}
for section in (S.EXPERIENCE, S.EDUCATION, S.CERTIFICATIONS, S.AWARDS):
    LABELS[section].update({"date": E.DATE, "dates": E.DATE_RANGE,
                            "date range": E.DATE_RANGE, "graduation year": E.DATE})


def is_short_entry(text):
    return bool(text) and len(text.split()) <= MAX_ITEM_WORDS and not PROSE_START.match(text) and not text.endswith((".", "!", "?"))


def split_entries(text):
    """Yield original substrings with offsets; split conjunctions only in short lists."""
    separator = r"[,|;•·\n]+"
    if is_short_entry(text) and " and " in text:
        separator += r"|\s+and\s+"
    start = 0
    for match in list(re.finditer(separator, text)) + [None]:
        end = match.start() if match else len(text)
        part = text[start:end]
        marker = BULLET.match(part)
        offset = marker.end() if marker else len(part) - len(part.lstrip())
        value = part[offset:].rstrip()
        if is_short_entry(value):
            yield start + offset, start + offset + len(value)
        start = match.end() if match else end


def url_kind(value):
    host = re.sub(r"^https?://", "", value, flags=re.I).split("/", 1)[0].lower()
    host = host.removeprefix("www.")
    if host == "linkedin.com":
        return E.LINKEDIN
    if host == "github.com":
        return E.GITHUB
    return E.WEBSITE


def contact_matches(text):
    matches = [(E.EMAIL, m.start(), m.end(), "email-pattern") for m in EMAIL.finditer(text)]
    for match in URL.finditer(text):
        value = match.group().rstrip(".,;)")
        matches.append((url_kind(value), match.start(), match.start() + len(value), "url-pattern"))
    for match in PHONE.finditer(text):
        value = match.group()
        digit_count = sum(char.isdigit() for char in value)
        if 10 <= digit_count <= 15 and not DATE_RANGE.fullmatch(value):
            if not any(start < match.end() and end > match.start() for _, start, end, _ in matches):
                matches.append((E.PHONE, match.start(), match.end(), "phone-pattern"))
    return matches


def line_matches(section, text, following=""):
    """Extract one line with explicit labels or conservative whole-line cues."""
    label, colon, value = text.partition(":")
    if colon:
        offset = len(label) + 1 + len(value) - len(value.lstrip())
        value = value.strip()
        kind = LABELS.get(section, {}).get(label.strip().lower())
        if kind and value:
            if kind in (E.DATE, E.DATE_RANGE):
                if DATE_RANGE.fullmatch(value):
                    yield E.DATE_RANGE, offset, offset + len(value), "explicit-date-label"
                elif kind == E.DATE and DATE.fullmatch(value):
                    yield E.DATE, offset, offset + len(value), "explicit-date-label"
            elif kind in (E.EMAIL, E.PHONE, E.LINKEDIN, E.GITHUB, E.WEBSITE, E.CREDENTIAL_URL):
                matches = contact_matches(value)
                for detected, start, end, rule in matches:
                    if detected == kind or kind == E.CREDENTIAL_URL and detected in (E.WEBSITE, E.LINKEDIN, E.GITHUB):
                        yield kind, offset + start, offset + end, "explicit-label-" + rule
            else:
                yield kind, offset, offset + len(value), "explicit-label"
            return
        if section == S.PROJECTS and label.lower().strip() in ("technologies", "technology", "tech stack"):
            for start, end in split_entries(value):
                yield E.TECHNOLOGY, offset + start, offset + end, "technology-list-label"
            return
    if section == S.CONTACT:
        yield from contact_matches(text)
        if contact_location(text):
            yield E.LOCATION, 0, len(text), "contact-address-shape"
        return
    if section in (S.SKILLS, S.LANGUAGES, S.INTERESTS):
        kind = {S.SKILLS: E.SKILL, S.LANGUAGES: E.LANGUAGE, S.INTERESTS: E.INTEREST}[section]
        for start, end in split_entries(text):
            if section != S.LANGUAGES or text[start:end].lower() in SPOKEN_LANGUAGES:
                yield kind, start, end, "section-list-entry"
        return
    if section not in (S.EXPERIENCE, S.EDUCATION, S.PROJECTS, S.CERTIFICATIONS, S.AWARDS):
        return
    if DATE_RANGE.fullmatch(text):
        yield E.DATE_RANGE, 0, len(text), "whole-line-date-range"
        return
    if DATE.fullmatch(text):
        yield E.DATE, 0, len(text), "whole-line-date"
        return
    short = len(text.split()) <= MAX_TITLE_WORDS and not PROSE_START.match(text) and not text.endswith((".", "!", "?"))
    if section == S.EXPERIENCE:
        date = DATE_RANGE.search(text)
        if date and date.end() == len(text):
            title = text[:date.start()].rstrip()
            if title and len(title.split()) <= MAX_TITLE_WORDS and ROLE.search(title) and not PROSE_START.match(title):
                yield E.ROLE, 0, len(title), "role-followed-by-date-range"
                yield E.DATE_RANGE, date.start(), date.end(), "role-followed-by-date-range"
                return
        if short and ROLE.search(text):
            yield E.ROLE, 0, len(text), "short-role-title"
        elif len(text.split()) <= MAX_TITLE_WORDS and not PROSE_START.match(text) and COMPANY.search(text):
            yield E.COMPANY, 0, len(text), "organization-suffix"
        elif short and re.fullmatch(r"[A-Z][A-Za-z .'-]+,\s*[A-Z][A-Za-z .'-]+", text):
            yield E.LOCATION, 0, len(text), "city-country-shaped-line"
        elif text.endswith((".", "!")) or BULLET.match(text):
            yield E.DESCRIPTION, 0, len(text), "experience-description-line"
    elif section == S.EDUCATION:
        degree = DEGREE.fullmatch(text)
        if degree:
            yield E.DEGREE, degree.start("degree"), degree.end("degree"), "degree-phrase"
            if degree.group("field"):
                yield E.FIELD_OF_STUDY, degree.start("field"), degree.end("field"), "degree-field-label"
        elif short and INSTITUTION.search(text):
            yield E.INSTITUTION, 0, len(text), "institution-designator"
        elif short and re.fullmatch(r"[A-Z][A-Za-z .'-]+,\s*[A-Z][A-Za-z .'-]+", text):
            yield E.LOCATION, 0, len(text), "city-country-shaped-line"
    elif section == S.PROJECTS:
        if PROSE_START.match(text) or text.endswith("."):
            yield E.PROJECT_DESCRIPTION, 0, len(text), "project-description-line"
        elif short and following and (PROSE_START.match(following) or following.endswith(".")):
            if all(word[:1].isupper() for word in text.split()):
                yield E.PROJECT_NAME, 0, len(text), "title-followed-by-description"
    elif section == S.CERTIFICATIONS:
        if short and re.search(r"\b(?:Certified|Certification|Certificate)\b", text, re.I):
            yield E.CERTIFICATION, 0, len(text), "certification-name-cue"
    elif section == S.AWARDS:
        if short and re.search(r"\b(?:Award|Medal|Prize)\b", text, re.I):
            yield E.AWARD, 0, len(text), "award-name-cue"
        elif text.endswith("."):
            yield E.DESCRIPTION, 0, len(text), "award-description-line"
