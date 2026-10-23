"""Skill formatting only. Every mapping below is a spelling/casing canonicalization of the
same token. Nothing here may relate two different technologies (no SQL -> SQL Server,
no C# -> .NET, no Web APIs -> ASP.NET Web API); that is a later taxonomy concern."""

from ..records.models import EntityRef
from .models import NormalizedValue
from .text_normalizer import clean_text, make_value, suspicious_warnings

SKILL_FORMATTING = {
    "asp.net core": "ASP.NET Core",
    "asp.net mvc": "ASP.NET MVC",
    "asp.net": "ASP.NET",
    ".net core": ".NET Core",
    ".net": ".NET",
    "vb net": "VB.NET",
    "vb.net": "VB.NET",
    "jquery": "jQuery",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "github": "GitHub",
    "bootstrap": "Bootstrap",
    "sql server": "SQL Server",
    "html": "HTML",
    "css": "CSS",
    "sql": "SQL",
}


def normalize_skill(raw: str, sources: tuple[EntityRef, ...] = ()) -> NormalizedValue:
    text, rules = clean_text(raw)
    key = text.casefold()
    if key in SKILL_FORMATTING and SKILL_FORMATTING[key] != text:
        rules.append(f"skill-mapping/{key}")
        text = SKILL_FORMATTING[key]
    return make_value(raw, text, rules, suspicious_warnings(raw), sources)
