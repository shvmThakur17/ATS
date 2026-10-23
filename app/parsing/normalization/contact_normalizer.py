"""Contact formatting only. No domains are corrected, no country codes are assumed."""

import re

from ..records.models import EntityRef
from .models import NormalizationStatus, NormalizedValue
from .text_normalizer import clean_text, make_value

EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s.]+")
PHONE_FORMATTING = re.compile(r"[\s\-().]")
PHONE = re.compile(r"\+?\d+")
MIN_PHONE_DIGITS, MAX_PHONE_DIGITS = 7, 15  # E.164 upper bound
URL = re.compile(r"(?:(?P<scheme>https?)://)?(?P<host>[^/\s?#]+)(?P<rest>[/?#]\S*)?", re.IGNORECASE)
HOST = re.compile(r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}")


def invalid(raw, code, message, rules, sources):
    return make_value(raw, None, rules, [f"{code}: {message}"], sources, NormalizationStatus.INVALID)


def normalize_email(raw: str, sources: tuple[EntityRef, ...] = ()) -> NormalizedValue:
    text, rules = clean_text(raw)
    if not EMAIL.fullmatch(text):
        return invalid(raw, "invalid-email", f"{text!r} is not a valid email shape", rules, sources)
    if text.lower() != text:
        rules.append("email/lowercase")
    return make_value(raw, text.lower(), rules, (), sources)


def normalize_phone(raw: str, sources: tuple[EntityRef, ...] = ()) -> NormalizedValue:
    text, rules = clean_text(raw)
    compact = PHONE_FORMATTING.sub("", text)
    digits = compact.lstrip("+")
    if not PHONE.fullmatch(compact) or not MIN_PHONE_DIGITS <= len(digits) <= MAX_PHONE_DIGITS:
        return invalid(raw, "invalid-phone",
                       f"{text!r} is not {MIN_PHONE_DIGITS}-{MAX_PHONE_DIGITS} digits with an optional leading '+'",
                       rules, sources)
    if compact != text:
        rules.append("phone/strip-formatting")
    return make_value(raw, compact, rules, (), sources)


def normalize_url(raw: str, sources: tuple[EntityRef, ...] = ()) -> NormalizedValue:
    text, rules = clean_text(raw)
    match = URL.fullmatch(text)
    if not match or not HOST.fullmatch(match["host"]):
        return make_value(raw, None, rules, [f"unrecognized-url: {text!r} is not an unambiguous URL"],
                          sources, NormalizationStatus.UNKNOWN)
    scheme = (match["scheme"] or "https").lower()
    host = match["host"].lower()
    normalized = f"{scheme}://{host}{match['rest'] or ''}"
    if not match["scheme"]:
        rules.append("url/add-https-scheme")
    elif normalized != text:
        rules.append("url/lowercase-scheme-and-host")
    return make_value(raw, normalized, rules, (), sources)
