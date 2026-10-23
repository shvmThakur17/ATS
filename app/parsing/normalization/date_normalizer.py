"""Deterministic date / date-range parsing. No clock is used, so results never change over time."""

import re

from ..records.models import EntityRef
from .models import NormalizationStatus as Status, NormalizedDateRange
from .text_normalizer import clean_text

MONTHS = {name: number for number, names in enumerate(
    (("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",),
     ("jun", "june"), ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"),
     ("oct", "october"), ("nov", "november"), ("dec", "december")), 1) for name in names}
MIN_YEAR, MAX_YEAR = 1900, 2100
ARTIFACT_SEPARATORS = "ûÛ�"

TOKEN = r"(?:[A-Za-z]{3,9}\.?\s+\d{4}|\d{1,2}/\d{4}|\d{4}-\d{1,2}|\d{4}|present|current)"
SEPARATOR = rf"(?:[-–—]|to|[{ARTIFACT_SEPARATORS}])"
RANGE = re.compile(rf"(?P<a>{TOKEN})\s*(?P<sep>{SEPARATOR})\s*(?P<b>{TOKEN})", re.IGNORECASE)
SINGLE = re.compile(TOKEN, re.IGNORECASE)
MONTH_NAME_YEAR = re.compile(r"([A-Za-z]{3,9})\.?\s+(\d{4})")
SLASH = re.compile(r"(\d{1,2})/(\d{4})")
ISO = re.compile(r"(\d{4})-(\d{1,2})")


class DateProblem(Exception):
    def __init__(self, code, message, status):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def check_year(year: int) -> int:
    if not MIN_YEAR <= year <= MAX_YEAR:
        raise DateProblem("invalid-year", f"year {year} is outside {MIN_YEAR}-{MAX_YEAR}", Status.INVALID)
    return year


def check_month(month: int) -> int:
    if not 1 <= month <= 12:
        raise DateProblem("invalid-month", f"month {month} is not between 1 and 12", Status.INVALID)
    return month


def parse_token(token: str):
    """Return (year, month, is_present, rule)."""
    if token.lower() in ("present", "current"):
        return None, None, True, "date/present"
    if match := MONTH_NAME_YEAR.fullmatch(token):
        month = MONTHS.get(match[1].lower())
        if month is None:
            raise DateProblem("unrecognized-month-name", f"{match[1]!r} is not a known month name", Status.UNKNOWN)
        return check_year(int(match[2])), month, False, "date/month-name-year"
    if match := SLASH.fullmatch(token):
        return check_year(int(match[2])), check_month(int(match[1])), False, "date/numeric-month-year"
    if match := ISO.fullmatch(token):
        return check_year(int(match[1])), check_month(int(match[2])), False, "date/iso-year-month"
    return check_year(int(token)), None, False, "date/year"


def fmt(year, month) -> str:
    return f"{year:04d}-{month:02d}" if month else f"{year:04d}"


def ends_before_start(start_year, start_month, end_year, end_month) -> bool:
    if end_year < start_year:
        return True
    return end_year == start_year and bool(start_month and end_month) and end_month < start_month


def normalize_date_range(raw: str, sources: tuple[EntityRef, ...] = ()) -> NormalizedDateRange:
    text, rules = clean_text(raw)
    rules, warnings = list(rules), []
    fields = dict(raw_value=raw, sources=tuple(sources))

    def fail(status, code, message, **extra):
        return NormalizedDateRange(normalized_value=None, status=status, rules=tuple(rules),
                                   warnings=tuple(warnings + [f"{code}: {message}"]), **fields, **extra)

    range_match = RANGE.fullmatch(text)
    if not range_match and not SINGLE.fullmatch(text):
        return fail(Status.UNKNOWN, "malformed-date", f"{text!r} is not a recognised date or date range")
    try:
        if range_match:
            if range_match["a"].lower() in ("present", "current"):
                raise DateProblem("malformed-range", "a range cannot start with Present", Status.UNKNOWN)
            start = parse_token(range_match["a"])
            end = parse_token(range_match["b"])
            separator = range_match["sep"]
        else:
            start, end, separator = parse_token(text), None, None
            if start[2]:
                raise DateProblem("malformed-date", "Present is not a date on its own", Status.UNKNOWN)
    except DateProblem as problem:
        return fail(problem.status, problem.code, problem.message)

    start_year, start_month, _, start_rule = start
    rules.append(start_rule)
    end_year = end_month = None
    is_current = False
    if separator is not None:
        end_year, end_month, is_current, end_rule = end
        rules.append(end_rule)
        if separator in ARTIFACT_SEPARATORS:
            # Both sides parsed as dates, so a lone artifact character here can only be the dash.
            rules.append("date-range/separator-artifact")
            warnings.append(f"separator-artifact: {separator!r} between two valid dates read as a dash; raw preserved")
        else:
            rules.append("date-range/separator-to" if separator.lower() == "to" else "date-range/separator-dash")
    structure = dict(start_year=start_year, start_month=start_month, end_year=end_year,
                     end_month=end_month, is_current=is_current)
    if end_year is not None and ends_before_start(start_year, start_month, end_year, end_month):
        return fail(Status.INVALID, "end-before-start",
                    f"end {fmt(end_year, end_month)} is before start {fmt(start_year, start_month)}", **structure)

    normalized = fmt(start_year, start_month)
    if separator is not None:
        normalized += " / " + ("Present" if is_current else fmt(end_year, end_month))
    return NormalizedDateRange(normalized_value=normalized,
                               status=Status.UNCHANGED if normalized == raw else Status.NORMALIZED,
                               rules=tuple(rules), warnings=tuple(warnings), **fields, **structure)
