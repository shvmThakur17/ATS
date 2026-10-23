"""Safe text normalization: Unicode NFC, zero-width removal, whitespace collapse.

Nothing else is touched: casing, punctuation and dashes (including U+2013) are preserved.
"""

import re
import unicodedata

from ..records.models import EntityRef
from .models import NormalizationStatus, NormalizedValue

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"))
# 'û' / 'Û' are valid letters, so they are only reported, never replaced.
SUSPICIOUS = "ûÛ�"


def clean_text(raw: str) -> tuple[str, list[str]]:
    """Return the cleaned text and the names of the rules that changed it."""
    rules, text = [], raw
    for rule, step in (("text/unicode-nfc", lambda t: unicodedata.normalize("NFC", t)),
                       ("text/remove-zero-width", lambda t: t.translate(ZERO_WIDTH)),
                       ("text/collapse-whitespace", lambda t: re.sub(r"\s+", " ", t).strip())):
        changed = step(text)
        if changed != text:
            rules.append(rule)
            text = changed
    return text, rules


def suspicious_warnings(text: str) -> list[str]:
    found = sorted({c for c in text if c in SUSPICIOUS})
    return [f"suspicious-character: possible extraction artifact {c!r} kept as-is; raw value preserved"
            for c in found]


def make_value(raw: str, normalized: str | None, rules=(), warnings=(), sources=(),
               status: NormalizationStatus | None = None) -> NormalizedValue:
    warnings = list(warnings)
    if status is None:
        if normalized is None or normalized == "":
            normalized, status = None, NormalizationStatus.UNKNOWN
            warnings.append("empty-value: value is empty after normalization")
        elif normalized == raw:
            status = NormalizationStatus.UNCHANGED
        else:
            status = NormalizationStatus.NORMALIZED
    return NormalizedValue(raw_value=raw, normalized_value=normalized, status=status,
                           rules=tuple(rules), warnings=tuple(warnings), sources=tuple(sources))


def normalize_text(raw: str, sources: tuple[EntityRef, ...] = ()) -> NormalizedValue:
    text, rules = clean_text(raw)
    return make_value(raw, text, rules, suspicious_warnings(raw), sources)
