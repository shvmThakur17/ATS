"""Bounded, linear-time evidence excerpts for match responses.

Resume text is attacker-controlled, so nothing here may take super-linear time. The text is cut to a small window
around the matched term BEFORE any redaction, and redaction is a single pass over that window (no backtracking
regular expressions over user text). Emails and phone-like numbers are replaced; the excerpt is then shortened.
"""

import re

EXCERPT_LENGTH = 300
SEARCH_CHARS = 20_000          # the term is looked for in at most this much of the chunk
WINDOW_CHARS = 1_200           # raw characters examined (and redacted) around the term
LEAD_CHARS = 120               # context kept before the matched term
MIN_PHONE_DIGITS = 7
PHONE_CHARS = frozenset("0123456789()+-. ")
PHONE_START = frozenset("0123456789+(")
TRAILING = ".,;:!?)]}>\"'"


def is_word(character: str) -> bool:
    return character.isalnum() or character == "_"


def redact_emails(text: str) -> str:
    """Any whitespace-delimited token containing '@' next to other text is treated as an email address."""
    tokens = []
    for token in text.split(" "):
        if "@" in token and any(c.isalnum() for c in token):
            stripped = token.rstrip(TRAILING)
            token = "[email]" + token[len(stripped):]
        tokens.append(token)
    return " ".join(tokens)


def redact_phones(text: str) -> str:
    """One left-to-right pass. A run of digits and phone punctuation with >= 7 digits, not glued to a word, is a number."""
    pieces, last, i, n = [], 0, 0, len(text)
    while i < n:
        if text[i] not in PHONE_CHARS:
            i += 1
            continue
        j = i
        while j < n and text[j] in PHONE_CHARS:
            j += 1
        start, end = i, j
        while start < end and text[start] not in PHONE_START:
            start += 1
        while end > start and not text[end - 1].isdigit():
            end -= 1
        digits = sum(1 for c in text[start:end] if c.isdigit())
        if digits >= MIN_PHONE_DIGITS and (start == 0 or not is_word(text[start - 1])) \
                and (end == n or not is_word(text[end])):
            pieces.append(text[last:start])
            pieces.append("[phone]")
            last = end
        i = j
    pieces.append(text[last:])
    return "".join(pieces)


def safe_excerpt(text: str | None, term: str | None = None) -> str | None:
    if not text:
        return None
    text = text[:SEARCH_CHARS]
    start = 0
    if term:
        found = re.search(re.escape(term), text, re.IGNORECASE)         # escaped literal: no backtracking risk
        if found:
            start = max(0, found.start() - LEAD_CHARS)
    window = text[start:start + WINDOW_CHARS]
    if start:
        window = window.split(None, 1)[1] if len(window.split(None, 1)) > 1 else window    # drop a cut-off first word
    redacted = redact_phones(redact_emails(" ".join(window.split())))
    prefix = "..." if start else ""
    room = EXCERPT_LENGTH - len(prefix)
    return prefix + (redacted if len(redacted) <= room else redacted[:room - 3] + "...")
