"""M5 deterministic section detection."""

from .models import SectionBlock, SectionDocumentIR, SectionType
from .section_detector import detect_sections, resolve_logical_text

__all__ = ["SectionBlock", "SectionDocumentIR", "SectionType", "detect_sections", "resolve_logical_text"]
