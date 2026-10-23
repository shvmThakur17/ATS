"""Deterministic physical layout analysis (M2)."""

from .layout_analyzer import analyze_layout, format_diagnostics
from .models import LayoutConfig, LayoutIR, LayoutLimitError, RegionKind, RelationKind

__all__ = ["analyze_layout", "format_diagnostics", "LayoutConfig", "LayoutIR",
           "LayoutLimitError", "RegionKind", "RelationKind"]
