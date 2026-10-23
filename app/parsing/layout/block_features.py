"""Read M1 geometry without copying text, styles, or changing provenance."""

from statistics import median

from ..models import PageIR
from .geometry import height, width
from .models import BlockFeatures


def block_features(page: PageIR) -> tuple[BlockFeatures, ...]:
    pw, ph = width(page.bounds), height(page.bounds)
    if pw <= 0 or ph <= 0:
        raise ValueError("Layout requires positive page dimensions")
    result = []
    for block in sorted(page.blocks, key=lambda b: b.source.object_id):
        box = block.source.box
        w, h = width(box), height(box)
        result.append(BlockFeatures(block.source, box.x0, box.y0, box.x1, box.y1,
                                    w, h, (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2,
                                    w * h, w / pw, h / ph))
    return tuple(result)


def line_scale(page: PageIR, config) -> float:
    heights = [height(line.source.box) for block in page.blocks for line in block.lines
               if height(line.source.box) > 0]
    return median(heights) if heights else height(page.bounds) * config.proximity_page_ratio
