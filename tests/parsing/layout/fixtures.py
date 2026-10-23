from app.parsing.models import Box, DocumentIR, PageIR, Point, SourceRef, TextBlock, TextLine


def document(*page_boxes, width=600, height=800):
    """Native geometry fixtures; one line per box, no synthetic semantics."""
    pages = []
    for number, boxes in enumerate(page_boxes):
        blocks = []
        for i, coords in enumerate(boxes):
            box = Box(*coords)
            block_id = f"p{number}/b{i}"
            block = SourceRef("synthetic", number, block_id, box)
            line = SourceRef("synthetic", number, f"{block_id}/l0", box)
            blocks.append(TextBlock(block, i, (TextLine(line, (), Point(1, 0), 0),)))
        bounds = Box(0, 0, width, height)
        pages.append(PageIR(number, bounds, bounds, bounds, 0,
                            (1, 0, 0, 1, 0, 0), (1, 0, 0, 1, 0, 0),
                            tuple(blocks), (), (), (), "NATIVE_TEXT"))
    return DocumentIR("synthetic", "resume-ir/1", "pdf-objects/1", "fixture/1", "fixture", tuple(pages), ())


def columns(ranges=((50, 270), (330, 550)), ys=(140, 210, 280)):
    return [(x0, y, x1, y + 20) for x0, x1 in ranges for y in ys]


def memberships(page):
    return {frozenset(region.block_ids) for region in page.regions}
