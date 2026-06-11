"""Render structured tabular data for Telegram.

Primary output is a PNG image (render_table_image) sent as a photo: it avoids
Telegram's "COPY CODE" header that <pre> blocks get, looks clean at any width,
and supports zoom on mobile. The text renderers (compact grid / vertical) are
kept as a fallback for when image rendering is unavailable (e.g. no font/Pillow).
"""
from __future__ import annotations

import html
from io import BytesIO
from typing import List, Optional, Sequence

# Max usable monospace width on a phone in portrait (conservative). Tables wider
# than this fall back to the vertical layout.
_MAX_TABLE_WIDTH = 32

# Cell truncation: tighter in grid mode (must fit side by side), looser in
# vertical mode (each value gets its own line).
_GRID_CELL_WIDTH = 18
_VERTICAL_CELL_WIDTH = 40

# Circled numbers ①..⑳ for vertical-layout row markers; falls back to "N." after.
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def _norm(value, width: int) -> str:
    """Collapse whitespace and truncate a single cell value to `width`."""
    s = " ".join(str(value if value is not None else "").split())
    if len(s) > width:
        return s[: width - 1] + "…"
    return s


def _col_widths(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[int]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(len(headers)):
            cell = row[i] if i < len(row) else ""
            widths[i] = max(widths[i], len(cell))
    return widths


def _bullet(idx: int) -> str:
    """Row marker for vertical layout (0-based)."""
    return _CIRCLED[idx] if idx < len(_CIRCLED) else f"{idx + 1}."


def render_compact_grid(headers: Sequence, rows: Sequence[Sequence]) -> str:
    """Borderless grid: columns separated by two spaces, underline under headers.

    Narrower than a bordered table (saves ~3 chars per column). Returns raw text.
    """
    h = [_norm(x, _GRID_CELL_WIDTH) for x in headers]
    r = [[_norm(c, _GRID_CELL_WIDTH) for c in row] for row in rows]
    widths = _col_widths(h, r)
    cols = len(h)

    def line(cells: Sequence[str]) -> str:
        return "  ".join(
            (cells[i] if i < len(cells) else "").ljust(widths[i]) for i in range(cols)
        ).rstrip()

    out = [line(h), "  ".join("─" * w for w in widths)]
    for row in r:
        out.append(line(row))
    return "\n".join(out)


def render_vertical(headers: Sequence, rows: Sequence[Sequence]) -> str:
    """Vertical layout: one block per row, first column as title, rest as
    aligned 'field: value' lines. Never overflows horizontally."""
    h = [_norm(x, _VERTICAL_CELL_WIDTH) for x in headers]
    r = [[_norm(c, _VERTICAL_CELL_WIDTH) for c in row] for row in rows]
    cols = len(h)

    # Align the "field:" labels (columns 1..n) so values line up.
    label_w = max((len(h[j]) for j in range(1, cols)), default=0) + 1  # +1 for ':'

    blocks: List[str] = []
    for idx, row in enumerate(r):
        title = row[0] if row else ""
        lines = [f"{_bullet(idx)} {title}"]
        for j in range(1, cols):
            val = row[j] if j < len(row) else ""
            label = (h[j] + ":").ljust(label_w)
            lines.append(f"   {label} {val}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _grid_width(headers: Sequence, rows: Sequence[Sequence]) -> int:
    """Natural width of the data (using the looser vertical cap), so the grid vs
    vertical decision reflects the real content — not the aggressive grid
    truncation. This sends genuinely long values to the vertical layout, where
    they're shown in full, instead of cramming them truncated into a grid."""
    h = [_norm(x, _VERTICAL_CELL_WIDTH) for x in headers]
    r = [[_norm(c, _VERTICAL_CELL_WIDTH) for c in row] for row in rows]
    widths = _col_widths(h, r)
    return sum(widths) + 2 * (len(widths) - 1)  # 2-space separators


def render_telegram_table(
    headers: Sequence,
    rows: Sequence[Sequence],
    max_width: int = _MAX_TABLE_WIDTH,
) -> str:
    """Render an HTML-safe <pre> block, choosing grid or vertical layout by width.

    - Single-column or grid that fits within `max_width` → compact grid.
    - Anything wider                                       → vertical layout.
    """
    cols = len(headers)
    if cols <= 1 or _grid_width(headers, rows) <= max_width:
        body = render_compact_grid(headers, rows)
    else:
        body = render_vertical(headers, rows)

    safe = html.escape(body, quote=False)  # escape & < > but keep box/marker chars
    return f"<pre>{safe}</pre>"


# ─────────────────────────────────────────────────────────────────────────────
#  PNG image rendering (primary output — no "COPY CODE" button, zoomable)
# ─────────────────────────────────────────────────────────────────────────────

# Font candidates tried in order. Covers Windows, Linux (Debian/Railway) and mac.
# Widths are measured per-string, so a proportional font still aligns correctly.
_FONT_CANDIDATES = (
    "DejaVuSansMono.ttf", "consola.ttf", "cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "DejaVuSans.ttf", "arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_FONT_BOLD_CANDIDATES = (
    "DejaVuSansMono-Bold.ttf", "consolab.ttf", "courbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "DejaVuSans-Bold.ttf", "arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

# Palette
_C_HEADER_BG = (44, 62, 80)      # dark slate
_C_HEADER_FG = (255, 255, 255)
_C_ROW_ALT = (242, 242, 242)     # zebra striping
_C_ROW_BG = (255, 255, 255)
_C_TEXT = (34, 34, 34)
_C_GRID = (200, 200, 200)
_C_TITLE = (44, 62, 80)


# Max column width in pixels before a cell wraps to multiple lines. Long values
# are never truncated in the image — they wrap so the full text is shown.
_MAX_COL_PX = 360


def _load_font(size: int, candidates: Sequence[str]):
    from PIL import ImageFont
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _clean_cell(value) -> str:
    """Collapse whitespace/newlines but DO NOT truncate (the image wraps text)."""
    return " ".join(str(value if value is not None else "").split())


def _wrap_text(text: str, font, max_px: float, measure) -> List[str]:
    """Word-wrap `text` to fit within `max_px`, breaking over-long words by char."""
    def width(s: str) -> float:
        return measure.textlength(s, font=font)

    if not text:
        return [""]

    lines: List[str] = []
    cur = ""
    for word in text.split():
        candidate = word if not cur else f"{cur} {word}"
        if width(candidate) <= max_px:
            cur = candidate
            continue
        if cur:
            lines.append(cur)
            cur = ""
        # The single word itself may exceed max_px → hard-break by character.
        if width(word) <= max_px:
            cur = word
        else:
            piece = ""
            for ch in word:
                if width(piece + ch) <= max_px or not piece:
                    piece += ch
                else:
                    lines.append(piece)
                    piece = ch
            cur = piece
    if cur:
        lines.append(cur)
    return lines or [""]


def render_table_image(
    headers: Sequence,
    rows: Sequence[Sequence],
    title: Optional[str] = None,
) -> bytes:
    """Render the table as a PNG and return the raw bytes.

    Cells longer than _MAX_COL_PX wrap to multiple lines so the full text is
    always shown — nothing is truncated. Row height grows to fit the tallest cell.
    Raises ImportError if Pillow is missing — callers should fall back to text.
    """
    from PIL import Image, ImageDraw

    h = [_clean_cell(x) for x in headers]
    cols = len(h)
    r = [[_clean_cell(row[i] if i < len(row) else "") for i in range(cols)] for row in rows]

    size = 22
    font = _load_font(size, _FONT_CANDIDATES)
    bold = _load_font(size, _FONT_BOLD_CANDIDATES)
    measure = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def tw(text: str, f) -> float:
        return measure.textlength(text, font=f)

    pad_x, pad_y = 18, 12
    line_gap = 6  # vertical gap between wrapped lines inside a cell

    # Column text width = natural width capped at _MAX_COL_PX.
    col_text_w: List[float] = []
    for i in range(cols):
        w = tw(h[i], bold)
        for row in r:
            w = max(w, tw(row[i], font))
        col_text_w.append(min(w, _MAX_COL_PX))
    col_w = [w + pad_x * 2 for w in col_text_w]

    asc = measure.textbbox((0, 0), "Ágxp", font=bold)
    base_lh = asc[3] - asc[1]

    def cell_height(n_lines: int) -> int:
        return int(n_lines * base_lh + (n_lines - 1) * line_gap + pad_y * 2)

    # Pre-wrap every cell and compute per-row heights.
    header_cells = [_wrap_text(h[i], bold, col_text_w[i], measure) for i in range(cols)]
    header_h = cell_height(max(len(c) for c in header_cells))

    row_cells: List[List[List[str]]] = []
    row_heights: List[int] = []
    for row in r:
        cells = [_wrap_text(row[i], font, col_text_w[i], measure) for i in range(cols)]
        row_cells.append(cells)
        row_heights.append(cell_height(max(len(c) for c in cells)))

    table_w = sum(col_w)
    margin = 20
    title_h = cell_height(1) if title else 0
    content_w = max(table_w, tw(title, bold) if title else 0)
    img_w = int(content_w + margin * 2)
    img_h = int(margin * 2 + title_h + header_h + sum(row_heights))

    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    def draw_cell(lines: List[str], x: int, y: int, fill, f) -> None:
        cy = y
        for ln in lines:
            draw.text((x, cy), ln, fill=fill, font=f)
            cy += base_lh + line_gap

    y = margin
    if title:
        draw.text((margin, y + pad_y), title, fill=_C_TITLE, font=bold)
        y += title_h

    top = y  # top of the grid (header row)

    # Header band
    draw.rectangle([margin, y, margin + table_w, y + header_h], fill=_C_HEADER_BG)
    x = margin
    for i in range(cols):
        draw_cell(header_cells[i], x + pad_x, y + pad_y, _C_HEADER_FG, bold)
        x += col_w[i]
    y += header_h

    # Data rows (zebra), each as tall as its tallest wrapped cell
    row_edges = [top, top + header_h]
    for idx, cells in enumerate(row_cells):
        rh = row_heights[idx]
        bg = _C_ROW_BG if idx % 2 == 0 else _C_ROW_ALT
        draw.rectangle([margin, y, margin + table_w, y + rh], fill=bg)
        x = margin
        for i in range(cols):
            draw_cell(cells[i], x + pad_x, y + pad_y, _C_TEXT, font)
            x += col_w[i]
        y += rh
        row_edges.append(y)
    bottom = y

    # Grid lines: horizontals at every row edge, verticals per column
    for yline in row_edges:
        draw.line([margin, yline, margin + table_w, yline], fill=_C_GRID, width=1)
    xx = margin
    for i in range(cols + 1):
        draw.line([xx, top, xx, bottom], fill=_C_GRID, width=1)
        if i < cols:
            xx += col_w[i]

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
