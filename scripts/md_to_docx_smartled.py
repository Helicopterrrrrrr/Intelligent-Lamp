from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt, RGBColor, Twips
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\project\SmartLED")
OUT_DIR = ROOT / "output" / "doc"
ASSET_DIR = OUT_DIR / "word_assets"
SOURCE_MD = OUT_DIR / "SmartLED_智能学习台灯结课报告_详细版.md"
TARGET_DOCX = OUT_DIR / "SmartLED_智能学习台灯结课报告_详细版.docx"

ASSET_DIR.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]

BLACK = RGBColor(0, 0, 0)
GRAY = RGBColor(90, 90, 90)
BLUE = RGBColor(46, 116, 181)
DARK = RGBColor(35, 35, 35)
LIGHT_FILL = "F3F3F3"
LIGHT_FILL_2 = "FFFFFF"
BORDER = "000000"
TITLE_FONT = "方正小标宋_GBK"
HEADING_FONT = "黑体"
BODY_FONT = "宋体"
TABLE_TARGET_WIDTH_DXA = 8306
TABLE_BASE_WIDTH_DXA = 9360
TABLE_SCALE = TABLE_TARGET_WIDTH_DXA / TABLE_BASE_WIDTH_DXA
FIGURE_WIDTH = Inches(5.45)
FIRST_LINE_INCH = 1 / 3
LEFT_MARGIN = Inches(1.25)
RIGHT_MARGIN = Inches(1.25)
TOP_MARGIN = Inches(1.0)
BOTTOM_MARGIN = Inches(1.0)
HEADER_DISTANCE = Twips(851)
FOOTER_DISTANCE = Twips(992)


def pick_font(size: int):
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def draw_center(draw, text, box, font, fill=(0, 0, 0), spacing=10):
    x1, y1, x2, y2 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x1 + (x2 - x1 - w) / 2
    y = y1 + (y2 - y1 - h) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="center")


def set_run_font(run, *, name=BODY_FONT, size=12, color=BLACK, bold=None, italic=None, underline=None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.color.rgb = color
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if underline is not None:
        run.underline = underline


def set_paragraph_format(par, *, before=0, after=0, line=1.5, first_line=FIRST_LINE_INCH, align=None):
    pf = par.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if first_line is not None:
        pf.first_line_indent = Inches(first_line)
    if align is not None:
        par.alignment = align


def apply_doc_style(doc: Document):
    styles = doc.styles
    body = styles["Normal"]
    body.font.name = BODY_FONT
    body._element.rPr.rFonts.set(qn("w:ascii"), BODY_FONT)
    body._element.rPr.rFonts.set(qn("w:hAnsi"), BODY_FONT)
    body._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    body.font.size = Pt(12)
    body.font.color.rgb = BLACK
    body.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    body.paragraph_format.space_before = Pt(0)
    body.paragraph_format.space_after = Pt(0)
    body.paragraph_format.line_spacing = 1.5
    body.paragraph_format.first_line_indent = Inches(FIRST_LINE_INCH)

    for name, font_name, size, align, before, after, page_break in [
        ("Heading 1", HEADING_FONT, 15, WD_ALIGN_PARAGRAPH.CENTER, 12, 6, True),
        ("Heading 2", HEADING_FONT, 14, WD_ALIGN_PARAGRAPH.LEFT, 8, 4, False),
        ("Heading 3", HEADING_FONT, 12, WD_ALIGN_PARAGRAPH.LEFT, 6, 2, False),
    ]:
        st = styles[name]
        st.font.name = font_name
        st._element.rPr.rFonts.set(qn("w:ascii"), font_name)
        st._element.rPr.rFonts.set(qn("w:hAnsi"), font_name)
        st._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
        st.font.size = Pt(size)
        st.font.color.rgb = BLACK
        st.font.bold = False
        st.paragraph_format.alignment = align
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.line_spacing = 1.5
        st.paragraph_format.page_break_before = page_break

    for name in ("List Bullet", "List Number"):
        st = styles[name]
        st.font.name = BODY_FONT
        st._element.rPr.rFonts.set(qn("w:ascii"), BODY_FONT)
        st._element.rPr.rFonts.set(qn("w:hAnsi"), BODY_FONT)
        st._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        st.font.size = Pt(12)
        st.font.color.rgb = BLACK
        st.paragraph_format.left_indent = Inches(0.33)
        st.paragraph_format.first_line_indent = Inches(-0.17)
        st.paragraph_format.space_after = Pt(0)
        st.paragraph_format.space_before = Pt(0)
        st.paragraph_format.line_spacing = 1.5

    if "Caption" in styles:
        st = styles["Caption"]
        st.font.name = BODY_FONT
        st._element.rPr.rFonts.set(qn("w:ascii"), BODY_FONT)
        st._element.rPr.rFonts.set(qn("w:hAnsi"), BODY_FONT)
        st._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        st.font.size = Pt(10.5)
        st.font.color.rgb = BLACK
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        st.paragraph_format.space_before = Pt(0)
        st.paragraph_format.space_after = Pt(0)
        st.paragraph_format.line_spacing = 1.0

    for name in ("Normal Table", "Table Grid"):
        if name in styles:
            st = styles[name]
            st.font.name = BODY_FONT
            st.font.size = Pt(10.5)
            st.font.color.rgb = BLACK


def set_section(section, *, first_page_blank=False):
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.top_margin = TOP_MARGIN
    section.bottom_margin = BOTTOM_MARGIN
    section.left_margin = LEFT_MARGIN
    section.right_margin = RIGHT_MARGIN
    section.header_distance = HEADER_DISTANCE
    section.footer_distance = FOOTER_DISTANCE
    section.different_first_page_header_footer = first_page_blank


def set_page_number_start(section, start=1):
    sect_pr = section._sectPr
    pg_num = sect_pr.find(qn("w:pgNumType"))
    if pg_num is None:
        pg_num = OxmlElement("w:pgNumType")
        sect_pr.append(pg_num)
    pg_num.set(qn("w:start"), str(start))



def add_header_footer(section, title_text):
    hdr = section.header.paragraphs[0]
    hdr.clear()
    hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hdr.paragraph_format.space_before = Pt(0)
    hdr.paragraph_format.space_after = Pt(0)
    hdr.paragraph_format.line_spacing = 1.0
    ppr = hdr._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), BORDER)
    pbdr.append(bottom)
    ppr.append(pbdr)
    r = hdr.add_run(title_text)
    set_run_font(r, name=BODY_FONT, size=9.5, color=GRAY)

    ftr = section.footer.paragraphs[0]
    ftr.clear()
    ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ftr.paragraph_format.space_before = Pt(0)
    ftr.paragraph_format.space_after = Pt(0)
    ftr.paragraph_format.line_spacing = 1.0
    r1 = ftr.add_run("- ")
    set_run_font(r1, name=BODY_FONT, size=9, color=GRAY)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run = ftr.add_run()
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(sep)
    run._r.append(end)
    r2 = ftr.add_run(" -")
    set_run_font(r2, name=BODY_FONT, size=9, color=GRAY)


def add_spacer(doc, after=12):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.0
    return p


def add_para(doc, text, *, size=12, color=BLACK, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=0, line=1.5, first_line=FIRST_LINE_INCH, font=BODY_FONT):
    p = doc.add_paragraph()
    set_paragraph_format(p, before=before, after=after, line=line, first_line=first_line, align=align)
    r = p.add_run(text)
    set_run_font(r, name=font, size=size, color=color, bold=bold, italic=italic)
    return p


INLINE_RE = re.compile(r"(\*\*[^*]+?\*\*|`[^`]+?`|\*[^*]+?\*|\[[^\]]+?\]\([^)]+?\))")


def add_rich_para(doc, text, *, before=0, after=0, line=1.5, first_line=FIRST_LINE_INCH, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    set_paragraph_format(p, before=before, after=after, line=line, first_line=first_line, align=align)
    pos = 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            r = p.add_run(text[pos:m.start()])
            set_run_font(r, name=BODY_FONT, size=12)
        token = m.group(0)
        if token.startswith("**") and token.endswith("**"):
            r = p.add_run(token[2:-2])
            set_run_font(r, name=BODY_FONT, size=12, bold=True)
        elif token.startswith("`") and token.endswith("`"):
            r = p.add_run(token[1:-1])
            set_run_font(r, name="Consolas", size=10)
        elif token.startswith("*") and token.endswith("*"):
            r = p.add_run(token[1:-1])
            set_run_font(r, name=BODY_FONT, size=12, italic=True)
        elif token.startswith("[") and "](" in token:
            label = token[1:token.index("]")]
            r = p.add_run(label)
            set_run_font(r, name=BODY_FONT, size=12, color=BLACK, underline=True)
        pos = m.end()
    if pos < len(text):
        r = p.add_run(text[pos:])
        set_run_font(r, name=BODY_FONT, size=12)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.first_line_indent = None
    if level == 1:
        p.paragraph_format.page_break_before = True
    r = p.add_run(text)
    set_run_font(r, name=HEADING_FONT, size={1: 15, 2: 14, 3: 12}[level], color=BLACK, bold=False)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.5
    r = p.add_run(text)
    set_run_font(r, name=BODY_FONT, size=12)
    return p


def add_number(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.5
    r = p.add_run(text)
    set_run_font(r, name=BODY_FONT, size=12)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    set_paragraph_format(p, before=0, after=0, line=1.0, first_line=None, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run(text)
    set_run_font(r, name=BODY_FONT, size=10.5, color=BLACK)
    return p


def add_rule(doc):
    p = doc.add_paragraph()
    set_paragraph_format(p, before=0, after=6, line=1.0, first_line=None)
    ppr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), BORDER)
    pbdr.append(bottom)
    ppr.append(pbdr)
    return p


def set_cell_margins(cell, top=80, start=108, bottom=80, end=108):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_table_borders(table, color=BORDER, size="4"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def scale_widths(widths, target=TABLE_TARGET_WIDTH_DXA):
    total = sum(widths)
    if total == target:
        return [int(w) for w in widths]
    scaled = [max(420, int(round(w * target / total))) for w in widths]
    scaled[-1] += target - sum(scaled)
    return scaled


def apply_table_geometry(table, widths, indent=0):
    widths = [int(w) for w in widths]
    total = sum(widths)
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:type"), "dxa")
    tbl_w.set(qn("w:w"), str(total))
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:type"), "dxa")
    tbl_ind.set(qn("w:w"), str(indent))
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    grid = tbl.tblGrid
    for c in list(grid):
        grid.remove(c)
    for w in widths:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(w))
        grid.append(gc)
    for ci, w in enumerate(widths):
        table.columns[ci].width = Inches(w / 1440)
    for row in table.rows:
        if len(row.cells) != len(widths):
            raise ValueError("Merged rows not supported")
        for ci, cell in enumerate(row.cells):
            cell.width = Inches(widths[ci] / 1440)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:type"), "dxa")
            tc_w.set(qn("w:w"), str(widths[ci]))
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    set_run_font(r, name=BODY_FONT, size=10.5)


def make_table(doc, headers, rows, widths=None):
    n = len(headers)
    if widths is None:
        if n == 2:
            widths = [1900, 7460]
        elif n == 3:
            widths = [1600, 1700, 6060]
        elif n == 4:
            widths = [1500, 1500, 2400, 3960]
        elif n == 5:
            widths = [1400, 1200, 2400, 2200, 2160]
        elif n == 6:
            widths = [1400, 1000, 1150, 1150, 2200, 2450]
        else:
            base = 9360 // n
            widths = [base] * n
            widths[-1] += 9360 - sum(widths)
    widths = scale_widths(widths)
    tbl = doc.add_table(rows=1, cols=n)
    tbl.style = "Table Grid"
    for i, h in enumerate(headers):
        tbl.rows[0].cells[i].text = h
    for row in rows:
        cells = tbl.add_row().cells
        for i in range(n):
            cells[i].text = str(row[i]) if i < len(row) else ""
    apply_table_geometry(tbl, widths, indent=0)
    set_table_borders(tbl)
    for i, cell in enumerate(tbl.rows[0].cells):
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                set_run_font(r, name=BODY_FONT, size=10.5, bold=True)
    for row in tbl.rows[1:]:
        for i, cell in enumerate(row.cells):
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i in (0, n - 1) else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    set_run_font(r, name=BODY_FONT, size=10.5)
    return tbl


def clean_inline_md(text: str) -> str:
    # strip common markdown syntax while keeping readable text
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def parse_blocks(section_text: str):
    lines = section_text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            blocks.append(("blank", ""))
            i += 1
            continue
        if line.startswith("### "):
            blocks.append(("h2", line[4:].strip()))
            i += 1
            continue
        if line.startswith("#### "):
            blocks.append(("h3", line[5:].strip()))
            i += 1
            continue
        if line.startswith("```"):
            lang = line[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", lang, code_lines))
            continue
        if line.startswith("|") and "|" in line:
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            blocks.append(("table", table_lines))
            continue
        if re.match(r"^\s*[-*] \S", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*] \S", lines[i]):
                items.append(clean_inline_md(lines[i].strip()[2:].strip()))
                i += 1
            blocks.append(("bullets", items))
            continue
        if re.match(r"^\s*\d+\.\s+\S", line):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+\S", lines[i]):
                items.append(clean_inline_md(re.sub(r"^\s*\d+\.\s+", "", lines[i]).strip()))
                i += 1
            blocks.append(("numbers", items))
            continue
        if line.strip() == "---":
            blocks.append(("rule", ""))
            i += 1
            continue
        # paragraph accumulation
        paras = [clean_inline_md(line.strip())]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip() or nxt.startswith(("### ", "#### ", "```")) or nxt.startswith("|") or re.match(r"^\s*[-*] \S", nxt) or re.match(r"^\s*\d+\.\s+\S", nxt) or nxt.strip() == "---":
                break
            paras.append(clean_inline_md(nxt.strip()))
            i += 1
        blocks.append(("para", " ".join(paras)))
    return blocks


def extract_sections(md_text: str):
    lines = md_text.splitlines()
    sections = []
    current_title = None
    current_lines = []
    for line in lines:
        if line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def add_block(doc, block):
    kind = block[0]
    if kind == "h2":
        add_heading(doc, block[1], 2)
    elif kind == "h3":
        add_heading(doc, block[1], 3)
    elif kind == "para":
        add_rich_para(doc, block[1])
    elif kind == "bullets":
        for item in block[1]:
            add_bullet(doc, item)
    elif kind == "numbers":
        for item in block[1]:
            add_number(doc, item)
    elif kind == "table":
        rows = [r.strip() for r in block[1] if r.strip()]
        if len(rows) < 2:
            return
        table_rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in rows]
        header = table_rows[0]
        data = table_rows[2:] if len(table_rows) > 2 else []
        make_table(doc, header, data)
        add_para(doc, "", after=4, first_line=None)
    elif kind == "code":
        lang = block[1]
        code_lines = block[2]
        if lang == "mermaid":
            # keep a textual note; the actual diagrams are inserted as images later
            add_para(doc, "（流程图见图示）", size=10, color=GRAY, italic=True, first_line=0.0, after=4)
        else:
            for ln in code_lines:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.18)
                p.paragraph_format.right_indent = Inches(0.18)
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                r = p.add_run(ln)
                set_run_font(r, name="Consolas", size=9.5, color=DARK)
    elif kind == "rule":
        add_rule(doc)


def make_system_figure(path: Path):
    img = Image.new("RGB", (1600, 900), (248, 250, 252))
    d = ImageDraw.Draw(img)
    title_font = pick_font(30)
    box_font = pick_font(20)
    note_font = pick_font(17)
    d.rounded_rectangle((50, 50, 1550, 850), radius=30, outline=(94, 112, 132), width=4)
    draw_center(d, "SmartLED 智能学习台灯系统总体架构", (80, 70, 1520, 130), title_font, fill=(28, 43, 58))
    boxes = [
        (90, 220, 300, 410, "设备端感知\nESP32-S3\nAHT20 / BH1750\nVL53L0X / 摄像头"),
        (390, 220, 600, 410, "后端服务\nFlask + SQLite\n状态融合 / 事件记录"),
        (690, 220, 900, 410, "视觉分析\nYOLO11n-pose\n目标选择 / 特征提取"),
        (990, 220, 1200, 410, "规则引擎\n在位 / 距离 / 姿态\n时序平滑 / 冷却"),
        (1290, 220, 1500, 410, "应用层\nVue 3 前端\n监控 / 摘要 / 绑定"),
    ]
    box_fill = (225, 239, 249)
    outline = (54, 96, 146)
    for x1, y1, x2, y2, label in boxes:
        d.rounded_rectangle((x1, y1, x2, y2), radius=20, fill=box_fill, outline=outline, width=4)
        draw_center(d, label, (x1 + 10, y1 + 16, x2 - 10, y2 - 16), box_font, fill=(25, 45, 65), spacing=6)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][2]
        x2 = boxes[i + 1][0]
        y = 315
        d.line((x1 + 8, y, x2 - 8, y), fill=outline, width=5)
        d.polygon([(x2 - 8, y), (x2 - 22, y - 9), (x2 - 22, y + 9)], fill=outline)
    d.rounded_rectangle((180, 520, 1420, 760), radius=24, outline=(146, 160, 176), width=4, fill=(255, 255, 255))
    draw_center(d, "核心闭环：采集 -> 融合 -> 姿态判断 -> 事件输出 -> 前端展示 -> 台灯控制", (220, 560, 1380, 620), box_font, fill=(35, 50, 65))
    draw_center(d, "系统重点不在单一模型，而在多传感器、规则引擎与自动照明策略的联动。", (220, 640, 1380, 710), note_font, fill=(65, 75, 85))
    img.save(path)


def make_pipeline_figure(path: Path):
    img = Image.new("RGB", (1600, 900), (248, 250, 252))
    d = ImageDraw.Draw(img)
    title_font = pick_font(28)
    box_font = pick_font(18)
    note_font = pick_font(16)
    d.rounded_rectangle((50, 50, 1550, 850), radius=30, outline=(94, 112, 132), width=4)
    draw_center(d, "姿态检测与状态融合核心处理流程", (80, 70, 1520, 130), title_font, fill=(28, 43, 58))
    flow = [
        (90, 220, 260, 350, "输入图像帧"),
        (310, 220, 480, 350, "YOLO 姿态估计"),
        (530, 220, 700, 350, "目标选择\nROI 约束"),
        (750, 220, 920, 350, "特征提取\n肩宽 / 颈角"),
        (970, 220, 1140, 350, "归一化标定\n个性化参考"),
        (1190, 220, 1360, 350, "规则判断\n在位 / 距离 / 姿态"),
        (1410, 220, 1540, 350, "事件输出"),
    ]
    fill = (225, 239, 249)
    outline = (54, 96, 146)
    for x1, y1, x2, y2, label in flow:
        d.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=fill, outline=outline, width=4)
        draw_center(d, label, (x1 + 10, y1 + 14, x2 - 10, y2 - 14), box_font, fill=(25, 45, 65), spacing=4)
    for i in range(len(flow) - 1):
        x1 = flow[i][2]
        x2 = flow[i + 1][0]
        y = 285
        d.line((x1 + 8, y, x2 - 8, y), fill=outline, width=5)
        d.polygon([(x2 - 8, y), (x2 - 22, y - 9), (x2 - 22, y + 9)], fill=outline)
    d.rounded_rectangle((150, 500, 1460, 760), radius=24, outline=(146, 160, 176), width=4, fill=(255, 255, 255))
    draw_center(d, "辅助机制：时序平滑、异常持续时间、冷却时间、事件日志与快照上传", (200, 545, 1410, 605), box_font, fill=(35, 50, 65))
    draw_center(d, "设计目标是减少误报，让短暂抖动不触发提醒，让持续异常才形成稳定事件。", (200, 625, 1410, 690), note_font, fill=(65, 75, 85))
    img.save(path)


def make_eval_figure(path: Path):
    img = Image.new("RGB", (1600, 900), (255, 255, 255))
    d = ImageDraw.Draw(img)
    title_font = pick_font(28)
    axis_font = pick_font(16)
    label_font = pick_font(15)
    value_font = pick_font(15)
    draw_center(d, "实验结果摘要（公开视频与实验室侧视样本）", (70, 40, 1530, 100), title_font, fill=(28, 43, 58))
    d.rounded_rectangle((1190, 120, 1510, 205), radius=18, fill=(248, 250, 252), outline=(220, 226, 233), width=2)
    d.rectangle((1220, 145, 1250, 162), fill=(79, 129, 189))
    d.text((1265, 140), "seated_rate", font=label_font, fill=(79, 129, 189))
    d.rectangle((1220, 175, 1250, 192), fill=(192, 80, 77))
    d.text((1265, 170), "raw_abnormal_rate", font=label_font, fill=(192, 80, 77))

    labels = [
        ("absent", 1.00, 0.00),
        ("calibration_normal", 0.90, 0.03),
        ("computer_abnormal", 0.82, 0.08),
        ("computer_normal", 0.80, 0.02),
        ("reading_abnormal", 0.95, 0.31),
        ("reading_normal", 0.86, 0.11),
    ]
    left = 120
    base_y = 760
    bar_w = 34
    group_gap = 120
    max_h = 450
    for i, (lab, seated, abnormal) in enumerate(labels):
        x = left + i * group_gap
        h1 = int(seated * max_h)
        h2 = int(abnormal * max_h)
        d.rectangle((x, base_y - h1, x + bar_w, base_y), fill=(79, 129, 189))
        d.rectangle((x + 46, base_y - h2, x + 46 + bar_w, base_y), fill=(192, 80, 77))
        d.text((x - 8, base_y + 18), lab, font=label_font, fill=(50, 50, 50))
        d.text((x - 4, base_y - h1 - 24), f"{seated:.2f}", font=value_font, fill=(79, 129, 189))
        d.text((x + 42, base_y - h2 - 24), f"{abnormal:.2f}", font=value_font, fill=(192, 80, 77))
    d.line((80, 760, 1080, 760), fill=(180, 180, 180), width=2)
    d.line((80, 320, 80, 760), fill=(180, 180, 180), width=2)
    for y, val in [(760, "0.00"), (650, "0.25"), (540, "0.50"), (430, "0.75"), (320, "1.00")]:
        d.line((74, y, 86, y), fill=(180, 180, 180), width=2)
        d.text((35, y - 10), val, font=axis_font, fill=(100, 100, 100))
    d.text((88, 290), "比率", font=axis_font, fill=(100, 100, 100))
    d.rounded_rectangle((1120, 560, 1490, 760), radius=20, fill=(248, 250, 252), outline=(220, 226, 233), width=2)
    notes = [
        "1. 正面学习场景较稳定。",
        "2. 侧视/俯拍/暗光仍会掉到 absent。",
        "3. reading_abnormal 的触发偏弱。",
        "4. 个性化标定提升了适应性。",
    ]
    y = 590
    for n in notes:
        d.text((1150, y), n, font=label_font, fill=(55, 55, 55))
        y += 34
    img.save(path)


def make_figures():
    fig1 = ASSET_DIR / "figure_system_overview_cn.png"
    fig2 = ASSET_DIR / "figure_pipeline_cn.png"
    fig3 = ASSET_DIR / "figure_eval_cn.png"
    make_system_figure(fig1)
    make_pipeline_figure(fig2)
    make_eval_figure(fig3)
    return fig1, fig2, fig3


def add_image(doc, path: Path, caption: str):
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(path), width=FIGURE_WIDTH)
        add_caption(doc, caption)


def parse_md_table(table_lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in table_lines]
    if len(rows) < 2:
        return None, None
    header = rows[0]
    # skip separator row at rows[1]
    data = rows[2:] if len(rows) > 2 else []
    return header, data


def add_md_table(doc, table_lines):
    header, data = parse_md_table(table_lines)
    if not header:
        return
    n = len(header)
    if n == 2:
        widths = [1900, 7460]
    elif n == 3:
        widths = [1600, 1800, 5960]
    elif n == 4:
        widths = [1500, 1500, 2400, 3960]
    elif n == 5:
        widths = [1400, 1200, 2400, 2200, 2160]
    elif n == 6:
        widths = [1400, 1000, 1150, 1150, 2200, 2450]
    else:
        base = 9360 // n
        widths = [base] * n
        widths[-1] += 9360 - sum(widths)
    widths = scale_widths(widths)
    tbl = doc.add_table(rows=1, cols=n)
    tbl.style = "Table Grid"
    for i, h in enumerate(header):
        tbl.rows[0].cells[i].text = h
    for row in data:
        cells = tbl.add_row().cells
        for i in range(n):
            cells[i].text = row[i] if i < len(row) else ""
    apply_table_geometry(tbl, widths, indent=0)
    set_table_borders(tbl)
    for i, cell in enumerate(tbl.rows[0].cells):
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                set_run_font(r, name=BODY_FONT, size=10.5, bold=True)
    for row in tbl.rows[1:]:
        for i, cell in enumerate(row.cells):
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i in (0, n - 1) else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    set_run_font(r, name=BODY_FONT, size=10.5)
    return tbl


def add_blocks(doc, blocks):
    for block in blocks:
        kind = block[0]
        if kind == "h2":
            add_heading(doc, block[1], 2)
        elif kind == "h3":
            add_heading(doc, block[1], 3)
        elif kind == "para":
            add_rich_para(doc, block[1])
        elif kind == "bullets":
            for item in block[1]:
                add_bullet(doc, item)
        elif kind == "numbers":
            for item in block[1]:
                add_number(doc, item)
        elif kind == "table":
            add_md_table(doc, block[1])
            add_para(doc, "", after=4, first_line=None)
        elif kind == "code":
            lang = block[1]
            code_lines = block[2]
            if lang == "mermaid":
                add_para(doc, "（流程图见下图）", size=10, color=GRAY, italic=True, first_line=0.0, after=4)
            else:
                for ln in code_lines:
                    p = doc.add_paragraph()
                    p.paragraph_format.left_indent = Inches(0.18)
                    p.paragraph_format.right_indent = Inches(0.18)
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.line_spacing = 1.0
                    r = p.add_run(ln)
                    set_run_font(r, name="Consolas", size=9.5, color=DARK)
        elif kind == "rule":
            add_rule(doc)


def strip_md(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def parse_section(section_text: str):
    lines = section_text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            blocks.append(("blank", ""))
            i += 1
            continue
        if line.startswith("### "):
            blocks.append(("h2", line[4:].strip()))
            i += 1
            continue
        if line.startswith("#### "):
            blocks.append(("h3", line[5:].strip()))
            i += 1
            continue
        if line.startswith("```"):
            lang = line[3:].strip()
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", lang, code_lines))
            continue
        if line.startswith("|") and "|" in line:
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            blocks.append(("table", table_lines))
            continue
        if re.match(r"^\s*[-*] \S", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*] \S", lines[i]):
                items.append(strip_md(lines[i].strip()[2:].strip()))
                i += 1
            blocks.append(("bullets", items))
            continue
        if re.match(r"^\s*\d+\.\s+\S", line):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+\S", lines[i]):
                items.append(strip_md(re.sub(r"^\s*\d+\.\s+", "", lines[i]).strip()))
                i += 1
            blocks.append(("numbers", items))
            continue
        if line.strip() == "---":
            blocks.append(("rule", ""))
            i += 1
            continue
        paras = [strip_md(line.strip())]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip() or nxt.startswith(("### ", "#### ", "```")) or nxt.startswith("|") or re.match(r"^\s*[-*] \S", nxt) or re.match(r"^\s*\d+\.\s+\S", nxt) or nxt.strip() == "---":
                break
            paras.append(strip_md(nxt.strip()))
            i += 1
        blocks.append(("para", " ".join(paras)))
    return blocks


def split_sections(md_text: str):
    lines = md_text.splitlines()
    sections = []
    current_title = None
    current_lines = []
    for line in lines:
        if line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def main():
    if not SOURCE_MD.exists():
        raise FileNotFoundError(SOURCE_MD)
    md_text = SOURCE_MD.read_text(encoding="utf-8")
    sections = split_sections(md_text)
    fig1, fig2, fig3 = make_figures()

    doc = Document()
    apply_doc_style(doc)
    section = doc.sections[0]
    set_section(section)
    add_header_footer(section, "智能台灯系统设计与姿态检测方法研究")
    set_page_number_start(section, 1)

    # title page
    title = "面向学习场景的智能台灯系统设计与姿态检测方法研究"
    subtitle = "SmartLED 智能学习台灯详细结课报告"
    add_spacer(doc, 70)
    add_para(doc, "结课报告", size=36, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=None, after=24, font=TITLE_FONT)
    add_spacer(doc, 28)
    add_para(doc, title, size=18, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=None, after=10, font=HEADING_FONT)
    add_para(doc, subtitle, size=15, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=None, after=20, font=HEADING_FONT)
    add_para(doc, "基于 YOLO Pose、肩宽归一化与个性化标定的桌前学习姿态识别系统", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=None, after=34)

    meta = make_table(
        doc,
        ["项目名称", "SmartLED 智能学习台灯"],
        [["技术路线", "ESP32-S3 + Flask + Vue 3 + YOLO11n-pose"], ["生成日期", "2026-06-19"]],
        widths=[1600, 7760],
    )
    # title page meta table should be lightly styled
    for row_idx, row in enumerate(meta.rows):
        for col_idx, cell in enumerate(row.cells):
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    set_run_font(r, name=BODY_FONT, size=12, bold=(col_idx == 0))
    add_spacer(doc, 90)
    add_para(doc, "二〇二六年六月", size=15, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=None, font=HEADING_FONT)
    doc.add_page_break()

    # section order as in markdown
    for title, body in sections:
        add_heading(doc, title, 1)
        blocks = parse_section(body)
        add_blocks(doc, blocks)

        if title == "4 系统总体设计":
            add_image(doc, fig1, "图 1  SmartLED 系统总体架构示意图")
            add_image(doc, fig2, "图 2  姿态检测与状态融合处理流程图")
        if title == "11 实验结果与分析":
            add_image(doc, fig3, "图 3  评测结果摘要图")

    TARGET_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(TARGET_DOCX))
    print(TARGET_DOCX)


if __name__ == "__main__":
    main()
