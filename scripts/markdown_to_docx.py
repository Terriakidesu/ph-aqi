from __future__ import annotations

import html
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_MD = PROJECT_ROOT / "paper_city_first_with_figures.md"
OUTPUT_DOCX = PROJECT_ROOT / "paper_city_first_with_figures.docx"
TEMPLATE_DOCX = PROJECT_ROOT / "conference-template-a4.docx"
TEMP_DIR = PROJECT_ROOT / ".docx_build"

EMU_PER_INCH = 914400
MAX_IMAGE_WIDTH_EMU = int(6.2 * EMU_PER_INCH)


@dataclass
class ImageRel:
    rid: str
    source: Path
    target_name: str
    width_emu: int
    height_emu: int


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def clean_inline(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = text.replace("`", "")
    return text


def paragraph(text: str, style: str | None = None, align: str | None = None) -> str:
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
    align_xml = f'<w:jc w:val="{align}"/>' if align else ""
    props = f"<w:pPr>{style_xml}{align_xml}</w:pPr>" if style_xml or align_xml else ""
    text = clean_inline(text)
    return f"<w:p>{props}<w:r><w:t xml:space=\"preserve\">{esc(text)}</w:t></w:r></w:p>"


def heading(text: str, level: int) -> str:
    style = "Title" if level == 1 else f"Heading{min(level, 3)}"
    return paragraph(text, style=style)


def table(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    grid = "".join("<w:gridCol w:w=\"2400\"/>" for _ in rows[0])
    row_xml = []
    for row in rows:
        cells = []
        for cell in row:
            cells.append(
                "<w:tc>"
                "<w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
                f"{paragraph(cell)}"
                "</w:tc>"
            )
        row_xml.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return (
        "<w:tbl>"
        "<w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "</w:tblBorders></w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        f"{''.join(row_xml)}"
        "</w:tbl>"
    )


def image_run(rel: ImageRel, descr: str) -> str:
    doc_id_match = re.search(r"(\d+)$", rel.rid)
    doc_id = doc_id_match.group(1) if doc_id_match else "1"
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{rel.width_emu}" cy="{rel.height_emu}"/>
        <wp:docPr id="{doc_id}" name="{esc(descr)}"/>
        <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="{esc(rel.target_name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel.rid}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{rel.width_emu}" cy="{rel.height_emu}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        cells = [clean_inline(cell.strip()) for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    return rows


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        px_w, px_h = img.size
    width_emu = min(MAX_IMAGE_WIDTH_EMU, int(px_w / 96 * EMU_PER_INCH))
    height_emu = int(width_emu * px_h / px_w)
    return width_emu, height_emu


def render_markdown(md_text: str) -> tuple[str, list[ImageRel]]:
    body = []
    images: list[ImageRel] = []
    lines = md_text.splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []
    in_equation = False
    equation_lines: list[str] = []

    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("```"):
            if in_code:
                body.append(paragraph("\n".join(code_lines), style="Code"))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if line.strip() == "$$":
            if in_equation:
                body.append(paragraph(" ".join(equation_lines), style="Equation", align="center"))
                equation_lines = []
                in_equation = False
            else:
                in_equation = True
            i += 1
            continue

        if in_equation:
            equation_lines.append(line)
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        img_match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if img_match:
            alt, path_text = img_match.groups()
            source = (PROJECT_ROOT / path_text).resolve()
            if source.exists():
                rid = f"rIdPaperImage{len(images) + 1}"
                target_name = f"paper_image{len(images) + 1}{source.suffix.lower()}"
                width_emu, height_emu = image_size(source)
                rel = ImageRel(rid, source, target_name, width_emu, height_emu)
                images.append(rel)
                body.append(image_run(rel, alt))
            else:
                body.append(paragraph(f"[Missing image: {path_text}]"))
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            rows = parse_table(table_lines)
            if len(rows) == 1 and len(rows[0]) == 1 and rows[0][0].lower().startswith("table "):
                body.append(paragraph(rows[0][0], style="Caption", align="center"))
            else:
                body.append(table(rows))
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            hashes, text = heading_match.groups()
            body.append(heading(text, len(hashes)))
            i += 1
            continue

        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:].strip())
                i += 1
            for item in items:
                body.append(paragraph(f"- {item}"))
            continue

        paragraph_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "!", "```", "$$")):
            paragraph_lines.append(lines[i].strip())
            i += 1
        body.append(paragraph(" ".join(paragraph_lines)))

    return "\n".join(body), images


def section_properties_from_template() -> str:
    if not TEMPLATE_DOCX.exists():
        return (
            '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
            'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
        )

    with zipfile.ZipFile(TEMPLATE_DOCX) as template:
        template_xml = template.read("word/document.xml").decode("utf-8")

    match = re.search(r"<w:sectPr[\s\S]*?</w:sectPr>", template_xml)
    if match:
        return match.group(0)

    return (
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
    )


def document_xml(body_xml: str) -> str:
    sect_pr = section_properties_from_template()
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
  <w:body>
    {body_xml}
    {sect_pr}
  </w:body>
</w:document>
"""


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:pPr><w:jc w:val="center"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:rPr><w:b/><w:sz w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:rPr><w:b/><w:sz w:val="23"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="Caption"/>
    <w:rPr><w:i/><w:sz w:val="20"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Code">
    <w:name w:val="Code"/>
    <w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="20"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Equation">
    <w:name w:val="Equation"/>
    <w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:sz w:val="22"/></w:rPr>
  </w:style>
</w:styles>
"""


def content_types(images: Iterable[ImageRel]) -> str:
    if TEMPLATE_DOCX.exists():
        with zipfile.ZipFile(TEMPLATE_DOCX) as template:
            content = template.read("[Content_Types].xml").decode("utf-8")
    else:
        content = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""

    image_defaults = set()
    for image in images:
        ext = image.source.suffix.lower().lstrip(".")
        content_type = "image/png" if ext == "png" else f"image/{ext}"
        image_defaults.add((ext, content_type))

    for ext, content_type in sorted(image_defaults):
        if f'Extension="{ext}"' not in content:
            content = content.replace(
                "</Types>",
                f'  <Default Extension="{ext}" ContentType="{content_type}"/>\n</Types>',
            )

    return content


def package_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def document_rels(images: list[ImageRel]) -> str:
    rels_path = TEMP_DIR / "word" / "_rels" / "document.xml.rels"
    if rels_path.exists():
        content = rels_path.read_text(encoding="utf-8")
    else:
        content = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""

    additions = []
    for image in images:
        if f'Id="{image.rid}"' in content:
            continue
        additions.append(
            f'<Relationship Id="{image.rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{image.target_name}"/>'
        )

    if additions:
        content = content.replace("</Relationships>", "\n  " + "\n  ".join(additions) + "\n</Relationships>")

    return content


def write_docx() -> None:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    if TEMPLATE_DOCX.exists():
        with zipfile.ZipFile(TEMPLATE_DOCX) as template:
            template.extractall(TEMP_DIR)
    else:
        (TEMP_DIR / "_rels").mkdir(parents=True)
        (TEMP_DIR / "word" / "_rels").mkdir(parents=True)

    (TEMP_DIR / "word" / "_rels").mkdir(parents=True, exist_ok=True)
    media_dir = TEMP_DIR / "word" / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    body_xml, images = render_markdown(INPUT_MD.read_text(encoding="utf-8"))

    (TEMP_DIR / "[Content_Types].xml").write_text(content_types(images), encoding="utf-8")
    if not (TEMP_DIR / "_rels" / ".rels").exists():
        (TEMP_DIR / "_rels" / ".rels").write_text(package_rels(), encoding="utf-8")
    (TEMP_DIR / "word" / "document.xml").write_text(document_xml(body_xml), encoding="utf-8")
    if not (TEMP_DIR / "word" / "styles.xml").exists():
        (TEMP_DIR / "word" / "styles.xml").write_text(styles_xml(), encoding="utf-8")
    (TEMP_DIR / "word" / "_rels" / "document.xml.rels").write_text(document_rels(images), encoding="utf-8")

    for image in images:
        shutil.copy2(image.source, TEMP_DIR / "word" / "media" / image.target_name)

    if OUTPUT_DOCX.exists():
        OUTPUT_DOCX.unlink()

    with zipfile.ZipFile(OUTPUT_DOCX, "w", zipfile.ZIP_DEFLATED) as docx:
        for path in TEMP_DIR.rglob("*"):
            if path.is_file():
                docx.write(path, path.relative_to(TEMP_DIR).as_posix())

    shutil.rmtree(TEMP_DIR)
    print(f"Saved DOCX to: {OUTPUT_DOCX}")


if __name__ == "__main__":
    write_docx()
