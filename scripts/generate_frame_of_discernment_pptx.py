"""Generate the frame-of-discernment PowerPoint slide deck.

The repository does not store the generated ``.pptx`` binary because some PR
systems reject binary attachments. Run this script to recreate a downloadable
PowerPoint file locally:

    python scripts/generate_frame_of_discernment_pptx.py

The output is written to ``docs/frame_of_discernment_production.pptx`` by
default. Pass another path as the first argument to write elsewhere.
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

SLIDE_WIDTH = 12_192_000
SLIDE_HEIGHT = 6_858_000


def _text_box(shape_id: int, x: int, y: int, cx: int, cy: int, paragraphs: list[str], *, size: int = 2400, bold: bool = False, color: str = "1F2937") -> str:
    bold_attr = ' b="1"' if bold else ""
    runs = "".join(
        f"""<a:p><a:r><a:rPr lang=\"en-US\" sz=\"{size}\"{bold_attr}><a:solidFill><a:srgbClr val=\"{color}\"/></a:solidFill></a:rPr><a:t>{escape(paragraph)}</a:t></a:r><a:endParaRPr lang=\"en-US\" sz=\"{size}\"/></a:p>"""
        for paragraph in paragraphs
    )
    return f"""<p:sp><p:nvSpPr><p:cNvPr id=\"{shape_id}\" name=\"TextBox {shape_id}\"/><p:cNvSpPr txBox=\"1\"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x=\"{x}\" y=\"{y}\"/><a:ext cx=\"{cx}\" cy=\"{cy}\"/></a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr wrap=\"square\"/><a:lstStyle/>{runs}</p:txBody></p:sp>"""


def _process_box(shape_id: int, x: int, y: int, cx: int, cy: int, text: str, *, fill: str, line: str, size: int = 1900) -> str:
    return f"""<p:sp><p:nvSpPr><p:cNvPr id=\"{shape_id}\" name=\"Process {shape_id}\"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x=\"{x}\" y=\"{y}\"/><a:ext cx=\"{cx}\" cy=\"{cy}\"/></a:xfrm><a:prstGeom prst=\"roundRect\"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val=\"{fill}\"/></a:solidFill><a:ln w=\"19050\"><a:solidFill><a:srgbClr val=\"{line}\"/></a:solidFill></a:ln></p:spPr><p:txBody><a:bodyPr anchor=\"mid\"/><a:lstStyle/><a:p><a:pPr algn=\"ctr\"/><a:r><a:rPr lang=\"en-US\" sz=\"{size}\" b=\"1\"><a:solidFill><a:srgbClr val=\"0F172A\"/></a:solidFill></a:rPr><a:t>{escape(text)}</a:t></a:r></a:p></p:txBody></p:sp>"""


def _arrow(shape_id: int, x: int, y: int, cx: int, cy: int) -> str:
    return f"""<p:sp><p:nvSpPr><p:cNvPr id=\"{shape_id}\" name=\"Arrow {shape_id}\"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x=\"{x}\" y=\"{y}\"/><a:ext cx=\"{cx}\" cy=\"{cy}\"/></a:xfrm><a:prstGeom prst=\"rightArrow\"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val=\"64748B\"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:sp>"""


def _slide_xml() -> str:
    shapes = [
        _text_box(2, 420_000, 250_000, 11_300_000, 620_000, ["Production of the Frame of Discernment"], size=3600, bold=True, color="0F172A"),
        _text_box(3, 460_000, 850_000, 11_000_000, 420_000, ["In this work, the frame Θ is the configured set of mutually exclusive hypotheses used by the r-GCN/Dempster-Shafer mass head."], size=1850, color="334155"),
        _process_box(21, 550_000, 1_600_000, 2_100_000, 850_000, "1. Define candidate hypotheses\n(e.g., radar/aircraft/operator IDs)", fill="DBEAFE", line="2563EB"),
        _arrow(41, 2_750_000, 1_810_000, 620_000, 360_000),
        _process_box(22, 3_500_000, 1_600_000, 2_100_000, 850_000, "2. Encode focal elements\nas bit masks", fill="E0F2FE", line="0284C7"),
        _arrow(42, 5_700_000, 1_810_000, 620_000, 360_000),
        _process_box(23, 6_450_000, 1_600_000, 2_100_000, 850_000, "3. Bound mass-vector size\n≤10: all non-empty subsets", fill="DCFCE7", line="16A34A"),
        _arrow(43, 8_650_000, 1_810_000, 620_000, 360_000),
        _process_box(24, 9_400_000, 1_600_000, 2_100_000, 850_000, "4. Large frames\nsingletons + type groups + Θ", fill="FEF3C7", line="D97706"),
        _text_box(4, 620_000, 2_850_000, 5_200_000, 2_100_000, ["Implementation details:", "• subset_masks(hypotheses) builds the ordered focal-element list.", "• For n ≤ 10, masks 1..(2ⁿ−1) represent every non-empty subset.", "• For n > 10, singleton masks are retained, aircraft variants are grouped by inferred type, and the final mask represents full-frame uncertainty Θ."], size=1750, color="1E293B"),
        _text_box(5, 6_400_000, 2_850_000, 5_000_000, 2_100_000, ["Use in the workflow:", "• Observation ETL converts candidate match scores into DS masses [non-match, match, uncertain].", "• Training validates label vectors against len(subset_masks(hypotheses)).", "• Model outputs masses over the same ordered frame, then belief/plausibility intervals support classification and fusion."], size=1750, color="1E293B"),
        _process_box(25, 910_000, 5_480_000, 10_000_000, 600_000, "Result: a deterministic, compact frame of discernment that keeps evidential fusion tractable while preserving singleton hypotheses and uncertainty.", fill="EDE9FE", line="7C3AED", size=1750),
    ]
    return f"""<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<p:sld xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\"><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val=\"F8FAFC\"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/><a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>{''.join(shapes)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"""


def build_pptx(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = {
        "[Content_Types].xml": """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Override PartName=\"/ppt/presentation.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml\"/><Override PartName=\"/ppt/slides/slide1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slide+xml\"/><Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/><Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/></Types>""",
        "_rels/.rels": """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"ppt/presentation.xml\"/><Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/><Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" Target=\"docProps/app.xml\"/></Relationships>""",
        "ppt/presentation.xml": f"""<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><p:presentation xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\"><p:sldSz cx=\"{SLIDE_WIDTH}\" cy=\"{SLIDE_HEIGHT}\" type=\"wide\"/><p:notesSz cx=\"6858000\" cy=\"9144000\"/><p:sldIdLst><p:sldId id=\"256\" r:id=\"rId1\"/></p:sldIdLst></p:presentation>""",
        "ppt/_rels/presentation.xml.rels": """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide\" Target=\"slides/slide1.xml\"/></Relationships>""",
        "ppt/slides/slide1.xml": _slide_xml(),
        "ppt/slides/_rels/slide1.xml.rels": """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"/>""",
        "docProps/core.xml": """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:dcterms=\"http://purl.org/dc/terms/\" xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"><dc:title>Production of the Frame of Discernment</dc:title><dc:creator>OpenAI Codex</dc:creator><cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy></cp:coreProperties>""",
        "docProps/app.xml": """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\"><Application>OpenAI Codex</Application><PresentationFormat>Widescreen</PresentationFormat><Slides>1</Slides></Properties>""",
    }
    with ZipFile(output_path, "w", ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def main() -> None:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/frame_of_discernment_production.pptx")
    build_pptx(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
