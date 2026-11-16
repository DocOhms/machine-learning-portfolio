# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Render the manuscript's system design as vector PDF and editable SVG."""
from pathlib import Path
import argparse

from reportlab.graphics.shapes import Drawing, Rect, Line, Polygon, String, STATE_DEFAULTS
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# Calculation provenance:
#   Drawing -> reportlab.graphics.shapes.Drawing (third-party package)
#   HexColor -> reportlab.lib.colors.HexColor (third-party package)
#   Line -> reportlab.graphics.shapes.Line (third-party package)
#   Polygon -> reportlab.graphics.shapes.Polygon (third-party package)
#   Rect -> reportlab.graphics.shapes.Rect (third-party package)
#   String -> reportlab.graphics.shapes.String (third-party package)
#   TTFont -> reportlab.pdfbase.ttfonts.TTFont (third-party package)
#   arrow -> local function in this file
#   box -> local function in this file
#   lines -> local function in this file
#   pdfmetrics.registerFont -> reportlab.pdfbase.pdfmetrics.registerFont (third-party package)
#   renderPDF.drawToFile -> reportlab.graphics.renderPDF.drawToFile (third-party package)
#   renderSVG.drawToFile -> reportlab.graphics.renderSVG.drawToFile (third-party package)
#   segment -> local function in this file
#   text -> local function in this file
def make_figure(output):
    pdfmetrics.registerFont(TTFont("FrameworkArial", r"C:\Windows\Fonts\arial.ttf"))
    pdfmetrics.registerFont(TTFont("FrameworkArialBold", r"C:\Windows\Fonts\arialbd.ttf"))
    STATE_DEFAULTS["fontName"] = "FrameworkArial"
    width, height = 266, 346
    d = Drawing(width, height)
    ink = HexColor("#183146")
    blue = HexColor("#396A89")
    amber = HexColor("#977028")
    green = HexColor("#416D61")
    pale_blue = HexColor("#F0F5F8")
    pale_amber = HexColor("#FBF6EC")
    pale_green = HexColor("#F0F6F3")
    white = HexColor("#FFFFFF")

    # Calculation provenance:
    #   Rect -> reportlab.graphics.shapes.Rect (third-party package)
    def box(x, top, w, h, fill, stroke=blue, radius=3, dashed=False):
        d.add(Rect(x, height-top-h, w, h, rx=radius, ry=radius,
                   fillColor=fill, strokeColor=stroke, strokeWidth=0.65,
                   strokeDashArray=[3, 2] if dashed else None))

    # Calculation provenance:
    #   String -> reportlab.graphics.shapes.String (third-party package)
    def text(x, top, value, size=8.1, bold=False, anchor="middle", color=ink):
        d.add(String(x, height-top-size, value,
                     fontName="FrameworkArialBold" if bold else "FrameworkArial",
                     fontSize=size, textAnchor=anchor, fillColor=color))

    # Calculation provenance:
    #   text -> local function in this file
    def lines(x, top, values, size=8.0, leading=9.0, bold_first=False):
        for i, value in enumerate(values):
            text(x, top+i*leading, value, size, bold_first and i == 0)

    # Calculation provenance:
    #   Line -> reportlab.graphics.shapes.Line (third-party package)
    def segment(x1, y1, x2, y2, color=blue, dashed=False):
        d.add(Line(x1, height-y1, x2, height-y2, strokeColor=color,
                   strokeWidth=0.8, strokeDashArray=[3, 2] if dashed else None))

    # Calculation provenance:
    #   Polygon -> reportlab.graphics.shapes.Polygon (third-party package)
    #   segment -> local function in this file
    def arrow(x1, y1, x2, y2, color=blue):
        segment(x1, y1, x2, y2, color)
        if abs(x2-x1) < 0.01:
            tip = [x2, height-y2, x2-2.1, height-y2+4, x2+2.1, height-y2+4]
        else:
            direction = 1 if x2 > x1 else -1
            tip = [x2, height-y2, x2-direction*4, height-y2-2.1,
                   x2-direction*4, height-y2+2.1]
        d.add(Polygon(tip, fillColor=color, strokeColor=color, strokeWidth=0.2))

    # A separate right-hand route carries naive benchmarks past model fitting.
    segment(244, 41, 258, 41, amber)
    segment(258, 41, 258, 299.5, amber, dashed=True)
    arrow(258, 299.5, 244, 299.5, amber)

    box(8, 3, 236, 18, white)
    text(126, 6.5, "NSSP or ILINet surveillance", 8.6, True)
    arrow(126, 21, 126, 28)
    box(8, 28, 236, 26, pale_blue)
    lines(126, 31, ["Weekly features + chronological partition", "Observed inputs through completed week t - 1"], 8.0, 9.5, True)
    segment(126, 54, 126, 59)
    segment(87, 59, 211, 59)
    arrow(87, 59, 87, 64)
    arrow(211, 59, 211, 64, amber)

    # Phase 1: fit only on earlier pre-test weeks. Branches stay separate.
    box(3, 64, 246, 80, pale_blue)
    text(10, 68, "1  FIT: earlier 80% of pre-test weeks", 8.0, True, "start")
    box(10, 84, 154, 20, white)
    text(87, 89, "Fit scaler + K-means", 8.1, True)
    segment(87, 104, 87, 110)
    segment(32, 110, 142, 110)
    for x, label in ((10, "RF"), (65, "GBR"), (120, "MLP")):
        arrow(x+22, 110, x+22, 118)
        box(x, 118, 44, 18, white)
        text(x+22, 121.5, label, 8.3, True)
        segment(x+22, 136, x+22, 149)
    segment(32, 149, 142, 149)
    arrow(87, 149, 87, 154)
    box(179, 84, 65, 52, pale_amber, amber)
    lines(211.5, 88.5, ["Per-series", "ARIMA /", "SARIMA", "candidates"], 8.0, 10.0)
    arrow(211.5, 136, 211.5, 154, amber)

    # Phase 2: later chronological validation, not the MLP stopping subset.
    box(3, 154, 246, 57, pale_blue)
    text(10, 158, "2  VALIDATE: latest 20% of pre-test weeks", 8.0, True, "start")
    box(10, 173, 154, 31, white)
    lines(87, 177, ["Validation MSE", "Normalized inverse-MSE weights"], 8.0, 10.0)
    box(179, 173, 65, 31, pale_amber, amber)
    lines(211.5, 175, ["RMSE / MAE", "+ coverage", "Select orders"], 7.9, 9.0)
    segment(87, 204, 87, 217)
    segment(211.5, 204, 211.5, 217, amber)
    segment(87, 217, 211.5, 217)
    arrow(126, 217, 126, 223)

    # Phase 3: rebuilding the fitted objects is distinct from validation.
    box(8, 223, 236, 32, pale_blue)
    text(126, 226, "3  REFIT: complete pre-test history", 8.2, True)
    lines(126, 237, ["Freeze weights/orders; rebuild preprocessing", "and hybrid / classical models"], 7.6, 8.5)
    arrow(126, 255, 126, 267)

    # Phase 4: held-out scoring, while completed observations may become history.
    box(3, 267, 246, 74, pale_green, green, dashed=True)
    text(10, 271, "4  FINAL TEST: rolling one-week forecasts", 8.0, True, "start", green)
    box(10, 286, 154, 27, white, green)
    lines(87, 289, ["Hybrid + classical forecasts", "Fixed parameters; history/state updates"], 7.65, 9.5)
    box(179, 286, 65, 27, pale_amber, amber)
    lines(211.5, 287, ["Persistence", "MA(3)", "Seasonal naive"], 7.6, 8.2)
    segment(87, 313, 87, 317, green)
    segment(211.5, 313, 211.5, 317, amber)
    segment(87, 317, 211.5, 317, green)
    arrow(126, 317, 126, 322, green)
    box(10, 322, 234, 17, white, green)
    text(127, 326, "Same-row metrics + descriptive state maps", 7.9, True)

    output.mkdir(parents=True, exist_ok=True)
    renderPDF.drawToFile(d, str(output / "framework_vector.pdf"))
    renderSVG.drawToFile(d, str(output / "framework_vector.svg"))
    svg = output / "framework_vector.svg"
    content = svg.read_text(encoding="utf-8")
    content = content.replace("font-family: FrameworkArialBold;", "font-family: Arial, Helvetica, sans-serif; font-weight: bold;")
    content = content.replace("font-family: FrameworkArial;", "font-family: Arial, Helvetica, sans-serif;")
    svg.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    make_figure(parser.parse_args().output_dir)
