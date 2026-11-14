#!/usr/bin/env python3
"""Build the editable CUDA GEMM project presentation from the Xidian template."""

from __future__ import annotations

import argparse
import csv
import tempfile
import zipfile
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import ChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


EMU_PER_INCH = 914400
FONT = "Noto Sans CJK SC"
MONO_FONT = "Noto Sans Mono CJK SC"

RED = RGBColor(180, 37, 42)
DARK_RED = RGBColor(141, 27, 32)
BLACK = RGBColor(35, 39, 43)
DARK_GRAY = RGBColor(83, 88, 93)
MID_GRAY = RGBColor(153, 158, 163)
LIGHT_GRAY = RGBColor(231, 233, 235)
PALE_GRAY = RGBColor(246, 247, 248)
PALE_RED = RGBColor(250, 236, 237)
PALE_GREEN = RGBColor(232, 244, 232)
GREEN = RGBColor(76, 133, 78)
BLUE = RGBColor(53, 104, 151)
PALE_BLUE = RGBColor(231, 240, 248)
WHITE = RGBColor(255, 255, 255)


def _set_run_font(run, size, color=BLACK, bold=False, font=FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_text(
    slide,
    x,
    y,
    w,
    h,
    text,
    size=12,
    color=BLACK,
    bold=False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.MIDDLE,
    font=FONT,
    margin=0.02,
    line_spacing=1.0,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.line_spacing = line_spacing
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(0)
        run = paragraph.add_run()
        run.text = line
        _set_run_font(run, size, color, bold, font)
    return box


def add_rich_text(
    slide,
    x,
    y,
    w,
    h,
    lines,
    valign=MSO_ANCHOR.MIDDLE,
    align=PP_ALIGN.LEFT,
    margin=0.04,
):
    """Add a text box whose lines are lists of (text, size, color, bold, font)."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    for line_index, segments in enumerate(lines):
        paragraph = frame.paragraphs[0] if line_index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(0)
        for text, size, color, bold, font in segments:
            run = paragraph.add_run()
            run.text = text
            _set_run_font(run, size, color, bold, font)
    return box


def add_box(
    slide,
    x,
    y,
    w,
    h,
    fill=WHITE,
    line=LIGHT_GRAY,
    radius=True,
    line_width=0.8,
):
    shape_type = (
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE
        if radius
        else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    )
    shape = slide.shapes.add_shape(
        shape_type, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(line_width)
    return shape


def add_line(slide, x1, y1, x2, y2, color=LIGHT_GRAY, width=1.0):
    line = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE,
        Inches(x1),
        Inches(y1),
        Inches(max(x2 - x1, 0.012)),
        Inches(max(y2 - y1, 0.012)),
    )
    line.fill.solid()
    line.fill.fore_color.rgb = color
    line.line.fill.background()
    if abs(x2 - x1) < abs(y2 - y1):
        line.width = Inches(width / 72.0)
    else:
        line.height = Inches(width / 72.0)
    return line


def add_arrow(slide, x, y, w=0.25, h=0.22, color=RED, direction="right"):
    kind = (
        MSO_AUTO_SHAPE_TYPE.DOWN_ARROW
        if direction == "down"
        else MSO_AUTO_SHAPE_TYPE.RIGHT_ARROW
    )
    arrow = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = color
    arrow.line.fill.background()
    return arrow


def add_circle_label(slide, x, y, diameter, text, fill=RED, size=10):
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.OVAL,
        Inches(x), Inches(y), Inches(diameter), Inches(diameter)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    add_text(
        slide, x, y, diameter, diameter, text,
        size=size, color=WHITE, bold=True, align=PP_ALIGN.CENTER
    )
    return shape


def add_card(slide, x, y, w, h, title, body, accent=RED, title_size=12, body_size=9):
    add_box(slide, x, y, w, h, WHITE, LIGHT_GRAY)
    add_line(slide, x + 0.14, y + 0.16, x + 0.58, y + 0.16, accent, 2.2)
    add_text(slide, x + 0.14, y + 0.27, w - 0.28, 0.34, title,
             title_size, accent, True)
    add_text(slide, x + 0.14, y + 0.66, w - 0.28, h - 0.76, body,
             body_size, DARK_GRAY, False, valign=MSO_ANCHOR.TOP,
             line_spacing=1.05)


def remove_all_slides(prs):
    slide_ids = prs.slides._sldIdLst
    for slide_id in list(slide_ids):
        prs.part.drop_rel(slide_id.rId)
        slide_ids.remove(slide_id)


def add_background(slide, image_path, prs):
    slide.shapes.add_picture(
        str(image_path), 0, 0, width=prs.slide_width, height=prs.slide_height
    )


def add_content_slide(prs, bg_path, title, page):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_background(slide, bg_path, prs)
    add_text(slide, 3.08, 0.06, 6.55, 0.57, title,
             22, WHITE, True, PP_ALIGN.RIGHT)
    add_text(slide, 8.40, 7.02, 1.05, 0.20, f"{page} / 8 ·",
             8.5, MID_GRAY, False, PP_ALIGN.RIGHT)
    return slide


def add_kpi(slide, x, value, label):
    add_line(slide, x, 1.77, x + 1.50, 1.77, RED, 1.7)
    add_text(slide, x, 1.90, 1.50, 0.43, value, 22, RED, True)
    add_text(slide, x, 2.36, 1.50, 0.27, label, 9, MID_GRAY)


def load_performance(csv_path):
    fallback = {
        "sizes": [256, 512, 1024, 2048, 4096],
        "custom": [1216.784, 3435.589, 4433.163, 4343.676, 3582.034],
        "fp32": [2899.823, 5435.289, 8629.545, 7757.533, 5678.101],
        "tf32": [4206.418, 8125.977, 14029.649, 9509.546, 11577.689],
    }
    if not csv_path.exists():
        return fallback

    grouped = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            size = int(row["m"])
            grouped.setdefault(size, {"custom": [], "fp32": [], "tf32": []})
            kind = row["kind"]
            value = float(row["gflops"])
            if kind == "custom_fp32" and row["passed"] == "1":
                grouped[size]["custom"].append(value)
            elif kind == "cublas_fp32":
                grouped[size]["fp32"].append(value)
            elif kind == "cublas_tf32":
                grouped[size]["tf32"].append(value)

    sizes = sorted(size for size in grouped if all(grouped[size].values()))
    if not sizes:
        return fallback
    return {
        "sizes": sizes,
        "custom": [max(grouped[size]["custom"]) for size in sizes],
        "fp32": [grouped[size]["fp32"][-1] for size in sizes],
        "tf32": [grouped[size]["tf32"][-1] for size in sizes],
    }


def add_performance_chart(slide, x, y, w, h, performance):
    data = ChartData()
    data.categories = [str(value) for value in performance["sizes"]]
    data.add_series("Custom FP32", performance["custom"])
    data.add_series("cuBLAS FP32", performance["fp32"])
    data.add_series("cuBLAS TF32", performance["tf32"])
    graphic = slide.shapes.add_chart(
        XL_CHART_TYPE.LINE_MARKERS,
        Inches(x), Inches(y), Inches(w), Inches(h), data
    )
    chart = graphic.chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.name = FONT
    chart.legend.font.size = Pt(8)
    chart.chart_style = 10
    chart.has_title = True
    chart.chart_title.text_frame.text = "多矩阵规模性能对比（GFLOPS）"
    for paragraph in chart.chart_title.text_frame.paragraphs:
        for run in paragraph.runs:
            _set_run_font(run, 11, BLACK, True)

    value_axis = chart.value_axis
    value_axis.minimum_scale = 0
    value_axis.maximum_scale = 16000
    value_axis.major_unit = 4000
    value_axis.has_major_gridlines = True
    value_axis.tick_labels.font.name = FONT
    value_axis.tick_labels.font.size = Pt(7.5)
    category_axis = chart.category_axis
    category_axis.tick_labels.font.name = FONT
    category_axis.tick_labels.font.size = Pt(7.5)

    colors = [RED, BLACK, GREEN]
    markers = [XL_MARKER_STYLE.CIRCLE, XL_MARKER_STYLE.SQUARE, XL_MARKER_STYLE.DIAMOND]
    for series, color, marker in zip(chart.series, colors, markers):
        series.format.line.color.rgb = color
        series.format.line.width = Pt(2.2)
        series.marker.style = marker
        series.marker.size = 6
        series.marker.format.fill.solid()
        series.marker.format.fill.fore_color.rgb = color
        series.marker.format.line.color.rgb = color
    add_text(slide, x + w - 1.13, y + h - 0.26, 1.05, 0.18,
             "矩阵边长 N", 7.5, MID_GRAY, False, PP_ALIGN.RIGHT)


def add_comparison_bar(slide, x, y, label, left_value, right_value, maximum, unit, invert=False):
    add_text(slide, x, y, 1.05, 0.25, label, 9.5, BLACK, True)
    bar_x = x + 1.10
    max_w = 2.95
    left_w = max(0.03, max_w * left_value / maximum)
    right_w = max(0.03, max_w * right_value / maximum)
    add_box(slide, bar_x, y + 0.02, max_w, 0.18, PALE_GRAY, PALE_GRAY, False, 0)
    add_box(slide, bar_x, y + 0.02, left_w, 0.18, RED, RED, False, 0)
    add_text(slide, bar_x + max_w + 0.08, y - 0.01, 0.72, 0.24,
             f"{left_value:g}{unit}", 8.5, RED, True)
    add_box(slide, bar_x, y + 0.32, max_w, 0.18, PALE_GRAY, PALE_GRAY, False, 0)
    add_box(slide, bar_x, y + 0.32, right_w, 0.18, DARK_GRAY, DARK_GRAY, False, 0)
    add_text(slide, bar_x + max_w + 0.08, y + 0.29, 0.72, 0.24,
             f"{right_value:g}{unit}", 8.5, DARK_GRAY, True)


def add_simple_roofline(slide, x, y, w, h):
    add_box(slide, x, y, w, h, WHITE, LIGHT_GRAY)
    add_text(slide, x + 0.15, y + 0.10, w - 0.30, 0.30,
             "DRAM Roofline（示意）", 10.5, BLACK, True)
    ox, oy = x + 0.36, y + h - 0.43
    top, right = y + 0.58, x + w - 0.20
    add_line(slide, ox, top, ox, oy, DARK_GRAY, 0.8)
    add_line(slide, ox, oy, right, oy, DARK_GRAY, 0.8)
    ridge_x, ridge_y = x + 1.13, y + 1.13
    # Diagonal line is approximated with a thin right triangle and a horizontal roof.
    roof = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.LINE_INVERSE,
        Inches(ox), Inches(ridge_y), Inches(ridge_x - ox), Inches(oy - ridge_y)
    )
    roof.line.color.rgb = BLUE
    roof.line.width = Pt(2.0)
    add_line(slide, ridge_x, ridge_y, right, ridge_y, GREEN, 2.0)
    add_text(slide, ridge_x - 0.18, ridge_y + 0.10, 0.78, 0.20,
             "ridge", 7.2, MID_GRAY)
    # Both measured configurations are to the right of the ridge and below the roof.
    add_circle_label(slide, x + 1.72, y + 1.77, 0.16, "", RED, 1)
    add_text(slide, x + 1.91, y + 1.70, 0.55, 0.26, "大 tile", 7.5, RED, True)
    square = add_box(slide, x + 1.72, y + 2.13, 0.16, 0.16,
                     DARK_GRAY, DARK_GRAY, False, 0)
    add_text(slide, x + 1.91, y + 2.05, 0.58, 0.26, "小 tile", 7.5, DARK_GRAY, True)
    add_text(slide, x + 0.40, y + h - 0.28, w - 0.56, 0.18,
             "算术强度 AI →", 7.2, MID_GRAY, False, PP_ALIGN.RIGHT)
    add_text(slide, x + 0.05, y + 0.45, 0.28, 1.65,
             "性\n能", 7.2, MID_GRAY, False, PP_ALIGN.CENTER)


def add_stall_bar(slide, x, y, label, large, small, maximum=60.0):
    add_text(slide, x, y - 0.03, 1.14, 0.30, label, 9.3, BLACK, True)
    start = x + 1.20
    max_w = 3.45
    for offset, value, color in [(0.00, large, RED), (0.27, small, DARK_GRAY)]:
        add_box(slide, start, y + offset, max_w, 0.16,
                PALE_GRAY, PALE_GRAY, False, 0)
        add_box(slide, start, y + offset, max(0.03, max_w * value / maximum), 0.16,
                color, color, False, 0)
        add_text(slide, start + max_w + 0.08, y + offset - 0.04, 0.62, 0.23,
                 f"{value:.2f}%", 8.2, color, True)


def build(template_path: Path, output_path: Path, csv_path: Path):
    performance = load_performance(csv_path)
    prs = Presentation(str(template_path))
    remove_all_slides(prs)
    prs.core_properties.title = "CUDA GEMM 高性能优化与参数化 Auto-Tuning"
    prs.core_properties.subject = "CUDA GEMM、cuBLAS、Nsight Compute 与 Roofline"
    prs.core_properties.author = "赵浩锦"

    with tempfile.TemporaryDirectory(prefix="gemm-ppt-") as temporary:
        temp_dir = Path(temporary)
        with zipfile.ZipFile(template_path) as archive:
            title_bg = temp_dir / "title-background.png"
            content_bg = temp_dir / "content-background.jpeg"
            title_bg.write_bytes(archive.read("ppt/media/image1.png"))
            content_bg.write_bytes(archive.read("ppt/media/image3.jpeg"))

        # Slide 1: cover
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        add_background(slide, title_bg, prs)
        add_text(slide, 1.07, 1.92, 5.55, 1.20,
                 "CUDA GEMM 高性能优化\n与参数化 Auto-Tuning",
                 29, BLACK, True, valign=MSO_ANCHOR.TOP)
        add_text(slide, 1.08, 3.44, 2.60, 0.44,
                 "赵浩锦", 25, RED, True)
        add_text(slide, 1.08, 3.98, 4.35, 0.56,
                 "西安电子科技大学｜计算机科学与技术",
                 15, DARK_GRAY)
        add_text(slide, 1.07, 5.10, 5.70, 0.60,
                 "从 Naive 到 Warp Tiling、Tensor Core 与 Roofline",
                 13, BLACK, True)
        add_box(slide, 6.82, 2.06, 2.08, 2.08, PALE_RED, RED, True, 2.2)
        add_text(slide, 7.08, 2.36, 1.56, 0.48,
                 "A × B", 26, RED, True, PP_ALIGN.CENTER)
        add_text(slide, 7.08, 2.86, 1.56, 0.42,
                 "= C", 25, BLACK, True, PP_ALIGN.CENTER)
        for row in range(4):
            for col in range(4):
                color = RED if (row + col) % 3 == 0 else MID_GRAY
                add_box(slide, 7.32 + col * 0.27, 3.42 + row * 0.14,
                        0.18, 0.08, color, color, False, 0)
        add_text(slide, 6.55, 5.45, 3.05, 0.58,
                 "CUDA C++  ×  cuBLAS\nNCU  ×  Roofline",
                 14, RED, True, PP_ALIGN.CENTER)

        # Slide 2: overview
        slide = add_content_slide(prs, content_bg, "项目概览", 2)
        add_text(slide, 0.77, 1.02, 8.55, 0.45,
                 "以 GEMM 为载体，建立“实现—测量—分析—再优化”的 GPU 性能工程闭环",
                 15.5, BLACK, True)
        add_kpi(slide, 0.78, "8", "递进优化版本")
        add_kpi(slide, 2.90, "11", "Auto-Tuning 配置")
        add_kpi(slide, 5.02, "4.43", "最高自定义 TFLOPS")
        add_kpi(slide, 7.15, "51–63%", "中大型矩阵 / cuBLAS FP32")
        steps = [
            ("01", "参数化 Kernel", "BM / BN / BK\nTM / TN"),
            ("02", "统一基准", "预热 + Event\n误差检查"),
            ("03", "自动搜索", "11 配置 ×\n5 种规模"),
            ("04", "工业对照", "cuBLAS FP32\n与 TF32"),
            ("05", "瓶颈解释", "NCU +\nRoofline"),
        ]
        for index, (number, title, body) in enumerate(steps):
            x = 0.70 + index * 1.76
            add_circle_label(slide, x, 3.27, 0.36, number, RED, 8.5)
            add_card(slide, x + 0.45, 3.16, 1.23, 1.58,
                     title, body, RED, 10.3, 8.3)
            if index < len(steps) - 1:
                add_arrow(slide, x + 1.67, 3.82, 0.18, 0.15, MID_GRAY)
        add_box(slide, 0.70, 5.22, 8.65, 0.80, BLACK, BLACK)
        add_rich_text(slide, 0.91, 5.34, 8.20, 0.53, [[
            ("目标｜", 11.5, WHITE, True, FONT),
            ("不仅给出最快参数，更用可复现实验解释“", 11.5, WHITE, False, FONT),
            ("为什么更快", 11.5, RGBColor(255, 202, 204), True, FONT),
            ("”。", 11.5, WHITE, False, FONT),
        ]], align=PP_ALIGN.CENTER)
        add_text(slide, 0.80, 6.25, 8.40, 0.28,
                 "环境｜RTX 4060 Laptop GPU · CUDA 13 · cuBLAS 13 · Nsight Compute 2026.2.1",
                 9.3, MID_GRAY, False, PP_ALIGN.CENTER)

        # Slide 3: optimization roadmap
        slide = add_content_slide(prs, content_bg, "优化路线：从访存到计算单元", 3)
        add_text(slide, 0.77, 1.00, 8.55, 0.42,
                 "逐级扩大数据复用范围，并将数据搬运、线程组织与硬件计算能力对齐",
                 14.5, BLACK, True)
        versions = [
            ("V0", "Naive", "1 thread\n→ 1 C", WHITE, DARK_GRAY),
            ("V1", "Shared Tiling", "Block 内\n复用 A/B", PALE_RED, RED),
            ("V2", "Register Tiling", "1 thread\n→ TM×TN", PALE_RED, RED),
            ("V3", "float4 Load", "16 Byte\n连续加载", PALE_RED, RED),
            ("V4", "Double Buffer", "寄存器预取\n双缓冲", PALE_BLUE, BLUE),
            ("V5", "cp.async", "Global→Shared\n异步搬运", PALE_BLUE, BLUE),
            ("V6", "Warp Tiling", "Warp 负责\n独立 C tile", PALE_GREEN, GREEN),
            ("V7", "Tensor Core", "WMMA\nFP16→FP32", RED, WHITE),
        ]
        for index, (version, title, body, fill, accent) in enumerate(versions):
            row = index // 4
            col = index % 4
            x = 0.68 + col * 2.20
            y = 1.62 + row * 1.73
            add_box(slide, x, y, 1.92, 1.30, fill,
                    RED if index == 7 else LIGHT_GRAY)
            add_text(slide, x + 0.12, y + 0.12, 0.52, 0.30,
                     version, 12, accent, True)
            add_text(slide, x + 0.64, y + 0.10, 1.14, 0.34,
                     title, 10.5, accent, True, PP_ALIGN.RIGHT)
            add_text(slide, x + 0.14, y + 0.53, 1.64, 0.58,
                     body, 9.3, accent if index == 7 else DARK_GRAY,
                     False, PP_ALIGN.CENTER)
            if col < 3:
                add_arrow(slide, x + 1.95, y + 0.54, 0.18, 0.18,
                          RED if row == 0 else BLUE)
        add_arrow(slide, 8.50, 2.98, 0.18, 0.29, RED, "down")
        add_text(slide, 0.78, 5.23, 1.05, 0.26,
                 "数据路径", 10.5, RED, True)
        pipeline = [
            ("Global Memory", PALE_GRAY, DARK_GRAY),
            ("Shared Memory", PALE_RED, RED),
            ("Registers", PALE_BLUE, BLUE),
            ("FP32 FMA / Tensor Core", PALE_GREEN, GREEN),
        ]
        px = 1.82
        widths = [1.45, 1.48, 1.16, 2.10]
        for i, ((label, fill, accent), width) in enumerate(zip(pipeline, widths)):
            add_box(slide, px, 5.10, width, 0.55, fill, accent)
            add_text(slide, px + 0.06, 5.16, width - 0.12, 0.39,
                     label, 9.3, accent, True, PP_ALIGN.CENTER)
            px += width
            if i < len(pipeline) - 1:
                add_arrow(slide, px + 0.06, 5.28, 0.24, 0.16, MID_GRAY)
                px += 0.39
        add_box(slide, 0.70, 6.03, 8.65, 0.48, PALE_RED, PALE_RED)
        add_text(slide, 0.88, 6.10, 8.28, 0.30,
                 "核心思路｜让更昂贵的上层存储服务更多 FMA，同时减少同步与重复搬运",
                 10.8, DARK_RED, True, PP_ALIGN.CENTER)

        # Slide 4: kernel internals
        slide = add_content_slide(prs, content_bg, "核心 Kernel：分块与寄存器外积", 4)
        add_text(slide, 0.77, 1.00, 8.55, 0.42,
                 "一个 Block 累加 C 的 BM×BN 子块；一次只搬入长度 BK 的 K 方向切片",
                 14.3, BLACK, True)
        matrix_items = [
            (0.78, 1.73, 2.18, 2.03, "A tile", "BM × BK", PALE_BLUE, BLUE),
            (3.50, 1.73, 2.18, 2.03, "B tile", "BK × BN", PALE_RED, RED),
            (6.34, 1.73, 2.18, 2.03, "C tile", "BM × BN", PALE_GREEN, GREEN),
        ]
        for x, y, w, h, title, dims, fill, accent in matrix_items:
            add_box(slide, x, y, w, h, fill, accent, False, 1.4)
            add_text(slide, x + 0.15, y + 0.22, w - 0.30, 0.34,
                     title, 13, accent, True, PP_ALIGN.CENTER)
            add_text(slide, x + 0.15, y + 0.76, w - 0.30, 0.50,
                     dims, 21, accent, True, PP_ALIGN.CENTER)
            # A small editable grid hints at tiled matrices.
            for row in range(3):
                for col in range(5):
                    cell_fill = accent if (row == 1 or col in (1, 2)) else WHITE
                    add_box(slide, x + 0.43 + col * 0.25, y + 1.43 + row * 0.13,
                            0.19, 0.075, cell_fill, cell_fill, False, 0)
        add_text(slide, 3.00, 2.47, 0.42, 0.38, "×", 23, BLACK, True, PP_ALIGN.CENTER)
        add_arrow(slide, 5.83, 2.53, 0.36, 0.23, GREEN)
        add_text(slide, 0.78, 4.08, 1.02, 0.28, "K 方向", 10.5, RED, True)
        for index, label in enumerate(["0", "BK", "2BK", "…", "K"]):
            x = 1.82 + index * 1.13
            add_box(slide, x, 4.00, 0.80, 0.46,
                    PALE_RED if index < 3 else PALE_GRAY,
                    RED if index < 3 else MID_GRAY)
            add_text(slide, x, 4.05, 0.80, 0.33, label,
                     9.5, RED if index < 3 else DARK_GRAY, True, PP_ALIGN.CENTER)
            if index < 4:
                add_arrow(slide, x + 0.86, 4.15, 0.18, 0.14, MID_GRAY)
        add_text(slide, 7.58, 4.04, 1.43, 0.35,
                 "循环 K / BK 次", 9.3, DARK_GRAY, True, PP_ALIGN.CENTER)
        stages = [
            ("Global", "float4", PALE_GRAY, DARK_GRAY, 1.30),
            ("Shared", "A/B tiles", PALE_RED, RED, 1.40),
            ("Registers", "regA / regB", PALE_BLUE, BLUE, 1.50),
            ("Outer Product", "TM × TN FMA", PALE_GREEN, GREEN, 1.72),
            ("C", "write back", WHITE, RED, 1.18),
        ]
        sx = 0.78
        for index, (title, body, fill, accent, width) in enumerate(stages):
            add_box(slide, sx, 4.83, width, 0.83, fill, accent)
            add_text(slide, sx + 0.06, 4.91, width - 0.12, 0.26,
                     title, 9.8, accent, True, PP_ALIGN.CENTER)
            add_text(slide, sx + 0.06, 5.22, width - 0.12, 0.24,
                     body, 8.3, DARK_GRAY if fill != RED else WHITE,
                     False, PP_ALIGN.CENTER, font=MONO_FONT)
            sx += width
            if index < len(stages) - 1:
                add_arrow(slide, sx + 0.07, 5.12, 0.21, 0.17, MID_GRAY)
                sx += 0.39
        add_box(slide, 0.70, 6.05, 8.65, 0.48, BLACK, BLACK)
        add_text(slide, 0.90, 6.11, 8.25, 0.32,
                 "accum[i][j] += regA[i] × regB[j]  ｜  每个 k 步完成 TM×TN 次 FMA",
                 11.0, WHITE, True, PP_ALIGN.CENTER, font=MONO_FONT)

        # Slide 5: autotuning and cuBLAS
        slide = add_content_slide(prs, content_bg, "Auto-Tuning 与 cuBLAS 性能对比", 5)
        add_text(slide, 0.77, 1.00, 8.55, 0.42,
                 "编译期生成专用 Kernel，运行时统一预热、计时、校验并选择最高 GFLOPS",
                 14.0, BLACK, True)
        add_card(slide, 0.68, 1.62, 2.72, 1.16,
                 "参数空间", "BM / BN / BK：Block 与 K tile\nTM / TN：每线程寄存器块",
                 RED, 11.3, 8.7)
        add_card(slide, 0.68, 2.96, 2.72, 1.16,
                 "公平测量", "5 次预热 + CUDA Event\n自适应 repeats；拷贝不计时",
                 BLUE, 11.3, 8.7)
        add_card(slide, 0.68, 4.30, 2.72, 1.16,
                 "正确性与参考线", "cuBLAS FP32 Pedantic 作参考\n另测 TF32 Tensor Core 上限",
                 GREEN, 11.3, 8.7)
        add_performance_chart(slide, 3.70, 1.52, 5.68, 4.40, performance)
        add_box(slide, 0.70, 6.10, 8.65, 0.50, PALE_RED, PALE_RED)
        add_rich_text(slide, 0.90, 6.16, 8.25, 0.32, [[
            ("结果｜", 10.8, DARK_RED, True, FONT),
            ("最高 ", 10.8, DARK_RED, False, FONT),
            ("4.43 TFLOPS", 12.5, RED, True, FONT),
            ("；512–4096 规模达到严格 FP32 cuBLAS 的 ", 10.8, DARK_RED, False, FONT),
            ("51%–63%", 12.5, RED, True, FONT),
        ]], align=PP_ALIGN.CENTER)

        # Slide 6: NCU and roofline
        slide = add_content_slide(prs, content_bg, "NCU 与 Roofline：低 Occupancy 之谜", 6)
        add_text(slide, 0.77, 1.00, 8.55, 0.42,
                 "对照高复用大 tile 与高 Occupancy 小 tile，建立“资源—流量—时间”证据链",
                 13.6, BLACK, True)
        add_card(slide, 0.70, 1.53, 2.58, 1.14,
                 "大 tile｜128×128×32", "TM×TN = 8×8 · 158 Reg/thread\n32.50 KiB Shared / block",
                 RED, 11.0, 8.6)
        add_card(slide, 3.58, 1.53, 2.58, 1.14,
                 "小 tile｜64×64×16", "TM×TN = 4×4 · 48 Reg/thread\n8.25 KiB Shared / block",
                 DARK_GRAY, 11.0, 8.6)
        add_circle_label(slide, 3.27, 1.90, 0.32, "VS", BLACK, 7.5)
        add_text(slide, 0.74, 2.94, 5.82, 0.27,
                 "红色：大 tile    深灰：小 tile", 8.5, MID_GRAY, False, PP_ALIGN.CENTER)
        add_comparison_bar(slide, 0.76, 3.26, "Occupancy", 16.67, 79.73, 100, "%")
        add_comparison_bar(slide, 0.76, 4.10, "L2 流量", 554, 1090, 1200, " MB")
        add_comparison_bar(slide, 0.76, 4.94, "NCU 时间", 4.66, 6.46, 7.0, " ms")
        add_simple_roofline(slide, 6.68, 1.54, 2.62, 3.92)
        add_box(slide, 6.68, 5.60, 2.62, 0.59, PALE_RED, PALE_RED)
        add_text(slide, 6.80, 5.68, 2.38, 0.40,
                 "DRAM 仅 2.7%–3.7% 峰值\n→ 不是显存带宽瓶颈",
                 8.8, DARK_RED, True, PP_ALIGN.CENTER)
        add_box(slide, 0.70, 6.30, 8.65, 0.35, BLACK, BLACK)
        add_text(slide, 0.87, 6.34, 8.30, 0.25,
                 "结论｜大 tile 用寄存器与 Occupancy 换取更高复用，将 L2 流量近乎减半并缩短执行时间",
                 9.8, WHITE, True, PP_ALIGN.CENTER)

        # Slide 7: stalls and next steps
        slide = add_content_slide(prs, content_bg, "Warp Stall 定位与下一步优化", 7)
        add_text(slide, 0.77, 1.00, 8.55, 0.42,
                 "只有当 Scheduler 无法持续发射时，Stall 才能解释真正的延迟来源",
                 14.0, BLACK, True)
        add_text(slide, 0.76, 1.50, 5.35, 0.30,
                 "Warp Active Cycles 分布（红：大 tile｜深灰：小 tile）",
                 10.3, BLACK, True)
        add_stall_bar(slide, 0.76, 1.94, "MIO Throttle", 24.84, 53.06)
        add_stall_bar(slide, 0.76, 2.80, "Long Scoreboard", 20.72, 6.14)
        add_stall_bar(slide, 0.76, 3.66, "Not Selected", 12.53, 15.86)
        add_stall_bar(slide, 0.76, 4.52, "Barrier", 1.93, 7.20)
        add_text(slide, 0.88, 5.24, 5.05, 0.46,
                 "小 tile：Shared/MIO 管线拥塞明显\n大 tile：Warp 少，长延迟更难隐藏",
                 9.7, DARK_GRAY, False, PP_ALIGN.CENTER)
        add_text(slide, 6.21, 1.50, 2.85, 0.30,
                 "分析结论", 11.5, RED, True)
        conclusions = [
            ("1", "Occupancy 不是最终目标", "应同时看数据复用、ILP 与实际发射率"),
            ("2", "优先削减片上访存压力", "减少 Shared 指令、Bank Conflict 与同步"),
            ("3", "扩展异步计算流水线", "深化 cp.async 双缓冲并调优 Tensor Core"),
        ]
        for index, (number, title, body) in enumerate(conclusions):
            y = 1.95 + index * 1.18
            add_circle_label(slide, 6.23, y, 0.34, number, RED, 9)
            add_text(slide, 6.72, y - 0.01, 2.35, 0.29,
                     title, 10.3, BLACK, True)
            add_text(slide, 6.72, y + 0.34, 2.35, 0.47,
                     body, 8.4, DARK_GRAY, False, valign=MSO_ANCHOR.TOP)
        add_box(slide, 0.70, 6.06, 8.65, 0.58, PALE_RED, PALE_RED)
        add_text(slide, 0.88, 6.14, 8.30, 0.39,
                 "方法沉淀｜CUDA Event 判断“快了多少”，NCU + Roofline 解释“为什么变快”",
                 10.8, DARK_RED, True, PP_ALIGN.CENTER)

        # Slide 8: closing
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        add_background(slide, title_bg, prs)
        add_line(slide, 1.08, 2.38, 1.92, 2.38, RED, 2.1)
        add_text(slide, 1.08, 2.78, 7.50, 0.66,
                 "谢谢各位老师", 31, BLACK, True)
        add_text(slide, 1.08, 3.82, 7.50, 0.46,
                 "敬请批评指正", 16, RED, True)
        add_text(slide, 1.08, 4.68, 7.70, 0.55,
                 "CUDA GEMM 高性能优化与参数化 Auto-Tuning",
                 13.5, DARK_GRAY)
        add_text(slide, 1.08, 5.28, 7.70, 0.45,
                 "赵浩锦｜西安电子科技大学｜计算机科学与技术",
                 10.2, MID_GRAY)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        prs.save(str(output_path))


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("/home/drink_ice/EDisk/baoyan/赵浩锦-推免自我介绍-3min.pptx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "report" / "CUDA_GEMM_项目介绍_赵浩锦.pptx",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=project_root / "profiles" / "autotune_results.csv",
    )
    args = parser.parse_args()
    build(args.template, args.output, args.csv)
    print(args.output)


if __name__ == "__main__":
    main()
