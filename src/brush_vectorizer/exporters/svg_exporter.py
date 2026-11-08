"""Plotter-ready SVG Exporter with Inkscape layers and custom kinematics attributes."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from brush_vectorizer.models import Stroke, StrokeVectorDocument
from brush_vectorizer.pipeline.painterly import _build_rounded_variable_width_polygon


def _build_Centerline_path_d(stroke: Stroke) -> str:
    """Build SVG path 'd' attribute from ordered machine waypoints."""
    if not stroke.waypoints:
        return ""
    parts = [f"M {stroke.waypoints[0].x_mm:.3f},{stroke.waypoints[0].y_mm:.3f}"]
    for wp in stroke.waypoints[1:]:
        parts.append(f"L {wp.x_mm:.3f},{wp.y_mm:.3f}")
    return " ".join(parts)


def _build_variable_width_ribbon_d(stroke: Stroke) -> str:
    """Build cusp-free closed SVG polygon path representing the physical brush footprint."""
    wps = stroke.waypoints
    if len(wps) < 2:
        return ""

    xs = np.array([wp.x_mm for wp in wps], dtype=np.float64)
    ys = np.array([wp.y_mm for wp in wps], dtype=np.float64)
    rs = np.array([wp.r_mm for wp in wps], dtype=np.float64)

    spine_xy = np.column_stack([xs, ys])
    poly_xy = _build_rounded_variable_width_polygon(
        spine_xy=spine_xy,
        radii=rs,
        edge_roughness=0.0,
        flat_caps=False,
    )
    if len(poly_xy) < 3:
        return ""

    pts = poly_xy.tolist()
    x0, y0 = pts[0]
    coords_str = " ".join(f"L {x:.3f},{y:.3f}" for x, y in pts[1:])
    return f"M {x0:.3f},{y0:.3f} {coords_str} Z"


def export_strokes_svg(
    document: StrokeVectorDocument,
    output_path: str | Path,
    include_variable_width_preview_layer: bool = True,
    layer_misalignment_mm: float = 0.0,
    plotter_display_mode: str = "overlay",
) -> Path:
    """Export StrokeVectorDocument to a Plotter-compatible SVG with custom attributes.

    Features:
    - Physical dimensions in millimeters (`width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}"`)
    - Same top-left orientation as the input raster image, uniformly scaled.
    - When `document.layers` is present (Multi-Layer Acrylic Plotter mode):
      - Emits K Inkscape centerline layers (`1_faint_wash` .. `K_black_accents`) matching the
        Painterly Multi-Layer decomposition, colored with each layer's paint color.
      - Emits an optional `0_preview_acrylic_footprint` layer with per-layer sub-groups
        showing the variable-width physical brush footprint underneath the centerlines.
    - When `document.layers` is empty (Legacy 2-layer mode):
      - Organized into `1_charcoal_wash` and `2_calligraphic_centerline`.
    - Custom `data-*` attributes on every `<path>` preserving `dwell_start_ms` and per-waypoint
      `t`, `r_mm`, `z_mm`, and `feed_mm_min` profiles.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    w_mm, h_mm = document.canvas_dims_mm
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")

    has_multilayer = len(document.layers) > 0
    svg_attrs = {
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:inkscape": "http://www.inkscape.org/namespaces/inkscape",
        "width": f"{w_mm:.3f}mm",
        "height": f"{h_mm:.3f}mm",
        "viewBox": f"0 0 {w_mm:.3f} {h_mm:.3f}",
        "data-source-width-px": str(document.source_dims_px[0]),
        "data-source-height-px": str(document.source_dims_px[1]),
        "data-stroke-count": str(document.stroke_count),
    }
    if has_multilayer:
        svg_attrs["data-mode"] = "plotter-multilayer"
        svg_attrs["data-tonal-layers"] = str(len(document.layers))

    svg = ET.Element("svg", svg_attrs)

    if has_multilayer:
        num_layers = len(document.layers)
        strokes_by_layer: dict[int, list[Stroke]] = {l.layer_index: [] for l in document.layers}
        for st in document.strokes:
            strokes_by_layer.setdefault(st.layer_index, []).append(st)

        # Layer 0a: Natural Paper Background Canvas (non-plotting reference layer)
        bg_group = ET.SubElement(
            svg,
            "g",
            {
                "id": "layer_0_paper_canvas",
                "inkscape:groupmode": "layer",
                "inkscape:label": "0_paper_canvas (non-plotting)",
            },
        )
        ET.SubElement(
            bg_group,
            "rect",
            {
                "x": "0",
                "y": "0",
                "width": f"{w_mm:.3f}",
                "height": f"{h_mm:.3f}",
                "fill": document.paper_hex,
            },
        )

        # Layer 0b: Opaque Acrylic Variable-Width Footprint Preview Layer
        if include_variable_width_preview_layer:
            preview_group_attrs = {
                "id": "layer_0_variable_width_preview",
                "inkscape:groupmode": "layer",
                "inkscape:label": "0_preview_acrylic_footprint (hide before plotting)",
                "opacity": "1.0" if plotter_display_mode == "acrylic" else "0.55",
            }
            if plotter_display_mode == "centerlines":
                preview_group_attrs["style"] = "display: none;"
            preview_group = ET.SubElement(svg, "g", preview_group_attrs)
            footprint_fill_op = "0.96" if plotter_display_mode == "acrylic" else "0.88"
            for l_info in document.layers:
                idx = l_info.layer_index
                sub_attrs = {
                    "id": f"preview_layer_{l_info.layer_name}",
                    "data-layer-index": str(idx),
                    "data-preview-for": f"layer_{l_info.layer_name}",
                }
                if layer_misalignment_mm > 0.01:
                    angle = idx * 2.39996323
                    mag = layer_misalignment_mm * ((idx + 1) / max(num_layers, 1))
                    dx = mag * math.cos(angle)
                    dy = mag * math.sin(angle)
                    sub_attrs["transform"] = f"translate({dx:.3f}, {dy:.3f})"

                sub_preview = ET.SubElement(preview_group, "g", sub_attrs)
                for stroke in strokes_by_layer.get(idx, []):
                    ribbon_d = _build_variable_width_ribbon_d(stroke)
                    if not ribbon_d:
                        continue
                    ET.SubElement(
                        sub_preview,
                        "path",
                        {
                            "id": f"preview_stroke_{stroke.stroke_id}",
                            "class": "footprint-preview-path",
                            "data-layer-index": str(idx),
                            "d": ribbon_d,
                            "fill": stroke.color_hex or l_info.color_hex,
                            "fill-opacity": footprint_fill_op,
                            "stroke": "none",
                        },
                    )

        # Layers 1..K: Back-to-Front Acrylic Paint Centerline Layers for Plotter
        for l_info in document.layers:
            idx = l_info.layer_index
            layer_attrs = {
                "id": f"layer_{l_info.layer_name}",
                "data-layer-index": str(idx),
                "data-color-hex": l_info.color_hex,
                "inkscape:groupmode": "layer",
                "inkscape:label": f"{l_info.layer_name} ({l_info.color_hex}, D>{l_info.density_threshold:.2f})",
                "fill": "none",
                "stroke": l_info.color_hex,
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            }
            if plotter_display_mode == "acrylic":
                layer_attrs["style"] = "display: none;"
            if layer_misalignment_mm > 0.01:
                angle = idx * 2.39996323
                mag = layer_misalignment_mm * ((idx + 1) / max(num_layers, 1))
                dx = mag * math.cos(angle)
                dy = mag * math.sin(angle)
                layer_attrs["transform"] = f"translate({dx:.3f}, {dy:.3f})"

            l_group = ET.SubElement(svg, "g", layer_attrs)

            for stroke in strokes_by_layer.get(idx, []):
                d_str = _build_Centerline_path_d(stroke)
                if not d_str:
                    continue
                mean_r_mm = float(np.mean([wp.r_mm for wp in stroke.waypoints]))
                t_profile = ",".join(f"{wp.t:.2f}" for wp in stroke.waypoints)
                r_profile = ",".join(f"{wp.r_mm:.2f}" for wp in stroke.waypoints)
                z_profile = ",".join(f"{wp.z_mm:.2f}" for wp in stroke.waypoints)
                feed_profile = ",".join(f"{wp.feed_mm_min:.0f}" for wp in stroke.waypoints)

                ET.SubElement(
                    l_group,
                    "path",
                    {
                        "id": f"stroke_{stroke.stroke_id}",
                        "class": "centerline-path",
                        "d": d_str,
                        "fill": "none",
                        "stroke": stroke.color_hex or l_info.color_hex,
                        "stroke-width": f"{max(0.22, min(mean_r_mm * 0.42, 0.95)):.3f}",
                        "data-stroke-id": str(stroke.stroke_id),
                        "data-layer-index": str(idx),
                        "data-stroke-type": stroke.stroke_type,
                        "data-dwell-start-ms": str(stroke.dwell_start_ms),
                        "data-t": t_profile,
                        "data-r-mm": r_profile,
                        "data-z-mm": z_profile,
                        "data-feed-mm-min": feed_profile,
                    },
                )
    else:
        if include_variable_width_preview_layer:
            # Non-plotting visual preview layer (hidden/locked by default for plotters or visible in viewers)
            preview_group = ET.SubElement(
                svg,
                "g",
                {
                    "id": "layer_0_variable_width_preview",
                    "inkscape:groupmode": "layer",
                    "inkscape:label": "0_preview_variable_width (hide before plotting)",
                    "opacity": "0.22",
                },
            )
            for stroke in document.strokes:
                ribbon_d = _build_variable_width_ribbon_d(stroke)
                if not ribbon_d:
                    continue
                fill_color = "#6b7280" if stroke.stroke_type == "charcoal_wash" else "#111827"
                ET.SubElement(
                    preview_group,
                    "path",
                    {
                        "id": f"preview_stroke_{stroke.stroke_id}",
                        "d": ribbon_d,
                        "fill": fill_color,
                        "stroke": "none",
                    },
                )

        # Layer 1: Broad Charcoal Wash Centerlines
        wash_group = ET.SubElement(
            svg,
            "g",
            {
                "id": "layer_1_charcoal_wash",
                "inkscape:groupmode": "layer",
                "inkscape:label": "1_charcoal_wash",
                "fill": "none",
                "stroke": "#2563eb",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            },
        )

        # Layer 2: Calligraphic Structural Centerlines
        callig_group = ET.SubElement(
            svg,
            "g",
            {
                "id": "layer_2_calligraphic_centerline",
                "inkscape:groupmode": "layer",
                "inkscape:label": "2_calligraphic_centerline",
                "fill": "none",
                "stroke": "#111827",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            },
        )

        for stroke in document.strokes:
            d_str = _build_Centerline_path_d(stroke)
            if not d_str:
                continue

            target_group = (
                wash_group if stroke.stroke_type == "charcoal_wash" else callig_group
            )
            mean_r_mm = float(np.mean([wp.r_mm for wp in stroke.waypoints]))
            t_profile = ",".join(f"{wp.t:.2f}" for wp in stroke.waypoints)
            r_profile = ",".join(f"{wp.r_mm:.2f}" for wp in stroke.waypoints)
            z_profile = ",".join(f"{wp.z_mm:.2f}" for wp in stroke.waypoints)
            feed_profile = ",".join(f"{wp.feed_mm_min:.0f}" for wp in stroke.waypoints)

            ET.SubElement(
                target_group,
                "path",
                {
                    "id": f"stroke_{stroke.stroke_id}",
                    "d": d_str,
                    "stroke-width": f"{max(0.25, min(mean_r_mm * 0.5, 1.2)):.3f}",
                    "data-stroke-id": str(stroke.stroke_id),
                    "data-stroke-type": stroke.stroke_type,
                    "data-dwell-start-ms": str(stroke.dwell_start_ms),
                    "data-t": t_profile,
                    "data-r-mm": r_profile,
                    "data-z-mm": z_profile,
                    "data-feed-mm-min": feed_profile,
                },
            )

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ", level=0)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out
