"""Layered Painterly SVG & Preview Exporter with Color Palette Gradients and Sumi-e / Charcoal Texture."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from brush_vectorizer.config import PainterlyConfig
from brush_vectorizer.pipeline.painterly import PainterlyDocument, _rgb_to_hex


def _polygon_to_svg_d(poly_xy: np.ndarray) -> str:
    """Convert (M, 2) polygon coordinates to compact SVG path 'd' string."""
    if len(poly_xy) < 3:
        return ""
    pts = poly_xy.tolist()
    x0, y0 = pts[0]
    coords_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts[1:])
    return f"M {x0:.2f},{y0:.2f} L {coords_str} Z"


def export_painterly_svg(
    doc: PainterlyDocument,
    output_path: str | Path,
    config: PainterlyConfig | None = None,
) -> Path:
    """Export a multi-layer monitor-targeted SVG with variable-width brush shapes, color gradients, and sumi-e filters."""
    cfg = config or PainterlyConfig()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    w, h = doc.width_px, doc.height_px
    res_scale = max(1.0, float(max(w, h)) / 480.0)
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "xmlns:inkscape": "http://www.inkscape.org/namespaces/inkscape",
            "width": f"{w}px",
            "height": f"{h}px",
            "viewBox": f"0 0 {w} {h}",
            "data-mode": "painterly-multilayer",
            "data-tonal-layers": str(len(doc.layers)),
            "data-total-strokes": str(doc.total_strokes),
            "data-res-scale": f"{res_scale:.4f}",
        },
    )

    defs = ET.SubElement(svg, "defs")

    # 1. Paper Grain Filter (#filter_paper_grain)
    if cfg.paper_grain_strength > 0.01:
        f_paper = ET.SubElement(
            defs,
            "filter",
            {"id": "filter_paper_grain", "x": "0%", "y": "0%", "width": "100%", "height": "100%"},
        )
        ET.SubElement(
            f_paper,
            "feTurbulence",
            {
                "type": "fractalNoise",
                "baseFrequency": f"{0.045 / res_scale:.4f}",
                "numOctaves": "4",
                "seed": "17",
                "result": "noise",
            },
        )
        slope = f"{cfg.paper_grain_strength * 0.14:.3f}"
        intercept = f"{1.0 - cfg.paper_grain_strength * 0.07:.3f}"
        f_ct = ET.SubElement(
            f_paper, "feComponentTransfer", {"in": "noise", "result": "grain"}
        )
        for ch in ("feFuncR", "feFuncG", "feFuncB"):
            ET.SubElement(
                f_ct, ch, {"type": "linear", "slope": slope, "intercept": intercept}
            )
        ET.SubElement(
            f_paper,
            "feBlend",
            {"in": "SourceGraphic", "in2": "grain", "mode": "multiply"},
        )

    # 2. Per-Layer Diffusion & Gaussian Blur Filters (#filter_layer_0 .. #filter_layer_{K-1})
    num_layers = len(doc.layers)
    turb_mult = float(cfg.turbulence_scale)
    filter_margin = "-22%" if turb_mult > 2.0 else "-12%"
    filter_span = "144%" if turb_mult > 2.0 else "124%"

    for layer in doc.layers:
        idx = layer.layer_index
        f_layer = ET.SubElement(
            defs,
            "filter",
            {
                "id": f"filter_layer_{idx}",
                "x": filter_margin,
                "y": filter_margin,
                "width": filter_span,
                "height": filter_span,
            },
        )
        if turb_mult > 0.01 and (
            idx < max(1, num_layers // 3)
            and cfg.dry_brush_strength > 0.02
            or turb_mult > 1.25
        ):
            base_freq_num = 0.025 if idx < max(1, num_layers // 3) else 0.038
            raw_scale = (
                max(1.5, 5.0 * cfg.dry_brush_strength)
                if idx < max(1, num_layers // 3)
                else max(1.0, 2.5 * max(0.3, cfg.dry_brush_strength))
            )
            ET.SubElement(
                f_layer,
                "feTurbulence",
                {
                    "type": "fractalNoise",
                    "baseFrequency": f"{base_freq_num / res_scale:.4f}",
                    "numOctaves": "3",
                    "seed": str(5 + idx),
                    "result": "turb",
                },
            )
            ET.SubElement(
                f_layer,
                "feDisplacementMap",
                {
                    "id": f"disp_layer_{idx}",
                    "in": "SourceGraphic",
                    "in2": "turb",
                    "scale": f"{raw_scale * turb_mult * res_scale:.1f}",
                    "xChannelSelector": "R",
                    "yChannelSelector": "G",
                    "result": "disp",
                },
            )
            blur_in = "disp"
        elif turb_mult > 0.01 and idx < max(2, (2 * num_layers) // 3) and cfg.dry_brush_strength > 0.02:
            ET.SubElement(
                f_layer,
                "feTurbulence",
                {
                    "type": "fractalNoise",
                    "baseFrequency": f"{0.06 / res_scale:.4f}",
                    "numOctaves": "3",
                    "seed": str(11 + idx),
                    "result": "turb",
                },
            )
            ET.SubElement(
                f_layer,
                "feDisplacementMap",
                {
                    "id": f"disp_layer_{idx}",
                    "in": "SourceGraphic",
                    "in2": "turb",
                    "scale": f"{max(0.8, 2.8 * cfg.dry_brush_strength) * turb_mult * res_scale:.1f}",
                    "xChannelSelector": "R",
                    "yChannelSelector": "G",
                    "result": "disp",
                },
            )
            blur_in = "disp"
        else:
            blur_in = "SourceGraphic"

        ET.SubElement(
            f_layer,
            "feGaussianBlur",
            {
                "id": f"blur_layer_{idx}",
                "in": blur_in,
                "stdDeviation": f"{layer.blur_sigma * res_scale:.2f}",
            },
        )

    # 4. Per-stroke linear gradients along each stroke's spine (from entry press to exit whip)
    for layer in doc.layers:
        for st in layer.strokes:
            grad_id = f"grad_s{st.stroke_id}"
            x1, y1 = st.spine_xy[0]
            x2, y2 = st.spine_xy[-1]
            if abs(x2 - x1) < 1e-2 and abs(y2 - y1) < 1e-2:
                x2 = x1 + 1.0

            lg = ET.SubElement(
                defs,
                "linearGradient",
                {
                    "id": grad_id,
                    "data-layer-index": str(layer.layer_index),
                    "gradientUnits": "userSpaceOnUse",
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                },
            )
            offsets = ("0%", "50%", "100%")
            shades = ("1.08", "1.00", "0.88")
            for off, sh_mult, rgb_val, op_val in zip(
                offsets, shades, st.rgb_stops, st.opacity_stops
            ):
                ET.SubElement(
                    lg,
                    "stop",
                    {
                        "offset": off,
                        "data-shade": sh_mult,
                        "stop-color": _rgb_to_hex(rgb_val),
                        "stop-opacity": f"{op_val:.3f}",
                    },
                )

    # Layer 0: Natural Paper Background Canvas
    bg_group_attrs = {
        "id": "layer_0_paper_canvas",
        "inkscape:groupmode": "layer",
        "inkscape:label": "0_paper_canvas",
    }
    if cfg.paper_grain_strength > 0.01:
        bg_group_attrs["filter"] = "url(#filter_paper_grain)"
    bg_group = ET.SubElement(svg, "g", bg_group_attrs)
    ET.SubElement(
        bg_group,
        "rect",
        {
            "x": "0",
            "y": "0",
            "width": str(w),
            "height": str(h),
            "fill": doc.paper_hex,
        },
    )

    # Layers 1..K: Back-to-Front Color/Tonal Brush Layers
    for layer in doc.layers:
        idx = layer.layer_index
        layer_attrs = {
            "id": f"layer_{layer.layer_name}",
            "data-layer-index": str(idx),
            "data-color-hex": layer.color_hex,
            "data-blur-sigma": f"{layer.blur_sigma:.2f}",
            "inkscape:groupmode": "layer",
            "inkscape:label": f"{layer.layer_name} ({layer.color_hex}, D>{layer.density_threshold:.2f})",
        }
        if (
            layer.blur_sigma > 0.02
            or turb_mult > 1.25
            or (turb_mult > 0.01 and idx < max(2, (2 * num_layers) // 3))
        ):
            layer_attrs["filter"] = f"url(#filter_layer_{idx})"

        if cfg.blend_mode != "normal":
            layer_attrs["style"] = f"mix-blend-mode: {cfg.blend_mode};"

        if cfg.layer_misalignment_px > 0.05:
            # Deterministic golden-angle offset per layer for Risograph / screenprint misregistration
            angle = idx * 2.39996323
            mag = (cfg.layer_misalignment_px * res_scale) * ((idx + 1) / max(num_layers, 1))
            dx = mag * math.cos(angle)
            dy = mag * math.sin(angle)
            layer_attrs["transform"] = f"translate({dx:.2f}, {dy:.2f})"

        l_group = ET.SubElement(svg, "g", layer_attrs)

        outline_mode = cfg.stroke_outline_mode
        outline_w = f"{cfg.stroke_outline_width * res_scale:.2f}"

        for st in layer.strokes:
            env_d = _polygon_to_svg_d(st.envelope_xy)
            if not env_d:
                continue

            if outline_mode == "hollow":
                fill_attr = "none"
                stroke_attr = f"url(#grad_s{st.stroke_id})"
                extra_stroke_attrs = {"stroke-width": outline_w, "stroke-linejoin": "round"}
            elif outline_mode == "outlined":
                fill_attr = f"url(#grad_s{st.stroke_id})"
                stroke_attr = _rgb_to_hex(st.rgb_stops[0])
                extra_stroke_attrs = {
                    "stroke-width": outline_w,
                    "stroke-opacity": "0.88",
                    "stroke-linejoin": "round",
                }
            else:
                fill_attr = f"url(#grad_s{st.stroke_id})"
                stroke_attr = "none"
                extra_stroke_attrs = {}

            path_attrs = {
                "id": f"stroke_{st.stroke_id}",
                "d": env_d,
                "fill": fill_attr,
                "stroke": stroke_attr,
                "data-layer": str(idx + 1),
                "data-max-radius-px": f"{float(np.max(st.radii_px)):.2f}",
                **extra_stroke_attrs,
            }
            ET.SubElement(l_group, "path", path_attrs)

            for b_idx, bristle in enumerate(st.bristles):
                b_d = _polygon_to_svg_d(bristle.polygon_xy)
                if not b_d:
                    continue
                if outline_mode == "hollow":
                    b_fill = "none"
                    b_stroke = _rgb_to_hex(bristle.rgb_color)
                    b_extra = {"stroke-width": f"{max(0.4, cfg.stroke_outline_width * 0.55) * res_scale:.2f}", "stroke-opacity": f"{bristle.opacity:.3f}"}
                else:
                    b_fill = _rgb_to_hex(bristle.rgb_color)
                    b_stroke = "none"
                    b_extra = {"fill-opacity": f"{bristle.opacity:.3f}"}

                ET.SubElement(
                    l_group,
                    "path",
                    {
                        "id": f"stroke_{st.stroke_id}_bristle_{b_idx}",
                        "class": "bristle-path",
                        "data-shade": "1.14",
                        "d": b_d,
                        "fill": b_fill,
                        "stroke": b_stroke,
                        **b_extra,
                    },
                )

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ", level=0)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def render_painterly_raster_image(
    doc: PainterlyDocument,
    config: PainterlyConfig | None = None,
) -> Image.Image:
    """Render the multi-layer painterly document to a high-resolution PIL Image with paper grain & sumi-e blending."""
    cfg = config or PainterlyConfig()
    w, h = doc.width_px, doc.height_px
    res_scale = max(1.0, float(max(w, h)) / 480.0)

    rng = np.random.default_rng(42)
    base_arr = np.empty((h, w, 3), dtype=np.float64)
    base_arr[:, :, 0] = doc.paper_rgb[0]
    base_arr[:, :, 1] = doc.paper_rgb[1]
    base_arr[:, :, 2] = doc.paper_rgb[2]

    if cfg.paper_grain_strength > 0.01:
        fine_noise = ndimage.gaussian_filter(rng.normal(0.0, 1.0, size=(h, w)), sigma=1.1 * res_scale)
        coarse_noise = ndimage.gaussian_filter(rng.normal(0.0, 1.0, size=(h, w)), sigma=4.5 * res_scale)
        grain = 1.0 + cfg.paper_grain_strength * (0.045 * fine_noise + 0.03 * coarse_noise)
        base_arr = np.clip(base_arr * grain[:, :, None], 0.0, 255.0)

    canvas = Image.fromarray(base_arr.astype(np.uint8), mode="RGB").convert("RGBA")
    num_layers = len(doc.layers)

    tooth_noise = ndimage.gaussian_filter(rng.uniform(0.0, 1.0, size=(h, w)), sigma=0.8 * res_scale)
    tooth_mod = np.clip(
        1.0 - cfg.dry_brush_strength * 0.28 * (tooth_noise - 0.35),
        0.55,
        1.0,
    )

    for layer in doc.layers:
        layer_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer_img, "RGBA")

        for st in layer.strokes:
            if len(st.envelope_xy) >= 3:
                rgb = st.rgb_stops[1]
                op_mid = st.opacity_stops[1]
                alpha = int(np.clip(round(op_mid * 255.0), 0, 255))
                pts = [tuple(map(float, pt)) for pt in st.envelope_xy]
                draw.polygon(pts, fill=(rgb[0], rgb[1], rgb[2], alpha))

            for bristle in st.bristles:
                if len(bristle.polygon_xy) >= 3:
                    b_rgb = bristle.rgb_color
                    b_alpha = int(np.clip(round(bristle.opacity * 255.0), 0, 255))
                    b_pts = [tuple(map(float, pt)) for pt in bristle.polygon_xy]
                    draw.polygon(b_pts, fill=(b_rgb[0], b_rgb[1], b_rgb[2], b_alpha))

        if layer.blur_sigma > 0.05:
            layer_img = layer_img.filter(ImageFilter.GaussianBlur(radius=layer.blur_sigma * res_scale))

        l_arr = np.asarray(layer_img, dtype=np.float64).copy()
        l_arr[:, :, 3] *= tooth_mod
        layer_img = Image.fromarray(np.clip(l_arr, 0, 255).astype(np.uint8), mode="RGBA")

        canvas = Image.alpha_composite(canvas, layer_img)

    return canvas.convert("RGB")

