"""Unit and integration tests for the Raster-to-Brushwork Vectorization Pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np

from brush_vectorizer.config import CanvasTransform, PainterlyConfig, PhysicalCalibrationConfig
from brush_vectorizer.exporters.json_exporter import export_strokes_json
from brush_vectorizer.exporters.painterly_svg import (
    export_painterly_svg,
    render_painterly_raster_image,
)
from brush_vectorizer.exporters.svg_exporter import export_strokes_svg
from brush_vectorizer.models import PixelPath
from brush_vectorizer.pipeline.painterly import run_painterly_pipeline
from brush_vectorizer.pipeline.runner import run_vectorization_pipeline_from_array
from brush_vectorizer.pipeline.stage1_preprocess import run_stage1_preprocessing
from brush_vectorizer.pipeline.stage2_medial_axis import run_stage2_medial_axis
from brush_vectorizer.pipeline.stage3_graph import run_stage3_graph_untangling
from brush_vectorizer.pipeline.stage4_classify import run_stage4_classification
from brush_vectorizer.pipeline.stage5_polarity import (
    evaluate_path_polarity,
    run_stage5_polarity_resolution,
)
from brush_vectorizer.pipeline.stage6_kinematics import run_stage6_kinematics
from brush_vectorizer.synthetic import generate_synthetic_color_artwork, save_rgb_image


class TestVectorizationPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PhysicalCalibrationConfig(
            canvas_dims_mm=(210.0, 297.0),
            brush_max_radius_mm=3.5,
            brush_drag_coeff=0.85,
            z_calibration_poly=(-0.15, -1.10, 0.0),
        )

    def test_stage1_and_stage2_spur_pruning(self) -> None:
        """Verify Stage 1 thresholding and Stage 2 EDT + spur pruning on a thick stroke with an edge bump."""
        lum = np.ones((120, 240), dtype=np.float64)
        # Draw a thick horizontal bar (radius ~ 10 px) with a small perpendicular bump (spur)
        lum[50:71, 30:210] = 0.1
        lum[46:50, 118:123] = 0.1  # 4px bump on top edge (<= 1.35 * 10)

        s1 = run_stage1_preprocessing(lum, self.config)
        self.assertTrue(np.any(s1.stroke_mask))
        self.assertAlmostEqual(float(s1.density_map[60, 120]), 0.9, places=2)

        s2 = run_stage2_medial_axis(s1.stroke_mask, self.config)
        # Centerline EDT radius should be around 10.5 px
        center_r = float(s2.edt_radius_px[60, 120])
        self.assertGreaterEqual(center_r, 9.5)
        self.assertTrue(np.any(s2.pruned_skeleton))

    def test_stage3_x_intersection_untangling(self) -> None:
        """Verify two crossing strokes at an X-intersection untangle into 2 continuous paths."""
        h, w = 200, 200
        lum = np.ones((h, w), dtype=np.float64)
        yy, xx = np.mgrid[0:h, 0:w]

        # Line 1: y = 0.5 * x + 50 (from x=20 to x=180)
        dist1 = np.abs(0.5 * xx - yy + 50.0) / np.hypot(0.5, 1.0)
        mask1 = (dist1 <= 4.5) & (xx >= 20) & (xx <= 180)

        # Line 2: y = -0.5 * x + 150 (from x=20 to x=180)
        dist2 = np.abs(-0.5 * xx - yy + 150.0) / np.hypot(0.5, 1.0)
        mask2 = (dist2 <= 4.5) & (xx >= 20) & (xx <= 180)

        lum[mask1 | mask2] = 0.1
        s1 = run_stage1_preprocessing(lum, self.config)
        s2 = run_stage2_medial_axis(s1.stroke_mask, self.config)
        s3 = run_stage3_graph_untangling(
            s2.pruned_skeleton, s2.edt_radius_px, s1.density_map, self.config
        )

        # Should untangle the 4 arms of the X intersection into 2 continuous through-strokes
        self.assertEqual(len(s3.untangled_paths), 2)
        for p in s3.untangled_paths:
            self.assertGreater(p.arc_length_px, 120.0)

    def test_stage4_broad_wash_subdivision(self) -> None:
        """Verify broad low-density strokes subdivide into multiple parallel charcoal_wash passes."""
        transform = CanvasTransform.from_image_and_canvas(200, 200, (100.0, 100.0))
        # scale = 0.5 mm/px; R_max = 3.5 mm = 7.0 px
        # Create a wide stroke with radius = 18 px (9.0 mm > 3.5 mm) and density = 0.45 (< 0.75)
        h, w = 200, 200
        stroke_mask = np.zeros((h, w), dtype=bool)
        stroke_mask[80:120, 20:180] = True
        edt = np.full((h, w), 18.0, dtype=np.float64)
        density = np.full((h, w), 0.45, dtype=np.float64)

        xs = np.linspace(25.0, 175.0, 80)
        ys = np.full_like(xs, 100.0)
        wide_path = PixelPath(
            path_id=0,
            coords_xy_px=np.column_stack([xs, ys]),
            radii_px=np.full_like(xs, 18.0),
            densities=np.full_like(xs, 0.45),
        )

        s4 = run_stage4_classification(
            [wide_path], stroke_mask, edt, density, transform, self.config
        )
        self.assertEqual(s4.wash_region_count, 1)
        self.assertGreater(s4.wash_subpass_count, 1)
        for p in s4.classified_paths:
            self.assertEqual(p.stroke_type, "charcoal_wash")

    def test_stage5_polarity_reversal(self) -> None:
        """Verify a stroke drawn backwards (thin striated start, thick uniform end) gets reversed."""
        h, w = 100, 200
        density = np.full((h, w), 0.85, dtype=np.float64)
        # Add high striation noise near x in [20, 50] (end A = actual tail)
        rng = np.random.default_rng(7)
        density[:, 20:55] = rng.uniform(0.15, 0.95, size=(h, 35))

        xs = np.linspace(20.0, 180.0, 100)
        ys = np.full_like(xs, 50.0)
        t = np.linspace(0.0, 1.0, 100)
        # Radius is small at A (t=0) and has blunt heavy belly near B (t=0.88)
        radii = 1.2 + 7.0 * (t ** 1.8) * np.exp(-((1.0 - t) / 0.25) ** 2)
        radii[-5:] = np.linspace(8.0, 5.5, 5)

        backward_path = PixelPath(
            path_id=0,
            coords_xy_px=np.column_stack([xs, ys]),
            radii_px=radii,
            densities=np.full_like(xs, 0.8),
        )

        metrics = evaluate_path_polarity(backward_path, density, self.config)
        self.assertGreater(metrics.score_b, metrics.score_a)
        self.assertTrue(metrics.reversed_path)

        s5 = run_stage5_polarity_resolution([backward_path], density, self.config)
        self.assertEqual(s5.reversed_count, 1)
        self.assertAlmostEqual(float(s5.oriented_paths[0].coords_xy_px[0, 0]), 180.0, places=2)

    def test_stage6_kinematics_and_profiles(self) -> None:
        """Verify Stage 6 B-spline fitting, bristle drag offset, Z polynomial, and 3-phase feedrate."""
        transform = CanvasTransform.from_image_and_canvas(200, 200, (200.0, 200.0))
        xs = np.linspace(20.0, 120.0, 50)
        ys = np.full_like(xs, 100.0)
        radii = np.full_like(xs, 2.0)
        densities = np.full_like(xs, 1.0)

        path = PixelPath(
            path_id=0,
            coords_xy_px=np.column_stack([xs, ys]),
            radii_px=radii,
            densities=densities,
        )

        s6 = run_stage6_kinematics([path], transform, self.config)
        self.assertEqual(len(s6.strokes), 1)
        stroke = s6.strokes[0]
        self.assertGreaterEqual(stroke.dwell_start_ms, 30)
        self.assertLessEqual(stroke.dwell_start_ms, 60)

        wps = stroke.waypoints
        self.assertEqual(wps[0].t, 0.0)
        self.assertEqual(wps[-1].t, 1.0)

        # Forward drag offset along +X tangent: x_machine(0) should equal 20.0 + alpha * r(0) = 20.0 + 0.85 * 2.0 = 21.7
        self.assertAlmostEqual(wps[0].x_mm, 20.0 + 0.85 * 2.0, delta=0.15)
        # Entry press feedrate clamped to F_min (400)
        self.assertAlmostEqual(wps[0].feed_mm_min, self.config.f_min_mm_min, places=1)
        # Exit whip at t=1.0 reaches F_max (3000), Z_clearance (0.0), and r_mm=0.1
        self.assertAlmostEqual(wps[-1].feed_mm_min, self.config.f_max_mm_min, places=1)
        self.assertAlmostEqual(wps[-1].z_mm, self.config.z_clearance_mm, places=2)
        self.assertAlmostEqual(wps[-1].r_mm, 0.1, places=2)

    def test_end_to_end_synthetic_and_exporters(self) -> None:
        """Run full pipeline on synthetic brushwork image and validate JSON + SVG outputs."""
        rgb = generate_synthetic_color_artwork(width_px=300, height_px=420, seed=123).astype(np.float64)
        lum = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]) / 255.0
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            artifacts = run_vectorization_pipeline_from_array(
                luminance=lum,
                config=self.config,
            )

            self.assertGreater(artifacts.document.stroke_count, 0)

            # Verify broad-to-fine layer ordering: all charcoal_wash strokes precede calligraphic_centerline
            types = [st.stroke_type for st in artifacts.document.strokes]
            if "charcoal_wash" in types and "calligraphic_centerline" in types:
                last_wash_idx = max(i for i, t in enumerate(types) if t == "charcoal_wash")
                first_callig_idx = min(
                    i for i, t in enumerate(types) if t == "calligraphic_centerline"
                )
                self.assertLess(last_wash_idx, first_callig_idx)

            json_path = export_strokes_json(artifacts.document, tmp_path / "out.json")
            svg_path = export_strokes_svg(artifacts.document, tmp_path / "out.svg")

            loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded_json["stroke_count"], artifacts.document.stroke_count)
            self.assertIn("waypoints", loaded_json["strokes"][0])

            tree = ET.parse(svg_path)
            root = tree.getroot()
            self.assertEqual(root.attrib["width"], "210.000mm")
            self.assertEqual(root.attrib["height"], "297.000mm")

    def test_multi_layer_painterly_pipeline(self) -> None:
        """Verify multi-threshold tonal slicing and variable-width painterly SVG/PNG export."""
        rgb = generate_synthetic_color_artwork(width_px=200, height_px=260, seed=99)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_path = save_rgb_image(rgb, tmp_path / "synth.png")
            p_cfg = PainterlyConfig(num_tonal_layers=5, bristle_count=3)
            doc = run_painterly_pipeline(img_path, config=p_cfg)

            self.assertEqual(len(doc.layers), 5)
            self.assertGreater(doc.total_strokes, 0)

            svg_out = export_painterly_svg(doc, tmp_path / "painterly.svg", config=p_cfg)
            self.assertTrue(svg_out.exists())
            tree = ET.parse(svg_out)
            root = tree.getroot()
            self.assertEqual(root.attrib["data-mode"], "painterly-multilayer")

            rendered = render_painterly_raster_image(doc, config=p_cfg)
            self.assertEqual(rendered.size, (200, 260))

    def test_web_studio_render_endpoints(self) -> None:
        """Verify Web Studio state and synchronous render worker for both painterly and plotter modes."""
        from brush_vectorizer.web.engine import StudioState, _execute_render_sync

        with tempfile.TemporaryDirectory() as tmpdir:
            state = StudioState(Path(tmpdir))
            res_painterly = _execute_render_sync(
                state=state,
                image_id="color_landscape_demo.png",
                mode="painterly",
                resolution="preview",
                raw_params={"num_tonal_layers": 4, "bristle_count": 2},
            )
            self.assertEqual(res_painterly["mode"], "painterly")
            self.assertGreater(res_painterly["total_strokes"], 0)
            self.assertIn("<svg", res_painterly["svg_text"])
            self.assertNotIn("png_data_url", res_painterly)

            res_plotter = _execute_render_sync(
                state=state,
                image_id="color_landscape_demo.png",
                mode="plotter",
                resolution="preview",
                raw_params={"num_tonal_layers": 4, "width_mm": 210.0, "height_mm": 297.0},
            )
            self.assertEqual(res_plotter["mode"], "plotter")
            self.assertGreater(res_plotter["total_strokes"], 0)
            # 4 paint layers + paper canvas + acrylic footprint preview = 6 interactive layers
            self.assertEqual(len(res_plotter["layers"]), 6)
            self.assertIn('data-mode="plotter-multilayer"', res_plotter["svg_text"])

    def test_painterly_color_palette_and_layer_color_overrides(self) -> None:
        """Verify K-color quantization from RGB source image and per-layer custom hex color overrides."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create a colorful synthetic image with warm cream background and 3 distinct color strokes
            rgb = np.full((180, 220, 3), (245, 240, 230), dtype=np.uint8)
            rgb[30:55, 25:195] = (210, 120, 60)   # Terracotta wash band
            rgb[80:102, 25:195] = (45, 115, 165)  # Cobalt blue mid-stroke
            rgb[130:148, 25:195] = (35, 25, 45)   # Deep indigo accent stroke

            color_img_path = tmp_path / "color_strokes.png"
            Image.fromarray(rgb, mode="RGB").save(color_img_path)

            custom_colors = ["#e07a5f", "#3d5a80", "#1d2d44"]
            p_cfg = PainterlyConfig(
                num_tonal_layers=3,
                color_palette_mode="auto_palette",
                layer_colors=custom_colors,
                bristle_count=2,
            )
            doc = run_painterly_pipeline(color_img_path, config=p_cfg)
            self.assertEqual(len(doc.layers), 3)
            for idx, layer in enumerate(doc.layers):
                self.assertEqual(layer.color_hex, custom_colors[idx])
                self.assertTrue(layer.auto_color_hex.startswith("#"))

            svg_out = export_painterly_svg(doc, tmp_path / "color.svg", config=p_cfg)
            svg_txt = svg_out.read_text(encoding="utf-8")
            self.assertIn('data-layer-index="0"', svg_txt)
            self.assertIn('data-shade=', svg_txt)

    def test_creative_and_abstract_painterly_effects(self) -> None:
        """Verify all 13 creative geometry, spine distortion, flow lock, and optical SVG effects."""
        rgb = generate_synthetic_color_artwork(width_px=180, height_px=220, seed=77)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_path = save_rgb_image(rgb, tmp_path / "creative_input.png")

            p_cfg = PainterlyConfig(
                num_tonal_layers=4,
                nib_flatness=0.75,
                nib_angle_deg=40.0,
                taper_power=2.2,
                radius_gamma=-0.5,
                polygon_decimation=0.55,
                spine_wave_amplitude=6.0,
                spine_wave_frequency=3.0,
                bristle_splay=1.8,
                stroke_extension_px=10.0,
                flow_lock_strength=0.65,
                flow_lock_angle_deg=35.0,
                blend_mode="difference",
                stroke_outline_mode="hollow",
                stroke_outline_width=1.5,
                stroke_hue_jitter=0.35,
                turbulence_scale=4.5,
                layer_misalignment_px=5.0,
            )
            doc = run_painterly_pipeline(img_path, config=p_cfg)
            self.assertEqual(len(doc.layers), 4)
            self.assertGreater(doc.total_strokes, 0)

            svg_out = export_painterly_svg(doc, tmp_path / "creative.svg", config=p_cfg)
            svg_txt = svg_out.read_text(encoding="utf-8")
            self.assertIn("mix-blend-mode: difference;", svg_txt)
            self.assertIn("translate(", svg_txt)
            self.assertIn('fill="none"', svg_txt)
            self.assertIn('stroke-width="1.50"', svg_txt)

    def test_multilayer_acrylic_plotter_pipeline(self) -> None:
        """Verify K-layer acrylic plotter pipeline matches Painterly layer count, palette, and centerline kinematics."""
        from PIL import Image
        from brush_vectorizer.pipeline.painterly import run_multilayer_plotter_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rgb = np.full((180, 220, 3), (245, 240, 230), dtype=np.uint8)
            rgb[30:55, 25:195] = (210, 120, 60)   # Layer 1 wash
            rgb[80:102, 25:195] = (45, 115, 165)  # Layer 2 midtone
            rgb[130:148, 25:195] = (35, 25, 45)   # Layer 3 dark accent

            img_path = tmp_path / "acrylic_test.png"
            Image.fromarray(rgb, mode="RGB").save(img_path)

            p_cfg = PainterlyConfig(num_tonal_layers=5, color_palette_mode="auto_palette")
            c_cfg = PhysicalCalibrationConfig(
                canvas_dims_mm=(210.0, 297.0),
                brush_max_radius_mm=3.0,
                wash_Stepover_factor=0.65,
                brush_drag_coeff=0.80,
            )

            p_doc = run_painterly_pipeline(img_path, config=p_cfg)
            plotter_doc = run_multilayer_plotter_pipeline(
                img_path,
                painterly_config=p_cfg,
                calibration_config=c_cfg,
            )

            # Same number of layers and matching layer names / colors
            self.assertEqual(len(plotter_doc.layers), len(p_doc.layers))
            for pl_layer, pa_layer in zip(plotter_doc.layers, p_doc.layers):
                self.assertEqual(pl_layer.layer_name, pa_layer.layer_name)
                self.assertEqual(pl_layer.color_hex, pa_layer.color_hex)

            self.assertGreater(plotter_doc.stroke_count, 0)
            svg_out = export_strokes_svg(plotter_doc, tmp_path / "acrylic_plotter.svg")
            svg_txt = svg_out.read_text(encoding="utf-8")
            self.assertIn('data-mode="plotter-multilayer"', svg_txt)
            self.assertIn('data-tonal-layers="5"', svg_txt)
            self.assertIn('class="centerline-path"', svg_txt)
            self.assertIn('class="footprint-preview-path"', svg_txt)
            for pl_layer in plotter_doc.layers:
                self.assertIn(f'id="layer_{pl_layer.layer_name}"', svg_txt)
                self.assertIn(f'id="preview_layer_{pl_layer.layer_name}"', svg_txt)

            # Verify Opaque Acrylic display mode in exported SVG
            svg_acrylic = export_strokes_svg(
                plotter_doc,
                tmp_path / "acrylic_only.svg",
                plotter_display_mode="acrylic",
            ).read_text(encoding="utf-8")
            self.assertIn('fill-opacity="0.96"', svg_acrylic)
            self.assertIn('style="display: none;"', svg_acrylic)

    def test_resolution_scale_invariance(self) -> None:
        """Verify full-resolution (>480px) renders scale brush widths and SVG filters proportionally to 480px preview."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create a 480x360 preview image and an identical 1440x1080 (3x) full-res image
            rgb_480 = np.full((360, 480, 3), (245, 240, 232), dtype=np.uint8)
            rgb_480[120:200, 60:420] = (30, 35, 45)
            path_480 = tmp_path / "img_480.png"
            Image.fromarray(rgb_480, mode="RGB").save(path_480)

            img_1440 = Image.fromarray(rgb_480, mode="RGB").resize((1440, 1080), Image.Resampling.NEAREST)
            path_1440 = tmp_path / "img_1440.png"
            img_1440.save(path_1440)

            p_cfg = PainterlyConfig(num_tonal_layers=3, color_palette_mode="monochrome")
            doc_480 = run_painterly_pipeline(path_480, config=p_cfg)
            doc_1440 = run_painterly_pipeline(path_1440, config=p_cfg)

            max_r_480 = max(float(np.max(st.radii_px)) for l in doc_480.layers for st in l.strokes)
            max_r_1440 = max(float(np.max(st.radii_px)) for l in doc_1440.layers for st in l.strokes)
            ratio = max_r_1440 / max(max_r_480, 1e-3)
            self.assertGreater(ratio, 2.5)
            self.assertLess(ratio, 3.5)

            svg_1440 = export_painterly_svg(doc_1440, tmp_path / "out_1440.svg", config=p_cfg).read_text(encoding="utf-8")
            self.assertIn('data-res-scale="3.0000"', svg_1440)

    def test_pyodide_static_build_and_wasm_bundle_execution(self) -> None:
        """Verify build_static_site produces a valid Pyodide bundle and engine.py runs without numba or tornado."""
        import zipfile
        from brush_vectorizer.web.build_static import build_static_site
        from brush_vectorizer.web.engine import StudioState, _execute_render_sync, get_export_payload

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dist_dir = build_static_site(output_dir=tmp_path / "dist", workspace_dir=tmp_path)

            self.assertTrue((dist_dir / "index.html").exists())
            self.assertTrue((dist_dir / "pyodide_worker.js").exists())
            self.assertTrue((dist_dir / "brush_vectorizer.zip").exists())
            self.assertTrue((dist_dir / "samples" / "manifest.json").exists())

            with zipfile.ZipFile(dist_dir / "brush_vectorizer.zip", "r") as zf:
                names = set(zf.namelist())
                self.assertIn("brush_vectorizer/web/engine.py", names)
                self.assertIn("brush_vectorizer/pipeline/painterly.py", names)
                self.assertNotIn("brush_vectorizer/web/server.py", names)
                self.assertNotIn("brush_vectorizer/cli.py", names)

            state = StudioState(tmp_path / "wasm_workspace")
            samples = state.list_samples()
            sample_ids = [s["id"] for s in samples]
            self.assertIn("color_landscape_demo.png", sample_ids)
            self.assertNotIn("synthetic_demo.png", sample_ids)

            res = _execute_render_sync(
                state=state,
                image_id="color_landscape_demo.png",
                mode="painterly",
                resolution="preview",
                raw_params={"num_tonal_layers": 4, "bristle_count": 2},
            )
            self.assertEqual(res["mode"], "painterly")
            self.assertGreater(res["total_strokes"], 0)

            exp_svg = get_export_payload(state, "svg")
            self.assertTrue(exp_svg["filename"].endswith(".svg"))
            self.assertIn("<svg", exp_svg["content"])


if __name__ == "__main__":
    unittest.main()
