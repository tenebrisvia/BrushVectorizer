"""Framework-agnostic Studio Engine for both Pyodide WebAssembly and local HTTP server."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import uuid
from PIL import Image

from brush_vectorizer.config import PainterlyConfig, PhysicalCalibrationConfig
from brush_vectorizer.exporters.json_exporter import export_strokes_json
from brush_vectorizer.exporters.painterly_svg import export_painterly_svg
from brush_vectorizer.exporters.svg_exporter import export_strokes_svg
from brush_vectorizer.pipeline.painterly import (
    PainterlyDocument,
    run_multilayer_plotter_pipeline,
    run_painterly_pipeline,
)
from brush_vectorizer.synthetic import (
    generate_synthetic_color_artwork,
    save_rgb_image,
)


CACHE_WRITE_LOCK = threading.Lock()


class StudioState:
    """In-memory / filesystem store for uploaded/sample images and most recent render cache."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.temp_dir = workspace_dir / ".web_cache"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.images: dict[str, Path] = {}
        self.last_render_cache: dict[str, object] = {}
        self._register_initial_samples()

    def _register_initial_samples(self) -> None:
        # Scan workspace root and optional samples/ subdirectory for existing .png / .jpg source images
        search_dirs = [self.workspace_dir]
        samples_subdir = self.workspace_dir / "samples"
        if samples_subdir.is_dir():
            search_dirs.append(samples_subdir)

        for folder in search_dirs:
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in sorted(folder.glob(ext)):
                    if p.name.endswith(".painterly.png") or "debug" in p.name:
                        continue
                    self.images[p.name] = p

        # Always provide a built-in colorful multi-layer alpine landscape demo image
        color_demo_path = self.temp_dir / "color_landscape_demo.png"
        if not color_demo_path.exists():
            rgb = generate_synthetic_color_artwork(width_px=520, height_px=620)
            save_rgb_image(rgb, color_demo_path)
        self.images.setdefault("color_landscape_demo.png", color_demo_path)

    def list_samples(self) -> list[dict[str, object]]:
        """Return metadata for all registered sample and uploaded images."""
        self._register_initial_samples()
        items: list[dict[str, object]] = []
        for name, path in self.images.items():
            if path.exists():
                try:
                    with Image.open(path) as im:
                        w, h = im.size
                    items.append({"id": name, "name": name, "width": w, "height": h})
                except Exception:
                    continue
        return items

    def register_uploaded_bytes(self, filename: str, body: bytes) -> dict[str, object]:
        """Save uploaded raw image bytes into temp_dir and register in self.images."""
        orig_name = Path(filename).name
        safe_name = f"upload_{uuid.uuid4().hex[:6]}_{orig_name}"
        dest = self.temp_dir / safe_name
        dest.write_bytes(body)

        self.images[safe_name] = dest
        with Image.open(dest) as im:
            w, h = im.size
        return {"id": safe_name, "name": orig_name, "width": w, "height": h}

    def get_image_path(self, image_id: str) -> Path:
        if image_id in self.images and self.images[image_id].exists():
            return self.images[image_id]
        for folder in (self.workspace_dir, self.workspace_dir / "samples", self.temp_dir):
            candidate = folder / image_id
            if candidate.exists():
                self.images[image_id] = candidate
                return candidate
        raise FileNotFoundError(f"Image '{image_id}' not found.")

    def create_resized_temp_image(self, src_path: Path, max_dim: int = 480) -> tuple[Path, float]:
        """Downscale image for fast interactive preview while dragging sliders (cached on disk/MEMFS)."""
        with Image.open(src_path) as img:
            w, h = img.size
            if max(w, h) <= max_dim:
                return src_path, 1.0

            scale = max_dim / float(max(w, h))
            out_path = self.temp_dir / f"preview_{max_dim}_{src_path.stem}.png"
            try:
                if out_path.exists() and out_path.stat().st_mtime_ns >= src_path.stat().st_mtime_ns:
                    return out_path, scale
            except OSError:
                pass

            new_w = max(16, int(round(w * scale)))
            new_h = max(16, int(round(h * scale)))
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized.save(out_path)
            return out_path, scale


def _execute_render_sync(
    state: StudioState,
    image_id: str,
    mode: str,
    resolution: str,
    raw_params: dict,
) -> dict:
    """Synchronous worker function executing either Painterly or Plotter mode (SVG only)."""
    t0 = time.perf_counter()
    src_path = state.get_image_path(image_id)
    with Image.open(src_path) as orig_img:
        orig_w, orig_h = orig_img.size

    is_preview = resolution == "preview"
    if is_preview:
        work_path, _scale_factor = state.create_resized_temp_image(src_path, max_dim=480)
    else:
        work_path, _scale_factor = src_path, 1.0

    if mode == "painterly":
        params = dict(raw_params)
        params.pop("plotter_display_mode", None)

        p_cfg = PainterlyConfig.model_validate(params)
        doc: PainterlyDocument = run_painterly_pipeline(work_path, config=p_cfg)

        with CACHE_WRITE_LOCK:
            svg_path = state.temp_dir / "latest_painterly.svg"
            export_painterly_svg(doc, svg_path, config=p_cfg)
            svg_text = svg_path.read_text(encoding="utf-8")

            json_path = state.temp_dir / "latest_painterly_config.json"
            json_path.write_text(json.dumps(p_cfg.model_dump(), indent=2), encoding="utf-8")

        layers_meta = [
            {
                "dom_id": "layer_0_paper_canvas",
                "name": "0_paper_canvas",
                "layer_index": -1,
                "threshold": 0.0,
                "gray": 1.0,
                "opacity": 1.0,
                "blur": 0.0,
                "default_blur": 0.0,
                "stroke_count": 1,
                "color_hex": doc.paper_hex,
                "auto_color_hex": doc.paper_hex,
            }
        ]
        for layer in doc.layers:
            layers_meta.append(
                {
                    "dom_id": f"layer_{layer.layer_name}",
                    "name": layer.layer_name,
                    "layer_index": layer.layer_index,
                    "threshold": round(layer.density_threshold, 3),
                    "gray": round(layer.target_gray, 2),
                    "opacity": round(layer.base_opacity, 2),
                    "blur": round(layer.blur_sigma, 2),
                    "default_blur": round(layer.default_blur_sigma, 2),
                    "stroke_count": len(layer.strokes),
                    "color_hex": layer.color_hex,
                    "auto_color_hex": layer.auto_color_hex,
                }
            )

        elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
        state.last_render_cache = {
            "mode": "painterly",
            "image_id": image_id,
            "resolution": resolution,
            "params": raw_params,
            "svg_path": svg_path,
            "json_path": json_path,
        }

        return {
            "mode": "painterly",
            "resolution": resolution,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "render_width": doc.width_px,
            "render_height": doc.height_px,
            "paper_hex": doc.paper_hex,
            "total_strokes": doc.total_strokes,
            "elapsed_ms": elapsed_ms,
            "svg_text": svg_text,
            "layers": layers_meta,
        }

    else:
        # Multi-Layer Acrylic Plotter mode (sharing Painterly K-layer decomposition)
        params = dict(raw_params)
        plotter_display_mode = str(params.pop("plotter_display_mode", "overlay"))
        c_dict: dict[str, object] = {}
        width_mm = float(params.pop("width_mm", 210.0))
        height_mm = float(params.pop("height_mm", 297.0))
        c_dict["canvas_dims_mm"] = (width_mm, height_mm)
        for c_key in (
            "brush_max_radius_mm",
            "brush_drag_coeff",
            "wash_Stepover_factor",
            "threshold_mode",
            "density_floor",
            "wash_density_threshold",
            "waypoint_spacing_mm",
            "min_waypoints",
            "num_spatial_bands",
        ):
            if c_key in params:
                c_dict[c_key] = params.pop(c_key)

        # Filter params to valid PainterlyConfig fields
        p_valid_keys = set(PainterlyConfig.model_fields.keys())
        p_dict = {k: v for k, v in params.items() if k in p_valid_keys}
        if "spur_prune_factor" in c_dict and "spur_prune_factor" not in p_dict:
            p_dict["spur_prune_factor"] = c_dict["spur_prune_factor"]

        p_cfg = PainterlyConfig.model_validate(p_dict)
        c_cfg = PhysicalCalibrationConfig.model_validate(c_dict)

        document = run_multilayer_plotter_pipeline(
            work_path,
            painterly_config=p_cfg,
            calibration_config=c_cfg,
            fit_canvas_to_image_aspect=True,
        )

        w_px, h_px = document.source_dims_px
        w_mm, h_mm = document.canvas_dims_mm
        res_scale = max(1.0, float(max(w_px, h_px)) / 480.0)
        scale_mm_per_px = w_mm / max(w_px, 1)
        misalign_mm = (p_cfg.layer_misalignment_px * res_scale) * scale_mm_per_px

        with CACHE_WRITE_LOCK:
            svg_path = state.temp_dir / "latest_plotter.svg"
            export_strokes_svg(
                document,
                svg_path,
                include_variable_width_preview_layer=True,
                layer_misalignment_mm=misalign_mm,
                plotter_display_mode=plotter_display_mode,
            )
            svg_text = svg_path.read_text(encoding="utf-8")

            json_path = state.temp_dir / "latest_plotter.strokes.json"
            export_strokes_json(document, json_path)

        layers_meta = [
            {
                "dom_id": "layer_0_paper_canvas",
                "name": "0_paper_canvas",
                "layer_index": -1,
                "threshold": 0.0,
                "gray": 1.0,
                "opacity": 1.0,
                "stroke_count": 1,
                "color_hex": document.paper_hex,
                "auto_color_hex": document.paper_hex,
            },
            {
                "dom_id": "layer_0_variable_width_preview",
                "name": "0_preview_acrylic_footprint",
                "layer_index": -2,
                "threshold": 0.0,
                "gray": 0.5,
                "opacity": 0.55,
                "stroke_count": document.stroke_count,
                "color_hex": "#64748b",
                "auto_color_hex": "#64748b",
            },
        ]
        for l_info in document.layers:
            layers_meta.append(
                {
                    "dom_id": f"layer_{l_info.layer_name}",
                    "name": l_info.layer_name,
                    "layer_index": l_info.layer_index,
                    "threshold": round(l_info.density_threshold, 3),
                    "gray": round(l_info.target_gray, 2),
                    "opacity": 1.0,
                    "stroke_count": l_info.stroke_count,
                    "color_hex": l_info.color_hex,
                    "auto_color_hex": l_info.auto_color_hex,
                    "air_travel_mm": l_info.air_travel_mm,
                }
            )

        elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
        state.last_render_cache = {
            "mode": "plotter",
            "image_id": image_id,
            "resolution": resolution,
            "params": raw_params,
            "svg_path": svg_path,
            "json_path": json_path,
        }

        return {
            "mode": "plotter",
            "resolution": resolution,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "render_width": w_px,
            "render_height": h_px,
            "canvas_width_mm": w_mm,
            "canvas_height_mm": h_mm,
            "paper_hex": document.paper_hex,
            "total_strokes": document.stroke_count,
            "air_travel_mm": document.air_travel_after_mm,
            "elapsed_ms": elapsed_ms,
            "svg_text": svg_text,
            "layers": layers_meta,
        }


def get_export_payload(
    state: StudioState,
    fmt: str,
    target_params: dict | None = None,
) -> dict[str, str]:
    """Ensure a full-resolution render exists for the current image/mode/params and return export file data."""
    cache = state.last_render_cache
    if not cache:
        raise RuntimeError("No active render found.")

    eff_params = target_params if target_params is not None else dict(cache["params"])
    if cache.get("resolution") != "full" or eff_params != cache.get("params"):
        _execute_render_sync(
            state,
            str(cache["image_id"]),
            str(cache["mode"]),
            "full",
            eff_params,
        )
        cache = state.last_render_cache

    stem = Path(str(cache["image_id"])).stem
    mode = str(cache["mode"])

    if fmt == "svg":
        path = Path(cache["svg_path"])
        return {
            "filename": f"{stem}_{mode}_full.svg",
            "mime_type": "image/svg+xml",
            "content": path.read_text(encoding="utf-8"),
        }
    elif fmt == "json":
        path = Path(cache["json_path"])
        return {
            "filename": f"{stem}_{mode}.json",
            "mime_type": "application/json",
            "content": path.read_text(encoding="utf-8"),
        }
    else:
        raise ValueError(f"Unsupported export format: {fmt}")
